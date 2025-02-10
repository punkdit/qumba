#!/usr/bin/env python

"""
search for triorthogonal matrices 
see:
https://arxiv.org/abs/2408.09685
"""

# XXX move to csscode.py ?

import numpy

from qumba.argv import argv
import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, sat, ForAll, PbEq
from qumba.umatrix import UMatrix, Expr
from qumba.csscode import CSSCode
from qumba.matrix import Matrix
from qumba.qcode import QCode
from qumba import construct
from qumba.lin import zeros2, parse, array2, pseudo_inverse, dot2, eq2, kernel, rand2, shortstr, linear_independent
from qumba.util import choose, cross


def dump_transverse(Hx, Lx, t=3):
    import CSSLO
    SX,LX,SZ,LZ = CSSLO.CSSCode(Hx, Lx)
    #CSSLO.CZLO(SX, LX)
    N = 1<<t
    zList, qList, V, K_M = CSSLO.comm_method(SX, LX, SZ, t, compact=True, debug=False)
    for z,q in zip(zList,qList):
        #print(z, q)
        print(CSSLO.CP2Str(2*q,V,N),"=>",CSSLO.z2Str(z,N))
    print()
    #return zList


def search():

    n, m0, m = 14, 3, 5 # [[14,2,2]]
    n, m0, m = 15, 4, 5 # [[15,1,3]]
    #n, m0, m = 17, 4, 5 # [[17,1,3]] repeated colums
    #n, m0, m = 17, 4, 6 # ?
    #n, m0, m = 17, 4, 7 # ?

    n = argv.get("n", n)
    m0 = argv.get("m0", m0)
    m = argv.get("m", m)

    blocks = argv.get("blocks", True)

    solver = Solver()
    add = solver.add

    if blocks:
        A = UMatrix.unknown(m0, n-m0)
        G1 = UMatrix.unknown(m-m0, n)
        Im = Matrix.identity(m0)
        G = UMatrix.unknown(m, n)
        G[:m0,:m0] = Im
        G[:m0,m0:] = A
        G[m0:,:] = G1

    else:
        G = UMatrix.unknown(m,n)

    #print(G)

    for i in range(m0):
        add(G[i,:].sum() == False) # even weight
    for i in range(m0, m):
        add(G[i,:].sum() == True) # odd weight

    for a in range(m):
      ga = G[a]
      for b in range(a+1, m):
        gb = G[b]
        gab = ga.prod(gb)
        add(gab.sum() == False) # biorthogonal
        for c in range(b+1, m):
            gc = G[c]
            gabc = gab.prod(gc)
            add(gabc.sum() == False) # triorthogonal

    get = lambda item : Expr.promote(item).get()

    # no missing Hx bits
    for col in range(n):
        add(Or([get(G[row,col]) for row in range(m0)]))

    # XXX G0 must be full rank XXX
    for row in range(m):
        add(Or([get(G[row,col]) for col in range(n)]))

    print("solver:")
    _G = G
    while 1:
    
        result = solver.check()
        if result != sat:
            print(result)
            return

        model = solver.model()

        G = _G.get_interp(model)

        add(_G[:,m:] != G[:,m:])
        #break
    
#        print()
#        print(G)
#        print()
#        for i in range(m):
#          for j in range(m):
#            ai, aj = G[i], G[j]
#            print( (ai.A * aj.A).sum(), end=" " )
#          print()
#        print()
    
        idxs = [i for i in range(m) if G[i].sum()%2]
        G1 = G[idxs, :] # odd weight
        #print(G1, G1.shape)
    
        idxs = [i for i in range(m) if G[i].sum()%2==0]
        G0 = G[idxs, :] # even weight
        #print(G0, G0.shape)
        #print()
    
        Hx = G0
        Hz = G.kernel()
    
        #print("Hx")
        #print(Hx, Hx.shape)
        #print("Hz")
        #print(Hz, Hz.shape)

        if Hx.rank() != len(Hx):
            print("*", end="", flush=True)
            continue
    
        code = QCode.build_css(Hx, Hz)

        #return

        if code.k:
            if code.d < 3:
                print("/", end="", flush=True)
            else:
                print(code)
                code = code.to_css()
                dump_transverse(code.Hx, code.Lx)
        else:
            print(".", end="", flush=True)


def make_double(S, T=None):
    ms, ns = S.shape
    if T is None:
        T = zeros2(0, ns)
    mt, nt = T.shape
    D = zeros2(ms+mt+1, 2*ns+nt)
    D[:ms, :ns] = S
    D[:ms, ns:2*ns] = S
    D[ms:ms+mt, 2*ns:] = T
    D[ms+mt, ns:] = 1
    return D
    


def te_codes():
    # see: 
    # https://arxiv.org/abs/2408.12752
    # https://www.youtube.com/watch?v=F266RMc0yEI

    # step 1: classical self-dual code of type-II
    # ...

    de = construct.get_713()

    print(de)

    de = de.to_css()
    S = de.Hx
    print(S)

    D = make_double(S)
    print(D)


def bravyi49():
    # see: https://arxiv.org/abs/1209.2426

    G = parse("""
    1111111111111110101010101010101010101010101010101
    0000000000000000000111100110011000011001100110011
    0000000000000001100000011001100110000000000000000
    0000000000000000000000000000000001111000000001111
    0000000000000000011110000000000000000111100000000
    0000000000000000000001111000011110000000000000000
    0000000000000000000000000111111110000000000000000
    0000000000000000000000000000000001111111100000000
    0000000000000000000000000000000000000000011111111
    1010101010101010000000000000000000000000000000000
    0110011001100110000000000000000000000000000000000
    0001111000011110000000000000000000000000000000000
    0000000111111110000000000000000000000000000000000
    1111111111111111111111111111111111111111111111111
    """)

    G = Matrix(G)
    G0 = G[:len(G)-1]
    print(G0)

    Hx = G0
    Hz = G.kernel()

    print("Hx:")
    for h in Hx:
        print(h, h.sum())
    print("Hz:")
    for h in Hz:
        print(h, h.sum())

    #return

#    print()
#    print(Hz, Hz.shape)
#    for i,h in enumerate(Hz):
#        if h.sum()==5:
#            break
#    print(h)
#    Hz = Hz[:i].concatenate(Hz[i+1:])
#    print(Hz, Hz.shape)

    code = QCode.build_css(Hx=Hx, Hz=Hz)
    print(code)

    #from qumba import db
    #code.desc = "triorthogonal"
    #db.add(code)
    
    #print(code.longstr())

    #css = code.to_css()
    #dump_transverse(css.Hx, css.Lx)

    #from qumba.diego import dump
    #dump(code)

def test_49_1_5():
    import json
    from pytket import Circuit

    # Load the circuit from a JSON files#
    with open('49_1_5.json', 'r') as f:
        data = json.load(f)
    circuit = Circuit.from_dict(data) 

    print(circuit)
    print(circuit.n_qubits)
    #for op in entire_circuit:
    #    print(op)



def is_morthogonal(G, m):
    k = len(G)
    if m==1:
        for v in G:
            if v.sum()%2 != 0:
                return False
        return True
    if m>2 and not is_morthogonal(G, m-1):
        return False
    items = list(range(k))
    for idxs in choose(items, m):
        v = G[idxs[0]]
        for idx in idxs[1:]:
            v = v * G[idx]
        if v.sum()%2 != 0:
            return False
    return True


def strong_morthogonal(G, m):
    k = len(G)
    assert m>=1
    if m==1:
        for v in G:
            if v.sum()%2 != 0:
                return False
        return True
    if not strong_morthogonal(G, m-1):
        return False
    items = list(range(k))
    for idxs in choose(items, m):
        v = G[idxs[0]]
        for idx in idxs[1:]:
            v = v * G[idx]
        if v.sum()%2 != 0:
            return False
    return True



def find_triorthogonal(m, k):
    # Bravyi, Haah, 1209.2426v1 sec IX.
    # https://arxiv.org/pdf/1209.2426.pdf

    verbose = argv.get("verbose")
    #m = argv.get("m", 6) # _number of rows
    #k = argv.get("k", None) # _number of odd-weight rows

    # these are the variables N_x
    xs = list(cross([(0, 1)]*m))

    maxweight = argv.maxweight
    minweight = argv.get("minweight", 1)

    xs = [x for x in xs if minweight <= sum(x)]
    if maxweight:
        xs = [x for x in xs if sum(x) <= maxweight]

    N = len(xs)

    lhs = []
    rhs = []

    # bi-orthogonality
    for a in range(m):
      for b in range(a+1, m):
        v = zeros2(N)
        for i, x in enumerate(xs):
            if x[a] == x[b] == 1:
                v[i] = 1
        if v.sum():
            lhs.append(v)
            rhs.append(0)

    # tri-orthogonality
    for a in range(m):
      for b in range(a+1, m):
       for c in range(b+1, m):
        v = zeros2(N)
        for i, x in enumerate(xs):
            if x[a] == x[b] == x[c] == 1:
                v[i] = 1
        if v.sum():
            lhs.append(v)
            rhs.append(0)

#    # dissallow columns with weight <= 1
#    for i, x in enumerate(xs):
#        if sum(x)<=1:
#            v = zeros2(N)
#            v[i] = 1
#            lhs.append(v)
#            rhs.append(0)

    if k is not None:
      # constrain to k _number of odd-weight rows
      assert 0<=k<m
      for a in range(m):
        v = zeros2(N)
        for i, x in enumerate(xs):
          if x[a] == 1:
            v[i] = 1
        lhs.append(v)
        if a<k:
            rhs.append(1)
        else:
            rhs.append(0)

    A = array2(lhs)
    rhs = array2(rhs)
    #print(shortstr(A))

    B = pseudo_inverse(A)
    soln = dot2(B, rhs)
    if not eq2(dot2(A, soln), rhs):
        print("no solution")
        return
    if verbose:
        print("soln:")
        print(shortstr(soln))

    soln.shape = (N, 1)
    rhs.shape = A.shape[0], 1

    K = array2(list(kernel(A)))
    #print(K)
    #print( dot2(A, K.transpose()))
    #sols = []
    #for v in span(K):
    best = None
    density = 1.0
    size = 99*N
    trials = argv.get("trials", 102400)
    #print("trials:", trials)
    count = 0
    #for trial in range(trials):
    while 1:
        u = rand2(len(K), 1)
        v = dot2(K.transpose(), u)
        #print(v)
        v = (v+soln)%2
        assert eq2(dot2(A, v), rhs)

        if v.sum() > size:
            continue
        size = v.sum()

        Gt = []
        for i, x in enumerate(xs):
            if v[i]:
                Gt.append(x)
        if not Gt:
            continue
        Gt = array2(Gt)
        G = Gt.transpose()
        assert is_morthogonal(G, 3)
        if G.shape[1]<m:
            continue

        if 0 in G.sum(1):
            continue

        if argv.strong_morthogonal and not strong_morthogonal(G, 3):
            continue

        #print(shortstr(G))
#        for g in G:
#            print(shortstr(g), g.sum())
#        print()

        yield G

        _density = float(G.sum()) / (G.shape[0]*G.shape[1])
        #if best is None or _density < density:
        if best is None or G.shape[1] <= size:
            best = G
            size = G.shape[1]
            density = _density

        if 0:
            #sols.append(G)
            Gx = even_rows(G)
            assert is_morthogonal(Gx, 3)
            if len(Gx)==0:
                continue
            GGx = array2(list(span(Gx)))
            assert is_morthogonal(GGx, 3)

        count += 1

    print("found %d solutions" % count)

#    if best is None:
#        return
#
#    G = best
#    #print(shortstr(G))
#
#    for g in G:
#        print(shortstr(g), g.sum())
#    print()
#    print("density:", density)
#    print("shape:", G.shape)
#
#    G = linear_independent(G)
#    for g in G:
#        print(shortstr(g), g.sum())
#
#    if 0:
#        A = list(span(G))
#        print(strong_morthogonal(A, 1))
#        print(strong_morthogonal(A, 2))
#        print(strong_morthogonal(A, 3))
#
#    #G = [row for row in G if row.sum()%2 == 0]
#    return array2(G)
#
#    #print(shortstr(dot2(G, G.transpose())))
#
#    if 0:
#        B = pseudo_inverse(A)
#        v = dot2(B, rhs)
#        print("B:")
#        print(shortstr(B))
#        print("v:")
#        print(shortstr(v))
#        assert eq2(dot2(B, v), rhs)


def main():

    m = argv.get("m", 6)
    k = argv.get("k", 1)

    found = set()
    for G in find_triorthogonal(m, k):

        G = Matrix(G)
    
        #print("G:")
        #print(G)
    
        if 0:
            j = 0
            while j < G.shape[1]:
                gj = G[:,j]
                #print(gj, gj.sum())
                if G[:, j].sum() == 1:
                    #print("puncture", j)
                    G = G.puncture(j)
                else:
                    j += 1
        
        G = G.linear_independent()
        
        m, n = G.shape
    
        idxs = [i for i in range(m) if G[i].sum()%2]
        G1 = G[idxs, :] # odd weight
        #print(G1, G1.shape)
    
        idxs = [i for i in range(m) if G[i].sum()%2==0]
        G0 = G[idxs, :] # even weight
        #print(G0, G0.shape)
        #print()
    
        Hx = G0
        Hz = G.kernel()
    
        #print(Hx.shape, Hz.shape)
    
        code = QCode.build_css(Hx, Hz)
        s = str(code)
        if s in found:
            continue
        print(s)
        found.add(s)

        if code.d < 3:
            continue
    
        print(G, G.shape)
        code = code.to_css()
        dump_transverse(code.Hx, code.Lx)



if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)

