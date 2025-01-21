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
from qumba.solve import zeros2, parse


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

