#!/usr/bin/env python

from random import shuffle

import numpy

from qumba.action import mulclose
from qumba.symplectic import SymplecticSpace
from qumba.matrix import Matrix
from qumba.qcode import QCode, fromstr, strop
from qumba.smap import SMap
from qumba import lin
from qumba.argv import argv



def main():

    n = argv.get("n", 3)

    space = SymplecticSpace(n)
    S = space.S
    H = space.H
    I = space.get_identity()
    CX = space.CX
    CZ = space.CZ

    P = I
    for i in range(n):
        P = P*H(i)
    print(P)

    gen  = [S(i) for i in range(n)]
    gen += [H(i) for i in range(n)]

    LC = mulclose(gen)
    print("|LC| =", len(LC))

    orbits = []
    found = []
    for E in [
        I, 
        CZ(0,1),
        CZ(0,2), CZ(1,2), 
        CZ(0,1) * CZ(1,2),
        CZ(0,2) * CZ(1,2),
    ]:
        orbit = []
        overlap = 0
        src = QCode.from_encoder(E)
        for g in LC:
            code = g*src
            for dode in orbit:
                if code.is_equiv(dode):
                    break
            else:
                orbit.append(code)
                for dode in found:
                    if dode.is_equiv(code):
                        print("*", end="")
                        overlap += 1
                        break
                else:
                    found.append(code)
        if overlap:
            print((overlap))
        print("orbit =", len(orbit))
        orbits.append(orbit)

    print(sum(len(o) for o in orbits))
    #print(54 + 3*18 + 27)
    print(len(found))




def get_graph(code): # XXX SLOW
    if isinstance(code, QCode):
        assert code.k == 0, "not a state"
        #print(code.longstr(False))
        #print()
        n = code.n
        H = code.H
    else:
        assert isinstance(code, Matrix)
        H = code
        n, nn = H.shape
        assert 2*n==nn

    H = list(H.rowspan())
    shuffle(H)
    H = Matrix(H)
    H = H.reshape(2**n, 2*n)
    s = strop(H)
    stabs = s.split("\n")
    idxs = set()
    A = lin.zeros2(n,n)
    for stab in stabs:
        if stab.count("X") != 1:
            continue
        if "Y" in stab:
            continue
        #print(stab)
        idx = stab.index("X")
        if idx in idxs:
            return
        idxs.add(idx)
        for i,s in enumerate(stab):
            if s=='Z':
                A[idx,i] = 1
    if len(idxs) != n:
        return
    A = Matrix(A)
    #print(A)
    if A==A.t:
        return A



def find_lagrangian(): # SLOW
    n = 4
    code = QCode.fromstr(""" XZZZ ZZXZ ZXZZ ZZZX """)
    code = QCode.fromstr(""" XIZI ZIXZ IXIZ IZZX """) # U shape
    pode = QCode.fromstr(""" XZZI ZXIZ ZIXZ IZZX """)
    #pode = QCode.fromstr(""" XZZZ ZXII ZIXI ZIIX """)

    assert get_graph(pode) is not None

    H = pode.H.normal_form()
    assert QCode(H).is_equiv(pode)

    found = set()
    for code in construct.all_codes(4,0,0):
        H = code.H.normal_form()
        dode = QCode(H)
        found.add(dode)
        assert dode.is_equiv(code)
    print(H.shape)
    print(len(found))

    space = code.space
    gen = [space.S(i) for i in range(n)]
    gen += [space.H(i) for i in range(n)]
    G = mulclose(gen)
    print("|G| =", len(G))

    orbits = []
    remain = set(found)
    graphs = set()
    while remain:
        code = remain.pop()
        orbit = [code]
        for g in G:
            dode = g*code
            H = dode.H.normal_form()
            eode = QCode(H)
            assert eode.k == 0
            assert eode.is_equiv(dode)
            assert eode in found
            if eode in remain:
                remain.remove(eode)
                orbit.append(eode)
        orbits.append( orbit )
    orbits.sort(key = len)

    for orbit in orbits:
        col = 0
        smap = SMap()
        for code in orbit:
            A = get_graph(code)
            if A is not None:
                graphs.add(A)
                smap[0, col] = str(A)
                col += n+1
        print(len(orbit))
        print(smap)
        print()

    print("graphs:", len(graphs))


def all_graph_states(n):
    S = numpy.empty(shape=(n,n), dtype=object)
    S[:] = '.'
    I,X,Z = ".XZ"

    def shortstr(S):
        return ("\n".join(''.join(row) for row in S))

    idxs = [(i,j) for i in range(n) for j in range(i+1,n)]
    assert len(idxs) == n*(n-1)//2
    N = len(idxs)

    graphs = set()
    for bits in numpy.ndindex((2,)*N):
        S[:] = I
        for i in range(n):
            S[i,i] = X

        for (i,bit) in enumerate(bits):
            if bit==0:
                continue
            j,k = idxs[i]
            S[j,k] = Z
            S[k,j] = Z

        s = shortstr(S)
        #print(s)
        #code = QCode.fromstr(s)
        #H = code.H.normal_form()
        H = fromstr(s)
        H = Matrix(H).normal_form()
        #print(H)
        #print()
        graphs.add(H)

    assert len(graphs) == 2**N
    return graphs


def lc_graph_states(n, verbose=True):

    # first we build all the graph states
    graphs = all_graph_states(n)

    if verbose:
        print("graphs:", len(graphs))

    space = SymplecticSpace(n)
    gens = [space.S(i) for i in range(n)]
    gens += [space.H(i) for i in range(n)]

    #graphs = list(graphs)
    orbits = []
    while graphs:
        H = graphs.pop()
        orbit = {H}
        bdy = [H]
        #A = get_graph(H)
        #print(A)
        while bdy:
            _bdy = []
            for H in bdy:
              #code = QCode(H)
              for g in gens:
                J = H*g.t
                #dode = g*code
                #J = dode.H
                #assert J == H*g.t
                J = J.normal_form()
                if J in orbit:
                    continue
                #B = get_graph(J) 
                if J in graphs:
                    graphs.remove(J)
                _bdy.append(J)
                orbit.add(J)
            bdy = _bdy
        orbits.append(orbit)
        if verbose:
            print(len(orbit), end=' ', flush=True)
    if verbose:
        print()

    orbits.sort(key=len)
    return orbits


def show_entanglement():
    n = argv.get("n", 3)

    graphs = all_graph_states(n)
#    for M in graphs:
#        print(M)
#        print()

    orbits = lc_graph_states(n)

    P = list(numpy.ndindex((2,)*n))
    assert len(P) == 2**n

    def get_ent(A):
        assert A.shape == (n,n)
        ranks = []
        for idxs in P:
            rows = [i for (i,ii) in enumerate(idxs) if ii==1]
            cols = [i for (i,ii) in enumerate(idxs) if ii==0]
            B = A[rows, :]
            B = B[:, cols]
            ranks.append(B.rank())
        return tuple(ranks)

    funcs = set()
    count = 0
    for orbit in orbits:
        orbit = [M for M in orbit if M in graphs]

        func = None
        for M in orbit:
            A = get_graph(M)
            gunc = get_ent(A)
            assert func is None or func==gunc
            func = gunc
        assert func
        funcs.add(func)
        count += len(orbit)

        print(len(orbit), end=' ', flush=True)
    print()
    print("total:", count)
    print("orbits:", len(orbits))
    print("funcs:", len(funcs))

    if argv.show_poset:
        show_poset(n, funcs)
        return

    if argv.show_stats:
        show_stats(n, funcs)
        return

    if argv.dump:
        f = open("funcs_%d.py"%n, "w")
        funcs = list(funcs)
        funcs.sort()
        for func in funcs:
            print(func, file=f)
        f.close()
        return

    from qumba.util import factorial
    def choose(m,n):
        top = factorial(m)
        bot = factorial(n) * factorial(m-n)
        assert top%bot == 0
        return top // bot

    from huygens.namespace import Canvas, path, grey, black

    mod = 99999
    dx = dy = 1.0
    r = 0.05
    if n==4:
        mod = 6 # 18
    elif n==5:
        mod = 3
        dx = dy = 0.7
    elif n==6:
        mod = 6
        dx = dy = 0.45
        

    def plot_func(f):
        cvs = Canvas()
        cols = [0]*(n+1)
        coords = {}
        for idx in P:
            row = sum(idx)
            col = cols[row] - 0.5*choose(n,row)
            x = dx*col
            y = dy*row
            coords[idx] = (x,y)
            cols[row] += 1
        for idx in P:
          for jdx in P:
            kdx = tuple(i-j for (i,j) in zip(idx,jdx))
            if min(kdx)>=0 and sum(kdx)==1:
                x0, y0 = coords[idx]
                x1, y1 = coords[jdx]
                cvs.stroke(path.line(x0,y0,x1,y1), [grey])
        for i,idx in enumerate(P):
            x, y = coords[idx]
            for j in range(f[i]):
                cvs.stroke(path.circle(x,y,(j+1)*r), [black])
        return cvs

    cvs = Canvas()

    funcs = list(funcs)
    funcs.sort(key = sum)
    x = 0
    y = 0
    for i,f in enumerate(funcs):
        print(f)
        fg = plot_func(f)
        cvs.insert(x, y, fg)
        bb = fg.get_bound_box()
        x += 1.1*bb.width
        if (i+1)%mod==0:
            x = 0
            y -= 1.1*bb.height

    cvs.writePDFfile("entanglement_%d.pdf"%n)


def show_orbits():
    n = argv.get("n", 4)
    orbits = lc_graph_states(n)

    print()
    smap = SMap()
    i = 0
    for orbit in orbits:
        for M in orbit:
            A = get_graph(M) # SLOW
            if A is not None:
                break
        col = i*(n+3)
        smap[0,col] = str(A)
        smap[n,col] = str(len(orbit))

        i += 1
        if i%8==0:
            print(smap)
            print()
            smap = SMap()
            i = 0
    if i:
        print(smap)
        print()

    counts = ([len(orbit) for orbit in orbits])
    print("found %d orbits" % (len(orbits)))
    print(counts, "==", sum(counts))


def show_stats(n, fs):
    print("show_stats", n)

    keys = []
    for f in fs:
        key = tuple(f.count(i) for i in range(n-1))
        keys.append(key)
    keys.sort()
    uniq = list(set(keys))
    uniq.sort(key = lambda key:keys.count(key))
    for key in uniq:
        print(keys.count(key), key)


def show_graphs_6():

    lines = open("graphs_6.py").readlines()
    assert len(lines) == 760

    fs = [eval(line) for line in lines]
    show_stats(6, fs)


def show_poset(n, fs):
    print("show_poset", n)

    fs = list(fs)
    fs.sort()
    print(fs[0])

    pairs = set()
    up = {f:set() for f in fs}
    dn = {f:set() for f in fs}
    for f in fs:
      for g in fs:
        if f is g:
            continue
        for (i,j) in zip(f,g):
            if i>j:
                break
        else:
            up[f].add(g)
            dn[g].add(f)
            pairs.add((f,g))

#    pairs = list(pairs)
#    shuffle(pairs)
#    f,g = pairs[0]
#    print(f)
#    print("<")
#    print(g)
#    return
    print("pairs:", len(pairs))

    for (f,g) in list(pairs):
        for h in up[g]:
            if (f,h) in pairs:
                pairs.remove((f,h))
    print("pairs:", len(pairs))

    names = {}
    for i,f in enumerate(fs):
        name = ''.join(str(k) for k in f[:len(f)//2])
        names[f] = "v"+name

    dot = open("poset_%d.dot"%n, "w")
    print("digraph {", file=dot)
    for (f,g) in pairs:
        print("  %s -> %s;"%(names[f], names[g]), file=dot)
    print("}", file=dot)
    dot.close()




if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))


