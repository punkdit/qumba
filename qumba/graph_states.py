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




def is_graph_state(code): # XXX SLOW
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

    assert is_graph_state(pode) is not None

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
            A = is_graph_state(code)
            if A is not None:
                graphs.add(A)
                smap[0, col] = str(A)
                col += n+1
        print(len(orbit))
        print(smap)
        print()

    print("graphs:", len(graphs))



def find_orbits():

    n = argv.get("n", 4)

    # first we build all the graph states
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
        #A = is_graph_state(H)
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
                #B = is_graph_state(J) 
                if J in graphs:
                    graphs.remove(J)
                _bdy.append(J)
                orbit.add(J)
            bdy = _bdy
        orbits.append(orbit)
        print(len(orbit), end=' ', flush=True)
    print()

    orbits.sort(key=len)

    print()
    smap = SMap()
    i = 0
    for orbit in orbits:
        for M in orbit:
            A = is_graph_state(M) # SLOW
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


