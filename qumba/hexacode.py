#!/usr/bin/env python

"""
hexacode and golay code experiments

"""


from random import shuffle, choice, randint
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul, lshift

from qumba.action import mulclose, mulclose_find
from qumba.matrix import Matrix, DEFAULT_P, pullback
from qumba.symplectic import symplectic_form, SymplecticSpace
from qumba.clifford import Clifford, half
from qumba.qcode import QCode, strop
from qumba.util import cross, allperms
from qumba import construct
from qumba.lagrel import cup, cap, Lagrangian


class Network:
    def __init__(self, codes, links):
        codes = list(codes)
        for code in codes:
            assert code.k == 0
        links = [(i,j) for (i,j) in links]
        self.codes = codes
        self.links = links


def contract(codes, links):

    Hs = []
    for code in codes:
        assert code.k == 0
        Hs.append(code.H)
    H = reduce(lshift, Hs)
    m, nn = H.shape
    lookup = [i for i in range(nn//2)]

    idxs = [0]
    for code in codes:
        i = idxs[-1]
        idxs.append(i + code.n)
    print(idxs)

    links = [(idxs[i]+j, idxs[k]+l) for (i,j,k,l) in links]
    print(links)

    while links:

        m, nn = H.shape
        n = nn//2

        print("H:", m, n)
        print(links)

        i, j = links.pop()
        assert i!=j, (i,j)
        if j<i:
            i,j = j,i
        assert 0<=i<j<n, (i,j)

        cols = list(range(n))
        cols = cols[:i] + cols[i+1:j] + cols[j+1:] + [i,j]
        #print(cols)
        cols = reduce(add, [(2*i,2*i+1) for i in cols]) # pairs
        #print(cols)
        H1 = H[:, cols]
        rel = Lagrangian(H[:, :-4], H[:, -4:])
        assert rel.is_lagrangian()
        #print(rel)
        rel = rel * cup
        assert rel.is_lagrangian()
        #print(rel)
        H = rel.A

        # repair indexes
        _links = []
        for (ii,jj) in links:
            assert ii not in [i,j]
            assert jj not in [i,j]
            if ii > j:
                ii -= 2
            elif ii > i:
                ii -= 1
            if jj > j:
                jj -= 2
            elif jj > i:
                jj -= 1
            _links.append((ii,jj))
        links = _links

    code = QCode(H)
    return code


def get_hexacode():
    code = QCode.fromstr("""
    XYXIIY
    IXYXIY
    IIXYXY
    YZYIIZ
    IYZYIZ
    IIYZYZ
    """) # Macwilliams-Sloane p598

    return code


def test_contract():

    code = get_hexacode()
    print(code)
    N, perms = code.get_autos()
    assert N==60

    if 0:
        dode = code.shorten(0)
        print(dode)
        print(dode.longstr())
        print(dode.get_autos())
        print(dode.is_gf4())
    
        eode = QCode.fromstr("XYX.Y .XYXY YZY.Z .YZYZ", None, "YXIIY ZYIIZ")
        assert dode.is_equiv(eode)
    

    codes = [code] * 3
    links = [(0,0,1,0), (0,1,1,1)]
    code = contract(codes, links)

    print(code)


def test_A5():

    from bruhat.gset import Group, Perm
    from bruhat.todd_coxeter import Schreier

    G = Group.alternating(5)
    print(G)

    Hs = G.conjugacy_subgroups()
    for H in Hs:
        if len(H)==5:
            break
    print(H, H.is_abelian())

    X = G.action_subgroup(H)
    print(X)


def test():
    from bruhat.gset import Group, Perm
    from bruhat.todd_coxeter import Schreier

    # Bring's curve reflection group
    ngens = 3
    a, b, c = range(ngens)
    rels = [
        (a,)*2, (b,)*2, (c,)*2,
        (a,b)*5, (b,c)*5, (a,c)*2,
    ]
    a1 = (b,a)
    b1 = (a,c)
    rels += [ (3*a1+b1)*3 ]
    graph = Schreier(ngens, rels)
    graph.build()
    assert len(graph) == 120 # == 12 * 10
    G = graph.get_gset()
    assert len(G) == 120

    #print(G.structure_description()) # C2 x A5

    a, b, c = G.gens

    R = Group.generate([a*b, b*c])
    assert len(R) == 60

    F = Group.generate([a*b])
    assert len(F) == 5
    faces = [gF for gF in G.left_cosets(F) if len(gF.intersect(R))==len(F)]
    assert len(faces) == 12

    E = Group.generate([a*c])
    edges = [gE for gE in G.left_cosets(E) if len(gE.intersect(R))==len(E)]
    assert len(edges) == 30

    V = Group.generate([b*c])
    verts = [gV for gV in G.left_cosets(V) if len(gV.intersect(R))==len(V)]
    assert len(verts) == 12

    import numpy
    H = numpy.zeros((len(faces), len(edges)), dtype=int)
    for i,face in enumerate(faces):
      for j,edge in enumerate(edges):
        if face.intersect(edge):
            H[i,j] = 1
    print(H)


    code = get_hexacode()
    dode = code.apply_perm([1,2,3,4,0,5])
    assert code.is_equiv(dode)




if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

    start_time = time()


    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        #import cProfile as profile
        #profile.run("%s()"%name)
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            test()
        profiler.print()



    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)

