#!/usr/bin/env python

from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce, cache

import numpy

from bruhat.gset import allgroups, Group, Perm
from bruhat.hecke import get_operators
from bruhat.gap import Gap
gap = Gap()

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba import construct


def span_ops(ops):
    N = len(ops)
    m, n = ops[0].shape
    if N > 12:
        print("(span_ops: N=%d is too big?)" % N, end=" ", flush=True)
        for op in ops:
            yield op # at least we tried...
        return
    for bits in numpy.ndindex((2,)*N):
        if sum(bits) == 0:
            continue
        op = Matrix.zeros((m, n))
        for (i,bit) in enumerate(bits):
            if bit:
                op = op + ops[i]
        yield op

        

found = set()


def get_selfdual(G, desc=None):

    s = G.structure_description()
    if desc and s != desc:
        return

    print("\t", G, G.structure_description(), end=", ")

    #print(G.perms)
    N = G.rank
    n = gap.Order(G, get=True)
    assert len(G) == n
    if argv.all_subgroups:
        _Hs = gap.AllSubgroups(G)
    else:
        _Hs = gap.ConjugacyClassesSubgroups(G)
    #print("\t", gap.Length(Hs, get=True))
    Hs = []
    for i in range(len(_Hs)):
        item = _Hs[i]
        if not argv.all_subgroups:
            item = gap.Representative(item)
        assert gap.IsPermGroup(item, get=True)
        H = gap.to_group(item, N=N)
        #print("\t", H)
        assert G.is_subgroup(H)
        if len(G) > len(H) > 1: # right ?!?
            Hs.append(H)

    print("subgroups:", len(Hs))
    for i in range(len(Hs)):
      for j in range(i, len(Hs)):

        X = G.left_action(Hs[i])
        Y = G.left_action(Hs[j])

        ops = list(get_operators(X.gens, Y.gens))
        #print(ops[0].shape, end=' ')
        ops = [Matrix(op) for op in ops]

        for H0 in span_ops(ops):
          for H in [H0, H0.t]:
            _, n = H.shape
            #if n <= 16: # this code is too small ...
            #    continue

            HHt = H*H.t # overlap of the rows
            if HHt.sum() != 0:
                continue

            H = H.linear_independent()
            m, n = H.shape
            assert 2*m <= n
            if 2*m == n:
                continue
            #print(HHt.shape)
            #print(H.shortstr(), H.shape)
            css = CSSCode(Hx=H, Hz=H, check=False)
            if css.n-css.k < 100:
                css.bz_distance()
            if css.d and css.d <= 2:
                continue

            if (css.n, css.k, css.d) == (24, 14, 4):
                print(H, css)

            if len(H) > 20:
                key = (css.n, css.k, css.d)
                wenum = None

            else:
                wenum = H.get_wenum()
                key = (css.d, wenum)

            if key not in found:
                found.add(key)
                weight = 0
                if wenum is not None:
                  for idx,wt in enumerate(wenum):
                    if idx and wt:
                        weight = idx
                        break
                print(css, weight)



def get_weight(H):
    m, n = H.shape

    vecs = [[] for i in range(n+1)]
    for u in numpy.ndindex((2,)*m):
        v = Matrix(u) * H
        w = v.sum()
        vecs[w].append(v)

    rows = []
    for w,vs in enumerate(vecs):
        if not w or not len(vs):
            continue
        rows += vs
        H1 = Matrix(rows)
        H1 = H1.row_reduce()
        if len(H1) == m:
            return w
    assert 0



def get_css_weight(css):

    Hx = css.Hx
    Hz = css.Hz

    wx = get_weight(Hx)
    wz = get_weight(Hz)
    return wx, wz

    



def get_css(G, desc=None):

    s = G.structure_description()
    if desc and s != desc:
        return

    print("\t", G, G.structure_description(), end=", ")

    #print(G.perms)
    N = G.rank
    n = gap.Order(G, get=True)
    assert len(G) == n
    _Hs = gap.ConjugacyClassesSubgroups(G)
    #print("\t", gap.Length(Hs, get=True))
    Hs = []
    for i in range(len(_Hs)):
        item = _Hs[i]
        item = gap.Representative(item)
        assert gap.IsPermGroup(item, get=True)
        H = gap.to_group(item, N=N)
        #print("\t", H)
        assert G.is_subgroup(H)
        index = len(G) // len(H)
        if argv.index and index != argv.index:
            continue
        if len(G) > len(H) > 1: # right ?!?
            Hs.append(H)

    del N
    print("subgroups:", len(Hs), end=', ', flush=True)

    print("build action... ", end='', flush=True)
    Xs = [G.left_action(H) for H in Hs]

    N = len(Xs)
    pairs = {}
    for i in range(N):
      for j in range(N):
        ops = list(get_operators(Xs[i].gens, Xs[j].gens))
        ops = [Matrix(op) for op in ops]
        if argv.show:
            print(i, j)
            for op in ops:
                print(op)
                print("-" * 7)
                print(op.normal_form())
                print("=" * 7)
            print()
        ops = [H.linear_independent() for H in span_ops(ops)]
        pairs[i,j] = ops

    print("done")

    for i in range(N):
     for jx in range(N):
       opxs = pairs[jx, i]
       for jz in range(N):
        opzs = pairs[jz, i]

        for Hx in (opxs):
         for Hz in (opzs):

            HHt = Hx*Hz.t
            if HHt.sum() != 0:
                continue

            #Hx = Hx.linear_independent()
            #Hz = Hz.linear_independent()
            mx, n = Hx.shape
            mz, n = Hz.shape
            assert mx+mz <= n
            if mx+mz == n:
                continue
            css = CSSCode(Hx=Hx, Hz=Hz, check=False, build=False)
            if css.n-css.k < 100:
                css.bz_distance()
            if css.d and css.d <= 2:
                continue

            wx, wz = (Hx.get_wenum(), Hz.get_wenum())
            if (wx,wz) in found or (wz,wx) in found:
                continue

            found.add((wx,wz))

            w = get_css_weight(css)
            print(css, w)





def main():

    gap = Gap()

    func = get_selfdual
    if argv.get_css:
        func = get_css

    print(func)

    if argv.PSL:
        m = argv.get("m", 3)
        n = argv.get("n", 2)
        N = argv.get("N", None)
        _G = gap.PSL(m, n)
        G = gap.to_group(_G, smaller=False, N=N)
        func(G)
        return

    if argv.Alternating:
        n = argv.get("n", 5)
        _G = gap.AlternatingGroup(n)
        G = gap.to_group(_G, smaller=False)
        func(G)
        return

    if argv.Symmetric:
        n = argv.get("n", 5)
        _G = gap.SymmetricGroup(n)
        G = gap.to_group(_G, smaller=False)
        func(G)
        return

    if argv.CoxeterBC:
        n = argv.get("n", 3)
        G = Group.coxeter_bc(n)
        func(G)
        return

    if argv.CoxeterD:
        n = argv.get("n", 4)
        G = Group.coxeter_d(n)
        func(G)
        return

    if argv.Mathieu:
        n = argv.get("n", 11)
        _G = gap.MathieuGroup(n)
        G = gap.to_group(_G, smaller=False)
        func(G)
        return

    desc = argv.desc
    n = argv.get("n", 27)
    n0 = argv.get("n0", n)
    n1 = argv.get("n1", n0+1)
    for _G in allgroups(n0, n1):
        G = gap.to_group(_G, smaller=False)
        func(G, desc)



def test_24():
    H = Matrix.parse("""
    11..1..11.1.11..1.1.1..1
    ..11.11.1.1...111.1..11.
    ..111..1.1.1..11.1.11..1
    1..11.1.11...11.11..1.1.
    1.1.11..1..11.1..11...11 
    """)

    H = Matrix.parse("""
    ..11......111.1.1.1.
    .11.....1..11..1..11
    1..111...1.1....1.1.
    11..1.1.1.1..11.....
    ......11..11.11..11.
    ....11111..11.1.1..1
    """)
    

    w = H.get_wenum()
    print(w)

    N, perms = H.get_autos()
    perms = [Perm(idxs) for idxs in perms]
    G = Group.generate(perms)
    print(len(G))
    print(G.structure_description())

    from bruhat.gap import Gap
    gap = Gap()

    M = gap.MathieuGroup(24)
    _G = gap.define(G)
    print(gap.IsomorphicSubgroups(M, _G, get=True)) # yes


def test_bring():
    css = construct.get_bring()
    print(css)

    perms = css.find_autos()
    perms = [Perm(idxs) for idxs in perms]
    G = Group(perms)
    print(G.structure_description())

    get_css(G)



if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "main"
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





