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


def span_ops(ops):
    N = len(ops)
    m, n = ops[0].shape
    if N >= 10:
        return
    assert N < 10, N
    for bits in numpy.ndindex((2,)*N):
        if sum(bits) == 0:
            continue
        op = Matrix.zeros((m, n))
        for (i,bit) in enumerate(bits):
            if bit:
                op = op + ops[i]
        yield op

        

found = set()


def get_equi(G):

    print(G, G.structure_description())

    #print(G.perms)
    N = G.rank
    n = gap.Order(G, get=True)
    assert len(G) == n
    #_Hs = gap.AllSubgroups(G)
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

        #m, n = ops[0].shape
        #if m!=21 and n!=21:
        #    continue

        #for H0 in ops:
        for H0 in span_ops(ops):
          for H in [H0, H0.t]:
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
            css = CSSCode(Hx=H, Hz=H)
            if css.k:
                css.bz_distance()
            if css.d <= 2:
                continue
            #if css.n > 20 and css.d <= 4:
            #    continue

            #print(css, H.shape)
            s = str(css)
            if css.d > 6:
                s += "FOUND!"

            if s not in found:
                found.add(s)
                print(s)

#        smap = SMap()
#        col = 0
#        for op in ops:
#            smap[0, col] = op.shortstr()
#            col += op.shape[1] + 1
#        #smap[op.shape[0], 0] = "-"*col
#        print(smap)
#        print()
#
#      print()



def main():

    gap = Gap()

    if argv.GL:
        m = argv.get("m", 3)
        n = argv.get("n", 2)
        N = argv.get("N", None)
        _G = gap.PSL(m, n)
        G = gap.to_group(_G, smaller=False, N=N)
        get_equi(G)
        return

    if argv.Alternating:
        n = argv.get("n", 5)
        _G = gap.AlternatingGroup(n)
        G = gap.to_group(_G, smaller=False)
        get_equi(G)
        return

    if argv.Mathieu:
        n = argv.get("n", 11)
        _G = gap.MathieuGroup(n)
        G = gap.to_group(_G, smaller=False)
        get_equi(G)
        return

    n0 = argv.get("n0", 33)
    n1 = argv.get("n1", 60)
    for _G in allgroups(n0, n1):
        G = gap.to_group(_G, smaller=False)
        get_equi(G)








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





