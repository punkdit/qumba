#!/usr/bin/env python

"""
Use linear relations for CSS circuits (aka phase-free ZX-calculus).

"""

import time
from functools import reduce, cache

#from qumba.qcode import QCode, SymplecticSpace
from qumba.matrix import Matrix
from qumba.rel import Relation # linear relations
from qumba.csscode import CSSCode

from qumba.argv import argv
from qumba.util import choose


class Space:
    @cache
    def __new__(cls, n):
        ob = object.__new__(cls)
        return ob

    def __init__(self, n):
        self.n = n


class Hom:
    @cache
    def __new__(cls, m, n=None):
        ob = object.__new__(cls)
        return ob

    def __init__(self, m, n=None):
        if n is None:
            n = m
        self.m = m
        self.n = n

    def __str__(self):
        return "(%s<--%s)"%(self.m, self.n)
    __repr__ = __str__

    def PX(self, *idxs):
        "prepare X state: |+..+>"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        else:
            idxs = list(idxs)
        idxs.sort(reverse=True)
        ops = [I]*n
        for i in idxs:
            ops.insert(i, w_)
        op = reduce(matmul, ops)
        return op

    def PZ(self, *idxs):
        "prepare Z state: |0..0>"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        else:
            idxs = list(idxs)
        idxs.sort(reverse=True)
        ops = [I]*n
        for i in idxs:
            ops.insert(i, b_)
        op = reduce(matmul, ops)
        return op

    def CX(self, idx=0, jdx=1):
        assert self.m == self.n, "wrong Hom %s"%(self,)
        


def test():
    pass

    

def reed_muller(r=1, m=4):

    assert 0<=r<=m, "r=%s, m=%d"%(r, m)

    n = 2**m # length

    one = Matrix([1]*n).A
    basis = [one]

    vs = [[] for i in range(m)]
    for i in range(2**m):
        for j in range(m):
            vs[j].append(i%2)
            i >>= 1
        assert i==0

    vs = [Matrix(v).A for v in vs]

    for k in range(r):
        for items in choose(vs, k+1):
            v = one
            #print(items)
            for u in items:
                v = v*u
            basis.append(v)

    H = Matrix(basis)
    H = H.linear_independent()
    return H



def test_rm():
    #from qumba.construct import reed_muller

    #code = reed_muller(2, 6)
    #code = code.to_css()

    #if code.k:
    #    code.bz_distance()

    #print(code)

    lookup = {}
    for m in [1,2,3,4,5]:
        Hs = [reed_muller(l, m) for l in range(m+1)]
        for idx,H in enumerate(Hs):
          for jdx,J in enumerate(Hs):
            A = H*J.t
            #print(int(A.max() == 0), end=' ')
            if A.max() > 0:
                continue
            css = CSSCode(Hx=H, Hz=J)
            if css.k==0:
                pass
            elif css.n < 64:
                css.bz_distance()
            else: 
                distance_z3_css(css)
            lookup[m,idx,jdx] = css
            print(str(css).ljust(14), end=' ', flush=True)
          print()
        print()

    #print(list(lookup.keys()))

    found = set()
    for m in [1,2,3,4]:
      for idx in range(m+1):
        for jdx in range(m+1):
            key = (m, idx, jdx)
            if key not in lookup:
                continue
            css = lookup[key]
            H = css.Hx
            #if H in found:
            #    continue
            found.add(H)
            p = H.get_tutte()
            #print(H)
            print(key, css, p)
            #print()

      print()






if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)
        ra.seed(_seed)

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


