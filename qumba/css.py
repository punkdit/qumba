#!/usr/bin/env python

"""
Use linear relations for CSS circuits (aka phase-free ZX-calculus).

"""

import time
from functools import reduce, cache
from operator import matmul

import numpy

#from qumba.qcode import QCode, SymplecticSpace
#from qumba import lin
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

    _w = Relation.white(0, 1)
    w_ = Relation.white(1, 0)
    _b = Relation.black(0, 1)
    b_ = Relation.black(1, 0)
    I = Relation.identity(1)

    @cache # singleton
    def __new__(cls, m, n=None):
        ob = object.__new__(cls)
        return ob

    def __init__(self, m, n=None):
        if n is None:
            n = m
        self.m = m
        self.n = n

    @property
    def op(self):
        return Hom(self.n, self.m)

    def __str__(self):
        return "(%s<--%s)"%(self.m, self.n)
    __repr__ = __str__

    def __getitem__(self, i):
        # i am a shape tuple
        return (self.m, self.n)[i]

    def PX(self, *idxs):
        "prepare X state: |+> on idxs"
        m, n = self
        assert m>n, "wrong Hom %s"%(self,)
        if not idxs:
            idxs = list(range(m-n))
        else:
            idxs = list(idxs)
        assert m == n+len(idxs), "wrong Hom %s"%(self,)
        idxs.sort(reverse=True)
        rels = [self.I]*n
        for i in idxs:
            assert 0<=i<m
            rels.insert(i, self.w_)
        assert len(rels) == m
        rel = reduce(matmul, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n), (str(rel), str(self))
        return rel

    def PZ(self, *idxs):
        "prepare Z state: |0> on idxs"
        m, n = self
        assert m>n, "wrong Hom %s"%(self,)
        if not idxs:
            idxs = list(range(m-n))
        else:
            idxs = list(idxs)
        assert m == n+len(idxs), "wrong Hom %s"%(self,)
        idxs.sort(reverse=True)
        rels = [self.I]*n
        for i in idxs:
            assert 0<=i<m
            rels.insert(i, self.b_)
        assert len(rels) == m
        rel = reduce(matmul, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n)
        return rel

    def MX(self, *idxs):
        "postselect <0| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.op.PX(*idxs).op
        return rel

    def MZ(self, *idxs):
        "postselect <+| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.op.PZ(*idxs).op
        return rel

    def CX(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        lhs = Matrix.identity(m)
        rhs = numpy.identity(m, dtype=int)
        rhs[jdx, idx] = 1
        rhs = Matrix(rhs)
        return Relation(lhs, rhs)

    def get_perm(self, idxs):
        """ send idxs[j]<---j """
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m == len(idxs), "wrong Hom %s"%(self,)
        assert len(set(idxs)) == m
        lhs = Matrix.identity(m)
        rhs = Matrix.get_perm(idxs)
        return Relation(lhs, rhs)

    def SWAP(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        f = list(range(self.n))
        f[idx], f[jdx] = f[jdx], f[idx]
        return self.get_perm(f)



def test():
    hom = Hom(3, 2)

    for i in range(3):
        c = hom.PX(i)
        c = hom.PZ(i)

    hom = Hom(2, 3)
    for i in range(3):
        c = hom.MX(i)
        c = hom.MZ(i)

    # test get_perm ----------------------------------------

    rel = Hom(3).get_perm([1,2,0])
    rhs = Hom(3,2).PX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).PX(1) * Hom(2,2).SWAP()
    assert lhs == rhs
    
    rel = Hom(3).get_perm([2, 0, 1])
    rhs = Hom(3,2).PX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).PX(2)
    assert lhs == rhs
    
    # test CX ----------------------------------------

    w_ww = Relation.white(1, 2)
    bb_b = Relation.black(2, 1)
    identity = Relation.identity(1)
    CX = (w_ww @ identity) * (identity @ bb_b)

    space = Hom(2)
    assert CX == space.CX()

    cx01 = space.CX(0, 1)
    cx10 = space.CX(1, 0)

    rel = cx01*cx10*cx01
    assert rel == space.SWAP()

    n = 3
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        cx = Hom(n).CX(i,j)
        lhs = Hom(n-1, n).MZ(i) * cx
        rhs = Hom(n-1, n).MZ(i)
        assert lhs == rhs
    
        lhs = Hom(n-1, n).MX(j) * cx
        rhs = Hom(n-1, n).MX(j)
        assert lhs == rhs

    I = Relation.identity(1)
    _ww = Relation.white(0, 3)
    _bb = Relation.black(0, 3)
    
    print(_ww, _ww.shape)
    print(_ww @ I)

    print(_bb, _bb.shape)
    print(_bb @ I)


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


