#!/usr/bin/env python3

"""
Spans of matrices.

"""


from random import shuffle, choice, randint
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

from qumba.solve import zeros2, solve
from qumba.smap import SMap
from qumba.matrix import Matrix, DEFAULT_P, pullback
from qumba.symplectic import SymplecticSpace


def reduce(left, right):
    # is this joint monotinicity ? or what.. not sure
    tgt, src = left.shape[0], right.shape[0]
    if tgt == src == 0:
        left, right = Matrix([[]], shape=(0,0)), Matrix([[]], shape=(0,0))
        return left, right
    m = left.concatenate(right)
    assert m.shape[0] == tgt+src
    #m = m.t.row_reduce().t
    m = m.t.linear_independent().t
    assert m.shape[0] == tgt+src
    left, right = m[:tgt, :], m[tgt:, :]
    return left, right


class Span(object):
    def __init__(self, left=None, right=None):
        if right is None:
            n = left.shape[1]
            right = Matrix.identity(n)
        if left is None:
            n = right.shape[1]
            left = Matrix.identity(n)
        left, right = reduce(left, right)
        assert left.shape[1] == right.shape[1]
        self.left = left
        self.right = right
        self.hom = (left.shape[0], right.shape[0]) # tgt, src
        self.n = left.shape[1]

    @property
    def subspace(self):
        return self.left.concatenate(self.right)

    @classmethod
    def identity(cls, n):
        I = Matrix.identity(n)
        return Span(I, I)

    def __str__(self):
        left = str(self.left).replace("\n", "")
        right = str(self.right).replace("\n", "")
        return "Span(%s, %s)"%(left, right)
        #return "Span(%s%s, %s%s)"%(left, self.left.shape, right, self.right.shape)
    __repr__ = __str__

    def relstr(self):
        smap = SMap()
        m, n = self.hom
        ab = self.left.concatenate(self.right)
        for row,u in enumerate(numpy.ndindex((2,)*m)):
            for col,v in enumerate(numpy.ndindex((2,)*n)):
                uv = numpy.array(u+v)
                uv.shape = (m+n,1)
                if solve(ab.A, uv) is not None:
                    smap[row, col] = '*'
                else:
                    smap[row, col] = '.'
        return str(smap)

    def eq(self, other):
        # strict equality...
        return self.hom==other.hom and self.n==other.n and self.left==other.left and self.right==other.right

    def __hash__(self):
        return hash(self.relstr())

    def __matmul__(self, other):
        assert isinstance(other, Span)
        left = self.left.direct_sum(other.left)
        right = self.right.direct_sum(other.right)
        return Span(left, right)

    def __mul__(self, other):
        assert isinstance(other, Span)
        assert self.hom[1] == other.hom[0]
        a, b = self.right, other.left
        c, d = pullback(a, b)
        left = self.left * c
        right = other.right * d
        return Span(left, right)

    def get_hom(self, other):
        a, b = self.left, self.right
        c, d = other.left, other.right
        ab = a.concatenate(b)
        cd = c.concatenate(d)
        cdi = cd.pseudo_inverse()
        f = cdi * ab
        if cd*f == ab:
            return f

    def is_iso(self, other):
        f = self.get_hom(other)
        if f is None:
            return False
        m, n = f.shape
        if m!=n:
            return False
        return f.rank() == m

    def __eq__(self, other):
        #return self.eq(other) or self.is_iso(other)
        return self.eq(other) or self.relstr() == other.relstr()

    def is_function(self):
        m, n = self.right.shape
        return m==n

    @property
    def t(self):
        return Span(self.right, self.left)

    @classmethod
    def black(cls, m, n):
        if m==1:
            # add the inputs
            A = zeros2(1, n)
            A[:] = 1
            left = Matrix(A)
            span = Span(left, None)
        elif n==1:
            span = cls.black(n, m)
            span = span.t
        else:
            span = cls.black(m, 1) * cls.black(1, n) # recurse
        return span

    @classmethod
    def white(cls, m, n):
        if n==1:
            # copy the inputs
            A = zeros2(m, 1)
            A[:] = 1
            left = Matrix(A)
            span = Span(left, None)
        elif m==1:
            span = cls.white(n, m)
            span = span.t
        else:
            span = cls.white(m, 1) * cls.white(1, n) # recurse
        return span

    def _is_symplectic(self):
        m, n = self.hom
        assert self.n%2 == 0
        assert m%2 == 0
        assert n%2 == 0
        top = SymplecticSpace(self.n//2)
        m = SymplecticSpace(m//2)
        n = SymplecticSpace(n//2)
        left, right = self.left, self.right
        # left : top --> m
        # right : top --> n
        left.t * m.F * left == top.F 
        right.t * n.F * right == top.F
        return left.t * m.F * left == top.F and right.t * n.F * right == top.F
        
    def is_lagrangian(self):
        m, n = self.hom
        if self.n != (m+n)//2: # maximal
            return False
        return self.is_isotropic()

    def is_isotropic(self):
        m, n = self.hom
        #assert self.n%2 == 0
        assert m%2 == 0
        assert n%2 == 0
        space = SymplecticSpace((m+n)//2)
        left, right = self.left, self.right
        A = self.subspace
        F = space.F
        return (A.t * F * A).sum() == 0
        

def test_special_comm_frob(i, swap, _g, g_gg, g_, gg_g):
    assert g_gg * (i @ g_gg) == g_gg * (g_gg @ i)
    assert g_gg * (i @ g_) == i
    assert g_gg * (g_ @ i) == i
    assert g_gg * swap == g_gg
    assert (i @ gg_g) * gg_g == (gg_g@i)*gg_g
    assert (i@_g) * gg_g == i
    assert (_g@i) * gg_g == i
    assert swap * gg_g == gg_g
    assert g_gg * gg_g == i


def test():

    black = Span.black
    white = Span.white

    b_ = black(1, 0)
    _b = black(0, 1)
    _w = white(0, 1)
    w_ = white(1, 0)
    b_bb = black(1, 2)
    bb_b = black(2, 1)
    ww_w = white(2, 1)
    w_ww = white(1, 2)
    _ww = white(0, 2)
    assert _ww == black(0, 2)

    assert _b*b_ == black(0, 0)

    g = _ww * (b_ @ w_)
    assert g == black(0, 0)

    I = Span(Matrix([[1]]))
    swap = Matrix([[0,1],[1,0]])
    swap = Span(swap)

    v = b_bb * ww_w

    assert (I@I) == Span(Matrix.identity(2))

    assert I == black(1, 1)
    assert I == white(1, 1)

    rhs = black(1,2)*black(2,1)
    assert I == black(1, 2)*black(2,1)
    assert I == white(1, 2)*white(2,1)

    assert black(2, 2) == black(2,1)*black(1,2)
    assert white(2, 2) == white(2,1)*white(1,2)

    test_special_comm_frob(I, swap, _w, w_ww, w_, ww_w)
    #test_special_comm_frob(I, swap, _b, b_bb, b_, bb_b)

    assert (b_bb * (I @ b_bb)) == ( b_bb * (b_bb @ I) )
    assert b_bb * (I @ b_) == I
    assert b_bb * (b_ @ I) == I
    assert (b_bb * swap) == ( b_bb )
    assert ((I @ bb_b) * bb_b) == ( (bb_b@I)*bb_b )
    assert (I@_b) * bb_b == I
    assert (_b@I) * bb_b == I
    assert (swap * bb_b) == ( bb_b )
    assert b_bb * bb_b == I

    # bialgebra
    lhs = bb_b * w_ww
    rhs = (w_ww @ w_ww) * (I @ swap @ I) * (bb_b @ bb_b)
    assert lhs == (rhs)

    # bialgebra
    lhs = ww_w * b_bb
    rhs = (b_bb @ b_bb) * (I @ swap @ I) * (ww_w @ ww_w)
    assert lhs == (rhs)

    #print(black(0, 1).hom)
    #print(white(0, 1).hom)
    #print(black(0, 3))
    #print(white(0, 3))
    #assert black(0, 2) == white(0, 2) # yes, but...

    if 0:
        print("_w")
        print( _w.relstr() )
        print("_b")
        print( _b.relstr() )
        print("ww_w")
        print( ww_w.relstr() )
        print("bb_b")
        print( bb_b.relstr() )
        print("w_ww")
        print( w_ww.relstr() )
        print("b_bb")
        print( b_bb.relstr() )

    g = (I @ swap @ I) * (bb_b @ ww_w)
    assert g.is_lagrangian()

    g = (b_bb * ww_w) @ (w_ww * bb_b)
    assert g.is_lagrangian()

    g = (b_bb * ww_w) @ (b_bb * ww_w)
    assert not g.is_lagrangian()

    ww_ww = white(2, 2)
    assert ww_ww == ww_w*w_ww
    bb_bb = black(2, 2)
    assert bb_bb == bb_b*b_bb
    g = (I@swap@I) * (ww_ww @ bb_bb) * (I@swap@I)
    assert g.is_lagrangian()
    g = (I@swap@I) * (I@I @ bb_bb) * (I@swap@I)
    assert not g.is_lagrangian()

    g = _b @ _w
    assert g.is_lagrangian()
    g = b_ @ w_
    assert g.is_lagrangian()

    found = set()
    cup = black(2,0)
    cap = black(0,2)
    for l in [b_@w_, w_@b_, cup]:
      for r in [_b@_w, _w@_b, cap]:
        g = l*r
        found.add(g)
        #print(g.relstr())
        #print()
    assert len(found) == 9

    # -------------------------------
    # test S

    S = (I @ b_bb) * (ww_w @ I)
    assert S.is_lagrangian()
    assert S.is_function()
    assert S*S == I@I

    H = swap
    assert H.is_lagrangian()
    assert H.is_function()
    assert (H*H) == ( I@I )

    Q = (I @ w_ww) * (bb_b @ I)
    assert Q.is_lagrangian()
    assert Q.is_function()
    assert Q*Q == I@I

    assert Q  == H*S*H
    assert (H*S).is_lagrangian()
    assert (S*H).is_lagrangian()

    # -------------------------------
    # test CNOT

    rhs = (I @ swap @ I) * (bb_b @ ww_w)
    assert rhs.is_lagrangian()
    rhs = rhs @ I @ I
    assert rhs.is_lagrangian()

    lhs = (w_ww @ b_bb) * (I @ swap @ I)
    assert lhs.is_lagrangian()
    lhs = I @ I @ lhs
    assert lhs.is_lagrangian()
    assert not (w_ww @ b_bb).is_lagrangian()

    CNOT = lhs * rhs

    s = SymplecticSpace(2)
    h = Span(s.get_CNOT(1, 0))
    assert h.is_lagrangian()
    assert CNOT == (h)
    assert CNOT.is_lagrangian()
    assert CNOT.is_function()
    
    h = Span(s.get_CNOT(0, 1))
    assert CNOT != h
    assert h.is_lagrangian()
    
    # -------------------------------
    # test CZ

    rhs = (I @ swap @ I) * (ww_w @ bb_b)
    assert rhs.is_lagrangian()
    rhs = rhs @ I @ I
    assert rhs.is_lagrangian()

    lhs = (w_ww @ b_bb) * (I @ swap @ I)
    assert lhs.is_lagrangian()
    lhs = I @ I @ lhs
    assert lhs.is_lagrangian()

    CZ = lhs * (I@I@H@I@I) * rhs
    assert CNOT != CZ
    assert CZ.is_function()

    s = SymplecticSpace(2)
    h = Span(s.get_CZ(0, 1))
    assert h.is_lagrangian()
    assert CZ == h
    assert CZ.is_lagrangian()
    

def test_complete():
    from bruhat.dev.geometry import all_codes
    n = 2
    I = Span.identity(n)
    s = SymplecticSpace(n)
    F = s.F
    found = set()
    count = 0
    for A in all_codes(n, 2*n):
        M = Matrix(A).t
        left = M[:n, :]
        right = M[n:, :]
        g = Span(left, right)
        if not g.is_lagrangian():
            continue
        if g.is_function():
            print("is_function:")
            print(g.relstr())
            print()
        else:
            assert g*g==g
        found.add(g)
        if len(found)%100 == 0:
            print(len(found))
        #print(span.relstr())
        #print()
        count += 1
    print(len(found), count)

    monoid = list(found)
    for i in range(1000):
        a = monoid[randint(0,len(found)-1)]
        b = monoid[randint(0,len(found)-1)]
        assert a*b in found



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
        import cProfile as profile
        profile.run("%s()"%name)

    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)


