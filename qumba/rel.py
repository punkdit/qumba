#!/usr/bin/env python

"""
Lagrangian relations 
also known as
Pauli flows

https://arxiv.org/abs/2105.06244v2

"""

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from random import choice, shuffle

import numpy

from qumba.matrix import Matrix, pullback
from qumba.symplectic import symplectic_form
from qumba.qcode import strop
from qumba.smap import SMap
from qumba.action import mulclose
from qumba.argv import argv
from qumba.construct import all_codes


def normalize(left, right, truncate=True):
    #print("normalize")
    assert left.shape[0] == right.shape[0]
    m, n = left.shape[1], right.shape[1]
    A = left.concatenate(right, axis=1)
    A = A.normal_form(truncate)
    left = A[:, :m]
    right = A[:, m:]
    return left, right


class Relation(object):

    def __init__(self, left, right=None):
        left = Matrix.promote(left)
        if right is None:
            right = Matrix.identity(left.shape[0])
        else:
            right = Matrix.promote(right)
        rows = left.shape[0]
        assert rows == right.shape[0]
        left, right = normalize(left, right)
        self.left = left
        self.right = right
        self.tgt = left.shape[1]
        self.src = right.shape[1]
        self.rows = rows
        self.A = left.concatenate(right, axis=1)
        self.shape = (left.shape, right.shape) # um..

    @classmethod
    def identity(cls, n):
        I = Matrix.identity(n)
        return cls(I, I)

    def __str__(self):
        w = self.left.shape[1]
        v = self.right.shape[1]
        smap = SMap()
        row = 0
        for row in range(self.left.shape[0]):
            smap[row,1] = "["
            smap[row,w+v+3] = "],"
            smap[row,w+2] = "|"
        smap[0,2] = str(self.left)
        smap[0,w+3] = str(self.right)
        smap[0,0] = "["
        smap[row,w+v+4] = "]"
        return str(smap)

    def __eq__(self, other):
        assert isinstance(other, self.__class__)
        assert self.tgt == other.tgt
        assert self.src == other.src
        if self.shape==other.shape and self.left==other.left and self.right==other.right:
            return True
        # just return False because we use normal_form
        return False
        # otherwise we do all this...
        lhs = other.A.t
        rhs = self.A.t
        u = lhs.solve(rhs)
        if u is None:
            return False
        v = rhs.solve(lhs)
        if v is None:
            return False
        assert (u*v).is_identity() and (v*u).is_identity()
        return True

    def __hash__(self):
        return hash((self.left, self.right))

    def __mul__(lhs, rhs):
        assert isinstance(rhs, lhs.__class__)
        assert lhs.src == rhs.tgt
        l, r = pullback(lhs.right.t, rhs.left.t)
        left = lhs.left.t * l
        right = rhs.right.t * r
        return lhs.__class__(left.t, right.t)

    def __matmul__(lhs, rhs):
        left = lhs.left.direct_sum(rhs.left)
        right = lhs.right.direct_sum(rhs.right)
        return lhs.__class__(left, right)

    @property
    #@cache # needs hash, needs _normal form
    def op(self):
        return self.__class__(self.right, self.left)

    @classmethod
    def get_swap(cls):
        z = zeros(2,2)
        i = Matrix.identity(2)
        l = z.concatenate(i)
        r = i.concatenate(z)
        swap = Relation(i, [[0,1],[1,0]])
        return swap


class Symplectic(Relation):
    def is_lagrangian(self):
        A = self.A
        m, nn = A.shape
        assert nn%2 == 0
        assert nn//2 == m, A.shape
        F = symplectic_form(m)
        # assert isotropic
        At = A.transpose()
        AFA = A * F * At
        return AFA.sum() == 0

    def __str__(self):
        left, right = self.left, self.right
        smap = SMap()
        smap[0,0] = strop(left)
        w = left.shape[1] // 2
        for i in range(left.shape[0]):
            smap[i,w] = "|"
        smap[0,w+1] = strop(right)
        return str(smap)

    @classmethod
    def get_swap(cls):
        z = zeros(2,2)
        i = Matrix.identity(2)
        l = z.concatenate(i)
        r = i.concatenate(z)
        swap = Symplectic(l.concatenate(r, axis=1).transpose())
        return swap



zeros = lambda a,b : Matrix.zeros((a,b))

def all_linear(tgt, src):
    for idxs in numpy.ndindex((2,)*(tgt*src)):
        left = numpy.array(idxs)
        left.shape = (tgt, src)
        left = left.transpose() # ?
        rel = Relation(left)
        yield rel

def all_subspaces(tgt, src):
    for idxs in numpy.ndindex((2,)*(tgt*src)):
        left = numpy.array(idxs)
        left.shape = (tgt, src)
        left = left.transpose()
        rel = Relation(left, zeros(src,0))
        yield rel


def test_linear():
    lins = [a.left for a in all_linear(3,3)]

    for trial in range(100):
        f = choice(lins)
        g = choice(lins)
        h = choice(lins)
        fg = f*g
        gh = g*h
        assert Relation(fg) == Relation(g)*Relation(f) # contravariant...
        assert ((Relation(fg)==Relation(gh)) 
            == (Relation(g)*Relation(f)==Relation(h)*Relation(g)))

    b_ = Relation([[1]], zeros(1,0))
    bb_ = Relation([[1,1]], zeros(1,0))
    bb_b = Relation([[1,1]], [[1]])
    _b = b_.op
    _bb = bb_.op
    b_bb = bb_b.op

    assert b_ * _b == b_ @ _b # elevator identity
    assert str(_b * b_) == "[[|]]"

    w_ = Relation([[0]], zeros(1,0))
    ww_ = Relation([[1,1]], zeros(1,0))
    ww_w = Relation([[1,1],[0,1]], [[0],[1]])

    _w = w_.op
    _ww = ww_.op
    w_ww = ww_w.op

    assert _w*b_bb == _w@_w
    r = (_w@_w)
    #print(r.A.shape)
    #print(r)
    #print(_w)

    assert b_ * _w == Relation([[1]],[[0]])

    assert _b@_b == _b * w_ww

    one = Relation(zeros(0,0), zeros(0,0))
    assert _b*w_ == one

    I = Relation([[1]], [[1]])
    assert I*I == I

    swap = Relation.get_swap()

    # commutative
    assert swap*bb_b == bb_b
    assert swap*ww_w == ww_w
    assert b_bb*swap == b_bb
    assert w_ww*swap == w_ww

    assert ww_w != bb_b

    # special
    assert b_bb*bb_b == I
    assert w_ww*ww_w == I

    # _assoc
    lhs = (bb_b @ I) * bb_b 
    rhs = (I @ bb_b) * bb_b 
    assert( lhs==rhs )

    # unital
    assert (I@_b)*bb_b == I
    assert (_b@I)*bb_b == I

    # copy
    assert bb_b * w_ == w_@w_

    # frobenius
    lhs = bb_b * b_bb
    rhs = (I @ b_bb) * (bb_b @ I)
    assert lhs==rhs
    rhs = (b_bb @ I) * (I @ bb_b)
    assert lhs==rhs

    lhs = ww_w * w_ww
    rhs = (I @ w_ww) * (ww_w @ I)
    assert lhs==rhs
    rhs = (w_ww @ I) * (I @ ww_w)
    assert lhs==rhs

    # bialgebra
    lhs = (b_bb @ b_bb) * (I @ swap @ I) * (ww_w @ ww_w)
    rhs = ww_w * b_bb
    assert lhs == rhs



def test_symplectic():
    one = Symplectic(zeros(0,0), zeros(0,0))
    #print(one)
    #print(one*one)
    assert one*one == one

    I = Symplectic.identity(2)
    h = Symplectic([[0,1],[1,0]])

    #print("h:")
    #print(h)
    hh = h*h

    assert h != I
    assert I==I
    assert I*I==I
    assert I*h==h*I==h
    assert h*h == I

    # black unit
    b_ = Symplectic([[0,1]], zeros(1,0))

    # phase=1
    b1 = Symplectic([[1,0],[1,1]])
    w1 = h*b1*h

    assert b1 != w1

    assert w1*w1 == I
    assert b1*b1 == I

    b1_ = b1*b_
    
    w1_ = h*b1_
    w_ = w1 * w1_

    _b1 = b1_.op
    _b = b_.op

    _w1 = w1_.op
    _w = w_.op

    assert h == w1*b1*w1
    assert h == b1*w1*b1

    assert _b * w_  == one

    assert _b1 == _w1
    assert _b != _b1
    assert _w != _w1

    swap = Symplectic.get_swap()
    
    assert swap != I@I
    assert swap*swap == I@I
    for a in [I,w1,b1,h]:
      for b in [I,w1,b1,h]:
        assert swap*(a@b) == (b@a)*swap

    #for v_ in all_subspaces(2, 1): # not Symplectic ...
    #    assert isinstance(v_@I, Symplectic)
    #    assert swap * (v_@I) == I@v_

    # copy
    bb_b = Symplectic([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,1],
    ], [
        [1,0],
        [1,0],
        [0,1],
    ])

    assert swap*bb_b == bb_b

    b_bb = bb_b.op
    ww_w = (h@h) * bb_b * h
    w_ww = ww_w.op

    assert ww_w != bb_b

    # special
    assert b_bb*bb_b == I
    assert w_ww*ww_w == I

    # _assoc
    lhs = (bb_b @ I) * bb_b 
    rhs = (I @ bb_b) * bb_b 
    assert( lhs==rhs )

    _b = b_.op
    w_ = h*b_

    # unital
    assert (I@_b)*bb_b == I
    assert (_b@I)*bb_b == I

    # copy
    assert bb_b * w_ == w_@w_

    # frobenius
    lhs = bb_b * b_bb
    rhs = (I @ b_bb) * (bb_b @ I)
    assert lhs==rhs
    rhs = (b_bb @ I) * (I @ bb_b)
    assert lhs==rhs

    lhs = ww_w * w_ww
    rhs = (I @ w_ww) * (ww_w @ I)
    assert lhs==rhs
    rhs = (w_ww @ I) * (I @ ww_w)
    assert lhs==rhs

    # bialgebra
    lhs = ww_w * b_bb
    rhs = (b_bb @ b_bb) * (I @ swap @ I) * (ww_w @ ww_w)
    assert lhs == rhs

    lhs = b_bb * (h@h) * bb_b
    rhs = b_ @ _b
    assert lhs==rhs

    assert( h == b1 * w_ww * (b1 @ b1_) )

    assert( b1 * b_bb == b_bb * (b1@I) )
    assert( b1 * b_bb == b_bb * (I@b1) )

    assert( w1 * w_ww == w_ww * (w1@I) )
    assert( w1 * w_ww == w_ww * (I@w1) )

    cup = bb_b * b_
    assert cup == ww_w * w_
    cap = _b * b_bb
    assert cap == _w * w_ww

    # snakes
    assert (I @ cap) * (cup @ I) == I
    assert (cap @ I) * (I @ cup) == I

    cnot = (I @ b_bb) * (ww_w @ I)
    assert cnot != I@I
    assert cnot * cnot == I@I

    gen = [cnot, w1@I, I@w1, h@I, I@h]
    G = list(gen)
    for _ in range(100):
        a = choice(G)
        b = choice(gen)
        G.append(a*b)

    for g in G:
        assert g * g.op == I@I
        assert g.op * g == I@I

    for a in G:
      for b in G:
        if a==b and hash(a) != hash(b):
            assert 0

    #print("w_")
    #print(w_)
    #print()
    assert str(w_) == "X| "

    #return

    G = mulclose(gen, verbose=True)
    assert len(G) == 720 # Sp(4,2)

    for g in G:
        assert g.is_lagrangian()

    for g in list(G)[:10]:
        str(g)
        #print(g)
        #print()
        #print(g * (w_@I))
        #print()
    
    bone = w_ * _w
    gen = [cnot, w1@I, I@w1, h@I, I@h, bone@I, I@bone]

    M = mulclose(gen, verbose=True)
    assert len(M) == 2295

    for m in M:
        assert m.is_lagrangian()


def test():
    test_linear()
    test_symplectic()


def main():

    n = argv.get("n", 4)

    found = set()
    for code in all_codes(n, 0, 0):
        #print(code.longstr())
        H = code.H
        Ht = H.t
        l, r = H[:, :n//2], H[:, n//2:]
        rel = Symplectic(l.t, r.t)
        assert rel.is_lagrangian()
        found.add(rel)
    found = list(found)
    print("found:", len(found))




if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "test"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))



