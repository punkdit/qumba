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


def normalize(left, right):
    #print("normalize")
    assert left.shape[0] == right.shape[0]
    m, n = left.shape[1], right.shape[1]
    A = left.concatenate(right, axis=1)
    A = A.normal_form()
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
        assert left.shape[0] == right.shape[0]
        left, right = normalize(left, right)
        self.left = left
        self.right = right
        #self.shape = left.shape + (right.shape[1],)
        #self.tgt = self.shape[0]
        #self.src = self.shape[2]
        self.tgt = left.shape[1]
        self.src = right.shape[1]
        self.A = left.concatenate(right, axis=1)
        self.shape = (left.shape, right.shape)

    @classmethod
    def identity(cls, n):
        I = Matrix.identity(n)
        return Relation(I, I)

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
        #s = "[%s <--- %s ---> %s]\n"%self.shape
        #s += "left: %s\n%s\nright: %s\n%s"%(
        #    self.left.shape, self.left.A, self.right.shape, self.right.A)
        #return "="*10 + '\n' + s + "\n" + "="*10

    def __eq__(self, other):
        assert isinstance(other, Relation)
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
        assert isinstance(rhs, Relation)
        assert lhs.src == rhs.tgt
        l, r = pullback(lhs.right.t, rhs.left.t)
        left = lhs.left.t * l
        right = rhs.right.t * r
        return Relation(left.t, right.t)

    def __matmul__(lhs, rhs):
        left = lhs.left.direct_sum(rhs.left)
        right = lhs.right.direct_sum(rhs.right)
        return Relation(left, right)

    @property
    #@cache # needs hash, needs _normal form
    def op(self):
        return Relation(self.right, self.left)



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
        #assert ((Relation(fg)==Relation(gh)) 
        #    == (Relation(f)*Relation(g)==Relation(g)*Relation(h)))

    #b_ = Relation(


def test_symplectic():
    one = Relation(zeros(0,0), zeros(0,0))
    #print(one)
    #print(one*one)
    assert one*one == one

    I = Relation.identity(2)
    h = Relation([[0,1],[1,0]])

    #print("h:")
    #print(h)
    hh = h*h

    assert h != I
    assert I==I
    assert I*I==I
    assert I*h==h*I==h
    assert h*h == I

    # black unit
    b_ = Relation([[0,1]], zeros(1,0))

    # phase=1
    b1 = Relation([[1,0],[1,1]])
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

    z = zeros(2,2)
    i = Matrix.identity(2)
    l = z.concatenate(i)
    r = i.concatenate(z)
    swap = Relation(l.concatenate(r, axis=1).transpose())
    
    assert swap != I@I
    assert swap*swap == I@I
    for a in [I,w1,b1,h]:
      for b in [I,w1,b1,h]:
        assert swap*(a@b) == (b@a)*swap

    for v_ in all_subspaces(2, 1):
        assert swap * (v_@I) == I@v_

    # copy
#    bb_b = Relation([
#        [1,0,0],
#        [0,0,1],
#        [0,1,0],
#        [0,0,1],
#    ], [
#        [1,1,0],
#        [0,0,1],
#    ])

    bb_b = Relation([
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
        rel = Relation(l.t, r.t)
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



