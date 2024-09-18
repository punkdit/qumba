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
from random import choice

import numpy

from qumba.matrix import Matrix, pullback
from qumba.smap import SMap
from qumba.action import mulclose
from qumba.argv import argv


def normalize(left, right):
    #print("normalize")
    assert left.shape[1] == right.shape[1]
    m, n = left.shape[0], right.shape[0]
    A = left.concatenate(right)
    #print(A)
    assert A.shape[1] == left.shape[1]
    A = A.t.row_reduce().t
    left = A[:m, :]
    right = A[m:, :]
    #print("-->")
    #print(A)
    return left, right


class Relation(object):

    def __init__(self, left, right=None):
        left = Matrix.promote(left)
        if right is None:
            right = Matrix.identity(left.shape[1])
        else:
            right = Matrix.promote(right)
        assert left.shape[1] == right.shape[1]
        left, right = normalize(left, right)
        self.left = left
        self.right = right
        self.shape = left.shape + (right.shape[0],)
        self.tgt = self.shape[0]
        self.src = self.shape[2]

    @classmethod
    def identity(cls, n):
        I = Matrix.identity(n)
        return Relation(I, I)

    def __str__(self):
        s = "[%s <--- %s ---> %s]\n"%self.shape
        s += "left: %s\n%s\nright: %s\n%s"%(
            self.left.shape, self.left.A, self.right.shape, self.right.A)
        return "="*10 + '\n' + s + "\n" + "="*10

    def __eq__(self, other):
        assert isinstance(other, Relation)
        assert self.tgt == other.tgt
        assert self.src == other.src
        if self.shape==other.shape and self.left==other.left and self.right==other.right:
            return True
        lhs = other.left.concatenate(other.right)
        #assert lhs.shape[0] == self.left.shape[0] + other.left.shape[0]
        rhs = self.left.concatenate(self.right)
        #assert rhs.shape[0] == self.right.shape[0] + other.right.shape[0]
        #print("solve:", lhs.shape, rhs.shape)
        u = lhs.solve(rhs)
        if u is None:
            return False
        v = rhs.solve(lhs)
        if v is None:
            return False
        return (u*v).is_identity() and (v*u).is_identity()

    # fail...
    def __hash__(self):
        return hash((self.left, self.right))

    def __mul__(lhs, rhs):
        assert isinstance(rhs, Relation)
        assert lhs.src == rhs.tgt
        l, r = pullback(lhs.right, rhs.left)
        left = lhs.left * l
        right = rhs.right * r
        return Relation(left, right)

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
        rel = Relation(left)
        yield rel

def all_subspaces(tgt, src):
    for idxs in numpy.ndindex((2,)*(tgt*src)):
        left = numpy.array(idxs)
        left.shape = (tgt, src)
        rel = Relation(left, zeros(0,src))
        yield rel


def test():
    lins = [a.left for a in all_linear(3,3)]
    for trial in range(100):
        f = choice(lins)
        g = choice(lins)
        h = choice(lins)
        fg = f*g
        gh = g*h
        assert Relation(fg) == Relation(f)*Relation(g)
        assert ((Relation(fg)==Relation(gh)) 
            == (Relation(f)*Relation(g)==Relation(g)*Relation(h)))
    print("OK")


def main():
    #test()

    n = 2 # qubits

    one = Relation(zeros(0,0), zeros(0,0))
    #print(one)
    #print(one*one)
    assert one*one == one

    I = Relation.identity(n)
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
    b_ = Relation([[0],[1]], zeros(0,1))

    # phase=1
    b1 = Relation([[1,1],[0,1]])
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

    z = zeros(n,n)
    i = Matrix.identity(n)
    l = z.concatenate(i)
    r = i.concatenate(z)
    swap = Relation(l.concatenate(r, axis=1))
    
    assert swap != I@I
    assert swap*swap == I@I
    for a in [I,w1,b1,h]:
      for b in [I,w1,b1,h]:
        assert swap*(a@b) == (b@a)*swap

    for v_ in all_subspaces(2, 1):
        assert swap * (v_@I) == I@v_

    # copy
    bb_b = Relation([
        [1,0,0],
        [0,0,1],
        [0,1,0],
        [0,0,1],
    ], [
        [1,1,0],
        [0,0,1],
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
    G = mulclose(gen, maxsize=100)

    for g in G:
        assert g * g.op == I@I
        assert g.op * g == I@I

    if 0: # SLOW
    
    #    G = mulclose(gen, maxsize=100)
    #    for g in G:
    #      for h in G:
    #        if g==h:
    #            assert hash(g) == hash(h)
    
        found = list(gen)
        while 1:
            bdy = []
            for g in found:
                for h in gen:
                    gh = g*h
                    if gh not in found and gh not in bdy:
                        bdy.append(gh)
            if not bdy:
                break
            found += bdy
            print(len(found))

        assert len(found) == 720 # Sp(4,2)



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



