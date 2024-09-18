#!/usr/bin/env python

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from random import choice

import numpy

from qumba.matrix import Matrix, pullback
from qumba.smap import SMap
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

    # phase=1
    b1 = Relation([[1,1],[0,1]])
    w1 = h*b1*h

    assert b1 != w1

    assert w1*w1 == I # S gate
    assert b1*b1 == I # R gate (=HSH)

    b1_ = Relation([[1],[0]], zeros(0,1))
    b_ = b1 * b1_
    
    w1_ = h*b1_
    w_ = w1 * w1_

    #print(b1_ == w1_) # False
    #print(b_ == w_) # False

    assert(b_ == b1_) # True
    assert(w_ == w1_) # True

    _b1 = b1_.op
    _b = b_.op

    _w1 = w1_.op
    _w = w_.op

    assert h == w1*b1*w1
    assert h == b1*w1*b1

    assert _b * w_  == one

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

    # special
    assert b_bb*bb_b == I
    assert w_ww*ww_w == I

    # _assoc
    lhs = (bb_b @ I) * bb_b 
    rhs = (I @ bb_b) * bb_b 
    assert( lhs==rhs )

    b_ = Relation([[0],[1]], zeros(0,1))
    _b = b_.op
    lhs = (I@_b)*bb_b 
    assert (I@_b)*bb_b == I
    assert (_b@I)*bb_b == I

    w_ = h*b_
    assert bb_b * w_ == w_@w_

    assert ww_w != bb_b

    assert swap*bb_b == bb_b
    assert swap*ww_w == ww_w

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

    b1_ = b1 * b_
    assert( h == b1 * w_ww * (b1 @ b1_) )



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



