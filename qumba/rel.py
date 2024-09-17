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
        #left, right = normalize(left, right)
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
    w1 = Relation([[1,1],[0,1]])
    #b1 = Relation([[1,1],[1,0]])

    b1 = h*w1*h
    #print("b1")
    #print(b1)

    #print("b2")
    #print(b1*b1)

    assert b1 != w1

    assert w1*w1 == I # S gate
    assert b1*b1 == I # R gate (=HSH)

    b1_ = Relation([[1],[0]], zeros(0,1))
    b_ = b1 * b1_
    
    w1_ = h*b1_
    w_ = w1 * w1_

    #print(b1_ == w1_) # False
    #print(b_ == w_) # True

    _b1 = b1_.op
    _b = b_.op

    _w1 = w1_.op
    _w = w_.op

    assert h == w1*b1*w1
    assert h == b1*w1*b1

    #print( _b * w_  == one) # hmm

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

    # copy
    #bb_b = Relation(i.concatenate(i))
    #bb_b = Relation(Matrix.identity(2*n), i.concatenate(i,axis=1))

    #print("w_")
    #print(w_)

    for v_ in all_subspaces(2, 1):
        assert swap * (v_@I) == I@v_

    print("search:")
    for bb_b in all_linear(4, 2):
        if swap*bb_b != bb_b:
            continue

        print("symmetric")
        b_bb = bb_b.op
        if b_bb*bb_b != I:
            continue

        print("special")
        #print(b_bb)

        lhs = (bb_b @ I) * bb_b 
        rhs = (I @ bb_b) * bb_b 
        if lhs==rhs:
            print("assoc")

        return

        for b_ in all_subspaces(2, 1):
            #print(b_)
            _b = b_.op
            #print(I@_b)
            lhs = (I@_b)*bb_b 
            print( (I@_b).shape )
            print(lhs.shape, I.shape)
            if (I@_b)*bb_b != I:
                continue
            assert (_b@I)*bb_b == I

            #print(bb_b)

            print( (bb_b*w_).shape, (w_@w_).shape )
    
            w_ = h*b_
            if bb_b * w_ == w_@w_:
                print("bb_b:")
                print(bb_b)

    print()
    return

    b_bb = bb_b.op

    print("bb_b")
    print(bb_b)


    ww_w = (h@h) * bb_b * h
    w_ww = ww_w.op

    #assert ww_w != bb_b

    assert swap*bb_b == bb_b
    assert swap*ww_w == ww_w

    #return

    #assert bb_b * w_ == w_@w_
    print(w_)
    #print(bb_b)

    lhs = bb_b * w_
    rhs = w_@w_

    print(lhs)
    print(rhs)

    #print(lhs == rhs)





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



