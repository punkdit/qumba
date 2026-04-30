#!/usr/bin/env python

"""
Linear relations, or jointly-monic spans

see also: lagrel.py

"""

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul, lshift
from random import choice, shuffle, randint

import numpy

from qumba.matrix import Matrix, pullback
from qumba.symplectic import symplectic_form
from qumba.qcode import strop, QCode, SymplecticSpace
from qumba.smap import SMap
from qumba.action import mulclose, mulclose_names
from qumba.syntax import Syntax
from qumba.argv import argv
from qumba import construct


def normalize(left, right, truncate=True):
    #print("normalize")
    assert left.shape[0] == right.shape[0]
    m, n = left.shape[1], right.shape[1]
    A = left.concatenate(right, axis=1)
    A = A.normal_form(truncate)
    left = A[:, :m]
    right = A[:, m:]
    return left, right

zeros = lambda a,b : Matrix.zeros((a,b))



class Relation:
    """ A linear relation
    """

    def __init__(self, left, right=None, p=2):
        left = Matrix.promote(left)
        if right is None:
            right = Matrix.identity(left.shape[0])
        else:
            right = Matrix.promote(right)
        assert left.shape[0] == right.shape[0], "%s %s"%(left.shape, right.shape)
        self._left = left
        self._right = right
        #if left.shape[1] or right.shape[1]: # ?!?!
        left, right = normalize(left, right)
        rank = left.shape[0]
        self.left = left
        self.right = right
        self.tgt = left.shape[1]
        self.src = right.shape[1]
        self.rank = rank
        self.A = left.concatenate(right, axis=1)
        #B = self._left.concatenate(self._right, axis=1)
        #AB = self.A.intersect(B)
        #assert len(AB) == rank # yes
        self.shape = (rank, self.tgt, self.src)
        self.p = p

    @property
    def nf(self): # normal form
        return self.__class__(self.left, self.right)

    def get_left(self, r): # __mul__ ?
        assert isinstance(r, Matrix)
        if len(r.shape)==1:
            n = len(r)
            r = r.reshape(1, n)
        null = Matrix.zeros(0,0).reshape(r.shape[0],0)
        r = Relation(r, null)
        self = Relation(self.left, self.right)
        op = self*r
        return op.left
    __call__ = get_left

    def get_right(self, l): # __rmul__ ?
        assert isinstance(l, Matrix)
        if len(l.shape)==1:
            n = len(l)
            l = l.reshape(1, n)
        null = Matrix.zeros(0,0).reshape(l.shape[0],0)
        l = Relation(null, l)
        self = Relation(self.left, self.right)
        op = l*self
        return op.right

    def get_matrix(self):
        right = self.right
        m,n = right.shape
        rows = []
        for i in range(n):
            v = [0]*n
            v[i] = 1
            v = Matrix(v).reshape(1,n)
            u = self.get_left(v)
            assert len(u)==1, "not invertible ...?"
            u = u.reshape(n)
            rows.append(u)
        U = Matrix(rows)
        return U

    @classmethod
    def identity(cls, n, p=2):
        I = Matrix.identity(n, p)
        return cls(I, I)

    @classmethod
    def black(cls, tgt, src, p=2):
        return Relation([[1]*tgt], [[1]*src], p)

    @classmethod
    def white(cls, tgt, src, p=2):
        n = tgt + src
        a = numpy.zeros((n-1, n), dtype=int)
        for i in range(n-1):
            a[i, i] = 1
            a[i, i+1] = -1
        A = Matrix(a, p)
        return Relation(A[:, :tgt], A[:, tgt:], p)

    @classmethod
    def promote(cls, item):
        if isinstance(item, cls):
            return item
        return cls(item)

    @classmethod
    def subspace(cls, M):
        left = M
        right = M[:, M.shape[1]:]
        return cls(left, right)

    def __str__(self):
        left, right = self._left, self._right
        w = left.shape[1]
        v = right.shape[1]
        smap = SMap()
        row = 0
        for row in range(left.shape[0]):
            smap[row,1] = "["
            smap[row,w+v+3] = "],"
            smap[row,w+2] = "|"
        smap[0,2] = str(left)
        smap[0,w+3] = str(right)
        smap[0,0] = "["
        smap[row,w+v+4] = "]"
        return str(smap)
        #return "%s %s"%(smap, self.shape)

    def __eq__(self, other):
        if other is None:
            return False
        assert isinstance(other, self.__class__)
        #assert self.tgt == other.tgt # too strict
        #assert self.src == other.src # too strict
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
        #assert isinstance(rhs, lhs.__class__)
        rhs = lhs.promote(rhs)
        #print()
        #print(lhs, "*")
        #print(rhs)
        assert lhs.src == rhs.tgt, (lhs.src, rhs.tgt)
        l, r = pullback(lhs.right.t, rhs.left.t)
        left = lhs.left.t * l
        right = rhs.right.t * r
        return lhs.__class__(left.t, right.t)

    def __rmul__(self, other):
        other = self.__class__(other)
        return other.__mul__(self)

    def __matmul__(lhs, rhs):
        left = lhs.left.direct_sum(rhs.left)
        right = lhs.right.direct_sum(rhs.right)
        return lhs.__class__(left, right)

    def __lshift__(lhs, rhs):
        left = lhs.left << rhs.left
        right = lhs.right << rhs.right
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
        assert Relation(fg) == Relation(g)*Relation(f) # _contravariant...
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

    assert b_ == Relation.black(1, 0)
    assert bb_ == Relation.black(2, 0)
    assert bb_b == Relation.black(2, 1)
    assert _b == Relation.black(0, 1)
    assert _bb == Relation.black(0, 2)
    assert b_bb == Relation.black(1, 2)

    w_ = Relation([[0]], zeros(1,0))
    ww_ = Relation([[1,1]], zeros(1,0))
    ww_w = Relation([[1,1],[0,1]], [[0],[1]])

    _w = w_.op
    _ww = ww_.op
    w_ww = ww_w.op

    assert w_ == Relation.white(1, 0)
    assert ww_ == Relation.white(2, 0)
    assert ww_w == Relation.white(2, 1)
    assert _w == Relation.white(0, 1)
    assert _ww == Relation.white(0, 2)
    assert w_ww == Relation.white(1, 2)

    assert _w*b_bb == _w@_w
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

    #print(b_ * _w)

    assert Relation.white(1,1) == I
    assert Relation.black(1,1) == I

    rel = (w_ @ I) * (I @ _b)

    rel = rel * Relation(Matrix.identity(2), zeros(2,0))
    #print(rel)
    #print( (w_@w_) * (_b@_b) )

    #print(w_ww * bb_b)
    #print(ww_w * b_bb)
    lhs = ww_w * w_ww

    A = Matrix.parse("""
    11..
    .11.
    ..11
    """)
    rhs = Relation(A[:,:2], A[:,2:])
    assert lhs == rhs
    assert lhs == Relation.white(2,2)

    lhs = (I @ w_ww) * (bb_b @ I)
    rhs = (b_bb @ I) * (I @ ww_w)
    assert lhs == rhs

    black = Relation.black
    white = Relation.white
    #print(white(3,0) @ black(0,1))
    #print(black(1,0))

    cup = white(2,0)
    assert cup == black(2,0)

    #print(lhs)
    #print( (lhs @ I) * (I @ cup) )
    #print( (I @ lhs) * (cup@I) )

    assert cup @ b_ == Relation([[1,1,0],[0,0,1]], zeros(2,0))

    op = (I @ w_ww @ I @ I) * (I @ I @ swap @ I) * (black(3,0) @ black(2,0))
    assert op == Relation([[1,1,1,0],[0,1,0,1]], zeros(2,0))


def test_tutte():
    black = Relation.black
    white = Relation.white

    I = Relation([[1]], [[1]])
    b_ = Relation([[1]], zeros(1,0))
    bb_ = Relation([[1,1]], zeros(1,0))
    bb_b = Relation([[1,1]], [[1]])
    _b = b_.op
    _bb = bb_.op
    b_bb = bb_b.op
    assert b_bb == black(1, 2)

    w_ = Relation([[0]], zeros(1,0))
    ww_ = Relation([[1,1]], zeros(1,0))
    ww_w = Relation([[1,1],[0,1]], [[0],[1]])
    assert ww_w == white(2, 1)

    _w = w_.op
    _ww = ww_.op
    w_ww = ww_w.op

    #print(w_, w_.shape)
    #ww_ = w_@w_
    #print(ww_, ww_.shape)

    M = Matrix.zeros((0,1))
    assert (M.is_loop(0))
    assert not (M.is_isthmus(0))

    M = Matrix([[1,0],[1,0]])
    assert M.is_isthmus(0)
    assert not M.is_loop(0)
    assert M.is_loop(1)
    assert not M.is_isthmus(1)

    M = Matrix([[1,0,1],[0,1,1]])
    assert M.delete(0) == Matrix([[0,1],[1,1]])
    assert M.contract(0) == Matrix([[1,1]])
    assert Matrix([[1,1]]).contract(0) == Matrix.zeros((0,1))

    assert bb_ == ww_
    M = (bb_b @ ww_w) * bb_

    I = black(1,1)

    op = (I @ I @ white(1,2) @ I) * (black(3,0) @ black(2,0))
    assert M==op

    return

    #print(M)
    #print(M.get_tutte())

    lhs = bb_b * w_ww
    rhs = ww_w * b_bb
    assert (lhs.op == rhs)

    def delete(M, i):
        M = Relation.promote(M)
        m, n = M.tgt, M.src
        assert 0<=i<n
        ops = [I]*n
        ops[i] = w_
        op = reduce(matmul, ops)
        return M*op

    m, n = 3, 4
    for trial in range(50):

        j = randint(0, n-1)
        M = Matrix.rand(m, n)
        Mj = M.delete(j)

        R = Relation(M.t)
        #print("R =")
        #print(R, R.shape)
        #print("j =", j)
        Rj = Relation(Mj.t)
        #print("Rj =")
        #print(Rj, Rj.shape)

        Sj = delete(R, j)
        #print("Sj =")
        #print(Sj, Sj.shape)
        #print( Rj==Sj )
        assert( Rj==Sj )
        #print()

    def contract(M, i):
        #print("contract(%d)"%i)
        #print(M, M.shape)
        #M = Relation.promote(M)
        m, n = M.shape
        right = Matrix.zeros((m,0))
        idxs = list(range(n))
        idxs.remove(i)
        idxs += [i]
        M1 = M[:, idxs]
        left = M1[:, :-1]
        right = M1[:, -1:]
        R = Relation(left, right)
        #print("R =")
        #print(R)
        R = R*w_
        #print("Rb_ =")
        #print(R)
        return R.left

    def rel_contract(R, i):
        lhs = [I for i in range(R.tgt)]
        lhs[i] = Relation.white(0, 1)
        lhs = reduce(matmul, lhs)
        R1 = lhs * R
        return R1

    def rel_delete(R, i):
        lhs = [I for i in range(R.tgt)]
        lhs[i] = Relation.black(0, 1)
        lhs = reduce(matmul, lhs)
        R1 = lhs * R
        return R1

    m, n = 3, 5
    for trial in range(5):
        j = randint(0, n-1)
        M = Matrix.rand(m, n)
        Mj = M.delete(j)

        R = Relation.subspace(M)
        Rj = Relation.subspace(Mj)
        Sj = rel_delete(R, j)
        assert Sj == Rj

    m, n = 3, 5
    for trial in range(5):
        j = randint(0, n-1)
        M = Matrix.rand(m, n)
        Mj = M.contract(j)
        Sj = contract(M, j)
        equiv = Mj.t.solve(Sj.t) is not None
        assert equiv
        R = Relation.subspace(M)
        Rj = Relation.subspace(Mj)
        Sj = rel_contract(R, j)
        assert Sj.left == Rj.left
        assert Sj == Rj

    m, n = 3, 5
    for trial in range(5):
        j = randint(0, n-1)
        M = Matrix.rand(m, n)
        R = Relation.subspace(M)
        R = R @ Relation.black(1,0)
        #print(R)
        #print()

    null = Relation(Matrix.zeros((0,0)))
    for l in [Relation.black(0,1), Relation.white(0,1)]:
      for r in [Relation.black(1,0), Relation.white(1,0)]:
        lr = (l*r).nf
        assert lr==null
    
    for l in [Relation.black(2,2), Relation.white(2,2)]:
        print(l, l.shape)
        print(l.nf, l.shape)

    



def test_pascal():
    # q-deformed pascal triangle
    p = 2

    rows = []

    rel = Relation.identity(0, p)
    row = [{rel}]
    rows.append(row)

    for n in range(1, 5):

        assert len(rows) == n

        row = []
        rel = Relation(zeros(0,n-1), zeros(0,1), p)
        assert rel.src == 1
        row.append({rel})
        #print(n, "choose", 0, "=", 1)
        for m in range(1, n):

            cell = set()
            for left in rows[n-1][m-1]:
                A = left.A
                A = A.direct_sum([[1]])
                rel = Relation(A[:,:n-1], A[:,n-1:], p)
                assert rel.src == 1
                cell.add(rel)

            for right in rows[n-1][m]:
                A = right.A
                assert A.shape[1] == n-1, (n, A.shape)
                for bits in numpy.ndindex((p,)*A.shape[0]):
                    v = numpy.array(bits)
                    v.shape = (A.shape[0],1)
                    B = numpy.concatenate((A, v), axis=1)
                    assert B.shape[1] == n
                    rel = Relation(B[:,:n-1], B[:,n-1:], p)
                    assert rel.src == 1
                    #print(rel)
                    assert rel not in cell
                    cell.add(rel)

            #print(n, "choose", m)
            row.append(cell)

        A = Matrix.identity(n)
        rel = Relation(A[:,:n-1], A[:,n-1:], p)
        assert rel.src == 1
        row.append({rel})
        #print(n, "choose", n, "=", 1)

        assert len(row) == n+1
        rows.append(row)

    for row in rows:
        print([len(cell) for cell in row])

    return
    for row in rows:
        for cell in row:
            for rel in cell:
                print(rel)


    
def test():
    test_linear()
    test_pascal()


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

    print("\nfinished in %.3f seconds.\n"%(time() - start_time))



