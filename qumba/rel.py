#!/usr/bin/env python

"""
Lagrangian relations 
also known as
Pauli flows

https://arxiv.org/abs/2105.06244v2

"""

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul, lshift
from random import choice, shuffle

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


class Relation(object):
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
        if left.shape[1] or right.shape[1]: # ?!?!
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
        #assert len(AB) == rank
        self.shape = (rank, self.tgt, self.src)
        self.p = p

    @property
    def nf(self): # normal form
        return self.__class__(self.left, self.right)

#    def get_left(self, r): # __mul__ ?
#        r = Matrix.promote(r)
#        if len(r.shape)==1:
#            n = len(r)
#            r = r.reshape(1, n)
#        left, right = self.left, self.right
#        assert r.shape[1] == right.shape[1]
#        A = right.intersect(r)
#        f = right.t.solve(A.t).t
#        l = f*left
#        return l
#
#    def get_right(self, l): # __rmul__ ?
#        if len(l.shape)==1:
#            n = len(l)
#            l = l.reshape(1, n)
#        left, right = self.left, self.right
#        assert l.shape[1] == left.shape[1]
#        #print("get_right", self.shape, l.shape)
#        A = left.intersect(l)
#        #print("\t", A.shape)
#        f = left.t.solve(A.t).t
#        l = f*right
#        return l


    def find_left(self, r): # __mul__ ?
        if len(r.shape)==1:
            n = len(r)
            r = r.reshape(1, n)
        null = Matrix.zeros(0,0).reshape(r.shape[0],0)
        r = Relation(r, null)
        self = Relation(self.left, self.right)
        op = self*r
        return op.left
    __call__ = find_left

    def find_right(self, l): # __mul__ ?
        if len(l.shape)==1:
            n = len(l)
            l = l.reshape(1, n)
        null = Matrix.zeros(0,0).reshape(l.shape[0],0)
        l = Relation(null, l)
        self = Relation(self.left, self.right)
        op = l*self
        return op.right

    # XXX XXX BROKEN
    def XXX_find_left(self, r): # __mul__ ?
        if len(r.shape)==1:
            n = len(r)
            r = r.reshape(1, n)
        left, right = self.left, self.right
        assert r.shape[1] == right.shape[1], ( r.shape[1] , right.shape[1] )
        I = Matrix.identity(left.shape[1])
        rows = []
        for row0 in r:
            for row1 in I: # XXX could vectorize this
                row = row1.concatenate(row0)
                rows.append(row)
        rows = Matrix(rows)
        print("find_left")
        print(strop(rows))
        print("intersect")
        print(strop(self.A))
        A = self.A.intersect(rows)
        #print("intersect:")
        #print(A)
        l = A[:, :left.shape[1]]
        #print(l)
        return l
    #__call__ = find_left

    def XXX_find_right(self, l): # __rmul__ ?
        if len(l.shape)==1:
            n = len(l)
            l = l.reshape(1, n)
        left, right = self.left, self.right
        assert l.shape[1] == left.shape[1]
        I = Matrix.identity(right.shape[1])
        rows = []
        for row0 in l:
            for row1 in I: # XXX could vectorize this
                row = row0.concatenate(row1)
                rows.append(row)
        rows = Matrix(rows)
        #print(rows)
        A = self.A.intersect(rows)
        #print("intersect:")
        #print(A)
        r = A[:, left.shape[1]:]
        #print(r)
        return r

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
        #assert isinstance(rhs, lhs.__class__)
        rhs = lhs.promote(rhs)
        assert lhs.src == rhs.tgt
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


class Lagrangian(Relation):
    @classmethod
    def all_rels(cls, tgt, src):
        n = tgt+src
        for code in construct.all_codes(n, 0, 0):
            H = code.H
            l, r = H[:, :2*tgt], H[:, 2*tgt:]
            rel = Lagrangian(l, r)
            assert rel.is_lagrangian()
            yield rel

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
        left, right = self._left, self._right
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
        swap = Lagrangian(l.concatenate(r, axis=1).transpose())
        return swap

    def to_qcode(self):
        H = self.A
        return QCode(H)

    def get_op(self):
        code = QCode(self.A)
        A = code.T
        k = self.tgt
        return Lagrangian(A[:, :k], A[:, k:])

    def is_diag(self):
        A = self.A
        m, n = A.shape
        if A.sum() != m:
            return False
#        for i in range(m):
#          for j in range(n):
#            i
        return True


zeros = lambda a,b : Matrix.zeros((a,b))

if 1:
    # make some globals

    I = Lagrangian.identity(2)
    h = Lagrangian([[0,1],[1,0]])
    b_ = Lagrangian([[0,1]], zeros(1,0))
    b1 = Lagrangian([[1,0],[1,1]])
    w1 = h*b1*h
    b1_ = b1*b_
    w1_ = h*b1_
    w_ = w1 * w1_
    _b1 = b1_.op
    _b = b_.op
    _w1 = w1_.op
    _w = w_.op

    swap = Lagrangian.get_swap()
    
    bb_b = Lagrangian([
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

    cnot = (I @ b_bb) * (ww_w @ I) 
    assert cnot * cnot.op == I@I # invertible
    assert cnot == (w_ww@I)*(I@bb_b)


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


    


def test_symplectic():
    one = Lagrangian(zeros(0,0), zeros(0,0))
    #print(one)
    #print(one*one)
    assert one*one == one

    I = Lagrangian.identity(2)
    h = Lagrangian([[0,1],[1,0]])

    #print("h:")
    #print(h)
    hh = h*h

    assert h != I
    assert I==I
    assert I*I==I
    assert I*h==h*I==h
    assert h*h == I

    # black unit
    b_ = Lagrangian([[0,1]], zeros(1,0))

    # phase=1
    b1 = Lagrangian([[1,0],[1,1]])
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

    gen = [b1, w1, w_*_w, b_*_b]
    names = mulclose_names(gen, "b w W B".split())
    assert len(names) == 15 # number of 2-qubit stabilizer states

    r = w1.get_op()

    code = w1.to_qcode()
    print(code.longstr())
    T = code.T
    assert r == Lagrangian(T[:, :2], T[:, 2:])
    assert r in names
    #print(names[r])
    assert r == b_*_w

    dode = QCode(T)
    print()
    print(dode.longstr())

    return

    print("w1 =")
    print(w1)
    print("r =")
    print(r)
    q = r.get_op()
    print("q =")
    print(q)
    assert r.get_op() == w1

    M = list(names.keys())
    for r in M:
        q = r.get_op()
        assert q != r, r
        print(r == q.get_op())

    return

    # ---------- 2 qubits ----------------

    swap = Lagrangian.get_swap()
    
    assert swap != I@I
    assert swap*swap == I@I
    for a in [I,w1,b1,h]:
      for b in [I,w1,b1,h]:
        assert swap*(a@b) == (b@a)*swap

    #for v_ in all_subspaces(2, 1): # not Lagrangian ...
    #    assert isinstance(v_@I, Lagrangian)
    #    assert swap * (v_@I) == I@v_

    # copy
    bb_b = Lagrangian([
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

    return

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


def test_code():

    # can we measure out the qubits of a CSS / non-CSS codes??

    I = Lagrangian.identity(2)
    h = Lagrangian([[0,1],[1,0]])
    b_ = Lagrangian([[0,1]], zeros(1,0))
    b1 = Lagrangian([[1,0],[1,1]])
    w1 = h*b1*h
    b1_ = b1*b_
    w1_ = h*b1_
    w_ = w1 * w1_
    _b1 = b1_.op
    _b = b_.op
    _w1 = w1_.op
    _w = w_.op

#    code = construct.get_513()
#    E = code.get_encoder()
#    print(E)
#    E2 = E << E
#    print(E2)
#    print(Lagrangian(E.t))
#    return

    dup = 2

    for code in [
        #construct.get_412(),
        construct.get_422(),
        construct.get_513(),
        #construct.get_713(),
        #construct.get_10_2_3(),
        #construct.get_622(),
        #construct.get_14_3_3(),
    ]:
        n = code.n
        space = code.space
        #for i in range(n):
        #    code = space.S(i) * code
        #code = space.S(0) * code
        #code = space.H(1) * code
        #print(code)
        #print(code.longstr())
        #print()

        E = code.get_encoder()
        #print(strop(E.t))
        #print()
        encode = Lagrangian(E.t)
    
        print("encode:")
        print(encode)
        print()
    
        prep = reduce(matmul, [w_]*n)
        prep = encode*prep
        #print("prep:")
        #print(prep)
        #print()
    
        #print("white measure:")
        #wgap = reduce(matmul, [w_ * _w]*n)
        #bgap = reduce(matmul, [b_ * _b]*n)
        wm = reduce(matmul, [_w]*n)
        bm = reduce(matmul, [_b]*n)
        #print(wm*encode)
        #print()

        op = [I]*n
        #op[1] = h
        op = reduce(matmul, op)
    
        u = wm*op*encode
        print(u.nf, u.shape)
        print()
    
        v = bm*op*encode
        print(v.nf, v.shape)
        print()
    
        f = list(range(n))
        f = space.SWAP(0,1)
        vf = v*f
        fprep = f*prep

        e2 = reduce(lshift, [encode]*dup)

        s2 = reduce(lshift, [space]*dup)
        f = []
        for i in range(n):
            for j in range(dup):
                f.append(i+j*n)
        p = s2.get_perm(f)
        e2 = p.t*e2*p
        print(e2.op.nf.op)

        bell = Matrix([[1,0,1,0],[0,1,0,1]])
        bell = Lagrangian(bell[:,:0], bell)

        s1 = SymplecticSpace(2)
        g = s1.S(0)*s1.H(0)
        bell = bell*g

        for r0 in Lagrangian.all_rels(0, dup):
        #for r0 in [_w@_w]:
        #for r0 in [bell]:
            measure = reduce(matmul, [r0]*n)
            me2 = measure * e2
            me2 = me2.nf
            if me2.is_diag():
                print(r0)
                print(me2)
                print()


class Module:
    def __init__(self, n):
        self.space = SymplecticSpace(n)
        self.n = n

    def get_identity(self):
        I = self.space.get_identity()
        return Lagrangian(I)

    # Argggh, why is this transposed
    def CX(self, i, j):
        CX = self.space.CX(i, j)
        CX = Lagrangian(CX.t)
        return CX

    def CZ(self, i, j):
        CZ = self.space.CZ(i, j)
        CZ = Lagrangian(CZ.t)
        return CZ

    def S(self, i):
        S = self.space.S(i)
        S = Lagrangian(S.t)
        return S

    def H(self, i):
        H = self.space.H(i)
        H = Lagrangian(H.t)
        return H

    # -----------------------------------------
    # note: Lagrangian relations act on Pauli's
    # via get_left, get_right

    def X(self, i):
        nn = 2*self.n
        x = [0]*nn
        x[2*i] = 1
        x = Matrix(x).reshape(1, nn)
        return x

    def Z(self, i):
        nn = 2*self.n
        z = [0]*nn
        z[2*i+1] = 1
        z = Matrix(z).reshape(1, nn)
        return z

    def Y(self, i):
        nn = 2*self.n
        y = [0]*nn
        y[2*i+1] = 1
        y = Matrix(y).reshape(1, nn)
        return y

    def get_pauli(self, desc):
        assert len(desc) == self.n
        nn = 2*self.n
        op = Matrix([0]*nn).reshape(1, nn)
        for i,c in enumerate(desc):
            if c in ".I":
                continue
            method = getattr(self, c)
            pauli = method(i)
            #print(pauli)
            op = pauli + op
        return op

    def MX(self, *idxs):
        n = self.n
        ops = [I]*n
        for i in idxs:
            ops[i] = _b
        #print(ops)
        op = reduce(matmul, ops)
        return op

    def MZ(self, *idxs):
        n = self.n
        ops = [I]*n
        for i in idxs:
            ops[i] = _w
        #print(ops)
        op = reduce(matmul, ops)
        return op


            
def css_prep(code):
    css = code.to_css()
    #print(css.longstr())
    n = code.n

    b_ancilla = reduce(matmul, [I]*n+[b_])
    w_ancilla = reduce(matmul, [I]*n+[w_])

    prep = Module(n).get_identity()

    ancilla = Module(n + 1)
    for i in range(css.mx):
        measure = ancilla.get_identity()
        for j in range(n):
            if css.Hx[i,j]:
                measure = measure * ancilla.CX(n, j)
        measure = w_ancilla.op * measure * w_ancilla
        prep = measure*prep
    for i in range(css.mz):
        measure = ancilla.get_identity()
        for j in range(n):
            if css.Hz[i,j]:
                measure = measure * ancilla.CX(j, n)
        measure = b_ancilla.op * measure * b_ancilla
        prep = measure*prep

    return prep


def test_prep():

    module = Module(2)
    assert cnot == module.CX(0,1)
    assert w1@I == module.S(0)
    assert I@h == module.H(1)

    # ------------------------------------------------------------------------

    code = QCode.fromstr("ZZZ")
    prep = css_prep(code)

    #print("prep =")
    #print(prep.nf)

    assert( (prep*prep.op) != Module(3).get_identity() )

    n = code.n
    module = Module(n)
    for i in range(n):
        x = module.X(i)
        l = prep(x)
        assert len(l) == 1, len(l)

        x = Lagrangian(x, zeros(1,0))
        px = prep*x # ??

        z = module.Z(i)
        l = prep(z)
        assert len(l) == 2, len(l)

        z = Lagrangian(z, zeros(1,0))
        pz = prep*z # ??

    # ------------------------------------------------------------------------

    code = QCode.fromstr("ZZZ XXI IXX")
    prep = css_prep(code)

    code = construct.get_713()
    prep = css_prep(code)

    print("prep:")
    print(prep)

    n = code.n
    module = Module(n)
    for i in range(n):
        x = module.X(i)
        l = prep(x)
        assert len(l) == 6, len(l)

        z = module.Z(i)
        l = prep(z)
        assert len(l) == 6

        for j in range(i+1, n):
            x = module.X(i) + module.X(j)
            l = prep(x)
            assert len(l) == 6


def test_goto():
    # See: https://www.nature.com/articles/srep19578

    n = 8
    syntax = Syntax()
    CX, H = syntax.CX, syntax.H

    prog = (CX(6,7)*CX(5,7)*CX(0,7)
        *CX(6,4)*CX(1,5)*CX(3,6)*CX(2,0)
        *CX(1,4)*CX(2,6)*CX(3,5)*CX(1,0)
        *H(1)*H(2)*H(3))

    print(prog)

    space = SymplecticSpace(n)
    mod = Module(n)
    rel = prog*mod

    print(rel)

    "IIIZZZZ ZIIIIZZ IZIIZIZ IIZIZZI"


def get_code(U, k=0):
    #print("get_code")
    assert isinstance(U, Lagrangian)
    rank, tgt, src = U.shape
    #print(rank, tgt, src)
    module = Module(src//2)
    H = Matrix([]).reshape(0, tgt)
    for i in range(src//2-k):
        r = module.X(i)
        #print("find_left:")
        #print(strop(r))
        op = U.find_left(r)
        #print(op, op.shape)
        if len(op):
            #print(op.shape)
            #print(strop(op))
            H = H.concatenate(op)
    #print(H)
    #H = Matrix(rows)
    return QCode(H)


def test_double():
    from qumba.triorthogonal import get_double

#    code = construct.get_422()
    code = construct.get_713()
    print(code.longstr())

    #dode = get_double([code])
    #print(dode)

    E = code.get_encoder()
    
    encode = Lagrangian(E.t)

    print("encode:")
    print(encode)
    print()
    
#    #print(code.H)
#    prep = css_prep(code) # what is this ??? XXX
#
#    print("prep:")
#    print(prep)
#    print()
#
#    print(prep == prep.op)
#
#    return


    n = code.n
    module = Module(n)
    
    H = strop(code.H)
    #H = strop(code.L)
    for h in H.split():
        print()
        print(h)
        op = module.get_pauli(h)
        #print(op)
        #lop = (encode.find_left(op))
        #print("op * h")
        #print(strop(lop), lop.shape)
        rop = (encode.find_right(op))
        print("h * op")
        print(strop(rop), rop.shape)

        lop = encode.find_left(rop)
        print(strop(lop), lop.shape)

        #rev = encode.op
        #rop = rev.find_left(op)
        #print(strop(rop), rop.shape)

    print(encode)
    print()

    m = module.MX(0, 1)
    print(m, m.shape)

    e = m*encode
    print()
    print(e)

    dode = get_code(e)
    print()
    print(dode)
    print(dode.longstr())

    #dode = dode.to_qcode()
    #print(dode)
    #print(dode.longstr())





def test():
    test_linear()
    test_symplectic()


def main():

    n = argv.get("n", 4)
    m = argv.get("m", n//2)

    found = set()
    for rel in Lagrangian.all_rels(n-m, m):
        #print(rel)
        #print()
        assert rel not in found
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

    print("\nfinished in %.3f seconds.\n"%(time() - start_time))



