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
        self.shape = (rank, self.tgt, self.src)
        self.p = p

    @property
    def nf(self): # normal form
        return self.__class__(self.left, self.right)

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

    print("finished in %.3f seconds.\n"%(time() - start_time))



