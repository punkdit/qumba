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
from random import choice, shuffle, randint

import numpy

from qumba.matrix import Matrix, pullback
from qumba.symplectic import symplectic_form
from qumba.qcode import fromstr, strop, QCode, SymplecticSpace
from qumba.smap import SMap
from qumba.action import mulclose, mulclose_names
from qumba.syntax import Syntax
from qumba.argv import argv
from qumba import construct
from qumba.rel import Relation, zeros


class Lagrangian(Relation):

    @property
    def source(self):
        return Module(self.src//2)

    @property
    def target(self):
        return Module(self.tgt//2)

    @classmethod
    def all_rels(cls, tgt, src):
        n = tgt+src
        for code in construct.all_codes(n, 0, 0):
            H = code.H
            l, r = H[:, :2*tgt], H[:, 2*tgt:]
            rel = Lagrangian(l, r)
            assert rel.is_lagrangian()
            yield rel

    @classmethod
    def fromstr(cls, s):
        assert "|" not in s, "not implemented"
        H = fromstr(s)
        return Lagrangian(H)

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
    def get_identity(cls, n, p=2):
        I = Matrix.identity(2*n, p)
        return cls(I, I)

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
        return True

    def get_code(self, k=0):
        rank, tgt, src = self.shape
        module = Module(src//2)
        H = Matrix([]).reshape(0, tgt)
        for i in range(src//2-k):
            r = module.X(i)
            op = self.get_left(r)
            if len(op):
                H = H.concatenate(op)
        H = H.linear_independent()
        return QCode(H)

#    def restrict(self, i):
#        A = self.A
#        r, tgt, src = self.shape
#        m, n = A.shape
#        assert tgt + src == n
#        assert 0<=2*i<n
#        idxs = list(n)
#        idxs.remove(i)
#        A = A[:, idxs]
#        if i < tgt:
#            tgt -= 1
#        else:
#            src -= 1
#        L = Lagrangian(A[:, :tgt], A[:, src:])
#        return L

    def _hitme(self, i, l, r):
        A = self.A
        m, nn = A.shape
        r, tgt, src = self.shape
        assert tgt + src == nn
        assert 0<=2*i<nn
        assert nn%2 == 0
        n = nn//2
        op = [I]*n
        if i < tgt:
            # measure i
            op[i] = l
            op = reduce(matmul, op)
            L = op * self
        else:
            # prepare i
            op[i] = r
            op = reduce(matmul, op)
            L = self * op
        return L

    def x_restrict(self, i):
        return self._hitme(i, _b, b_)

    def y_restrict(self, i):
        return self._hitme(i, _w1, w1_) # == b1_,_b1

    def z_restrict(self, i):
        return self._hitme(i, _w, w_)

    def _tutte(self, x, y, z, depth=0):
        if self.rank == 0:
            return 1
        i = 0
        xl = self.x_restrict(i)
        yl = self.y_restrict(i)
        zl = self.z_restrict(i)
        px = py = pz = None
        if xl.rank:
            px = xl._tutte(x, y, z, depth+1)
        if yl.rank:
            py = yl._tutte(x, y, z, depth+1)
        if zl.rank:
            pz = zl._tutte(x, y, z, depth+1)
        if xl.rank == 0 and yl.rank == 0 and zl.rank == 0:
            #assert 0
            p = 1
        elif xl.rank == 0 and yl.rank == 0:
            p = x*y*pz
        elif xl.rank == 0 and zl.rank == 0:
            p = x*py*z
        elif yl.rank == 0 and zl.rank == 0:
            p = px*y*z
        elif xl.rank == 0:
            p = x*(py + pz)
        elif yl.rank == 0:
            p = y*(px + pz)
        elif zl.rank == 0:
            p = z*(px + py)
        else:
            p = px + py + pz
        print(" "*depth, p, xl.rank, yl.rank, zl.rank)
        return p

    def get_tutte(self):
        from sage import all_cmdline as sage
        R = sage.PolynomialRing(sage.ZZ, list("xyz"))
        x, y, z = R.gens()
        p = self._tutte(x, y, z)
        return p


if 1:
    # make some globals

    I = Lagrangian.identity(2)
    h = Lagrangian([[0,1],[1,0]])
    b_ = Lagrangian([[0,1]], zeros(1,0))
    b1 = Lagrangian([[1,0],[1,1]]) # S gate
    w1 = h*b1*h
    b1_ = b1*b_
    w1_ = h*b1_
    w_ = w1 * w1_
    _b1 = b1_.op
    _b = b_.op
    _w1 = w1_.op
    _w = w_.op

    assert _b1 == _w1

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

    assert str(b_).strip() == "Z|" # the Z-state aka |0>
    assert str(w_).strip() == "X|" # the X-state aka |+>
    assert str(_b).strip() == "|Z" # the Z-measure aka <0|
    assert str(_w).strip() == "|X" # the X-measure aka <+|

    assert swap*bb_b == bb_b

    b_bb = bb_b.op
    ww_w = (h@h) * bb_b * h
    w_ww = ww_w.op

    cnot = (I @ b_bb) * (ww_w @ I) 
    assert cnot * cnot.op == I@I # invertible
    assert cnot == (w_ww@I)*(I@bb_b)


def is_xop(op, i):
    A = op.A
    m, nn = A.shape
    v = [0]*nn
    v[2*i] = 1
    v = Matrix([v])
    u = A.t.solve(v.t)
    return u is not None

def is_yop(op, i):
    A = op.A
    m, nn = A.shape
    v = [0]*nn
    v[2*i:2*i+2] = [1,1]
    v = Matrix([v])
    u = A.t.solve(v.t)
    return u is not None

def is_zop(op, i):
    A = op.A
    m, nn = A.shape
    v = [0]*nn
    v[2*i+1] = 1
    v = Matrix([v])
    u = A.t.solve(v.t)
    return u is not None


def delete(op, i):
    A = op.A
    A = A.puncture(2*i)
    A = A.puncture(2*i)
    A = A.linear_independent()
    op = Lagrangian.subspace(A)
    assert op.is_lagrangian()
    return op


def get_tutte(op, R=None, depth=0):
    assert 0, "FAIL"
    if R is None:
        from sage import all_cmdline as sage
        R = sage.PolynomialRing(sage.ZZ, list("xyz"))

    #print("get_tutte", depth)
    #print(op, op.shape)

    x, y, z = R.gens()
    n = op.shape[0]

    if n:
        i = randint(0, n-1)

    if n==0:
        p = 1

    elif is_xop(op, i):
        op = delete(op, i)
        p = x * get_tutte(op, R, depth+1)

    elif is_yop(op, i):
        op = delete(op, i)
        #p = y * get_tutte(op, R, depth+1) # ???
        p = (x*z)*get_tutte(op, R, depth+1) # ???

    elif is_zop(op, i):
        op = delete(op, i)
        p = z * get_tutte(op, R, depth+1)

    else:
        xop = [I] * n
        xop[i] = _b
        xop = reduce(matmul, xop)

        yop = [I] * n
        yop[i] = _b1
        yop = reduce(matmul, yop)

        zop = [I] * n
        zop[i] = _w
        zop = reduce(matmul, zop)

        xop = xop*op
        assert xop.shape[0] == n-1
        xp = get_tutte(xop, R, depth+1)

        yop = yop*op
        assert yop.shape[0] == n-1
        yp = get_tutte(yop, R, depth+1)

        zop = zop*op
        assert zop.shape[0] == n-1
        zp = get_tutte(zop, R, depth+1)

        p = xp+yp+zp # ???
        #p = xp+zp # ???


    return p


def test_tutte():

    rel = b_ @ w_
    assert not is_xop(rel, 0)
    assert is_xop(rel, 1)
    assert not is_yop(rel, 0)
    assert not is_yop(rel, 1)
    assert is_zop(rel, 0)
    assert not is_zop(rel, 1)

    rel = bb_b * b_
    for i in range(2):
        assert not is_xop(rel, i)
        assert not is_yop(rel, i)
        assert not is_zop(rel, i)

    for n in [1,2,3]:
        found = set()
        for op in Lagrangian.all_rels(n,0):
            #print(op)
            #print([is_xop(op, 0), is_xop(op, 1)])
    
            p = get_tutte(op)
            q = get_tutte(op)
            assert p==q, "%s != %s"%(p, q)
            if p in found:
                continue
            found.add(p)
            print(p)
    
        print(len(found))
        print()


def FAIL_generating_poly(A):
    m, nn = A.shape
    n = nn//2
    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, list("xy"))
    x, y = R.gens()
    p = 0
    for bits in numpy.ndindex((2,)*n):
        idxs = []
        for i,bit in enumerate(bits):
            if bit:
                idxs.append(2*i)
                idxs.append(2*i+1)
        B = A[:, idxs]
        B = B.row_reduce()
        left = len(A) - len(B)
        right = sum(bits) - len(B)
        if right< 0:
            print(A)
            print(bits)
            print(idxs)
            print(B)
        assert right>=0
        p += (x-1)**left * (y-1)**right
    return p


def generating_poly(A):
    # not great: pauli Y != Z == X
    # but it is monoidal
    m, nn = A.shape
    n = nn//2
    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, list("xy"))
    x, y = R.gens()
    p = 0
    for bits in numpy.ndindex((2,)*nn):
        for i in range(n):
            if bits[2*i] and bits[2*i+1]:
                break
        else:
            idxs = [i for i,bit in enumerate(bits) if bit]
            B = A[:, idxs]
            B = B.row_reduce()
            left = len(A) - len(B)
            right = sum(bits) - len(B)
            #p += (x-1)**left * (y-1)**right # has -ve coefficients
            p += x**left * y**right
    return p


def test_char():
    # maybe we want to use polymatroids:
    # https://arxiv.org/abs/2207.04421 etc.

    for n in [1,2,3,4]:
        found = set()
        for op in Lagrangian.all_rels(n, 0):
            A = op.A
            p = generating_poly(A)
            #print(A, p)
            if p not in found:
                print(op)
                print(p)
            found.add(p)
        print(len(found))



def detect(lhs, rhs):
    "return (detect'ors, lhs*rhs)"
    rhs = lhs.promote(rhs)
    assert lhs.src == rhs.tgt, (lhs.src, rhs.tgt)
    l, r = pullback(lhs.right.t, rhs.left.t)

    left = lhs.left.t * l
    right = rhs.right.t * r
    f = left.concatenate(right)
    k = f.kernel().t # find the detect'ing regions

    ll = lhs.right.t * l
    rr = rhs.left.t * r
    assert ll==rr
    d = (ll*k).t
    d = Lagrangian(d)
    op = Lagrangian(left.t, right.t)

    return d, op




class Module:

    # singleton
    @cache
    def __new__(cls, n):
        ob = object.__new__(cls)
        return ob

    def __init__(self, n):
        self.space = SymplecticSpace(n)
        self.n = n

    def __str__(self):
        return "Module(%d)"%(self.n,)
    __repr__ = __str__

    def get_identity(self):
        I = self.space.get_identity()
        return Lagrangian(I)

    # Argggh, SymplecticSpace gives transposed ops ...?!?
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

    def HSH(self, i):
        HSH = self.space.HSH(i)
        HSH = Lagrangian(HSH.t)
        return HSH

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

    def PX(self, *idxs): # or RX / reset ?
        "prepare X state: |+..+>"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        else:
            idxs = list(idxs)
        idxs.sort(reverse=True)
        ops = [I]*n
        for i in idxs:
            ops.insert(i, w_)
        op = reduce(matmul, ops)
        return op

    def PZ(self, *idxs): # or RZ / reset ?
        "prepare Z state: |0..0>"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        else:
            idxs = list(idxs)
        idxs.sort(reverse=True)
        ops = [I]*n
        for i in idxs:
            ops.insert(i, b_)
        op = reduce(matmul, ops)
        return op

    def PY(self, *idxs): # or RY / reset ?
        "prepare Y state"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        else:
            idxs = list(idxs)
        idxs.sort(reverse=True)
        ops = [I]*n
        for i in idxs:
            ops.insert(i, w1_)
        op = reduce(matmul, ops)
        return op

    def MX(self, *idxs):
        "measure X on idxs"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        ops = [I]*n
        for i in idxs:
            ops[i] = _w
        op = reduce(matmul, ops)
        return op

    def MZ(self, *idxs):
        "measure Z on idxs"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        ops = [I]*n
        for i in idxs:
            ops[i] = _b
        op = reduce(matmul, ops)
        return op

    def MY(self, *idxs):
        "measure Y on idxs"
        n = self.n
        if not idxs:
            idxs = list(range(n))
        ops = [I]*n
        for i in idxs:
            ops[i] = _w1 # ?!?! is this right ??
        op = reduce(matmul, ops)
        return op



def test_symplectic():

    assert Module(3) is Module(3)

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
    #print(code.longstr())
    T = code.T
    assert r == Lagrangian(T[:, :2], T[:, 2:])
    assert r in names
    #print(names[r])
    assert r == b_*_w

    #dode = QCode(T)
    #print()
    #print(dode.longstr())

    #print("w1 =")
    #print(w1)
    #print("r =")
    #print(r)
    #q = r.get_op()
    #print("q =")
    #print(q)
    #assert r.get_op() == w1

#    M = list(names.keys())
#    for r in M:
#        q = r.get_op()
#        assert q != r, r
#        #print(r == q.get_op())

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

    a = (_w*b_)
    b = (_b*b_)
    assert a==b
    assert _b*w_ == a

    I = Lagrangian.get_identity(1)

    op = (w_ww * ww_w)
    assert op == I

    assert w_ww.shape == (3,2,4) # rank, left, right

    d, ww = detect(w_ww, ww_w)
    assert ww == I
    assert d == Lagrangian.fromstr("ZZ")

    mzz = ww_w*w_ww

    d, op = detect(mzz, mzz)
    assert d == Lagrangian.fromstr("ZZ")


def other_test():

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


def test_module():

    module = Module(2)

    lhs = module.CX(0,1) * module.H(1) * module.PX(0,1)
    rhs = module.CX(1,0) * module.H(0) * module.PX(0,1)
    assert lhs == rhs

    #print(module.PZ(0,1))
    lhs = (module.MX(0,1) * module.PZ(0,1))
    rhs = (module.MZ(0,1) * module.PZ(0,1))
    #print(lhs.nf, lhs.shape)
    #print(rhs.nf, rhs.shape)

    def show(op):
        print(op, op.shape)

    # rank==0 means impossible 
    # rank>0 means possible 
    # yes ? ?????
    assert (_w * b_).rank == 0
    assert (_w * w_).rank == 1
    assert (_b * b_).rank == 1
    assert (_b * w_).rank == 0

    #print( (_b * w_) @ module.get_identity() )

    module = Module(4)
    op = module.MX(0,1,2,3) * module.PX(0,1,2,3)
    assert op.rank == 4


def test_syntax():

    s = Syntax()
    X, Z, Y = s.X, s.Z, s.Y
    S, H, CX, MX, PX, PZ = s.S, s.H, s.CX, s.MX, s.PX, s.PZ

    mod = Module(5)
    op = MX(0) * mod
    assert op.source == Module(5)
    assert op.target == Module(4)

    #print(mod.PX(3))
    op = PX(3) * mod
    #print(op)

    module = Module(0)
    lhs = PX(2)*PX(1)*PZ(0)*module
    rhs = module.PZ(0) @ module.PX(0) @ module.PX(0)
    assert lhs == rhs
    lhs = PZ(0)*PX(1)*PX(2)*module # tricky... hmmm
    assert lhs == rhs


    


def test_goto():
    # test fault tolerant circuits from Goto paper
    # See: https://www.nature.com/articles/srep19578

    n = 8
    syntax = Syntax()
    CX, H = syntax.CX, syntax.H
    PX, PZ = syntax.PX, syntax.PZ
    MX, MZ = syntax.MX, syntax.MZ

    prog = (CX(6,7)*CX(5,7)*CX(0,7)*CX(6,4)*CX(1,5)*CX(3,6)*CX(2,0)
        *CX(1,4)*CX(2,6)*CX(3,5)*CX(1,0)*H(1)*H(2)*H(3))

    prog = (CX(6,7)*CX(5,7)*CX(0,7)*CX(6,4)*CX(1,5)*CX(3,6)*CX(2,0)
        *CX(1,4)*CX(2,6)*CX(3,5)*CX(1,0))

    for i in reversed(range(n)):
        s = PX(i) if i in [1,2,3] else PZ(i)
        prog = prog*s

    print(prog)

    module = Module(0)
    state = prog*module

    assert state.source == module
    assert state.target == Module(8)

    module = state.target

    assert module.MZ(7) * state == module.MX(7) * state # weird but true..
    assert module.MZ(7) != module.MX(7)

    prep = module.MZ(7) * state
    print(prep, prep.shape)

    other = MZ(7) * state
    print(other, other.shape)
    assert prep == other

    return

    # um um, how does this work
    mod = Module(n)
    nn = prep.src
    for i in range(n):
        #v = [0]*nn
        #v[i] = 1
        #v = Matrix([v])
        v = mod.X(i)
        print(v, prep * v)
        v = mod.Z(i)
        print(v, prep * v)


def test_left_right():
#    code = construct.get_422()
    code = construct.get_713()
    print(code.longstr())

    E = code.get_encoder()
    encode = Lagrangian(E.t)

    print("encode:")
    print(encode)
    print()
    
    n = code.n
    module = Module(n)
    
    H = strop(code.H)
    #H = strop(code.L)
    for h in H.split():
        print()
        print(h)
        op = module.get_pauli(h)
        #print(op)
        #lop = (encode.get_left(op))
        #print("op * h")
        #print(strop(lop), lop.shape)
        rop = (encode.get_right(op))
        print("h * op")
        print(strop(rop), rop.shape)

        lop = encode.get_left(rop)
        print(strop(lop), lop.shape)

        #rev = encode.op
        #rop = rev.get_left(op)
        #print(strop(rop), rop.shape)

    print(encode)
    print()

    m = module.MX(0, 1)
    print(m, m.shape)

    e = m*encode
    print()
    print(e)

    dode = e.get_code()
    print()
    print(dode)
    print(dode.longstr())

    #dode = dode.to_qcode()
    #print(dode)
    #print(dode.longstr())


def get_subcode(code, dode):
    code = code.to_qcode()
    dode = dode.to_qcode()

    idxs = list(range(code.n, dode.n))
    
    E = dode.get_encoder()
    encode = Lagrangian(E.t)
    module = Module(dode.n)
    m = module.MZ(*idxs)

    e = m*encode
    eode = e.get_code(1)
    return eode


def test_double():
    from qumba.triorthogonal import get_double

    S0 = construct.get_713()

    T1 = get_double([S0])
    T1.bz_distance()
    assert T1.n == 15

    S1 = QCode.fromstr("""
    XIIIIIIIIXIIXIIXI
    IXIIIIIIIXIIXIIIX
    IIXIIIIIIIIXIXXII
    IIIXIIIIXXIIXIIII
    IIIIXIIIXIXIXXXXX
    IIIIIXIIIIXXIXIII
    IIIIIIXIXXXIIXXXX
    IIIIIIIXIIXXIIXII
    ZIIIIIIIIZIIZIIZI
    IZIIIIIIIZIIZIIIZ
    IIZIIIIIIIIZIZZII
    IIIZIIIIZZIIZIIII
    IIIIZIIIZIZIZZZZZ
    IIIIIZIIIIZZIZIII
    IIIIIIZIZZZIIZZZZ
    IIIIIIIZIIZZIIZII
    """)

    T2 = get_double([S0, S1])
    T2.bz_distance()
    print(T2)

    T2 = T2.to_qcode()

    code = get_subcode(S1, T2)
    print(S1, T2, code)
    print(code.is_equiv(S1))
    

    S2 = construct.get_golay(23)
    print(S2)
    T3 = get_double([S0, S1, S2])
    #T3.bz_distance()
    print(T3)

    T3 = T3.to_qcode()
    code = get_subcode(S2, T3)
    print(code)



class Gadget:
    def __init__(self, n):

        m = Module(2*n)

        # Prep,Control,Measure
        PX, CX, MX = m.PX, m.CX, m.MX
        PZ, CZ, MZ = m.PZ, m.CZ, m.MZ
        PY, MY = m.PY, m.MY

        idxs = list(range(n))

        measure = MZ(*idxs)
        cx = reduce(mul, [CX(i,i+n) for i in idxs])
        prep = PX(*idxs)
        fwd = [measure, cx, prep]

        measure = MX(*idxs)
        cx = reduce(mul, [CX(i+n,i) for i in idxs])
        prep = PZ(*idxs)
        rev = [measure, cx, prep]

        self.fwd = fwd
        self.rev = rev
        self.my = MY(*idxs)

    def teleport(self, g, rev=False):
        measure, cx, prep = [self.fwd, self.rev][int(rev)]
        return measure * cx * g * prep
        
    def y_teleport(self, g, b0, b1, rev=False):
        _, cx, prep = [self.fwd, self.rev][int(rev)]

        m = Module(2)

        MX, MZ, MY = m.MX, m.MZ, m.MY

        if rev:
            ops = [MX, MX]
        else:
            ops = [MZ, MZ]
        if b0:
            ops[0] = MY
        if b1:
            ops[1] = MY
        measure = ops[0](0) @ ops[1](1)

        return measure * cx * g * prep
        


def test_k2():

    # Note:
    # CSS state prep & measure in XYZ basis gives whole clifford group on k=2

    n = 2

    module = Module(n)
    
    gen = [module.CX(0,1), module.H(0), module.H(1), module.S(0), module.S(1)]
    G = mulclose(gen)
    assert len(G) == 720

    if argv.css:
        # here we restrict the 720 to only CSS state prep:
        fwd = []
        rev = []
        px = module.PX(0,1)
        pz = module.PZ(0,1)
        for g in G:
            op = g * px
            H = op.left
            c = QCode(H)
            if c.is_css():
                fwd.append(g)
    
            op = g * pz
            H = op.left
            c = QCode(H)
            if c.is_css():
                rev.append(g)
    else:
        fwd = rev = G

    print(len(fwd), len(rev))

    module = Module(2*n)
    II = Lagrangian.identity(2*n)

    #gen = [module.CX(0,1), module.H(0), module.H(1), module.S(0), module.S(1)]
    #G = mulclose(gen)
    #assert len(G) == 720

    found = set()

    gadget = Gadget(n)

    I = Lagrangian.identity(n)
    for g in fwd:
        g = g<<II
        op = gadget.teleport(g, False)
        found.add(op)
        if argv.css_y:
            op = gadget.y_teleport(g, 1, 0, False)
            found.add(op)
            op = gadget.y_teleport(g, 0, 1, False)
            found.add(op)
    #print(len(found))
    
    for g in rev:
        g = g<<II
        op = gadget.teleport(g, True)
        found.add(op)
        if argv.css_y:
            op = gadget.y_teleport(g, 1, 0, True)
            found.add(op)
            op = gadget.y_teleport(g, 0, 1, True)
            found.add(op)

    print(len(found))

    gen = []
    s2 = SymplecticSpace(2)
    for g in found:
        #print(g, "?")
        if g*g.op != II:
            continue
        #print()
        op = g.get_matrix()
        print(op)
        assert s2.is_symplectic(op)
        print(s2.get_name(op))
        print()

        gen.append(op)

    G = mulclose(gen)
    print(len(G))


def test_toric():

    code = construct.get_toric(1,3)
    n = code.n
    print(code)

    module = Module(2*n)
    II = Lagrangian.identity(2*n)

    k = SymplecticSpace(code.k)
    gen = [k.CX(0,1), k.H(0), k.H(1), k.S(0), k.S(1)]
    G = mulclose(gen)
    assert len(G) == 720

    I = Matrix.identity(2*(code.n - code.k))
    G = [I<<g for g in G]

    E = code.get_encoder()
    Ei = ~E

    G = [E * g * Ei for g in G]

    for g in G:
        dode = g*code
        assert dode.is_equiv(code)
    
    I = Matrix.identity(2*code.n)
    G = [g<<I for g in G]
    G = [Lagrangian(g) for g in G]


    found = set()

    gadget = Gadget(n)

    for g in G:
        op = gadget.teleport(g, False)
        found.add(op)
    
        op = gadget.teleport(g, True)
        found.add(op)

    print("#gadget:", len(found))


    gen = []
    space = SymplecticSpace(n)
    sk = SymplecticSpace(code.k)
    for g in found:
        #print(g)
        if g*g.op != II:
            continue
        #print()
        op = g.get_matrix()
        assert space.is_symplectic(op)

        dode = op*code
        assert dode.is_equiv(code)

        op = dode.get_logical(code)

        print(op)
        assert sk.is_symplectic(op)
        print(sk.get_name(op))
        print()

        gen.append(op)

    G = mulclose(gen, verbose=True)
    print(len(G))



def test_encode():

    code = construct.get_513()
    print(code.longstr())

    n = code.n
    space = code.space

    E = code.get_encoder()
    encode = Lagrangian(E.t)
    print("encode:")
    print(encode)
    print()
    

def measure_x(rel, idxs):
    nn = rel.src
    n = nn//2
    mod = Module(n+1)
    rel = rel @ Lagrangian.get_identity(1) # ancilla
    rel = rel * mod.PX(n)
    for i in idxs:
        rel = mod.CX(n, i)*rel
    rel = mod.MX(n) * rel
    return rel

def measure_z(rel, idxs):
    nn = rel.src
    n = nn//2
    mod = Module(n+1)
    rel = rel @ Lagrangian.get_identity(1) # ancilla
    rel = rel * mod.PZ(n)
    for i in idxs:
        rel = mod.CX(i, n)*rel
    rel = mod.MZ(n) * rel
    return rel


def test_422():

    code = construct.get_422()
    n = code.n
    nn = 2*n
    
    if 0:
        E = code.get_encoder()
        encode = Lagrangian(E.t)
        print("encode:")
        print(encode)
        print()


    rel = Module(4).get_identity()
    rel = measure_x(rel, [0,1,2,3])
    lhs = rel

    mod = Module(5)
    rel = mod.get_identity()
    a = 4
    for i in range(4):
        rel = mod.CX(a, i) * rel
    rel = mod.MX(a) * rel * mod.PX(a)
    assert rel == lhs

    rel = measure_z(rel, [0,1,2,3])
    rel = measure_x(rel, [0,1])
    rel = measure_x(rel, [0,2])

    print(rel)
    print()

    v = Matrix([[0]*nn])
    print(rel*v)
    



def test():
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



