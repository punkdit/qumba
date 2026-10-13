#!/usr/bin/env python

"""
Use linear relations for CSS _circuits (aka phase-free ZX-calculus).

See previous: css.py 

"""

import time
from functools import reduce, cache
from operator import matmul, lshift
from random import choice, randint, random

import numpy

from qumba.matrix import Matrix, pullback
from qumba.rel import Relation # linear relations
from qumba.csscode import CSSCode
from qumba import construct

from qumba.argv import argv
from qumba.util import choose, all_perms
from qumba.smap import SMap


class Circuit:
    def __init__(self, op, tgt, src):
        assert isinstance(op, Operator)
        self.op = op
        self.tgt = tgt
        self.src = src

#    @property
#    def tgt(self)

    def __mul__(self, other):
        return Composite(self, other)

    @classmethod
    def promote(cls, item):
        if isinstance(item, Circuit):
            op = item
        elif isinstance(item, (Matrix, Relation)):
            op = Operator(item)
        else:
            assert 0, "whats this: %s"%(item.__class__,)
        return op


class Operator(Circuit):
    "The atomic Operator's in a Composite Circuit"
    def __init__(self, Ax, Az=None, check=True):
        if isinstance(Ax, Circuit):
            Ax = Ax.op.Ax
        if isinstance(Az, Circuit):
            Az = Az.op.Ax
        Ax = Relation.promote(Ax)
        if Az is None:
            Az = Ax.dual
        Az = Relation.promote(Az)
        self.Ax = Ax
        self.Az = Az
        self.pair = (Ax, Az)
        if check:
            assert Ax.dual == Az
        Circuit.__init__(self, self, Ax.tgt, Ax.src) # i am my own Operator

    @property
    @cache
    def reverse(self):
        Ax, Az = self.Ax, self.Az
        return Operator(Ax.reverse, Az.reverse)

    @property
    @cache
    def dual(self):
        Ax, Az = self.Ax, self.Az
        return Operator(Az, Ax)

    def mul(self, other):
        assert isinstance(other, Operator), type(other)
        Ax = self.Ax * other.Ax
        Az = self.Az * other.Az
        return Operator(Ax, Az)

    def __matmul__(self, other):
        assert isinstance(other, Operator), type(other)
        Ax = self.Ax @ other.Ax
        Az = self.Az @ other.Az
        return Operator(Ax, Az)

    def __eq__(self, other):
        assert isinstance(other, Operator), (type(self), type(other))
        return self.pair == other.pair

    def __hash__(self):
        return hash(self.pair)

    def __str__(self):
        return "(%s | %s)"%(self.Ax.spanstr(), self.Az.spanstr())

    @classmethod
    def black(cls, l, r):
        Rx = Relation.black(l, r)
        Rz = Relation.white(l, r)
        return Operator(Rx, Rz)

    @classmethod
    def white(cls, l, r):
        Rx = Relation.white(l, r)
        Rz = Relation.black(l, r)
        return Operator(Rx, Rz)

    @classmethod
    def identity(cls, n):
        Rx = Relation.identity(n)
        return Operator(Rx, Rx)

    @classmethod
    def zero(cls, l, r):
        Rx = Relation.zero(l, r)
        return Operator(Rx)

    @classmethod
    def from_code(cls, code):
        assert code.is_css()
        css = code.to_css()
    
        left = code.Hx.concatenate(code.Lx)
        right = Matrix.zeros((code.mx, code.k))
        right = right.concatenate(Matrix.identity(code.k))
        Ex = Relation(left, right)
    
        left = code.Hz.concatenate(code.Lz)
        right = Matrix.zeros((code.mz, code.k))
        right = right.concatenate(Matrix.identity(code.k))
        Ez = Relation(left, right)
    
        return Operator(Ex, Ez)
    
    def to_code(op):
        assert isinstance(op, Operator)
        Ex, Ez = op.pair
    
        #print("from_encoder")
        #print(Ex, Ex.shape)
        assert isinstance(Ex, Relation)
        k = Ex.src
        assert Ez.src == k
    
        #if k==0:
        #    print(Ex)
    
        w = Relation.white(1,0)
        if k:
            op = reduce(matmul, [w]*k)
            #print(op, op.shape)
            Hx = (Ex*op).left
            Hz = (Ez*op).left
        else:
            Hx = Ex.left
            Hz = Ez.left
    
        mx, n = Hx.shape
        #print("Hx =")
        #print(Hx, Hx.shape)
    
        mz = len(Hz)
        assert mz+mx+k == n
        #print("Hz =")
        #print(Hz, Hz.shape)
    
        op = Relation(Matrix.identity(k), Matrix.zeros((k,0)))
        HLx = (Ex*op).left
        HHLx = Hx.concatenate(HLx).linear_independent()
        assert HHLx[:mx, :] == Hx
        Lx = HHLx[mx:, :]
    
        #print("Lx =")
        #print(Lx, Lx.shape)
    
        op = Relation(Matrix.identity(k), Matrix.zeros((k,0)))
        HLz = (Ez*op).left
        HHLz = Hz.concatenate(HLz).linear_independent()
        assert HHLz[:mz, :] == Hz
        Lz = HHLz[mz:, :]
    
        code = CSSCode(Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz)
        return code


get_encoder = Operator.from_code
from_encoder = Operator.to_code


def witness(lhs, rhs):
    """
        compose Relation's and return the detector and witness
    """
    assert isinstance(lhs, Relation)
    assert isinstance(rhs, Relation)

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
    op = Relation(left.t, right.t)
    w = None
    return d, w, op
    


class Composite(Circuit):
    def __init__(self, lhs, rhs):
        lhs = Circuit.promote(lhs)
        rhs = Circuit.promote(rhs)
        self.lhs = lhs
        self.rhs = rhs
        lop = lhs.op
        rop = rhs.op
        #op = lop.mul(rop.op)

        dz, wx, Ax = witness(lop.Ax, rop.Ax)
        dx, wz, Az = witness(lop.Az, rop.Az)
        op = Operator(Ax, Az)

        Circuit.__init__(self, op, op.tgt, op.src)

        self.dx = dx
        self.wx = wx
        self.dz = dz
        self.wz = wz

    def __str__(self):
        return "(%s*%s=%s, (%d|%d))"%(self.lhs, self.rhs, self.op, 
            len(self.dx), len(self.dz))

    @property
    @cache
    def reverse(self):
        assert 0, "todo"

    

class Hom:

    _w = Operator.white(0, 1)
    w_ = Operator.white(1, 0)
    _b = Operator.black(0, 1)
    b_ = Operator.black(1, 0)
    I = Operator.identity(1)

    @cache # singleton
    def __new__(cls, m, n=None):
        ob = object.__new__(cls)
        return ob

    def __init__(self, m, n=None):
        if n is None:
            n = m
        self.m = m
        self.n = n

    @property
    def reverse(self):
        return Hom(self.n, self.m)

    def __str__(self):
        return "(%s<--%s)"%(self.m, self.n)
    __repr__ = __str__

    def __getitem__(self, i):
        # i am a shape tuple
        return (self.m, self.n)[i]

    def identity(self):
        assert self.m == self.n
        return Operator.identity(self.m)

    def RX(self, *idxs):
        "prepare X state: |+> on idxs"
        m, n = self
        assert m>n, "wrong Hom %s"%(self,)
        if not idxs:
            idxs = list(range(m-n))
        else:
            idxs = list(idxs)
        assert m == n+len(idxs), "wrong Hom %s"%(self,)
        idxs.sort(reverse=True)
        rels = [self.I]*n
        for i in idxs:
            assert 0<=i<m
            rels.insert(i, self.w_)
        assert len(rels) == m
        rel = reduce(matmul, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n), (str(rel), str(self))
        return rel

    def RZ(self, *idxs):
        "prepare Z state: |0> on idxs"
        m, n = self
        assert m>n, "wrong Hom %s"%(self,)
        if not idxs:
            idxs = list(range(m-n))
        else:
            idxs = list(idxs)
        assert m == n+len(idxs), "wrong Hom %s"%(self,)
        idxs.sort(reverse=True)
        rels = [self.I]*n
        for i in idxs:
            assert 0<=i<m
            rels.insert(i, self.b_)
        assert len(rels) == m
        rel = reduce(matmul, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n)
        return rel

    def MX(self, *idxs):
        "postselect <+| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.reverse.RX(*idxs).reverse
        return rel

    def MZ(self, *idxs):
        "postselect <0| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.reverse.RZ(*idxs).reverse
        return rel

    def CX(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        lhs = Matrix.identity(m)
        rhs = numpy.identity(m, dtype=int)
        rhs[jdx, idx] = 1
        rhs = Matrix(rhs)
        rel = Relation(lhs, rhs)
        return Operator(rel)

    def PERM(self, idxs):
        """ send idxs[j]<---j """
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m == len(idxs), "wrong Hom %s"%(self,)
        assert len(set(idxs)) == m
        lhs = Matrix.identity(m)
        rhs = Matrix.get_perm(idxs)
        rel = Relation(lhs, rhs)
        return Operator(rel)
    get_perm = PERM

    def SWAP(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        f = list(range(self.n))
        f[idx], f[jdx] = f[jdx], f[idx]
        rel = self.get_perm(f)
        return Operator.promote(rel)


def test_composite():

    white = Operator.white
    black = Operator.black
    _w = white(0, 1)
    w_ = white(1, 0)
    _b = black(0, 1)
    b_ = black(1, 0)
    I = Operator.identity(1)
    zero = Operator.zero(0,0)

    cx01 = (I @ black(1,2))*(white(2,1) @ I)
    cx10 = (I @ white(1,2))*(black(2,1) @ I)

    circ = _b * b_
    circ = _w * w_

    circ = (I@_b) * cx01*cx10 * (b_@I)
    #print(circ)

    circ = white(1,2)*white(2,1)
    print(circ)



def test_random():

    def getrand(n):
        hom = Hom(n,n)
        ops = [hom.CX(i,j) for i in range(n) for j in range(n) if i!=j]
        op = hom.identity()
        count = 0
        while count < n**2:
            g = choice(ops)
            op = g*op
            count += 1
        return op

    n = 5
    op = getrand(n)

    i = randint(0,n-1)
    mx = Hom(n-1, n).MZ(i) # 1 <-- 2 --> 2
    op = mx*op

    n -= 1
    op = getrand(n)*op

    i = randint(0,n-1)
    mx = Hom(n-1, n).MZ(i) # 1 <-- 2 --> 2
    op = mx*op
        
    s = str(op)


def eqop(A, B):
    assert isinstance(A, Circuit)
    assert isinstance(B, Circuit)
    return A.op == B.op


def test_hom():

    hom = Hom(3, 2)

    for i in range(3):
        c = hom.RX(i)
        c = hom.RZ(i)

    hom = Hom(2, 3)
    for i in range(3):
        c = hom.MX(i)
        c = hom.MZ(i)

    # test get_perm ----------------------------------------

    rel = Hom(3).get_perm([1,2,0])
    rhs = Hom(3,2).RX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).RX(1) * Hom(2,2).SWAP()
    assert lhs.op == rhs.op
    
    rel = Hom(3).get_perm([2, 0, 1])
    rhs = Hom(3,2).RX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).RX(2)
    assert lhs.op == rhs.op
    
    # test CX ----------------------------------------

    w_ww = Operator.white(1, 2)
    bb_b = Operator.black(2, 1)
    identity = Operator.identity(1)
    CX = (w_ww @ identity) * (identity @ bb_b)

    space = Hom(2)
    assert CX.op == space.CX().op

    cx01 = space.CX(0, 1)
    cx10 = space.CX(1, 0)

    rel = cx01*cx10*cx01
    assert eqop(rel, space.SWAP())

    n = 3
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        cx = Hom(n).CX(i,j)
        assert cx.src == n
        assert cx.tgt == n
        mz = Hom(n-1, n).MZ(i)
        assert mz.src == n
        assert mz.tgt == n-1
        lhs = mz * cx
        rhs = mz 
        assert eqop(lhs, rhs)
    
        lhs = Hom(n-1, n).MX(j) * cx
        rhs = Hom(n-1, n).MX(j)
        assert eqop(lhs, rhs)

    # ------------------------------------------------

    b3 = Operator.black(3,1)
    w3 = Operator.white(3,1)

    iso = Hom(9,9).PERM([0,3,6,1,4,7,2,5,8])
    Ex = (b3@b3@b3)*w3
    #print("Ex:")
    #print(Ex)

    Hx = (Ex*Operator.white(1,0)).op.Ax.left
    #print(Hx)

    Ez = (w3@w3@w3)*b3
    #print("Ez:")
    #print(Ez)

    assert Ez.op == Ex.op.dual
    assert Ex.op == Ez.op.dual

    Hz = (Ez*Operator.white(1,0)).op.Ax.left
    #print(Hz)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.distance()
    #print(code)
    #print(code.longstr())

    assert eqop(get_encoder(code), Operator(Ex, Ez))

    # ------------------------------------------------
    # test from_encoder

    for trial in range(10):

        code = CSSCode.random(20, 5, 5)
        Ex, Ez = get_encoder(code).pair
        assert Ex.dual == Ez
        assert Ez.dual == Ex
        dode = from_encoder(Operator(Ex, Ez))
        assert code.is_equiv(dode)
    

    # ------------------------------------------------

    code = construct.get_422().to_css()
    Ex = get_encoder(code).pair[0]
    left = Ex.left
    #print(left)
    #print(left @ left)

    EE = Ex.tensor(Ex)
    #print(EE)
    dode = from_encoder(Operator(EE))
    #dode.bz_distance()
    #print(dode)
    assert dode.n == 16
    assert dode.k == 4
    

def test():

    from qumba.rel import test_linear
    test_linear()
    test_hom()
    test_random()
    test_composite()



if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)
        ra.seed(_seed)

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


