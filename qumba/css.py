#!/usr/bin/env python

"""
Use linear relations for CSS _circuits (aka phase-free ZX-calculus).

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


class Hom:
    # Space = Hom below

    _w = Relation.white(0, 1)
    w_ = Relation.white(1, 0)
    _b = Relation.black(0, 1)
    b_ = Relation.black(1, 0)
    I = Relation.identity(1)

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
    def op(self):
        return Hom(self.n, self.m)

    def __str__(self):
        return "(%s<--%s)"%(self.m, self.n)
    __repr__ = __str__

    def __getitem__(self, i):
        # i am a shape tuple
        return (self.m, self.n)[i]

    def identity(self):
        assert self.m == self.n
        return Relation.identity(self.m)

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
        rel = reduce(lshift, rels)
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
        rel = reduce(lshift, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n)
        return rel

    def MX(self, *idxs):
        "postselect <+| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.op.RX(*idxs).op
        return rel

    def MZ(self, *idxs):
        "postselect <0| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.op.RZ(*idxs).op
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
        return Relation(lhs, rhs)

    def PERM(self, idxs):
        """ send idxs[j]<---j """
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m == len(idxs), "wrong Hom %s"%(self,)
        assert len(set(idxs)) == m
        lhs = Matrix.identity(m)
        rhs = Matrix.get_perm(idxs)
        return Relation(lhs, rhs)
    get_perm = PERM

    def SWAP(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        f = list(range(self.n))
        f[idx], f[jdx] = f[jdx], f[idx]
        return self.get_perm(f)

Space = Hom


class Circuit:
    def __init__(self, op):
        assert isinstance(op, Operator)
        self.op = op

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
        Ax = Relation.promote(Ax)
        if Az is None:
            Az = Ax.dual
        Az = Relation.promote(Az)
        self.Ax = Ax
        self.Az = Az
        self.pair = (Ax, Az)
        if check:
            assert Ax.dual == Az
        Circuit.__init__(self, self) # i am my own Operator

    def mul(self, other):
        assert isinstance(other, Operator)
        Ax = self.Ax * other.Ax
        Az = self.Az * other.Az
        return Operator(Ax, Az)

    def __matmul__(self, other):
        assert isinstance(other, Operator)
        Ax = self.Ax @ other.Ax
        Az = self.Az @ other.Az
        return Operator(Ax, Az)

    def __eq__(self, other):
        assert isinstance(other, Operator)
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
            op = reduce(lshift, [w]*k)
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

        Circuit.__init__(self, op)

        self.dx = dx
        self.wx = wx
        self.dz = dz
        self.wz = wz

    def __str__(self):
        return "(%s*%s=%s, (%d|%d))"%(self.lhs, self.rhs, self.op, 
            len(self.dx), len(self.dz))


    
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
    #d = Relation(d)
    op = Relation(left.t, right.t)

    return d, op


def test_detect():

    _w = Relation.white(0, 1)
    w_ = Relation.white(1, 0)
    _b = Relation.black(0, 1)
    b_ = Relation.black(1, 0)
    I = Relation.identity(1)
    zero = Relation.zero(0,0)

    d, op = detect(_b, b_)
    assert d==Matrix.identity(1)
    assert op==zero


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
        
    print(op.nf)




def test():

    from qumba.rel import test_linear
    test_linear()

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
    assert lhs == rhs
    
    rel = Hom(3).get_perm([2, 0, 1])
    rhs = Hom(3,2).RX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).RX(2)
    assert lhs == rhs
    
    # test CX ----------------------------------------

    w_ww = Relation.white(1, 2)
    bb_b = Relation.black(2, 1)
    identity = Relation.identity(1)
    CX = (w_ww << identity) * (identity << bb_b)

    space = Hom(2)
    assert CX == space.CX()

    cx01 = space.CX(0, 1)
    cx10 = space.CX(1, 0)

    rel = cx01*cx10*cx01
    assert rel == space.SWAP()

    n = 3
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        cx = Hom(n).CX(i,j)
        lhs = Hom(n-1, n).MZ(i) * cx
        rhs = Hom(n-1, n).MZ(i)
        assert lhs == rhs
    
        lhs = Hom(n-1, n).MX(j) * cx
        rhs = Hom(n-1, n).MX(j)
        assert lhs == rhs

    # ------------------------------------------------

    b3 = Relation.black(3,1)
    w3 = Relation.white(3,1)

    iso = Hom(9,9).PERM([0,3,6,1,4,7,2,5,8])
    Ex = (b3<<b3<<b3)*w3
    #print("Ex:")
    #print(Ex)

    Hx = (Ex*Relation.white(1,0)).left
    #print(Hx)

    Ez = (w3<<w3<<w3)*b3
    #print("Ez:")
    #print(Ez)

    assert Ez == Ex.dual
    assert Ex == Ez.dual

    Hz = (Ez*Relation.white(1,0)).left
    #print(Hz)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.distance()
    #print(code)
    #print(code.longstr())

    assert get_encoder(code) == Operator(Ex, Ez)

    # ------------------------------------------------
    # test from_encoder

    Hx = get_rm(1,5)
    Hz = get_rm(2,5)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.bz_distance()
    #print(code)
    #print(code.longstr())

    Ex, Ez = get_encoder(code).pair

    #print(Ex)
    #print(Ez)
    assert Ex!=Ez

    dode = from_encoder(Operator(Ex, Ez))
    dode.bz_distance()

    assert code.is_equiv(dode)

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
    



@cache
def get_rm(r=1, m=4):

    n = 2**m # length
    if r==-1:
        H = Matrix.zeros((0, n))
        return H

    assert 0<=r<=m, "r=%s, m=%d"%(r, m)

    one = Matrix([1]*n).A
    basis = [one]

    vs = [[] for i in range(m)]
    for i in range(2**m):
        for j in range(m):
            vs[j].append(i%2)
            i >>= 1
        assert i==0

    vs = [Matrix(v).A for v in vs]

    for k in range(r):
        for items in choose(vs, k+1):
            v = one
            #print(items)
            for u in items:
                v = v*u
            basis.append(v)

    H = Matrix(basis)
    H = H.linear_independent()
    return H


def test_rm():

    #for r in [-1,0,1,2,3]:
    #    print("r =", r)
    #    H = get_rm(r,3)
    #    print(H, H.shape)
    #return

    RM = {}
    lookup = [[] for m in range(5)]
    for N in range(5):
      for l in range(-1,N+1):
        for r in range(-1,N+1):
            Hx = get_rm(l, N)
            Hz = get_rm(r, N)
            if (Hx*Hz.t).max():
                continue
            code = CSSCode(Hx=Hx, Hz=Hz)
            if code.k and code.mx and code.mz:
                code.bz_distance()
            else:
                code.distance_z3()
            key = (N, l, r)
            RM[key] = code
            lookup[N].append(code)
            name = "RM(%d,%d,%d)"%(N,l,r)
            code.name = name
            if l>=0 and r>=0:
                print("%s=%s"%(name,code), end=' ')
        print()

    if 0:
        # does not seem to get anywhere
        outer = RM[2,0,0] # [[4,2,2]]
        print(outer)
        inner = RM[2,0,0]
    
        Ei = get_encoder(inner).pair[0]
        Eo = get_encoder(outer).pair[0]
    
        #P = Space(4).PERM([0,2,1,3])
        for f in all_perms(list(range(4))):
            P = Space(4).PERM(f)
            E = (Eo<<Eo)*P*Ei
            code = from_encoder(Operator(E))
            code.bz_distance()
            print(f, code) # [[8,2,2]]
    
        return

    op = get_rm(0,0)
    assert op == Matrix([[1]])
    #print(op, op.shape)

    def pair_cx(left, right):
        assert left.n == right.n
        n = left.n
        Ex = get_encoder(left).pair[0] << get_encoder(right).pair[0]
        hom = Space(2*n)
        CX = hom.CX
        for i in range(n):
            Ex = CX(n+i, i) * Ex
        dode = from_encoder(Operator(Ex))
        return dode

    dode = pair_cx(RM[2,1,0], RM[2,0,1])
    assert dode.is_equiv(RM[3,1,1])

    dode = pair_cx(RM[3,2,0], RM[3,1,1])
    assert dode.is_equiv(RM[4,2,1])

    dode = pair_cx(RM[3,1,0], RM[3,0,1])
    assert dode.is_equiv(RM[4,1,1])

    def find(code):
        for k,dode in RM.items():
            if dode.is_equiv(code):
                return dode

    for left in lookup[3]:
      for right in lookup[3]:
        code = pair_cx(left, right)
        dode = find(code)
        if dode:
            print(left.name, right.name, "->", dode.name)

    return

#    for l in [2,1,0]:
#        rm = RM[3,l,2-l]
#        print(rm)
#        print(rm.longstr())
#        print()
    
    smap = SMap()
    keys = list(RM.keys())
    keys.sort()
    w,h = 32, 18
    for (N,l,r) in keys:
        if N!=3:
            continue
        desc = "RM(%s,%s,%s)"%(N,l,r)
        smap[h*l, w*r] = desc
        smap[h*l+1, w*r] = str(RM[N,l,r])
        smap[h*l, w*r+18] = RM[N,l,r].longstr()

    #print()
    #print(smap)

    def lzero(code):
        Hz = code.Hz.concatenate(code.Lz)
        return CSSCode(Hx=code.Hx, Hz=Hz)

    def lplus(code):
        Hx = code.Hx.concatenate(code.Lx)
        return CSSCode(Hx=Hx, Hz=code.Hz)

    rm311 = RM[3,1,1]
    rm310 = RM[3,1,0]
    rm301 = RM[3,0,1]

    assert( lzero(rm310).is_equiv( rm311 ) )
    assert( lplus(rm301).is_equiv( rm311 ) )

    keys = [(N,l,r) for (N,l,r) in keys if N==4]
    #print( lzero(RM[4,1,1]).is_equiv( RM[4,1,2] ) )
    for lkey in keys:
      for rkey in keys:
        if lkey==rkey:
            continue
        if lzero(RM[lkey]).is_equiv(RM[rkey]):
            print("lzero", lkey, rkey)
        if lplus(RM[lkey]).is_equiv(RM[rkey]):
            print("lplus", lkey, rkey)



def test_tutte():
    for m in [1,2,3,4]:
        for l in range(m+1):
            H = get_rm(l, m)
            print(H.shape, H.get_tutte())

    return

    lookup = {}
    for m in [1,2,3,4]:
        Hs = [get_rm(l, m) for l in range(m+1)]
        for idx,H in enumerate(Hs):
          for jdx,J in enumerate(Hs):
            A = H*J.t
            #print(int(A.max() == 0), end=' ')
            if A.max() > 0:
                continue
            css = CSSCode(Hx=H, Hz=J)
            if css.k==0:
                pass
            elif css.n < 64:
                css.bz_distance()
            else: 
                distance_z3_css(css)
            lookup[m,idx,jdx] = css
            print(str(css).ljust(14), end=' ', flush=True)
          print()
        print()

    #print(list(lookup.keys()))

    found = set()
    for m in [1,2,3,4]:
      for idx in range(m+1):
        for jdx in range(m+1):
            key = (m, idx, jdx)
            if key not in lookup:
                continue
            css = lookup[key]
            H = css.Hx
            #if H in found:
            #    continue
            found.add(H)
            p = H.get_tutte()
            #print(H)
            print(key, css, p)
            #print()

      print()






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


