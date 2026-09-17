#!/usr/bin/env python

"""
Use linear relations for CSS circuits (aka phase-free ZX-calculus).

"""

import time
from functools import reduce, cache
from operator import matmul

import numpy

#from qumba.qcode import QCode, SymplecticSpace
#from qumba import lin
from qumba.matrix import Matrix
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

    def PX(self, *idxs):
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

    def PZ(self, *idxs):
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
        "postselect <0| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.op.PX(*idxs).op
        return rel

    def MZ(self, *idxs):
        "postselect <+| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.op.PZ(*idxs).op
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


def get_encoder(code):
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

    return (Ex, Ez)


def from_encoder(Ex, Ez=None):
    if Ez is None:
        Ez = Ex.dual

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



def test():
    hom = Hom(3, 2)

    for i in range(3):
        c = hom.PX(i)
        c = hom.PZ(i)

    hom = Hom(2, 3)
    for i in range(3):
        c = hom.MX(i)
        c = hom.MZ(i)

    # test get_perm ----------------------------------------

    rel = Hom(3).get_perm([1,2,0])
    rhs = Hom(3,2).PX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).PX(1) * Hom(2,2).SWAP()
    assert lhs == rhs
    
    rel = Hom(3).get_perm([2, 0, 1])
    rhs = Hom(3,2).PX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).PX(2)
    assert lhs == rhs
    
    # test CX ----------------------------------------

    w_ww = Relation.white(1, 2)
    bb_b = Relation.black(2, 1)
    identity = Relation.identity(1)
    CX = (w_ww @ identity) * (identity @ bb_b)

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
    Ex = (b3@b3@b3)*w3
    #print("Ex:")
    #print(Ex)

    Hx = (Ex*Relation.white(1,0)).left
    #print(Hx)

    Ez = (w3@w3@w3)*b3
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

    assert get_encoder(code) == (Ex, Ez)

    # ------------------------------------------------
    # test from_encoder

    Hx = get_rm(1,5)
    Hz = get_rm(2,5)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.bz_distance()
    #print(code)
    #print(code.longstr())

    Ex, Ez = get_encoder(code)

    #print(Ex)
    #print(Ez)
    assert Ex!=Ez

    dode = from_encoder(Ex, Ez)
    dode.bz_distance()

    assert code.is_equiv(dode)

    for trial in range(10):

        code = CSSCode.random(20, 5, 5)
        Ex, Ez = get_encoder(code)
        assert Ex.dual == Ez
        assert Ez.dual == Ex
        dode = from_encoder(Ex, Ez)
        assert code.is_equiv(dode)
    

    # ------------------------------------------------


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
    
        Ei = get_encoder(inner)[0]
        Eo = get_encoder(outer)[0]
    
        #P = Space(4).PERM([0,2,1,3])
        for f in all_perms(list(range(4))):
            P = Space(4).PERM(f)
            E = (Eo@Eo)*P*Ei
            code = from_encoder(E)
            code.bz_distance()
            print(f, code) # [[8,2,2]]
    
        return

    op = get_rm(0,0)
    assert op == Matrix([[1]])
    #print(op, op.shape)

    def pair_cx(left, right):
        assert left.n == right.n
        n = left.n
        Ex = get_encoder(left)[0] @ get_encoder(right)[0]
        hom = Space(2*n)
        CX = hom.CX
        for i in range(n):
            Ex = CX(n+i, i) * Ex
        dode = from_encoder(Ex)
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


