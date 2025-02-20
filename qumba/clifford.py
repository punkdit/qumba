#!/usr/bin/env python

"""
build clifford/pauli groups (and some non-cliffords) in sage

ported from:
https://github.com/punkdit/bruhat/blob/master/bruhat/clifford_sage.py

"""

from random import choice
from operator import mul, matmul, add
from functools import reduce
#from functools import cache
from functools import lru_cache
cache = lru_cache(maxsize=None)

import numpy

from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
    PolynomialRing)
from sage import all_cmdline 

from qumba.lin import zeros2, identity2
from qumba.action import mulclose, mulclose_names, mulclose_find
from qumba.argv import argv

if argv.sage or 1:
    # much faster for big matrices
    print("using qumba.matrix_sage")
    from qumba.matrix_sage import Matrix
else:
    # BROKEN
    print("using qumba.matrix_numpy")
    from qumba.matrix_numpy import Matrix

from qumba.clifford_ring import degree
K = CyclotomicField(degree)
root = K.gen() # primitive eighth root of unity
w8 = root ** (degree // 8)
half = K.one()/2
w4 = w8**2
r2 = w8 + w8**7
ir2 = r2 / 2
assert r2**2 == 2

def simplify_latex(self):
    M = self.M
    m, n = self.shape
    idxs = [(i,j) for i in range(m) for j in range(n)]
    for idx in idxs:
        if M[idx] != 0:
            break
    else:
        assert 0
    scale = M[idx]
    if scale != 1 and scale != -1:
        M = (1/scale) * M
        s = {
            r2 : r"\sqrt{2}",
            1/r2 : r"\frac{1}{\sqrt{2}}",
            #2/r2 : r"\frac{2}{\sqrt{2}}",
            2/r2 : r"\sqrt{2}",
            #r2/2 : r"\sqrt{2}/2",
        }.get(scale, latex(scale))
        if "+" in s:
            s = "("+s+")"
        s = "%s %s"%(s, latex(M))
    else:
        s = latex(M)
    s = s.replace(r"\zeta_{8}^{2}", "i")
    return s


matrix = lambda rows : Matrix(K, rows)
I = matrix([[1, 0], [0, 1]])
H = (r2/2) * matrix([[1, 1], [1, -1]])
X = matrix([[0, 1], [1, 0]])
Z = matrix([[1, 0], [0, -1]])
Y = matrix([[0, -w4], [w4, 0]])


class Coset(object):
    def __init__(self, group, items):
        self.group = group
        self.items = list(items)
        self.g = items[0] # pick one

    def __mul__(self, other):
        assert isinstance(other, Coset)
        gh = self.g * other.g
        result = self.group.lookup[gh]
        return result

    def __hash__(self):
        return id(self)


class FactorGroup(object):
    def __init__(self, G, H):
        remain = set(G)
        cosets = []
        while remain:
            g = iter(remain).__next__()
            coset = []
            for h in H:
                gh = g*h
                remain.remove(gh)
                coset.append(gh)
            coset = Coset(self, coset) # promote
            cosets.append(coset)
            #if len(remain)%1000 == 0:
            #    print(len(remain), end=" ", flush=True)
        #print()
        self.G = G
        self.H = H
        lookup = {g:coset for coset in cosets for g in coset.items}
        self.lookup = lookup
        #cosets = [Coset(self, coset) for coset in cosets] # promote
        self.cosets = cosets

    def __getitem__(self, idx):
        return self.cosets[idx]

    def __len__(self):
        return len(self.cosets)


class Clifford(object):
    "clifford group on n qubits"

    @cache
    def __new__(cls, n):
        ob = object.__new__(cls)
        return ob

    def __init__(self, n):
        self.n = n
        self.K = K
        self.w = w8
        self.I = Matrix.identity(K, 2**n)

    def wI(self):
        w = self.w
        return w*self.I
    get_wI = wI

    def w2I(self):
        w2 = self.w**2
        return w2*self.I
    get_w2I = w2I

    def get_identity(self):
        return self.I
    get_I = get_identity

    def mkop(self, i, g, name):
        n = self.n
        K = self.K
        assert 0<=i<n
        I = Matrix.identity(K, 2)
        items = [I]*n
        items[i] = g
        gi = reduce(matmul, items)
        gi.name = ("%s(%d)"%(name, i),)
        return gi
        #while len(items)>1:
        #    #items[-2:] = [items[-2] @ items[-1]]
        #    items[:2] = [items[0] @ items[1]]
        #return items[0]
        
    @cache
    def Z(self, i=0):
        K = self.K
        Z = Matrix(K, [[1, 0], [0, -1]])
        Zi = self.mkop(i, Z, "Z")
        return Zi
    get_Z = Z
        
    @cache
    def S(self, i=0):
        K = self.K
        w = self.w
        S = Matrix(K, [[1, 0], [0, w*w]])
        Si = self.mkop(i, S, "S")
        return Si
    get_S = S
        
    @cache
    def T(self, i=0):
        K = self.K
        w = self.w
        T = Matrix(K, [[1, 0], [0, w]])
        Ti = self.mkop(i, T, "T")
        return Ti
    get_T = T
        
    @cache
    def X(self, i=0):
        K = self.K
        X = Matrix(K, [[0, 1], [1,  0]])
        Xi = self.mkop(i, X, "X")
        return Xi
    get_X = X

    @cache
    def Y(self, i=0):
        K = self.K
        Y = Matrix(K, [[0, -w4], [w4,  0]])
        Yi = self.mkop(i, Y, "Y")
        return Yi
    get_Y = Y
        
    @cache
    def H(self, i=0):
        K = self.K
        w = self.w
        r2 = w+w.conjugate()
        ir2 = r2 / 2
        H = Matrix(K, [[ir2, ir2], [ir2, -ir2]])
        Hi = self.mkop(i, H, "H")
        return Hi
    get_H = H

    @cache
    def CZ(self, idx=0, jdx=1):
        n = self.n
        K = self.K
        assert 0<=idx<n
        assert 0<=jdx<n
        assert idx!=jdx
        N = 2**n
        A = zeros2(N, N)
        ii, jj = 2**(n-idx-1), 2**(n-jdx-1)
        for i in range(N):
            if i & ii and i & jj:
                A[i, i] = -1
            else:
                A[i, i] = 1
        return Matrix(K, A, "CZ(%d,%d)"%(idx,jdx))
    get_CZ = CZ

    @cache
    def CY(self, idx=0, jdx=1):
        CX = self.CX(idx, jdx)
        S = self.S(jdx)
        Si = S.d
        CY = S*CX*Si
        CY.name = ("CY(%d,%d)"%(idx,jdx),)
        return CY
    get_CY = CY

    @cache
    def CNOT(self, idx=0, jdx=1):
        assert idx != jdx
        CZ = self.CZ(idx, jdx)
        H = self.H(jdx)
        CX = H*CZ*H
        CX.name = ("CX(%d,%d)"%(idx,jdx),)
        return CX
    CX = CNOT
    get_CNOT = CNOT
    get_CX = CNOT

    @cache
    def SWAP(self, idx=0, jdx=1):
        assert idx != jdx
        #HH = self.H(idx) * self.H(jdx)
        #CZ = self.CZ(idx, jdx)
        #g = HH*CZ*HH*CZ*HH*CZ
        idxs = list(range(self.n))
        idxs[idx],idxs[jdx] = idxs[jdx],idxs[idx]
        return self.get_P(*idxs)
        #assert g==self.get_P(*idxs)
        #return g
    get_SWAP = SWAP

    def get_P(self, *perm):
        #print("get_P", perm)
        I = self.I
        n = self.n
        N = 2**n
        idxs = list(numpy.ndindex((2,)*n))
        lookup = {idx:i for (i,idx) in enumerate(idxs)}
        #print(lookup)
        p = [lookup[tuple(idx[perm[i]] for i in range(n))] for idx in idxs]
        rows = []
        for i in p:
            row = [0]*N
            row[i] = 1
            rows.append(row)
        name = "P%s"%(perm,)
        M = Matrix(self.K, rows, name)
        return M
    P = get_P

    def get_expr(self, expr, rev=False):
        if expr == ():
            op = self.I
        elif type(expr) is tuple:
            if rev:
                expr = reversed(expr)
            op = reduce(mul, [self.get_expr(e) for e in expr]) # recurse
        else:
            expr = "self.get_"+expr
            op = eval(expr, {"self":self})
        return op

    def get_pauli(self, desc):
        assert len(desc) == self.n
        op = self.I
        for i,c in enumerate(desc):
            if c in ".I":
                continue
            method = getattr(self, "get_"+c)
            pauli = method(i)
            op = pauli*op
        return op

    @cache
    def pauli_group(self, phase=0):
        names = [()]
        gen = [self.get_identity()]
        if phase==2:
            names += [('w2I')]
            gen += [self.get_w2I()]
        elif phase==1:
            names += [('wI')]
            gen += [self.get_wI()]
        else:
            assert phase==0
        for i in range(self.n):
            X, Z = self.get_X(i), self.get_Z(i)
            names.append("X(%d)"%i)
            names.append("Z(%d)"%i)
            gen += [X, Z]
        names = mulclose_names(gen, names)
        return names



dim = 2 # qubits

def tpow(v, n, one=matrix([[1]])):
    return reduce(matmul, [v]*n) if n>0 else one


@cache
def green(m, n, phase=0):
    # m legs <--- n legs
    assert phase in [0, 1, 2, 3]
    r = w4**phase
    G = reduce(add, [
        (r**i) * tpow(ket(i), m) * tpow(bra(i), n)
        for i in range(dim)])
    return G

@cache
def red(m, n, phase=0):
    # m legs <--- n legs
    assert phase in [0, 1, 2, 3]
    G = green(m, n, phase)
    R = tpow(H, m) * G * tpow(H, n)
    return R



def ket(i):
    v = [0]*dim
    v[i] = 1
    v = matrix([[w] for w in v])
    return v

def bra(i):
    v = ket(i)
    v = v.transpose()
    return v


#def green(m, n, phase=0):
#    # m legs <--- n legs
#    assert phase in [0, 1, 2, 3]
#    r = w4**phase
#    ket_0 = [0]*(dim**m)
#    ket_0[0] = 1
#    ket_0 = matrix([[w] for w in ket_0])
#    
#    ket_1 = [0]*(dim**m)
#    ket_1[-1] = 1
#    ket_1 = matrix([[w] for w in ket_1])
#
#    return ket_0 + ket_1
    

def test_spider():

    v0 = ket(0) # |0>
    v1 = ket(1) #    |1>
    u0 = bra(0) # <0|
    u1 = bra(1) #    <1|

#    print(green(1,0))
#    print(green(2,0))
#    print(green(3,0))
#    return

    I = v0*u0 + v1*u1
    assert green(1, 1) == I
    assert red(1, 1) == I

    assert tpow(ket(0), 1) == v0
    assert tpow(ket(0), 2) == v0@v0
    assert green(2, 2) == v0 @ v0 @ u0 @ u0 + v1 @ v1 @ u1 @ u1
    a = ket(0) * bra(0)
    assert green(1,2).shape == (v0@u0@u0).shape

    assert v0 == (r2/2)*red(1,0)
    assert v1 == (r2/2)*red(1,0,2)

    assert green(1, 1, 2) == Z
    assert red(1, 1, 2) == X

    assert green(2,0) == v0@v0 + v1@v1
    lhs, rhs = green(3,0) , v0@v0@v0 + v1@v1@v1
    assert lhs == rhs

    # green frobenius algebra
    mul = green(1, 2)
    comul = green(2, 1)
    unit = green(1, 0)
    counit = green(0, 1)

    assert mul * (I @ unit) == I
    assert mul * (I @ mul) == mul * (mul @ I)
    assert mul * comul == I

    cap = counit * mul
    cup = comul * unit

    assert (I @ cap)*(cup @ I) == I

    assert ( mul * (v0 @ v0) ) == v0
    assert ( mul * (v1 @ v1) ) == v1
    assert ( mul * (v0 @ v1) ).is_zero()
    assert ( mul * (v1 @ v0) ).is_zero()

    A = counit * mul * (X @ I) * comul
    assert A.is_zero()


    assert mul * (X@X) == X*mul
    assert (X@X) * comul == comul*X
    assert counit * X == counit
    assert X * unit == unit

    assert green(2, 1) * green(1, 2) + (X@I) * green(2, 1) * green(1, 2) * (X@I) == I@I

    # red frobenius algebra
    _mul = red(1, 2)
    _comul = red(2, 1)
    _unit = red(1, 0)
    _counit = red(0, 1)

    u0 = H*ket(0) # |->
    u1 = H*ket(1) #    |+>

    assert u0 == (r2/2)*green(1,0)
    assert u1 == (r2/2)*green(1,0,2)

    assert _mul * (I @ _unit) == I
    assert _mul * (I @ _mul) == _mul * (_mul @ I)
    assert _mul * _comul == I

    cap = _counit * _mul
    cup = _comul * _unit

    assert (I @ cap)*(cup @ I) == I

    assert ( _mul * (u0 @ u0) ) == u0
    assert ( _mul * (u1 @ u1) ) == u1
    assert ( _mul * (u0 @ u1) ).is_zero()
    assert ( _mul * (u1 @ u0) ).is_zero()

    A = _counit * _mul * (Z @ I) * _comul
    assert A.is_zero()

    assert _mul * (Z@Z) == Z*_mul
    assert (Z@Z) * _comul == _comul*Z
    assert _counit * Z == _counit
    assert Z * _unit == _unit

    assert red(2, 1) * red(1, 2) + (Z@I) * red(2, 1) * red(1, 2) * (Z@I) == I@I





def test_clifford():

    c2 = Clifford(2)
    II = c2.I
    XI = c2.X(0)
    IX = c2.X(1)
    ZI = c2.Z(0)
    IZ = c2.Z(1)
    wI = c2.wI()

    Pauli = mulclose([wI*wI, XI, IX, ZI, IZ])
    assert len(Pauli) == 64, len(Pauli)

    assert c2 is Clifford(2)

    SI = c2.S(0)
    IS = c2.S(1)
    HI = c2.H(0)
    IH = c2.H(1)
    CZ = c2.CZ(0, 1)

    #C2 = mulclose([SI, IS, HI, IH, CZ], verbose=True) # slow
    #assert len(C2) == 92160

    CNOT = c2.CNOT(0, 1)
    SWAP = c2.SWAP(0, 1)

    assert SWAP * CZ * SWAP == CZ

    c2 = Clifford(3)
    A = c2.CNOT(0, 1)
    B = c2.CNOT(1, 2)
    assert A*B != B*A

    A = c2.CZ(0, 1)
    B = c2.CZ(1, 2)
    assert A*B == B*A

    CNOT01 = c2.CNOT(0, 1)
    CNOT10 = c2.CNOT(1, 0)
    CZ01 = c2.CZ(0, 1)
    CZ10 = c2.CZ(1, 0)
    SWAP01 = c2.SWAP(0, 1)

    assert HI*HI == II
    assert HI*XI*HI == ZI
    assert HI*ZI*HI == XI
    assert HI*IX*HI == IX
    assert SWAP01 == CNOT01*CNOT10*CNOT01

    lhs = c2.get_expr(('CZ(0,1)', 'CNOT(0,1)'))
    rhs = c2.get_expr('CZ(0,1)') * c2.get_expr('CNOT(0,1)')
    assert lhs==rhs

    c = Clifford(1)
    X, S, Y = c.X(), c.S(), c.Y()
    assert Y == S*X*S.d

    c = Clifford(2)
    M = c.CY()
    assert M[2:, 2:] == Y


def test_CY():
    c = Clifford(2)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S

    gen = [H(0), H(1), S(0), S(1), S(0)**3, S(1)**3, CX()]
    g = mulclose_find(gen, CY())
    assert g.name == ('S(1)', 'CX(0,1)', 'S(1)', 'S(1)', 'S(1)')

    gen = [H(0), H(1), S(0), S(1), S(0)**3, S(1)**3, CZ()]
    g = mulclose_find(gen, CY())
    assert g.name == ('H(1)', 'S(0)', 'CZ(0,1)', 'H(1)', 'CZ(0,1)')

    gen = [H(0), H(1), S(0), S(1), S(0)**3, S(1)**3, CX(1,0)]
    g = mulclose_find(gen, CY())
    assert g.name == ('H(0)', 'H(1)', 'CX(1,0)', 'H(1)', 'CX(1,0)', 'H(0)', 'S(0)')



def test_clifford3():
    c3 = Clifford(3)
    wI = c3.wI()
    III = c3.I
    XII = c3.X(0)
    IXI = c3.X(1)
    IIX = c3.X(2)
    ZII = c3.Z(0)
    IZI = c3.Z(1)
    IIZ = c3.Z(2)
    SII = c3.S(0)
    ISI = c3.S(1)
    IIS = c3.S(2)
    HII = c3.H(0)
    IHI = c3.H(1)
    IIH = c3.H(2)
    CNOT01 = c3.CNOT(0, 1)
    CNOT10 = c3.CNOT(1, 0)
    CNOT02 = c3.CNOT(0, 2)
    CNOT20 = c3.CNOT(2, 0)
    CNOT12 = c3.CNOT(1, 2)
    CNOT21 = c3.CNOT(2, 1)
    CNOT12 = c3.CNOT(1, 2)
    CZ01 = c3.CZ(0, 1)
    CZ02 = c3.CZ(0, 2)
    CZ12 = c3.CZ(1, 2)
    SWAP01 = c3.SWAP(0, 1)
    SWAP02 = c3.SWAP(0, 2)
    SWAP12 = c3.SWAP(1, 2)

    assert HII*HII == III
    assert HII*XII*HII == ZII
    assert HII*ZII*HII == XII
    assert HII*IXI*HII == IXI

    III = c3.I
    assert c3.get_pauli("XXX") == c3.get_X(0)*c3.get_X(1)*c3.get_X(2)
    assert c3.get_pauli("ZZZ") == c3.get_Z(0)*c3.get_Z(1)*c3.get_Z(2)

    XXX = c3.get_pauli("XXX")
    ZZZ = c3.get_pauli("ZZZ")
    assert XXX*XXX == III
    assert ZZZ*ZZZ == III
    assert XXX*ZZZ == -ZZZ*XXX

    I = Clifford(1).I
    c2 = Clifford(2)
    assert CZ01 == c2.CZ(0,1)@I
    assert SWAP01 == CNOT01*CNOT10*CNOT01

    assert CNOT01*CNOT02 == CNOT02*CNOT01
    assert CNOT12*CNOT01 == CNOT02*CNOT01*CNOT12



def test_perm():

    cliff = Clifford(3)

    perm = [1, 2, 0]
    P = cliff.get_P(*perm)
    #print(P)

    k0 = (r2/2)*red(1,0) #   |0>
    k1 = (r2/2)*red(1,0,2) # |1>
    bits = [
        k0 @ k1 @ k1,
        k1 @ k0 @ k1,
        k1 @ k1 @ k0]
    for i, l in enumerate(bits):
      for j, r in enumerate(bits):
        if ( P*l == r ):
            assert perm[i] == j
            #print(i, "--->", j)
        else:
            assert perm[i] != j
            #print(i, "-X->", j)



def test_bruhat():
    K = CyclotomicField(8)
    w = w8
    one = K.one()
    w2 = w*w
    r2 = w+w.conjugate()
    ir2 = r2 / 2
    
    I = Matrix(K, [[1, 0], [0, 1]])
    w2I = Matrix(K, [[w2, 0], [0, w2]])
    S = Matrix(K, [[1, 0], [0, w2]])
    X = Matrix(K, [[0, 1], [1, 0]])
    Z = Matrix(K, [[1, 0], [0, -1]])
    H = Matrix(K, [[ir2, ir2], [ir2, -ir2]])
    
    Pauli1 = mulclose([w2I, X, Z])
    Cliff1 = mulclose([w2I, S, H])
    
    assert len(Pauli1) == 16
    assert len(Cliff1) == 192
    
    CNOT = Matrix(K,
       [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]])
    
    CZ = Matrix(K,
       [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0,-1]])
    
    SWAP = Matrix(K,
       [[1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])
    
    II = I @ I
    w2II = w2*II
    XI = X @ I
    IX = I @ X
    ZI = Z @ I
    IZ = I @ Z
    SI = S @ I
    IS = I @ S
    HI = H @ I
    IH = I @ H

    Pauli2 = mulclose([XI, IX, ZI, IZ])

    lhs = CZ * CNOT
    rhs = CNOT * CZ

    g = rhs * lhs.inverse()
    print(g)
    print(g == ZI)
    print(g == IZ)

    return

    Cliff2 = mulclose([SI, IS, HI, IH, CZ], verbose=True)
    #assert len(Cliff2) == 92160

    torus = []
    for g in Cliff2:
        if g.is_diagonal():
            #print(g)
            torus.append(g)
    print("torus:", len(torus))

    while 1:
        gen = [choice(torus) for i in range(4)]
        T = mulclose(gen)
        if len(T) == len(torus):
            break
    print("gen", len(gen))

    # ------------------------------------------------------------------
    # See:
    # https://arxiv.org/abs/2003.09412
    # Hadamard-free circuits expose the structure of the Clifford group
    # Sergey Bravyi, Dmitri Maslov
    # although this is for the projective Clifford group only 

    n = 2
    W = mulclose([HI, IH, SWAP]) # Weyl group
    assert len(W) == 8

    #B = mulclose([XI, IX, CNOT, CZ, SI, IS])
    B = mulclose([XI, IX, CNOT, CZ, SI, IS]+gen, verbose=True)
    print("Borel:", len(B))

    #T = [g for g in torus if g in B]
    #print(len(T))
    #return
    #B.extend(torus)

    # build the double cosets:
    dcs = {w:set() for w in W}
    total = 0
    for w in W:
        dc = dcs[w]
        size = 0
        for l in B:
          lw = l*w
          for r in B:
            lwr = lw*r
            dc.add(lwr)
          if len(dc) > size:
            size = len(dc)
            print(size, end=" ", flush=True)
        print("//")
        total += size
    print("total:", total) # == 92160 ?

    for w in W:
        for u in W:
            if u==w:
                continue
            a = dcs[w].intersection(dcs[u])
            #assert len(a) == 0
            print(len(a), end=" ")
    print()

    return dcs


def test_cocycle():
    K = CyclotomicField(8)
    w = w8
    one = K.one()
    w2 = w*w
    r2 = w+w.conjugate()
    ir2 = r2 / 2
    
    I = Matrix(K, [[1, 0], [0, 1]])
    w2I = Matrix(K, [[w2, 0], [0, w2]])
    S = Matrix(K, [[1, 0], [0, w2]])
    X = Matrix(K, [[0, 1], [1, 0]])
    Z = Matrix(K, [[1, 0], [0, -1]])
    H = Matrix(K, [[ir2, ir2], [ir2, -ir2]])
    
    Pauli1 = mulclose([w2I, X, Z])
    Cliff1 = mulclose([w2I, S, H])
    
    assert len(Pauli1) == 16
    assert len(Cliff1) == 192
    
    CZ = Matrix(K,
       [[1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0,-1]])
    
    II = I @ I
    w2II = w2*II
    XI = X @ I
    IX = I @ X
    ZI = Z @ I
    IZ = I @ Z
    SI = S @ I
    IS = I @ S
    HI = H @ I
    IH = I @ H

    pauli_gen = [XI, IX, ZI, IZ]
    Pauli2 = mulclose([w2II] + pauli_gen)
    assert len(Pauli2) == 64

    Phase2 = mulclose([w2II])
    assert len(Phase2) == 4

    F4 = FactorGroup(Pauli2, Phase2)
    assert len(F4) == 16

    sy_gen = [ZI, IZ, XI, IX]
    pauli_lin = lambda g: [int(g*h != h*g) for h in sy_gen]
    
    # no phase generator needed here
    Cliff2 = mulclose([SI, IS, HI, IH, CZ], verbose=True)
    assert len(Cliff2) == 92160

    F2 = FiniteField(2)
    cliff_lin = lambda g:Matrix(F2, [pauli_lin(g*h*g.inverse()) for h in pauli_gen]).transpose()
    Sp4 = mulclose([cliff_lin(g) for g in [SI,IS,HI,IH,CZ]])
    assert len(Sp4) == 720

    Mp4 = FactorGroup(Cliff2, Pauli2)
    assert len(Mp4) == 1440

    hom = {} # Mp4 --> Sp4
    for coset in Mp4:
        g = coset.g # Cliff2
        assert g in Cliff2
        h = cliff_lin(g)
        hom[coset] = h
    
    Sp4_i = Matrix(F2, [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    kernel = [g for g in Mp4 if hom[g]==Sp4_i]
    assert len(kernel) == 2
    
    homi = {g:[] for g in Sp4}
    for g in Mp4:
        homi[hom[g]].append(g)
    #print([len(v) for v in homi.values()])
    def cocyc(g, h): # Sp4xSp4 --> Z/2
        gh = g*h
        gi, hi, ghi = homi[g], homi[h], homi[gh]
        lhs = gi[0]*hi[0]
        assert lhs in ghi
        return ghi.index(lhs)
    items = list(Sp4)
    for _ in range(64):
        g, h, k = [choice(items) for _ in range(3)]
        #print(cocyc(g, h), end=" ")
        lhs = cocyc(g, h*k) + cocyc(h, k)
        rhs = cocyc(g*h, k) + cocyc(g, h)
        assert lhs%2 == rhs%2
        print("%s=%s"%(lhs%2,rhs%2), end=" ")


def test_CCZ():

    I = Clifford(1).I

    c2 = Clifford(2)
    wI = c2.wI()
    II = c2.I
    XI = c2.X(0)
    IX = c2.X(1)
    ZI = c2.Z(0)
    IZ = c2.Z(1)
    SI = c2.S(0)
    IS = c2.S(1)
    HI = c2.H(0)
    IH = c2.H(1)
    CNOT01 = c2.CNOT(0, 1)
    CNOT10 = c2.CNOT(1, 0)
    CZ01 = c2.CZ(0, 1)
    CZ10 = c2.CZ(1, 0)
    SWAP01 = c2.SWAP(0, 1)

    assert HI*HI == II
    assert HI*XI*HI == ZI
    assert HI*ZI*HI == XI
    assert HI*IX*HI == IX
    assert SWAP01 == CNOT01*CNOT10*CNOT01

    #print( CNOT01 )
    #print( CNOT10 )
    #print( CZ01 )
    #print( IH * CZ01 * IH )
    #print( HI * CZ01 * HI )

    c3 = Clifford(3)
    wI = c3.wI()
    III = c3.I
    XII = c3.X(0)
    IXI = c3.X(1)
    IIX = c3.X(2)
    ZII = c3.Z(0)
    IZI = c3.Z(1)
    IIZ = c3.Z(2)
    SII = c3.S(0)
    ISI = c3.S(1)
    IIS = c3.S(2)
    HII = c3.H(0)
    IHI = c3.H(1)
    IIH = c3.H(2)
    CNOT01 = c3.CNOT(0, 1)
    CNOT10 = c3.CNOT(1, 0)
    CNOT02 = c3.CNOT(0, 2)
    CNOT20 = c3.CNOT(2, 0)
    CNOT12 = c3.CNOT(1, 2)
    CNOT21 = c3.CNOT(2, 1)
    CNOT12 = c3.CNOT(1, 2)
    CZ01 = c3.CZ(0, 1)
    CZ02 = c3.CZ(0, 2)
    CZ12 = c3.CZ(1, 2)
    SWAP01 = c3.SWAP(0, 1)
    SWAP02 = c3.SWAP(0, 2)
    SWAP12 = c3.SWAP(1, 2)

    assert HII*HII == III
    assert HII*XII*HII == ZII
    assert HII*ZII*HII == XII
    assert HII*IXI*HII == IXI

    assert CZ01 == c2.CZ(0,1)@I
    assert SWAP01 == CNOT01*CNOT10*CNOT01

    assert CNOT01*CNOT02 == CNOT02*CNOT01
    assert CNOT12*CNOT01 == CNOT02*CNOT01*CNOT12

    N = 2**c3.n
    rows = []
    for i in range(N):
        row = [0]*N
        row[i] = -1 if i==N-1 else 1
        rows.append(row)
    CCZ = Matrix(c3.K, rows)

    g = CCZ * XII*IXI*IIX * CCZ.inverse()
    print("g =")
    print(g)

    half = c3.K.one() / 2
    op = half*(IIX + ZII*IIX + IZI*IIX - ZII*IZI*IIX)

    ns =locals()

    #names = "wI SII ISI IIS HII IHI IIH CZ01 CZ02 CZ12".split()
    names = "XII IXI IIX CZ01 CZ02 CZ12".split()
    gen = [ns[name] for name in names]
    #name = mulclose_find(gen, names, g, verbose=True)
    #print(name)


def test_platonic():

    c = Clifford(1)
    I = c.I
    wI = c.wI()
    S = c.get_S()
    H = c.get_H()
    J = w8*H

    Z = S*S
    X = H*Z*H
    G = mulclose([Z, X])
    assert len(G) == 8

    i = w4
    A = half*Matrix(K, [[-1+i,-1+i],[1+i,-1-i]])
    B = -half*Matrix(K, [[-1-i,-1-i],[1-i,-1+i]])
    
    # binary tetrahedral group
    G_2T = mulclose([A, B])
    assert len(G_2T) == 24
    names = mulclose_names([A, B], "AB")
    for g,name in names.items():
        if g == -I:
            print(g, ''.join(name), "in 2T")
    print(Z in G_2T, X in G_2T)

    # binary octahedral group
    #C = ir2 * Matrix(K, [[1+i, 0], [0, 1-i]]) # has order 8
    C = w8*S # simpler to generate
    print(C, "=C")
    #print([I==C**j for j in range(10)])
    G_2O = mulclose([A, C])
    assert len(G_2O) == 48
    names = mulclose_names([A, C], "AC")
    for g,name in names.items():
        if g == B:
            print(g, ''.join(name), "in 2O")
    print(Z in G_2O, X in G_2O)

    # Clifford group
    G = mulclose([S, H])
    assert len(G) == 192
    assert wI in G
    for g in G_2O:
        assert g in G
    print(Z in G, X in G)

    names = mulclose_names([I, w8*I, S, H], list("IwSH"))
    assert len(names) == 192
    for g,name in names.items():
        if g == C:
            print(g, ''.join(name), "= C in Clifford")

    # Semi-Clifford group
    G = mulclose([S, J])
    assert w4*I in G
    assert len(G) == 96
    print(Z in G, X in G)

    #names = mulclose_names([I, w4*I, S, J], list("IiSJ"))
    names = mulclose_names([I, S, J], list("ISJ"))
    for g,name in names.items():
        if g == w4*I:
            print(g, ''.join(name), "in Semi-Clifford")


def test_higher():
    "higher Clifford group elements"
    c1 = Clifford(1)
    c2 = Clifford(2)
    ring = c1.K
    w = c1.w
    T = Matrix(ring, [[1,0],[0,w]])
    I = c1.get_identity()
    assert [T**n==I for n in range(1,9)] == [False]*7+[True]
    H, S = c1.get_H(), c1.get_S()
    X, Y, Z = c1.X(), c1.Y(), c1.Z()
    assert T*T == S
    Tx = H*T*H
    Sx = H*S*H
    Sy = S*H*S*H*S.d
    Ty = S*H*T*H*S.d
    assert Sx*Sx == X
    assert Sy*Sy == Y
    assert Tx*Tx == Sx
    assert Ty*Ty == Sy
    assert Ty.d*X*Ty == H
    I1 = c1.get_identity()
    M = I1.direct_sum(Sx)
    assert M*M == c2.get_CNOT()


def graph_state(A, normalize=False):

    A = numpy.array(A, dtype=int)
    n = len(A)
    assert n>1
    A = A + A.transpose()
    A = numpy.clip(A, 0, 1)
    assert numpy.all(A==A.transpose())
    assert A.min() in [0,1]
    assert A.max() in [0,1]

    #print("graph_state")
    #print(A)

    lhs = reduce(matmul, [green(1, A[i].sum()) for i in range(n)])
    links = []
    for i in range(n):
      for j in range(i+1, n):
        if A[i,j]==0:
            continue
        links.append((i,j))

    legs = []
    N = 0
    for i in range(n):
        leg = []
        for j in range(A[i].sum()):
            leg.append(N)
            N += 1
        legs.append(leg)

    #if N==0:
    #    return lhs # <------------- return

    #print("legs:", legs)
    #print("links:", links)
    idxs = []
    for (i,j) in links:
        #ii,jj = (legs[i].pop(0), legs[j].pop(0))
        idxs.append(legs[i].pop(0))
        idxs.append(legs[j].pop(0))
    #print(idxs)
    idxs = list(enumerate(idxs))
    #print(idxs)
    idxs.sort(key = lambda idx:idx[1])
    #print(idxs)
    perm = [idx[0] for idx in idxs]
    #print("perm:", perm)
    iperm = [{perm[i]:i for i in range(N)}[j] for j in range(N)]
    #print("iperm:", iperm)

    if N>0:
        c = Clifford(N)
        P = c.get_P(*iperm)
    
        assert N%2 == 0, N
        cup = (H@I) * green(2,0)
        assert green(2,0) == red(2,0)
        state = reduce(matmul, [cup]*(N//2))
        state = P*state
        assert lhs.shape == (2**n, 2**N)
        state = lhs*state
    else:
        state = lhs

    if normalize:
        # HACK THIS:
        #print(state.shape)
        r = (state.dagger()*state).M[0,0]
        w = 1
        while r > 1:
            assert int(r)%2 == 0
            r //= 2
            w *= ir2
        #print(bits)
        state = w*state
        assert (state.dagger()*state).M[0,0] == 1

    return state
    

def graph_op(m, n, A, normalize=True):
    A = numpy.array(A, dtype=int)
    mn = m+n
    assert A.shape == (mn, mn)

    idxs = list(range(m)) + list(reversed(range(m, mn)))
    #print(idxs)
    #print(A)
    A = A[idxs, :]
    A = A[:, idxs]
    #print(A)

    u = graph_state(A)

    cap = green(0, 2)
    for i in range(n):
        #print(u.shape)
        u = u @ I
        #print(u.shape)
        op = Clifford(mn-i-1).I @ cap
        #print(op.shape)
        u = op * u

    assert u.shape == (2**m, 2**n)

    if normalize:
        # HACK THIS:
        v = u.dagger()
        r = (u*v).M[0,0]
        w = 1
        while r != 1:
            w *= r2
            r *= 2
        assert r==1, (r, w)
        u = w*u

    return u
        
        


def test_graph_state():

    # See:
    # https://arxiv.org/abs/1902.03178
    # eq (11)

    n = 4

    c = Clifford(n)
    X, Z = c.X, c.Z

    A = numpy.zeros((n,n), dtype=int)
    A[0,1] = 1
    A[0,2] = 1
    A[1,3] = 1
    
    u = graph_state(A)

    for op in [
        X(0)*Z(1)*Z(2),
        X(1)*Z(0)*Z(3),
        X(2)*Z(0),
        X(3)*Z(1),
    ]:
        assert op*u == u

    #print("\n\ntest:")

    A[:] = 0
    A[0,3] = A[1,2] = 1
    u = graph_state(A)
    #print(u.dagger()*u) # == 4 arghh?!?!
    for op in [
        X(0)*Z(3),
        X(1)*Z(2),
        X(2)*Z(1),
        X(3)*Z(0),
    ]:
        #print(op.shape)
        assert op*u == u


    # ---------------------------------------

    n = 4
    c = Clifford(n)
    X, Z = c.X, c.Z
    A = numpy.zeros((n,n), dtype=int)
    idxs = [(i,j) for i in range(n) for j in range(i+1,n)]
    N = len(idxs)
    for bits in numpy.ndindex((2,)*N):
        A[:] = 0
        for i in range(N):
            if bits[i]:
                A[idxs[i]] = 1
        if A.sum()>3:
            continue
        A = A+A.transpose()
        #print(A)
        u = graph_state(A, normalize=True)
        #print(u.shape)
        for i in range(n):
            op = X(i)
            for j in range(n):
                if A[i,j]:
                    op = op*Z(j)
            #if op*u == u:
                #print(".",end="",flush=True)
            #else:
                #print("*",end="",flush=True)
            assert op*u==u
        assert (u.dagger()*u).M[0,0] == 1 


def test_graph_op():

    n = 2
    nn = 2*n
    A = numpy.zeros((nn,nn), dtype=int)
    A[0,3] = 1
    A[1,2] = 1
    op = graph_op(2,2,A)
    assert op*op == I@I
    assert op == (H@H)*Clifford(2).P(1,0)
    assert op*op.dagger() == I@I, "non-unitary "
    A0 = A

    A = numpy.zeros((nn,nn), dtype=int)
    A[0,2] = 1
    A[1,3] = 1
    op = graph_op(2,2,A)
    assert op==H@H
    assert op*op.dagger() == I@I, "non-unitary "

    A = numpy.zeros((nn,nn), dtype=int)
    A[0,2] = 1
    A[0,3] = 1
    A[1,3] = 1
    op = graph_op(2,2,A)
    assert op*op.dagger() == I@I, "non-unitary "

    rhs = Clifford(2).CNOT(0,1) * (H@H)
    assert op == rhs


def test_prep():
    g2, r2 = Z, X
    assert g2 == green(1,1,2)
    g_, _g = green(1, 0), green(0,1)
    r_, _r = red(1, 0), red(0,1)

    assert (r2*g_ == g_)
    assert (g2*g_ != g_)

    print(g2*g_)
    print(r2*r_)


def get_cliff1():
    K = CyclotomicField(8)
    w = w8
    one = K.one()
    w2 = w*w
    r2 = w+w.conjugate()
    ir2 = r2 / 2
    
    I = Matrix(K, [[1, 0], [0, 1]])
    w2I = Matrix(K, [[w2, 0], [0, w2]], ('w2I',))
    assert w2I**2 == -I
    S = Matrix(K, [[1, 0], [0, w2]], ('S',))
    X = Matrix(K, [[0, 1], [1, 0]])
    Z = Matrix(K, [[1, 0], [0, -1]])
    H = Matrix(K, [[ir2, ir2], [ir2, -ir2]], ('H',))
    
    Pauli1 = mulclose([w2I, X, Z])
    Cliff1 = mulclose([w2I, S, H])
    
    assert len(Pauli1) == 16
    assert len(Cliff1) == 192

    return Cliff1


def test_heirarchy():
    Cliff1 = get_cliff1()

    names = set()
    for g in Cliff1:
        name = g.name
        print(name)
        #name = tuple(n for n in name if n!='w2I')
        #names.add(name)
    #print("names:", len(names))

    nI = -I
    assert nI in Cliff1

    refls = [g for g in Cliff1 if g*g==I]
    count = 0
    n = len(refls)
    for i in range(n):
      g = refls[i]
      for j in range(i+1,n):
        h = refls[j]
        if g*h == -h*g:
            G = mulclose([g,h])
            assert len(G) == 8
            count += 1
    print("found:", count)


    
def test():
    test_clifford()
    test_clifford3()
    test_bruhat()
    test_CCZ()
    test_spider()
    test_perm()
    #test_cocycle() # sloooow
    test_higher()
    test_CY()
    print("done")



if __name__ == "__main__":

    from time import time
    start_time = time()

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%_seed)
        seed(_seed)

    profile = argv.profile
    fn = argv.next() or "test"

    print("%s()"%fn)

    if profile:
        import cProfile as profile
        profile.run("%s()"%fn)

    else:
        fn = eval(fn)
        fn()

    print("\nOK: finished in %.3f seconds"%(time() - start_time))
    print()

