#!/usr/bin/env python

"""
build clifford/pauli groups in sage

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

from sage.all_cmdline import FiniteField, CyclotomicField, latex, block_diagonal_matrix
from sage import all_cmdline 

from qumba.solve import zeros2, identity2
from qumba.action import mulclose, mulclose_names, mulclose_find
from qumba.argv import argv

K = CyclotomicField(8)
w8 = K.gen() # primitive eighth root of unity
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


class Matrix(object):
    def __init__(self, ring, rows, name=()):
        M = all_cmdline.Matrix(ring, rows)
        M.set_immutable()
        self.M = M
        self.ring = ring
        self.shape = (M.nrows(), M.ncols())
        if type(name) is str:
            name = (name,)
        self.name = name

    def __eq__(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        assert self.shape == other.shape
        return self.M == other.M

    def __hash__(self):
        return hash(self.M)

    def __str__(self):
        lines = str(self.M).split("\n")
        lines[0] = "[" + lines[0]
        lines[-1] = lines[-1] + "]"
        lines[1:] = [" "+l for l in lines[1:]]
        return '\n'.join(lines)
    __repr__ = __str__

    def __mul__(self, other):
        assert isinstance(other, Matrix), type(other)
        assert self.ring == other.ring
        assert self.shape[1] == other.shape[0], (
            "cant multiply %sx%s by %sx%s"%(self.shape + other.shape))
        M = self.M * other.M
        name = self.name + other.name
        return Matrix(self.ring, M, name)

    def __add__(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        M = self.M + other.M
        return Matrix(self.ring, M)

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        M = self.M - other.M
        return Matrix(self.ring, M)

    def __neg__(self):
        M = -self.M
        return Matrix(self.ring, M)

    def __pow__(self, n):
       assert n>=0
       if n==0:
           return Matrix.identity(self.ring, self.shape[0])
       return reduce(mul, [self]*n)

    def __rmul__(self, r):
        M = r*self.M
        return Matrix(self.ring, M)

    def __matmul__(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        M = self.M.tensor_product(other.M)
        return Matrix(self.ring, M)
    tensor_product = __matmul__

    def direct_sum(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        #M = self.M.direct_sum(other.M)
        M = block_diagonal_matrix(self.M, other.M)
        return Matrix(self.ring, M)

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.M[idx]
        M = self.M[idx]
        return Matrix(self.ring, M)

    def _latex_(self):
        M = self.M
        s = M._latex_()
        if "zeta_" not in s:
            return s
        return simplify_latex(self)

    @classmethod
    def identity(cls, ring, n):
        rows = []
        for i in range(n):
            row = [0]*n
            row[i] = 1
            rows.append(row)
        return Matrix(ring, rows)

    def inverse(self):
        M = self.M.inverse()
        return Matrix(self.ring, M)

    def transpose(self):
        M = self.M.transpose()
        return Matrix(self.ring, M)

    @property
    def t(self):
        return self.transpose()

    def dagger(self):
        M = self.M.conjugate_transpose()
        name = tuple(name+".d" for name in reversed(self.name))
        return Matrix(self.ring, M, name)

    @property
    def d(self):
        return self.dagger()

    def is_diagonal(self):
        M = self.M
        return M.is_diagonal()

    def is_zero(self):
        return self == -self

    def rank(self):
        return self.M.rank()


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
        K = CyclotomicField(8)
        self.K = K
        self.w = K.gen()
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
        HH = self.H(idx) * self.H(jdx)
        CZ = self.CZ(idx, jdx)
        return HH*CZ*HH*CZ*HH*CZ

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


def green(m, n, phase=0):
    # m legs <--- n legs
    assert phase in [0, 1, 2, 3]
    r = w4**phase
    G = reduce(add, [
        (r**i) * tpow(ket(i), m) * tpow(bra(i), n)
        for i in range(dim)])
    return G

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



def test_spider():

    v0 = ket(0) # |0>
    v1 = ket(1) #    |1>
    u0 = bra(0) # <0|
    u1 = bra(1) #    <1|

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
    assert len(Pauli) == 64

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
    w = K.gen()
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
    w = K.gen()
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
    H,S = c1.get_H(),c1.get_S()
    R = H*S*H
    I1 = c1.get_identity()
    M = I1.direct_sum(R)
    assert M*M == c2.get_CNOT()



def test():
    test_clifford()
    test_clifford3()
    test_bruhat()
    test_CCZ()
    test_spider()
    test_perm()
    #test_cocycle() # sloooow
    test_higher()



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


