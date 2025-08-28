#!/usr/bin/env python

from functools import reduce
from operator import matmul, add


import numpy
from numpy import linalg
from numpy import random
from numpy import exp, pi, cos, arccos, sin

from qumba.argv import argv

EPSILON = 1e-4
scalar = numpy.complex64

w8 = numpy.exp(1j*pi/4)
w4 = 1j


def bitlog(N):
    N0 = N
    n = 0
    while N>1:
        N //= 2
        n += 1
    assert N0==2**n
    return n


class Matrix:
    def __init__(self, A, name=()):
        A = numpy.array(A, dtype=scalar)
        self.A = A
        shape = A.shape
        N, M = A.shape
        self.n = bitlog(N)
        self.m = bitlog(M)
        self.shape = shape

    def __eq__(self, other):
        assert self.shape == other.shape
        if self.shape != other.shape:
            return False
        return numpy.allclose(self.A, other.A, atol=EPSILON)

    def __repr__(self):
        return "Matrix(%s)"%(self.shape,)

    def __str__(self):
        n = self.n
        m = self.m
        A = self.A
        #print("__str__", n,m)
        idxs = list(numpy.ndindex((2,)*n))
        jdxs = list(numpy.ndindex((2,)*m))
        terms = []
        for i,l in enumerate(idxs):
          for j,r in enumerate(jdxs):
            a = A[i,j]
            if abs(a) < EPSILON:
                #print("A[%d,%d]==0"%(i,j), end=" ")
                continue
            ls = ''.join(str(li) for li in l)
            rs = ''.join(str(ri) for ri in r)
            if ls and rs:
                term = "|%s><%s|"%(ls, rs)
            elif ls:
                term = "|%s>"%(ls,)
            elif rs:
                term = "<%s|"%(rs,)
            else:
                term = "<>"
            if abs(a-1)<EPSILON:
                pass
            elif abs(a+1)<EPSILON:
                term = "-"+term
            elif abs(a.real)<EPSILON:
                term = "%.4fj"%(a.imag)+term
            elif abs(a.imag)<EPSILON:
                term = "%.4f"%(a.real)+term
            else:
                term = str(a)+term
            terms.append(term)
            assert len(terms) <= 1024, "%s too big"%(self.shape,)
        s = "+".join(terms)
        s = s.replace("+-", "-")
        #print()
        return s

    @classmethod
    def identity(self, N):
        A = numpy.identity(N)
        return Matrix(A)

    def __add__(self, other):
        assert self.shape == other.shape
        A = self.A+other.A
        return Matrix(A)

    def __sub__(self, other):
        assert self.shape == other.shape
        A = self.A-other.A
        return Matrix(A)

    def __neg__(self):
        return Matrix(-self.A)

    def __rmul__(self, r):
        return Matrix(r*self.A)

    def __matmul__(self, other):
        A = numpy.kron(self.A, other.A)
        return Matrix(A)

    def __mul__(self, other):
        assert self.shape[1] == other.shape[0], "%s*%s"%(self.shape, other.shape)
        A = numpy.dot(self.A, other.A)
        if A.shape == (1,1):
            return A[0,0]
        return Matrix(A)

    @property
    def d(self):
        "dagger"
        A = self.A.transpose()
        A = A.conjugate()
        return Matrix(A)

    @property
    def t(self):
        "transpose"
        A = self.A.transpose()
        return Matrix(A)

    def trace(self):
        return self.A.trace()


class Space:

    #@cache
    def __new__(cls, n):
        ob = object.__new__(cls)
        return ob

    def __init__(self, n):
        self.n = n
        self.w = exp(1j*pi/4)
        self.I = Matrix.identity(2**n)

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
        assert 0<=i<n
        I = Matrix.identity(2)
        items = [I]*n
        items[i] = g
        gi = reduce(matmul, items)
        gi.name = ("%s(%d)"%(name, i),)
        return gi
        #while len(items)>1:
        #    #items[-2:] = [items[-2] @ items[-1]]
        #    items[:2] = [items[0] @ items[1]]
        #return items[0]

    def PZ(self):
        N = 2**self.n
        u = [[0] for i in range(N)]
        u[0] = [1]
        return Matrix(u)
        
    #@cache
    def Z(self, i=0):
        Z = Matrix([[1, 0], [0, -1]])
        Zi = self.mkop(i, Z, "Z")
        return Zi
    get_Z = Z
        
    #@cache
    def S(self, i=0):
        w = self.w
        S = Matrix([[1, 0], [0, w*w]])
        Si = self.mkop(i, S, "S")
        return Si
    get_S = S
        
    #@cache
    def T(self, i=0):
        w = self.w
        T = Matrix([[1, 0], [0, w]])
        Ti = self.mkop(i, T, "T")
        return Ti
    get_T = T
        
    #@cache
    def X(self, i=0):
        X = Matrix([[0, 1], [1,  0]])
        Xi = self.mkop(i, X, "X")
        return Xi
    get_X = X

    #@cache
    def Y(self, i=0):
        Y = Matrix([[0, -w4], [w4,  0]])
        Yi = self.mkop(i, Y, "Y")
        return Yi
    get_Y = Y
        
    #@cache
    def H(self, i=0):
        w = self.w
        r2 = w+w.conjugate()
        ir2 = r2 / 2
        H = Matrix([[ir2, ir2], [ir2, -ir2]])
        Hi = self.mkop(i, H, "H")
        return Hi
    get_H = H

    #@cache
    def CZ(self, idx=0, jdx=1):
        n = self.n
        assert 0<=idx<n
        assert 0<=jdx<n
        assert idx!=jdx
        N = 2**n
        A = numpy.zeros((N, N))
        ii, jj = 2**(n-idx-1), 2**(n-jdx-1)
        for i in range(N):
            if i & ii and i & jj:
                A[i, i] = -1
            else:
                A[i, i] = 1
        return Matrix(A, "CZ(%d,%d)"%(idx,jdx))
    get_CZ = CZ

    #@cache
    def CY(self, idx=0, jdx=1):
        CX = self.CX(idx, jdx)
        S = self.S(jdx)
        Si = S.d
        CY = S*CX*Si
        CY.name = ("CY(%d,%d)"%(idx,jdx),)
        return CY
    get_CY = CY

    #@cache
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

    #@cache
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
        M = Matrix(rows, name)
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

    #@cache
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



ket0 = Matrix([[1,0]]).d
ket1 = Matrix([[0,1]]).d


def test():


    u0 = Matrix([[1,0],[0,0]])
    u1 = Matrix([[0,0],[0,1]])

    assert u0 == ket0.d @ ket0
    assert u1 == ket1.d @ ket1

    assert str(u0@u1 - u1@u0) == "|01><01|-|10><10|"

    assert u0*u0 == u0
    assert u0*u0 != u1


    CZ = Space(2).CZ()

    s = Space(1)
    I = s.I
    H = s.H()
    S = s.S()
    Z = s.Z()

    assert S*S == Z
    assert H==H.d
    assert S != S.d
    assert S*S.d == I



if __name__ == "__main__":

    numpy.set_printoptions(
        precision=4, threshold=1024, suppress=True, 
        formatter={'float': '{:0.4f}'.format}, linewidth=200)

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

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



