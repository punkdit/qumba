#!/usr/bin/env python

from functools import reduce
from operator import mul
from cmath import *

import numpy
class Scalar(object):
    dtype = numpy.complex128
    @classmethod
    def zeros(cls, arg):
        return numpy.zeros(arg, dtype=cls.dtype)
    @classmethod
    def array(cls, A):
        return numpy.array(A, dtype=cls.dtype)

class RealScalar(Scalar):
    dtype = numpy.float64
    

EPSILON = 1e-8
NO_NAME = "?"


class Space(object):
    def __init__(self, n, name=NO_NAME, scalar=Scalar):
        assert type(n) is int
        self.n = n
        self.name = name
        self.scalar = scalar
        self.dtype = scalar.dtype

    def __str__(self):
        return self.name

    def __hash__(self):
        return id(self)

    # __eq__ is object identity

    def identity(self):
        A = numpy.identity(self.n, dtype=self.dtype)
        return Lin(self, self, A)

    def zeros(self):
        A = numpy.zeros((self.n, self.n), dtype=self.dtype)
        return Lin(self, self, A)

    def perm(self, f):
        A = numpy.identity(self.n, dtype=self.dtype)
        A = A[:, f]
        return Lin(self, self, A)


def fstr(x):
    re = x.real
    im = x.imag
    if abs(im) < EPSILON:
        x = re
        if abs(x) < EPSILON:
            s = " 0."
        elif abs(x-1) < EPSILON:
            s = " 1."
        elif abs(x+1) < EPSILON:
            s = "-1."
        else:
            s = "%.6f"%x
    else:
        s = str(x)
    return s


def astr(A):
    m, n = A.shape
    lines = []
    for i in range(m):
        row = ' '.join(fstr(x) for x in A[i, :])
        if i==0:
            row = '[[%s]'%row
        elif i==m-1:
            row = ' [%s]]'%row
        else:
            row = ' [%s]'%row
        lines.append(row)
    return '\n'.join(lines)

    


class Lin(object):
    "_Linear operator"

    def __init__(self, tgt, src, A=None):
        assert tgt.scalar is src.scalar
        scalar = tgt.scalar
        if A is None:
            A = scalar.zeros(self.ring, tgt.n, src.n)
        if type(A) is list:
            A = scalar.array(A)
        #for row in A:
        #    assert None not in row
        assert A.shape == (tgt.n, src.n), "%s != %s" % ( A.shape , (tgt.n, src.n) )
        self.src = src
        self.tgt = tgt
        self.scalar = scalar
        self.hom = (tgt, src) # yes it's backwards, just like shape is.
        self.shape = A.shape
        self.A = A.copy()

    def __str__(self):
        return astr(self.A)

    def __eq__(self, other):
        assert self.hom == other.hom, "um..?"
        return numpy.allclose(self.A, other.A, atol=EPSILON)

    def __add__(self, other):
        assert self.hom == other.hom
        A = self.A + other.A
        return Lin(self.tgt, self.src, A)

    def __sub__(self, other):
        assert self.hom == other.hom
        A = self.A - other.A
        return Lin(self.tgt, self.src, A)

    def __mul__(self, other):
        assert self.src == other.tgt
        A = numpy.dot(self.A, other.A)
        return Lin(self.tgt, other.src, A)

    def __rmul__(self, r):
        A = r*self.A
        return Lin(self.tgt, self.src, A)

    def __neg__(self):
        A = -self.A
        return Lin(self.tgt, self.src, A)

#    def __matmul__(self, other):

    def __pow__(self, n):
        assert self.tgt == self.src
        if n==0:
            return self.tgt.identity()
        lin = reduce(mul, [self]*n)
        return lin

    def transpose(self):
        A = self.A.transpose()
        return Lin(self.src, self.tgt, A)


def test_pauli(n):

    print("test_pauli(%d)"%n)
    V = Space(n, "V")

    N = V.zeros()
    I = V.identity()
    assert I==I
    assert I*I == I
    assert I+N == I
    assert I*N == N

    X = numpy.zeros((n, n), dtype=numpy.complex128)
    for i in range(n):
        X[i, (i+1)%n] = 1.
    X = Lin(V, V, X)
    assert X**0 == I
    assert X != I
    assert X**n == I

    phase = lambda i : exp(2*pi*i*1.j/n)

    Z = numpy.zeros((n, n), dtype=numpy.complex128)
    for i in range(n):
        Z[i, i] = phase(i)
    Z = Lin(V, V, Z)
    assert Z**0 == I
    assert Z != I
    assert Z**n == I

    Xi = X**(n-1)
    Zi = Z**(n-1)

    assert X*Z*Xi*Zi == phase(1)*I


    H = numpy.zeros((n, n), dtype=numpy.complex128)
    for i in range(n):
      for j in range(n):
        H[i, j] = phase(i*j)
    H *= n**(-1/2)
    Hi = H.transpose().conjugate()
    H = Lin(V, V, H)
    Hi = Lin(V, V, Hi)

    if n==2:
        assert Hi == H
    assert Hi*H == I

    lhs = H*Z*Hi
    assert (lhs == X)

    rhs = H*X*Hi
    assert (rhs == Zi)


def all_perms(items):
    items = tuple(items)
    if len(items)<=1:
        yield items
        return
    n = len(items)
    for i in range(n):
        for rest in all_perms(items[:i] + items[i+1:]):
            yield (items[i],) + rest


def all_signed_perms(V):
    n = V.n
    items = list(range(n))
    signs = [2*numpy.array(v)-1 for v in numpy.ndindex((2,)*n)]
    for f in all_perms(items):
        P = V.perm(f)
        for sign in signs:
            A = P.A * sign
            SP = Lin(V, V, A)
            yield SP

    
def test_symplectic():
    n = 4
    nn = 2*n
    scalar = RealScalar
    V = Space(nn, "V", scalar)
    I = V.identity()

    F = scalar.zeros((nn, nn))
    for i in range(n):
        F[i, n+i] = 1.
        F[n+i, i] = -1.
    F = Lin(V, V, F)
    print(F)

    count = 0
    total = 0
    for P in all_signed_perms(V):
        total += 1
        #print(P)
        #print()
        Pi = P.transpose()
        #assert P*Pi == I
        if P*F*Pi == F:
            count += 1
    print(count, "of", total)


def test():
    for n in [2, 3]:
        test_pauli(n)
    test_symplectic()


if __name__ == "__main__":

    from time import time
    start_time = time()

    from qumba.argv import argv

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


        

