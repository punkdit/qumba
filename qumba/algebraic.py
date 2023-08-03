#!/usr/bin/env python3

"""
Algebraic groups: matrix groups over Z/pZ.

see also:
    quantale.py
    combinatorial.py
    orthogonal.py

"""


import sys, os
from time import time
start_time = time()
import random
from random import randint, choice
from functools import reduce
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

#scalar = numpy.int64
scalar = numpy.int8 # CAREFUL !!

from bruhat.action import mulclose, mulclose_hom
from bruhat.spec import isprime
from bruhat.argv import argv
from bruhat.solve import parse, enum2, row_reduce, span, shortstr, rank, shortstrx, pseudo_inverse, intersect
from bruhat.solve import zeros2, identity2
from bruhat.dev import geometry
from bruhat.util import cross, allperms, choose
from bruhat.smap import SMap

EPSILON = 1e-8

DEFAULT_P = argv.get("p", 2)


def qchoose_2(n, m, p=DEFAULT_P):
    if m>n:
        return
    col = m
    row = n-m
    for A in geometry.get_cell(row, col, p):
        yield A


def pdiv(i, j, p=DEFAULT_P):
    " i/j mod p "
    assert j!=0
    for k in range(1, p):
        if (j*k)%p == i:
            return k
    assert 0

def pinv(i, p=DEFAULT_P):
    return pdiv(1, i, p)


def swap_row(A, j, k):
    row = A[j, :].copy()
    A[j, :] = A[k, :]
    A[k, :] = row

def swap_col(A, j, k):
    col = A[:, j].copy()
    A[:, j] = A[:, k]
    A[:, k] = col

def row_reduce_p(A, p=DEFAULT_P, truncate=False, inplace=False, check=False, verbose=False):
    """ Remove zero rows if truncate==True
    """

    zero = 0
    one = 1

    assert len(A.shape)==2, A.shape
    m, n = A.shape
    if not inplace:
        A = A.copy()

    if m*n==0:
        if truncate and m:
            A = A[:0, :]
        return A

    if verbose:
        print("row_reduce")
        #print("%d rows, %d cols" % (m, n))

    i = 0
    j = 0
    while i < m and j < n:
        if verbose:
            print("i, j = %d, %d" % (i, j))
            print("A:")
            print(A)

        assert i<=j
        if i and check:
            assert (A[i:,:j]!=0).sum() == 0

        # first find a nonzero entry in this col
        for i1 in range(i, m):
            if A[i1, j]:
                break
        else:
            j += 1 # move to the next col
            continue # <----------- continue ------------

        if i != i1:
            if verbose:
                print("swap", i, i1)
            swap_row(A, i, i1)

        assert A[i, j] != zero
        for i1 in range(i+1, m):
            if A[i1, j]:
                r = p-pdiv(A[i1, j], A[i, j], p)
                if verbose:
                    print("add %d times row %s to %s" % (r, i, i1))
                A[i1, :] += r*A[i, :]
                A %= p
                assert A[i1, j] == zero

        i += 1
        j += 1

    if truncate:
        m = A.shape[0]-1
        #print("sum:", m, A[m, :], A[m, :].sum())
        while m>=0 and (A[m, :]!=0).sum()==0:
            m -= 1
        A = A[:m+1, :]

    if verbose:
        print()

    return A


# see orthogonal.py for numba version
def normal_form_p(A, p=DEFAULT_P, truncate=True):
    "reduced row-echelon form"
    #print("normal_form")
    #print(A)
    A = row_reduce_p(A, truncate=truncate)
    #print(A)
    m, n = A.shape
    j = 0
    for i in range(m):
        while j<n and A[i, j] == 0:
            j += 1
        if j==n:
            break
        r = A[i, j]
        if r != 1:
            A[i, :] = pinv(r, p) * A[i, :]
            A %= p
        assert A[i, j] == 1
        i0 = i-1
        while i0>=0:
            r = A[i0, j]
            if r!=0:
                A[i0, :] += (p-r)*A[i, :]
                A %= p
            i0 -= 1
        j += 1
    #print(A)
    return A

def test_row_reduce():
    p = 3
    m, n = 4, 4
    A = numpy.zeros((m, n))
    for idx in numpy.ndindex(A.shape):
        A[idx] = random.randint(0, p-1)
    B = row_reduce_p(A, p=p, verbose=False)
    C = normal_form_p(B, p)
    print(A)
    print(B)
    print(C)

_cache = {}
def normal_form(A, p=DEFAULT_P):
    "reduced row-echelon form"
    if p!=2:
        return normal_form_p(A, p)
    key = A.tobytes()
    if key in _cache:
        return _cache[key]
    #print("normal_form")
    #print(A)
    A = row_reduce(A)
    #print(A)
    m, n = A.shape
    j = 0
    for i in range(m):
        while A[i, j] == 0:
            j += 1
        i0 = i-1
        while i0>=0:
            r = A[i0, j]
            if r!=0:
                A[i0, :] += A[i, :]
                A %= p
            i0 -= 1
        j += 1
    #print(A)
    _cache[key] = A
    return A



def all_matrices(m, n, p=DEFAULT_P):
    shape = ((p,)*m*n)
    for idxs in numpy.ndindex(shape):
        M = numpy.array(idxs)
        M.shape = (m, n)
        yield M

def all_codes(m, n, p=DEFAULT_P):
    assert p==2
    for m1 in range(m+1):
        for M1 in geometry.all_codes(m1, n):
            M = numpy.zeros((m, n), dtype=scalar)
            M[:m1, :] = M1
            yield M


class Matrix(object):
    def __init__(self, A, p=DEFAULT_P, shape=None, name="?"):
        if type(A) == list or type(A) == tuple:
            A = numpy.array(A, dtype=scalar)
        else:
            A = A.astype(scalar) # makes a copy
        if shape is not None:
            A.shape = shape
        self.A = A
        #n = A.shape[0]
        #assert A.shape == (n, n)
        assert int(p) == p
        assert p>=0
        self.p = p
        #self.n = n
        if p>0:
            self.A %= p
        self.key = (self.p, self.A.tobytes())
        self._hash = hash(self.key)
        self.shape = A.shape
        self.name = name

    @classmethod
    def perm(cls, items, p=DEFAULT_P, name="?"):
        n = len(items)
        A = numpy.zeros((n, n), dtype=scalar)
        for i, ii in enumerate(items):
            A[ii, i] = 1
        return Matrix(A, p, name=name)

    @classmethod
    def identity(cls, n, p=DEFAULT_P):
        A = numpy.identity(n, dtype=scalar)
        return Matrix(A, p, name="I")

    def get_bitkey(self):
        assert self.p == 2
        A = numpy.packbits(self.A)
        return A.tobytes()

    def __str__(self):
        return str(self.A)

    def __repr__(self):
        return "Matrix(%s)"%str(self.A)

    def shortstr(self):
        return shortstr(self.A)

    def __hash__(self):
        return self._hash

    def is_zero(self):
        return self.A.sum() == 0

    def __len__(self):
        return len(self.A)

    def __eq__(self, other):
        assert self.p == other.p
        return self.key == other.key

    def __ne__(self, other):
        assert self.p == other.p
        return self.key != other.key

    def __lt__(self, other):
        assert self.p == other.p
        return self.key < other.key

    def __add__(self, other):
        A = self.A + other.A
        return Matrix(A, self.p)

    def __sub__(self, other):
        A = self.A - other.A
        return Matrix(A, self.p)

    def __neg__(self):
        A = -self.A
        return Matrix(A, self.p)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            A = numpy.dot(self.A, other.A)
            return Matrix(A, self.p, name=self.name+other.name)
        else:
            return NotImplemented

    def __rmul__(self, r):
        A = r*self.A
        return Matrix(A, self.p)

    def __getitem__(self, idx):
        A = self.A[idx]
        #print("__getitem__", idx, type(A))
        if type(A) is scalar:
            return A
        return Matrix(A, self.p)

    def transpose(self):
        A = self.A
        return Matrix(A.transpose(), self.p)

    def sum(self):
        return self.A.sum()

    def mask(self, A):
        return Matrix(self.A * A, self.p) # pointwise multiply !

    def normal_form(self, cols=None):
        A = self.A
        m, n = A.shape
        if cols is not None:
            assert len(cols)==n
            A0 = A
            A = A[:, cols]
            inv = [cols.index(i) for i in range(n)]
            B = A[:, inv]
            #print()
            #print(A0)
            #print(A, cols)
            #print(B, inv)
            assert str(A0) == str(B)
        A = normal_form(A, self.p)
        if cols is not None:
            A = A[:, inv]
        return Matrix(A, self.p)

    @classmethod
    def all_codes(cls, m, n, p=DEFAULT_P):
        assert p==2
        for A in geometry.all_codes(m, n):
            yield cls(A, p)

    def get_pivots(M):
        A = M.A
        m, n = A.shape
        pivots = []
        col = 0
        for row in range(m):
            while col < n:
                if A[row, col]:
                    pivots.append(col)
                    col += 1
                    break
                col += 1
        return tuple(pivots)

    def inverse(self):
        assert self.p == 2
        B = pseudo_inverse(self.A)
        return Matrix(B, self.p)

    def span(self):
        V = []
        assert self.p == 2
        for v in span(self.A):
            v = Matrix(v, self.p)
            V.append(v)
        return V

    def order(self):
        n = len(self)
        I = Matrix.identity(n)
        count = 1
        g = self
        while g != I:
            g = self*g
            count += 1
        return count


def test_matrix():
    M = Matrix([[1,1,0],[0,1,0]])
    M1 = Matrix([[1,0,0],[0,1,0]])
    assert M.normal_form() == M1

    M = Matrix([[1,2,0],[0,1,2]], p=3)
    M1 = Matrix([[1,0,2],[0,1,0]], p=3)
    assert M.normal_form() == M1


# https://math.stackexchange.com/questions/34271/
# order-of-general-and-special-linear-groups-over-finite-fields

def order_gl(n, q):
    order = 1
    for i in range(n):
        order *= (q**n - q**i)
    return order

def order_sl(n, q):
    order = order_gl(n, q)
    assert order % (q-1) == 0
    return order//(q-1)


def order_sp(n, q):
    # n = 2*m
    assert n%2==0
    m = n//2
    N = q**(m**2)
    for i in range(m):
        N *= (q**(2*(i+1)) - 1)
    return N

assert order_sp(2, 2)==6     # 3!
assert order_sp(4, 2)==720   # 6!


class Algebraic(object):
    def __init__(self, gen, order=None, p=DEFAULT_P, G=None, verbose=False, **kw):
        self.__dict__.update(kw)
        self.gen = list(gen)
        if G is not None:
            assert order is None or order==len(G)
            order = len(G)
        self.order = order
        self.G = G # elements
        self.p = p
        assert gen
        A = gen[0]
        self.n = len(A)
        assert p == A.p
        self.I = Matrix.identity(self.n, p)
        self.verbose = verbose

    def get_elements(self):
        if self.G is None:
            I = self.I
            G = mulclose(self.gen, maxsize=self.order, verbose=self.verbose)
            G.remove(I)
            G.add(I)
            G = list(G)
            self.G = G
            self.order = len(self.G)
        return self.G

    def sample(self):
        gen = self.gen
        N = 10*len(gen)
        A = choice(gen)
        for i in range(N):
            A = A*choice(gen)
        return A

    def __len__(self):
        if self.order is None:
            self.get_elements()
        return self.order

    def __getitem__(self, idx):
        if self.G is None:
            self.get_elements()
        return self.G[idx]

    def __contains__(self, g):
        return g in self.get_elements()

    def __eq__(self, other):
        return set(self.get_elements()) == set(other.get_elements())

    def left_stabilizer(self, M):
        # find subgroup that stabilize the rowspace of M
        V = M.span()
        H = []
        for g in self:
            for v in V:
                if g*v not in V:
                    break
            else:
                H.append(g)
        return Algebraic(H)
    
    def right_stabilizer(self, M):
        # find subgroup that stabilize the rowspace of M
        V = M.span()
        H = []
        for g in self:
            for v in V:
                if v*g not in V:
                    break
            else:
                H.append(g)
        return Algebraic(H)
    
    def show(self):
        items = [M.A for M in self]
        M = numpy.array(items)
    
        smap = SMap()
        for (i, j) in numpy.ndindex(M.shape[1:]):
            if numpy.alltrue(M[:, i, j] == 0):
                smap[i,j] = "."
            elif numpy.alltrue(M[:, i, j] == 1):
                smap[i,j] = "1"
            else:
                smap[i,j] = "*"
        return str(smap)

    @classmethod
    def SL(cls, n, p=DEFAULT_P, **kw):
        "special linear group"
        assert int(n)==n
        assert int(p)==p
        assert n>0
        assert isprime(p)
    
        I = numpy.identity(n, scalar)
        gen = []
        for i in range(n):
            for j in range(n):
                if i==j:
                    continue
                A = I.copy()
                A[i, j] = 1
                gen.append(Matrix(A, p))
        order = order_sl(n, p)
        return cls(gen, order, p=p, **kw)
    
    @classmethod
    def GL(cls, n, p=DEFAULT_P, **kw):
        "general linear group"
        assert int(n)==n
        assert int(p)==p
        assert n>0
        assert isprime(p)
    
        H = cls.SL(n, p)
        gen = list(H.gen)
        for i in range(2, p):
            A = Matrix(i*numpy.identity(n, scalar), p)
            gen.append(A)
        order = order_gl(n, p)
        return GL(gen, order, p=p, **kw)

    # See:
    # Pairs of Generators for Matrix _Algebraics. I
    # D. E. Taylor 2006
    # https://www.maths.usyd.edu.au/u/don/papers/genAC.pdf
    # Also:
    # http://doc.sagemath.org/html/en/reference/groups/sage/groups/matrix_gps/symplectic.html
    
    @classmethod
    def Sp_4_2(cls, F, **kw):
        A = numpy.array([[1,0,1,1],[1,0,0,1],[0,1,0,1],[1,1,1,1]], dtype=scalar)
        B = numpy.array([[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,1,0,0]], dtype=scalar)
        gen = [Matrix(A, 2), Matrix(B, 2)]
        return Sp(gen, 720, p=2, invariant_form=F, **kw)

    @classmethod
    def Symplectic(cls, nn, p=DEFAULT_P, **kw):
        gen = []
        assert nn%2 == 0
        n = nn//2
        assert isprime(p)
        F = numpy.zeros((nn, nn), dtype=scalar)
        for i in range(n):
            F[i, n+i] = 1
            F[n+i, i] = p-1
        F = Matrix(F, p)
        I = numpy.identity(nn, dtype=scalar)
        I = Matrix(I, p)
        G = Symplectic([I], order_sp(nn, p), p=p, invariant_form=F, **kw)
        return G

    @classmethod
    def Sp(cls, n, p=DEFAULT_P, **kw):
        gen = []
        assert n%2 == 0

        assert isprime(p)
        F = numpy.zeros((n, n), dtype=scalar)
        for i in range(n//2):
            F[i, n-i-1] = 1
            F[i + n//2, n//2-i-1] = p-1
        F = Matrix(F, p)

        for i in range(1, p):
            vals = set((i**k)%p for k in range(p+1))
            if len(vals)==p-1:
                fgen = i # generates GL(1, p)
                break
        else:
            assert 0
        for i in range(1, p):
            if (i*fgen)%p == 1:
                ifgen = i
                break
        else:
            assert 0

        if n==2:
            G = Sp.SL(2, p, invariant_form=F, **kw)
            return G
        if n==4 and p==2:
            return Sp.Sp_4_2(F, **kw)

        if p==2:
            m = n//2
            A = numpy.zeros((n, n), dtype=scalar)
            B = numpy.zeros((n, n), dtype=scalar)
            for i in range(n):
                A[i, i] = 1
            A[0, m-1] = 1
            A[0, n-1] = 1
            A[m, n-1] = 1
            for i in range(m-1):
                B[i+1, i] = 1
                B[i+m, i+m+1] = 1
            B[0, m] = 1
            B[n-1, m-1] = 1
            gen = [Matrix(A, 2), Matrix(B, 2)]
        else:
            m = n//2
            A = numpy.zeros((n, n), dtype=scalar)
            B = numpy.zeros((n, n), dtype=scalar)
            for i in range(n):
                A[i, i] = 1
            A[0, 0] = fgen
            A[n-1, n-1] = ifgen

            for i in range(m-1):
                B[i+1, i] = 1
                B[m+i, m+i+1] = 1
            B[0, 0] = 1
            B[0, m] = 1
            B[n-2, m-1] = 1
            B[n-1, m-1] = ifgen

            gen = [Matrix(A, p), Matrix(B, p)]

        G = Sp(gen, order_sp(n, p), p=p, invariant_form=F, **kw)
        return G

    # rip this stuff out of sage:

    @classmethod
    def SO_9_2(cls, **kw):
        p = 2
        gens = [
            [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[1,1,0,0,1,0,0,0,1],[0,0,0,0,0,1,0,0,1],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]],
            [[1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,0]]]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, order=47377612800, p=p,
            invariant_bilinear_form = Matrix(
                [[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0]]),
            invariant_quadratic_form = Matrix(
                [[1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0]]))
        return G

    @classmethod
    def SO_7_3(cls, **kw):
        p = 3
        gens = [
[[2,0,0,0,0,0,0],[0,2,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],
[[2,1,2,0,0,0,0],[0,2,0,0,0,0,0],[0,1,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],
[[1,2,2,0,0,0,0],[2,2,1,2,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[0,2,1,0,0,0,0],[2,1,1,1,0,0,0]],
        ]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, order=9170703360, p=p,
            invariant_bilinear_form = Matrix(
[[0,1,0,0,0,0,0],[1,0,0,0,0,0,0],[0,0,2,0,0,0,0],[0,0,0,2,0,0,0],[0,0,0,0,2,0,0],[0,0,0,0,0,2,0],[0,0,0,0,0,0,2]], p),
            invariant_quadratic_form = Matrix(
[[0,1,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]], p))
        return G

    @classmethod
    def SO_7_2(cls, **kw):
        p = 2
        gens = [
            [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[1,1,0,1,0,0,1],
             [0,0,0,0,1,0,1],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]],
            [[1,0,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],
             [0,0,0,0,0,1,0],[0,0,0,0,0,0,1],[0,1,0,0,0,0,0]]]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, order=1451520, p=p,
            invariant_bilinear_form = Matrix(
                [[0,0,0,0,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1],
                 [0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], p),
            invariant_quadratic_form = Matrix(
                [[1,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]], p))
        return G

    @classmethod
    def SO_5_2(cls, **kw):
        p = 2
        gens = [
            [[1,0,0,0,0],[1,0,1,0,1],[1,0,1,1,1],[0,1,0,0,1],[0,1,1,1,1]],
            [[1,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,1,0,0,0]]]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, p=p,
            invariant_bilinear_form = Matrix(
                [[0,0,0,0,0],[0,0,0,1,0],[0,0,0,0,1],[0,1,0,0,0],[0,0,1,0,0]], p),
            invariant_quadratic_form = Matrix(
                [[1,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]], p))
        return G

    @classmethod
    def SO_5_3(cls, **kw):
        p = 3
        gens = [
                [[2,0,0,0,0],[0,2,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
                [[2,1,2,0,0],[0,2,0,0,0],[0,1,1,0,0],[0,0,0,1,0],[0,0,0,0,1]],
                [[1,2,2,0,0],[2,2,1,2,0],[0,0,0,0,1],[0,2,1,0,0],[2,1,1,1,0]],
        ]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, p=p,
            invariant_bilinear_form = Matrix(
                [[0,1,0,0,0],[1,0,0,0,0],[0,0,2,0,0],[0,0,0,2,0],[0,0,0,0,2]], p),
            invariant_quadratic_form = Matrix(
                [[0,1,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1]], p))
        return G



    @classmethod
    def SO_3_5(cls, **kw):
        p = 5
        gens = [
            [[2,0,0],[0,3,0],[0,0,1]], [[3,2,3],[0,2,0],[0,3,1]], [[1,4,4],[4,0,0],[2,0,4]]]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, p=p,
            invariant_bilinear_form = Matrix([[0,1,0],[1,0,0],[0,0,2]], p),
            invariant_quadratic_form = Matrix([[0,1,0],[0,0,0],[0,0,1]], p))
        return G

    @classmethod
    def SO_3_3(cls, **kw):
        p = 3
        gens = [
            [[2,0,0],[0,2,0],[0,0,1]],
            [[2,1,2],[0,2,0],[0,1,1]],
            [[1,2,2],[2,0,0],[2,0,2]],
            ]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, p=p,
            invariant_bilinear_form = Matrix([[0,1,0],[1,0,0],[0,0,2]], 3),
            invariant_quadratic_form = Matrix([[0,1,0],[0,0,0],[0,0,1]], 3))
        return G

    @classmethod
    def SO_3_2(cls, **kw):
        p = 2
        gens = [[[1,0,0],[1,1,1],[0,0,1]],[[1,0,0],[0,0,1],[0,1,0]]]
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, p=p,
            invariant_bilinear_form = Matrix([[0,0,0],[0,0,1],[0,1,0]], p),
            invariant_quadratic_form = Matrix([[1,0,0],[0,0,0],[0,1,0]], p))
        return G

    @classmethod
    def make(cls, gens, invariant_bilinear_form, invariant_quadratic_form, order=None, p=DEFAULT_P, **kw):
        gens = [Matrix(A, p) for A in gens]
        G = Algebraic(gens, order, p=p,
            invariant_bilinear_form = Matrix(invariant_bilinear_form, p),
            invariant_quadratic_form = Matrix(invariant_quadratic_form, p))
        return G


    @classmethod
    def SO_2_2(cls, **kw):
        return cls.make(
            [[[1,0],[0,1]],
             [[0,1],[1,0]]],
            [[0,1],[1,0]],
            [[0,1],[0,0]],
            2 , 2 )

    @classmethod
    def SO_4_2(cls, **kw):
        return cls.make(
            [[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
             [[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]]],
            [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],
            [[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]],
            72 , 2 )

    @classmethod
    def SO_6_2(cls, **kw):
        return cls.make(
            [[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]],
             [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,1],[0,0,0,0,0,1],[0,0,1,0,1,0]],
             [[1,0,0,0,0,0],[0,1,0,0,1,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[1,0,0,0,0,1]],
             [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]],
            [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]],
            [[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,0],[0,0,0,0,0,1],[0,0,0,0,0,0]],
            40320 , 2 )

    @classmethod
    def SO_8_2(cls, **kw):
        return cls.make(
            [[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0]],
             [[0,1,0,0,0,0,0,0],[1,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,0],[0,1,0,1,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,1,0,0,0,1,0,0],[0,0,1,0,1,0,0,0]],
             [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0],[0,0,1,0,1,0,0,0]]],
            [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]],
            [[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0]],
            348364800 , 2 )

    @classmethod
    def SO_10_2(cls, **kw):
        return cls.make(
            [[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0]],
             [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0,0,0],[0,0,1,0,1,0,0,0,0,0]],
             [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]],
            [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0]],
            [[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0]],
            46998591897600 , 2 )

    @classmethod
    def SO_2_3(cls, **kw):
        return cls.make(
            [[[2,0],[0,2]],
             [[2,0],[0,2]],
             [[1,0],[0,1]]],
            [[0,1],[1,0]],
            [[0,1],[0,0]],
            2 , 3 )

    @classmethod
    def SO_4_3(cls, **kw):
        return cls.make(
            [[[0,2,2,2],[0,1,1,2],[1,0,2,0],[1,2,2,0]],
             [[0,2,2,1],[0,2,1,1],[1,1,0,2],[2,0,0,1]],
             [[1,0,0,0],[2,1,0,2],[0,0,1,0],[2,0,0,1]]],
            [[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,2]],
            [[0,1,0,0],[0,0,0,0],[0,0,2,0],[0,0,0,1]],
            576 , 3 )

    @classmethod
    def SO_6_3(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0],[0,2,0,0,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]],
             [[2,1,0,2,0,0],[0,2,0,0,0,0],[0,0,0,0,0,1],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0]],
             [[1,2,0,2,0,0],[2,2,0,1,2,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,2,0,1,0,0],[2,1,0,1,1,0]]],
            [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,2,0,0,0],[0,0,0,2,0,0],[0,0,0,0,2,0],[0,0,0,0,0,2]],
            [[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],
            12130560 , 3 )

    @classmethod
    def SO_8_3(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0],[0,0,0,1,1,0,0,0],[0,0,2,2,1,0,0,0],[0,0,1,2,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
             [[2,1,0,2,0,0,0,0],[0,2,0,0,0,0,0,0],[0,0,0,0,1,1,0,0],[0,1,0,1,0,0,0,0],[0,0,2,0,2,1,0,0],[0,0,1,0,2,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
             [[1,2,0,2,0,0,0,0],[2,2,0,1,2,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,2,0,1,0,0,0,0],[2,1,0,1,1,0,0,0]]],
            [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,2,0,0,0,0],[0,0,0,0,2,0,0,0],[0,0,0,0,0,2,0,0],[0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,2]],
            [[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
            19808719257600 , 3 )

    @classmethod
    def SO_10_3(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[2,1,0,2,0,0,0,0,0,0],[0,2,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,1,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[1,2,0,2,0,0,0,0,0,0],[2,2,0,1,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,2,0,1,0,0,0,0,0,0],[2,1,0,1,1,0,0,0,0,0]]],
            [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,0],[0,0,0,0,0,0,2,0,0,0],[0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,0,0,2]],
            [[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
            None , 3 )

    # these have e == -1:

    @classmethod
    def SO_2_2_1(cls, **kw):
        return cls.make(
            [[[1,1],[1,0]],
             [[1,0],[1,1]]],
            [[0,1],[1,0]],
            [[1,1],[0,1]],
            6 , 2 )

    @classmethod
    def SO_4_2_1(cls, **kw):
        return cls.make(
            [[[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],
             [[0,1,0,0],[1,1,1,0],[0,1,0,1],[0,0,1,0]]],
            [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],
            [[0,1,0,0],[0,0,0,0],[0,0,1,1],[0,0,0,1]],
            120 , 2 )

    @classmethod
    def SO_6_2_1(cls, **kw):
        return cls.make(
            [[[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]],
             [[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,1],[0,0,0,0,0,1],[0,0,1,0,1,1]],
             [[1,0,0,0,0,0],[0,1,0,0,1,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[1,0,0,0,0,1]],
             [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]]],
            [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,0,0,0,1,0]],
            [[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,1,1,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1],[0,0,0,0,0,0]],
            51840 , 2 )

    @classmethod
    def SO_8_2_1(cls, **kw):
        return cls.make(
            [[[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0]],
             [[0,1,0,0,0,0,0,0],[1,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,0],[0,1,0,1,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,1,0,0,0,1,0,0],[0,1,1,0,1,1,0,0]],
             [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0],[0,0,1,0,1,1,0,0]]],
            [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]],
            [[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0]],
            394813440 , 2 )

    @classmethod
    def SO_10_2_1(cls, **kw):
        return cls.make(
            [[[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0]],
             [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,1,0,0,0,0],[0,0,1,0,1,1,0,0,0,0]],
             [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[1,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]],
            [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,0]],
            [[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0]],
            50030759116800 , 2 )

    @classmethod
    def SO_12_2_1(cls, **kw):
        # sage code:
        # G = SO(12, GF(2), -1)
        # for g in G.gens():
        #     print(g)
        #     print()

        data = ("""
        [1 0 0 0 0 0 0 0 0 0 0 0]
        [0 1 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 1 0 0 0 0 0 0 0 0]
        [0 0 1 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 1 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0]
        [0 0 0 0 1 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 0 0 1 0 0 0]
        [0 0 0 0 0 0 0 0 0 1 0 0]""",
        """
        [0 1 0 0 0 0 0 0 0 0 0 0]
        [1 0 0 0 1 0 0 0 0 0 0 0]
        [0 0 1 0 0 0 0 0 0 0 0 0]
        [0 1 0 1 0 1 0 0 0 0 0 0]
        [0 0 0 0 0 0 1 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0]
        [0 0 0 0 0 0 0 0 1 0 0 0]
        [0 0 0 0 0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 0 0 0 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 1]
        [0 1 0 0 0 1 0 0 0 0 0 0]
        [0 1 1 0 1 1 0 0 0 0 0 0]""",
        """
        [0 1 0 0 0 0 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 1 0 0 0 0 0 0 0 0 0]
        [0 0 0 1 0 1 0 0 0 0 0 0]
        [0 0 0 0 0 0 1 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0]
        [0 0 0 0 0 0 0 0 1 0 0 0]
        [0 0 0 0 0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 0 0 0 1 0]
        [0 0 0 0 0 0 0 0 0 0 0 1]
        [0 0 0 0 0 1 0 0 0 0 0 0]
        [0 0 1 0 1 1 0 0 0 0 0 0]""")
        invariant_bilinear_form = parse("""
        [0 1 0 0 0 0 0 0 0 0 0 0]
        [1 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 1 0 0 0 0 0 0 0 0]
        [0 0 1 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 0 0 0 0 0 0]
        [0 0 0 0 1 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0]
        [0 0 0 0 0 0 1 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 0 1 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 0 0 0 0 1 0]
        """)
        invariant_quadratic_form = parse("""
        [0 1 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 1 1 0 0 0 0 0 0 0 0]
        [0 0 0 1 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 1 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 1 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 0]
        [0 0 0 0 0 0 0 0 0 0 0 1]
        [0 0 0 0 0 0 0 0 0 0 0 0]
        """)
        gen = [parse(d) for d in data]
        #for g in gen:
        #    print(g)
        return cls.make(gen, invariant_bilinear_form, invariant_quadratic_form, 
            103231467131240448000, 2)

    @classmethod
    def SO_2_3_1(cls, **kw):
        return cls.make(
            [[[2,2],[2,1]],
             [[1,1],[1,2]],
             [[1,0],[0,1]]],
            [[2,1],[1,1]],
            [[1,1],[0,2]],
            4 , 3 )

    @classmethod
    def SO_4_3_1(cls, **kw):
        return cls.make(
            [[[0,2,0,0],[2,1,0,1],[0,2,0,1],[0,0,1,0]],
             [[1,2,0,2],[1,1,1,1],[1,0,0,1],[1,2,1,2]],
             [[1,0,0,0],[1,1,2,1],[2,0,1,0],[1,0,0,1]]],
            [[0,1,0,0],[1,0,0,0],[0,0,2,0],[0,0,0,2]],
            [[0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],
            720 , 3 )

    @classmethod
    def SO_6_3_1(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0],[0,2,0,0,0,0],[0,0,0,1,1,0],[0,0,2,2,1,0],[0,0,1,2,1,0],[0,0,0,0,0,1]],
             [[2,1,0,2,0,0],[0,2,0,0,0,0],[0,0,0,0,1,1],[0,1,0,1,0,0],[0,0,2,0,2,1],[0,0,1,0,2,1]],
             [[1,2,0,2,0,0],[2,2,0,1,2,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,2,0,1,0,0],[2,1,0,1,1,0]]],
            [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,2,0,0],[0,0,0,0,2,0],[0,0,0,0,0,2]],
            [[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,2,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],
            13063680 , 3 )

    @classmethod
    def SO_8_3_1(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
             [[2,1,0,2,0,0,0,0],[0,2,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,1,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
             [[1,2,0,2,0,0,0,0],[2,2,0,1,2,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,2,0,1,0,0,0,0],[2,1,0,1,1,0,0,0]]],
            [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0],[0,0,0,2,0,0,0,0],[0,0,0,0,2,0,0,0],[0,0,0,0,0,2,0,0],[0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,2]],
            [[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
            20303937239040 , 3 )

    @classmethod
    def SO_10_3_1(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,2,2,1,0,0,0,0,0],[0,0,1,2,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[2,1,0,2,0,0,0,0,0,0],[0,2,0,0,0,0,0,0,0,0],[0,0,0,0,1,1,0,0,0,0],[0,1,0,1,0,0,0,0,0,0],[0,0,2,0,2,1,0,0,0,0],[0,0,1,0,2,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[1,2,0,2,0,0,0,0,0,0],[2,2,0,1,2,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,2,0,1,0,0,0,0,0,0],[2,1,0,1,1,0,0,0,0,0]]],
            [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,0],[0,0,0,0,0,0,2,0,0,0],[0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,0,0,2]],
            [[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
            None , 3 )

    @classmethod
    def SO_2_5(cls, **kw):
        return cls.make(
            [[[2,0],[0,3]],
             [[3,0],[0,2]],
             [[1,0],[0,1]]],
            [[0,1],[1,0]],
            [[0,1],[0,0]],
            4 , 5 )

    @classmethod
    def SO_4_5(cls, **kw):
        return cls.make(
            [[[0,1,0,0],[1,4,4,0],[0,0,0,4],[0,3,4,0]],
             [[0,4,0,0],[4,4,0,3],[0,4,0,4],[0,0,4,0]],
             [[4,0,0,0],[0,4,0,0],[0,0,1,0],[0,0,0,1]]],
            [[0,1,0,0],[1,0,0,0],[0,0,2,0],[0,0,0,2]],
            [[0,1,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]],
            14400 , 5 )

    @classmethod
    def SO_6_5(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0],[0,3,0,0,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,0,1]],
             [[3,2,0,3,0,0],[0,2,0,0,0,0],[0,0,0,0,0,1],[0,3,0,1,0,0],[0,0,1,0,0,0],[0,0,0,0,1,0]],
             [[1,4,0,4,0,0],[4,2,0,1,4,0],[0,0,1,0,0,0],[0,0,0,0,0,1],[0,2,0,1,0,0],[2,3,0,3,1,0]]],
            [[0,1,0,0,0,0],[1,0,0,0,0,0],[0,0,2,0,0,0],[0,0,0,2,0,0],[0,0,0,0,2,0],[0,0,0,0,0,2]],
            [[0,1,0,0,0,0],[0,0,0,0,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]],
            29016000000 , 5 )

    @classmethod
    def SO_8_5(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0,0,0],[0,3,0,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
             [[3,2,0,3,0,0,0,0],[0,2,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,3,0,1,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
             [[1,4,0,4,0,0,0,0],[4,2,0,1,4,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1],[0,2,0,1,0,0,0,0],[2,3,0,3,1,0,0,0]]],
            [[0,1,0,0,0,0,0,0],[1,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0],[0,0,0,2,0,0,0,0],[0,0,0,0,2,0,0,0],[0,0,0,0,0,2,0,0],[0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,2]],
            [[0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]],
            None , 5 )

    @classmethod
    def SO_10_5(cls, **kw):
        return cls.make(
            [[[2,0,0,0,0,0,0,0,0,0],[0,3,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[3,2,0,3,0,0,0,0,0,0],[0,2,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,3,0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
             [[1,4,0,4,0,0,0,0,0,0],[4,2,0,1,4,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1],[0,2,0,1,0,0,0,0,0,0],[2,3,0,3,1,0,0,0,0,0]]],
            [[0,1,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0],[0,0,2,0,0,0,0,0,0,0],[0,0,0,2,0,0,0,0,0,0],[0,0,0,0,2,0,0,0,0,0],[0,0,0,0,0,2,0,0,0,0],[0,0,0,0,0,0,2,0,0,0],[0,0,0,0,0,0,0,2,0,0],[0,0,0,0,0,0,0,0,2,0],[0,0,0,0,0,0,0,0,0,2]],
            [[0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]],
            None , 5 )

    @classmethod
    def SO(cls, n, p=DEFAULT_P, e=None, **kw):
        if e==None or e==1:
            attr = "SO_%d_%d"%(n, p)
        else:
            attr = "SO_%d_%d_1"%(n, p)
        method = getattr(cls, attr)
        if method:
            return method(**kw)
        assert 0, (n, p)


