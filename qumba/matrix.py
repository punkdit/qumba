#!/usr/bin/env python3

"""
matrix groups over Z/pZ.

"""


from random import shuffle, choice
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

from qumba.solve import (shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2,
    kernel, span)
from qumba.solve import int_scalar as scalar
from qumba.action import mulclose


DEFAULT_P = 2 # qubits


def flatten(H):
    if H is not None and len(H.shape)==3:
        H = H.view()
        m, n, _ = H.shape
        H.shape = m, 2*n
    return H


def complement(H):
    H = flatten(H)
    H = row_reduce(H)
    m, nn = H.shape
    #print(shortstr(H))
    pivots = []
    row = col = 0
    while row < m:
        while col < nn and H[row, col] == 0:
            #print(row, col, H[row, col])
            pivots.append(col)
            col += 1
        row += 1
        col += 1
    while col < nn:
        pivots.append(col)
        col += 1
    W = zeros2(len(pivots), nn)
    for i, ii in enumerate(pivots):
        W[i, ii] = 1
    #print()
    return W




class Matrix(object):
    def __init__(self, A, p=DEFAULT_P, shape=None, name="?"):
        if type(A) == list or type(A) == tuple:
            A = numpy.array(A, dtype=scalar)
        elif isinstance(A, Matrix):
            A, p = A.A, A.p
        elif isinstance(A, numpy.ndarray):
            A = A.astype(scalar) # will always make a copy
        else:
            raise TypeError
        A = flatten(A)
        if shape is not None:
            A.shape = shape
        self.A = A
        assert int(p) == p
        assert p>=0
        self.p = p
        if p>0:
            self.A %= p
        self.key = (self.p, self.A.tobytes())
        self._hash = hash(self.key)
        self.shape = A.shape
        assert name != "?"
        if type(name) is str:
            name = name,
        self.name = name

    @classmethod
    def promote(cls, item, p=DEFAULT_P, name=""):
        if item is None:
            return None
        if isinstance(item, Matrix):
            return item
        return Matrix(item, p, name=name)

    @classmethod
    def perm(cls, items, p=DEFAULT_P, name=""):
        n = len(items)
        A = numpy.zeros((n, n), dtype=scalar)
        for i, ii in enumerate(items):
            A[ii, i] = 1
        return Matrix(A, p, name=name)

    @classmethod
    def identity(cls, n, p=DEFAULT_P):
        A = numpy.identity(n, dtype=scalar)
        return Matrix(A, p, name="I")

    def shortstr(self):
        s = shortstr(self.A)
        lines = s.split()
        lines = [" ["+line+"]" for line in lines]
        lines[0] = "["+lines[0][1:]
        lines[-1] = lines[-1] + "]"
        return '\n'.join(lines)
    __str__ = shortstr

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
        assert self.shape == other.shape
        return self.key == other.key

    def __ne__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        return self.key != other.key

    def __lt__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        return self.key < other.key

    def __add__(self, other):
        assert self.p == other.p
        A = self.A + other.A
        return Matrix(A, self.p)

    def __sub__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        A = self.A - other.A
        return Matrix(A, self.p)

    def __neg__(self):
        A = -self.A
        return Matrix(A, self.p)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.p == other.p
            A = numpy.dot(self.A, other.A)
            return Matrix(A, self.p, name=self.name+other.name)
        else:
            return NotImplemented

    def __rmul__(self, r):
        A = r*self.A
        return Matrix(A, self.p)

    def direct_sum(self, other):
        "direct_sum"
        A = direct_sum(self.A, other.A)
        return Matrix(A, self.p)
    #__add__ = direct_sum # ??
    #__lshift__ = direct_sum # ???

    def __getitem__(self, idx):
        A = self.A[idx]
        #print("__getitem__", idx, type(A))
        if type(A) is scalar:
            return A
        return Matrix(A, self.p)

    def transpose(self):
        A = self.A
        name = self.name + ("t",)
        return Matrix(A.transpose(), self.p, None, name)

    @property
    def t(self):
        return self.transpose()

    def sum(self):
        A = self.A
        return A.astype(numpy.int64).sum()

    def kernel(self):
        K = kernel(self.A)
        K = Matrix(K, self.p)
        return K

    def where(self):
        return list(zip(*numpy.where(self.A))) # list ?

    def concatenate(self, *others):
        A = numpy.concatenate((self.A,)+tuple(other.A for other in others))
        return Matrix(A, self.p)

    def max(self):
        return self.A.max()

    def min(self):
        return self.A.min()

    def copy(self):
        return Matrix(self.A, self.p)

    def rowspan(self):
        m, n = self.A.shape
        for u in span(self.A):
            u.shape = 1, n
            u = Matrix(u)
            yield u


def test():
    I = Matrix.identity(5)
    assert I*I == I



if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

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


