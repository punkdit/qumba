#!/usr/bin/env python

"""
Matrix object that uses numpy array's with object dtype populated with sage scalars.
"""

from random import choice
from operator import mul, matmul, add
from functools import reduce
#from functools import cache
from functools import lru_cache
cache = lru_cache(maxsize=None)

import numpy

#from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
#    PolynomialRing)
from sage import all_cmdline 

#from qumba.argv import argv
#from qumba.clifford_ring import degree
#K = CyclotomicField(degree)
#root = K.gen() # primitive eighth root of unity
#w8 = root ** (degree // 8)
#half = K.one()/2
#w4 = w8**2
#r2 = w8 + w8**7
#ir2 = r2 / 2
#assert r2**2 == 2
#
#def simplify_latex(self):
#    M = self.M
#    m, n = self.shape
#    idxs = [(i,j) for i in range(m) for j in range(n)]
#    for idx in idxs:
#        if M[idx] != 0:
#            break
#    else:
#        assert 0
#    scale = M[idx]
#    if scale != 1 and scale != -1:
#        M = (1/scale) * M
#        s = {
#            r2 : r"\sqrt{2}",
#            1/r2 : r"\frac{1}{\sqrt{2}}",
#            #2/r2 : r"\frac{2}{\sqrt{2}}",
#            2/r2 : r"\sqrt{2}",
#            #r2/2 : r"\sqrt{2}/2",
#        }.get(scale, latex(scale))
#        if "+" in s:
#            s = "("+s+")"
#        s = "%s %s"%(s, latex(M))
#    else:
#        s = latex(M)
#    s = s.replace(r"\zeta_{8}^{2}", "i")
#    return s


class Matrix(object):
    def __init__(self, ring, rows, name=()):
        #M = all_cmdline.Matrix(ring, rows)
        #M.set_immutable()
        M = numpy.array(rows, dtype=object)
        one = ring.one()
        for idx in numpy.ndindex(M.shape):
            M[idx] = one*M[idx]
        self.M = M
        self.ring = ring
        self.shape = M.shape
        if type(name) is str:
            name = (name,)
        self.name = name

    def __eq__(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        assert self.shape == other.shape
        return numpy.alltrue(self.M == other.M)

    def __hash__(self):
        items = tuple(self.M[i] for i in numpy.ndindex(self.shape))
        return hash(items)
        #return hash(self.M)

    def __str__(self):
        lines = str(self.M).split("\n")
        lines[0] = "[" + lines[0]
        lines[-1] = lines[-1] + "]"
        lines[1:] = [" "+l for l in lines[1:]]
        return '\n'.join(lines)
    __repr__ = __str__

    def __len__(self):
        return self.shape[0]

    def __slowmul__(self, other):
        assert isinstance(other, Matrix), type(other)
        assert self.ring == other.ring
        assert self.shape[1] == other.shape[0], (
            "cant multiply %sx%s by %sx%s"%(self.shape + other.shape))
        #print("__mul__: numpy.dot", self.M.shape, other.M.shape)
        M = numpy.dot(self.M, other.M)
        #print("__mul__: numpy.dot finished")
        name = self.name + other.name
        return Matrix(self.ring, M, name)

    def __veryslowmul__(self, other):
        assert isinstance(other, Matrix), type(other)
        assert self.ring == other.ring
        assert self.shape[1] == other.shape[0], (
            "cant multiply %sx%s by %sx%s"%(self.shape + other.shape))
        #print("__mul__: numpy.dot", self.M.shape, other.M.shape)
        m, n = self.shape
        _, l = other.shape
        M = self.M
        N = other.M
        zero = self.ring.zero()
        MN = [[
            sum([M[i,j]*N[j,k] for j in range(n)], zero)
            for k in range(l)] 
            for i in range(m)]
        #print("__mul__: numpy.dot finished")
        name = self.name + other.name
        return Matrix(self.ring, MN, name)

    def __fastmul__(self, other):
        assert isinstance(other, Matrix), type(other)
        assert self.ring == other.ring
        assert self.shape[1] == other.shape[0], (
            "cant multiply %sx%s by %sx%s"%(self.shape + other.shape))
        #print("__mul__: numpy.dot", self.M.shape, other.M.shape)
        m, n = self.shape
        _, l = other.shape
        M = self.M
        N = other.M
        zero = self.ring.zero()
        #cache = {}
        #def mul(a, b):
        #    key = (a,b)
        #    ab = cache.get(key)
        #    if ab is None:
        #        ab = a*b
        #        cache[key] = ab
        #    return ab
        MN = [[
            sum([M[i,j]*N[j,k] for j in range(n) if M[i,j]!=zero and N[j,k]!=zero], zero)
            for k in range(l)] 
            for i in range(m)]
        #print("__mul__: numpy.dot finished")
        name = self.name + other.name
        return Matrix(self.ring, MN, name)

    #__mul__ = __slowmul__
    __mul__ = __fastmul__

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
        #M = self.M.tensor_product(other.M)
        M = numpy.kron(self.M, other.M)
        return Matrix(self.ring, M)
    tensor_product = __matmul__

    def direct_sum(self, other):
        assert isinstance(other, Matrix)
        assert self.ring == other.ring
        #M = self.M.direct_sum(other.M)
        #M = block_diagonal_matrix(self.M, other.M)
        M = numpy.zeros((self.shape[0]+other.shape[0], self.shape[1]+other.shape[1]), 
            dtype=object)
        M[:self.shape[0], :self.shape[1]] = self.M
        M[self.shape[0]:, self.shape[1]:] = other.M
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
        one = ring.one()
        zero = ring.zero()
        rows = []
        for i in range(n):
            row = [zero]*n
            row[i] = one
            rows.append(row)
        return Matrix(ring, rows)

    def order(self):
        I = self.identity(self.ring, len(self))
        count = 1
        A = self
        while A != I:
            count += 1
            A = self*A
        return count

    def sage_matrix(self):
        M = all_cmdline.Matrix(self.ring, self.M)
        return M

    def inverse(self):
        #M = self.M.inverse()
        M = self.sage_matrix()
        M = M.inverse()
        return Matrix(self.ring, M)

    def transpose(self):
        M = self.M.transpose()
        return Matrix(self.ring, M)

    def trace(self):
        return self.M.trace()

    @property
    def t(self):
        return self.transpose()

    def dagger(self):
        #M = self.M.conjugate_transpose()
        M = self.M.transpose()
        m, n = M.shape
        M = [[M[i,j].conjugate() for j in range(n)] for i in range(m)]
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
        M = self.sage_matrix()
        return M.rank()

    def eigenvectors(self):
        M = self.sage_matrix()
        evs = M.eigenvectors_right()
        vecs = []
        for val,vec,dim in evs:
            #print(val, dim)
            #print(type(vec[0]))
            vec = all_cmdline.Matrix(self.ring, vec[0])
            vec = vec.transpose() 
            vec = Matrix(self.ring, vec)
            vecs.append((val, vec, dim))
        #print(type(self.M))
        return vecs


