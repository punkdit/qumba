#!/usr/bin/env python3

"""
Algebraic groups: matrix groups over Z/pZ.

"""


from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

from qumba.qcode import QCode
from qumba.solve import shortstr, dot2, identity2


#scalar = numpy.int64
scalar = numpy.int8 # careful !!

DEFAULT_P = 2 # qubits


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

    def __getitem__(self, idx):
        A = self.A[idx]
        #print("__getitem__", idx, type(A))
        if type(A) is scalar:
            return A
        return Matrix(A, self.p)

    def transpose(self):
        A = self.A
        return Matrix(A.transpose(), self.p)






def test():

    from qumba import construct
    code = construct.get_10_2_3()
    code = code.to_qcode()
    print(code)

    E = code.get_encoder()
    Ei = code.get_decoder()

    M = QCode.trivial(code.m)

    L = QCode.trivial(code.k)
    L = L.apply_CNOT(0, 1) # logical gate

    logop = (M+L).get_encoder()

    A = dot2(E, logop, Ei)
    n = 6
    nn = 2*n
    A = A[:nn,:nn]

    tgt = Matrix(A)

    gen = []
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        A = identity2(nn)
        A[2*jdx+1, 2*idx+1] = 1
        A[2*idx, 2*jdx] = 1
        g = Matrix(A)
        #print(shortstr(g.A), idx, jdx)
        #print()
        gen.append(g)

    found = set(gen)
    bdy = list(found)
    while bdy:
        _bdy = []
        for g in gen:
          for h in bdy:
            gh = g*h
            if gh in found:
                continue
            if gh == tgt:
                print("found")
                return
            _bdy.append(gh)
            found.add(gh)

        bdy = _bdy
        print(len(found), len(bdy))




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




