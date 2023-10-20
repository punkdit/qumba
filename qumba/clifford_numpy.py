#!/usr/bin/env python

"""
build clifford/pauli groups using integer numpy matrices

XXX need rationals
FAIL

"""

from random import choice
from operator import mul, matmul, add
from functools import reduce
#from functools import cache
from functools import lru_cache
cache = lru_cache(maxsize=None)

import numpy

from qumba.solve import zeros2, identity2, shortstr
from qumba.action import mulclose
from qumba.argv import argv


scalar = numpy.int64


class Matrix(object):
    def __init__(self, A, shape=None, name="?"):
        if type(A) == list or type(A) == tuple:
            A = numpy.array(A, dtype=scalar)
        elif isinstance(A, Matrix):
            A = A.A
        elif isinstance(A, numpy.ndarray):
            A = A.astype(scalar) # will always make a copy
        else:
            raise TypeError
        if shape is not None:
            A.shape = shape
        self.A = A
        self.key = self.A.tobytes()
        self._hash = hash(self.key)
        self.shape = A.shape
        #assert name != "?"
        assert name != ""
        if type(name) is str:
            name = name,
        self.name = name

    @classmethod
    def promote(cls, item, name="?"):
        if item is None:
            return None
        if isinstance(item, Matrix):
            return item
        return Matrix(item, name=name)


    def shortstr(self):
        return str(self.A)
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
        assert self.shape == other.shape
        return self.key == other.key

    def __ne__(self, other):
        assert self.shape == other.shape
        return self.key != other.key

    def __lt__(self, other):
        assert self.shape == other.shape
        return self.key < other.key

    def __add__(self, other):
        A = self.A + other.A
        return Matrix(A)

    def __sub__(self, other):
        assert self.shape == other.shape
        A = self.A - other.A
        return Matrix(A)

    def __neg__(self):
        A = -self.A
        return Matrix(A)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.p == other.p
            A = numpy.dot(self.A, other.A)
            return Matrix(A, name=self.name+other.name)
        else:
            return NotImplemented

    def __rmul__(self, r):
        A = r*self.A
        return Matrix(A)

    def transpose(self):
        A = self.A
        name = self.name
        names = []
        for n in reversed(name):
            assert len(n), name
            if n.endswith(".t"):
                names.append(n[:-2])
            elif n[0] == "H":
                names.append(n) # symmetric
            else:
                names.append(n+".t")
        #name = tuple(n[:-2] if n.endswith(".t") else n+".t" for n in reversed(name))
        name = tuple(names)
        return Matrix(A.transpose(), None, name)

    @property
    def t(self):
        return self.transpose()



def test():
    mul = Matrix([[1,0,0,-1],[0,1,1,0]])
    print(mul)




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


