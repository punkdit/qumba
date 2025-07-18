#!/usr/bin/env python

"""

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


def unify(R, S):
    if R.coerce_map_from(S) is not None:
        return R
    if S.coerce_map_from(R) is not None:
        return S
    assert 0



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
        #assert self.ring == other.ring
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

    def gapstr(self):
        rows = []
        #for idx in numpy.ndindex(self.shape):
        m, n = self.shape
        for i in range(m):
          row = []
          for j in range(n):
            r = self.M[i,j] 
            r = str(r)
            r = r.replace(" ", "")
            r = r.replace("zeta4", "E(4)")
            r = r.replace("zeta8", "E(8)")
            assert "zeta" not in r, r
            row.append(r)
          row = "[" + ', '.join(row) + "]"
          rows.append(row)
        return "[" + ", ".join(rows) + "]"

    def __len__(self):
        return self.shape[0]

    def __mul__(self, other):
        assert isinstance(other, Matrix), type(other)
        assert self.shape[1] == other.shape[0], (
            "cant multiply %sx%s by %sx%s"%(self.shape + other.shape))
        ring = unify(self.ring, other.ring)
        M = self.M * other.M
        name = self.name + other.name
        return Matrix(ring, M, name)

    def __add__(self, other):
        assert isinstance(other, Matrix)
        ring = unify(self.ring, other.ring)
        M = self.M + other.M
        return Matrix(ring, M)

    def __sub__(self, other):
        assert isinstance(other, Matrix)
        ring = unify(self.ring, other.ring)
        M = self.M - other.M
        return Matrix(ring, M)

    def __neg__(self):
        M = -self.M
        return Matrix(self.ring, M)

    def __pow__(self, n):
       assert n>=0
       if n==0:
           return Matrix.identity(self.ring, self.shape[0])
       return reduce(mul, [self]*n)

    def __rmul__(self, r):
        if isinstance(r, int):
            ring = self.ring
        else:
            ring = unify(r.parent(), self.ring)
        M = r*self.M
        return Matrix(ring, M)

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

    def stack(self, other):
        "concatenate rows of self & other"
        #assert self.ring == other.ring
        ring = unify(self.ring, other.ring)
        return Matrix(ring, self.M.stack(other.M))

    def augment(self, other):
        "concatenate cols of self & other"
        #assert self.ring == other.ring
        ring = unify(self.ring, other.ring)
        return Matrix(ring, self.M.augment(other.M))

    def __getitem__(self, idx):
        if type(idx) is int:
            return self.M[idx]
        elif type(idx) is tuple and len(idx)==2 and type(idx[0])==type(idx[1])==int:
            return self.M[idx]
        M = self.M[idx]
        return Matrix(self.ring, M)

    def __call__(self, val):
        M = self.M(val)
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

    def order(self):
        I = self.identity(self.ring, len(self))
        count = 1
        A = self
        while A != I:
            count += 1
            A = self*A
        return count

    def inverse(self):
        M = self.M.inverse()
        return Matrix(self.ring, M)
    __invert__ = inverse

    def transpose(self):
        M = self.M.transpose()
        return Matrix(self.ring, M)

    def trace(self):
        return self.M.trace()

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

    def determinant(self):
        return self.M.determinant()

    def eigenvectors(self, ring=None):
        if ring is None:
            ring = self.ring
        evs = self.M.eigenvectors_right()
        vecs = []
        for val,vec,dim in evs:
            #print(val, dim)
            #print(vec[0])
            #print(type(vec[0]))
            vec = all_cmdline.Matrix(ring, vec[0])
            vec = vec.transpose() 
            vec = Matrix(ring, vec)
            vecs.append((val, vec, dim))
        #print(type(self.M))
        return vecs


    def cokernel(self):
        K = all_cmdline.kernel(self.M)
        B = K.basis()
        #print(K, type(K))
        M = Matrix(self.ring, B)
        # XXX fix empty M shape
        return M

    def kernel(self):
        K = self.t.cokernel().t
        return K

    def charpoly(self):
        return self.M.characteristic_polynomial()





def test():
    "Orthogonal matrixes over GF(2)[x] / x**2 "
    from util import cross
    from random import shuffle
    from sage.all_cmdline import GF, PolynomialRing
    K = GF(2)
    R = PolynomialRing(K, "x")
    x = R.gens()[0]
    S = R.quo(x**2)
    print(S)

    x = S.gens()[0]
    zero = x**2
    one = (x+1)**2
    print("one:", one)
    print("x:", x)
    #return

    els = [zero, one, one+x, x]
    found = []
    n = argv.get("n", 2)
    I = Matrix.identity(S, n)
    rows = list(cross([els]*n))
    shuffle(rows)
    #for cols in cross( [rows]*n ):
    print("search:")
    while 1:
        cols = [choice(rows) for i in range(n)]
        M = Matrix(S, cols)
        if M*M.t == I:
            found.append(M)
            print('/', end='', flush=True)
            if len(found) > 1:
                G = mulclose(found, verbose=True)
                print("|G|=", len(G))
                #del G # free mem
                #break
    print()
    #for g in G:
    #    print(g)


if __name__ == "__main__":

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




