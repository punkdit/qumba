#!/usr/bin/env python


import numpy

from qumba.argv import argv

EPSILON = 1e-8
scalar = numpy.complex64

class Matrix:
    def __init__(self, A, n):
        A = numpy.array(A, dtype=scalar)
        N = 2**n
        assert A.shape == (N, N)
        self.A = A
        self.shape = A.shape
        self.n = n

    def __eq__(self, other):
        if self.n != other.n:
            return False
        return numpy.allclose(self.A, other.A)

    def __repr__(self):
        return "Matrix(%s)"%(self.shape,)

    def __str__(self):
        n = self.n
        A = self.A
        idxs = list(numpy.ndindex((2,)*n))
        terms = []
        for i,l in enumerate(idxs):
          for j,r in enumerate(idxs):
            a = A[i,j]
            if abs(a) < EPSILON:
                continue
            ls = ''.join(str(li) for li in l)
            rs = ''.join(str(ri) for ri in r)
            term = "|%s><%s|"%(ls, rs)
            if abs(a-1)<EPSILON:
                pass
            elif abs(a+1)<EPSILON:
                term = "-"+term
            else:
                term = str(a)+term
            terms.append(term)
        s = "+".join(terms)
        s = s.replace("+-", "-")
        return s

    def __add__(self, other):
        assert self.n == other.n
        A = self.A+other.A
        return Matrix(A, self.n)

    def __sub__(self, other):
        assert self.n == other.n
        A = self.A-other.A
        return Matrix(A, self.n)

    def __neg__(self):
        return Matrix(-self.A, self.n)

    def __rmul__(self, r):
        return Matrix(r*self.A, self.n)

    def __matmul__(self, other):
        A = numpy.kron(self.A, other.A)
        return Matrix(A, self.n+other.n)

    def __mul__(self, other):
        assert self.n == other.n
        A = numpy.dot(self.A, other.A)
        return Matrix(A, self.n)



def test():

    u0 = Matrix([[1,0],[0,0]], 1)
    u1 = Matrix([[0,0],[0,1]], 1)

    assert str(u0@u1 - u1@u0) == "|01><01|-|10><10|"

    assert u0*u0 == u0
    assert u0*u0 != u1




if __name__ == "__main__":

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



