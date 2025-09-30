#!/usr/bin/env python

"""

"""

import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add
from random import random, randint

import numpy

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.smap import SMap
from qumba.qcode import strop, QCode
from qumba import construct 
from qumba.matrix_sage import Matrix
from qumba.clifford import Clifford, w4, r2, ir2
from qumba.action import mulclose
from qumba.util import choose
from qumba import pauli

# See: 
# https://math.stackexchange.com/questions/2294595/cauchy-binet-formula-general-form

def exterior(M, k):
    R = M.ring
    m, n = M.shape
    rows = [[M[idxs, jdxs].determinant() 
        for jdxs in choose(n, k)]
        for idxs in choose(m, k)]
    return Matrix(R, rows)


def npy_exterior(M, k):
    m, n = M.shape
    rows = [[numpy.linalg.det(M[idxs, :][:, jdxs]) 
        for jdxs in choose(n, k)]
        for idxs in choose(m, k)]
    return numpy.array(rows)


def test_plucker(m, n):

    vs = ["x%d%d"%(i,j) for i in range(m) for j in range(n)]
    vs += ["a%d%d"%(i, j) for i in range(n) for j in range(n)]
    #print(vs)
    lookup = {v:idx for (idx,v) in enumerate(vs)}
    R = sage.PolynomialRing(sage.ZZ, vs)
    vs = R.gens()

    x = {(i,j):vs[n*i+j] for i in range(m) for j in range(n)} 
    #print(x)

    a = {(i,j):vs[lookup["a%d%d"%(i,j)]] for i in range(n) for j in range(n)} 
    A = [[a[i,j] for j in range(n)] for i in range(n)]
    A = Matrix(R, A)
    #print("A =")
    #print(A)

    M = [[x[i,j] for j in range(n)] for i in range(m)]
    M = Matrix(R, M)
    M = M.t
    #print("M =")
    #print(M)

    def coords(M):
        P = []
        for idxs in choose(n, m):
            N = M[idxs, :]
            P.append(N.determinant())
        P = Matrix(R, P).t
        return P
    P = coords(M)
    #print("P =")
    #print(P)

    Am = exterior(A, m)
    #print(Am, Am.shape)

    AmP = Am*P
    #print(AmP)

    assert AmP == coords(A*M)

    
def test_projector():

    #for m in range(1, 5):
    #  for n in range(m, m+2):
    #    test_plucker(m, n)

    code = construct.get_422()

    P = code.get_projector()
    assert P*P == P

    for val, vec, d in P.eigenvectors():
        if val == 1:
            break
    else:
        assert 0

    print(vec)


    N, K = vec.shape

    #PK = exterior(P, K)
    #print(PK.shape)

    P = 2*P

    P = numpy.array(P, dtype=int)
    print(P, P.shape)
    found = numpy.where(P)
    found = list(zip(*found))
    print("found:", len(found))

    #P = P[:10, :10]
    PK = npy_exterior(P, K)
    PK = numpy.round(PK)
    PK = PK.astype(int)
    print(PK.shape)
    found = numpy.where(PK)
    found = list(zip(*found))
    #for (i,j) in zip(*found):
    #    print(i,j,PK[i,j])
    print("found:", len(found))


def test_exterior():

    n = 4

    R = sage.ZZ
    I = Matrix.identity(R, n)
    O = Matrix.zeros(R, 2, 2)

    A = I<<O
    print(A)

    print(exterior(A, 2))




if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    fn = eval(name)

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            fn()
        profiler.print()

    else:
        fn()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)



