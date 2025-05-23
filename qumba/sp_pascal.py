#!/usr/bin/env python3

"""
Pascal triangle for Sp(F_2)

copied from bruhat

Warning: the symplectic structure is the uturn not ziporder !

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
from collections import namedtuple

import numpy

from qumba.argv import argv
from qumba.lin import parse, shortstr, rank, shortstrx, zeros2, dot2
from qumba.util import cross, allperms
from qumba.smap import SMap

"""

These are the cardinalities:

    k=0  k=1   k=2   k=3   k=4
n=0   1                            
n=1   1    3                           
n=2   1   15    15                           
n=3   1   63   315   135                           
n=4   1  255  5355 11475  2295                           

"""


@cache
def B(n, k, q=2):
    assert 0<=n
    assert 0<=k
    if k>n:
        return 0
    if k==0:
        return 1
    return (1+q**(2*n-k)) * B(n-1,k-1) + (q**k) * B(n-1,k)


@cache
def omega(n):
    nn = 2*n
    F = zeros2(nn,nn)
    for i in range(nn):
        F[i, nn-i-1] = 1
    return F


@cache
def getbits(k):
    return list(numpy.ndindex((2,)*k))


def fill(A, idxs):
    k = len(idxs)
    for bits in getbits(k):
        B = A.copy()
        for idx, bit in zip(idxs, bits):
            B[idx] = bit
        yield B


class RowReduced(object):
    "_Row-echelon reduced matrix"
    def __init__(self, pivots, A, left=None, right=None):
        assert len(pivots) == len(A)
        for a, pivot in zip(A, pivots):
            assert a[pivot] == 1
            # etc
        self.pivots = pivots
        self.A = A
        self.left = left
        self.right = right
    def __getitem__(self, i):
        return [self.pivots, self.A][i]
    def __str__(self):
        return shortstr(self.A)


def row_reduced(pivots, A, parent=None):
    assert len(pivots) == len(A)
    prev = None
    for a, pivot in zip(A, pivots):
        assert a[pivot] == 1
        assert prev is None or pivot > prev
        prev = pivot
        # etc.
    return (pivots, A)

#RowReduced = namedtuple("pivots", "A") 
#RowReduced = lambda *args, parent=None : args # fastest


@cache
def grassmannian(n, k):
    return list(i_grassmannian(n, k))

def i_grassmannian(n, k):
    assert 0<=n
    assert 0<=k
    nn = 2*n
    if k>n:
        return

    if k==0:
        yield RowReduced([], zeros2(0, nn))
        return

    F = omega(n)

    # left parent
    src = grassmannian(n-1, k-1)
    for left in src:
        pivs = [i+1 for i in left.pivots]

        B = zeros2(k, nn)
        B[:k-1, 1:nn-1] = left.A
        B[k-1, nn-1] = 1 # choose the new guy's twin
        item = RowReduced(pivs+[nn-1], B, left, None)
        yield item

        B = zeros2(k, nn)
        B[1:k, 1:nn-1] = left.A
        B[0,0] = 1 # choose the new guy
        idxs = [(i,nn-1) for i in range(k)]
        for j in range(1, nn-1):
            if j not in pivs:
                idxs.append((0, j))
        for C in fill(B, idxs):
            if dot2(C, F, C.transpose()).sum() == 0:
                item = RowReduced([0]+pivs, C, left, None)
                yield item

    # right parent
    src = grassmannian(n-1, k)
    for right in src:
        pivs = [i+1 for i in right.pivots]
        
        B = zeros2(k, nn)
        B[:, 1:nn-1] = right.A
        idxs = [(i,nn-1) for i in range(k)]
        for C in fill(B, idxs):
            #assert dot2(C, F, C.transpose()).sum() == 0 # yes, always
            item = RowReduced(pivs, C, None, right)
            yield item

    #return items


def find_sy():
    from qumba.transversal import Solver, UMatrix

    n = 2
    F4 = omega(n)
    F6 = omega(n+1)

    solve = Solver()
    E = UMatrix.unknown(2*n, 2*(n+1))
    print(E)

    u = UMatrix.unknown(1, 2*n)
    print(u * E)
    

def show():
    for n in range(1, 6):
      for m in range(n+1):
        print(len(grassmannian(n,m)), end=" ", flush=True)
      print()


def test():

    print("test()")

    assert B(0,0) == 1
    assert B(2,1) == 15
    assert B(3,2) == 315
    assert B(4,2) == 5355
    assert B(5,3) == 782595
    print( len( grassmannian(4, 3) ))
    print( len( grassmannian(5, 4) ))

    F = omega(3)

    A = zeros2(3,3)
    A[0,0] = 1
    #for A in fill(A, [(0,1), (2,2), (2,0)]):
    #    print(shortstr(A).replace('\n',''))
    assert len(list(fill(A, [(0,1), (2,2), (2,0)]))) == 8

    count = 0
    for (pivots, A) in grassmannian(3, 2):
        #print(shortstr(A), pivots)
        count += 1
    assert count == 315

    items = grassmannian(5, 4)
    assert len(items) == B(5, 4)

    return

    print("[[6,0,?]]:", end=" ", flush=True)
    count = 0
    for item in i_grassmannian(6, 6):
        count += 1
    print(count)

    # takes about 4 hours to run this
    count = 0
    for item in i_grassmannian(6, 5):
        count += 1
    print("[[6,1,?]]:", count) # 103378275

    #items = grassmannian(6, 5)
    #print("[[6,1,?]]:", len(items))

    #items = grassmannian(7, 6)
    #print("[[7,1,?]]:", len(items))


def dump():
    for n in range(12):
     print(2*n+1, end=":")
     for k in range(n+1):
        print(B(n,k), end=" ")
     print()

def main():
    print("main()")
    items = grassmannian(5, 4)
    assert len(items) == B(5, 4)


if __name__ == "__main__":
    fn = argv.next() or "test"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))






