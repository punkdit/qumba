#!/usr/bin/env python3

"""

"""

from random import random, randint, shuffle
from functools import reduce
from operator import mul

import numpy

from qumba.lin import parse, shortstr, shortstrx, span, int_scalar
from qumba.lin import row_reduce, dot2
from qumba.matrix import Matrix
from qumba.argv import argv
from qumba.util import choose, cross, all_perms


def zeros(m, n):
    A = numpy.zeros((m, n), dtype=int_scalar)
    return A

def identity(m):
    A = numpy.identity(m, dtype=int_scalar)
    return A


def get_cell(row, col, p=2):
    """
        return all matrices in bruhat cell at (row, col)
        These have shape (col, col+row).
    """

    if col == 0:
        yield zeros(0, row)
        return

    if row == 0:
        yield identity(col)
        return

    # recursive steps:
    m, n = col, col+row
    for left in get_cell(row, col-1, p):
        A = zeros(m, n)
        A[:m-1, :n-1] = left
        A[m-1, n-1] = 1
        yield A

    els = list(range(p))
    vecs = list(cross((els,)*m))
    for right in get_cell(row-1, col, p):
        for v in vecs:
            A = zeros(m, n)
            A[:, :n-1] = right
            A[:, n-1] = v
            yield A


def all_codes(m, n, q=2):
    """
        All full-rank generator matrices of shape (m, n)
    """
    assert m<=n
    col = m
    row = n-m
    for A in get_cell(row, col, q):
        yield Matrix(A, q)


def main():

    q = 2
    m = argv.get("m", 2)
    n = argv.get("n", 4)
    for n in [0,1,2,3,4,5]:
      for m in range(n+1):
        cells = list(all_codes(m, n))
        print(len(cells), end=' ')
      print()




if __name__ == "__main__":

    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))






