#!/usr/bin/env python

"""
Build _monoids/_comonoids/_frobenius etc. over F_2
See: Carboni 1991
"""

import numpy

from qumba.matrix import Matrix
from qumba.transversal import UMatrix, z3
from qumba.argv import argv


def get_swap(n):
    A = numpy.empty((n, n, n, n), dtype=object)
    A[:] = 0
    for (i,j) in numpy.ndindex((n,n)):
        A[i,j,j,i] = 1
    A.shape = (n*n, n*n)
    A = UMatrix(A)
    return A


def main():

    n = 3 # also works for n=4
    n = 5

    I = UMatrix(numpy.identity(n, dtype=object))
    swap = get_swap(n)

    # (out,in)
    comul = UMatrix.unknown(n*n, n)
    mul = UMatrix.unknown(n, n*n)
    _mul = UMatrix.unknown(n, n*n)

    solver = z3.Solver()
    Add = solver.add

    Add(mul * swap == mul)
    Add(_mul * swap == _mul)
    Add(swap * comul == comul)

    # _frobenius
    op = comul * mul
    Add( (mul @ I) * (I @ comul) == op )
    Add( (I@mul) * (comul@I) == op )

    op = comul * _mul
    Add( (_mul @ I) * (I @ comul) == op )
    Add( (I@_mul) * (comul@I) == op )

    # special
    Add(mul*comul == I)
    Add(_mul*comul == I)

    result = solver.check()
    assert result == z3.sat, result

    # unique ?
    Add(mul != _mul)

    result = solver.check()
    assert result == z3.unsat, result

    #model = solver.model()
    #comul = comul.get_interp(model)
    #print(comul)





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




