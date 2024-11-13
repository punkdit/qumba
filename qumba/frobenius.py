#!/usr/bin/env python

"""
Build _monoids/_comonoids/_frobenius etc. over F_2
See: Carboni 1991

see : qumba/algebras.py
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
    n = 2 # qubits

    I = UMatrix(numpy.identity(n, dtype=object))
    one = Matrix([[1]])
    swap = get_swap(n)
    unknown = UMatrix.unknown

    # black/white phase=0 spiders
    bb_b = unknown(n*n, n)
    b_bb = unknown(n, n*n)
    ww_w = unknown(n*n, n)
    w_ww = unknown(n, n*n)

    # phase=1
    _b1 = unknown(1, n)
    b1_ = unknown(n, 1)
    _w1 = unknown(1, n)
    w1_ = unknown(n, 1)

    h = UMatrix([[0,1],[1,0]])

    # phase=1
    w1 = UMatrix([[1,1],[0,1]])
    b1 = UMatrix([[1,1],[1,0]])

    #assert h*h == I
    #assert w1*h == b1
    #assert b1*h == w1

    # phase=0
    _b = _b1 * b1
    b_ = b1 * b1_ 
    _w = _w1 * w1
    w_ = w1 * w1_ 

    solver = z3.Solver()
    Add = solver.add

    Add( _b * w1_ == one )
    Add( _b * w_ == one )
    Add( _b * b_ == one )

    # comm
    Add(swap*bb_b == bb_b)
    Add(b_bb*swap == b_bb)
    Add(swap*ww_w == ww_w)
    Add(w_ww*swap == w_ww)

    # unit
    print(I.direct_sum(_b)*bb_b)
    Add(I.direct_sum(_b) * bb_b == I)
    
    # bialgebra
    lhs = b_bb.direct_sum(b_bb) * (I.direct_sum(swap).direct_sum(I)) * ww_w.direct_sum(ww_w)
    rhs = ww_w * b_bb

    Add(lhs==rhs)

    result = solver.check()
    #assert result == z3.sat, result
    print(result)



def main_unsat():

    n = 3 # also works for n=4
    n = argv.get("n", 3)

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
    print(result)

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




