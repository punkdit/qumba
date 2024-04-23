#!/usr/bin/env python

import numpy

from qumba import clifford_ring
clifford_ring.degree = 24
from qumba.clifford import Clifford, r2, K, w4, half, PolynomialRing
w24 = K.gen()
one = K.one()
w3 = w24**8
r3 = w4 * (w3 - w3.conjugate())
assert r3 ** 2 == 3

from qumba import construct
from qumba.argv import argv



def test():
    # See: 
    # https://www.iqst.ca/events/csqic05/talks/nathan%20b.pdf

    c = Clifford(1)
    X, Y, Z, S, H, I = c.X(), c.Y(), c.Z(), c.S(), c.H(), c.I
    M = S*H
    assert M.order() == 24

    evs = M.eigenvectors()
    #for val, vec, dim in evs:
    #    print(vec, val)

    T1, T0 = [ev[1] for ev in evs]

    norm0 = (T0.d*T0)[0][0]
    norm1 = (T1.d*T1)[0][0]

    TT0 = (one/norm0) * T0 * T0.d
    TT1 = (one/norm1) * T1 * T1.d
    assert TT0.trace() == 1
    assert TT1.trace() == 1

    rho = (half * (I + (one/r3)*(X + Y + Z)))
    assert rho.trace() == 1
    assert rho == TT0
    rho = (half * (I - (one/r3)*(X + Y + Z)))
    assert rho.trace() == 1
    assert rho == TT1

    R = PolynomialRing(K, "e")
    e = R.gen()
    m = numpy.array(rho)
    print(m, m.shape, m.dtype)
    m = e*m
    print(m, m.shape, m.dtype)
    print(type(m[0,0]))

    return

    code = construct.get_513()
    P = code.get_projector()

    assert P*P == P

    n = code.n
    c = Clifford(n)

    



if __name__ == "__main__":

    from time import time
    start_time = time()

    print(argv)

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        from random import seed
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














