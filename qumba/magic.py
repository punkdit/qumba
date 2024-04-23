#!/usr/bin/env python

from operator import mul, matmul, add
from functools import reduce


import numpy

from qumba import clifford_ring
clifford_ring.degree = 24
from qumba.clifford import Clifford, r2, K, w4, half, PolynomialRing, Matrix
w24 = K.gen()
one = K.one()
w8 = w24**3
assert w8**2 == w4
w3 = w24**8
w6 = w24**4
r3 = w4 * (w3 - w3.conjugate())
assert r3 ** 2 == 3

from qumba import construct
from qumba.argv import argv


def get_eigenval(M, v):
    "return eigenval if v is eigvec of M, None otherwise"
    r = (v.d*v)[0][0]
    if r==0:
        return 0
    val = (v.d * M * v)[0][0]
    val = val / r
    if M*v == val*v:
        return val
    return None


def test():
    # See these slides:
    # https://www.iqst.ca/events/csqic05/talks/nathan%20b.pdf
    # And paper by Bravyi & Kitaev:
    # https://arxiv.org/abs/quant-ph/0403025

    c = Clifford(1)
    X, Y, Z, S, H, I = c.X(), c.Y(), c.Z(), c.S(), c.H(), c.I
    M = S*H # page 13 of slides, missing a phase...
    assert M.order() == 24

    M = w8*M # this is the correct Bravyi-Kitaev T gate
    assert M.order() == 6

    evs = M.eigenvectors()
    assert [ev[0] for ev in evs] == [w6, w6**5]

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
    #rho = Matrix(R, rho)
    #m = e*rho

    rho = (1-e) * TT0 + e * TT1
    assert rho(half) == half*(TT0 + TT1)

    code = construct.get_513()
    P = code.get_projector()

    assert P*P == P

    n = code.n
    c = Clifford(n)

    rho5 = reduce(matmul, [rho]*n)

    M5 = reduce(matmul, [M]*n)
    assert M5*P == P*M5 # logical gate

    found = []
    evs = M5.eigenvectors()
    for val, vec, dim in evs:
        assert val == get_eigenval(M5, vec)
        v = P*vec
        val = get_eigenval(M5, v)
        if val == 0:
            continue
        #print(val)
        found.append(v)
    T1enc, T0enc = found # the encoded magic states

    assert (T0enc.d * T1enc)[0,0] == 0

    r0 = one / (T0enc.d * T0enc)[0,0]
    r1 = one / (T1enc.d * T1enc)[0,0]

    Q = r0*T0enc*T0enc.d + r1*T1enc*T1enc.d
    
    assert (Q*Q == Q)
    assert (Q*P == P*Q)
    assert (Q == P)

    # page 21 in slides
    a = r0*(T0enc.d * rho5 * T0enc)[0,0]
    b = r1*(T1enc.d * rho5 * T1enc)[0,0]
    e1 = a / (a+b)

    # page 22 in slides
    assert e1(0.15) < 0.15 # below threshold
    assert e1(0.20) > 0.20 # above threshold

    # Bravyi-Kitaev page 6:
    epsilon_0 = (1/2)*(1 - (3/7)**0.5)
    assert abs(e1(epsilon_0) - epsilon_0) < 1e-8



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














