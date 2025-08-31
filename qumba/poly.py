#!/usr/bin/env python

from functools import reduce
from operator import matmul, add

from qumba.argv import argv

from qumba.qcode import strop, QCode
from qumba import construct 

from qumba.matrix_sage import Matrix
from qumba import clifford
from qumba.clifford import Clifford, w4

from sage import all_cmdline as sage


def test():

    base = clifford.K
    one = base.one()
    half = one/2

    K = sage.PolynomialRing(base, list("xyzw"))
    x,y,z,w = K.gens()

    c = Clifford(1)
    I = c.I
    X = c.X()
    Y = c.Y()
    Z = c.Z()

    rho = half*(w*I + x*X + y*Y + z*Z)
    assert rho.trace() == w

    code = construct.get_513()
    print(code)
    H = strop(code.H, "I")

    n = code.n
    space = Clifford(n)
    P = space.I
    for h in H.split():
        s = space.get_pauli(h)
        P = half*(space.I+s)*P
    assert P*P == P

    if 0:
        assert P.conjugate()==P
        P = P.change_ring(sage.QQ)

    Pd = P.d

    rho = reduce(matmul, [rho]*n)
    rho = P*rho*Pd
    div = rho.trace()
    print("div =", div)

    L = strop(code.L, "I").split()
    LX = space.get_pauli(L[0])
    LZ = space.get_pauli(L[1])
    LY = w4*LX*LZ

    x = (rho*LX).trace()
    y = (rho*LY).trace()
    z = (rho*LZ).trace()

    print("x =", x)
    print("y =", y)
    print("z =", z)

    N = 2**n
    x = (x*N)/(div*N)
    #print(x.parent())


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



