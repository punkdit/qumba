#!/usr/bin/env python

"""

"""

from operator import mul, matmul, add
from functools import reduce
from functools import cache

from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
    PolynomialRing)
from sage import all_cmdline as sage

from qumba.action import mulclose, mulclose_names, mulclose_find
from qumba.matrix_sage import Matrix
from qumba.argv import argv


def test():

    q = argv.get("q", 3)
    assert q==3, "um... ??"

    K = CyclotomicField(q*4)
    print(K)

    z, = K.gens()
    w3 = z**4
    J = z**q
    assert J**2 == -1

    r3 = 2*z - z**3
    assert r3**2 == 3

    one = K.one()
    for i in range(1,q):
        assert w3**i != one
    assert w3**q == one
    half = one//2
    ir3 = one/r3

    I = Matrix.identity(K, q)

    rows = []
    for i in range(q):
        row = [0]*q
        row[(i+1)%q] = 1
        rows.append(row)
    X = Matrix(K, rows)

    rows = [[0]*q for i in range(q)]
    for i in range(q):
        rows[i][i] = w3**i
    Z = Matrix(K, rows)

    Pauli = mulclose([X, Z])
    assert len(Pauli) == q**3

    rows = [[w3**(i*j) for i in range(q)] for j in range(q)]
    H = ir3*Matrix(K, rows)
    assert H.order() == 4
    assert (H.d)*H == I
    assert ~H == H.d

    for g in Pauli:
        assert (~H)*g*H in Pauli

    qhalf = 2 # fix for q!=3
    
    rows = [[0]*q for i in range(q)]
    for i in range(q):
        rows[i][i] = w3**(i*(i+1)*qhalf)
    S = Matrix(K, rows)

    for g in Pauli:
        assert (~S)*g*S in Pauli
    assert ~S == S.d
    
    Cliff = mulclose([X, S, H])
    print(len(Cliff))

    op = J*I
    print(op in Cliff)

    # yes, works:
    #for g in Cliff:
    #    for h in Pauli:
    #        assert (g.d)*h*g in Pauli
    assert type(Pauli) is set
    


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




