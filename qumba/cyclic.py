#!/usr/bin/env python

"""
Build some cyclic codes, we get all gf4 linear cyclic codes easily,
these include the self-dual codes.

More general algorithm is here:
https://arxiv.org/abs/1007.1697

"""


from random import shuffle
from functools import reduce
from operator import add, mul

import numpy

from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
    PolynomialRing, GF, factor)
from sage import all_cmdline

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, Matrix, get_weight, fromstr
from qumba import construct
from qumba.argv import argv
from qumba.distance import distance_z3


def all_cyclic(n, dmin=1, gf4_linear=True):
    F = GF(4)
    z2 = F.gen()
    R = PolynomialRing(F, "x")
    x = R.gen()

    A = factor(x**n - 1)
    #print(A)
    space = SymplecticSpace(n)

    mkpauli = lambda a:''.join({0:'.', 1:'X', z2:'Z', z2+1:'Y'}[a[i]] for i in range(n))

    factors = [a for (a,j) in A]
    N = len(factors)
    #print("factors:", N)

    scalars = [1, z2, z2+1] if gf4_linear else [1]

    # build all the principle ideals
    for bits in numpy.ndindex((2,)*N):
        a = reduce(mul, (factors[i] for i in range(N) if bits[i]), x**0)
        #if sum(bits)==N:
        #    assert n%2==0 or a == x**n+1, a
        #    continue 
        #print(gen)
    
        rows = []
        for i in range(n):
          for scalar in scalars:
            gen = mkpauli(scalar*a)
            s = ''.join(gen[(k+i)%n] for k in range(n))
            rows.append(s)
        s = ' '.join(rows)
        H = space.fromstr(s)
        H = H.linear_independent()
        U = H * space.F * H.t
        if U.sum():
            continue # not isotropic
        #print(H)
        code = QCode(H)
        #print(code.longstr())
        #print(code, end=' ', flush=True)
        d = distance_z3(code)
        code.d = d
        if d >= dmin:
            yield code
        #print(gen, code)
        #print(code.longstr())

def main():
    for n0 in range(2, 20):
        if argv.even:
            n = 2*n0
        else:
            n = 2*n0 + 1
        for code in all_cyclic(n, 3):
            sd = code.is_selfdual()
            H = code.H
            rws = [get_weight(h) for h in H.A]
            if code.k:
                print(code, set(rws), "*" if sd else "")
                #print(code.longstr())
            assert code.is_gf4_linear()
            tgt = code.apply_perm([(i+1)%n for i in range(n)])
            assert tgt.is_equiv(code)


def test_513():
    code = construct.get_513()
    assert code.is_gf4_linear()


def test_golay():
    n = 24
    #code = construct.get_golay()
    #n = code.n
    #print(code)

    n = 23
    for code in all_cyclic(n):
        print(code)

        print(code.longstr())
        assert code.is_gf4_linear()
        assert code.is_selfdual()
    
        tgt = code.apply_perm([(i+1)%n for i in range(n)])
        assert tgt.is_equiv(code)


def test_13_1_5():
    """
    Disambiguate some [[13,1,5]] cyclic codes..
    refs:
    https://errorcorrectionzoo.org/c/stab_13_1_5
    https://arxiv.org/abs/quant-ph/9704019 page 10-11
    """
    code = construct.get_xzzx(2,3)
    #assert distance_z3(code) == 5
    print(code)
    print(code.longstr())
    print()
    H = code.H
    n = code.n
    space = SymplecticSpace(n)
    u = space.fromstr("X.ZZ.X.......")
    u = space.fromstr("XYZYYZYX.....")
    gen = "XZ.ZX........"
    for i in range(n):
        s = ''.join(gen[(k+i)%n] for k in range(n))
        u = space.fromstr(s)
        #print(H * space.F * u.t)

    # how to find the perms:
    #N, perms = code.get_autos()
    from qumba.action import Perm, Group
    items = list(range(n))
    gen = [
        Perm([0, 8, 7, 11, 5, 4, 9, 2, 1, 6, 12, 3, 10], items),
        Perm([1, 0, 8, 12, 11, 5, 10, 9, 2, 7, 6, 4, 3], items)]
    G = Group.generate(gen)
    assert len(G) == 26

    for g in G:
        if g.order() == n:
            break

    del code, H

    gen = ("XY.Z.YX......") # [[13,1,3]]
    gen = ("XZ.Z.ZX......") # [[13,1,3]]
    gen = ("X.ZZ.X.......")
    gen = ("ZXIIIXZ......")
    gen = ("XYZYYZYX.....")
    rows = []
    for i in range(n-1):
        s = ''.join(gen[(k+i)%n] for k in range(n))
        rows.append(s)
    s = ' '.join(rows)
    H = space.fromstr(s)
    #print(H)
    code = QCode(H)
    print(code)
    assert distance_z3(code) == 5, distance_z3(code)

    #u = space.fromstr("X.ZZ.X.......")
    #print(H * space.F * u.t)

    #from qumba.transversal import find_local_cliffords
    #for E in find_local_cliffords(code, code):
    #    print(E)

    print(code.longstr())

    return



if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next()
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
        main()


    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")


