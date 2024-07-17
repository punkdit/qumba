#!/usr/bin/env python


from random import shuffle
from functools import reduce
from operator import add

import numpy

from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
    PolynomialRing, GF, factor)
from sage import all_cmdline

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, Matrix, get_weight, fromstr
from qumba.construct import get_xzzx
from qumba.argv import argv



def main():
    F = GF(4)
    z2 = F.gen()
    R = PolynomialRing(F, "x")
    x = R.gen()

    n = 13
    A = factor(x**n - 1)
    print(A)

    for item in A:
        a = item[0]
        print(a)
        print(a*(1+x))
        print()
        #for i in a:
        #    print('\t', i)
        #    assert i in F
        #print(a[5])


def test_13_1_5():
    """
    refs:
    https://errorcorrectionzoo.org/c/stab_13_1_5
    https://arxiv.org/abs/quant-ph/9704019 page 10-11
    """
    from qumba.distance import distance_z3
    code = get_xzzx(2,3)
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


