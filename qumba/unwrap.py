#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle, choice

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, zeros2, solve2, normal_form)
from qumba.matrix import Matrix
from qumba.qcode import QCode, SymplecticSpace
from qumba import construct
from qumba.distance import distance_z3
from qumba.autos import get_isos
from qumba.argv import argv


def unwrap(code, check=True):
    H0 = code.deepH
    m, n, _ = H0.shape
    Sx = H0[:, :, 0]
    Sz = H0[:, :, 1]
    Sxz = numpy.concatenate((Sx, Sz), axis=1)
    Szx = numpy.concatenate((Sz, Sx), axis=1)
    H = zeros2(2*m, 2*n, 2)
    H[:m, :, 0] = Sxz
    H[m:, :, 1] = Szx
    code = QCode(H, check=check)
    return code


# from bruhat.unwrap
def unwrap_encoder(code):
    E = code.get_encoder()
    Ei = code.space.invert(E)
    space = SymplecticSpace(2*code.n)

    n, m, k = code.n, code.m, code.k
    E2 = zeros2(4*n, 4*n)
    E2[::2, ::2] = E
    E2[1::2, 1::2] = Ei.t
    E2 = Matrix(E2)
    assert space.is_symplectic(E2)
    F = space.F

    perm = list(range(4*n))
    for i in range(m):
        a, b = perm[4*i+2:4*i+4]
        perm[4*i+2:4*i+4] = b, a
    E2 = E2[:, perm]
    assert space.is_symplectic(E2)

    #HT = E2.t[:4*m, :]
    #print(strop(HT))
    #print()

    code2 = QCode.from_encoder(E2, 2*m)
    #print(code2.longstr(), code2)
    return code2




def zxcat(code, duality):
    #print(duality)
    pairs = []
    perm = []
    for (i, j) in enumerate(duality):
        if i==j:
            return None
        assert i!=j
        if i < j:
            pairs.append((i, j))
            perm.append(i)
            perm.append(j)
    assert len(pairs)*2 == len(duality)
    #print(pairs)

    right = code.apply_perm(perm)

    inner = construct.get_422()
    left = len(pairs) * inner

    #print(left)
    #print(right)
    right = QCode.trivial(left.n - right.n) + right

    code = left << right
    return code


def test_codetables():
    for code in QCode.load_codetables():
        if code.n < 11:
            continue
        if code.n > 20:
            break
        if code.k == 0:
            continue
        code2 = unwrap(code)
        code2.get_params()
        if code2.d is None:
            code2.d = distance_z3(code2)
        print(code, code2)


def test_all_codes():
    n, k, d = argv.get("params", (4, 1, 2))
    found = set()
    for code in construct.all_codes(n, k, d):
        dode = unwrap(code)
        dode.get_params()
        desc = "%s %s"%(code, dode)
        if desc not in found:
            print(desc)
            print(code.longstr())
            print("-->")
            print(dode.longstr())
            found.add(desc)


def test_zx():
    for code in QCode.load_codetables():
        if code.n > 8:
            break
        if code.k == 0:
            continue
        print()
        code2 = unwrap(code)
        code2.get_params()
        print(code, code2)
        dode = code2.get_dual()
        #iso = code2.get_iso(dode)
        for iso in get_isos(code2, dode):
            break
        else:
            continue
        print(iso)

        code = zxcat(code2, iso)
        if code is None:
            continue
        #print(code)
        #print(code.longstr())
        assert code.is_selfdual()
        #print(code.get_params())
        print("found:", code)
        #if code.n > 5:
        #    break


test = test_zx


if __name__ == "__main__":

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





