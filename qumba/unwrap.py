#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle, choice
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul


import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, zeros2, solve2, normal_form)
from qumba.matrix import Matrix
from qumba.qcode import QCode, SymplecticSpace, strop
from qumba import construct
from qumba.distance import distance_z3
from qumba.autos import get_isos, is_iso
from qumba.action import Perm
from qumba.argv import argv

@cache
def toziporder(n):
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(i+n)
    return perm

def unwrap_matrix(H, ziporder=False):
    if isinstance(H, Matrix):
        H = H.A
    H0 = H.view()
    m, nn = H.shape
    n = nn//2
    H0.shape = m, n, 2
    m, n, _ = H0.shape
    Sx = H0[:, :, 0]
    Sz = H0[:, :, 1]
    Sxz = numpy.concatenate((Sx, Sz), axis=1)
    Szx = numpy.concatenate((Sz, Sx), axis=1)
    H = zeros2(2*m, 2*n, 2)
    H[:m, :, 0] = Sxz
    H[m:, :, 1] = Szx
    if ziporder:
        P = toziporder(n)
        H = H[:, P, :].copy()

    H.shape = 2*m, 2*nn
    H = Matrix(H)
    return H


def unwrap(code, ziporder=False, check=True):
    H = unwrap_matrix(code.H, ziporder)
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


def get_pairs(duality):
    pairs = []
    perm = []
    n = len(duality.items)
    for i in range(n):
        j = duality[i]
        if i==j:
            return None
        assert i!=j
        if i < j:
            pairs.append((i, j))
            perm.append(i)
            perm.append(j)
    assert len(pairs)*2 == n
    return pairs


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


def get_zx_dualities(code):
    space = code.space
    n = space.n
    H = space.H
    g = reduce(mul,[H(i) for i in range(n)])
    dode = code.apply(g)
    items = list(range(n))
    perms = [Perm(perm,items) for perm in get_isos(code,dode)]
    return perms

def get_zx_wrap(code):
    "these are the fixed-point-free involutory zx dualities"
    n = code.n
    perms = get_zx_dualities(code)
    items = list(range(n))
    I = Perm(items,items)
    assert I*I==I
    #print(len(perms))
    zxs = []
    for g in perms:
        if g*g!=I:
            continue
        for i in range(n):
            if g[i]==i:
                break
        else:
            zxs.append(g)
    return zxs

def wrap(code, zx):
    nn = code.nn
    pairs = get_pairs(zx)
    #print(code.H)
    Hx = code.H[:,0:nn:2]
    #print(Hx)
    cols = reduce(add,pairs)
    H = Hx[:, list(cols)]
    H = H.linear_independent()
    code = QCode(H)
    return code


def scramble(code):
    H = code.H
    m, nn = H.shape
    #print(H, H.shape, H.rank(), m)
    while 1:
        J = Matrix.rand(m, m)
        #print(J, J.shape)
        H1 = J*H
        if H1.rank() < m:
            continue
        #print()
        #print(H1, H1.shape, H1.rank())
        break
    code = QCode(H1)
    return code


def test_wrap():
    import transversal
    a_code = construct.get_toric(2,2)
    b_code = unwrap(construct.get_412())
    n = a_code.n
    iso = iter(get_isos(a_code, b_code)).__next__()
    iso = Perm(iso, list(range(n)))
    print(iso)

    a_zxs = get_zx_wrap(a_code)
    b_zxs = get_zx_wrap(b_code)
    print(a_zxs)
    print(b_zxs)
    for g in a_zxs:
        g = iso*g*(~iso)
        assert g in b_zxs
    for g in b_zxs:
        g = (~iso)*g*iso
        assert g in a_zxs
    return

    for zx in zxs:
        dode = wrap(code, zx)
        print()
        print(dode)
        print(strop(dode.H))
        gens = []
        for M in transversal.find_local_clifford(dode, dode):
            gens.append(M)
        print("local cliffords:", len(gens))
        gens = list(get_isos(dode,dode))
        print("autos:", len(gens))
    

def test_wrap_toric():
    import transversal
    code = construct.get_toric(2,2)
    #code = unwrap(construct.get_412())
    if 0:
        toric = construct.get_toric(3,1)
        code = construct.get_513()
        code = unwrap(code)
        code.distance("z3")
        print(code)
        print(strop(code.H))
        print(toric)
        print(strop(toric.H))
        assert is_iso(code, construct.get_toric(3,1))
        for perm in get_isos(code, construct.get_toric(3,1)):
            print(perm)
        return

    zxs = get_zx_wrap(code)
    print("zx-dualities:", len(zxs))

    for zx in zxs:
        eode = wrap(code, zx)
        print()
        print(eode)
        #print(eode.longstr())
        print(strop(eode.H))
        src = unwrap(eode)
        assert src.distance("z3") == code.d
        gens = []
        for M in transversal.find_local_clifford(eode, eode):
            gens.append(M)
        print("local cliffords:", len(gens))
        gens = list(get_isos(eode,eode))
        print("autos:", len(gens))


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





