#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel)
from qumba.qcode import QCode, SymplecticSpace
from qumba import css
from qumba.argv import argv



def find_logical_autos(code):
    K = SymplecticSpace(code.k)
    kk = 2*code.k

    space = code.space
    N, perms = code.get_autos()
    M = code.get_symplectic()
    Mi = dot2(space.F, M.transpose(), space.F)
    I = identity2(code.nn)
    assert eq2(dot2(M, Mi), I)
    assert eq2(dot2(Mi, M), I)

    gens = []
    for f in perms:
        A = space.get_perm(f)
        dode = QCode.from_symplectic(dot2(A, M), code.m)
        assert dode.equiv(code)

        MiAM = dot2(Mi, A, M)
        L = MiAM[-kk:, -kk:]
        assert K.is_symplectic(L)
        gens.append(L)

    from sage.all_cmdline import GF, matrix, MatrixGroup
    field = GF(2)
    gens = [matrix(field, kk, A.copy()) for A in gens]
    G = MatrixGroup(gens)
    print(G.structure_description())


def fixed(f):
    return [i for i in range(len(f)) if f[i]==i]

def is_identity(f):
    for i in range(len(f)):
        if f[i] != i:
            return False
    return True

def mul(f, g):
    return [f[g[i]] for i in range(len(g))]
        

def find_logicals(Ax, Az):

    Hx = linear_independent(Ax)
    Hz = linear_independent(Az)
    code = QCode.build_css(Hx, Hz)
    space = code.space

    perms = css.find_autos(Ax, Az)
    print("perms:", len(perms))

    duality = css.find_zx_duality(Ax, Az)

    dode = code.apply_perm(duality)
    dode = dode.apply_H()
    assert code.equiv(dode)

    n = code.n
    kk = 2*code.k
    K = SymplecticSpace(code.k)
    M = code.get_symplectic()
    Mi = dot2(space.F, M.transpose(), space.F)
    I = identity2(code.nn)
    assert eq2(dot2(M, Mi), I)
    assert eq2(dot2(Mi, M), I)

    gens = []
    A = dot2(space.get_H(), space.get_perm(duality))
    gens.append(A)

    for f in perms:
        A = space.get_perm(f)
        gens.append(A)

    for f in perms:
        # zx duality
        zx = mul(duality, f)
        if not is_identity(mul(zx, zx)) or len(fixed(zx))%2 != 0:
            continue
        # XXX there's more conditions to check

        A = I
        remain = set(range(n))
        for i in fixed(zx):
            A = dot2(space.get_S(i), A)
            remain.remove(i)
        for i in range(n):
            if i not in remain:
                continue
            j = zx[i]
            assert zx[j] == i
            remain.remove(i)
            remain.remove(j)
            A = dot2(space.get_CZ(i, j), A)
        gens.append(A)
        break # we only need one of these

    print("gens:", len(gens))

    logicals = []
    for A in gens:
        dode = QCode.from_symplectic(dot2(A, M), code.m)
        assert dode.equiv(code)
        MiAM = dot2(Mi, A, M)
        L = MiAM[-kk:, -kk:]
        assert K.is_symplectic(L)
        logicals.append(L)
    print("logicals:", len(logicals))

    from sage.all_cmdline import GF, matrix, MatrixGroup
    field = GF(2)
    logicals = [matrix(field, kk, A.copy()) for A in logicals]
    G = MatrixGroup(logicals)
    print(G.structure_description())
    print("|G| =", len(G))


def test_bring():
    # Bring's code
    Ax = parse("""
    ......1......1.......1.1...1..
    .1.........11..1...........1..
    .1....1..1....1.1.............
    ....1..1....1........1.......1
    .....1..1.1.......11..........
    ...1....1.....11......1.......
    .........1...1...1......11....
    1.1.......1...............1..1
    ..1.1......1......1...1.......
    .......1...............11.1.1.
    ...1.1..........1...1....1....
    1................1.11.......1.
    """)
    Az = parse("""
    ......1.........1...1..1....1.
    1............1...1...1.......1
    ..1........1...........1..11..
    .....1....1.............111...
    ....1.1.......1......11.......
    ........1.1.1..1.............1
    .1.....1.1..1...........1.....
    ....1..1..........11........1.
    ........11....1..1.1..........
    .1...1.....1....1.1...........
    ...1.........1.1.........1.1..
    1.11................1.1.......
    """)
    find_logicals(Ax, Az)


def test_5_1_3():
    H = """
    XZZX.
    .XZZX
    X.XZZ
    ZX.XZ
    """
    #H = "XX ZZ"
    code = QCode.fromstr(H)
    print(code)

    N, perms = code.get_autos()
    assert N==10 # dihedral group

def test_autos():
    codes = QCode.load_codetables()
    for code in codes:
        if code.n > 10:
            break
        print("[[%s, %s, %s]]"%(code.n, code.k, code.d), end=" ", flush=True)
        N, perms = code.get_autos()
        print("perms:", N)


def test_iso():
    code = QCode.fromstr("X. .Z")
    dode = code.apply_perm([1,0])
    iso = code.get_iso(dode)
    assert iso == [1, 0]


def test_10_2_3():
    Ax = parse("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..X.XXX...
    """)

    Az = parse("""
    .ZZ..Z..Z.
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    ZZ.....Z.Z
    """)

    Hx = linear_independent(Ax)
    Hz = linear_independent(Az)
    code = QCode.build_css(Hx, Hz)

    print(code)
    print()

    N, perms = code.get_autos()
    assert N==20

    dode = code.apply_perm(perms[0])
    assert code.equiv(dode)

    dode = code.apply_H()
    print("dode:")
    print(dode)
    assert not code.equiv(dode)

    iso = code.get_iso(dode)
    print(iso)
    eode = code.apply_perm(iso)
    assert eode.equiv(dode)

    print("eode:")
    print(eode)

    return

    """
    0123456789
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..X.XXX...
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    ZZ.....Z.Z
    """

    idxs = list(range(code.n))
    pairs = [(i,j) for i in idxs for j in idxs if i!=j]
    print(len(pairs))
    assert code.equiv(code)

    N = len(pairs)
    for idx in range(N):
     print("idx =", idx)
     ci = code.apply_CNOT(*pairs[idx])
     for jdx in range(idx+1, N):
      cj = ci.apply_CNOT(*pairs[jdx])
      for kdx in range(jdx+1, N):
        ck = cj.apply_CNOT(*pairs[kdx])
        #print(c)
        if ck.equiv(code):
            print("FOUND!")
            return
    #find(Ax, Az)


def get_m24():
    # Golay code:
    H = parse("""
    1...........11...111.1.1
    .1...........11...111.11
    ..1.........1111.11.1...
    ...1.........1111.11.1..
    ....1.........1111.11.1.
    .....1......11.11..11..1
    ......1......11.11..11.1
    .......1......11.11..111
    ........1...11.111...11.
    .........1..1.1.1..1.111
    ..........1.1..1..11111.
    ...........11...111.1.11
    """)
    return QCode.build_css(H, H)


def test_symplectic():

    for s in [
        "XI IX",
        "XI IZ",
        "ZI IZ",
        "ZI IX",
        "XX ZZ",
        "ZZXX XYZZ",
    ]:
        code = QCode.fromstr(s)
    
        space = code.space
        F = space.F
        M = code.get_symplectic()
        assert space.is_symplectic(M)
        assert QCode.from_symplectic(M, code.m) == code
    
        A = space.get_S()
        assert space.is_symplectic(A)
        AM = dot2(A, M)
        assert space.is_symplectic(AM)
        dode = QCode.from_symplectic(AM, code.m)
        eode = code.apply_S()
        assert eode == dode
    
        A = space.get_CNOT(0, 1)
        assert space.is_symplectic(A)
        AM = dot2(A, M)
        assert space.is_symplectic(AM)
        dode = QCode.from_symplectic(AM, code.m)
        eode = code.apply_CNOT(0, 1)
        assert eode == dode

        f = list(range(code.n))
        shuffle(f)
        A = space.get_perm(f)
        assert space.is_symplectic(A)
        AM = dot2(A, M)
        assert space.is_symplectic(AM)
        dode = QCode.from_symplectic(AM, code.m)
        eode = code.apply_perm(f)
        assert eode == dode
    
    toric = QCode.fromstr("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    ZZ.....Z.Z
    """)


    # RM [[16,6,4]]
    H = parse("""
    11111111........
    ....11111111....
    ........11111111
    11..11..11..11..
    .11..11..11..11.
    """)

    rm = QCode.build_css(H, H)
    assert rm.is_css()

    #for code in QCode.load_codetables():
    #    print(code, code.is_css())
    code = QCode.fromstr("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    YZ.X..XY.Z
    """)
    assert code.is_css()


def test():
    test_5_1_3()
    test_autos()
    test_iso()
    test_10_2_3()
    test_symplectic()
    test_bring()



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





