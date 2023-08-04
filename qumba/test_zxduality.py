#!/usr/bin/env python

from time import time
start_time = time()

import numpy

from qumba.algebraic import Matrix, Algebraic
from qumba.solve import parse, shortstr, linear_independent, eq2, dot2, identity2
from qumba.isomorph import Tanner, search
from qumba.qcode import QCode, symplectic_form
from qumba.argv import argv


def find_autos_css(Ax, Az):
    ax, n = Ax.shape
    az, _ = Az.shape

    U = Tanner.build2(Ax, Az)
    V = Tanner.build2(Ax, Az)
    perms = []
    for g in search(U, V):
        # each perm acts on ax+az+n in that order
        assert len(g) == n+ax+az
        for i in range(ax):
            assert 0<=g[i]<ax
        for i in range(ax, ax+az):
            assert ax<=g[i]<ax+az
        g = [g[i]-ax-az for i in range(ax+az, ax+az+n)]
        perms.append(g)
    return perms


def find_zx_duality(Ax, Az):
    # find a zx duality
    ax, n = Ax.shape
    az, _ = Az.shape
    U = Tanner.build2(Ax, Az)
    V = Tanner.build2(Az, Ax)
    for duality in search(U, V):
        break
    else:
        return None

    dual_x = [duality[i] for i in range(ax)]
    dual_z = [duality[i]-ax for i in range(ax,ax+az)]
    dual_n = [duality[i]-ax-az for i in range(ax+az,ax+az+n)]

    Ax1 = Ax[dual_z, :][:, dual_n]
    Az1 = Az[dual_x, :][:, dual_n]

    assert eq2(Ax1, Az)
    assert eq2(Az1, Ax)
    return dual_n


def main_bring():
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
    find(Ax, Az)


def main_5_1_3():
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


def test_iso():
    code = QCode.fromstr("X. .Z")
    dode = code.permute([1,0])

    iso = code.get_iso(dode)



def main_10_2_3():
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

    dode = code.permute(perms[0])
    assert code.equiv(dode)

    dode = code.apply_H()
    print(dode)
    assert not code.equiv(dode)

    #iso = code.get_iso(dode)
    #print(iso)

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


def find(Ax, Az):

    print(Ax.shape)

    autos = find_autos_css(Ax, Az)
    print("autos:", len(autos))

    duality = find_zx_duality(Ax, Az)
    print("duality:", duality)

    Hx = linear_independent(Ax)
    Hz = linear_independent(Az)
    code = QCode.build_css(Hx, Hz)

    print(code)
    print()

    M = code.get_symplectic()
    #print(shortstr(M))

    F = symplectic_form(code.n)
    assert eq2(F, dot2(M, F, M.transpose()))

    I = identity2(2*code.n)
    assert eq2(dot2(F, F), I) # self inverse

    Mi = dot2(F, M.transpose(), F)
    assert eq2(dot2(M, Mi), I)

    Hx = code.H
    Tz = code.T
    Lx = code.L[0::2, :]
    Lz = code.L[1::2, :]

    # encoder
    E = numpy.concatenate((Tz, Lz)).transpose()

    # decoder
    D = numpy.concatenate((Hx, Lx))
    D = dot2(D, F)

    DE = dot2(D, E)
    assert eq2(DE, identity2(code.n))
    
    return

    F = code.overlap(code)
    print(shortstr(F))

    for g in autos:
        tgt = code.permute(g)
        F = code.overlap(tgt)
        #print(shortstr(F))


def test():
    c = QCode.fromstr("XI IZ")
    c.build()
    print(c)
    print()

    c1 = c.apply_CNOT(0, 1)
    print(c1)




if __name__ == "__main__":

    start_time = time()


    profile = argv.profile
    name = argv.next() or "main"
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
    print("finished in %.3f seconds"%t)
    print("OK!\n")





