#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)
from qumba.qcode import QCode, SymplecticSpace
from qumba import csscode, construct
from qumba.construct import get_422, get_513, golay, get_10_2_3, reed_muller
from qumba.argv import argv


def find_logical_autos(code):
    K = SymplecticSpace(code.k)
    kk = 2*code.k

    space = code.space
    N, perms = code.get_autos()
    M = code.get_encoder()
    Mi = dot2(space.F, M.transpose(), space.F)
    I = identity2(code.nn)
    assert eq2(dot2(M, Mi), I)
    assert eq2(dot2(Mi, M), I)

    gens = []
    for f in perms:
        A = space.get_perm(f)
        dode = QCode.from_encoder(dot2(A, M), code.m)
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

    perms = csscode.find_autos(Ax, Az)
    print("perms:", len(perms))

    duality = csscode.find_zx_duality(Ax, Az)

    dode = code.apply_perm(duality)
    dode = dode.apply_H()
    assert code.equiv(dode)

    n = code.n
    kk = 2*code.k
    K = SymplecticSpace(code.k)
    M = code.get_encoder()
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
        dode = QCode.from_encoder(dot2(A, M), code.m)
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


def test_codetables():
    for code in QCode.load_codetables():
        if code.n > 10:
            break
        print("[[%s, %s, %s]]"%(code.n, code.k, code.d), end=" ", flush=True)
        N, perms = code.get_autos()
        if code.is_css():
            print("css, ", end="")
        if code.is_selfdual():
            print("selfdual, ", end="")
        print("autos:", N)


def test_isomorphism():
    code = QCode.fromstr("X. .Z")
    dode = code.apply_perm([1,0])
    iso = code.get_isomorphism(dode)
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

    #print(code)
    #print()

    N, perms = code.get_autos()
    assert N==20

    dode = code.apply_perm(perms[0])
    assert code.equiv(dode)

    dode = code.apply_H()
    #print("dode:")
    #print(dode)
    assert not code.equiv(dode)

    iso = code.get_isomorphism(dode)
    #print(iso)
    eode = code.apply_perm(iso)
    assert eode.equiv(dode)

    #print("eode:")
    #print(eode)

    duality = csscode.find_zx_duality(Ax, Az)
    print(duality)

    n = code.n
    pairs = []
    remain = set(range(n))
    for i, j in enumerate(duality):
        assert duality[j] == i
        assert i != j
        if i<j:
            pairs.append((i, j))

    dode = code
    for (i,j) in pairs:
        dode = dode.apply_CZ(i, j)
        print(dode.get_params())
        print(dode.equiv(code))
    print(code.get_logical(dode))

    return

    dode = code
    print(dode.longstr())
    print(dode.get_params())
    print()
    for (i,j) in pairs:
        dode = dode.apply_CNOT(i, j)
    #print(dode.longstr())
    #print(dode.get_params())
    #print()
    for (i,j) in pairs:
        dode = dode.apply_CNOT(j, i)
    #print(dode.longstr())
    #print(dode.get_params())
    #print()
    for (i,j) in pairs:
        dode = dode.apply_CNOT(i, j)
    #print(dode.longstr())
    #print(dode.get_params())
    #print()
    f = dode.get_isomorphism(code)
    dode = dode.permute(f)
    print(code.equiv(dode))
    print(dode.longstr())
    print(dode.get_params())
    print()

    print(code.get_logical(dode))
        


def test_toric_cnot():

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

    code = get_10_2_3()
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
    print("NOT FOUND")


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
        M = code.get_encoder()
        assert space.is_symplectic(M)
        assert QCode.from_encoder(M, code.m) == code
    
        A = space.get_S()
        assert space.is_symplectic(A)
        AM = dot2(A, M)
        assert space.is_symplectic(AM)
        dode = QCode.from_encoder(AM, code.m)
        eode = code.apply_S()
        assert eode == dode
    
        A = space.get_CNOT(0, 1)
        assert space.is_symplectic(A)
        AM = dot2(A, M)
        assert space.is_symplectic(AM)
        dode = QCode.from_encoder(AM, code.m)
        eode = code.apply_CNOT(0, 1)
        assert eode == dode

        f = list(range(code.n))
        shuffle(f)
        A = space.get_perm(f)
        assert space.is_symplectic(A)
        AM = dot2(A, M)
        assert space.is_symplectic(AM)
        dode = QCode.from_encoder(AM, code.m)
        eode = code.apply_perm(f)
        assert eode == dode
    

    code = get_513()
    N, perms = code.get_autos()
    assert N==10 # dihedral group

    rm = reed_muller()
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


def test_concatenate_show():
    bflip = QCode.fromstr("ZZI IZZ")
    print(bflip.longstr())
    print()
    pflip = bflip.get_dual()
    print(pflip.longstr())

    outer = bflip
    print(shortstr(outer.get_encoder()))
    print()
    inner = 3*pflip
    E1 = inner.get_encoder()
    print(inner)
    print(shortstr(E1))
    print()

    trivial = QCode.from_symplectic(identity2(12))
    rhs = trivial + outer
    E0 = rhs.get_encoder()
    print(shortstr(E0))
    print()

    F = dot2(E1, E0)
    result = QCode.from_symplectic(F, k=1)
    print(result)
    print(result.get_params())
    print(shortstr(F))
    print(result.longstr())
    assert result.get_params() == (9, 1, 3)
    



#def test_concatenate_422():
def test_concatenate():
    inner = get_422()

    right = QCode.trivial(2) + QCode.fromstr("XX ZZ")
    result = inner << right
    assert result.equiv( QCode.fromstr("XXXX ZZZZ IZZI IXXI") )

    # --------------------

    code = QCode.fromstr("XXXX ZZZZ", None, "XXII ZIZI XIXI ZZII")

    right = QCode.trivial(4) + code
    #right = right.apply_perm([0, 1, 4, 5, 2, 3, 6, 7])
    #print(shortstr(right.get_encoder()))
    #print()

    left = code + code
    #print(shortstr(left.get_encoder()))

    result = left << right
    print(result.get_params())
    print(result.longstr())
    assert result.get_params() == (8, 2, 2)


def test_concatenate_steane():
    surface = QCode.fromstr("ZZZII IIZZZ XIXXI IXXIX")
    assert surface.get_params() == (5, 1, 2)

    surface = surface.apply_perm([1, 2, 3, 0, 4])
    inner = get_422()

    right = QCode.trivial(2) + surface
    left = inner + QCode.trivial(3)

    code = left << right
    assert code.get_params() == (7, 1, 3)
    assert code.is_selfdual()
    assert code.is_css()


def test_css():
    code = get_10_2_3()
    print(code)

    c1 = code.to_css()
    print(c1)


def test_concatenate_zx():
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

    duality = csscode.find_zx_duality(Ax, Az)
    #print(duality)
    pairs = []
    perm = []
    for (i, j) in enumerate(duality):
        assert i!=j
        if i < j:
            pairs.append((i, j))
            perm.append(i)
            perm.append(j)
    assert len(pairs)*2 == len(duality)
    #print(pairs)

    Hx = linear_independent(Ax)
    Hz = linear_independent(Az)
    right = QCode.build_css(Hx, Hz)
    #print("perm:", perm)
    right = right.apply_perm(perm)

    inner = get_422()
    left = len(pairs) * inner

    #print(left)
    #print(right)
    right = QCode.trivial(left.n - right.n) + right

    code = left << right
    #print(code)
    #print(code.longstr())
    assert code.is_selfdual()
    #print(code.get_params())



def test_concatenate_toric():
    toric = get_10_2_3()
    dual = toric.get_dual()
    iso = toric.get_isomorphism(dual)
    print(iso)
    
    outer = toric + dual

    print("outer:", outer)

    f = []
    for (i,j) in enumerate(iso):
        f.append(i)
        f.append(toric.n + j)
    print("f:", f)
    #P = outer.space.get_perm(f)
    outer = outer.apply_perm(f)
    print(outer.longstr())

    inner = get_422()
    print(inner)

    left = (outer.n // inner.k) * inner

    print("left:", left)

    trivial = QCode.trivial(left.m)
    right = trivial + outer
    print(right)

    result = left << right
    print(result)
    print(result.longstr())
    print(result.is_selfdual())

    # hmm...


def test_biplanar():
    code = construct.biplanar()
    print(code)
    Hx, Hz = code.Hx, code.Hz
    #print(shortstr(code.Hx))
    #print(Hx.sum(0), Hx.sum(1))
    #print(shortstr(code.Lx))


def test():
    test_codetables()
    test_isomorphism()
    test_10_2_3()
    test_symplectic()
    test_concatenate()
    #test_bring() # slow..



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





