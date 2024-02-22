#!/usr/bin/env python

from random import shuffle, randint
from operator import add, matmul, mul
from functools import reduce

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)
from qumba.qcode import QCode, SymplecticSpace, strop, Matrix
from qumba.csscode import CSSCode, find_logicals
from qumba.autos import get_autos
from qumba import csscode, construct
from qumba.construct import get_422, get_513, golay, get_10_2_3, reed_muller
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.util import cross
from qumba.symplectic import Building
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv



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
        print(code.longstr())
        print()


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
    find_logicals(Ax, Az)

    return
        

    Hx = linear_independent(Ax)
    Hz = linear_independent(Az)
    code = QCode.build_css(Hx, Hz)

    #print(code)
    #print()

    N, perms = code.get_autos()
    assert N==20

    dode = code.apply_perm(perms[0])
    assert code.is_equiv(dode)

    dode = code.apply_H()
    #print("dode:")
    #print(dode)
    assert not code.is_equiv(dode)

    iso = code.get_isomorphism(dode)
    #print(iso)
    eode = code.apply_perm(iso)
    assert eode.is_equiv(dode)

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
        print(dode.is_equiv(code))
    print(code.get_logical(dode))


def test_toric_logicals():
    css = construct.toric(3, 3)
    Ax, Az = css.Ax, css.Az
    find_logicals(Ax, Az)




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
        #AM = dot2(A, M)
        AM = A*M
        assert space.is_symplectic(AM)
        dode = QCode.from_encoder(AM, code.m)
        eode = code.apply_S()
        assert eode == dode
    
        A = space.get_CNOT(0, 1)
        assert space.is_symplectic(A)
        #AM = dot2(A, M)
        AM = A*M
        assert space.is_symplectic(AM)
        dode = QCode.from_encoder(AM, code.m)
        eode = code.apply_CNOT(0, 1)
        assert eode == dode

        f = list(range(code.n))
        shuffle(f)
        A = space.get_perm(f)
        assert space.is_symplectic(A)
        #AM = dot2(A, M)
        AM = A*M
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

    #F = dot2(E1, E0)
    F = E1*E0
    result = QCode.from_symplectic(F, k=1)
    print(result)
    print(result.get_params())
    print(F)
    print(result.longstr())
    assert result.get_params() == (9, 1, 3)
    



#def test_concatenate_422():
def test_concatenate():
    inner = get_422()

    right = QCode.trivial(2) + QCode.fromstr("XX ZZ")
    result = inner << right
    assert result.is_equiv( QCode.fromstr("XXXX ZZZZ IZZI IXXI") )

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


def test_hgp():
    n, m = 6, 4
    for H in construct.classical_codes(n, m, 3):
        print("H=")
        print(shortstr(H))
        Hx, Hz = construct.hypergraph_product(H, H.transpose())
        code = CSSCode(Hx=Hx, Hz=Hz)
        print(code)
        #print(code.distance())
        #print("Hz =")
        #print(shortstr(Hz))
        #print("Hx =")
        #print(shortstr(Hx))
        #print()



def test_biplanar():
    code = construct.biplanar()
    print(code)
    Hx, Hz = code.Hx, code.Hz
    #print(shortstr(code.Hx))
    #print(Hx.sum(0), Hx.sum(1))
    #print(shortstr(code.Lx))


def test_clifford():
    from qumba.clifford_sage import Clifford, green, red, I, r2, half

    n = 2
    space = SymplecticSpace(n)
    c3 = Clifford(n)

    src, tgt = [], []
    for i in range(n):
        E = space.get_H(i)
        U = space.translate_clifford(E)
        src.append(U)
        tgt.append(E)
        assert U == c3.get_H(i)

        E = space.get_S(i)
        U = space.translate_clifford(E)
        src.append(U)
        tgt.append(E)
        assert U == c3.get_S(i)

    for i in range(n):
      for j in range(n):
        if i==j:
            continue

        E = space.get_CNOT(i, j)
        U = space.translate_clifford(E)
        src.append(U)
        tgt.append(E)
        assert U == c3.get_CNOT(i, j)

        E = space.get_CZ(i, j)
        U = space.translate_clifford(E)
        src.append(U)
        tgt.append(E)
        assert U == c3.get_CZ(i, j)

    hom = mulclose_hom(src, tgt, verbose=True, maxsize=1000)
    gen = [c3.wI(), c3.get_X(0), c3.get_X(1), c3.get_Z(0), c3.get_Z(1)] 
    G = mulclose(gen)
    assert len(G) == 128 # Pauli with w8 phase group
    #assert len(G) == 64
    for s,t in hom.items():
        U = space.translate_clifford(t)
        for g in G:
            if g*U == s:
                break
        else:
            assert 0


def test_encoder():
    #code = QCode.fromstr("XX ZZ")
    #code = QCode.fromstr("ZZI IZZ")
    code = construct.get_422()
    #code = construct.get_513()
    #code = construct.get_10_2_3()
    space = code.space
    m, n, k = code.m, code.n, code.k

    from qumba.clifford_sage import Clifford, green, red, I, r2, half

    k0 = (r2/2)*red(1,0) #   |0>
    k1 = (r2/2)*red(1,0,2) # |1>

    cliff = Clifford(n)

    #print(code.longstr())
    stabs = strop(code.H).split()
    stabs = [cliff.get_pauli(stab) for stab in stabs]
    for g in stabs:
      for h in stabs:
        assert g*h == h*g
    P = reduce(mul, [half*(stab + cliff.I) for stab in stabs])
    PP = P*P
    assert PP == P
    P0 = P # save

    E = code.get_encoder() # symplectic matrix
    D = code.get_encoder(True) # symplectic matrix

    E = space.translate_clifford(E)
    D = space.translate_clifford(D)
    pauli = D*E
    assert pauli * pauli == cliff.I
    D = pauli*D
    assert D*E == cliff.I
    assert E*D == cliff.I

    #p = half * red(1,0) * red(0,1)
    p = half * green(1,0) * green(0,1) # why is this green ??!?!?
    assert p*p == p

    # codespace projector
    items = [I]*n
    for i in range(m):
        items[i] = p
    p = reduce(matmul, items) 
    P = E * p * D
    PP = P*P
    assert P == PP

    #print(P0)
    #print()
    #print(P)
    #print( pauli*P == P0 )
    #print( pauli*P*pauli == P0 )

    assert P*P0 == P0*P

    if argv.slow:
    
        gen = []
        for i in range(n):
            gen.append(cliff.get_X(i))
            gen.append(cliff.get_Z(i))
        G = mulclose(gen, verbose=True)
        print(len(G))
    
        for g in G:
          for h in G:
            if P*g == h*P0:
                print("found!")
                return
        print("no pauli")

    #assert P == P0 # up to some pauli? ... argh...

    # logical encoder
    #L = E * (red(1,0) @ red(1,0) @ I @ I)


def test_spider():
    s1 = SymplecticSpace(1)
    s2 = SymplecticSpace(2)
    S = s1.get_S()
    HSH = S.transpose()
    CNOT = s2.get_CNOT()
    CZ = s2.get_CZ()

    from qumba.solve import shortstr
    from qumba.clifford_sage import Clifford, Matrix, K, r2
    c2 = Clifford(2)
    c4 = Clifford(4)

    E = S.to_spider()
    E1 = c2.get_CNOT()
    assert Matrix(K, E) == E1

    E = HSH.to_spider()
    E1 = c2.get_CNOT(1, 0)
    assert Matrix(K, E) == E1

    E = CNOT.to_spider()
    E1 = c4.get_CNOT(0, 2)*c4.get_CNOT(3, 1)
    assert Matrix(K, E) == E1

    E = CZ.to_spider()
    E1 = c4.get_CNOT(0, 3)*c4.get_CNOT(2, 1)
    assert Matrix(K, E) == E1

    SI = s2.get_S(0)
    E = SI.to_spider()
    E1 = c4.get_CNOT()
    assert Matrix(K, E) == E1

    code = get_513()
    c1 = Clifford(1)
    I, H = c1.get_identity(), c1.get_H()
    IH = reduce(matmul, [I,H]*code.n)
    print("IH", IH.shape)
    e = code.get_encoder()
    E = e.to_spider(verbose=True)
    print(type(E), E.shape)
    E = Matrix(K, list(E))
    E = E*IH
    E = 4*r2*E
    print(E.shape)
    #for i in range(1024):
    #  for j in range(1024):
    #    if E[i,j]:
    #        print(E[i,j], end=" ")
    #  print()
    return E
    



def test_1():

    for ht in [
        "X Z", "Z X", "Y Z",
    ]:
        h, t = ht.split()
        code = QCode.fromstr(Hs=h, Ts=t)
        #print(code)

    space = SymplecticSpace(1)
    I = space.get_identity()
    H, S = space.get_H(0), space.get_S(0)
    G = mulclose([I, H, S])
    assert len(G) == 6

    for g in [
        I, S, H, H*S, S*H, S*H*S # == H*S*H
    ]:
        code = QCode.from_encoder(g)
        print("*".join(g.name))
        print(strop(code.H), strop(code.T))
        print("-----")
        c2 = unwrap_encoder(code)

        h, t = (strop(c2.H), strop(c2.T))
        smap = SMap()
        smap[0, 0] = h
        smap[0, 3] = t
        print(smap)
        E = c2.get_encoder()
        #translate_clifford(c2.space, E, verbose=True)
        print()
        #print(g.to_spider())
        #print()
    print("="*79)


    space = SymplecticSpace(2)
    CX = space.get_CNOT(0, 1)
    SWAP = space.get_perm([1, 0])
    I = space.get_identity()
    HI = space.get_H(0)
    IH = space.get_H(1)
    for E in [
        I,
        CX,
        SWAP,
        SWAP*CX,
        CX*SWAP,
        CX*SWAP*CX,
    ]:
        E = E*IH
        code = QCode.from_encoder(E)
        h, t = (strop(code.H), strop(code.T))
        smap = SMap()
        smap[0, 0] = h
        smap[0, 3] = t
        print(E.name)
        print(smap)
        print()


def test_412_unwrap():
    code = QCode.fromstr("XYZI IXYZ ZIXY")
    print(code.longstr())

    E = code.get_encoder()
    Ei = code.get_encoder(True)
    v = [[0,0, 0,0, 0,1, 1,0]] # IIZX
    v = Matrix(v)
    print(Ei*v.t)

    #return

    #translate_clifford(code.space, code.get_encoder(), verbose=True)

    dode = unwrap_encoder(code)
    print(dode.longstr())

    space = dode.space
    E = dode.get_encoder()
    Ei = space.invert(E)
    #[space.get_CZ(2*i, 2*i+1) for i in range(4)]
    cz = reduce(mul, [space.get_CZ(2*i, 2*i+1) for i in range(4)])
    logop = space.get_H()
    perm = reduce(add, [[2*i+1, 2*i] for i in range(4)])
    logop = space.get_perm(perm) * logop

    I = space.get_identity()
    #print(Ei * logop * E == I)
    eode = QCode.from_encoder(logop*E, m=dode.m)
    print(eode.get_params())
    print(eode.is_equiv(dode))
    
    #space.translate_clifford(E, verbose=True)


def test_conjugacy():
    s = SymplecticSpace(2)
    gen = [s.get_S(0), s.get_S(1), s.get_H(0), s.get_H(1), s.get_CNOT(0,1)]
    G = mulclose(gen)
    print("|G| =", len(G))

    equs = {g:set() for g in G}
    inv = {g:s.invert(g) for g in G}
    for g in G:
      for h in G:
        equs[g].add( inv[h]*g*h )

    cgys = set()
    for g in G:
        cgy = equs[g]
        cgy = list(cgy)
        cgy.sort(key = str)
        cgy = tuple(cgy)
        cgys.add(cgy)
    print("cgys:", len(cgys))
    cgys = list(cgys)
    cgys.sort(key = len)
    for cgy in cgys:
        print("\t%d"%len(cgy))
        best = None
        for g in cgy:
            name = s.get_name(g)
            print("\t\t%s"%("*".join(name)))
            if best is None or len(best[0]) > len(name):
                best = [name]
            elif len(best[0]) == len(name):
                best.append(name)
        #for name in best:
        #    print("\t\t%s"%("*".join(name)))
        if len(cgy) > 10:
            break
    H = mulclose(cgy)
    print(len(H))


def test_genon():

    code = QCode.fromstr("XYZI IXYZ ZIXY")
    print(code.longstr())

    G = list(get_autos(code))
    assert len(G) == 4 # Z/4

    space = SymplecticSpace(1)
    H = space.get_H()
    S = space.get_S()
    cliff = mulclose([H,S])
    print(len(cliff))
    
    n = code.n
    items = [cliff for i in range(n)]
    gates = [reduce(Matrix.direct_sum, item) for item in cross(items)]
    
    from qumba.action import Group
    n = code.n
    bits = list(numpy.ndindex((2,)*n))
    I = space.get_identity()
    
    G = Group.symmetric(n)
    count = 0
    found = set()
    for g in G:
        perm = [g[i] for i in range(n)]
        #print(perm)
        dode = code.apply_perm(perm)
        hit = 0
        for gate in gates:
            eode = dode.apply(gate)
            if not eode.is_equiv(code):
                continue
            L = eode.get_logical(code)
            if L==I:
                print(perm, "==I")
            #if L not in found:
            #    print(perm)
            #    print(L)
            found.add(L)
            count += 1
            hit += 1
        assert hit == 1
    print(count, len(found))

    

def test_majorana():
    from bruhat.algebraic import qchoose_2
    from bruhat.action import Group

    n, m = 3, 2
    #G = mulclose([ Matrix.perm([1,0,2]), Matrix.perm([0,2,1])])
    perms = [[1,0,2], [0,2,1], [1,2,0], [2,0,1]]
    #assert len(G) == 6
    nn = 2*n
    G = Group.symmetric(nn)
    print(len(G))
    space = SymplecticSpace(n)
    uspace = SymplecticSpace(nn)
    F = space.F
    count = 0
    found = 0
    for H in qchoose_2(nn, m):
        H = Matrix(H)
        U = H*F*H.transpose()
        if U.sum():
            continue
        count += 1
        if space.is_majorana(H):
            code = QCode(H)
            found += 1
            #print("*", end="")
            #for perm in perms:
            #    dode = code.apply_perm(perm)
            #    assert space.is_majorana(dode.H)

        code = QCode(H)
        dode = unwrap(code)
        if uspace.is_majorana(dode.H):
            print("/", end="", flush=True)
            
        for g in G:
            items = [g[i] for i in range(nn)]
            #print(g, items)
            eode = dode.apply_perm(items)
            if uspace.is_majorana(eode.H):
                print("+", end="", flush=True)
                break
        else:
            print("-", end="", flush=True)
            #assert 0
#        HH = dode.H
#        print(uspace.is_majorana(HH))
        #print(shortstr(H))
        #print()
    print(count, found)


def test_normal_form():
    #code = QCode.fromstr("XYZI IXYZ ZIXY")
    code = QCode.fromstr("""
    .ZZZZ
    Y.ZXY
    ZX.ZY
    Z.XYZ
    """)
    dode = code.normal_form()
    assert dode.is_equiv(code)
    #E = get_encoder(code)
    #test_clifford_encoder(code, E)

    code = QCode.fromstr("""XXXX ZZZZ""")
    code = construct.get_832() # FAIL
    code = construct.get_713() # FAIL

    code = construct.get_512() # works
    E = code.get_clifford_encoder()
    test_clifford_encoder(code, E)

    # 822 toric code
    code = QCode.fromstr("""
    XXX..X..
    X.XX...X
    .X..XXX.
    ZZ.ZZ...
    .ZZZ..Z.
    Z...ZZ.Z
    """)
    #E = code.get_clifford_encoder()
    #test_clifford_encoder(code, E)

    code = QCode.fromstr("XZXX IXIZ ZIZI") # works
    E = code.get_clifford_encoder()
    test_clifford_encoder(code, E)
    return

    code = QCode.fromstr("XXZX IIXZ ZZII") # FAIL
    #E = get_encoder(code)
    from qumba.clifford_sage import Clifford, red, green, Matrix
    c = Clifford(code.n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    E0 = H(0)*CX(0,1)*CZ(0,2)*CX(0,3)
    E1 = H(2)*CZ(2,3)
    E2 = CZ(3,0)*CZ(3,1)
    E = E0*E1*E2
    test_clifford_encoder(code, E)
    return

    for code in construct.all_codes():
        E = code.get_clifford_encoder()
        test_clifford_encoder(code, E)

    return

    for code in QCode.load_codetables():
        if code.n < 4:
            continue
        if code.k != 1:
            continue
        if code.n > 10:
            break
        print("[[%s, %s, %s]]"%(code.n, code.k, code.d))
        dode = code.normal_form()
        assert dode.is_equiv(code)
        E = code.get_clifford_encoder()
        test_clifford_encoder(code, E)


def test_grassl():
    # from Grassl 2002 "Algorithmic aspects of quantum error-correcting codes"
    from qumba.clifford_sage import Clifford, red, green, Matrix
    if 0:
        code = QCode.fromstr("XIXYY ZIZXX IXYYX IZXXZ")
        c = Clifford(code.n)
        E = get_encoder(code) # FAIL

    elif 0:
        code = QCode.fromstr("YXIZ ZYXI IZYX")
        c = Clifford(code.n)
        CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
        #E = get_encoder(code)

        E2 = CY(2,3)*CZ(2,0)*H(2)
        E1 = CZ(1,3)*CY(1,2)*H(1)
        E0 = CZ(0,2)*CY(0,1)*H(0)
        E = E0*E1*E2 
        E = c.get_P(3,2,1,0) * E
        # WORKS

    elif 1:
        code = QCode.fromstr("XYZI IXYZ ZIXY")
        c = Clifford(code.n)
        CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
        #E = get_encoder(code)

        E2 = CZ(2,0)*CY(2,3)*H(2)
        E1 = CY(1,2)*CZ(1,3)*H(1)
        E0 = CY(0,1)*CZ(0,2)*H(0)
        E = E0*E1*E2 # WORKS

    elif 0:
        c = Clifford(5)
        CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
        swap = c.get_P(4,3,2,1,0)

        code = QCode.fromstr("XXZIZ YIYZZ YZZYI XZIZX") # standard form
        assert code.is_equiv(QCode.fromstr("XIXYY ZIZXX IXYYX IZXXZ"))
        E0 = CZ(0,1) * CZ(0,3) * CX(0,4) * H(0)
        E1 = S(1) * CZ(1,2) * CZ(1,3) * CY(1,4) * H(1)
        E2 = S(2) * CZ(2,0) * CZ(2,1) * CY(2,4) * H(2)
        E3 = CZ(3,0) * CZ(3,2) * CX(3,4) * H(3)
        E = E0 * E1 * E2 * E3
        E = swap * E  # put the logical at the end
    
    else:
        c = Clifford(5)
        CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
        swap = c.get_P(4,3,2,1,0)

        # hadamard on qubit 1
        code = QCode.fromstr("XZZIZ YIYZZ YXZYI XXIZX")
        E0 = CZ(0,1) * CX(0,3) * CX(0,4) * H(0)
        E1 = S(1) * CZ(1,2) * CX(1,3) * CY(1,4) * H(1)
        E2 = S(2) * CZ(2,0) * CZ(2,1) * CY(2,4) * H(2)
        E3 = H(3) * CZ(3,0) * CZ(3,2) * CX(3,4) * H(3)
        E = E0 * E1 * E2 * E3
        E = swap * E  # put the logical at the end

    test_clifford_encoder(code, E)


def test_clifford_encoder(code, E):
    from qumba.clifford_sage import Clifford, red, green, Matrix
    c = Clifford(code.n)
    P = code.get_projector()
    assert P.rank() == 2**code.k
    assert E.rank() == 2**code.n

    zero = red(1,0)
    one = red(1,0,2)
    plus = green(1,0)
    minus = green(1,0,2)
    states = [zero, one, plus, minus]

    stabs = []
    for h in code.H:
        #print(strop(h))
        stabs.append(c.get_pauli(strop(h)))

    for idx in range(4):
        # zero, one, plus, minus
        v = [0]*code.n
        if code.k:
            v[-1] = idx
        v = reduce(matmul, [states[i] for i in v])
        u = E*v
        print("idx", idx)
        for g in stabs:
            print('\t', g*u == u )
        #assert P*u==u
        #for g in stabs:
        #    assert g*u == u

    for idx in range(4):
        # zero, one, plus, minus
        v = [0]*code.n
        if code.k:
            v[-1] = idx
        v = reduce(matmul, [states[i] for i in v])
        u = E*v
        assert P*u==u
        for g in stabs:
            assert g*u == u

    if code.k == 1:
        lhs = E.d * P * E
        rr = red(1,0)*red(0,1)
        I = Clifford(1).get_identity()
        rhs = [rr]*code.n
        rhs[-1] = I
        rhs = reduce(matmul, rhs)
        assert (2**code.m)*lhs == rhs


def test_422_logical():
    code = construct.get_422()
    print(code.longstr())
    E = code.get_encoder()

    
    s = code.space
    CX, CZ, H, S, SWAP = s.CX, s.CZ, s.H, s.S, s.SWAP
    H0, H1, H2, H3 = [H(i) for i in range(4)]
    S0, S1, S2, S3 = [S(i) for i in range(4)]
    I = s.get_identity()

    gen = [CX(0,1),CX(1,0), H0, H1] #, S0, S1]
    G = mulclose(gen)
    #assert len(G)==720
    print("|G| =", len(G))

    g = H0 * CX(1,0) * CX(0,1) * H1
    E1 = E*g
    dode = QCode.from_encoder(E1, k=2)
    assert dode.is_equiv(code)



def test_422_encode():
    code = construct.get_422()
    print(code.longstr())

    s = code.space
    CX, CZ, H, S, SWAP = s.CX, s.CZ, s.H, s.S, s.SWAP
    H0, H1, H2, H3 = [H(i) for i in range(4)]
    S0, S1, S2, S3 = [S(i) for i in range(4)]
    I = s.get_identity()
    s4 = S0*S1*S2*S3
    assert s4*s4 == I

    E = code.get_encoder()

    def find_gate(tgt):
        idxs = list(range(4))
        gen = [CX(i,j) for i in idxs for j in idxs if i!=j]
        #gen += [CZ(i,j) for i in idxs for j in range(i+1,4)]
        #gen += [SWAP(0,1),SWAP(2,3)]
        gen += [H0,H1,H2,H3]
        g = mulclose_find(gen, tgt)
        return g

    #E1 = E*H0*H1
    #g = find_gate(E1)
    #print(g.name)

    g = CX(0,1)*CX(0,2)*CX(2,0)*CX(0,3)*CX(1,0)*CX(3,1)*CX(1,3)*H(1)
    assert E==g

    dode = QCode.from_encoder(g, k=2)
    assert code==dode

    #code = QCode.fromstr("XXXX ZZZZ", None, "XXII ZIZI IXIX IIZZ")
    #E = code.get_encoder()
    D = E.pseudo_inverse()
    g = H0*H1*H2*H3
    h = D*g*E
    assert g*E == E*h
    #h = find_gate(h)
    #print(h.name)
    assert h == H0*CX(1,0)*CX(0,1)*H1*SWAP(2,3)*H2*H3

    dode = code.apply(s4)
    assert dode.is_equiv(code)
    L = dode.get_logical(code)

    g = S0*S1*S2*S3
    dode = code.apply(g)
    assert dode.is_equiv(code)
    L = dode.get_logical(code)
    h = D*g*E
    assert g*E == E*h
    #print("h =", code.space.get_name(h))
    #return

    s2 = SymplecticSpace(2)
    gen = [s2.get_identity(), 
        s2.H(0), s2.H(1), s2.S(0), s2.S(1), 
        s2.CX(0,1), s2.CX(1,0), s2.CZ(), s2.get_perm([1,0])]
    G = mulclose(gen)
    G = list(G)
    assert len(G)==720
    idx = G.index(L)
    #print(G[idx].name)
    #print(dode.longstr())


    # --------------------------------------------
    # Clifford

    from qumba.clifford_sage import Clifford, red, green, Matrix
    P = code.get_projector()
    c = Clifford(code.n)
    CX, CY, CZ, H, S, SWAP = c.CX, c.CY, c.CZ, c.H, c.S, c.SWAP
    H0, H1, H2, H3 = [H(i) for i in range(4)]
    S0, S1, S2, S3 = [S(i) for i in range(4)]
    X0, X1, X2, X3 = [c.X(i) for i in range(4)]
    Z0, Z1, Z2, Z3 = [c.Z(i) for i in range(4)]
    I = c.get_identity()

    g = S(0)*S(1)
    h = S(2)*S(3)
    assert g.d==g*g*g
    for l in [g*h, g.d*h, g*h.d,g.d*h.d]:
        assert l*P == P*l

    E = CX(0,1)*CX(0,2)*CX(2,0)*CX(0,3)*CX(1,0)*CX(3,1)*CX(1,3)*H(1)

    # test the encoder...
    assert E*X0 == X0*X1*X2*X3*E
    assert E*Z0 == Z0*Z1*Z2*E
    assert E*X1 == Z0*Z1*Z2*Z3*E
    assert E*Z1 == X3*E
    assert E*X2 == X0*X1*E
    assert E*Z2 == Z0*Z2*E
    assert E*X3 == X0*X2*E
    assert E*Z3 == Z0*Z1*E

    # transversal hadamard -------------------
    g = H0*H1*H2*H3
    assert P*g == g*P
    h = H0*CX(1,0)*CX(0,1)*H1*SWAP(2,3)*H2*H3
    assert g*E == E*h

    #ops = mulclose([S1, H1])
    #print(len(ops))

    # transversal CZ -------------------
    for op in [H1*S1*H1, H1*S1.d*H1]:
        h = CZ(2,3) * op *CX(0,1)
        for i0,s0 in enumerate([S0, S0.d]):
         for i1,s1 in enumerate([S1, S1.d]):
          for i2,s2 in enumerate([S2, S2.d]):
           for i3,s3 in enumerate([S3, S3.d]):
            g = s0*s1*s2*s3
            if P*g!=g*P:
                continue
            #print('.',end='')
            if(g*E==E*h) :
                print("found!", op.name, i0, i1, i2, i3)

    # switch to computational basis
    E = E*H(0)*H(1)
    test_clifford_encoder(code, E)

    # test the encoder...
    assert E*Z0 == X0*X1*X2*X3*E
    assert E*X0 == Z0*Z1*Z2*E
    assert E*Z1 == Z0*Z1*Z2*Z3*E
    assert E*X1 == X3*E
    assert E*X2 == X0*X1*E
    assert E*Z2 == Z0*Z2*E
    assert E*X3 == X0*X2*E
    assert E*Z3 == Z0*Z1*E

    # transversal hadamard -------------------
    g = H0*H1*H2*H3
    assert P*g == g*P
    h = E.d*g*E
    assert g*E == E*h

    # transversal CZ -------------------
    for op in [H1*S1*H1, H1*S1.d*H1]:
        h = CZ(2,3) * op *CX(0,1)
        for i0,s0 in enumerate([S0, S0.d]):
         for i1,s1 in enumerate([S1, S1.d]):
          for i2,s2 in enumerate([S2, S2.d]):
           for i3,s3 in enumerate([S3, S3.d]):
            g = s0*s1*s2*s3
            if P*g!=g*P:
                continue
            #print('.',end='')
            if(g*E==E*h) :
                print("found!", op.name, i0, i1, i2, i3)

    



def test_822():
    src = QCode.fromstr("XYZI IXYZ ZIXY")
    print(src)

    code = unwrap(src)
    print(code)

    # note: code.get_autos does not get anything more than below ->

    E = code.get_encoder()
    D = code.get_decoder()
    space = code.space
    fibers = [(i, i+src.n) for i in range(src.n)]
    g = space.get_identity()
    for i,j in fibers:
        g = space.CZ(i, j) * g
    dode = code.apply(g)
    assert dode.is_equiv(code)

    gen = []
    gen.append(dode.get_logical(code))

    print("fibers:", fibers)

    def lift_perm(perm):
        assert len(perm) == src.n
        f = [fibers[perm[i]] for i in range(src.n)]
        #print(f)
        perm = {}
        for fsrc, ftgt in zip(fibers, f):
            perm[fsrc[0]] = ftgt[0]
            perm[fsrc[1]] = ftgt[1]
        perm = [perm[i] for i in range(code.n)]
        return perm

    I = SymplecticSpace(2).get_identity()

    perm = lift_perm([1,2,3,0])
    dode = code.apply_perm(perm)
    assert dode.is_equiv(code)
    gen.append(dode.get_logical(code))

    #perm = lift_perm([3,2,1,0]) # logical identity
    #perm = lift_perm([1,0,3,2]) # logical identity
    perm = lift_perm([0,3,2,1])
    #perm = lift_perm([2,1,0,3])
    dode = code.apply_perm(perm)
    # now we lift H.H.H.H gate
    perm = list(range(code.n))
    for f in fibers:
        perm[f[0]], perm[f[1]] = perm[f[1]], perm[f[0]]
    dode = dode.apply_perm(perm)
    assert dode.is_equiv(code)
    L = dode.get_logical(code)
    assert L in gen
    assert I != L

    def lift_S(code, fiber):
        return code.apply_CX(*fiber)
    def lift_SH(code, fiber):
        code = code.apply_swap(*fiber) # lift H
        code = code.apply_CX(*fiber) # lift S
        return code
    def lift_HS(code, fiber):
        code = code.apply_CX(*fiber) # lift S
        code = code.apply_swap(*fiber) # lift H
        return code
    def lift_SHS(code, fiber): # == HSH
        return code.apply_CX(fiber[1], fiber[0])
    lift_HSH = lift_SHS

    perm = lift_perm([1,0,2,3])
    dode = code.apply_perm(perm)
    dode = lift_SH(dode, fibers[0])
    dode = lift_HS(dode, fibers[1])
    dode = lift_S(dode, fibers[2])
    dode = lift_SHS(dode, fibers[3])
    assert dode.is_equiv(code)
    gen.append(dode.get_logical(code))

    perm = lift_perm([0,1,3,2])
    dode = code.apply_perm(perm)
    dode = lift_S(dode, fibers[0])
    dode = lift_SHS(dode, fibers[1])
    dode = lift_SH(dode, fibers[2])
    dode = lift_HS(dode, fibers[3])
    assert dode.is_equiv(code)
    gen.append(dode.get_logical(code))

    for perm, cliff in [
        ([0,2,1,3], "SHS SH HS S"),
        ([0,2,3,1], "HS S SHS SH"),
        ([0,3,1,2], "SH HS S SHS"),
        #([1,0,2,3], "SH HS S SHS"), # above
        ([1,2,0,3], "S SHS SH HS"),
        ([1,3,0,2], "HS S SHS SH"),
        ([1,3,2,0], "SHS SH HS S"),
        ([2,0,1,3], "HS S SHS SH"),
        ([2,0,3,1], "SHS SH HS S"),
        ([2,1,3,0], "SH HS S SHS"),
        ([2,3,1,0], "S SHS SH HS"),
        ([3,0,2,1], "S SHS SH HS"),
        ([3,1,0,2], "SHS SH HS S"),
        ([3,1,2,0], "HS S SHS SH"),
        ([3,2,0,1], "SH HS S SHS"),
    ]:
        #print(perm)
        perm = lift_perm(perm)
        dode = code.apply_perm(perm)
        cliff = cliff.split()
        for i, c in enumerate(cliff):
            func = eval("lift_%s"%c, locals())
            dode = func(dode, fibers[i])
        assert dode.is_equiv(code)
        gen.append(dode.get_logical(code))

    #for L in gen:
    #    print(L)

    G = mulclose(gen)
    print("G:", len(G))

    # --- Clifford -----

    from qumba.clifford_sage import Clifford, red, green, Matrix
    c = Clifford(code.n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    SWAP = c.SWAP
    P = c.get_P

    def lift_S(fiber):
        E = CX(*fiber)
        return E
    def lift_SH(fiber):
        E = SWAP(*fiber) # lift H
        E = CX(*fiber)*E # lift S
        return E
    def lift_HS(fiber):
        E = CX(*fiber) # lift S
        E = SWAP(*fiber)*E # lift H
        return E
    def lift_SHS(fiber): # == HSH
        E = CX(fiber[1], fiber[0])
        return E
    lift_HSH = lift_SHS

    perm = lift_perm([1,0,2,3])
    g = P(*perm)
    g = lift_SH(fibers[0])*g
    g = lift_HS(fibers[1])*g
    g = lift_S(fibers[2])*g
    g = lift_SHS(fibers[3])*g

    P = code.get_projector() # slow...
    #print(P.shape)

    assert P*g == g*P


def test_822_clifford():
    src = QCode.fromstr("XYZI IXYZ ZIXY")
    print(src)

    code = unwrap(src)
    print(code)

    fibers = [(i, i+src.n) for i in range(src.n)]
    print("fibers:", fibers)

    # --- Clifford -----

    from qumba.clifford_sage import Clifford, red, green, Matrix
    c = Clifford(code.n)
    CX, CY, CZ, H, S, SWAP = c.CX, c.CY, c.CZ, c.H, c.S, c.SWAP
    I = c.get_identity()

    P = code.get_projector() # slow...
    
    g = I
    for (i,j) in fibers:
        g = CZ(i,j)*g

    assert g*P != P*g

    g = I
    for (i,j) in fibers:
        g = SWAP(i,j)*g
    for i in range(c.n):
        g = H(i)*g

    assert g*P == P*g


def test_10_2_3_clifford():
    src = construct.get_513()
    code = unwrap(src)
    fibers = [(i, i+src.n) for i in range(src.n)]

    from qumba.clifford_sage import Clifford, red, green, Matrix
    c = Clifford(code.n)
    CX, CY, CZ, H, S, SWAP = c.CX, c.CY, c.CZ, c.H, c.S, c.SWAP
    I = c.get_identity()

    print("P...")
    P = code.get_projector() # slow...
    
    # Transversal CZ
    print("CZ...")
    g = I
    for (i,j) in fibers:
        g = CZ(i,j)*g
    assert g*P == P*g

    # Transversal HH SWAP
    print("HH SWAP...")
    g = I
    for (i,j) in fibers:
        g = SWAP(i,j)*g
    for i in range(c.n):
        g = H(i)*g
    assert g*P == P*g

    # Transversal CX SWAP
    print("CX SWAP...")
    def lift_SH(fiber):
        E = c.SWAP(*fiber) # lift H
        E = c.CX(*fiber)*E # lift S
        return E
    g = I
    for fiber in fibers:
        g = g*lift_SH(fiber)
    assert g*P == P*g




def test_422():
    code = construct.get_422()
    E = code.get_encoder()
    D = code.get_decoder()
    s = code.space
    I = s.get_identity()
    assert D*E == I
    
    code1 = QCode.from_encoder(E*s.CZ(2,3), k=2)
    #print(code1.longstr())
    assert code.is_equiv(code1)
    
    H4 = s.H(0)*s.H(1)*s.H(2)*s.H(3)
    code1 = code.apply(H4)
    assert code.is_equiv(code1)
    
    s2 = SymplecticSpace(2)
    gen = [s2.H(0), s2.H(1), s2.S(0), s2.S(1), s2.CX()]
    G = mulclose(gen)
    assert len(G)==720
    I2 = s2.get_identity()
    G = [I2.direct_sum(g) for g in G]
    
    found = []
    for g in G:
        code1 = QCode.from_encoder(E*g, k=2)
        assert code.is_equiv(code1)
        code2 = code1.apply(H4)
        L = code2.get_logical(code1)
        if L == s2.CZ():
            found.append(code1)
    assert len(found) == 48 # two kinds up to 4!==24 perms
    #for code in found:
    #    print()
    #    print(strop(code.L))

    code = QCode.fromstr(Hs="XXXX ZZZZ", Ls="ZIZI YYII ZZII YIYI")

    from qumba.clifford_sage import Clifford, red, green, Matrix
    #E = code.get_clifford_encoder()
    E = code.get_encoder()
    E = code.space.translate_clifford(E)
    #test_clifford_encoder(code, E) # FAIL

    D = E.d
    P = code.get_projector()
    c = Clifford(4)
    I = c.get_identity()
    assert D*E == I

    CZ, H = c.CZ, c.H
    g = CZ(2,3)
    h = H(0)*H(1)*H(2)*H(3)

    assert h*P == P*h

    return

    Pauli = c.pauli_group(2)
    print(len(Pauli))

#    lhs = E*g
#    rhs = h*E
    #lhs, rhs = g*E, E*h
    print(lhs == rhs)

    for p in Pauli:
        if lhs*p == rhs:
            print("found")
            break


def test_double():
    # ------------------------
    # double Sp(1) --> Sp(2)

    s = SymplecticSpace(1)
    S, H = s.S(), s.H()
    F = s.F
    G = [s.get_identity(), S, H, S*H, H*S, S*H*S]
    
    s2 = SymplecticSpace(2)
    cx, swap = s2.CX(), s2.get_perm([1,0])
    assert cx*swap*cx == swap*cx*swap
    G1 = [s2.get_identity(), cx, swap, cx*swap, swap*cx, cx*swap*cx]
    
    # convert to ziporder
    p = Matrix.perm([0,2,1,3])
    assert p == p.t # ooof

    for g, g1 in zip(G,G1):
        gi = F*g*F
        assert gi == g.pseudo_inverse().t
        ggi = g.direct_sum(gi)
        ggi = p.t * ggi * p
        assert s2.is_symplectic(ggi)
        assert ggi == g1

    # ------------------------
    # double Sp(2) --> Sp(4)

    s2 = SymplecticSpace(2)
    F = s2.F
    cx, swap = s2.CX(), s2.get_perm([1,0])
    h0, h1, s0, s1 = s2.H(0), s2.H(1), s2.S(0), s2.S(1)
    gen2 = [cx, swap, h0, h1, s0, s1]
    G2 = mulclose(gen2)
    assert len(G2) == 720

    s4 = SymplecticSpace(4)
    gen4 = [
        s4.CX(0,2)*s4.CX(3,1), 
        s4.get_perm([2,3,0,1]),
        s4.get_perm([1,0,2,3]),
        s4.get_perm([0,1,3,2]),
        s4.CX(0,1), 
        s4.CX(2,3)
    ]
    G4 = mulclose(gen4)
    assert len(G4) == 720

    # extend to hom
    hom = mulclose_hom(gen2, gen4)

    # convert to ziporder
    items = []
    for i in range(4):
        items.append(i)
        items.append(i+4)
    p = Matrix.perm(items)

    for g in G2:
        #print(g)
        gi = F*g*F
        assert gi == g.pseudo_inverse().t
        ggi = g.direct_sum(gi)
        ggi = p.t * ggi * p
        assert s4.is_symplectic(ggi)
        g4 = hom[g]
        assert ggi == g4
        #print(ggi)
        #print(g4)
        #print()



def test():
    print("\ntest()")
    get_422()
    test_isomorphism()
    test_symplectic()
    test_concatenate()
    test_10_2_3()
    test_codetables()
    #test_bring() # slow..
    test_grassl()
    test_normal_form()


if __name__ == "__main__":

    from time import time
    start_time = time()
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





