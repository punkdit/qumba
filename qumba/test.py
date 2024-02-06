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
from qumba.action import mulclose, mulclose_hom
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
    
    from qumba.util import cross
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


def get_encoder(code):
    from qumba.clifford_sage import Clifford, red, green, Matrix
    code = code.normal_form()
    n = code.n
    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S

    E = I = c.get_identity()
    for src,h in enumerate(code.H):
        op = strop(h).replace('.', 'I')
        ctrl = op[src]
        print(src, op, ctrl, end=":  ")
        if ctrl=='I':
            E0 = I
        elif ctrl in 'XZY':
            E0 = H(src)
            print("H(%d)"%src, end=" ")
        else:
            assert 0, ctrl
        
        for tgt,opi in enumerate(op):
            if tgt==src:
                continue
            if opi=='I':
                pass
            elif opi=='X':
                E0 = CX(src,tgt)*E0
                print("CX(%d,%d)"%(src,tgt), end=" ")
            elif opi=='Z':
                E0 = CZ(src,tgt)*E0
                print("CZ(%d,%d)"%(src,tgt), end=" ")
            elif opi=='Y':
                E0 = CY(src,tgt)*E0
                print("CY(%d,%d)"%(src,tgt), end=" ")
            else:
                assert 0, opi

        if ctrl in 'XI':
            pass
        elif ctrl == 'Z':
            E0 = H(src)*E0
            print("H(%d)"%src, end=" ")
        elif ctrl == 'Y':
            E0 = S(src)*E0
            print("S(%d)"%src, end=" ")
        else:
            assert 0, ctrl
        print()
        E = E * E0
    return E


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

    # 822 toric code
    code = QCode.fromstr("""
    XXX..X..
    X.XX...X
    .X..XXX.
    ZZ.ZZ...
    .ZZZ..Z.
    Z...ZZ.Z
    """)
    E = get_encoder(code)
    test_clifford_encoder(code, E)

    code = QCode.fromstr("XZXX IXIZ ZIZI") # works
    E = get_encoder(code)
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
        E = get_encoder(code)
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
        E = get_encoder(code)
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





