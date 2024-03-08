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
from qumba.action import Perm, mulclose_find, mulclose
from qumba import csscode 
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


def get_fibers(duality):
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


def test():
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
    pairs = get_fibers(zx)
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


class Cover(object):
    def __init__(self, base, total, fibers=None, zx=None):
        if fibers is None:
            fibers = get_fibers(zx)
        self.base = base
        self.total = total
        self.fibers = fibers
        assert len(fibers) == base.n
        for (i,j) in fibers:
            assert i!=j
            assert 0<=i<total.n
            assert 0<=j<total.n

    @classmethod
    def frombase(cls, base):
        code = unwrap(base)
        n = base.n
        fibers = [(i,i+n) for i in range(n)]
        return Cover(base, code, fibers)

    @classmethod
    def fromzx(cls, total, zx):
        base = wrap(total, zx)
        fibers = get_fibers(zx)
        return Cover(base, total, fibers)

    def H(self, idx):
        base, total, fibers = self.base, self.total, self.fibers
        fiber = fibers[idx]
        return total.space.SWAP(*fiber)

    def S(self, idx):
        base, total, fibers = self.base, self.total, self.fibers
        fiber = fibers[idx]
        return total.space.CX(*fiber)

    #def CX(self, idx, jdx):

    def get_ZX(self):
        # Transversal HH SWAP
        base, total, fibers = self.base, self.total, self.fibers
        space = total.space
        g = space.get_identity()
        for (i,j) in fibers:
            g = space.SWAP(i,j)*g
        for i in range(total.n):
            g = space.H(i)*g
        dode = total.apply(g)
        assert dode.is_equiv(total)
        return g

    def get_CZ(self):
        base, total, fibers = self.base, self.total, self.fibers
        space = total.space
        assert 'Y' not in strop(base.H)
        g = space.get_identity()
        for (i,j) in fibers:
            g = space.CZ(i,j)*g
        dode = total.apply(g)
        assert dode.is_equiv(total)
        return g

    def lift(self, M):
        "lift a symplectic on the base code to a symplectic on the total code"
        base, total, fibers = self.base, self.total, self.fibers
        nn = base.nn
        assert M.shape == (nn, nn)
        assert base.apply(M).is_equiv(base)
        F = base.space.F
        I = base.space.get_identity()
        Mi = F*M*F
        assert M*Mi.t == I
        MM = M.direct_sum(Mi) # this is block direct sum
        perm = []
        for i in range(nn):
            perm.append(i)
            perm.append(i+nn)
        P = Matrix.get_perm(perm)
        MM = P.t*MM*P # switch to ziporder
        assert total.space.is_symplectic(MM)
        idxs = []
        for (i,j) in fibers:
            idxs.append(i)
            idxs.append(j)
        P = total.space.get_perm(idxs)
        PMM = P.t*MM*P # switch to fiber order
        assert total.space.is_symplectic(PMM)
        assert total.apply(PMM).is_equiv(total)
        return PMM


def test_gaussian():
    N = argv.get("N", 8)
    for a in range(1,N):
      print(a, end=" ")
      for b in range(0,a+1):
      #for b in range(1,8):
        if (a+b) <= 1:
            continue
        if (a+b)%2:
            code = construct.get_xzzx(a, b)
        else:
            code = construct.get_toric(a, b)
        code.distance("z3")
        print(code, end=" ", flush=True)
      print()


def test_bring():
    total = construct.get_bring()
    total.distance()
    print(total)
    zxs = total.find_zx_dualities()
    print(len(zxs))

    total = total.to_qcode()

    codes = []
    for zx in zxs:
        cover = Cover.fromzx(total, zx)
        base = cover.base
        base.distance("z3")
        print(base)
        #print(strop(base.H))
        codes.append(base)

#    for a in codes:
#      for b in codes:
#        print(int(is_iso(a,b)), end=" ", flush=True)
#      print()




def test_wrap():
    N = argv.get("N", 8)
    for a in range(1,N):
        b = 0
      #for b in range(0,a+1):
      #for b in range(1,8):
        if (a+b) <= 2:
            continue
        if (a+b)%2:
            code = construct.get_xzzx(a, b)
        else:
            code = construct.get_toric(a, b)
        #if code.n > 20:
            #continue
        if (a+b)%2:
            continue
        code.distance("z3")
        print(a, b, code)
        zxs = code.find_zx_dualities()
        print("zxs:", len(zxs))
        for zx in zxs:
            dode = wrap(code, zx)
            dode.distance("z3")
            print(dode, "Y's:", strop(dode.H).count("Y"))
        print()
        print()


def find_perm_local_cliffords(code):
    from qumba.action import Group
    import transversal
    n = code.n
    space = code.space
    assert n<6, "um.."
    G = Group.symmetric(n)
    for g in G:
        perm = [g[i] for i in range(n)]
        dode = code.apply_perm(perm)
        P = space.get_perm(perm)
        for Q in transversal.find_local_clifford(code, dode):
            QP = Q*P
            assert code.apply(QP).is_equiv(code)
            #print(space.get_name(QP))
            yield QP


def test_unwrap_cliffords():
    import transversal
    #total = unwrap(construct.get_412())
    total = unwrap(construct.get_513()) # [[10,2,3]]
    #total = construct.get_toric(2,2)
    #total = construct.get_toric(1,3) # [[10,2,3]]
    #total = construct.get_toric(4,0) # too big
    print(total.longstr())

    zxs = get_zx_wrap(total)
    print("zx _dualities:", len(zxs))

    logops = set()
    covers = []
    for zx in zxs:
        base = wrap(total, zx)
        cover = Cover(base, total, None, zx)
        covers.append(cover)
        print()
        print(zx)
        print(base)
        #if base.d == 3:
        #    continue
        #print(base.longstr())
        A = base.H
        A = A.concatenate(A.sum(0).reshape(1,A.shape[1]))
        print(strop(A))
        gens = []
        for M in transversal.find_local_clifford(base, base):
            gens.append(M)
        print("local cliffords:", len(gens))
        gens = list(get_isos(base,base))
        print("autos:", len(gens))
        gens = list(find_perm_local_cliffords(base))
        print("perm local cliffords:", len(gens))
        gens = transversal.search_gate(base, base, row_weight=3)
        #gens = list(transversal.search_gate(base, base, row_weight=2))
        #print("row_weight 2 cliffords:", len(gens))
        for g in gens:
            g1 = cover.lift(g)
            tgt = total.apply(g1)
            assert tgt.is_equiv(total)
            l = tgt.get_logical(total)
            if l not in logops:
                logops.add(l)
                print(l)
                G = mulclose(logops)
                print("|G| =", len(G))
                #if len(G) >= 6:
                #    break
        g = cover.get_ZX()
        tgt = total.apply(g)
        assert tgt.is_equiv(total)
        l = tgt.get_logical(total)
        logops.add(l)
        if 'Y' in strop(base.H):
            continue
        g = cover.get_CZ()
        tgt = total.apply(g)
        assert tgt.is_equiv(total)
        l = tgt.get_logical(total)
        logops.add(l)
        
    #print("found logops:", len(logops))
    #logops = list(set(logops))
    print("uniq logops:", len(logops))
    G = mulclose(logops)
    print("|G| =", len(G))

    for a in covers:
      for b in covers:
        print(int(is_iso(a.base,b.base)), end=" ")
      print()


def test_wrap_16():
    import transversal
    code = construct.get_toric(4)
    space = code.space
    print(code)

    g = Matrix.parse("""
    1...............................
    .1..............................
    ..1.............................
    ...1............................
    ....1...........................
    .....1..........................
    ......1.........................
    .......1........................
    ........1.......................
    .........1.1...1................
    ........1.1.1...................
    ...........1....................
    ............1...................
    ...........1.1.1................
    ........1...1.1.................
    ...............1................
    ................1...............
    .................1..............
    ..................1.............
    ...................1............
    ....................1...........
    .....................1..........
    ......................1.........
    .......................1........
    ........................1.......
    .........................1.1...1
    ........................1.1.1...
    ...........................1....
    ............................1...
    ...........................1.1.1
    ........................1...1.1.
    ...............................1
    """) # 32x32


    rows = cols = 4
    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    perm = space.get_perm(perm)
    g = g*perm
    print(g)
    dode = code.apply(g)
    assert dode.is_equiv(code)
    l = dode.get_logical(code)
    print(l)
    #print(SymplecticSpace(2).get_name(l))
    #return

    if 0:
        n = 4
        CX = SymplecticSpace(n).CX
        SWAP = SymplecticSpace(n).SWAP
        gen = [SWAP(i,j) for i in range(n) for j in range(i+1,n)]
        gen += [CX(i,j) for i in range(n) for j in range(n) if i!=j]
        h = mulclose_find(gen, Matrix.parse("""
        ..1.....
        .1.1.1..
        ..1.1.1.
        .....1..
        ......1.
        .1...1.1
        1.1...1.
        .1......
        """))
        print("h =", h.name)
        
        return

    zxs = get_zx_wrap(code)
    print("zxs:", len(zxs))

    covers = [Cover.fromzx(code, zx) for zx in zxs]
    bases = []
    for cover in covers:
        fibers = cover.fibers
        idxs = []
        for (i,j) in fibers:
            idxs.append(2*i)
            idxs.append(2*i+1)
        #print(idxs)
        g1 = g[idxs,:][:,idxs]
        base = cover.base
        bases.append(base)
        if base.space.is_symplectic(g1):
            dode = base.apply(g1)
            if dode.is_equiv(base):
                print(fibers)
                dode.distance()
                print(strop(dode.H), dode)
                print(g1)
                print("L =")
                print(dode.get_logical(base))
                print()
        #else:
        #    print("not symplectic")

    print("is_iso")
    for a in bases:
      for b in bases:
        print(int(is_iso(a,b)), end=" ")
      print()

    return


    codes = []
    for zx in zxs:
        eode = wrap(code, zx)
        codes.append(eode)
        print()
        print(zx)
        print(eode)
        #print(eode.longstr())
        print(strop(eode.H))
        total = unwrap(eode)
        gens = []
        for M in transversal.find_local_clifford(eode, eode):
            gens.append(M)
        print("local cliffords:", len(gens))
        gens = list(get_isos(eode,eode))
        print("autos:", len(gens))

    for a in codes:
      for b in codes:
        print(int(is_iso(a,b)), end=" ")
      print()


def test_dehn():
    rows = cols = argv.get("rows", 4)
    code = construct.get_toric(rows)
    print(code)
    print(strop(code.H))
    print()

    space = code.space
    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    perm = space.get_perm(perm)

    CX = space.CX

    g = space.get_identity()
    #for i in [4,12]:
    for i in range(rows):
        if i%2==0:
            continue
        idx = i*cols
        #g *= CX(j+0,j+1)*CX(j+0,j+3)*CX(j+2,j+1)*CX(j+2,j+3)
        for j in range(cols):
            src, tgt = (idx+j, idx+(j+1)%cols)
            if j%2==0:
                g *= CX(src, tgt)
            else:
                g *= CX(tgt, src)
    #print(g)
    g = g*perm

    dode = code.apply(g)
    assert code.is_equiv(dode)
    l = code.get_logical(dode)
    #print(SymplecticSpace(2).get_name(l))
    #print(l*l)

    zxs = get_zx_wrap(code)
    print("zxs:", len(zxs))

    covers = [Cover.fromzx(code, zx) for zx in zxs]
    for cover in covers:
        perm = [None]*code.n
        for (i,j) in cover.fibers:
            perm[i] = j
            perm[j] = i
        perm = code.space.get_perm(perm)
        if g*perm == perm*g:
            print("found cover:", cover.fibers)

    #return

    bases = []
    for cover in covers:
        fibers = cover.fibers
        idxs = []
        for (i,j) in fibers:
            idxs.append(2*i)
            idxs.append(2*i+1)
        #print(idxs)
        g1 = g[idxs,:][:,idxs]
        base = cover.base
        #bases.append(base)
        if not base.space.is_symplectic(g1):
            continue
        dode = base.apply(g1)
        if not dode.is_equiv(base):
            continue

        print("lift:", cover.lift(g1) == g)
        print(fibers)
        dode.distance()
        print(strop(dode.H), dode)
        print(g1)
        print("L =")
        print(dode.get_logical(base))
        print()
        bases.append(base)

    print("is_iso")
    for a in bases:
      for b in bases:
        print(int(is_iso(a,b)), end=" ")
      print()


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





