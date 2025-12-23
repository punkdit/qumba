#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle, choice
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul


import numpy

from qumba.lin import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, zeros2, solve2, normal_form)
from qumba.matrix import Matrix
from qumba.qcode import QCode, SymplecticSpace, strop
from qumba.csscode import CSSCode
from qumba import construct
from qumba.distance import distance_z3
from qumba.autos import get_isos, is_iso
from qumba.action import Perm, mulclose_find, mulclose
#from qumba import autos, transversal
from qumba import csscode 
from qumba.argv import argv
from qumba.gcolor import dump_transverse
from qumba.util import choose


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


def wrap(code, zx=None, fibers=None):
    nn = code.nn
    if fibers is None:
        fibers = get_fibers(zx)
    #print(code.H)
    Hx = code.H[:,0:nn:2]
    #print(Hx)
    cols = reduce(add, fibers)
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

    # protocol methods ---------------------------------------
    def X(self, idx):
        return self.total.space.get_identity()
    Y = X
    Z = X

    def CX(self, idx, jdx):
        assert idx != jdx
        CX = self.total.space.CX
        fibers = self.fibers
        i, j = fibers[idx], fibers[jdx]
        op = CX(i[0], j[0]) * CX(j[1], i[1])
        return op

    def CZ(self, idx, jdx):
        assert idx != jdx
        CX = self.total.space.CX
        fibers = self.fibers
        i, j = fibers[idx], fibers[jdx]
        op = CX(i[0], j[1]) * CX(j[1], i[0])
        return op

    def CY(self, idx, jdx):
        assert idx != jdx
        CX = self.total.space.CX
        fibers = self.fibers
        i, j = fibers[idx], fibers[jdx]
        op = CX(j[0], j[1]) * CX(i[0], j[0]) * CX(j[1], i[1]) * CX(j[0], j[1])
        return op

    def H(self, idx):
        base, total, fibers = self.base, self.total, self.fibers
        fiber = fibers[idx]
        return total.space.SWAP(*fiber)

    def S(self, idx):
        base, total, fibers = self.base, self.total, self.fibers
        fiber = fibers[idx]
        return total.space.CX(*fiber)

    def P(self, *perm):
        base, total, fibers = self.base, self.total, self.fibers
        assert len(perm) == len(fibers)
        fibers = [fibers[i] for i in perm]
        perm = reduce(add, fibers)
        return total.space.P(*perm)

    def get_expr(self, name):
        #print("Cover.get_expr", name)
        op = self.total.space.get_identity()
        for item in name:
            if item.endswith(".d"):
                assert item[0] == "S"
                item = item.replace(".d", "")
            e = "self."+item
            op = op * eval(e, {"self":self})
        return op

    # end protocol methods -------------------------------------

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
        #assert base.apply(M).is_equiv(base)
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
        #assert total.apply(PMM).is_equiv(total)
        return PMM

    def descend(self, g):
        "descend a symplectic on the total code to the base code"
        nn = self.total.nn
        assert g.shape == (nn,nn)
        fibers = self.fibers
        idxs = []
        for (i,j) in fibers:
            idxs.append(2*i)
            idxs.append(2*i+1)
        #print(idxs)
        g1 = g[idxs,:][:,idxs]
        base = self.base
        nn = base.nn
        assert g1.shape == (nn,nn)
        #bases.append(base)
        if not base.space.is_symplectic(g1):
            return None
        dode = base.apply(g1)
        if not dode.is_equiv(base):
            return None
        return g1



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


def get_avoid(code):
    css = code.to_css()
    Hx = css.Hx
    Lx = css.Lx
    mx = len(Hx)
    k = len(Lx)

    avoid = []
    for idx in numpy.ndindex((2,)*k):
        idx = numpy.array(idx)
        if idx.sum()==0:
            continue
        l = dot2(idx, Lx)
        for jdx in numpy.ndindex((2,)*mx):
            jdx = numpy.array(jdx)
            lh = (l + dot2(jdx, Hx)) % 2
            if lh.sum() == code.d:
                avoid.append(lh)

    #print("avoid:", len(avoid))
    return avoid


def test_zx():
    from qumba import db
    from qumba import autos, transversal
    from qumba.action import Perm, Group, mulclose
    code = list(db.get(_id="67a4b0119edf81e4b7e670ac"))[0]
    print(code.longstr())
    css = code.to_css()

    space = code.space
    n = code.n
    items = list(range(n))

    dode = space.H() * code
    #print(dode.longstr())

    #tau = dode.get_isomorphism(code)
    tau = [0, 1, 3, 2, 10, 6, 5, 7, 14, 11, 4, 9, 13, 12, 8]
    tau = Perm(tau, items)

    #css = code.to_css()
    #print(css)
    #G = autos.get_autos_css(css) # fails..
    #print("|G| =", len(G))

    N, perms = code.get_autos()
    perms = [Perm(perm,items) for perm in perms]
    G = mulclose(perms)
    assert len(G) == N

    logops = set()

    for g in G:

        if 0:
            # automorphism gate, these don't help any
            lop = space.get_perm([g[i] for i in items])
            dode = lop*code
            assert dode.is_equiv(code)
            L = dode.get_logical(code)
            logops.add(L)

        f = g*tau
        perm = [f[i] for i in items]

        # H-type gate
        lop = space.get_perm(perm)
        lop = space.H() * lop
        dode = lop*code
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        logops.add(L)

        s = f.fixed()
        if not (f*f).is_identity():
            continue

        # S-type gate
        pairs = [(i,f(i)) for i in range(n) if i<f(i)]
        succ = " "
        #for lh in avoid:
        #    for (i,j) in pairs:
        #        if lh[i] and lh[j]:
        #            succ = "X"
        print(pairs, s, succ)

        lop = reduce(mul, [space.CZ(i,j) for (i,j) in pairs])
        lop = lop * reduce(mul, [space.S(i) for i in s])
        dode = lop*code
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        logops.add(L)

    #logops = mulclose(logops, verbose=True)
    print("logops:", len(logops))

    f = open("bring.gap", 'w')
    vs = []
    for i,L in enumerate(logops):
        m = "M%d"%i
        vs.append(m)
        print("%s := %s;"%(m, L.gap()), file=f)
    print("G := Group([%s]);"%(','.join(vs)), file=f)




    #zxs = total.find_zx_dualities()
    #print("involutory fixed-point free zx-dualities:", len(zxs))


def test_fold():

    n = 24

    # [[24, 4, 4]]
    # https://qecdb.org/codes/6900a99d0a060a5626b32e6b
    s = ("""
    IIIIXIIXIIIXIIIXXXIIIIII
    IIIXXIIIIIIIIXIXIIIIXIXI
    XIIIIIIIIIXIIXIIIIXIIIXX
    IIIIIIIIXIIIIIXIIIXXIXIX
    XIIIIIXXIIXIXIIIXIIIIIII
    IIXIIIIIIIIXIXXXIIIIIIIX
    IIIXXIXIIIIIIIIIXIIXIXII
    IXXIIXIIIXIXIIIIIXIIIIII
    XIIIIXIXXIIIIIIIIXXIIIII
    IIXIIIXIIXIIXIXIIIIIIXII
    """).replace("I", ".").replace("X", "1")
    H = Matrix.parse(s)
    print(H.sum(1))
    H = H.normal_form()
    print(H, H.shape)
    m = len(H)
    print()

#    A = H.A
#    for i in range(m):
#      for j in range(m):
#        print((A[i]*A[j]).sum(), end=' ')
#      print()
#    return

    J = []
    for v in H.rowspan():
        if v.sum() == 8:
            J.append(v[0])
    J = Matrix(J)
    print(J.shape)
    J = J.normal_form()
    print(J, J.shape)
    print(J.sum(1))
    print(J.rank())

    css = CSSCode(Hx=H, Hz=H)
    print(css)
    print(css.longstr())
    Hx = css.Hx

    Hx = css.Hx.A
    Lx = css.Lx
    print(Hx, type(Hx))

    m, n = Hx.shape
    for i in range(m):
      for j in range(m):
        print((Hx[i]*Hx[j]).sum(), end=' ')
      print()

    return

    dump_transverse(Hx, Lx)
    dump_transverse(Hx, Lx)


def send_to_golay(J, H_G=None):
    n = 24

    if H_G is None:
        code = construct.get_golay()
        css = code.to_css()
        H_G = Matrix(css.Hx)

    #print()
    #print("golay:")
    #print(H_G)

    from qumba.umatrix import UMatrix, Solver, And, Or
    solver = Solver()
    Add = solver.add

    # permutation matrix
    P = UMatrix.unknown(n, n)
    
    for i in range(n):
      for j in range(n):
        rhs = reduce(And, [P[i,k]==0 for k in range(n) if k!=j])
        Add( Or(P[i,j]==0, rhs) )
        rhs = reduce(And, [P[k,j]==0 for k in range(n) if k!=i])
        Add( Or(P[i,j]==0, rhs) )

    for i in range(n):
        Add( reduce(Or, [P[i,j]!=0 for j in range(n)]) )

    Add(J*P*H_G.t == 0)
    #Add(H*P*H_G.t == 0) # unsat

    result = solver.check()
    result = str(result)
    if result == "sat":
        model = solver.model()
        P1 = P.get_interp(model)
        return P1

    print(result)


def find_stabilizer(gen, H):
    print("find_stabilizer")
    for g in gen:
        print(g)
    print(H, H.shape)

    rows = [list(row) for row in list(H.A)]
    s = str(rows).replace(" ", "")
    print(s)



def bruhat_golay():
    from bruhat.golay import Vec, infty, m24_gen, bricks

    brick = bricks[0]

    gen = m24_gen() # M24 group generators, agrees with gap
    #for g in gen:
    #    print(g)
    #return

    octad = Vec(infty,19,15,5,11,1,22,2)
    octads = {octad}
    bdy = list(octads)
    while bdy:
        _bdy = []
        for g in gen:
            for octad in bdy:
                o = g * octad
                if o not in octads:
                    octads.add(o)
                    _bdy.append(o)
        bdy = _bdy
    assert len(octads) == 759
    octads = list(octads)
    octads.sort()
    #for octad in octads:
    #    print(octad)
    found = set(octads)
    #send = {v:v for v in octads}

    rows = []
    for v in octads:
        rows.append(v.v)
    H = Matrix(rows)
    #H = H.linear_independent()
    H = H.row_reduce()
    print(H, H.shape)
    H_G = H # Golay

    # https://qecdb.org/codes/67a38c449d65c7b4098268cb 
    # [[24,8,4]]
    s = ("""
    IIIIIIIXXIXIXIIIIXXIIIXX
    IIIIXXXIIIIIIXXIIXIXIIXI
    IIIXIIIIXIXIIXXIXIIIXXII
    IIXIXIXIIIIIIIIXIIXIXXIX
    IXIXIIIIIIIXIIXIXIIIIXXX
    IXIXIIIIIIXIXIIXIXXXIIII
    XIIIIXIIIXIIIXIIIIXXIXXI
    XIXIIIXIIIIIXXIIXIIXIIXI
    """).replace("I", ".").replace("X", "1")
    H = Matrix.parse(s)
    H = H.normal_form()
    print(H, H.shape)
    print(H.get_wenum())
    m, n = H.shape

    P = send_to_golay(H, H_G)
    idxs = P.to_perm()
    print(idxs)
    H = H*P

    find_stabilizer(gen, H)

    return

    vecs = []
    for v in H.rowspan():
        if v.sum() != 8:
            continue
        idxs = [i for i in range(n) if v[0,i]]
        vec = Vec(*idxs)
        assert vec in found
        vecs.append(vec)

    vec = vecs[0]
    orbit = set([vec])
    bdy = list(orbit)
    word = {vec:()}
    while bdy:
        _bdy = []
        for v in bdy:
            for g in gen:
                v1 = g*v
                if v1 in orbit:
                    continue
                word[v1] = (g,)+word[v]
                orbit.add(v1)
                _bdy.append(v1)
        bdy = _bdy
        print(len(bdy))
    print("orbit:", len(orbit), len(word))
    assert brick in word
    g = reduce(mul, word[brick])
    print(g)

    vecs = [g*v for v in vecs]
    vecs.sort(key = lambda v : (v*brick).sum())

    from huygens.namespace import Canvas
    cvs = Canvas()

    count = 0
    x = y = 0
    for vec in vecs:
        cvs.insert(x, y, vec.render(2.0,1.2))

        count += 1
        if count%8==0:
            y -= 2.0
            x = 0
        else:
            x += 3.0

    cvs.writePDFfile("output_bricks.pdf")
    print("output_bricks.pdf")



def test_24_16_4():
    # this is another [24,16,4] self-orthogonal code from
    # https://arxiv.org/abs/2008.05051
    # Short Shor-style syndrome sequences Nicolas Delfosse, Ben W. Reichardt
    dot = chr(183)
    s = """
    1 1 1 1 · · · · · · · · · · · · · · · · · · · ·
    · · · · 1 1 1 1 · · · · · · · · · · · · · · · ·
    · · · · · · · · 1 1 1 1 · · · · · · · · · · · ·
    · · · · · · · · · · · · 1 1 1 1 · · · · · · · ·
    · · · · · · · · · · · · · · · · 1 1 1 1 · · · ·
    · · · · · · · · · · · · · · · · · · · · 1 1 1 1
    · · 1 1 · · 1 1 · · 1 1 · · 1 1 · · 1 1 · · 1 1
    · 1 · 1 · 1 · 1 · 1 · 1 · 1 · 1 · 1 · 1 · 1 · 1
    """.replace(" ", "").replace(dot, ".")
    print(s)
    H = Matrix.parse(s)
    print(H)
    print(H.get_wenum()) 
    # (1, 0, 0, 0, 6, 0, 0, 0, 15, 0, 0, 0, 212, 0, 0, 0, 15, 0, 0, 0, 6, 0, 0, 0, 1)


def test_w8():
    from qumba import autos, transversal

#    H = Matrix.parse("""
#    1.......1.1..11....1.11.
#    .1......1.11..1.1.1..1..
#    ..1.....1..1.1.1..11.1..
#    ...1.......111.11111.111
#    ....1...1..11.1.11..1...
#    .....1..1.1.11.1.111111.
#    ......1......1.11..1111.
#    .......1....1..1..1111.1
#    .........1.11...1..11.11 
#    """)
#    print(H.get_wenum())
#
#    code = QCode.build_css(H,H)
#    print(code)

    # https://qecdb.org/codes/67a38c449d65c7b4098268cb 
    # [[24,8,4]]
    s = ("""
    IIIIIIIXXIXIXIIIIXXIIIXX
    IIIIXXXIIIIIIXXIIXIXIIXI
    IIIXIIIIXIXIIXXIXIIIXXII
    IIXIXIXIIIIIIIIXIIXIXXIX
    IXIXIIIIIIIXIIXIXIIIIXXX
    IXIXIIIIIIXIXIIXIXXXIIII
    XIIIIXIIIXIIIXIIIIXXIXXI
    XIXIIIXIIIIIXXIIXIIXIIXI
    """).replace("I", ".").replace("X", "1")
    H = Matrix.parse(s)
    H = H.normal_form()
    print(H, H.shape)
    print(H.get_wenum())

    code = QCode.build_css(H, H)
    print(code) # Autos: S6xS4 = 17280

    P = send_to_golay(H)
    #print(P)
    print(P.to_perm())
    # [18, 14, 23, 6, 9, 11, 20, 17, 3, 5, 2, 19, 7, 0, 10, 13, 12, 21, 8, 15, 16, 4, 1, 22]

    H = H*P

    code = construct.get_golay()
    css = code.to_css()
    H_G = Matrix(css.Hx)

    print("solve:")
    print(H_G.t.solve(H.t))
    #from qumba.gcolor import dump_transverse
    #dump_transverse(css.Hx, css.Lx)

    if 0:
        #perms = css.find_autos()
        gen = []
        for g in transversal.find_isomorphisms_css(code):
            gen.append(g)
            print(".", flush=True, end="")
            if len(gen) > 3:
                break
        print()

        from bruhat.gset import Group, Perm
        perms = []
        for g in gen:
            perm = g.to_perm()
            perm = Perm(perm)
            print(perm)
            perms.append(perm)
    
        G = Group.generate(perms, verbose=True)
        print(G)
        print(G.gapstr())
        print(G.structure_description())
    
        return


def test_golay():

    code = construct.get_golay()
    css = code.to_css()
    H0 = Matrix(css.Hx)

    print(H0, H0.shape)
    m,n = H0.shape

    while 1:
        J = Matrix.rand(m,m)
        H1 = J*H0
        if H1.rank() == m:
            break
        print(H1.rank())

    print(H1)
    print()


    H = H1[:m-1]
    print(H, H.shape)

    code = QCode.build_css(H,H)
    print(code)


@cache
def getbits(dim):
    return list(numpy.ndindex((2,)*dim))


def getops(lookup, dim, sub):
    n = 2**dim
    ops = []
    coords = list(range(dim))
    for idxs in choose(coords, sub):
        jdxs = tuple(j for j in coords if j not in idxs)
        #print(idxs, jdxs)
        for cits in getbits(dim-sub):
            op = [0]*n
            src = [0]*dim
            for (i,bit) in zip(jdxs,cits):
                src[i] = bit
            #print(cits, src)
            for bits in getbits(sub):
                #dest = [0]*dim
                dest = list(src)
                for (i,bit) in zip(idxs,bits):
                    dest[i] = bit
                dest = tuple(dest)
                op[lookup[dest]] = 1
                #print("\t", dest)
            #print("\t", op)
            ops.append(op)
        #print()
    ops = Matrix(ops)
    return ops
    
    
def build_cube(dim, xdim=3, zdim=2):
    assert 0 <= xdim <= dim
    assert 0 <= zdim <= dim
    verts = getbits(dim)
    n = len(verts)
    assert n==2**dim
    lookup = {bits:i for (i,bits) in enumerate(verts)}

    Hx = getops(lookup, dim, xdim)
    Hx = Hx.linear_independent()
    Hz = getops(lookup, dim, zdim)
    Hz = Hz.linear_independent()

    if (Hx*Hz.t).max()>0:
        return None

    css = CSSCode(Hx=Hx.A, Hz=Hz.A)
    if css.k>0:
        css.bz_distance()
    return css


    


def test_cube():
    # Only the (n,n,2) cubes give Clifford hierarchy
#    dim = 5
#    for xdim in [5,4,3,2]:
#      for zdim in range(1,xdim+1):

    for (dim,xdim,zdim) in [
        (2, 2, 2), # CZ  [[4,2,2]]
        (2, 2, 1), # k=0
        (3, 3, 3), # CZ  
        (3, 3, 2), # CCZ [[8,3,(4,2)]]
        (3, 3, 1), # k=0
        (3, 2, 2), # k=0
        (3, 2, 1), # None
        (4, 4, 4), # CZ
        (4, 4, 3), # CZ
        (4, 4, 2), # CCCZ [[16,4,(8,2)]]
        (4, 3, 3), # CZ
        #(4, 3, 2), # k=0
        #(4, 2, 2), # None
        (5, 5, 2),
    ]:

        if xdim!=dim or zdim!=2:
            continue
        css = build_cube(dim,xdim,zdim)
        print((dim,xdim,zdim), css)
        if css is None or css.k==0:
            continue
        dump_transverse(css.Hx, css.Lx, 5)




def test_logical():
    from qumba import autos, transversal
    from qumba import db
    from qumba.action import Perm, Group, mulclose

    _id = argv._id
    n = argv.n
    if n==4:
        code = construct.get_422()
    elif n==8:
        code = construct.get_832()
    elif n==16:
        code = construct.get_16_4_2()
    elif _id:
        print("fetching https://qecdb.org/codes/%s" % _id)
        code = list(db.get(_id=_id))[0]


    else:
    
        #code = construct.get_10_2_3()
        #  logicals: 12
    
        #code = list(db.get(_id="6900a99d0a060a5626b32e6a"))[0] # [[15,5,3]] SD
        # logicals: O(9,2) x S3 == Sp(8,2)xS3
    
        #code = list(db.get(_id="67a4b0119edf81e4b7e670ac"))[0] # [[15,5,3]] 5,5 surface
        # logicals: O(9,2) x S3 == Sp(8,2)xS3
    
        code = list(db.get(_id="6900a99d0a060a5626b32e6b"))[0] # [[24,4,4]]
        # logicals: O(9,2) == Sp(8,2), FT logicals: 432
    
        #code = list(db.get(_id="67a4b0129edf81e4b7e670ad"))[0] # [[30,8,3]]
    
        #code = list(db.get(_id="6705236219cca60cf657a938"))[0] # [[21,3,5]]
        # 120960 autos, cyclic, logicals: 4320, FT logicals: 12
    
        #code = list(db.get(_id="67476b7f44a05d87e042051b"))[0] # [[21,3,5]]
        # 5760 autos, logicals: 4320, FT logicals: 36
    
        #code = list(db.get(_id="672e583d4d73ae0fe405d860"))[0] # [[23,3,5]]
        # logicals: 4320, FT logicals: 36
    
        #code = list(db.get(_id="67a4c83cfab4d3b49d2d29c1"))[0] # [[21,9,3]]
        # logicals: more than Sp(16,2)

    print(code)
    print(code.longstr())
    css = code.to_css()
    Hx = Matrix(css.Hx)
    Hz = Matrix(css.Hz)
    print("Hx.wenum:", Hx.get_wenum())
    print("Hz.wenum:", Hz.get_wenum())

    space = code.space
    n = code.n
    items = list(range(n))

    #perms = css.find_autos()
    gen = []
    for g in transversal.find_isomorphisms_css(code):
        gen.append(g)
        print(".", flush=True, end="")
        if len(gen) > 3:
            break
    print()

    if argv.autos:
        from bruhat.gset import Group, Perm
        perms = []
        for g in gen:
            perm = g.to_perm()
            perm = Perm(perm)
            print(perm)
            perms.append(perm)
    
        G = Group.generate(perms, verbose=True)
        print(G)
        print(G.gapstr())
        print(G.structure_description())
    
        return

    G = mulclose(gen, verbose=True, maxsize=10000)
    print("|G| =", len(G))

    avoid = []
    if argv.avoid: # use to find the fault tolerant logicals
        avoid = get_avoid(code)

    print("tau...", end='', flush=True)
    dode = space.H() * code

    try:
        tau = iter(transversal.find_isomorphisms_css(code, dode)).__next__()
        print(" found")
    except StopIteration:
        print("no zx-duality")
        tau = None

    logops = set()

    G = list(G)
    shuffle(G)
    for g in G[:100]:
        # some automorphism gates.. these don't seem to help ... ??
        perm = [None]*n
        for (i,j) in g.where():
            perm[i] = j
        lop = space.get_perm(perm)
        dode = lop*code
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        logops.add(L)

    if argv.limit:
        G = G[:argv.limit]
    print("G:", len(G))

    if argv.physical:
        f = open(argv.physical, 'w')
        vs = []
        for i,L in enumerate(G):
            m = "M%d"%i
            vs.append(m)
            print("%s := %s;"%(m, L.gap()), file=f)
        print("G := Group([%s]);"%(','.join(vs)), file=f)
        f.close()
        print("wrote to", argv.physical)

    for g in G:
        if tau is None:
            break

        f = g*tau
        perm = [None]*n
        for (i,j) in f.where():
            perm[i] = j
        f = Perm(perm, items)
        #perm = [f[i] for i in items]

        # H-type gate
        lop = space.get_perm(perm)
        lop = space.H() * lop
        dode = lop*code
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        logops.add(L)

        s = f.fixed()
        if not (f*f).is_identity():
            continue

        # S-type gate
        pairs = [(i,f(i)) for i in range(n) if i<f(i)]
        succ = True
        for lh in avoid:
            for (i,j) in pairs:
                if lh[i] and lh[j]:
                    succ = False
        if not succ:
            continue

        print(pairs, s)

        ops = [space.CZ(i,j) for (i,j) in pairs] + [space.S(i) for i in s]
        #lop = space.I()
        #if len(pairs):
        #    lop = reduce(mul, [space.CZ(i,j) for (i,j) in pairs])
        #if s:
        #    lop = lop * reduce(mul, [space.S(i) for i in s])

        dode = code
        for (i,j) in pairs:
            dode = space.CZ(i,j) * dode
            d = dode.distance("z3")
            assert d>=code.d, d
        for i in s:
            dode = space.S(i)*dode
        assert dode.is_equiv(code)

        #lop = reduce(mul, ops)
        #dode = lop*code
        #assert dode.is_equiv(code)
        L = dode.get_logical(code)
        logops.add(L)

    if argv.gen:
        print("logops:", len(logops))
        logops = mulclose(logops, verbose=True, maxsize=100000)

    print("logops:", len(logops))
    logops = list(logops)
    shuffle(logops)
    gen = logops[:100]

    if argv.name:
        f = open(argv.name, 'w')
        vs = []
        for i,L in enumerate(gen):
            m = "M%d"%i
            vs.append(m)
            print("%s := %s;"%(m, L.gap()), file=f)
        print("G := Group([%s]);"%(','.join(vs)), file=f)
        f.close()
        print("wrote to", argv.name)

    css = code.to_css()
    dump_transverse(css.Hx, css.Lx, 4)

    return logops


def test_bring():
    total = construct.get_bring()
    total.distance()
    print(total)

    #items = get_zx_dualities(total.to_qcode())
    #print(items)
    #return

    G = total.find_autos()
    print(len(G))

    zxs = total.find_zx_dualities(False, False) # 120 of these
    print(len(zxs))
    return

    zxs = total.find_zx_dualities()
    print("involutory fixed-point free zx-dualities:", len(zxs))

    total = total.to_qcode()

    codes = []
    for zx in zxs:
        #print(zx)
        cover = Cover.fromzx(total, zx)
        base = cover.base
        base.distance("z3")
        print(base)
        #print(base.longstr(False))
        #print(strop(base.H))
        codes.append(base)

    for a in codes:
      for b in codes:
        assert is_iso(a,b)



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


def toric_wraps(code):
    keys, lookup = code.keys, code.lookup
    print(code.keys, len(code.keys))
    rows = cols = max(key[1] for key in keys)+1
    assert len(keys) == rows**2

    for dcol in range(1, rows, 2):
      for drow in range(0, rows, 2):
        fibers = []
        for idx,(a,b) in enumerate(keys):
            c, d = drow-a, dcol-b
            jdx = lookup[c%rows, d%cols]
            #print((a,b), "-->", keys[jdx])
            if idx < jdx:
                fibers.append((idx, jdx))
        print(drow, dcol)
        yield fibers


def test_dehn():
    rows = cols = argv.get("rows", 4)
    code = construct.get_toric(rows)
    print(code)
    #print(strop(code.H))
    #print()

    space = code.space
    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    perm = space.get_perm(perm)

    CX = space.CX

    g0 = space.get_identity()
    g1 = space.get_identity()
    #for i in [4,12]:
    for i in range(rows):
        if i%2==0:
            continue
        idx = i*cols
        for j in range(cols):
            src, tgt = (idx+j, idx+(j+1)%cols)
            if j%2==0:
                g0 *= CX(src, tgt)
            else:
                g1 *= CX(tgt, src)
    #print(g)
    assert g0*g1 == g1*g0
    g = g0*g1*perm

    dode = code.apply(g)
    assert code.is_equiv(dode)
    l = code.get_logical(dode)
    #print(SymplecticSpace(2).get_name(l))
    #print(l*l)

    if 0:
        c1 = code.apply(g1*perm)
        print(c1.longstr())
        print("dist:", c1.distance("z3")) # ouch it's only distance rows/2 here... 
        c2 = c1.apply(g0)
        assert c2.is_equiv(code)
        return

    if rows > 4:

        for fibers in toric_wraps(code):
            base = wrap(code, None, fibers)
        
            cover = Cover(base, code, fibers)
            h = cover.descend(g)
            if h is None:
                continue
            #base.distance("z3")
            print(base)
            #print(strop(base.H))
            dode = base.apply(h)
            print("L =")
            print(dode.get_logical(base))
            print()
    
        return

    zxs = get_zx_wrap(code)
    print("zxs:", len(zxs))
    #for zx in zxs:
    #    print(zx)
    #return

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
        base = cover.base
        g1 = cover.descend(g)
        if g1 is None:
            continue

        dode = base.apply(g1)
        print("lift:", cover.lift(g1) == g)
        print(fibers)
        keys = cover.total.keys
        for (i,j) in fibers:
            print("\t", keys[i], keys[j])
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





