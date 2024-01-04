#!/usr/bin/env python

from random import shuffle
from functools import reduce
from operator import add

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, get_weight, fromstr
from qumba.csscode import CSSCode, find_zx_duality
from qumba.argv import argv

def get_412():
    return QCode.fromstr("XYZI IXYZ ZIXY")

def get_422():
    return QCode.fromstr("XXXX ZZZZ", None, "XXII ZIZI XIXI ZZII")

def get_832():
    """ Build the [[8,3,2]] code.

    We _number the qubits like this:
    0 ----------- 1
    |\           /|
    | \         / |
    |  2-------3  |
    |  |       |  |
    |  |       |  |
    |  6-------7  |
    | /         \ |
    |/           \|
    4 ----------- 5

    See:
    https://earltcampbell.com/2016/09/26/the-smallest-interesting-colour-code/
    """
    return QCode.fromstr(
        "XXXXXXXX ZZZZIIII ZZIIZZII ZIZIZIZI ZZZZZZZZ",
        None,
        "XXIIXXII IZIZIIII IXIXIXIX ZZIIIIII XXXXIIII IZIIIZII"
    )


def get_513(idx=None):
    if idx is None:
        H = """
        XZZX.
        .XZZX
        X.XZZ
        ZX.XZ
        """
    else:
        # some other 5,1,3 codes:
        data = """
        Y.ZZX ZYZ.Y ZZXZ. .ZZYZ
        XZZX. ZY.XZ .ZXZZ Z.ZZX
        YZ.ZY ZXZ.Y Z.YZZ .ZZYZ
        Y.ZZX ZX.XX .ZXYX ZZZ.Z
        YZZ.Z ZYZZ. .ZXZX ZZ.XY
        XZZY. .XZZX Z.YYX ZZ.ZZ
        X.ZXX ZX.XZ .ZYXY ZZZZ.
        X.ZXZ ZX.ZZ .ZXYZ ZZZ.Y
        YZXZ. .YYZX ZZ.YX Z.ZZZ
        Y.ZZY ZY.ZZ .ZXZX ZZZX.
        X.ZZX ZY.ZZ ZZY.X .ZZXZ
        YZX.X .XYZX Z.ZXX ZZ.ZZ
        Y.ZZZ ZYZ.Y .ZYZY ZZ.YZ
        XZX.X ZXXZ. Z.ZYX .ZZZZ
        YZZ.Y .YZZX ZZYZ. Z.ZXZ
        XZZ.Y ZYZZ. .ZXZX ZZ.YZ
        XZZX. .XZZZ Z.XXZ ZZ.ZX
        XZ.ZY .YZZX ZZY.X Z.ZYY
        XZXZ. ZXX.Z .ZZYZ Z.ZZY
        Y.ZZX ZXZY. .ZYYX ZZ.ZZ
        XZZZ. ZX.ZZ ZZX.X Z.ZYY
        YZZZ. ZX.ZX ZZX.Y Z.ZXZ
        YZ.ZX .YZZY ZZY.Y Z.ZYX
        Y.ZZY ZX.ZX ZZY.Y .ZZXX
        YZZ.X .YZZY ZZXZ. Z.ZYZ
        X.ZZZ ZYZ.Y ZZYZ. .ZZXX
        YZ.ZY .YZZX ZZX.X Z.ZYY
        YZ.ZY ZXZZ. ZZY.X .ZZXZ
        YZ.XZ .XZZZ Z.YYZ ZZZ.X
        X.ZYZ ZX.ZZ .ZXXZ ZZZ.X
        YZ.ZX ZYZZ. ZZY.Z .ZZYY
        """.strip().split('\n')
        H = data[idx]

    code = QCode.fromstr(H)
    return code


def get_713():
    H = "X..X.XX .X.XXX. ..X.XXX Z..Z.ZZ .Z.ZZZ. ..Z.ZZZ"
    code = QCode.fromstr(H)
    return code


def golay():
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


def get_10_2_3():
    toric = QCode.fromstr("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    ZZ.....Z.Z
    """, d=3)
    return toric


def get_512():
    toric = QCode.fromstr("""
    ZZZ..
    ..ZZZ
    X.XX.
    .XX.X
    """, d=2)
    return toric


def toric(rows, cols, delta_row=0, delta_col=0):
    n = 2*rows*cols

    def getidx(r, c, k):
        assert k in [0, 1]
        r, c = r+(c//cols)*delta_row, c+(r//rows)*delta_col
        idx = 2*((r%rows)*cols + (c%cols)) + k
        return idx

    Hx, Hz = [], []
    for r in range(rows):
      for c in range(cols):
        X = [0]*n
        for key in [(r, c, 0), (r, c, 1), (r, c-1, 0), (r-1, c, 1)]:
            X[getidx(*key)] = 1
        Hx.append(X)

        Z = [0]*n
        for key in [(r, c, 0), (r, c, 1), (r, c+1, 1), (r+1, c, 0)]:
            Z[getidx(*key)] = 1
        Hz.append(Z)

    Hx = numpy.array(Hx)
    Hz = numpy.array(Hz)

    code = CSSCode(Hx=Hx, Hz=Hz)
    return code


def reed_muller():
    # RM [[16,6,4]]
    H = parse("""
    11111111........
    ....11111111....
    ........11111111
    11..11..11..11..
    .11..11..11..11.
    """)

    rm = QCode.build_css(H, H)
    return rm


def biplanar(w=24, h=12):
    "ibm biplanar code"
    wrap = lambda re, im : (re%w, im%h)

    zstabs = []
    xstabs = []
    for x in range(w):
      for y in range(h):
        if x%2==0 and y%2==1:
            zop = [wrap(x+dx, y+dy) for (dx, dy) in 
                [(1,0), (0,1), (-1,0), (0,-1), (-3,-6), (6,-3)]]
            zstabs.append(zop)
        if x%2==1 and y%2==0:
            xop = [wrap(x+dx, y+dy) for (dx, dy) in 
                [(1,0), (0,1), (-1,0), (0,-1), (+3,-6), (-6,+3)]]
            xstabs.append(xop)

    keys = reduce(add, zstabs + xstabs)
    keys = list(set(keys))
    keys.sort()
    lookup = dict((key,idx) for idx,key in enumerate(keys))
    n = len(keys)

    Ax = zeros2((len(xstabs), n))
    for i,op in enumerate(xstabs):
        for key in op:
            Ax[i, lookup[key]] = 1

    Az = zeros2((len(zstabs), n))
    for i,op in enumerate(zstabs):
        for key in op:
            Az[i, lookup[key]] = 1

    code = CSSCode(Ax=Ax, Az=Az)
    return code


def classical_codes(n, m, distance=3):
    from bruhat.algebraic import qchoose_2

    for H in qchoose_2(n, m):
        if H.sum(0).min() == 0: # distance=1
            continue
        if H.sum(1).min() == 1: # dead bit
            continue
        K = kernel(H)
        k, n = K.shape
        assert k==n-m, k
        d = n
        for idx in numpy.ndindex((2,)*k):
            v = dot2(idx, K)
            if 0 < v.sum() < d:
                d = v.sum()
        #print("d =", d)
        if d>=distance:
            yield H


def kron(A, B):
    if 0 in A.shape or 0 in B.shape:
        C = zeros2(A.shape[0]*B.shape[0], A.shape[1]*B.shape[1])
    else:
        #print("kron", A.shape, B.shape)
        C = numpy.kron(A, B)
        #print("\t", C.shape)
    return C

def hypergraph_product(A, B, check=False):
    #print("hypergraph_product: A=%s, B=%s"%(A.shape, B.shape))

    ma, na = A.shape
    mb, nb = B.shape

    Ima = identity2(ma)
    Imb = identity2(mb)
    Ina = identity2(na)
    Inb = identity2(nb)

    Hz0 = kron(Ina, B.transpose()), kron(A.transpose(), Inb)
    Hz = numpy.concatenate(Hz0, axis=1) # horizontal concatenate

    Hx0 = kron(A, Imb), kron(Ima, B)
    #print("Hx0:", Hx0[0].shape, Hx0[1].shape)
    Hx = numpy.concatenate(Hx0, axis=1) # horizontal concatenate

    assert dot2(Hx, Hz.transpose()).sum() == 0

    return Hx, Hz




def test():

    from qumba.lattices import lattices

    idx = argv.get("idx", 1)

    # face--edge, edge--vert
    A, B = lattices[idx]
    print("face--edge")
    print(shortstr(A), A.shape)

    print("edge-vert")
    print(shortstr(B), B.shape)

    assert A.shape[1] == B.shape[0]
    nface, nedge = A.shape
    nedge, nvert = B.shape

    #rows = []
    #for i in range(nface):
        #for j in range(nvert):

    AB = numpy.dot(A, B) // 2

    print("face-vert")
    AB = AB.astype(object)
    print(shortstr(AB), AB.shape)

    for i in range(nvert):
        XYZ = list('XYZ')
        shuffle(XYZ)
        for j in range(nface):
            if AB[j, i] == 0:
                AB[j, i] = "I"
            else:
                AB[j, i] = XYZ.pop()
    print(AB, AB.shape)

    stabs = [''.join(row) for row in AB]
    print(stabs)

    H = fromstr(stabs)
    print(shortstr(H), H.shape, rank(H))

    edges = []
    for i in range(nedge):
        b = B[i]
        edge = tuple(numpy.where(b)[0])
        edges.append(edge)
    shuffle(edges)

    estabs = []
    while edges:
        j0, j1 = edges.pop()
        i = 0
        while i < len(edges):
            edge = edges[i]
            if j0 in edge or j1 in edge:
                edges.pop(i)
            else:
                i += 1

        op = ['I']*nvert
        for stab in stabs:
            s0, s1 = stab[j0], stab[j1]
            if s0 == 'I' and s1 != 'I':
                op[j1] = stab[j1]
            elif s1 == 'I' and s0 != 'I':
                op[j0] = stab[j0]
        op = ''.join(op)
        #print(op)
        estabs.append(op)
    H = fromstr(stabs + estabs)
    H = linear_independent(H)
    print(shortstr(H), H.shape, rank(H))

    code = QCode(H)
    from qumba.distance import distance_z3
    d = distance_z3(code)
    code.d = d
    print(code)
    #print(code.longstr())



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
        test()


    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")






