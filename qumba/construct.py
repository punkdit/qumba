#!/usr/bin/env python

from random import shuffle
from functools import reduce
from operator import add

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, Matrix, get_weight, fromstr
from qumba.csscode import CSSCode, find_zx_duality
from qumba.argv import argv

def get_412():
    return QCode.fromstr("XYZI IXYZ ZIXY")

def get_422():
    return QCode.fromstr("XXXX ZZZZ", None, "XXII ZIZI XIXI ZZII")

def get_832():
    r""" Build the [[8,3,2]] code.

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


def get_833():
    "[[8,3,3]] non-css code from https://arxiv.org/abs/quant-ph/9702029"
    code = QCode.fromstr(
        "XXXXXXXX ZZZZZZZZ XIXIZYZY XIYZXIYZ XZIYIYXZ",
        None,
        "XXIIIZIZ IZIZIZIZ XIXZIIZI IIZZIIZZ XIIZXZII IIIIZZZZ"
    )
    return code


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
    return QCode.fromstr("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    ZZ.....Z.Z
    """, d=3)


def get_512():
    return QCode.fromstr("""
    ZZZ..
    ..ZZZ
    X.XX.
    .XX.X
    """, d=2)


def get_11_2_3():
    return QCode.fromstr("""
    ZYX........
    Z..YY...X..
    XX..XX.....
    .ZZ..ZZ....
    ..X...YY..Z
    ...Z....Z..
    ....ZZ..ZZ.
    .....XX..XX
    .......X..X
    """)


def get_622():
    return QCode.fromstr("""
    XXXIXI
    ZZIZIZ
    IYZXII
    IIYYYY
    """, d=2)


def get_14_3_3():
    "Landahl jaunty code"
    return QCode.fromstr("""
    XXXIIXIIIIIIII
    ZZIIZIIIIZIIII
    IYZYXIIIIIIIII
    IIZIIYYIIIXIII
    IIXXIIXXIIIIII
    IIIZZIIZZIIIII
    ZIIIIZIIIIZIIZ
    IIIIIIZZIIZZII
    IIIIIIIXXIIXXI
    IIIIXIIIYYIIZI
    IIIIIIIIIIXYZY
    """, d=3)
    


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


def get_bring():
    "Bring's code"
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
    return CSSCode(Ax=Ax, Az=Az)


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



def all_css(n, k, distance=3):
    from bruhat.algebraic import qchoose_2
    m = (n-k)
    assert m%2 == 0
    m = m//2

    for Hx in qchoose_2(n, m):
        K = kernel(Hx)
        #print(Hx, Hx.shape)
        #print(K, K.shape)
        for Mz in qchoose_2(K.shape[0], m):
            #print(Mz)
            Hz = dot2(Mz, K)
            #print(Hz)
            assert dot2(Hx, Hz.transpose()).sum() == 0
            code = QCode.build_css(Hx, Hz)
            if code.d >= distance:
                yield code
        #print()
        #return




def all_codes(n=4, k=1, d=2):
    from bruhat.sp_pascal import i_grassmannian

    space = SymplecticSpace(n)
    gen = []
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(2*n - i - 1)
        gen.append(space.get_S(i))
        gen.append(space.get_H(i))
    found = []

    F = space.F
    for _,H in i_grassmannian(n, n-k):
        H = H[:, perm] # reshuffle to qumba symplectic
        H = Matrix(H)
        code = QCode(H, check=False)
        if code.get_distance() < d:
            continue
        yield code



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


def make_torus(a, b=0):
    lookup = {}
    c = max(a, b)
    deltas = [(0, 0), (a,b), (-a,-b), (-b,a), (b,-a)]
    deltas = set()
    for (di,dj) in [(a,b), (-b,a)]:
      for k in [-2,-1,0,1,2]:
        deltas.add((k*di,k*dj))
    deltas.remove((0,0))
    deltas = list(deltas)
    deltas.sort()
    deltas = [(0,0)] + deltas
    #print(deltas)
    n = 0
    keys = []
    for i in range(a+b):
        for j in range(c):
            for di, dj in deltas:
                tgt = i+di, j+dj
                if tgt in lookup:
                    lookup[i,j] = lookup[tgt]
                    break
            else:
                key = (i,j)
                lookup[key] = n
                keys.append(key)
                n += 1
    # hack this....
    for i in range(2*(a+b)):
        for j in range(-2*c,2*c):
            for di, dj in deltas:
                tgt = i+di, j+dj
                if tgt in lookup:
                    lookup[i,j] = lookup[tgt]
                    break
    assert len(keys) == n == a**2 + b**2, len(keys)
    return lookup, keys

def get_xzzx(a, b=0):
    lookup, keys = make_torus(a, b)
    n = len(keys)
    rows = []
    for (i,j) in keys:
        row = ['.']*n
        row[lookup[i,j]] = 'X'
        row[lookup[i,j+1]] = 'Z'
        row[lookup[i+1,j+1]] = 'X'
        row[lookup[i+1,j]] = 'Z'
        row = ''.join(row)
        rows.append(row)
    assert len(rows) == n
    rows = ' '.join(rows)
    H = fromstr(rows)
    code = QCode(A=H)
    code.lookup = lookup
    code.keys = keys
    code.coords = (a, b)
    return code


def get_toric(a, b=0):
    lookup, keys = make_torus(a, b)
    n = len(keys)
    assert n%2 == 0
    rows = []
    for (i,j) in keys:
        row = ['.']*n
        op = 'Z' if (i+j)%2 else 'X'
        row[lookup[i,j]] = op
        row[lookup[i,j+1]] = op
        row[lookup[i+1,j+1]] = op
        row[lookup[i+1,j]] = op
        row = ''.join(row)
        rows.append(row)
    assert len(rows) == n
    rows = ' '.join(rows)
    H = fromstr(rows)
    code = QCode(A=H)
    code.lookup = lookup
    code.keys = keys
    code.coords = (a, b)
    return code


def get_surface(rows, cols):
    n = rows*cols
    parity = lambda i,j:'X' if (i+j)%2 else 'Z'
    ops = []
    get = lambda i,j:cols*i + j
    for i in range(rows-1):
      for j in range(cols-1):
        op = parity(i,j)
        row = ['.']*n
        row[get(i,j)] = op
        row[get(i+1,j)] = op
        row[get(i,j+1)] = op
        row[get(i+1,j+1)] = op
        ops.append(''.join(row))
    for j in range(0,cols-1,2): # top boundary
        i = 0
        op = parity(i-1,j)
        row = ['.']*n
        row[get(i,j)] = op
        row[get(i,j+1)] = op
        ops.append(''.join(row))
    for j in range(rows%2,cols-1,2): # bottom boundary
        i = rows-1
        op = parity(i,j)
        row = ['.']*n
        row[get(i,j)] = op
        row[get(i,j+1)] = op
        ops.append(''.join(row))
    for i in range(1,rows-1,2): # left boundary
        j = 0
        op = parity(i,j-1)
        row = ['.']*n
        row[get(i,j)] = op
        row[get(i+1,j)] = op
        ops.append(''.join(row))
    for i in range(1-cols%2,rows-1,2): # right boundary
        j = cols-1
        op = parity(i,j)
        row = ['.']*n
        row[get(i,j)] = op
        row[get(i+1,j)] = op
        ops.append(''.join(row))
        
    assert len(ops) == n-1
    ops = ' '.join(ops)
    H = fromstr(ops)
    assert rank(H) == n-1
    code = QCode(H)
    return code



def test_xzzx():

    code = get_toric(2, 2)
    assert code.n == 8
    assert code.k == 2
    assert code.get_distance() == 2

    from qumba.distance import distance_z3
    code = get_xzzx(1,3)
    assert code.n == 10
    assert code.k == 2
    assert code.get_distance() == 3
    code = get_xzzx(3,4)
    assert code.n == 25
    assert code.k == 1
    assert distance_z3(code) == 3+4
    #print(code.longstr())


def get_css(param):
    H = Hx = Hz = None
    code = None
    if param == (56, 14, 6):
        Hx = Hz = """
    11111111................................................
    11.........1..1...1..1...1..........1...................
    1......1..11...1........1...1...............1...........
    .11.....1........11...............1.......1.........1...
    ..11....11.........1............1.............11........
    ........111................11.....11.............1......
    .........111.11........................1.......1......1.
    ......11....1..11..............1..................1....1
    .....11.....11........1..........1....11................
    ............111..........11...11.........1..............
    ...............111.....11.................11....1.......
    ................111.11..................1.........11....
    ...11..............11........11.........11..............
    ...................111..........11..111.................
    ....11................11.....1.....1............11......
    ......................111.......11..........111.........
    .........................111........11.....1....11......
    ..........................111...........11..11.....1....
    .............................111..11................11.1
    .....................................111..11........111.
    .............................................111..11.111
        """
    elif param == (32,12,4):
        Hx = Hz = """
    11111111........................
    11......1..11...1...1..1........
    1......1..11..1...1.....1..1....
    .11.....11.....1............11.1
    ........111.....1111..........1.
    ......11.11..1.......11........1
    ...........11111......1...1..1..
    .....11.....11.....1...1.1....1.
    ..11..........11.1......11....1.
    ...11...........11..111...1.....
        """
    elif param==(30, 5, 3): 
        # bruhat$ ./qcode.py  build_geometry key=5,4 idx=7 homology=1 show
        H = """
.....X...X.....X..X........X..
...X.....X.............X.X..X.
........X....X......X.X.....X.
XXX................X....X.....
...............X.X.X..X......X
...X......X.X........X.......X
....X......X..X........X..X...
......XX........XX.......X....
............X.X.X...X...X.....
X...XX.X.............X........
..X...X......X............XX..
...........Z......Z.......ZZ..
..............Z.Z......Z.Z....
.....Z.Z.......Z.Z............
......Z......Z..Z...Z.........
...Z...Z.............Z...Z....
...................ZZ.Z.Z.....
...Z.....Z.....Z.............Z
.Z........Z.Z...........Z.....
ZZ..Z......Z..................
.............Z.........Z..Z.Z.
........ZZ........Z.........Z.
........Z.Z...........Z......Z
Z.Z..Z.....................Z..
....Z.......Z.Z......Z........
        """
    elif param==(40,6,4):
        # bruhat$ ./qcode.py  build_geometry key=5,4 idx=8 homology=1 show
        H = ("""
.....X.....X..X.....X.....X.............
.........X..........XXX..........X......
X....................X..X.....X.....X...
.......XX........XX....................X
...X.X.X............................XX..
.X..X.....................X........X..X.
......X....X.....X.....X.....X..........
........XXX........X.....X..............
.X...........X.....X..............X..X..
...............XX.X..............X....X.
X........................X.X.X.....X....
..XX...........X...........XX...........
......X.........X.......X......X..X.....
..X.........XX........XX................
....X.......X.................X.X......X
.Z.................Z.....Z.........Z....
..........Z..............Z.ZZ...........
..ZZ.........Z.......................Z..
............Z....Z.....Z...............Z
............Z........ZZ.......Z.........
........ZZ........Z..............Z......
...Z...Z.......Z..Z.....................
......Z....Z..Z................Z........
Z...Z.........................Z....Z....
........Z.Z.....................Z......Z
.........Z...Z.....Z..Z.................
.....Z..............ZZ..............Z...
....Z.........Z...........Z.....Z.......
Z.....Z.................Z....Z..........
...............ZZ...........Z..Z........
..Z....................Z...Z.Z..........
.....Z.Z...Z.....Z......................
........................Z.........Z.ZZ..
.Z..............Z.................Z...Z.
    """)
    elif param==(40,10,4):
        # bruhat$ ./qcode.py  build_geometry key=5,5 idx=5 homology=1 show
        H = """
..X...X..........X..............X....X..
...X......X.......X..X........X.........
..................X.....X..X.X....X.....
........X.X.X.............X............X
...X.X...X....X..........X..............
.X...........X........XX..X.............
...........X.....X............XX....X...
.....X..X..........................X.XX.
.......X......XXX...X...................
....X.................X..X..X..X........
XX.......X...................X..X.......
X..........XX...X..X....................
.......X.....X.............X........X.X.
...................X........X....XXX....
..X.X...............X...X..............X
...........................ZZ..Z..Z.Z...
....Z....Z..............ZZ...Z..........
.......Z.......Z.................Z.Z..Z.
.................ZZ..........ZZ.Z.......
.....Z..Z.....Z.....Z..................Z
.Z.Z.....ZZ...............Z.............
.......Z.....ZZ.......Z..Z..............
Z.Z.........Z...................Z......Z
......Z....Z.....Z.Z.............Z......
....Z...........Z..ZZ.......Z...........
.............Z.......Z.Z......Z.....Z...
............Z..ZZ......Z..Z.............
......Z.Z.Z..........Z...............Z..
...Z.Z............Z........Z..........Z.
ZZ.........Z..........Z........Z........
    """
    elif param == (30,8,3): # Bring's code
        H = """
.......X.X........XX..X.......
.X..XX..........X..........X..
X.................X......XX..X
....X...X.X.X.............X...
...X.....X.X....X...X.........
......X......X.X.X..........X.
..X...X...X.............X..X..
..X..X.....X...X.....X........
X...........X.X.........X...X.
...X...X.........X...X.X......
.............XX....X...X.X....
..Z.....Z.ZZ........Z.........
....ZZ......Z..Z............Z.
...Z..Z.........ZZ.........Z..
........Z...Z.Z....Z..Z.......
...Z................Z..Z.Z...Z
.........Z.Z.Z.Z...Z..........
Z......Z.........ZZ.........Z.
ZZ......................Z..Z.Z
.Z...Z.Z.............ZZ.......
..Z...........Z......Z.ZZ.....
....Z....Z......Z.Z.......Z...
        """
    elif param==(15,5,3): # ./qcode.py  build_geometry key=5,5 idx=3 homology=1 show
        H = """
X..X....XX....X
..X.XXX.X......
XX..X......X.X.
..XX...X..XX...
.X....XX.X..X..
.Z.......ZZZ..Z
...ZZ...Z.Z..Z.
ZZZ....ZZ......
Z...ZZ...Z..Z..
..ZZ..Z.....Z.Z
        """
    elif param==(30,7,3): # ./qcode.py  build_geometry key=6,4 idx=31 homology=1 show
        H = """
..........X..X...X....X....XX.
X.........X....X..XX......X...
..X.....X...........XX....X..X
.X..XX.X.....X....X...........
..XX..........X........XX...X.
.X.........X....X.......XX.X..
......XX.X....X..........X...X
X....X..XX.XX.................
...XX.X........X.....XX.......
....ZZ..Z............Z........
.........Z..Z.Z........Z......
......Z...............Z..Z.Z..
.......Z..........Z.......Z..Z
Z.........ZZ...............Z..
.....Z......ZZ...Z............
...Z...........Z...Z...Z......
..Z.....Z..Z............Z.....
Z.....Z..Z.....Z..............
................Z...Z....Z...Z
..Z.......Z...............Z.Z.
.Z.ZZ...................Z.....
.................Z..ZZZ.......
.......Z.....ZZ.............Z.
    """
    elif param==(36,8,4):
        H = """
...........X.......X....X..X.X.....X
........X........XX.......X.X.....X.
......................X..X..XXX.X...
......XXX....X.............X...X....
...X.....X...X.X.......X........X...
..XX........X...X.X................X
..........X....X...XXX....X.........
XXX.................X.X........X....
.X..X....X.X.....X...............X..
.....XX.....X........X...X.......X..
....X..X..X...X.X.............X.....
.............Z........Z........ZZ...
.........Z.......Z..........Z...Z...
Z..............Z....Z..Z............
........Z..................ZZZ......
..Z.........Z.......ZZ..............
.Z....................Z..Z.......Z..
......ZZ.................Z....Z.....
...................Z....Z.Z.......Z.
Z....ZZ........................Z....
.......Z......Z.........Z..Z........
.....Z....Z...Z......Z..............
....Z.......Z...Z................Z..
..........Z.....Z..Z...............Z
.ZZZ.....Z..........................
........Z....Z.........Z..........Z.
...........Z.....ZZ................Z
....Z......Z.................ZZ.....
"""
    elif param==(27,11,3):
        H = """
...X.X.XX........X........X
....X.X....X..XX..........X
..XX.....X......X.X.X......
.X...........XX...XX.X.....
..X..X....X....X.....X...X.
....X....XX.X..........XX..
XX.....X...X....X.......X..
X...........X....X.X..X..X.
..ZZ....Z..........Z.ZZ....
Z.......ZZ.......Z..Z...Z..
.......Z...Z.ZZ..Z....Z....
....Z.Z...Z..Z.....Z.....Z.
.Z....Z..Z....Z.Z......Z...
...ZZZ.........Z....Z..Z...
Z......Z..Z.Z..Z..........Z
............Z...Z.Z..Z..ZZ.
        """
    elif param==(48,18,4): # ./qcode.py build_geometry key=6,6  homology=1  show idx=49
        H = """
.......X......X...X..............X........X....X
.X..............XX...X......X.................X.
X.X........................X.X..X........X......
........X.....X......X....XX..........X.........
.....X.......X.X........X.........X........X....
..X.......XX.............X.................X...X
....................X....X.........X..X.....XX..
.X.............X...XX...........X....X..........
X......X....X.....................XX....X.......
...X..............X..........X.........X....X.X.
...X..X...............X...X....X.....X..........
....XX...........X............XX..........X.....
....X....X.XX......X................X...........
.............X..X.....X.............X....X...X..
........X.X............X......X........XX.......
......Z..Z..Z.............Z........Z..Z.........
Z....................Z..Z..ZZ.....Z.............
Z.............................ZZZ....Z..Z.......
...Z...Z.....Z....Z...Z...........Z.............
..Z...Z...Z...........ZZ.................Z......
.........Z........ZZ.........Z..ZZ..............
.Z.....Z.........Z..Z..............Z......Z.....
....Z.........Z............Z........Z....ZZ.....
.Z......Z.ZZ.......Z.Z..........................
...ZZ......Z.............Z.....Z............Z...
.....Z..Z....Z................Z.......Z......Z..
..Z..Z...........Z...........Z.............Z..Z.
...............Z....Z..ZZ..............Z....Z...
............Z...Z...................Z..ZZ.....Z.
..............ZZ..........Z..........Z.....Z...Z
"""


    if type(H) is str:
        css = QCode.fromstr(H).to_css()
        Hx = css.Hx
        Hz = css.Hz

    if Hx is None:
        print("param %r not found"%(param,))
        return

    if type(Hx) is str:
        Hx = parse(Hx.strip())
        Hz = parse(Hz.strip())

    code = CSSCode(Hx=Hx, Hz=Hz)

    return code


def test_css():
    param = argv.param
    code = get_css(param)
    if code is None:
        return

    print(code)

    from qumba.csscode import distance_z3_css
    d_x, d_z = distance_z3_css(code)
    print("d =", (d_x, d_z))



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






