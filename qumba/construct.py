#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle
from functools import reduce
from operator import add

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, get_weight, fromstr
from qumba.csscode import CSSCode, find_zx_duality
from qumba.argv import argv


def get_422():
    return QCode.fromstr("XXXX ZZZZ", None, "XXII ZIZI XIXI ZZII")


def get_513():
    H = """
    XZZX.
    .XZZX
    X.XZZ
    ZX.XZ
    """
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
        for key in [(r, c, 0), (r, c, 1), (r, c+1, 1), (r+1, c, 0)]:
            X[getidx(*key)] = 1
        Hx.append(X)

        Z = [0]*n
        for key in [(r, c, 0), (r, c, 1), (r, c-1, 0), (r-1, c, 1)]:
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



def test():

    from qumba.lattices.db import items

    lattice = {}
    
    # face--edge, edge--vert
    A, B = items[0]
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

    estabs = []
    for i in range(nedge):
        b = B[i]
        j0, j1 = numpy.where(b)[0]
        op = ['I']*nvert
        for stab in stabs:
            s0, s1 = stab[j0], stab[j1]
            if s0 == 'I' and s1 != 'I':
                op[j1] = stab[j1]
            elif s1 == 'I' and s0 != 'I':
                op[j0] = stab[j0]
        op = ''.join(op)
        estabs.append(op)
    H = fromstr(stabs + estabs)
    print(shortstr(H), H.shape, rank(H))
    

    return

    stabs.pop()
    code = QCode.fromstr(stabs)
    print(code)
    print(code.longstr())

#    HL = numpy.concatenate((code.H, code.L[::2]))
#    H1 = []
#    for v in span(HL):
#        if v.sum() == 0:
#            continue
#        if get_weight(v) == 2:
#            #print(v)
#            H1.append(v)
#    H = numpy.concatenate((code.H, H1))
#    H = linear_independent(H)
#    code = QCode(H)
#    print(code.get_params())
#    print(code.longstr())
#
#    HL = numpy.concatenate((code.H, code.L[1::2]))
#    H1 = []
#    for v in span(HL):
#        if v.sum() == 0:
#            continue
#        if get_weight(v) == 2:
#            #print(v)
#            H1.append(v)
#    H = numpy.concatenate((code.H, H1))
#    H = linear_independent(H)
#    code = QCode(H)
#    print(code.get_params())
#    print(code.longstr())



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






