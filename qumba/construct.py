#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle
from functools import reduce
from operator import add

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum)
from qumba.qcode import QCode, SymplecticSpace
from qumba.csscode import CSSCode
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
    """)
    return toric


def get_rm():
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

    Hx = zeros2((len(xstabs), n))
    for i,op in enumerate(xstabs):
        for key in op:
            Hx[i, lookup[key]] = 1

    Hz = zeros2((len(zstabs), n))
    for i,op in enumerate(zstabs):
        for key in op:
            Hz[i, lookup[key]] = 1

    Hx = linear_independent(Hx)
    Hz = linear_independent(Hz)
    code = CSSCode(Hx=Hx, Hz=Hz)
    return code

