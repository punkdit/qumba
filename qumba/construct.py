#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)
from qumba.qcode import QCode, SymplecticSpace
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


