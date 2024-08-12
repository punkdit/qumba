#!/usr/bin/env python


from random import shuffle, randint
from operator import add, matmul, mul
from functools import reduce

import numpy


from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)


from qumba.qcode import QCode, SymplecticSpace
from qumba.csscode import CSSCode
from qumba import construct 
from qumba.syntax import Syntax
from qumba.argv import argv




def get_surf9():
    code = QCode.fromstr("""
    XX.XX....
    .XX......
    ....XX.XX
    ......XX.
    Z..Z.....
    .ZZ.ZZ...
    ...ZZ.ZZ.
    .....Z..Z
    """)
    return code


def get_wenum(code):
    css = code.to_css()
    Hz = css.Hz
    Lz = css.Lz
    wenum = {w:[] for w in range(code.n+1)}
    for ik in numpy.ndindex((2,)*css.k):
      if ik == (0,)*css.k:
        continue
      for imz in numpy.ndindex((2,)*css.mz):
        h = dot2(imz, css.Hz) + dot2(ik, css.Lz)
        h %= 2
        wenum[h.sum()].append(h)

    return wenum

def get_gauge_wenum(code):
    css = code.to_css()
    Gz = css.Gz
    Lz = css.Lz
    mz = len(Gz)
    wenum = {w:[] for w in range(code.n+1)}
    for ik in numpy.ndindex((2,)*css.k):
      if ik == (0,)*css.k:
        continue
      for imz in numpy.ndindex((2,)*mz):
        h = dot2(imz, css.Gz) + dot2(ik, css.Lz)
        h %= 2
        wenum[h.sum()].append(h)

    return wenum


def test_decode():
    code = get_surf9()
    print(code)
    print(code.longstr())

    ancilla = QCode.fromstr("Z")
    print(ancilla)

    circ = ancilla + code
    print(circ)
    print(circ.longstr())

    s = Syntax()
    CX, CZ, H = s.CX, s.CZ, s.H

    a, b, c, d = 1, 2, 4, 5
    op = CX(1+c, 0) * CX(1+d, 0) * CX(1+a, 0) * CX(1+b, 0)
    dode = op*circ

    print()
    print(dode)
    print(dode.longstr())


def test_support():
    #code = get_surf9()
    #code = construct.get_713()
    #code = construct.get_bring()
    code = construct.get_832()
    d = code.d

    print(code)
    n = code.n
    wenum = get_wenum(code)
    print([len(wenum[i]) for i in range(code.n+1)])
    hs = wenum[3]
    for l in hs:
        print(l)
    
    print("found:", end="")
    for i in range(n):
     for j in range(i+1, n):
        for h in wenum[d]:
            if h[i] and h[j]:
                break
        else:
            print((i, j), end=" ")
    print()


def test():
    # gauge colour (color) code 
    Gx = Gz = parse("""
    1111...........
    ....1111.......
    11..11.........
    ..11..11.......
    1.1.1.1........
    .1.1.1.1.......
    ........1111...
    ........11..11.
    ........1.1.1.1
    11......11.....
    ..11......11...
    1.1.....1.1....
    .1.1.....1.1...
    ....11......11.
    ....1.1.....1.1
    1...1...1...1..
    .1...1...1...1.
    ..1...1...1...1
    """)
    Hx = Hz = parse("""
    11111111.......
    1111....1111...
    11..11..11..11.
    1.1.1.1.1.1.1.1
    """)

    #Hz = numpy.concatenate((Gz, Hz))
    #code = CSSCode(Hx=Hx, Hz=Hz)

    code = CSSCode(Gz=Gz, Gx=Gx)
    print(code)
    print(code.distance())
    print(code.Lx)
    print(code.Lz)

    Lx, Lz = code.Lz, code.Lz

    wenum = get_gauge_wenum(code)

    print([len(wenum[i]) for i in range(code.n+1)])

    n, d = code.n, 3
    print("found:", end="")
    for i in range(n):
     for j in range(i+1, n):
        for h in wenum[d]:
            if h[i] and h[j]:
                break
        else:
            print((i, j), end=" ")
    print()






if __name__ == "__main__":

    from time import time
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




