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
    flag = QCode.fromstr("X")

    circ = code + ancilla + flag
    print(circ)
    print(circ.longstr())

    n = code.n
    s = Syntax()
    CX, CZ, H = s.CX, s.CZ, s.H

    a, b, c, d = 1, 2, 4, 5
    op = CX(c, n) * CX(n+1, n) * CX(d, n) * CX(a, n) * CX(n+1, n) * CX(b, n)
    dode = op*circ

    print()
    print(dode)
    print(dode.longstr())


def get_pairs(code):
    n = code.n
    d = code.d
    wenum = get_wenum(code)
    hs = wenum[d]
    found = []
    for i in range(n):
     for j in range(i+1, n):
        for h in wenum[d]:
            if h[i] and h[j]:
                break
        else:
            found.append((i, j))
    return found


def test_support():
    #code = get_surf9()
    #code = construct.get_10_2_3()
    #code = construct.get_713()
    #code = construct.get_bring()
    #code = construct.get_832()

    from qumba.db import get_codes

    for code in get_codes():
        d = code.d
    
        print(code)
        n = code.n
        wenum = get_wenum(code)
        print([len(wenum[i]) for i in range(code.n+1)])
        hs = wenum[3]
        for l in hs:
            print(l)
        
        print("found:", get_pairs(code))


def test_gcolour():
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



def test_find():
    code = construct.get_10_2_3()
    #code = construct.get_513()
    #code = construct.get_toric(2,2) # 8,2,2

    src = code
    print(src.longstr())
    E = src.get_encoder()
    #print(E)

    n = src.n
    s = src.space
    CX, CZ, S, H, P, SWAP, invert = s.CX, s.CZ, s.S, s.H, s.P, s.SWAP, s.invert

    g = CX(8,9)
    #g = S(8)*S(9) # nope
    #g = H(8)*H(9) # nope

    tgt = QCode.from_encoder(E*g, k=2)
    assert (tgt.is_equiv(src))

    L = tgt.get_logical(src)
    #assert L == SymplecticSpace(2).CX()

    Ei = invert(E)
    physical = E*g*Ei
    print(physical)

    assert (physical*src).is_equiv(src)

    gates = [CX(i,j) for i in range(n) for j in range(n) if i!=j]

    # FAIL:

    circuit = []
    for trial in range(1000):
        found = gate_search(s, gates, physical)
        if found is None:
            continue
        if not found:
            break
        g = found[-1]
        code = g*src
        d = code.distance()
        if d == src.d:
            print("*")
            src = code
            circuit.insert(0, g)
            physical = invert(g)*physical
        
    print()
    if physical == s.get_identity():
        print([g.name for g in circuit], len(circuit))
    else:
        print("not found")


def gate_search(space, gates, target):
    gates = list(gates)
    A = space.get_identity()
    found = []
    while A != target:
        if len(found) > 100:
            return
        count = (A+target).sum()
        print(count, end=' ')
        shuffle(gates)
        for g in gates:
            B = g*A # right to left
            if (B+target).sum() <= count:
                break
        else:
            print("X ", end='', flush=True)
            return None
        found.insert(0, g)
        A = B
    print("^", end='')
    return found



def test_search_clifford():
    #code = construct.get_10_2_3()
    #code = construct.get_513()
    #code = construct.get_toric(2,2) # [[8,2,2]]
    #code = construct.get_toric(3,1) # [[10,2,3]]
    #code = construct.get_toric(4,2) # [[20,2,4]]
    code = construct.get_toric(3,3) # [[18,2,3]]
    #code = construct.get_toric(4,0) # [[16,2,4]]
    code.distance()
    print(code)

    n = code.n
    s = code.space
    CX, CZ = s.CX, s.CZ
    gates =  [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    #gates += [CZ(i,j) for i in range(n) for j in range(i+1,n)]

    while 1:
        circ = search_clifford(code, gates)
        if circ is not None:
            print([g.name for g in circ])

    # i think this is the knight move gate on [[10,2,3]]:
    # [('CX(3,8)',), ('CX(2,7)',), ('CX(0,5)',), ('CX(1,6)',), ('CX(4,9)',), 
    #  ('CX(8,3)',), ('CX(7,2)',), ('CX(9,4)',), ('CX(6,1)',), ('CX(5,0)',)]



def search_clifford(code, gates):
    "search for logical clifford on code"
    d = code.d
    assert d is not None

    circ = []
    src = code
    r = src.m
    count = 0
    while r or count < 10:
        count += 1
        shuffle(gates)
        for g in gates:
            tgt = g*src
            if tgt.distance() < d:
                continue
            src = tgt
            circ.append(g)
            break
        r = src.H.intersect(code.H).rank()
        print(r, end=' ', flush=True)
    print("/", end=' ', flush=True)

    count = 0
    while r < src.m:
        count += 1
        if count > 20:
            print("?")
            return # Fail
        found = []
        for g in gates:
            tgt = g*src
            if tgt.distance() < d:
                continue
            r1 = tgt.H.intersect(code.H).rank()
            if r1 > r:
                found = [g]
                r = r1
            elif r1 == r:
                found.append(g)
        if not found:
            print("X")
            return
        shuffle(found)
        g = found[0]
        src = g*src
        circ.append(g)
        r = src.H.intersect(code.H).rank()
        print(r, end=' ', flush=True)

    assert src.is_equiv(code)
    print()
    print(src.get_logical(code))
    circ = list(reversed(circ))
    return circ

    

                
        

def test_clifford():
    code = construct.get_10_2_3()
    #code = construct.get_513()
    #code = construct.get_toric(2,2) # 8,2,2

    n = code.n
    s = code.space
    CX, CZ = s.CX, s.CZ
    gates =  [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    #gates += [CZ(i,j) for i in range(n) for j in range(i+1,n)]

    print(code)
    d = code.d
    assert d is not None

    bdy = {code}
    found = {code.H}

    while bdy:
        print(len(bdy))
        _bdy = set()
        for src in bdy:
          for g in gates:
            tgt = g*src
            if tgt.distance() < d:
                continue
            if tgt in found:
                continue
#            for other in found:
#                if tgt.is_equiv(other):
#                    break
            else:
                print(".", end='', flush=True)
                found.add(tgt.H)
                _bdy.add(tgt)
        bdy = _bdy

    print()
                
        
def test_clifford_pairs():
    # equivalent & slower than just searching non-distance decreasing gates 
    code = construct.get_10_2_3()
    #code = construct.get_513()
    #code = construct.get_toric(2,2) # 8,2,2

    n = code.n
    s = code.space
    CX, CZ = s.CX, s.CZ

    print(code)
    d = code.d
    assert d is not None

    bdy = {code}
    found = set(bdy)

    gates =  {(i,j):CX(i,j) for i in range(n) for j in range(n) if i!=j}
    while bdy:
        print(len(bdy))
        _bdy = set()
        for src in bdy:
          pairs = get_pairs(src)
          #for g in gates:
          for pair in pairs:
           for (i,j) in [pair,reversed(pair)]:
            g = gates[i,j]
            tgt = g*src
            if tgt.distance() < d:
                continue
            if tgt in found:
                continue
            for other in found:
                if tgt.is_equiv(other):
                    break
            else:
                print(".", end='', flush=True)
                found.add(tgt)
                _bdy.add(tgt)
        bdy = _bdy

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




