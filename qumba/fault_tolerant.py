#!/usr/bin/env python


from random import shuffle, randint
from operator import add, matmul, mul
from functools import reduce

import numpy


from qumba.lin import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)


from qumba.qcode import QCode, SymplecticSpace, strop
from qumba.csscode import CSSCode
from qumba import construct 
from qumba.syntax import Syntax
from qumba.argv import argv

from qumba.matrix import Matrix
from qumba.umatrix import UMatrix, Solver, If, Not, And, Or, PbLe




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
                

def test_state_prep():
    "fault tolerant state prep for CSS codes"

    #code = construct.get_surface(3,3)
    code = construct.get_512() # sat
    code = QCode.fromstr("XXXXII IIXXXX ZZZZII IIZZZZ") # sat
    #code = construct.get_713() # unsat
    #code = construct.get_toric(2,2) # [[8,2,2]] sat
    #code = construct.get_10_2_3() # unsat
    #code = construct.get_913() # unsat
    #code = construct.get_toric(3,3) # [[18,2,3]] ??

    space = code.space
    n = code.n
    nn = 2*n

    print(code)
    print(code.longstr())
    print()

    E = code.get_encoder()
    H = code.H
    L = code.L

    if 0:
        print("E:")
        print(E, E.shape)
        print()
    
        print("H:")
        print(H, H.shape)
        print()
    
        print("L:")
        print(L, L.shape)
        print()

    #return

    M = []
    for i in range(code.m):
        u = [0]*nn
        u[2*i] = 1
        M.append(u)
    M = Matrix(M).t
    EM = E*M
    #print(EM, EM.shape)
    assert EM == H.t

    print("finding logops...")
    rows = []
    kk = 2*code.k
    H = code.H
    L = code.L[:kk]
    ldxs = list(Matrix(list(i)) for i in numpy.ndindex((2,)*kk) if sum(i))
    assert len(ldxs) == 2**kk-1
    logops = []
    for idx in numpy.ndindex((2,)*code.m):
        idx = Matrix(idx)
        for ldx in ldxs:
            l = idx*H + ldx*L
            d = l.sum()  # CSS weight only FIX
            assert d>=code.d
            if d==code.d:
                logops.append(l)
                print(l, strop(l))

    print("logops:", len(logops))
    del L

    def dump_ft(E):
        #for i in range(2*code.n):
        for i in range(1,2*code.n,2):
            u = E[:,i]
            print(u, end=" ")
            for l in logops:
                lu = Matrix(l.A * u.A)
                c = lu.sum() or "." #if lu==u else "."
                print(c, end="")
            print(" ", i)

    dump_ft(E)
    #print(code.longstr())
    print()

    #return

    # find a fault tolerant unitary encoder U:

    solver = Solver()
    add = solver.add

    m = code.m
    U = UMatrix.unknown(*E.shape)
    V = UMatrix.unknown(m,m)
    W = UMatrix.unknown(m,m)
    add(V*W==Matrix.identity(m))
    add(U*M == H.t*V)
    add(U.t*space.F*U == space.F)

    #for i in range(1,2*code.m,2):
    for i in range(1,2*code.n,2):
        print(i, end=" ")
        u = U[:,i]
        #print(u, end=" ")
        for l in logops:
            lu = UMatrix(l.A * u.A)
            items = list(lu)
            items = [item.get() for item in items] # if str(item)!="0"]
            #print(item, type(item))
            #add(If(lu==u, Not(And(*[item for item in items])), True))
            #add(Not(And(*[item for item in items])))
            add(PbLe([(i,True) for i in items], 1))
        #print(i)

    def x_type(i):
        for j in range(n):
            add(U[2*j+1, 2*i] == 0) # X type
            add(U[2*j, 2*i+1] == 0) # Z type
    def z_type(i):
        for j in range(n):
            add(U[2*j, 2*i] == 0) # Z type
            add(U[2*j+1, 2*i+1] == 0) # X type

    css = code.to_css()
    mx, mz = css.mx, css.mz
    for i in range(n):
        if i < mx:
            x_type(i)
        elif i < mx+mz:
            z_type(i)
        else:
            x_type(i)


    print()
    print("solve:")
    count = 0
    while 1:
        result = solver.check()
        print(result)
        if str(result) != "sat":
            break
        count += 1
    
        model = solver.model()
        U1 = U.get_interp(model)
        V1 = V.get_interp(model)
        W1 = W.get_interp(model)
        print()
        print("U1:")
        print(U1)
        print()
        assert (V1*W1) == Matrix.identity(m)
        dump_ft(U1)

        add(U != U1)
    
        dode = QCode.from_encoder(U1, k=code.k)
        assert (dode.is_equiv(code))

        print(dode.longstr())
        #print(".", end='', flush=True)
        break

    #print(count)


def test_ancilla():
    "fault tolerant state prep with ancilla"

    #code = construct.get_surface(3,3)
    code = construct.get_512() # sat
    code = construct.get_713() # unsat
    #code = construct.get_toric(2,2) # [[8,2,2]] sat
    #code = construct.get_10_2_3() # unsat
    #code = construct.get_913() # unsat
    #code = construct.get_toric(3,3) # [[18,2,3]] ??

    # ancilla
    a = 6
    ancilla = QCode.from_encoder(Matrix.identity(2*a), k=a)
    #print(ancilla)
    #print(ancilla.longstr())

    src, code = code, code + ancilla

    space = code.space
    n = code.n
    nn = 2*n

    print(code)
    print(code.longstr())
    print()

    if 1:
        E = code.get_encoder()
        H = code.H
        L = code.L

        print("E:")
        print(E, E.shape)
        print()
    
        print("H:")
        print(H, H.shape)
        print()
    
        print("L:")
        print(L, L.shape)
        print()

        #return
    
        M = []
        for i in range(code.m):
            u = [0]*nn
            u[2*i] = 1
            M.append(u)
        M = Matrix(M).t
        EM = E*M
        #print(EM, EM.shape)
        assert EM == H.t

    print("finding logops...")
    rows = []
    kk = 2*src.k
    H = code.H
    L = code.L[:kk]
    ldxs = list(Matrix(list(i)) for i in numpy.ndindex((2,)*kk) if sum(i))
    assert len(ldxs) == 2**kk-1
    logops = []
    for idx in numpy.ndindex((2,)*code.m):
        idx = Matrix(idx)
        for ldx in ldxs:
            l = idx*H + ldx*L
            d = l.sum()  # CSS weight only FIX
            assert d>=src.d
            if d==src.d:
                logops.append(l)
                print(l)

    print("logops:", len(logops))
    del L

    def dump_ft(E):
        #for i in range(2*code.n):
        for i in range(1,2*code.n,2):
            u = E[:,i]
            print(u, end=" ")
            for l in logops:
                lu = Matrix(l.A * u.A)
                c = lu.sum() or "." #if lu==u else "."
                print(c, end="")
            print(" ", i)

    dump_ft(E)
    #print(code.longstr())
    print()

    #return

    # find a fault tolerant unitary encoder U:

    solver = Solver()
    add = solver.add

    m = code.m
    U = UMatrix.unknown(*E.shape)
    V = UMatrix.unknown(m,m)
    W = UMatrix.unknown(m,m)
    add(V*W==Matrix.identity(m))
    add(U*M == H.t*V)
    add(U.t*space.F*U == space.F)

    #for i in range(1,2*code.m,2):
    for i in range(1,2*code.n,2):
        print(i, end=" ")
        u = U[:,i]
        print(u, end="\n")
        for l in logops:
            lu = UMatrix(l.A * u.A)
            print('\t', lu)
            items = list(lu)
            items = [item.get() for item in items] # if str(item)!="0"]
            #print(item, type(item))
            #add(If(lu==u, Not(And(*[item for item in items])), True))
            #add(Not(And(*[item for item in items])))
            add(PbLe([(i,True) for i in items], 1))
        #print(i)

    def x_type(i):
        for j in range(n):
            add(U[2*j+1, 2*i] == 0) # X type
            add(U[2*j, 2*i+1] == 0) # Z type
    def z_type(i):
        for j in range(n):
            add(U[2*j, 2*i] == 0) # Z type
            add(U[2*j+1, 2*i+1] == 0) # X type

    css = code.to_css()
    mx, mz = css.mx, css.mz
    for i in range(n):
        if i < mx:
            x_type(i)
        elif i < mx+mz:
            z_type(i)
        else:
            x_type(i)


    print()
    print("solve:")
    count = 0
    while 1:
        result = solver.check()
        print(result)
        if str(result) != "sat":
            break
        count += 1
    
        model = solver.model()
        U1 = U.get_interp(model)
        V1 = V.get_interp(model)
        W1 = W.get_interp(model)
        print()
        print("U1:")
        print(U1)
        print()
        assert (V1*W1) == Matrix.identity(m)
        dump_ft(U1)

        add(U != U1)
    
        dode = QCode.from_encoder(U1, k=code.k)
        assert (dode.is_equiv(code))

        print(dode.longstr())
        #print(".", end='', flush=True)
        break

    #print(count)


def test_prep():
    "fault tolerant state prep "

    #code = construct.get_surface(3,3)
    #code = construct.get_513() # unsat
    #code = construct.get_512()
    code = QCode.fromstr("XXXXII IIXXXX ZZZZII IIZZZZ IYIYIY ZIZIZI")

    space = code.space
    n = code.n
    nn = 2*n

    print(code)
    print(code.longstr())
    print()

    E = code.get_encoder()
    H = code.H
    L = code.L

    if 0:
        print("E:")
        print(E, E.shape)
        print()
    
        print("H:")
        print(H, H.shape)
        print()
    
        print("L:")
        print(L, L.shape)
        print()

    #return

    M = []
    for i in range(code.m):
        u = [0]*nn
        u[2*i] = 1
        M.append(u)
    M = Matrix(M).t
    EM = E*M
    #print(EM, EM.shape)
    assert EM == H.t

    def weight(l):
        s = strop(l)
        return len(s) - s.count('.')
    assert weight(numpy.array([1,0,0,0])) == 1
    assert weight(numpy.array([1,1,0,0])) == 1
    assert weight(numpy.array([1,1,0,1])) == 2

    print("finding minimum weight logops...")
    rows = []
    kk = 2*code.k
    H = code.H
    L = code.L[:kk]
    ldxs = list(Matrix(list(i)) for i in numpy.ndindex((2,)*kk) if sum(i))
    assert len(ldxs) == 2**kk-1
    logops = []
    for idx in numpy.ndindex((2,)*code.m):
        idx = Matrix(idx)
        for ldx in ldxs:
            l = idx*H + ldx*L
            d = weight(l)
            assert d>=code.d
            if d==code.d:
                logops.append(l)
                print(strop(l))
    print("logops:", len(logops))
    del L

    print("E =")
    print(E)
    def dump_ft(E):
        print("dump_ft:")
        #for i in range(2*code.n):
        for i in range(1,2*code.n,2):
            u = E[:,i]
            print(u, end=" ")
            for l in logops:
                lu = Matrix(l.A * u.A)
                c = lu.sum() or "." #if lu==u else "."
                print(c, end="")
            print(" i=%d"%i)

    dump_ft(E)
    #print(code.longstr())
    print()

    # find a fault tolerant unitary encoder U:

    solver = Solver()
    add = solver.add

    m = code.m
    U = UMatrix.unknown(*E.shape)
    V = UMatrix.unknown(m,m)
    W = UMatrix.unknown(m,m)
    add(V*W==Matrix.identity(m))
    add(U*M == H.t*V)
    add(U.t*space.F*U == space.F)

    #for i in range(1,2*code.m,2):
    for i in range(1,2*code.n,2):
        print(i, end=" ")
        u = U[:,i]
        #print(u, end=" ")
        for l in logops:
            lu = UMatrix(l.A * u.A)
            items = list(lu)
            items = [item.get() for item in items] # if str(item)!="0"]
            #print(item, type(item))
            #add(If(lu==u, Not(And(*[item for item in items])), True))
            #add(Not(And(*[item for item in items])))
            add(PbLe([(i,True) for i in items], 1))
        #print(i)

#    def x_type(i):
#        for j in range(n):
#            add(U[2*j+1, 2*i] == 0) # X type
#            add(U[2*j, 2*i+1] == 0) # Z type
#    def z_type(i):
#        for j in range(n):
#            add(U[2*j, 2*i] == 0) # Z type
#            add(U[2*j+1, 2*i+1] == 0) # X type
#
#    css = code.to_css()
#    mx, mz = css.mx, css.mz
#    for i in range(n):
#        if i < mx:
#            x_type(i)
#        elif i < mx+mz:
#            z_type(i)
#        else:
#            x_type(i)


    print()
    print("solve:")
    count = 0
    while 1:
        result = solver.check()
        print(result)
        if str(result) != "sat":
            break
        count += 1
    
        model = solver.model()
        U1 = U.get_interp(model)
        V1 = V.get_interp(model)
        W1 = W.get_interp(model)
        print()
        print("U1:")
        print(U1)
        print()
        assert (V1*W1) == Matrix.identity(m)
        if code.k:
            dump_ft(U1)

        add(U != U1)
    
        dode = QCode.from_encoder(U1, k=code.k)
        assert (dode.is_equiv(code))

        print(dode.longstr())
        #print(".", end='', flush=True)
        #break

    #print(count)


def test_goto():
    # See: https://www.nature.com/articles/srep19578

    n = 8
    syntax = Syntax()
    CX, H = syntax.CX, syntax.H

    prog = (CX(6,7)*CX(5,7)*CX(0,7)
        *CX(6,4)*CX(1,5)*CX(3,6)*CX(2,0)
        *CX(1,4)*CX(2,6)*CX(3,5)*CX(1,0)
        *H(1)*H(2)*H(3))

    print(prog)

    space = SymplecticSpace(n)
    E = prog*space

    #print(E)
    print(strop(E.t))

    "IIIZZZZ ZIIIIZZ IZIIZIZ IIZIZZI"
    

def find_chaimap(src, tgt, injective=False):
    
    # mx ---Hx.t--> n --Hz--> mz : 0
    # |             |         |
    # |fx           |fn       |fz
    # |             |         |
    # v             v         v
    # mx ---Hx.t--> n --Hz--> mz : 1

    src = src.to_css()
    tgt = tgt.to_css()

    Hz0, Hxt0 = Matrix(src.Hz), Matrix(src.Hx).t
    Hz1, Hxt1 = Matrix(tgt.Hz), Matrix(tgt.Hx).t
    assert ( Hz0 * Hxt0 ).is_zero()
    assert ( Hz1 * Hxt1 ).is_zero()

    mx0, mx1 = Hxt0.shape[1], Hxt1.shape[1]
    n0, n1 = Hz0.shape[1], Hz1.shape[1]
    assert (n0, n1) == (Hxt0.shape[0], Hxt1.shape[0])
    mz0, mz1 = Hz0.shape[0], Hz1.shape[0]

    solver = Solver()
    add = solver.add

    fx = UMatrix.unknown(mx1, mx0) # mx1<--mx0
    fn = UMatrix.unknown(n1, n0) # n1<--n0
    fz = UMatrix.unknown(mz1, mz0) # mx1<--mx0

    if injective:
        for i in range(n1):
            add(PbLe([(fn[i,j].get(),True) for j in range(n0)], 1))
        for j in range(n0):
            add(PbLe([(fn[i,j].get(),True) for i in range(n1)], 1))

    add(Hxt1*fx == fn*Hxt0)
    add(Hz1*fn == fz*Hz0)

    print()
    print("solve:")
    count = 0
    prev = None
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break
        count += 1
    
        model = solver.model()
        _fx = fx.get_interp(model)
        _fn = fn.get_interp(model)
        _fz = fz.get_interp(model)

        assert (Hxt1*_fx == _fn*Hxt0)
        assert (Hz1*_fn == _fz*Hz0)

        result = (_fx, _fn, _fz)
        assert result != prev
        yield result
        prev = (_fx, _fn, _fz)

        add(Not(And(fx==_fx, fn==_fn, fz==_fz)))
        add(fn != _fn)

        #break
        #if count>10:
        #    break

    print("count =", count)





def test_transversal():

    #code = construct.get_surface(3,3)
    #code = construct.get_512()
    #code = construct.get_713()
    #code = construct.get_toric(2,2)
    #code = construct.get_10_2_3()
    #code = construct.get_913()
    #code = construct.get_toric(3,3)

    #src = tgt = code
    #src = construct.get_surface(3,3)
    #tgt = construct.get_surface(4,4)

    tgt = construct.get_toric(2,2)
    src = construct.get_toric(4,4)

    #tgt = construct.get_512()
    #tgt = construct.get_713()
    #tgt = construct.get_10_2_3()

    #print(src.longstr())

    code = src + tgt
    print(src, tgt)
    #print(code.longstr())
    space = code.space

    found = set()
    logical = set()
    for (fx, fn, fz) in find_chaimap(src, tgt, True):
        #print(".", end="", flush=True)
        #print(fn)
        #print("-"*20)
        assert fn not in found
        found.add(fn)

        m,n = fn.shape
        assert fn.shape == (tgt.n, src.n)
        op = space.get_identity()
        for i in range(m): # tgt index
          for j in range(n): # src index
            if fn[i,j]:
                op = op * space.CX(j,i+n) # src is ctrl, tgt is tgt
                #op = op * space.CX(i+n,j) # nope
        #print(op)

        dode = op*code
        succ = dode.is_equiv(code)
        assert succ

        L = dode.get_logical(code)
        if L not in logical:
            logical.add(L)
            print("logical:")
            print(L)
            print("fn:")
            print(fn)



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




