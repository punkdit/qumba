#!/usr/bin/env python3

"""
Algebraic groups: matrix groups over Z/pZ.

"""


from random import shuffle
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

from qumba.qcode import QCode, get_weight
from qumba.solve import shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2
from qumba.matrix import Matrix, SymplecticSpace


def cnots_priority(src, tgt):
    src = src.to_qcode()
    tgt = tgt.to_qcode()
    print("cnots_priority:", src, "-->", tgt)

    n = src.n

    pairs = []
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        pairs.append((idx, jdx))

    space = SymplecticSpace(src.n)
    gates = []
    for (i0,i1) in pairs:
      M = space.get_CNOT(i0, i1)
      for (j0,j1) in pairs:
        if i0==j0 or i0==j1 or i1==j0 or i1==j1:
            continue
        MN = M*space.get_CNOT(j0,j1)
        gates.append(MN)
    print(len(gates))

    src.get_distance()
    src.name = []
    src.weight = src.get_overlap(tgt)
    found = set([src])
    bdy = list(found)
    count = 0
    while bdy: # and count < 3:
      count += 1

      code = bdy.pop()
      print(". \tbdy:", len(bdy), len(found), code.weight)
      E = code.get_symplectic()
      E = Matrix.promote(E)
      for M in gates:
        ME = M*E
        dode = QCode.from_symplectic(ME.A, code.m)
        if dode.get_distance() < code.d:
            continue
        if dode in found:
            continue
        dode.name = [(idx,jdx)] + code.name
        #print((idx, jdx), dode)
        dode.weight = dode.get_overlap(tgt)
        if dode.weight == tuple(range(tgt.m, tgt.m+tgt.k+1)):
            print()
            yield dode
        #elif code.n > 8:
            #print("%d%d."%(idx,jdx), end="", flush=True)
            #print(".", end="", flush=True)
        print("%s%s%s."%dode.weight, end="", flush=True)
        found.add(dode)
        bdy.append(dode)
      #shuffle(bdy)
      bdy.sort(key = lambda code : (code.weight, -len(code.name)))
      #bdy.sort(key = lambda code : code.weight)
      #print([(c.weight, len(c.name)) for c in bdy], tgt.m)
    print()


def test():

    from qumba import construct
    code = construct.get_713()
    code = construct.get_513()
    #code = construct.toric(2,2)
    code = construct.get_10_2_3()

    code = code.to_qcode()

    print("src:")
    print(code.longstr())

    D = Matrix(code.get_decoder())
    L = Matrix.identity(2*code.m) << SymplecticSpace(code.k).get_CNOT(0,1)
    E = Matrix(code.get_encoder())

    ELD = E*L*D
    n = code.n
    space = SymplecticSpace(n)
    assert space.is_symplectic(ELD)

    def metric(A, B):
        AB = (A.A + B.A)%2
        d = AB.astype(numpy.int64).sum()
        #print(shortstr(AB), d)
        #print()
        return d

    pairs = []
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        pairs.append((idx, jdx))

    print( (ELD).shortstr() )

    best = None
    while 1:
      found = set()
      M0 = space.identity()
      d0 = metric(M0, ELD)
      while d0:
        done = True
        shuffle(pairs)
        for (i,j) in pairs:
            #if i in found or j in found:
            #    continue
            M = space.get_CNOT(i,j)
            M1 = M*M0
            #print(M.shortstr())
            d1 = metric(M1, ELD)
            if d1 < d0:
                d0 = d1
                M0 = M1
                found.add(i)
                found.add(j)
                #print(M0.shortstr())
                print(found, d0, (i, j))
                done = False
        if done:
            break
      if best is None or d0 < best:
        print()
        print(M0 == ELD)
        D = (M0.A + ELD.A)%2
        print(shortstr(D))
        best = d0
      if best==0:
        break

    return

    dode = QCode.from_symplectic((E*L).A, code.m)
    #print(dode.get_params())
    #print(dode.longstr())
    assert code.equiv(dode)
    #print(code.get_logop(dode))

    print("tgt:")
    print(dode.longstr())

    

    src, tgt = code, dode
    #print(src.get_overlap(tgt))
    #print(tgt.get_overlap(tgt))
    #return

    I = zeros2(code.kk)
    for code in cnots_priority(src, tgt):
        print(code.longstr())
        print(code.name)
        L = src.get_logop(code)
        if not eq2(L, I):
            break
        
    print(L)
    assert code.equiv(tgt)

    return

    E = code.get_encoder()
    Ei = code.get_decoder()

    M = QCode.trivial(code.m)

    L = QCode.trivial(code.k)
    L = L.CNOT(0, 1) # logical gate

    logop = (M+L).get_encoder()

    A = dot2(E, logop, Ei)
    n = 6
    nn = 2*n
    A = A[:nn,:nn]

    tgt = Matrix(A, name=[])

    gen = []
    for idx in range(n):
      for jdx in range(n):
        if idx==jdx:
            continue
        A = identity2(nn)
        A[2*jdx+1, 2*idx+1] = 1
        A[2*idx, 2*jdx] = 1
        name="C(%d,%d)"%(idx, jdx)
        name = [(idx, jdx)]
        g = Matrix(A, name=name)
        gen.append(g)

    for src in find(tgt, gen):
        P = src.A
        P = P[::2, ::2]
        print(shortstr(P))
        perm = list(numpy.where(P)[1]) + list(range(len(P), code.nn))
        name = src.name
        print(name, len(name))

        dode = code
        for (idx, jdx) in reversed(name):
            dode = dode.CNOT(jdx, idx)
            print(dode.get_params())
        print(perm)
        dode = dode.permute(perm)
        assert dode.equiv(code)


def find(tgt, gen):
    gen = list(gen)
    nn = tgt.shape[0]
    I = Matrix.identity(nn)
    best = None

    #while 1:
    for trial in range(10000):
#      print()
      src = tgt
      while src != I:
        shuffle(gen)
        weight = src.sum()
#        print(weight-nn, end=" ")
        if weight == nn:
            if best is None or len(src.name) < len(best.name):
                print()
                print(shortstr(src.A))
                yield src
                best = src
            break

        for g in gen:
            gh = g*src
            if gh.sum() >= weight:
                continue
            src = gh
            break
        else:
            break
      else:
        #print(src.name)
        assert src == I


def search_1(tgt, gen):
    bdy = [tgt]
    while bdy:
        weight = bdy[0].sum()
        print(weight, len(bdy))
        _bdy = []
        for g in gen:
          for h in bdy:
            gh = g*h
            if gh.sum() >= weight:
                continue
            _bdy.append(gh)
        bdy = _bdy


def search(tgt, gen):
    weight = tgt.sum()
    count = 1
    found = set(gen)
    bdy = list(found)
    while bdy and count < 5:
        _bdy = []
        for g in gen:
          for h in bdy:
            gh = g*h
            if gh.sum() > weight:
                continue
            if gh in found:
                continue
            if gh == tgt:
                print("found", gh.name)
                return gh
            _bdy.append(gh)
            found.add(gh)

        bdy = _bdy
        print(len(found), len(bdy))
        count += 1




if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

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


