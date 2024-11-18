#!/usr/bin/env python

"""
find automorphisms of QCode's that permute the qubits.
"""

from time import time
start_time = time()
from random import shuffle, choice

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, zeros2, solve2, normal_form, enum2)
from qumba.qcode import QCode, SymplecticSpace
from qumba import construct
from qumba.argv import argv


def get_autos_slow(code):
    n, m = code.n, code.m

    H = code.H
    m, nn = H.shape
    Ht = H.transpose()
    #Ht.shape = (m, nn//2, 2)

    def accept(idxs):
        iidxs = []
        for i in idxs:
            iidxs.append(2*i)
            iidxs.append(2*i+1)
        A = Ht[iidxs, :]
        B = Ht[:len(iidxs), :]
        U = solve2(A, B)
        return U is not None

    cols = list(range(n))

    gen = []
    stack = [0]
    idx = 0
    while 1:
      while len(stack) < n and idx < n:
        #assert accept(stack)
        #print("stack:", stack, "idx:", idx)
        while idx < n:
            if idx not in stack and accept(stack + [idx]):
                stack.append(idx)
                idx = 0
                break
            idx += 1
        else:
            idx = stack.pop()+1
            while idx >= n and len(stack):
                idx = stack.pop()+1

      if not stack:
          break
      print(stack)
      gen.append(stack)
      if len(gen) > 200:
        return
      dode = code.apply_perm(stack)
      assert dode.is_equiv(code)

      idx = stack.pop()+1
      while idx >= n and len(stack):
          idx = stack.pop()+1

    print(len(gen))


def get_isos(src, tgt):
    if src.n != tgt.n or src.m != tgt.m:
        return 

    n, m = src.n, src.m

    H = src.H.A.copy()
    m, nn = H.shape

    rhs = normal_form(tgt.H.A.copy())
    #print(shortstr(rhs))

    forms = [None]
    for i in range(1, 1+nn//2):
        H1 = normal_form(H[:, :2*i])
        #print("normal_form")
        #print(shortstr(H1))
        #print()
        forms.append(H1)

    def accept(idxs):
        iidxs = []
        for i in idxs:
            iidxs.append(2*i)
            iidxs.append(2*i+1)
        lhs = H[:, iidxs]
        #print("accept", idxs, iidxs)
        #print("lhs:")
        #print(lhs, lhs.shape)
        lhs = normal_form(lhs, False)
        #print("normal_form:")
        #print(lhs, lhs.shape)
        #print([f.shape for f in forms[1:]])
        return eq2(lhs, rhs[:, :len(iidxs)])

    cols = list(range(n))

    gen = []
    stack = [0]
    idx = 0
    while 1:
      while len(stack) < n and idx < n:
        #assert accept(stack)
        #print("stack:", stack, "idx:", idx)
        while idx < n:
            if idx not in stack and accept(stack + [idx]):
                stack.append(idx)
                idx = 0
                break
            idx += 1
        else:
            idx = stack.pop()+1
            while idx >= n and len(stack):
                idx = stack.pop()+1

      if not stack:
          break
      #print(stack)
      #gen.append(stack)
      yield list(stack)
      #if len(gen) > 4000:
      #  return
      dode = src.apply_perm(stack)
      assert dode.is_equiv(tgt)
      #print("is_equiv!")

      idx = stack.pop()+1
      while idx >= n and len(stack):
          idx = stack.pop()+1


def is_iso(code, dode):
    for iso in get_isos(code, dode):
        return True
    return False


def get_autos(code):
    return list(get_isos(code, code))


def test():
    #code = construct.get_713()
    #code = construct.get_rm()
    #code = construct.get_m24()

    code = construct.get_10_2_3()
    gen = get_autos(code)
    assert len(gen) == 20


def test_css():
    H = parse("""
    X....XX.XXX....X..XXXXX
    .X...X.XX.XX.X..XXX.X.X
    ..X....XXX..X.X.XXXX.XX
    ...X....X.XX.XXXXX.XX.X
    ....XXX..X.XX..XX.XXX.X
    """) # [[23, 13, 3]] |G|=21504 

    _H = parse("""
    X.....XX.XX.XX...X...X..X
    .X....X..X....X.XX.X..XXX
    ..X...X.X..XX...XX..XXX..
    ...X.....XXX.XXX.XXX.....
    ....X..X....XXX.XXX.X..X.
    .....XX.X.XXXX.XX...X....
    """) # autos: 8

    _H = parse("""
    X......XX.X.X..X.X..X.XXXX.
    .X....XX.XXXX...X.X..X...XX
    ..X...X.X.X..XXXX.X..XX...X
    ...X.....XXXXX.....XXXX.XX.
    ....X.X.X.XXX.X..XXX..X..X.
    .....X...XXXX.XXX.X...XXX..
    """)

    _H = parse("""
    XXXX.X.XX..X...
    XXX.X.XX..X...X
    XX.X.XX..X...XX
    X.X.XX..X...XXX
    """) # [[15, 7, 3]] |G|=20160, is G=M_21=L_3(4) or G=Alt(8) ?

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
    """) # Golay

    #print(H)
    from qumba.csscode import CSSCode
    code = CSSCode(Hx=H, Hz=H)
    if code.k:
        code.bz_distance()
    print(code)

    m, n = H.shape
    wenum = {i:[] for i in range(n+1)}
    span = []
    for u in enum2(m):
        v = dot2(u, H)
        d = v.sum()
        wenum[d].append(v)
        if d:
            span.append(v)

    print([len(wenum[i]) for i in range(n+1)])

    #for d in range(1, n+1):
    #    if wenum[d]:
    #        break
    #for v in wenum[d]:
    #    print(shortstr(v))

    from pynauty import Graph, autgrp
    N = len(span)
    graph = Graph(N+n)

    colours = {d:[] for d in range(n+1)}
    for idx, v in enumerate(span):
        #print(idx, v, v.sum())
        d = v.sum()
        colours[d].append(idx)
        for i in range(n):
            if v[i]:
                graph.connect_vertex(N+i, idx)
    
    labels = []
    for d in range(n+1):
        if colours[d]:
            labels.append(colours[d])
    labels.append(list(range(N, N+n)))
    print([len(lbl) for lbl in labels])

    labels = [set(l) for l in labels]

    fix = labels[1].pop()
    labels.insert(0, {fix})

    items = []
    for l in labels:
        items += list(l)
    items.sort()
    #print(items, N+n)
    assert items == list(range(N+n))
    print(N+n, "vertices")


    graph.set_vertex_coloring(labels)

    aut = autgrp(graph)
    print(len(aut))
    gen = aut[0]
    order = int(aut[1])
    print("autos:", order)
    #for perm in gen:
    #    print(perm)

    for perm in gen:
        f = [perm[i] - N for i in range(N, N+n)]
        dode = code.apply_perm(f)
        assert dode.is_equiv(code)
    



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





