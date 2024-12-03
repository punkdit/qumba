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
from qumba.action import mulclose
from qumba import construct
from qumba.argv import argv


def very_slow_get_autos(code):
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


def slow_get_isos(src, tgt):
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


def slow_is_iso(code, dode):
    for iso in get_isos(code, dode):
        return True
    return False


def slow_get_autos(code):
    return list(get_isos(code, code))


def get_autos_selfdualcss(code):

    #assert code.tp == "selfdualcss"
    css = code.to_css()
    assert eq2(css.Hx, css.Hz)
    H = css.Hx

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

    print("building...", end=" ", flush=True)
    colours = {d:[] for d in range(n+1)}
    for idx, v in enumerate(span):
        #print(idx, v, v.sum())
        d = v.sum()
        colours[d].append(idx)
        for i in range(n):
            if v[i]:
                graph.connect_vertex(N+i, idx)
    print(" done.")
    
    labels = []
    for d in range(n+1):
        if colours[d]:
            labels.append(colours[d])
    labels.append(list(range(N, N+n)))
    print([len(lbl) for lbl in labels])

    labels = [set(l) for l in labels]

    #fix = labels[1].pop()
    #labels.insert(0, {fix})

    items = []
    for l in labels:
        items += list(l)
    items.sort()
    #print(items, N+n)
    assert items == list(range(N+n))
    print(N+n, "vertices")

    graph.set_vertex_coloring(labels)

    print("autgrp...")
    aut = autgrp(graph)
    print(len(aut))
    gen = aut[0]
    order = int(aut[1])
    print("autos:", order)
    #for perm in gen:
    #    print(perm)

    code = code.to_qcode()
    ops = []
    for perm in gen:
        f = [perm[i] - N for i in range(N, N+n)]
        dode = code.apply_perm(f)
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        #print(L)
        #print()
        ops.append(L)

    dode = code.apply_S()
    ops.append(dode.get_logical(code))
    assert dode.is_equiv(code)
    dode = code.apply_H()
    ops.append(dode.get_logical(code))
    assert dode.is_equiv(code)

    G = mulclose(ops, verbose=True)
    print("|G| =", len(G))


class Graph:
    def __init__(self, directed=False):
        self.directed = directed
        self.verts = []
        self.edges = []
        self.e_colours = {} # map colour -> vert

    def vert(self, colour=None):
        assert colour is not None
        i = len(self.verts)
        self.verts.append(colour)
        return i

    def edge(self, i, j, colour=None):
        if colour is None:
            self.edges.append((i, j))
            return
        e_colours = self.e_colours
        c_vert = e_colours.get(colour)
        if c_vert is None:
            c_vert = self.vert(colour)
            e_colours[colour] = c_vert
        vert = self.vert("e")
        self.edge(i, vert)
        self.edge(j, vert)
        self.edge(vert, c_vert)

    def to_dot(self, fname):
        f = open(fname, "w")
        print("graph {", file=f)
        for i,lbl in enumerate(self.verts):
            print('  %s [label="%s:%s"];'%(i,lbl,i), file=f)
        for (i,j) in self.edges:
            print("  %s -- %s;" % (i,j), file=f)
        print("}", file=f)
        f.close()

    def __str__(self):
        return "Graph(%s, %s)"%(self.verts, self.edges)
    __repr__ = __str__

    def get_autos(self):
        from pynauty import Graph, autgrp
        verts, edges = self.verts, self.edges
        N = len(verts)
        labels = {k:set() for k in set(verts)}
        for idx,label in enumerate(verts):
            labels[label].add(idx)
        directed = False # True is much slower ?!?!
        graph = Graph(N, directed)

        print("Graph: |verts|=%d ... " % N, end="", flush=True)
        adj = {i:[] for i in range(N)}
        for (i,j) in edges:
            adj[i].append(j)
            #graph.connect_vertex(i, j)
        graph.set_adjacency_dict(adj)
        #print(labels)
        labels = list(labels.values())
        #print(labels)
        graph.set_vertex_coloring(labels)
    
        print("autgrp ...", end="", flush=True)
        aut = autgrp(graph)
        print(" done")
        gen = aut[0]
        order = int(aut[1])
        return gen, order


def _get_autos(code, span, keys):

    n = code.n
    graph = Graph()
    v_bits = [graph.vert("q") for i in range(n)]
    #for key, us in span.items():
    #  for u in us:
    for key in keys:
      for u in span[key]:
        #print(str(v).replace("\n ",""), wx, wz, wy)
        v_check = graph.vert(colour=key)
        for i in range(n):
            if u[i] == 1:
                graph.edge(v_check, v_bits[i], colour="X")
            elif u[i] == 256:
                graph.edge(v_check, v_bits[i], colour="Z")
            elif u[i] == 257:
                graph.edge(v_check, v_bits[i], colour="Y")
            else:
                assert u[i] == 0

    #print(graph)
    #graph.to_dot("code.dot")

    gen, order = graph.get_autos()
    #for f in gen:
    #    print({i:j for (i,j) in enumerate(f)})
    gen = [f[:n] for f in gen]

    ops = []
    for f in gen:
        dode = code.apply_perm(f)
        if not dode.is_equiv(code):
            return None
        L = dode.get_logical(code)
        #print(L)
        #print()
        ops.append(L)

    return gen, order, ops


def get_autos(code):

    #assert code.tp == "selfdualcss"
    code = code.to_qcode()
    H = code.H.A
    m, nn = H.shape
    assert nn%2 == 0
    n = nn//2
    assert H.dtype == numpy.int8
    #H = H.reshape(m, n, 2)
    assert m < 30

    N = 2**m
    print("get_autos...", end=" ", flush=True)

    span = {}
    for u in enum2(m):
        u = numpy.array(u, dtype=numpy.int8)
        #print(u.dtype)
        #u = u.reshape(1,m)
        v = dot2(u, H)
        v.shape = (n, 2)
        u = numpy.frombuffer(v.tobytes(), dtype=numpy.int16)
        wx = int((u==1).sum())
        wz = int((u==256).sum())
        wy = int((u==257).sum())
        key = (wx, wz, wy)
        span.setdefault(key, []).append(u)

    #for (key,value) in span.items():
    #    print('\t', key, len(value) )

    keys = list(span.keys())
    keys.remove((0,0,0))
    keys.sort(key = lambda k:sum(k))
    #print(keys)
    w = sum(keys[0])
    #print(keys)

    print("N=%d"%N)

    #last = 0
    idx = 1
    while idx <= len(keys):
        #w_keys = [k for k in keys if sum(k)<=w]
        w_keys = keys[:idx]
        #if len(w_keys) > last:
        result = _get_autos(code, span, w_keys)
        if result is not None:
            return result
        #last = len(w_keys)
        idx += 1

def get_isos(code, dode):
    return []

def is_iso(code, dode):
    return False


def get_autos_css(code):
    from qumba.transversal import find_lw_css
    code = code.to_css()

    print("get_autos_css", code)

    print("find_lw_css ... ", end="", flush=True)
    hx, hz = find_lw_css(code)
    wx = hx[0].sum()
    wz = hz[0].sum()
    print(wx, len(hx), "--", wz, len(hz))

    hx = [h.A.reshape(code.n) for h in hx]
    hz = [h.A.reshape(code.n) for h in hz]

    span = {(wx,0,0):hx, (0,wz,0):hz}
    result = _get_autos(code.to_qcode(), span, list(span.keys()))

    return result


def test_bring():
    code = construct.get_bring()
    code.bz_distance()
    print(code)

    result = get_autos_css(code)
    assert result is not None
    gen, order, ops = result
    L = mulclose(ops, verbose=True)
    print("|G| =", order, "logicals:", len(L))

    from qumba.transversal import find_isomorphisms_css
    code = code.to_qcode()
    dode = code.get_dual()
    for g in find_isomorphisms_css(code, dode, ffinv=True):
        break
    print(g)


def test_logicals():
    from qumba import db
    _id = "6705229919cca60cf657a904"
    code = list(db.get(_id = _id))[0]
    print(code)
    gen, order, ops = get_autos_css(code)
    L = mulclose(ops, verbose=True)
    print("|G| =", order, "logicals:", len(L))

    space = SymplecticSpace(code.k)
    for g in L:
        name = space.get_name(g)
        if len(name) == 1:
            print(name)


def test():
    from qumba import db
    n = argv.get("n")
    d = argv.get("d")
    ns = range(17, 28) if n is None else [n]
    for n in ns:
        kw = {"n":n, "css":True}
        if d is not None:
            kw["d"] = d
        for code in db.get(**kw):
            result = get_autos_css(code)
            if result is not None:
                gen, order, ops = result
                L = mulclose(ops, verbose=True)
                print("|G| =", order, "logicals:", len(L))
            else:
                print("fail")
            print()

    return

    from qumba.construct import biplanar
    #code = biplanar(6, 12)              # [[36, 8, 4]]      |G| = 144
    #code = construct.biplanar(12, 12)   # [[72, 12, 6]]     |G| = 432
    code = construct.biplanar(24, 6)    # [[72, 8, 6]]      |G| = 72
    #code = construct.biplanar(18, 12)   # [[108, 8, 10]]    |G| = 108
    #code = construct.biplanar(24, 12)   # [[144, 12, 12?]]  |G| = 144
    code = construct.biplanar(30, 6)    # [[90, 8, 10]]     |G| = 90
    code = construct.biplanar(30, 12)   # [[180, 8, ?]]     |G| = 90
    code = construct.biplanar(60, 12)   # [[360, 12, ?]]    |G| = 180
    code = construct.biplanar(48, 12)   # [[288, 12, 18?]]   |G| = 144
    code = construct.biplanar(84, 12)   # [[504, 12, ?]]   |G| = 

    return

    from qumba.transversal import find_isomorphisms
    for n in range(10, 18):
    #for n in range(20, 30):
        for code in db.get(n=n, desc="codetables"):
            #if code.n - code.k > 22:
            #    continue
            print(code)
            gen, order, ops = get_autos(code)
            #print(gen, len(gen))
            print("|G| =", order)

            count = 0
            for g in find_isomorphisms(code):
                print(".", end="", flush=True)
                count += 1
            assert count == order

            print()
            
    for code,N in [
        (QCode.fromstr("XI IZ"), 1),
        (QCode.fromstr("XX ZZ"), 2),
        (construct.get_412(), 4),
        (construct.get_713(), 168),
        (construct.get_10_2_3(), 20),
        (construct.reed_muller(), 322560),
    ]:
        gen, order, ops = get_autos(code)
        assert order == N, order
        #print(gen, len(gen))


def test_css():
    H = parse("""
    X....XX.XXX....X..XXXXX
    .X...X.XX.XX.X..XXX.X.X
    ..X....XXX..X.X.XXXX.XX
    ...X....X.XX.XXXXX.XX.X
    ....XXX..X.XX..XX.XXX.X
    """) # [[23, 13, 3]] |G|=21504 , logicals = 129024 

    _H = parse("""
    X.....XX.XX.XX...X...X..X
    .X....X..X....X.XX.X..XXX
    ..X...X.X..XX...XX..XXX..
    ...X.....XXX.XXX.XXX.....
    ....X..X....XXX.XXX.X..X.
    .....XX.X.XXXX.XX...X....
    """) # autos: 8

    H = parse("""
    X......XX.X.X..X.X..X.XXXX.
    .X....XX.XXXX...X.X..X...XX
    ..X...X.X.X..XXXX.X..XX...X
    ...X.....XXXXX.....XXXX.XX.
    ....X.X.X.XXX.X..XXX..X..X.
    .....X...XXXX.XXX.X...XXX..
    """) # |G|=24, logicals = 144

    _H = parse("""
    XXXX.X.XX..X...
    XXX.X.XX..X...X
    XX.X.XX..X...XX
    X.X.XX..X...XXX
    """) # [[15, 7, 3]] |G|=20160, is G=M_21=L_3(4) or G=Alt(8) ? logicals = 120960

    _H = parse("""
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
    """) # Golay |G|=244823040 G=M_24

    _H = parse("""
X.........XX.X.X.XXX.
.X.........XXX.X...X.
..X.........X.X..X.XX
...X.....X.X.X.XX..XX
....X....X.XX.X...X..
.....X.......XX.XXXXX
......X..XXX.XX...X.X
.......X..X.X.XXX....
........XXX..XXXXX...
    """) # [[21, 3, 5]] |G| = 5760, logicals = 36

    _H = parse("""
XX.X.X...XXXX........
X.X.X...XXXX........X
.X.X...XXXX........XX
X.X...XXXX........XX.
.X...XXXX........XX.X
X...XXXX........XX.X.
...XXXX........XX.X.X
..XXXX........XX.X.X.
.XXXX........XX.X.X..
    """) # [[21, 3, 5]] |G| = 120960 , logicals = 36

    #print(H)
    from qumba.csscode import CSSCode
    code = CSSCode(Hx=H, Hz=H)
    if code.k:
        code.bz_distance()
    print(code)

    get_autos_selfdualcss(code)




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





