#!/usr/bin/env python

from random import shuffle, randint, choice
from operator import add, matmul, mul, lshift
from functools import reduce, cache

import numpy

from bruhat.gset import allgroups, Group, Perm, Coset
from bruhat.hecke import get_operators
from bruhat.gap import Gap

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba.symplectic import SymplecticSpace
from qumba.qcode import QCode
from qumba.util import get_complete_pairings, choose
from qumba import construct
from qumba.smap import SMap


def span_ops(ops):
    N = len(ops)
    m, n = ops[0].shape
    if N > argv.get("span_ops", 16):
        print("(span_ops: N=%d is too big?)" % N, end="\n", flush=True)
        for op in ops:
            yield op # at least we tried...
        return
    for bits in numpy.ndindex((2,)*N):
        if sum(bits) == 0:
            continue
        op = Matrix.zeros((m, n))
        for (i,bit) in enumerate(bits):
            if bit:
                op = op + ops[i]
        yield op

        
def get_weight(H):
    m, n = H.shape

    vecs = [[] for i in range(n+1)]
    for u in numpy.ndindex((2,)*m):
        v = Matrix(u) * H
        w = v.sum()
        vecs[w].append(v)

    rows = []
    for w,vs in enumerate(vecs):
        if not w or not len(vs):
            continue
        rows += vs
        H1 = Matrix(rows)
        H1 = H1.row_reduce()
        if len(H1) == m:
            return w
    assert 0



def get_css_weight(css):

    Hx = css.Hx
    Hz = css.Hz

    wx = get_weight(Hx)
    wz = get_weight(Hz)
    return wx, wz

    
def get_pairs(G, Hs):
    print("get_pairs ...", end=' ', flush=True)
    Xs = [G.left_action(H) for H in Hs]
    N = len(Xs)
    pairs = {}
    for i in range(N):
      for j in range(N):
        ops = list(get_operators(Xs[i].gens, Xs[j].gens))
        ops = [Matrix(op) for op in ops]
        if argv.show:
            print(i, j)
            for op in ops:
                print(op)
                print("-" * 7)
                print(op.normal_form())
                print("=" * 7)
            print()
        ops = [H.row_reduce() for H in span_ops(ops)]
        pairs[i,j] = ops
    print("done.")
    return pairs



def find_autos(H):
    if len(H) > 24:
        return

    print(H, H.shape)

    w = H.get_wenum()
    print(w)

    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, "x")
    x = R.gens()[0]
    f = 0
    for (i,c) in enumerate(w):
        f += c*(x**i)
    print(f)
    print(sage.factor(f))

    N, perms = H.get_autos(verbose=True)
    #print("|G| =", N, "(nauty)")
    perms = [Perm(idxs) for idxs in perms]
    from bruhat.gap import Gap
    gap = Gap()
    G = Group(None, perms)
    N = gap.Order(G, get=True)
    print("|G| =", N)

    if N <= 244823040: # M(24) 
        print(G.structure_description())

    if argv.M24:
    
        M = gap.MathieuGroup(24)
        _G = gap.define(G)
        print(gap.IsomorphicSubgroups(M, _G, get=True)) # yes

    print()



found = set()


def get_selfdual(G):
    print("\t", G, G.structure_description(), end=", ")

    #print(G.perms)
    N = G.rank
    n = gap.Order(G, get=True)
    assert len(G) == n
    if argv.all_subgroups:
        _Hs = gap.AllSubgroups(G)
    else:
        _Hs = gap.ConjugacyClassesSubgroups(G)
    #print("\t", gap.Length(Hs, get=True))
    Hs = []
    for i in range(len(_Hs)):
        item = _Hs[i]
        if not argv.all_subgroups:
            item = gap.Representative(item)
        assert gap.IsPermGroup(item, get=True)
        H = gap.to_group(item, N=N)
        #print("\t", H)
        assert G.is_subgroup(H)
        if len(G) > len(H) > 1: # right ?!?
            Hs.append(H)

    N = len(Hs)
    print("subgroups:", N)

    Xs = [G.left_action(Hs[i]) for i in range(N)]

    for i in range(len(Hs)):
      for j in range(i, len(Hs)):
        ops = list(get_operators(Xs[i].gens, Xs[j].gens))
        #print(ops[0].shape, end=' ')
        ops = [Matrix(op) for op in ops]

        for H0 in span_ops(ops):
          m, n = H0.shape
          if argv.code_n and (m != argv.code_n and n != argv.code_n):
                break
          for H in [H0, H0.t]:
            _, n = H.shape
            if argv.code_n and n != argv.code_n:
                continue

            HHt = H*H.t # overlap of the rows
            if HHt.sum() != 0:
                continue

            H = H.row_reduce()
            m, n = H.shape
            assert 2*m <= n
            if 2*m == n:
                continue
            #print(HHt.shape)
            #print(H.shortstr(), H.shape)
            css = CSSCode(Hx=H, Hz=H, check=False, build=False)
            if css.n-css.k < 100:
                css.bz_distance()
            if css.d and css.d <= 2:
                continue

            if (css.n, css.k, css.d) == (24, 14, 4):
                print(H, css)

            if len(H) > 20:
                key = (css.n, css.k, css.d)
                wenum = None

            else:
                wenum = H.get_wenum()
                key = (css.d, wenum)

            if key not in found:
                print(css, end=' ', flush=True)
                found.add(key)
                if argv.weight:
                    weight = get_weight(H)
                    print(weight)
                else:
                    print()
                if argv.dump:
                    print(H, H.shape)
                if (css.n, css.k, css.d) == argv.dumpcode:
                    print(H, H.shape)
                if argv.find_autos:
                    find_autos(H)



def conjugacy_subgroups(G):
    gap = Gap()

    N = G.rank
    n = gap.Order(G, get=True)
    assert len(G) == n
    _Hs = gap.ConjugacyClassesSubgroups(G)
    #print("\t", gap.Length(Hs, get=True))
    Hs = []
    for i in range(len(_Hs)):
        item = _Hs[i]
        item = gap.Representative(item)
        assert gap.IsPermGroup(item, get=True)
        H = gap.to_group(item, N=N)
        #print("\t", H)
        assert G.is_subgroup(H)
        Hs.append(H)

    return Hs



def get_css(G):
    gap = Gap()

    print("\t", G, G.structure_description(), end=", ")

    #print(G.perms)
    N = G.rank
    n = gap.Order(G, get=True)
    assert len(G) == n
    _Hs = gap.ConjugacyClassesSubgroups(G)
    #print("\t", gap.Length(Hs, get=True))
    Hs = []
    for i in range(len(_Hs)):
        item = _Hs[i]
        item = gap.Representative(item)
        assert gap.IsPermGroup(item, get=True)
        H = gap.to_group(item, N=N)
        #print("\t", H)
        assert G.is_subgroup(H)
        index = len(G) // len(H)
        if argv.index and index != argv.index:
            continue
        if len(G) > len(H) > 1: # right ?!?
            Hs.append(H)

    del N
    print("subgroups:", len(Hs), end=', ', flush=True)

    print("build action... ", end='', flush=True)

    N = len(Hs)
    pairs = get_pairs(G, Hs)
    print("done")

    for i in range(N):
     for jx in range(N):
       opxs = pairs[jx, i]
       if argv.code_n and opxs and opxs[0].shape[1] != argv.code_n:
           break 
       for jz in range(N):
        opzs = pairs[jz, i]

        for Hx in (opxs):

         for Hz in (opzs):

            HHt = Hx*Hz.t
            if HHt.sum() != 0:
                continue

            #Hx = Hx.linear_independent()
            #Hz = Hz.linear_independent()
            mx, n = Hx.shape
            mz, n = Hz.shape
            assert mx+mz <= n
            if mx+mz == n:
                continue
            css = CSSCode(Hx=Hx, Hz=Hz, check=False, build=False)
            if css.n-css.k < 100:
                css.bz_distance()
            if css.d and css.d <= 2:
                continue

            wx, wz = (Hx.get_wenum(), Hz.get_wenum())
            if (wx,wz) in found or (wz,wx) in found:
                continue

            found.add((wx,wz))

            w = get_css_weight(css)
            print(css, w[0], w[1], "sd" if css.is_selfdual() else "")
            if argv.dump or (css.n, css.k, css.d) == argv.dumpcode:
                print("Hx =")
                print(css.Hx)
                print("Hz =")
                print(css.Hz)
                print()


def rand_ops(ops, weight):
    n = len(ops)
    assert n >= weight
    idxs = list(range(n))
    shuffle(idxs)
    ops = [ops[i] for i in idxs[:weight]]
    return reduce(add, ops)


def search_bimodule(G):
    print("\t", G, G.structure_description(), end=", ")

    Hs = conjugacy_subgroups(G)
    N = len(Hs)
    print("conjugacy_subgroups:", N)
    Xs = [G.left_action(Hs[i]) for i in range(N)]

    #Hs = [H for H in Hs if len(H)==6]
    Hs = [H for H in Hs if 1<len(H) <len(G)]
    #found = set() # use global

    trials = argv.get("trials", 10)

    for H in Hs:
      for K in Hs:
        dcosets = set()
        for g in G:
            dc = list({h * g * k for h in H for k in K})
            dc.sort()
            dc = tuple(dc)
            dcosets.add(dc)
        dcosets = list(dcosets)
        dcosets.sort(key = len)
        for dc in dcosets:
            if len(dc) == len(G):
                continue # these are 2BGA's ?
            if argv.code_n and 2*len(dc) != argv.code_n:
                continue
            #print("(%d-%d)-bimodule dim=%d, "%(len(H), len(K), len(dc)), end='', flush=True)
            #print("*", end='', flush=True)
            lookup = {g:i for i,g in enumerate(dc)}
            #for i,g in enumerate(dc):
            #    print("\t", i, g)
            L = {}
            R = {}
            for h in H:
                perm = ([lookup[h*g] for g in dc])
                L[h] = Matrix.get_perm(perm)
                #print(L[h])
            for k in K:
                perm = ([lookup[g*k] for g in dc])
                R[k] = Matrix.get_perm(perm)
                #print(R[k])
            for h in H:
              for k in K:
                assert L[h]*R[k] == R[k]*L[h]
            n = R[k].shape[0]
            lhs = list(L.values())
            rhs = list(R.values())
            #for lw in [1,2,3,4]:
            for lw in [2,3,4]:
              if len(lhs) < lw:
                continue
              #lops = [reduce(add, ops) for ops in choose(lhs, lw)]
              lops = [rand_ops(lhs, lw) for _ in range(trials)]
              #for rw in range(lw, 4+1):
              for rw in [8-lw]:
               if len(rhs) < rw:
                 continue
               #rops = [reduce(add, ops) for ops in choose(rhs, rw)]
               rops = [rand_ops(rhs, rw) for _ in range(trials)]
               for A in lops:
                for B in rops:
                    #assert A*B == B*A
                    Hx = A.concatenate(B, axis=1)
                    Hz = B.concatenate(A, axis=0).t
                    #assert (Hx*Hz.t).sum() == 0
                    Hx = Hx.row_reduce()
                    Hz = Hz.row_reduce()
                    #assert (Hx*Hz.t).sum() == 0
                    css = CSSCode(Hx=Hx, Hz=Hz, check=False, build=False)
                    if css.k==0:
                        continue
                    css.bz_distance()
                    key = str(css)
                    if key in found:
                        continue
                    found.add(key)
                    if css.d==2:
                        #print('.', flush=True, end='')
                        pass
                    else:
                        print(css, lw+rw)

            

def search(G):

    print("search", G, G.structure_description())
    #GG = G*G
    #GG.get_gens() # randomly choose gens

    #G.get_gens() # ?
    e = G.identity
    GG = Group(
        gens=[g.cross(e) for g in G.gens]+[e.cross(g) for g in G.gens])
    assert len(GG) == len(G)**2

    N = len(GG)
    for i in range(1, N):
        if N%i==0:
            print(i, end=' ')
    print()

    assert argv.code_n
    if len(G)**2 % argv.code_n:
        return 

    trials = argv.get("trials", 1000)
    w = argv.get("w", 8)

    #for g in G:
    #  for h in G:
    #    assert g.cross(h) in GG

    print(GG, end=' ')

    Ls = conjugacy_subgroups(GG)
    N = len(Ls)
    print("conjugacy_subgroups:", N)

    n = argv.code_n // 2

    for L in Ls:

        if len(GG) // len(L) != n:
            continue

        print(L, L.structure_description())
        assert GG.is_subgroup(L)

        X = GG.left_cosets(L)
        lookup = {perm:idx for (idx,perm) in enumerate(X)}
        assert len(X) == n

        left = {}
        right = {}
        for g in G:
            ge = g.cross(e)
            #assert ge in GG
            #for x in X:
            #    gex = ge*x
            #    assert isinstance(gex, Coset)
            #    assert gex in lookup
            perm = Perm([lookup[ge*x] for x in X])
            left[g] = perm
            eg = e.cross(g)
            perm = Perm([lookup[(~eg)*x] for x in X])
            right[g] = perm
            #print("\t\t", perm)
        #print()

        for g in G:
          for h in G:
            L, R = left[g], right[h]
            assert L*R == R*L

        Ls = set(left.values())
        Rs = set(right.values())

        if len(Ls) < 4:
            continue

        Ls = [Matrix.get_perm(L) for L in Ls]
        Rs = [Matrix.get_perm(R) for R in Rs]
        for L in Ls:
          for R in Rs:
            assert L*R == R*L

        smap = SMap()
        col = 0
        for L in Ls:
            smap[0,col] = str(L)
            col += L.shape[1]+1
        #print(smap)
        #print()

        for _ in range(trials):
            A = Matrix.zeros((n,n))
            B = Matrix.zeros((n,n))
            for i in range(w):
                A += choice(Ls)
                B += choice(Rs)

            if A.sum() == 0:
                continue
            if B.sum() == 0:
                continue

            #assert A*B == B*A
            Hx = A.concatenate(B, axis=1)
            Hz = B.concatenate(A, axis=0).t
            #assert (Hx*Hz.t).sum() == 0
            Hx = Hx.row_reduce()
            Hz = Hz.row_reduce()
            #assert (Hx*Hz.t).sum() == 0
            css = CSSCode(Hx=Hx, Hz=Hz, check=False, build=False)
            if css.k==0:
                continue
            css.bz_distance()
            key = str(css)
            if key in found:
                continue
            found.add(key)
            if css.d<=2:
                print(css)
            else:
                print("\t", css)

            
            






def main():

    gap = Gap()

    func = get_selfdual
    if argv.get_css:
        func = get_css
    elif argv.get_selfdual:
        func = get_selfdual
    elif argv.search_bimodule:
        func = search_bimodule
    elif argv.search:
        func = search
    else:
        print("no func")
        return

    print(func)
    assert not argv.max_ops, "deprecated"

    if argv.PSL:
        m = argv.get("m", 3)
        n = argv.get("n", 2)
        N = argv.get("N", None)
        _G = gap.PSL(m, n)
        G = gap.to_group(_G, smaller=False, N=N)
        func(G)
        return

    if argv.Alternating:
        n = argv.get("n", 5)
        _G = gap.AlternatingGroup(n)
        G = gap.to_group(_G, smaller=False)
        func(G)
        return

    if argv.Symmetric:
        n = argv.get("n", 5)
        _G = gap.SymmetricGroup(n)
        G = gap.to_group(_G, smaller=False)
        func(G)
        return

    if argv.CoxeterBC:
        n = argv.get("n", 3)
        G = Group.coxeter_bc(n)
        func(G)
        return

    if argv.CoxeterD:
        n = argv.get("n", 4)
        G = Group.coxeter_d(n)
        func(G)
        return

    if argv.Mathieu:
        n = argv.get("n", 11)
        _G = gap.MathieuGroup(n)
        G = gap.to_group(_G, smaller=False)
        func(G)
        return


    desc = argv.desc
    n = argv.get("n", 6)
    n0 = argv.get("n0", n)
    n1 = argv.get("n1", n0+1)
    print("allgroups", n0, n1)
    for _G in allgroups(n0, n1):
        G = gap.to_group(_G, smaller=False)
        s = G.structure_description()
        print(s)
        if desc and s != desc:
            continue
        func(G)


def test_autos():

    H = Matrix.parse("""
    ..11......111.1.1.1.
    .11.....1..11..1..11
    1..111...1.1....1.1.
    11..1.1.1.1..11.....
    ......11..11.11..11.
    ....11111..11.1.1..1
    """)

    _H = Matrix.parse("""
    11..1..11.1.11..1.1.1..1
    ..11.11.1.1...111.1..11.
    ..111..1.1.1..11.1.11..1
    1..11.1.11...11.11..1.1.
    1.1.11..1..11.1..11...11 
    """)
    
    H = Matrix.parse("""
    1.............1.1111..1..11...1.1.1.
    .1............111.111...1...1..11.1.
    ..1.........1...11..11.11..1..1.1..1
    ...1........11...11.11....1..11..1.1
    ....1........111.....111.1...1.1.11.
    .....1......11.1...1..11...11..1.1.1
    ......1......11...1.1.1...1.1111..1.
    .......1....1...1..11.1...111.111...
    ........1...1..1..1.1..11...11..11.1
    .........1....1..11..1.111...11.11..
    ..........1..1...1.1.11..111.....111
    ...........1...11..1.1.111.1...1..11
    ..................111111......111111
    """)

    find_autos(H)

    from qumba.gcolor import dump_transverse
    css = CSSCode(Hx=H, Hz=H)
    dump_transverse(css.Hx, css.Lx)
    


def test_12():
    Hx = Matrix.parse("""
    1.1...1.111.
    .11....11.11
    ...11.11..11
    ....111.11.1
    """)
    Hz = Matrix.parse("""
    1.1...11..11
    .11...1.11.1
    ...1.111.1.1
    ....11.11.11
    """)
    css = CSSCode(Hx=Hx, Hz=Hz)
    n = css.n
    N, perms = css.get_autos()
    
    perms = [Perm(idxs) for idxs in perms]
    G = Group.generate(perms)
    assert len(G) == 72
    #assert G.structure_description() == "(C3 x A4) : C2"

    for i in range(n):
        # point stabilizer is S3
        H = [g for g in G if g[i]==i]
        assert len(H) == 6
        orbits = set()
        for j in range(12):
            o = list(set(h[j] for h in H))
            o.sort()
            o = tuple(o)
            orbits.add(o)
        orbits = list(orbits)
        orbits.sort(key=len)
        #print(orbits)
        

    quads = [
        (0, 3, 8, 11),
        (1, 5, 6, 10),
        (2, 4, 7, 9),
    ] # ?

    for j in range(1, 12):
        H = [g for g in G if g[0]==0 and g[j]==j]
        #print(len(H))

    count = 0
    for pairs in get_complete_pairings(list(range(n))):
        #print(pairs)
        count += 1
    assert count == 10395

    # try concatenating with 422... gives [[24,4,4]] at best.

    code = construct.get_422()
    e4 = code.get_encoder()
    E = css.to_qcode().get_encoder()
    rhs = SymplecticSpace(n).get_identity() << E
    lhs = reduce(lshift, [e4]*6)

    for pairs in get_complete_pairings(list(range(n))):
        idxs = [i + 4*j for i in range(2) for j in range(6)] # stabs
        #print("\t", pairs)
        pairs = reduce(add, pairs)
        #pairs = list(range(12))
        #pairs = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #print(pairs)
        #src = [2 + i + 4*j for i in range(2) for j in range(6)] # logicals
        src = [2 + i + 4*j for j in range(6) for i in range(2)] # logicals
        jdxs = [src[i] for i in pairs]
        jdxs = [None] * n
        for i,j in enumerate(pairs):
            jdxs[j] = src[i]
        #print(jdxs == [src[i] for i in pairs])
        #print("\t", jdxs)
        idxs += jdxs
        P = SymplecticSpace(2*n).get_perm(idxs)
        #print(P)
        
        E = lhs * P.t * rhs
        assert SymplecticSpace(2*n).is_symplectic(E)
    
        code = QCode.from_encoder(E, k=4)
        assert code.is_css()
        #code.distance("z3")
        #print(code, "sd" if code.is_selfdual() else "css")
        #print(code.longstr())
        css = code.to_css()
        css.bz_distance()

        #print(css)
        if css.d > 4:
            break
        print(".", flush=True, end='')

    print(css)


def test_bring():
    css = construct.get_bring()
    print(css)

    perms = css.find_autos()
    perms = [Perm(idxs) for idxs in perms]
    G = Group(perms)
    print(G.structure_description())

    get_css(G)


            
        


if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "main"
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





