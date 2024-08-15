#!/usr/bin/env python

"""
Build some cyclic codes, we get all gf4 linear cyclic codes easily,
these include the self-dual codes.

More general algorithm is here:
https://arxiv.org/abs/1007.1697

"""


from random import shuffle
from functools import reduce
from operator import add, mul

import numpy

from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
    PolynomialRing, GF, factor)
from sage import all_cmdline

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, Matrix, get_weight, fromstr
from qumba import construct
from qumba.argv import argv
from qumba.distance import distance_z3
from qumba.transversal import find_local_cliffords
from qumba.action import mulclose
from qumba.autos import is_iso


def all_cyclic_gf4(n, dmin=1, gf4_linear=True):
    F = GF(4)
    z2 = F.gen()
    R = PolynomialRing(F, "x")
    x = R.gen()

    A = factor(x**n - 1)
    #print(A)
    space = SymplecticSpace(n)

    mkpauli = lambda a:''.join({0:'.', 1:'X', z2:'Z', z2+1:'Y'}[a[i]] for i in range(n))

    factors = [a for (a,j) in A]
    N = len(factors)
    #print("factors:", N)

    scalars = [1, z2, z2+1] if gf4_linear else [1]

    # build all the principle ideals
    for bits in numpy.ndindex((2,)*N):
        a = reduce(mul, (factors[i] for i in range(N) if bits[i]), x**0)
        #if sum(bits)==N:
        #    assert n%2==0 or a == x**n+1, a
        #    continue 
        #print(gen)
    
        rows = []
        for i in range(n):
          for scalar in scalars:
            gen = mkpauli(scalar*a)
            s = ''.join(gen[(k+i)%n] for k in range(n))
            rows.append(s)
        s = ' '.join(rows)
        H = space.fromstr(s)
        H = H.linear_independent()
        U = H * space.F * H.t
        if U.sum():
            continue # not isotropic
        #print(H)
        code = QCode(H)
        #print(code.longstr())
        #print(code, end=' ', flush=True)
        d = distance_z3(code)
        code.d = d
        if d >= dmin:
            yield code
        #print(gen, code)
        #print(code.longstr())


def all_cyclic_gf2_poly(n):
    assert n%2, n
    F = GF(2)
    z2 = F.gen()
    R = PolynomialRing(F, "x")
    x = R.gen()

    A = factor(x**n - 1)
    #print(A)
    space = SymplecticSpace(n)

    factors = [a for (a,j) in A]
    N = len(factors)
    #print("factors:", N)

    Hs = []
    for bits in numpy.ndindex((2,)*N):
        a = reduce(mul, (factors[i] for i in range(N) if bits[i]), x**0)
        yield a


def all_cyclic_gf2(n):
    assert n%2, n
    F = GF(2)
    z2 = F.gen()
    R = PolynomialRing(F, "x")
    x = R.gen()

    A = factor(x**n - 1)
    #print(A)
    space = SymplecticSpace(n)

    factors = [a for (a,j) in A]
    N = len(factors)
    #print("factors:", N)

    Hs = []
    for bits in numpy.ndindex((2,)*N):
        a = reduce(mul, (factors[i] for i in range(N) if bits[i]), x**0)
        if sum(bits)==N:
            assert a == x**n+1, a
            continue 
        gen = numpy.array([int(a[k]) for k in range(n)], dtype=int)
        yield gen


def all_cyclic_sp(n, dmin):

    space = SymplecticSpace(n)
    nn = 2*n
    for hx0 in all_cyclic_gf2(n):
      for hz in all_cyclic_gf2(n):
        for i in range(n):
            idxs = [(i+k)%n for k in range(n)]
            hx = hx0[idxs]
            h = numpy.zeros((n,2), dtype=int)
            h[:,0] = hx
            h[:,1] = hz
            #print(hx, hz)
            #print(h)
            rows = []
            for j in range(n):
                idxs = [(j+k)%n for k in range(n)]
                h1 = h[idxs, :].copy()
                h1.shape = nn,
                rows.append(h1)
            #H = numpy.array(rows)
            H = Matrix(rows)
            if (H*space.F*H.t).sum()==0:
                H = H.linear_independent()
                code = QCode(H)
                yield code
                #print(H)
                #print()

        #print()


def main():
    n = 5
    for h in all_cyclic_gf2(n):
        print(h)
    return

    for code in all_cyclic_sp(n, 3):
        print(code)
        print(code.longstr())
        pass


def main_gf4():
    gf4_linear = argv.get("gf4_linear", True)
    n = argv.get("n", 20)
    for n0 in range(2, n):
        if argv.even:
            n = 2*n0
        else:
            n = 2*n0 + 1
        for code in all_cyclic_gf4(n, 3, gf4_linear):
            sd = code.is_selfdual()
            H = code.H
            rws = [get_weight(h) for h in H.A]
            if gf4_linear:
                assert code.is_gf4_linear()
            tgt = code.apply_perm([(i+1)%n for i in range(n)])
            assert tgt.is_equiv(code)
            if code.k==0:
                continue
            print(code, set(rws), 
                "*" if sd else "", 
                "gf4" if code.is_gf4_linear() else "")
            print(code.longstr())
            print()
            continue
            L = tgt.get_logical(code)
            assert (L**n).is_identity()
            if not L.is_identity():
                print("L")
            #N, perms = code.get_autos()
            gen = list(find_local_cliffords(code))
            if len(gen)==1:
                continue
            gen = [code.get_logical(g*code) for g in gen] + [L]
            G = mulclose(gen)
            print("|G| =", len(G))
            print()


def test_713():
    code = QCode.fromstr("""
    X.XXX..
    Z.ZZZ..
    .XXX..X
    .ZZZ..Z
    XXX..X.
    ZZZ..Z.
    """)

    code = QCode.fromstr("""
XXX.X..
ZZZ.Z..
ZZ.Z..Z
YY.Y..Y
X.X..XX
Z.Z..ZZ
    """)

    tgt = construct.get_713()

    print(is_iso(code, tgt))



def test_513():
    code = construct.get_513()
    assert code.is_gf4_linear()


def test_golay():
    n = 24
    #code = construct.get_golay()
    #n = code.n
    #print(code)

    n = 23
    for code in all_cyclic(n):
        print(code)

        print(code.longstr())
        assert code.is_gf4_linear()
        assert code.is_selfdual()
    
        tgt = code.apply_perm([(i+1)%n for i in range(n)])
        assert tgt.is_equiv(code)


def test_13_1_5():
    """
    Disambiguate some [[13,1,5]] cyclic codes..
    refs:
    https://errorcorrectionzoo.org/c/stab_13_1_5
    https://arxiv.org/abs/quant-ph/9704019 page 10-11
    """
    code = construct.get_xzzx(2,3)
    #assert distance_z3(code) == 5
    print(code)
    print(code.longstr())
    print()
    H = code.H
    n = code.n
    space = SymplecticSpace(n)
    u = space.fromstr("X.ZZ.X.......")
    u = space.fromstr("XYZYYZYX.....")
    gen = "XZ.ZX........"
    for i in range(n):
        s = ''.join(gen[(k+i)%n] for k in range(n))
        u = space.fromstr(s)
        #print(H * space.F * u.t)

    # how to find the perms:
    #N, perms = code.get_autos()
    from qumba.action import Perm, Group
    items = list(range(n))
    gen = [
        Perm([0, 8, 7, 11, 5, 4, 9, 2, 1, 6, 12, 3, 10], items),
        Perm([1, 0, 8, 12, 11, 5, 10, 9, 2, 7, 6, 4, 3], items)]
    G = Group.generate(gen)
    assert len(G) == 26

    for g in G:
        if g.order() == n:
            break

    del code, H

    gen = ("XY.Z.YX......") # [[13,1,3]]
    gen = ("XZ.Z.ZX......") # [[13,1,3]]
    gen = ("X.ZZ.X.......")
    gen = ("ZXIIIXZ......")
    gen = ("XYZYYZYX.....")
    rows = []
    for i in range(n-1):
        s = ''.join(gen[(k+i)%n] for k in range(n))
        rows.append(s)
    s = ' '.join(rows)
    H = space.fromstr(s)
    #print(H)
    code = QCode(H)
    print(code)
    assert distance_z3(code) == 5, distance_z3(code)

    #u = space.fromstr("X.ZZ.X.......")
    #print(H * space.F * u.t)

    #from qumba.transversal import find_local_cliffords
    #for E in find_local_cliffords(code, code):
    #    print(E)

    print(code.longstr())

    return




def all_cyclic(n):

    K = GF(2)
    R = PolynomialRing(K, "x")
    x = R.gen()

    A = factor(x**n - 1)

    def conv(f, h):
        v = zeros2(n)
        for i in range(n):
            for j in range(n):
                v[i] += f[(i+j)%n]*h[(-j)%n]
        return v % 2

#    def sy(f, g, h, l):
#        fb = list(reversed(f))
#        lb = list(reversed(l))
#        #return numpy.alltrue(conv(fb, h) == conv(g, lb))
#        hb = list(reversed(h))
#        return numpy.alltrue(conv(fb, l) == conv(g, hb))

    def sy(f, g):
        fb = list(reversed(f))
        gb = list(reversed(g))
        return numpy.alltrue( conv(fb, g) == conv(f, gb) )

    if n==5:
        #print(conv([0,1,0,0,0], [0,1,0,0,0])) # x*x --> [0 0 1 0 0 0]
        #print(conv([1,0,0,0,0], [0,1,0,0,0])) # x*x --> [0 0 1 0 0 0]
        f = [1,1,0,0,0]
        g = [0,0,0,1,1]
        #assert sy(f,g,f,g)
    
        #fb = list(reversed(f))
        #gb = list(reversed(g))
        #print( conv(fb, g), conv( f, gb ) )
        #print(numpy.alltrue( conv(fb, g) == conv( f, gb ) ) )
    
        assert not sy(f, g)
        #return
    
        #print( conv([1,1,1,1,1], [1,1,0,0,0]) )

    z = zeros2(n)

    F = list(all_cyclic_gf2(n)) + [z]
    #print(F)

    pairs = []
    for f in F:
      for h in F:
        #if h.sum() == 0:
        #    continue # classical codes...
        fb = list(reversed(f))
        if conv(fb, h).sum() == 0:
            pairs.append((f, h))
    print("pairs:", len(pairs))

    # <(f,g), (f,g)> = 0

    def mkpauli(f, g):
        op = []
        for i in range(n):
            if f[i] and g[i]:
                op.append("Y")
            elif f[i]:
                op.append("X")
            elif g[i]:
                op.append("Z")
            else:
                op.append('.')
        return ''.join(op)

    g = z
    for (f,h) in pairs:

      for g in numpy.ndindex((2,)*n):
        if not sy(f, g):
            continue

        #print( shortstr(f), shortstr(g), "--", shortstr(z), shortstr(h) )
        #print(f, g, h)
        l, r = mkpauli(f, g), mkpauli(z, h) 
        #if l.count(".")==n:
        #    continue
        a,b,c = (l+r).count("X"), (l+r).count("Y"), (l+r).count("Z")
        if [a,b,c].count(0)>1:
            continue # classical codes
        #print()
        #print( shortstr(f), shortstr(g), "--", shortstr(z), shortstr(h) )
        #print(l,r)
        stabs = []
        for i in range(n):
            stabs.append( ''.join(l[(j-i)%n] for j in range(n)) )
            stabs.append( ''.join(r[(j-i)%n] for j in range(n)) )
        #print(stabs)
        stabs = ' '.join(stabs)
        H = fromstr(stabs)
        #print(shortstr(H))
        #print()
        H = linear_independent(H)
        if len(H) == n:
            continue # stabilizer states
        code = QCode(H)
        code.cyclic_gens = (f, g, h)
        yield code
    

def test_cyclic():
    n = argv.get("n", 7)
    print("all_cyclic", n)
    count = 0
    for code in all_cyclic(n):
        if code.k:
            code.d = code.distance("z3")
            print(code, "+" if sum(code.cyclic_gens[1])==0 else " ", end=" ", flush=True)
            #print(code, "+" if code.is_css() else " ", 
            #    'l' if code.is_gf4_linear() else " ")
            count += 1
            if count % 8 == 0:
                print()
    print()




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
        main()


    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")


