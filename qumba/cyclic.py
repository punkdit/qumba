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

from qumba.lin import (parse, shortstr, linear_independent,
    eq2, dot2, identity2, solve,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span)
from qumba.qcode import QCode, SymplecticSpace, Matrix, get_weight, fromstr, strop
from qumba import construct
from qumba.argv import argv
from qumba.distance import distance_z3, distance_z3_lb, distance_meetup
from qumba.transversal import find_local_cliffords
from qumba.transversal import find_autos, find_autos_lc
from qumba.autos import get_autos, get_isos
from qumba.action import mulclose, Perm, Group
from qumba.autos import is_iso
from qumba.smap import SMap
from qumba.util import all_primes


def get_cyclic_perms(n):
    """compute GL(1,Z/n) this is the Galois group
    of the n-th cyclotomic integers"""
    space = SymplecticSpace(n)
    gl = set()
    for i in range(1, n):
        for j in range(1, n):
            if (i*j)%n == 1:
                gl.add(i)
                gl.add(j)
    gl = list(gl)
    gl.sort()
    perms = []
    for i in gl:
        #print(i, i%3, i%5)
        perm = [(j*i)%n for j in range(n)]
        p = space.get_perm(perm)
        perms.append(p)
    return perms


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
    print("factors:", N)

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
        code = QCode(H, cyclic=True)
        #print(code.longstr())
        #print(code, end=' ', flush=True)
        #d = distance_z3(code, verbose=True)
        if distance_z3_lb(code, dmin-1):
            yield code
        #print(gen, code)
        #print(code.longstr())


def main_multi():
    
    p, l = 5, 25

    F = GF(2)
    z2 = F.gen()
    R = PolynomialRing(F, 2, "x")
    x0, x1 = R.gens()

    poly = (x0**p - 1) * (x1**l - 1)
    A = factor(poly)

    factors = [a for (a,j) in A]
    print(factors)
    N = len(factors)
    print("factors:", N)


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
    #assert n%2, n
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
        if sum(bits)==N and n%2:
            assert a == x**n+1, a
            continue 
        gen = numpy.array([int(a[k]) for k in range(n)], dtype=int)
        yield gen


def is_cyclic_gf2(H):
    m, n = H.shape
    idxs = [(i+1)%n for i in range(n)]
    J = H[:, idxs]
    return solve(H.transpose(), J.transpose()) is not None

def is_iso_gf2(H, J):
    if H.shape != J.shape:
        return False
    A = solve(H.transpose(), J.transpose())
    Ai = solve(J.transpose(), H.transpose())
    assert (A is None) == (Ai is None)
    return A is not None


def count_cyclic_gf2():
    # TODO: submit to OEIS
    # 1,2,3,2,3,4,7,2,7,4,3,4,3,8,31,2,7,8,3,4,63,4,7,4,7,4,15,8,3,
    for n in range(1, 30):
        vs = list(all_cyclic_gf2(n))    
        N = len(vs)
        print("%d,"%N, end="", flush=True)
    print()

        
def main_cyclic_gf2():
    n = argv.get("n", 17)

    gens = get_cyclic_perms(n)
    gens = [g[::2,::2] for g in gens]
    print("gens:", len(gens))

    rows = []
    for i in range(n):
        row = [0]*n
        row[(i+1)%n] = 1
        rows.append(row)
    C = Matrix(rows)
    #print(C)
    #for g in gens:
    #    print(C*g == g*C)
    #return

    vs = list(all_cyclic_gf2(n))    
    N = len(vs)
    Hs = [
        numpy.array([numpy.array([v[(i+j)%n] for i in range(n)]) 
        for j in range(n)]) for v in vs ]
    Hs = [linear_independent(H) for H in Hs]
    print("Hs:", N)


    for H0 in Hs:
        h = H0[0]
        print(shortstr(h), H0.shape, h.sum())

    found = set()
    for g in gens:
        rows = []
        for H in Hs:
            J = dot2(H, g)
            assert is_cyclic_gf2(J)
            row = [int(is_iso_gf2(J, H1)) for H1 in Hs]
            rows.append(row)
        P = Matrix(rows)
        if P not in found:
            print(P.is_identity(), (P*P).is_identity(), (P*P.t).is_identity())
            print()
        found.add(P)
    #found = list(found)
    print("found:", len(found))

    #G = mulclose(found)
    #assert len(G) == len(found)
    G = found

    for g in G:
      for h in G:
        assert g*h in G
        assert g*h == h*g, "non-abelian ???!"
        


def all_cyclic_css(n):
    vs = list(all_cyclic_gf2(n))    
    N = len(vs)
    print()
    print("all_cyclic_css(n=%d)   n mod 4 = %d,  n mod 8 = %d"%(n, n%4, n%8))

    Hs = [
        numpy.array([numpy.array([v[(i+j)%n] for i in range(n)]) for j in range(n)])
        for v in vs
    ]
    Hs = [linear_independent(H) for H in Hs]
    print("Hs:", N)

    for i in range(N):
      for j in range(i, N):
        if dot2(Hs[i], Hs[j].transpose()).sum():
            continue
        code = QCode.build_css(Hs[i], Hs[j], cyclic=True)
        if code.k == 0 or code.d_upper_bound <= 2:
            continue
        yield code


def main_sd():
    n = argv.get("n")
    if n is None:
        ns = list(range(64, 128))
    else:
        ns = [n]

    for n in ns:
      for v in all_cyclic_gf2(n):
        H = numpy.array([numpy.array([v[(i+j)%n] for i in range(n)]) for j in range(n)])
        if dot2(H, H.transpose()).sum():
            #print("/", end="", flush=True)
            continue
        H = linear_independent(H)
        code = QCode.build_css(H, H, cyclic=True)
        if code.k == 0 or code.d_upper_bound <= 2:
            continue
        print(code)
        if argv.show:
            print(code.longstr())




def unique(codes, min_d=3):
    from qumba.transversal import find_isomorphisms_css, find_isomorphisms
    def isomorphic_css(code, dode):
        for iso in find_isomorphisms_css(code, dode):
            return True
        return False

    def isomorphic(code, dode):
        for iso in find_isomorphisms(code, dode):
            return True
        return False

    found = {} # map each code to a unique representative

    for code in codes:
        if code.d is not None and code.d < min_d:
            print("d.", end="", flush=True)
            continue
        #print("\n", code.d, min_d)
        for prev in list(found.keys()):
            if code.is_equiv(prev):
                found[code] = found[prev]
                print("e.", end="", flush=True)
                break
            if code.n < 30 and code.is_css() and prev.is_css() and isomorphic_css(code, prev):
                found[code] = found[prev]
                print("i.", end="", flush=True)
                break
            if code.n < 30 and isomorphic(code, prev):
                found[code] = found[prev]
                print("i.", end="", flush=True)
                break
            #if isomorphic(code.get_dual(), prev): # up to n=15 didn't find any... hmm..
            #    found[code] = found[prev]
            #    print("d")
            #    break
            #print("/", end="")
            if code.n < 10 and code.k == prev.k and code.get_isomorphism(prev): # slooooow
                found[code] = found[prev]
                print("i.", end="", flush=True)
                break
        else:
            found[code] = code # i am the representative 
            yield code
    print("\nfound:", len(set(found.values())), "of", len(set(found.keys())))


def main_all_css():
    min_d = argv.get("min_d", 3)
    #n = argv.get("n", 19)
    #for n in all_primes(100):
    for idx in range(2, argv.get("idx", 11)):
        n = 2*idx + 1
        if n < 5:
            continue
        #if n%4 == 3:
        #codes = list(all_cyclic_css(n))
        for code in unique(all_cyclic_css(n), min_d):
            code = code.to_css()
            code.bz_distance()
            code = code.to_qcode()
            print(code.cssname, "*" if code.is_selfdual() else " ")
            if argv.store_db:
                from qumba import db
                db.add(code)
        print()


def main_css():
    min_d = argv.get("min_d", 3)
    n = argv.get("n", 15)
    count = 0
    for code in unique(all_cyclic_css(n), min_d):
        code = code.to_css()
        code.bz_distance()
        code = code.to_qcode()
        print(code.cssname, "*" if code.is_selfdual() else " ")
        if argv.store_db:
            from qumba import db
            db.add(code)
        count += 1
        if count%6 == 0:
            break
    print()

        
        


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
                code = QCode(H, cyclic=True)
                yield code
                #print(H)
                #print()

        #print()



def main_gf4():
    gf4_linear = argv.get("gf4_linear", True)
    n = argv.get("n", 20)
    d = argv.get("d", 3)
    found = set()
    for code in all_cyclic_gf4(n, d, gf4_linear):
        sd = code.is_selfdual()
        H = code.H
        rws = list(set([get_weight(h) for h in H.A]))
        rws.sort()
        rws = tuple(rws)
        if gf4_linear:
            assert code.is_gf4()
        tgt = code.apply_perm([(i+1)%n for i in range(n)])
        assert tgt.is_equiv(code)
        if code.k==0:
            print(code, "skipping")
            continue
        if sd and argv.distance:
            #print(strop(code.H))
            css = code.to_css()
            assert css.k == code.k
            d = css.bz_distance()
            #print(css)
            code.d = min(d)
        elif argv.distance:
            distance_z3(code, verbose=True)
        elif argv.distance_meetup:
            #print()
            #d = distance_meetup(code, max_m=3, verbose=False)
            #print("distance_meetup(3):", d)
            #d = distance_meetup(code, max_m=4, verbose=False)
            #print("distance_meetup(4):", d)
            code.d = distance_meetup(code, verbose=True)
        key = (code.n, code.k, code.d, rws, sd)
        if argv.filter and key in found:
            continue
        found.add(key)
        print(code, set(rws), 
            "*" if sd else "", 
            "gf4" if code.is_gf4() else "")
        #if code.k > 1:
        #    distance_z3_lb(code, 5, verbose=True)
        if argv.show:
            print(strop(code.H))
        #print(code.longstr())
        #print()
        if argv.search_qr:
            search_qr(code)
        if argv.store_db:
            from qumba import db
            db.add(code)
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
    assert code.is_gf4()


def test_golay():
    n = 24
    #code = construct.get_golay()
    #n = code.n
    #print(code)

    n = 23
    for code in all_cyclic(n):
        print(code)

        print(code.longstr())
        assert code.is_gf4()
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
#        #return numpy.all(conv(fb, h) == conv(g, lb))
#        hb = list(reversed(h))
#        return numpy.all(conv(fb, l) == conv(g, hb))

    def sy(f, g):
        fb = list(reversed(f))
        gb = list(reversed(g))
        return numpy.all( conv(fb, g) == conv(f, gb) )

    if n==5:
        #print(conv([0,1,0,0,0], [0,1,0,0,0])) # x*x --> [0 0 1 0 0 0]
        #print(conv([1,0,0,0,0], [0,1,0,0,0])) # x*x --> [0 0 1 0 0 0]
        f = [1,1,0,0,0]
        g = [0,0,0,1,1]
        #assert sy(f,g,f,g)
    
        #fb = list(reversed(f))
        #gb = list(reversed(g))
        #print( conv(fb, g), conv( f, gb ) )
        #print(numpy.all( conv(fb, g) == conv( f, gb ) ) )
    
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
        H = linear_independent(H) # bottleneck
        if len(H) == n:
            continue # stabilizer states
        code = QCode(H, cyclic=True)
        code.cyclic_gens = (f, g, h)
        yield code
    

def main():
    n = argv.get("n", 7)
    min_d = argv.get("min_d", 3)
    print("all_cyclic", n)
    params = argv.params
    count = 0
    for code in unique(all_cyclic(n), min_d):
        if code.k == 0:
            continue
        #code.d = code.distance("z3")
        #if code.d <= 2:
        #    continue
        count += 1
        if argv.show or (code.n, code.k, code.d) == params:
            print()
            print(code, 'gf4' if code.is_gf4() else "")
            print('"'+strop(code.H).replace("\n", " ")+'"')

        else:
            #print(code, "+" if sum(code.cyclic_gens[1])==0 else " ", end=" ", flush=True)
            print(str(code) + ( "css" if code.is_css() else 
                ('gf4' if code.is_gf4() else "   ")), end = "", flush=True)
            if count % 8 == 0:
                print()
        if argv.store_db:
            from qumba import db
            db.add(code)
    print()


def find_prime():


    for n in all_primes(50):
        if n != 29:
            continue

        print(n, n%4)
        space = SymplecticSpace(n)

        #for bits in numpy.ndindex((2,)*(n-6)):
        m = (n-6)//2
        for left in numpy.ndindex((2,)*m):
            bits = left + (0,) + tuple(reversed(left))
            if numpy.sum(bits) > 8:
                continue
            #print(bits)
            bits = list("XXZZ") + [".Z"[bit] for bit in bits] + list("ZZ")
            code = is_cyclic(space, bits)
            if code is None:
                continue
            if code.k == 0:
                continue
            code.distance("z3")
            gf4 = code.is_gf4()
            print("found", ''.join(bits), code, 'gf4' if gf4 else '')
            if gf4:
                break


def is_cyclic(space, word):
    n = space.n
    F = space.F
    w = ''.join(word)
    words = [w]
    v0 = space.parse(w)
    for j in range(1,n):
        w1 = ''.join(w[(i+j)%n] for i in range(n))
        words.append(w1)
        v1 = space.parse(w1)
        if (v1*F*v0.t).sum():
            return None
    H = space.parse(' '.join(words))
    H = H.linear_independent()
    code = QCode(H)
    return code


def get_code(spec=None):
    if spec is None:
        spec = argv.code
    if spec == (5,1,3):
        code = construct.get_513() # GF4 linear

    elif spec == (9,3,3):
        # [[9,3,3]]
        code = QCode.fromstr("""
        YZZYZ...Z
        ZYZZYZ...
        .ZYZZYZ..
        ..ZYZZYZ.
        ...ZYZZYZ
        Z...ZYZZY
        """)

    elif spec == (13,1,5): # GF4 linear
        code = QCode.fromstr("""
        XXZZ.Z...Z.ZZ
        ZXXZZ.Z...Z.Z
        ZZXXZZ.Z...Z.
        .ZZXXZZ.Z...Z
        Z.ZZXXZZ.Z...
        .Z.ZZXXZZ.Z..
        ..Z.ZZXXZZ.Z.
        ...Z.ZZXXZZ.Z
        Z...Z.ZZXXZZ.
        .Z...Z.ZZXXZZ
        Z.Z...Z.ZZXXZ
        ZZ.Z...Z.ZZXX
        """)

#    elif spec == (17,1,7): # GF4 linear
#        # [[17,1,7]]
#        code = QCode.fromstr("""
#
#        """)
    elif spec == (17,1,7,0): # GF4 linear
        # [[17,1,7]]
        code = QCode.fromstr("""
        XXZ.ZZZ.Z.Z.ZZZ.Z
        ZXXZ.ZZZ.Z.Z.ZZZ.
        .ZXXZ.ZZZ.Z.Z.ZZZ
        Z.ZXXZ.ZZZ.Z.Z.ZZ
        ZZ.ZXXZ.ZZZ.Z.Z.Z
        ZZZ.ZXXZ.ZZZ.Z.Z.
        .ZZZ.ZXXZ.ZZZ.Z.Z
        Z.ZZZ.ZXXZ.ZZZ.Z.
        .Z.ZZZ.ZXXZ.ZZZ.Z
        Z.Z.ZZZ.ZXXZ.ZZZ.
        .Z.Z.ZZZ.ZXXZ.ZZZ
        Z.Z.Z.ZZZ.ZXXZ.ZZ
        ZZ.Z.Z.ZZZ.ZXXZ.Z
        ZZZ.Z.Z.ZZZ.ZXXZ.
        .ZZZ.Z.Z.ZZZ.ZXXZ
        Z.ZZZ.Z.Z.ZZZ.ZXX
        """, d=7)

    elif spec == (17,1,7,1): # GF4 linear
        # [[17,1,7]]
        code = QCode.fromstr("""
        XXZZ..Z.....Z..ZZ
        ZXXZZ..Z.....Z..Z
        ZZXXZZ..Z.....Z..
        .ZZXXZZ..Z.....Z.
        ..ZZXXZZ..Z.....Z
        Z..ZZXXZZ..Z.....
        .Z..ZZXXZZ..Z....
        ..Z..ZZXXZZ..Z...
        ...Z..ZZXXZZ..Z..
        ....Z..ZZXXZZ..Z.
        .....Z..ZZXXZZ..Z
        Z.....Z..ZZXXZZ..
        .Z.....Z..ZZXXZZ.
        ..Z.....Z..ZZXXZZ
        Z..Z.....Z..ZZXXZ
        ZZ..Z.....Z..ZZXX
        """, d=7)

    elif spec == (29,1,11):
        # [[29,1,11]]
        code = QCode.fromstr("""
        ..ZZXXZZ...Y.Y...Y...Y...Y.Y.
        ...ZZXXZZ...Y.Y...Y...Y...Y.Y
        Y...ZZXXZZ...Y.Y...Y...Y...Y.
        .Y...ZZXXZZ...Y.Y...Y...Y...Y
        Y.Y...ZZXXZZ...Y.Y...Y...Y...
        .Y.Y...ZZXXZZ...Y.Y...Y...Y..
        ..Y.Y...ZZXXZZ...Y.Y...Y...Y.
        ...Y.Y...ZZXXZZ...Y.Y...Y...Y
        Y...Y.Y...ZZXXZZ...Y.Y...Y...
        .Y...Y.Y...ZZXXZZ...Y.Y...Y..
        ..Y...Y.Y...ZZXXZZ...Y.Y...Y.
        ...Y...Y.Y...ZZXXZZ...Y.Y...Y
        Y...Y...Y.Y...ZZXXZZ...Y.Y...
        .Y...Y...Y.Y...ZZXXZZ...Y.Y..
        ..Y...Y...Y.Y...ZZXXZZ...Y.Y.
        ...Y...Y...Y.Y...ZZXXZZ...Y.Y
        Y...Y...Y...Y.Y...ZZXXZZ...Y.
        .Y...Y...Y...Y.Y...ZZXXZZ...Y
        Y.Y...Y...Y...Y.Y...ZZXXZZ...
        .Y.Y...Y...Y...Y.Y...ZZXXZZ..
        ..Y.Y...Y...Y...Y.Y...ZZXXZZ.
        ...Y.Y...Y...Y...Y.Y...ZZXXZZ
        Z...Y.Y...Y...Y...Y.Y...ZZXXZ
        ZZ...Y.Y...Y...Y...Y.Y...ZZXX
        XZZ...Y.Y...Y...Y...Y.Y...ZZX
        XXZZ...Y.Y...Y...Y...Y.Y...ZZ
        ZXXZZ...Y.Y...Y...Y...Y.Y...Z
        ZZXXZZ...Y.Y...Y...Y...Y.Y...
        """, d=11)

    else:
        assert 0, "no code specified"

    return code



def find_gates():

    code = get_code()
    print(code)
    space = code.space
    n = code.n

    if argv.store_db:
        from qumba import db
        db.add(code)

    if 0:
        found = []
        #for P in find_autos(code):
        #    found.append(P)
        for U in find_autos_lc(code):
            print(U)
            found.append(U)
    
        print(len(found))
    
        logical = set()
        for g in found:
            l = code.get_logical(g*code)
            if l not in logical:
                print(l)
                logical.add(l)
        print(len(logical))
    
        return
    
        found = []
        for M in find_local_cliffords(code):
            #print(M)
            found.append(M)
        print("local cliffords:", len(found))

    search_qr(code)


def find_residues():
    for n in all_primes(200):
        if n < 5:
            continue

        code = build_qr_code(n)
        if code is None:
            assert (n%4) != 1
            continue

        print(n, n%4, n%8, code, end=" ")
        print("gf4" if code.is_gf4() else " ", flush=True)
        if n%8 in [1, 5]:
            search_qr(code)


def search_qr(code):

    n = code.n
    space = code.space

    #G = Group.dihedral(n)
    G = Group.cyclic(n)
    #print(len(G))

    for a in range(2, n): # look for generator
        perm = {0:0}
        for i in range(1,n):
            perm[i] = (i*a)%n
        assert len(perm) == n
        perm = Perm(perm, list(range(n)))
        #print(perm)
    
        H = Group.generate([perm])
        #print(len(H))
        if len(H) == n-1:
            break
    else:
        assert 0

    #for h in H:
        #print(int(h in G), end=" ")
    #print()

    #return

    #H, S = space.H, space.S

    get_perm = lambda perm : space.get_perm([perm[i] for i in range(n)])

    for g in G:
        sigma = get_perm(g)
        assert (sigma*code).is_equiv(code)

    found = list(G)
    logical = []
    pattern = []
    #for h in H:
    for k in range(0, n-1):
        h = perm ** k
        sigma = get_perm(h)
        dode = sigma*code
        equiv = dode.is_equiv(code)
        if equiv:
            L = dode.get_logical(code)
            pattern.append(".")
            #print(L)
            logical.append(L)
            #continue

        for op in "SH":
            M = reduce(mul, [getattr(space, op)(i) for i in range(n)])
            eode = M*dode
            if eode.is_equiv(code):
                pattern.append(op)
                L = eode.get_logical(code)
                #print(L)
                logical.append(L)
        #pattern.append("*")

        #if len(mulclose(logical)) == 6:
        #    break

    print(''.join(pattern))

    print()
    print("|G| =", len(mulclose(logical)))
    print()
    

def gen_group():
    n = argv.get("n", 5)

    gl = list(range(1, n))
    gens = []
    for a in gl:
        found = set( (a**i)%n for i in range(n) )
        if len(found) == n-1:
            gens.append(a)

    print("gens:", gens)

    rotate = [(i+1)%n for i in range(n)]
    print(rotate)

    perms = [rotate]

    for a in gens:
        cycle = []
        for i in range(n-1):
            cycle.append( (a**i)%n )

        perm = [0] + [None]*(n-1)
        for i in range(n-1):
            perm[cycle[i]] = cycle[(i+1)%(n-1)]
        print(perm)
        perms.append(perm)
        #break

    items = list(range(n))
    perms = [Perm(perm, items) for perm in perms]

    a, b = perms[:2]
    assert (a*b != b*a)

    G = Group.generate(perms)
    assert len(G) == n*(n-1)


def build_qr_code(n):
    space = SymplecticSpace(n)

    gl = list(range(1, n))

    residues = {(i**2)%n for i in gl}
    #print(residues)

    smap = SMap()
    op = []
    for i in range(n):
        smap[0,i] = str(i)[-1:]
        if i==0:
            op.append('.')
            #smap[1,i] = "."
            #smap[2,i] = "."
        elif i in residues:
            op.append('Z')
            smap[1,i] = "*"
            smap[2,i] = "."
        else:
            op.append('X')
            smap[1,i] = "."
            smap[2,i] = "*"

    #print(smap)
    H0 = ''.join(op)
    #print(H0)

    H = []
    for i in range(n):
        H1 = ''.join(H0[(i-j)%n] for j in range(n))
        H.append(H1)
    H = "\n".join(H)
    #print(H)

    F = space.F
    H = space.fromstr(H)

    if argv.gf4:
        op = reduce(mul, [space.S(i) for i in range(n)])
        op *= reduce(mul, [space.H(i) for i in range(n)])
        H1 = H*op
        H = H.concatenate(H1)

    #if 1: # check if we get GF(4) linear code
    #    H = H.linear_independent()
    #    op = reduce(mul, [space.H(i)*space.S(i) for i in range(n)])
    #    H1 = H*op
    #    U = H1.t.solve(H.t)
    #    print(n%8, U is not None)

    A = H * F * H.t
    if A.sum() != 0:
        return None

    H = H.linear_independent()
    code = QCode(H, cyclic=True)

    if code.n < 20:
        code.distance("z3")

    return code

    
def find_qr_distance():
    from qumba.unwrap import unwrap
    for n in all_primes(100):
        if n < 50 or (n%4)!=1:
            continue

        code = build_qr_code(n)
        print(code)

        css = unwrap(code)
        css = css.to_css()
        print(css.bz_distance())
        print()
    


def make_cyclic():
    H0 = "..Y.Y...Y...Y...Y.Y...ZZXXZZ."
    n = len(H0)
    space = SymplecticSpace(n)
        
    H = []
    for i in range(n):
        H1 = ''.join(H0[(i-j)%n] for j in range(n))
        H.append(H1)
    H = "\n".join(H)
    #print(H)

    F = space.F
    H = space.fromstr(H)

    if argv.gf4:
        op = reduce(mul, [space.S(i) for i in range(n)])
        op *= reduce(mul, [space.H(i) for i in range(n)])
        H1 = H*op
        H = H.concatenate(H1)

    A = H * F * H.t
    assert A.sum() == 0
    H = H.linear_independent()
    code = QCode(H, cyclic=True)

    print(code)
    print(strop(code.H))

    if argv.distance:
        code.distance("z3", True)
        print(code)


def test_residues():
    #n = argv.get("n", 5)
    n = argv.n

    code = build_qr_code(n)
    H = code.H
    print(strop(H))
    print()

    m, nn = H.shape

#    for i in range(1, m):
#      for j in range(i+1, m):
#        h = H[0] + H[i] + H[j]
#        print(strop(h), get_weight(h.A))

    best = n
    best_h = None
    #for h in H.rowspan():
    for bits in numpy.ndindex((2,)*m):
        bits = Matrix(bits)
        h = bits * H
        #print(h, h.shape)
        A = h.A.copy()
        A.shape = (nn,)
        w = get_weight(A) 
        if 0 < w < best:
            best = w
            best_h = h
            print(strop(best_h), w)

    print(strop(best_h), best)



def find_cyclic():
    n = argv.get("n", 5)
    nn = 2*n

    from qumba.transversal import UMatrix, Solver, z3
    solver = Solver()
    Add = solver.add

    H0 = UMatrix.unknown(1, nn)
    H0[0,0:2] = [1,0] # X

    if n==29:
        print(" guess from smaller examples..? ")
        H0[0,2:4] = [1,0] # X
        H0[0,4:6] = [0,1] # Z
        H0[0,6:8] = [0,0] # .
        H0[0,8:10] = [0,1] # Z

    H = UMatrix.unknown(n, nn)
    for i in range(n):
      for j in range(n):
        jj = (j-i)%n
        H[i, 2*j:2*j+2] = H0[0, 2*jj:2*jj+2]

    #print(H)

    space = SymplecticSpace(n)
    F = space.F

    Add( H * F * H.t == 0 ) # isotropic

    # X or Z, no Y
    for i in range(1, n):
        Add( H0[0,2*i]*H0[0,2*i+1] == 0 )

    if argv.gf4:
        op = reduce(mul, [space.S(i) for i in range(n)])
        op *= reduce(mul, [space.H(i) for i in range(n)])
        V = op*H0.t
        W = UMatrix.unknown(n, 1)
        Add( H.t * W == V )

    while 1:
        result = solver.check()
        if result != z3.sat:
            break

        model = solver.model()
        h0 = H0.get_interp(model)

        h = H.get_interp(model)

        #for i in range(n):
        #    Add(H0 != H[i:i+1, :])
        Add( H0 != h0 )

        #(strop(h))
        h = h.linear_independent()
        if len(h) == n or len(h) == 0: # k==0
            continue
        code = QCode(h, cyclic=True)

        code.distance("z3")

        if code.d > 2:
            print(strop(h0), code,
                #"css" if code.is_css() else "   ", 
                'gf4' if code.is_gf4() else "   ")

        #break

    print("done.")


def test_galois():

    left = get_code((17,1,7,0))
    right = get_code((17,1,7,1))
    space = left.space

    print(left)
    n = left.n

    gl = list(range(1, n))
    gens = []
    for a in gl:
        found = set( (a**i)%n for i in range(n) )
        if len(found) == n-1:
            gens.append(a)

    print("gens:", gens)

    rotate = [(i+1)%n for i in range(n)]
    print("rotate:", rotate)

    logical = []

    for a in gens:
        cycle = []
        for i in range(n-1):
            cycle.append( (a**i)%n )

        perm = [0] + [None]*(n-1)
        for i in range(n-1):
            perm[cycle[i]] = cycle[(i+1)%(n-1)]
        print(perm)

        P = space.get_perm(perm)

        #if n==17:
        #    P = P*P

        dode = P*left

        #print(dode.is_equiv(left), dode.is_equiv(right))

        for g in find_local_cliffords(right, dode, constant=True):
            eode = g*dode
            assert eode.is_equiv(right)
            print("found")
            break
        else:
            print("not found")

    
def main_galois():
    code = QCode.fromstr("""
    XIIXIXXIIXXXXXIIIXXIXXXIXIXIIII
    IIXIXXIIXXXXXIIIXXIXXXIXIXIIIIX
    IXIXXIIXXXXXIIIXXIXXXIXIXIIIIXI
    XIXXIIXXXXXIIIXXIXXXIXIXIIIIXII
    IXXIIXXXXXIIIXXIXXXIXIXIIIIXIIX
    ZZZIIZZIZZZZZIZIIIZIIZIZIZZIIII
    ZZIIZZIZZZZZIZIIIZIIZIZIZZIIIIZ
    ZIIZZIZZZZZIZIIIZIIZIZIZZIIIIZZ
    IIZZIZZZZZIZIIIZIIZIZIZZIIIIZZZ
    IZZIZZZZZIZIIIZIIZIZIZZIIIIZZZI
    """)

    assert code.is_cyclic()
    print(code)
    n = code.n
    space = code.space

    gl = list(range(1, n))
    gens = []
    for a in gl:
        found = set( (a**i)%n for i in range(n) )
        if len(found) == n-1:
            gens.append(a)

    print("gens:", gens)


    for a in gens:
        cycle = []
        for i in range(n-1):
            cycle.append( (a**i)%n )

        perm = [0] + [None]*(n-1)
        for i in range(n-1):
            perm[cycle[i]] = cycle[(i+1)%(n-1)]
        #print(perm)

        P = space.get_perm(perm)
        dode = P*code
        print(dode.is_equiv(code), dode.is_cyclic())


def main_toric():
    #code = construct.get_toric(1, 3)
    code = construct.get_513()
    n = code.n
    N, gens = code.get_autos()

    #G = mulclose([Matrix.get_perm(g) for g in gens])
    #assert len(G) == N
    print("code auts:", N)

    from bruhat.gset import Perm, Group
    gens = [Perm(g) for g in gens]
    G = Group.generate(gens)
    print(G)

    mul = zeros2(N, N*N)
    for i,g in enumerate(G):
      for j,h in enumerate(G):
        k = G.lookup[g*h]
        mul[k, i + N*j] = 1
    #print(shortstr(mul), mul.shape)
    mul = Matrix(mul)

    from qumba.umatrix import UMatrix, Solver, Not, Or, And
    solver = Solver()
    add = solver.add

    U = UMatrix.get_perm(solver, N)

    #for g in G:
    #    add( U*g == g*U )
    add( U*mul == mul*(U@U) )

    found = []
    while 1:

        result = solver.check()
        if str(result) != "sat":
            break
        
        model = solver.model()
        u = U.get_interp(model)
    
        #print(u)

        found.append(u)
        add(U != u)

    print(len(found))

    for g in found:
        print(g.order(), end=" ")
    print()

    #A = mulclose(found)
    #print(len(A))

    space = code.space
    for g in found:
        perm = g.to_perm()
        print(perm)
        u = space.get_perm(perm)
        dode = u*code
        print(dode, dode.is_equiv(code))




            


def main_galois_15():
    code = QCode.fromstr("""
    XIXIIXXIXXXIIII
    IXIIXXIXXXIIIIX
    XIIXXIXXXIIIIXI
    IIXXIXXXIIIIXIX
    IXXIXXXIIIIXIXI
    ZZZZIZIZZIIZIII
    ZZZIZIZZIIZIIIZ
    ZZIZIZZIIZIIIZZ
    ZIZIZZIIZIIIZZZ
    """)
    assert code.is_cyclic()
    print(code)

    n = code.n
    space = code.space

    gen = []
    for p in get_cyclic_perms(n):
        dode = p*code
        assert dode.is_cyclic()
        if code.is_equiv(dode):
            L = code.get_logical(dode)
            #print(L, "\n")
            gen.append(L)
    G = mulclose(gen)
    print("G:", len(G))




def test_cyclotomic():

    code = get_code()
    space = code.space

    print(code)
    n = code.n

    gl = list(range(1, n))
    gens = []
    for a in gl:
        found = set( (a**i)%n for i in range(n) )
        if len(found) == n-1:
            gens.append(a)

    print("gens:", gens)

    rotate = [(i+1)%n for i in range(n)]
    print("rotate:", rotate)

    logical = []

    for a in gens:
        cycle = []
        for i in range(n-1):
            cycle.append( (a**i)%n )

        perm = [0] + [None]*(n-1)
        for i in range(n-1):
            perm[cycle[i]] = cycle[(i+1)%(n-1)]
        print(perm)

        P = space.get_perm(perm)

        #if n==17:
        #    P = P*P

        dode = P*code

        eode = space.get_perm(rotate)*dode
        print("cyclic:", eode.is_equiv(dode))
        print("gf4:", dode.is_gf4())

        for g in find_local_cliffords(code, dode, constant=True):
            eode = g*dode
            assert eode.is_equiv(code)
            L = eode.get_logical(code)
            logical.append(L)
            print("found")
            print(L)
            break
        else:
            print("not found")

    G = mulclose(logical)
    print("gates:", len(G))



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


