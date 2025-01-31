#!/usr/bin/env python
"""
_looking for transversal logical clifford operations
"""

from functools import reduce
from operator import add, matmul, mul
from random import shuffle, choice

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, ForAll

from bruhat.action import mulclose, Group, Perm, mulclose_find

from qumba.qcode import QCode, SymplecticSpace, fromstr, shortstr, strop
from qumba.matrix import Matrix, scalar
from qumba import csscode
#from qumba.action import mulclose, Group, Perm, mulclose_find
from qumba.util import allperms
from qumba import equ
from qumba import construct 
from qumba import autos
from qumba.unwrap import unwrap, Cover
from qumba.argv import argv
from qumba.umatrix import UMatrix
from qumba.lin import zeros2
from qumba.csscode import CSSCode
from qumba import unwrap
from qumba.util import choose


def find_equivariant(X):
    G = X.G
    n = len(X)
    print("find_equivariant", n)

    perms = []
    for g in G:
        send = X(g)
        items = send.items # Coset's
        lookup = dict((v,k) for (k,v) in enumerate(items))
        send = {i:lookup[send[items[i]]] for i in range(len(items))}
        assert len(send) == n
        perms.append(send)

    solver = Solver()
    Add = solver.add

    hx = UMatrix.unknown(n)
    hz = UMatrix.unknown(n)

    # we have a transitive G action, so we can fix the 0 bit:
    hx[0] = 1
    hz[0] = 1

    Hx = [[hx[g[i]] for i in range(n)] for g in perms]
    Hx = UMatrix(Hx)

    c = Hx * hz.t
    Add(c==0)

    weight = lambda h : Sum([If(h[i].get(),1,0) for i in range(1, n)])+1
    row_weight = argv.get("row_weight")
    print("row_weight:", row_weight)
    if row_weight is not None:
        Add(weight(hx) <= row_weight)
        Add(weight(hz) <= row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("unsat")
            return
    
        model = solver.model()
        _hx = hx.get_interp(model)
        _hz = hz.get_interp(model)

        #Add(hx != _hx)
        #Add(hz != _hz)
    
        #hx = Hx[0]
        #print("hx:", _hx)
        #print("hz:", _hz)
    
        Hx = Matrix([[_hx[g[i]] for i in range(n)] for g in perms])
        for _hx in Hx:
            Add(hx != _hx)
        Hx = Hx.linear_independent()
    
        Hz = Matrix([[_hz[g[i]] for i in range(n)] for g in perms])
        for _hz in Hz:
            Add(hz != _hz)
        Hz = Hz.linear_independent()
    
        code = csscode.CSSCode(Hx=Hx.A, Hz=Hz.A)
        if code.k == 0:
            continue

        yield code


def find_equivariant_sd(X):
    G = X.G
    n = len(X)
    print("find_equivariant_sd", n)

    perms = []
    for g in G:
        send = X(g)
        items = send.items # Coset's
        lookup = dict((v,k) for (k,v) in enumerate(items))
        send = {i:lookup[send[items[i]]] for i in range(len(items))}
        assert len(send) == n
        perms.append(send)

    solver = Solver()
    Add = solver.add

    h = UMatrix.unknown(n)

    # we have a transitive G action, so we can fix the 0 bit:
    h[0] = 1

    H = [[h[g[i]] for i in range(n)] for g in perms]
    H = UMatrix(H)

    c = H * h.t
    Add(c==0)

    weight = lambda h : Sum([If(h[i].get(),1,0) for i in range(1, n)])+1
    row_weight = argv.get("row_weight")
    print("row_weight:", row_weight)
    if row_weight is not None:
        Add(weight(h) <= row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("unsat")
            return
    
        model = solver.model()
        _h = h.get_interp(model)

        #Add(hx != _hx)
        #Add(hz != _hz)
    
        #hx = Hx[0]
        #print("hx:", _hx)
        #print("hz:", _hz)
    
        Hx = Matrix([[_h[g[i]] for i in range(n)] for g in perms])
        for _h in Hx:
            Add(h != _h)
        Hx = Hx.linear_independent()
        Hz = Hx
    
        code = csscode.CSSCode(Hx=Hx.A, Hz=Hz.A)
        if code.k == 0:
            print(".", end="", flush=True)
            continue

        code.bz_distance()
        yield code


def fail_search_equivarant(X):
    G = X.G
    n = len(X)
    print("search_equivarant", n)

    perms = []
    for g in G:
        send = X(g)
        items = send.items # Coset's
        lookup = dict((v,k) for (k,v) in enumerate(items))
        send = [lookup[send[items[i]]] for i in range(len(items))]
        U = Matrix.perm(send)
        perms.append(U)
    G = perms

    irreps = []
    v = [0]*n
    v[0] = 1
    v = Matrix(v)
    orbit = []
    for g in G:
        vg = v*g
        orbit.append(vg)
        print(vg)

    return []



def prune(Xs):
    i = 0
    while i < len(Xs):
        j = i+1
        while j < len(Xs) and i < len(Xs):
            if Xs[i].isomorphic(Xs[j]):
                Xs.pop(j)
            else:
                j += 1
        i += 1


def isomorphic_css(code, dode):
    for iso in find_isomorphisms_css(code, dode):
        return True
    return False

def isomorphic(code, dode):
    for iso in find_isomorphisms(code, dode):
        return True
    return False


def unique_css(codes, min_d=3, found=None):
    if found is None:
        found = {} # map each code to a unique representative

    for code in codes:

        if code.n < 40:
            #print("/", end="", flush=True)
            dx, dz = code.bz_distance()
            #print("\\", end="", flush=True)
            if code.d < min_d:
                continue

        else:
            w = csscode.distance_meetup(code, max_m=2)
            if w is not None and w < min_d:
                continue

        #if code.d is not None and code.d < min_d:
        #    print("d.", end="", flush=True)
        #    continue
        #print("\n", code.d, min_d)
        for prev in list(found.keys()):
            if code.is_equiv(prev):
                found[code] = found[prev]
                print("e.", end="", flush=True)
                break
            if code.get_dual().is_equiv(prev):
                found[code] = found[prev]
                print("f.", end="", flush=True)
                break
            #print("i?", end='', flush=True)
            #if isomorphic_css(code, prev):
            #    found[code] = found[prev]
            #    print("y", end="", flush=True)
            #    break
            #print("n", end="", flush=True)
        else:
            found[code] = code # i am the representative
            yield code
    print("\nfound:", len(set(found.values())), "of", len(set(found.keys())))




def get_group():
    # gap> LoadPackage( "AtlasRep", false );
    # gap> G := AtlasGroup("L2(11)");
    # Group([ (2,10)(3,4)(5,9)(6,7), (1,2,11)(3,5,10)(6,8,9) ])

    if argv.cyclic:
        n = argv.get("n", 10)
        G = Group.cyclic(n)
        G.name = "C_%d"%n

    elif argv.dihedral:
        n = argv.get("n", 10)
        G = Group.dihedral(n)
        G.name = "D_%d"%n

    elif argv.symmetric:
        m = argv.get("m", 4)
        G = Group.symmetric(m)
        G.name = "S_%d"%m

    elif argv.coxeter_bc:
        m = argv.get("m", 3)
        G = Group.coxeter_bc(m)
        n = len(G)
        G.name = "CoxeterBC_%d"%m

    elif argv.Sp2:
        items = list(range(16))
        a = Perm([0, 15, 9, 6, 3, 12, 10, 5, 13, 2, 4, 11, 14, 1, 7, 8], items)
        b = Perm([0, 2, 8, 10, 1, 3, 9, 11, 4, 6, 12, 14, 5, 7, 13, 15], items)
        G = Group.generate([a,b])
        assert len(G) == 720
        G.name = "Sp2"

    elif argv.GL32:
        n = 7
        items = list(range(1, n+1))
        a = Perm.fromcycles([(1,), (2,), (7,), (3,5), (4,6)], items)
        b = Perm.fromcycles([(1,6,3), (5,), (2,4,7)], items)
        G = Group.generate([a,b])
        assert len(G) == 168
        G.name = "GL32"

    elif argv.L2_8:
        n = 9
        items = list(range(1, n+1))
        a = Perm.fromcycles([ (1,2),(3,4),(6,7),(8,9) ], items)
        b = Perm.fromcycles([ (1,3,2),(4,5,6),(7,8,9) ], items)
        G = Group.generate([a,b])
        assert len(G) == 504
        X = G.tautological_action() # No solution
        X = X*X
        Xs = X.get_components()

    elif argv.L2_11:
        n = 11
        items = list(range(1, n+1))
        a = Perm.fromcycles([ (2,10),(3,4),(5,9),(6,7), ], items)
        b = Perm.fromcycles([(1,2,11),(3,5,10),(6,8,9) ], items)
        G = Group.generate([a,b])
        assert len(G) == 660
        X = G.tautological_action() # No solution
        X = X*X
        Xs = X.get_components()

    elif argv.L2_13:
        n = 14
        items = list(range(1, n+1))
        a = Perm.fromcycles([(1,12),(2,6),(3,4),(7,11),(9,10),(13,14)], items)
        b = Perm.fromcycles([ (1,6,11),(2,4,5),(7,8,10),(12,14,13) ], items)
        G = Group.generate([a,b])
        assert len(G) == 1092, len(G)
        X = G.tautological_action() # No solution
        X = X*X
        Xs = X.get_components()

    elif argv.L2_16:
        n = 17
        items = list(range(1, n+1))
        a = Perm.fromcycles([ (1,2),(3,4),(5,6),(7,9),(10,11),(12,13),(14,16),(15,17), ], items)
        b = Perm.fromcycles([ (1,3,2),(4,5,7),(6,8,10),(11,12,14),(13,15,17) ], items)
        G = Group.generate([a,b])
        assert len(G) == 4080
        X = G.tautological_action() # No solution
        Xs = [X]

    elif argv.M11:
        n = 11
        items = list(range(1, n+1))
        a = Perm.fromcycles([(2,10),(4,11),(5,7),(8,9)], items)
        b = Perm.fromcycles([(1,4,3,8),(2,5,6,9)], items)
        G = Group.generate([a,b])
        assert len(G) == 7920
        X = G.tautological_action()
        Xs = [X]
        G.name = "M11"

    elif argv.M20:
        n = 20
        items = list(range(1, n+1))
        a = Perm.fromcycles(
            [(1,2,4,3),(5,11,7,12),(6,13),(8,14),(9,15,10,16),(17,19,20,18)], items)
        b = Perm.fromcycles(
            [(2,5,6),(3,7,8),(4,9,10),(11,17,12),(13,16,18),(14,15,19)], items)
        G = Group.generate([a,b], verbose=True)
        assert len(G) == 960
        X = G.tautological_action()
        Xs = [X]
        G.name = "M20"

    elif argv.M21:
        n = 21
        items = list(range(1, n+1))
        a = Perm.fromcycles([(1,2),(4,6),(5,7),(8,12),(9,14),(10,15),(11,17),(13,19)], items)
        b = Perm.fromcycles([(2,3,5,4),(6,8,13,9),(7,10,16,11),(12,18),(14,20,21,15),(17,19)], items)
        G = Group.generate([a,b], verbose=True)
        assert len(G) == 20160
        X = G.tautological_action()
        Xs = [X]
        G.name = "M21"

    elif argv.alternating:
        m = argv.get("m", 4)
        G = Group.alternating(m)
        G.name = "A_%d"%m

        for g in G:
          for h in G:
            if g*h != h*g:
                print("non-abelian")
                break
          else:
            continue
          break

    else:
        return

    G.desc = "?"
    return G


def search_equivariant():
    # build equivariant CSS codes from hecke operators.
    # does not work very well. todo: use hecke operators to
    # find irreps.
    Hs = None
    Xs = None
    n = None

    G = get_group()

    n = argv.get("n", n or len(G))
    print("|G| =", len(G))

    Hs = [H for H in G.conjugacy_subgroups() if len(H)<len(G)]
    #Hs = [H for H in G.subgroups()]
    Hs.sort(key = len)
    print("indexes:", [len(G)//len(H) for H in Hs])
    Xs = [G.action_subgroup(H) for H in Hs]
    print("|Xs| =", len(Xs))

    #for X in Xs:
    #Y = Xs[16]

    for Y in Xs[1:]:

        n = len(Y)
        space = SymplecticSpace(n)

        if argv.n and n != argv.n:
            continue

        Hs = set()
        for X in Xs:
            for H in X.hecke(Y):
                H = H.astype(scalar)
                H = Matrix(H)
                H = H.row_reduce()
                Hs.add(H)
    
        Hs = list(Hs)
        Hs.sort(key = lambda H: (len(str(H)), str(H)))

        autos = []
        for g in G:
            send = Y(g)
            items = send.items # Coset's
            lookup = dict((v,k) for (k,v) in enumerate(items))
            send = [lookup[send[items[i]]] for i in range(len(items))]
            g = space.P(*send)
            autos.append(g)
    
        for A in Hs:
          for B in Hs:
            if (A*B.t).sum() == 0:
                code = QCode.build_css(A, B)
                if code.k == 0:
                    continue
                if code.d and code.d <= 2:
                    #print(".", end='', flush=True)
                    continue
                css = code.to_css()
                if code.n < 40:
                    css.bz_distance()
                code = css.to_qcode()
                if argv.show:
                    print()
                    print(code.cssname)
                    print(strop(code.H))
                else:
                    print(code, end='', flush=True)
                L = set()
                for g in autos:
                    dode = g*code
                    assert dode.is_equiv(code)
                    l = dode.get_logical(code)
                    L.add(l)
                L = mulclose(L)
                if len(L) > 1:
                    print("|L| =", len(L))
        print()



def make_two_block(Lx, Rx, Lz=None, Rz=None):
    Hx = Lx.concatenate(Rx, axis=1)
    if Lz is None:
        Lz = Rx.t
        Rz = Lx.t
    Hz = Lz.concatenate(Rz, axis=1)
    Hx = Hx.linear_independent()
    Hz = Hz.linear_independent()
    code = CSSCode(Hx=Hx.A, Hz=Hz.A, check=True)
    return code


#def search_random_hecke(Ms, wa, wb):


def find_hecke_random(Ms):
    Ms = list(Ms)
    if Ms[0].shape[1] > 40:
        return
    print("find_hecke_random", len(Ms), Ms[0].shape)

    idxs = list(range(len(Ms)))
    found = set()
    trials = argv.get("trials", 100)
    #ws = [(wa,wb) for wa in range(1,7) for wb in range(wa,7)]
    if argv.wa and argv.wb:
        ws = [(argv.wa, argv.wb)]
    for (wa,wb) in ws:
      trial = 0
      while trial < trials:
        trial += 1
        if wa+wb>len(Ms):
            break
        shuffle(Ms)
        L = reduce(add, Ms[0:wa])
        R = reduce(add, Ms[wa:wa+wb])
        if L[0].sum() + R[0].sum() > 12:
            return # <------------------------ return <<<<<----

        if L*R.t != R*L.t:
            continue

        css = make_two_block(L, R, R, L)
        if css.k == 0:
            continue
        if argv.k and css.k != argv.k:
            continue
        css.bz_distance() # <--- half the time spent here
        if css.d < 3:
            #print(".", end=" ", flush=True)
            continue
        trials = max(trials, 10000) # more !
        s = str(css)
        if s not in found:
            found.add(s)
            #print(Matrix(css.Hx), css.Hx.sum(1)[0], css.Hx.sum(0)[0], wa, wb)
            rws = css.Hx.sum(1)
            cws = css.Hx.sum(0)
            print("\t"+s, "\trw=%d, cw=%d, wa=%d, wb=%d"%(
                numpy.max(rws), numpy.max(cws), wa, wb))
                #css.Hx.sum(1)[0], css.Hx.sum(0)[0], wa, wb))


def find_hecke_exhaustive(Ms):
    Ms = list(Ms)
    if Ms[0].shape[1] > 40:
        return
    shuffle(Ms)
    print("find_hecke_exhaustive", len(Ms), Ms[0].shape)

    idxs = list(range(len(Ms)))
    found = set()
    trials = argv.get("trials", 100)
    #ws = [(wa,wb) for wa in range(1,7) for wb in range(wa,7)]
    if argv.wa and argv.wb:
        ws = [(argv.wa, argv.wb)]
    M0 = Ms[0]
    for (wa,wb) in ws:
      for Ls in choose(Ms[1:], wa-1):
       for Rs in choose(Ms[1:], wb-1):
        
        L = reduce(add, Ls+(M0,))
        R = reduce(add, Rs+(M0,))
        if L[0].sum() + R[0].sum() > 12:
            return # <------------------------ return <<<<<----

        if L*R.t != R*L.t:
            continue

        css = make_two_block(L, R, R, L)
        if css.k == 0:
            continue
        if argv.k and css.k != argv.k:
            continue
        css.bz_distance() # <--- half the time spent here
        if css.d < 3:
            #print(".", end=" ", flush=True)
            continue
        trials = max(trials, 10000) # more !
        s = str(css)
        if s not in found:
            found.add(s)
            #print(Matrix(css.Hx), css.Hx.sum(1)[0], css.Hx.sum(0)[0], wa, wb)
            rws = css.Hx.sum(1)
            cws = css.Hx.sum(0)
            print("\t"+s, "\trw=%d, cw=%d, wa=%d, wb=%d"%(
                numpy.max(rws), numpy.max(cws), wa, wb))
                #css.Hx.sum(1)[0], css.Hx.sum(0)[0], wa, wb))


def search_hecke(G):
    n = argv.get("n", len(G))
    print("|G| =", len(G))

    Hs = [H for H in G.conjugacy_subgroups() if len(H)<len(G)]
    #Hs = [H for H in G.subgroups()]
    Hs.sort(key = len)
    print("indexes:", [len(G)//len(H) for H in Hs])
    Xs = [G.action_subgroup(H) for H in Hs]
    print("|Xs| =", len(Xs))

    #for X in Xs:
    #Y = Xs[16]

    #Xs = [X for X in Xs if len(X) <= 40]

    for X in Xs:
      for Y in Xs:
        Ms = set()
        for M in X.hecke(Y):
            M = M.astype(scalar)
            M = Matrix(M)
            Ms.add(M)
        if argv.exhaustive:
            find_hecke_exhaustive(Ms)
        else:
            find_hecke_random(Ms)


def test_hecke():
    G = get_group()

    if G is None:
        from bruhat.small_groups import groups
    else:
        groups = [G]

    for G in groups:
        print(len(G), G.desc)
        search_hecke(G)


def test_equivariant():
    # find equivariant CSS codes


    Hs = None
    Xs = None
    n = None

    G = get_group()

    n = argv.get("n", n or len(G))
    print("|G| =", len(G))

    if Xs is None:
        #Hs = [H for H in G.conjugacy_subgroups() if len(H)<len(G)]
        Hs = [H for H in G.subgroups() if len(H)<len(G)]
        print("indexes:")
        print('\t', [len(G)//len(H) for H in Hs])
        Xs = [G.action_subgroup(H) for H in Hs if len(G)//len(H) == n]

    print("|Xs| =", len(Xs))

    if argv.prune and len(Xs)>1:
        print("prune...")
        prune(Xs)
        print("|Xs| =", len(Xs))

#    for H in Hs:
#        if len(G)//len(H) != n:
#            continue
#        print("|H| =", len(H))
#        X = G.action_subgroup(H)
    show = argv.show

    find = find_equivariant
    if argv.sd:
        find = find_equivariant_sd

    found = {}
    for X in Xs:
        count = 0
        for code in unique_css(find(X), found=found):
            print(code, end="\t", flush=True)
            count += 1
            if count % 6 == 0:
                print()
            if argv.store_db:
                from qumba.db import add
                code = code.to_qcode(homogeneous=True, G=G.name)
                add(code)
        #print("found:", count)
    

def two_block(G, w=2):
    """
    Construct two-block group algebra codes
    # ref: https://arxiv.org/abs/2306.16400
    """

    perms = list(G)
    perms.sort(key = str)
    lookup = {p:i for (i,p) in enumerate(perms)}

    n = len(G)

    def left(g):
        L = zeros2(n,n)
        for i in range(n):
            h = g*perms[i]
            j = lookup[h]
            L[i,j] = 1
        return Matrix(L)

    def right(g):
        R = zeros2(n,n)
        for i in range(n):
            h = perms[i]*g
            j = lookup[h]
            R[i,j] = 1
        return Matrix(R)

    for g in G:
      for h in G:
        L = left(g)
        R = right(h)
        assert L*R == R*L

#    for g in G:
#        print(left(g),'\n')
    lb = [left(g) for g in G]
    rb = [right(g) for g in G]

    idxs = list(range(len(G)))

    trials = argv.get("trials", 100)

    found = set()
    #while 1:
    for trial in range(trials):

        shuffle(idxs)
        ll = [lb[i] for i in idxs]
        shuffle(idxs)
        rr = [rb[i] for i in idxs]
        L = reduce(add, ll[:w])
        R = reduce(add, rr[:w])

        assert L*R == R*L

        #L = L.linear_independent()
        #R = R.linear_independent()

        Hx = L.concatenate(R, axis=1)
        #print(Hx, Hx.shape)

        Hz = R.t.concatenate(L.t, axis=1)
        #print(Hz, Hz.shape)

        Hx = Hx.linear_independent()
        Hz = Hz.linear_independent()

        code = CSSCode(Hx=Hx.A, Hz=Hz.A, check=True)
        #print(code)
        if code.k == 0:
            continue
        code.bz_distance()
        if code.dx <= 2 or code.dz <= 2:
            continue

        s = str(code)
        if s not in found:
            #print(s)
            #print(Hx)
            #print(Hx.sum(1))
            found.add(s)
            yield code

        if 0:
            code = code.to_qcode()
            n = code.n
            dode = code.apply_H()
            f = list(range(n//2, n)) + list(range(n//2))
            #for i in range(n//2):
                #f.append(i)
            dode = dode.apply_perm(f)
            if not dode.is_equiv(code):
                continue
    
            perm = Perm(f, list(range(n)))
            cover = unwrap.Cover.fromzx(code, perm)
            print(code)
            print(cover.base)
            print(cover.base.longstr())
    
            break


def test_two_block():
    G = get_group()
    w = argv.get("w", 2)
    print(G)
    for code in two_block(G, w):
        print(code)


def all_two_block():
    from bruhat.small_groups import groups
    w = argv.get("w", 4)
    for G in groups:
        print(len(G), G.desc)
        for code in two_block(G, w):
            print("\t%s"%code)


def test_bimodule():
    G = Group.symmetric(4)
    print(G)

    Hs = G.subgroups()
    print(len(Hs))

#    from bruhat.biset import Biset
#    for H in Hs:
#      for J in Hs:
#        X = Biset.double_coset(G, H, J)
#        print(X, len(H), len(J), len(X.items))
#        m = len(G) // len(H)
#        n = len(G) // len(J)
#        lookup = {(i//m, i%m):x for (i,x) in enumerate(X.items)}
#        for g in G:
#            
#
#    return

    for H in Hs:
        X = G.action_subgroup(H)
        if len(H) == 1:
            break

    for H in Hs:

        #if not H.is_cyclic():
        #    continue

        G_H = G.left_cosets(H)
        X = G.left_action(G_H, H)

        #H_G = G.right_cosets(H)
        H_G = G_H
        Y = G.right_action(H_G, H)

        print(len(H), H.is_cyclic(), end=" ")
        print(set(G_H) == set(H_G))

#        assert len(G_H) == len(H_G)
#        n = len(G_H)
#        for g in G:
#            L = zeros2(n,n)
#            for i,x in enumerate(G_H):
#                y = X(g)[x]
#                j = G_H.index(y)
#                L[i,j] = 1
#            L = Matrix(L)
#
#            for h in G:
#                R = zeros2(n,n)
#                for i,x in enumerate(H_G):
#                    y = Y(h)[x]
#                    j = H_G.index(y)
#                    R[i,j] = 1
#                R = Matrix(R)
#                print(int(L*R==R*L), end="")
#            print()


    return

    items = list(X.items)
    items.sort(key = str)
    lookup = {p:i for (i,p) in enumerate(items)}
    n = len(items)

    def left(g):
        L = zeros2(n,n)
        for i,x in enumerate(items):
            y = X(g)[x]
            j = lookup[y]
            L[i,j] = 1
        return Matrix(L)

    def right(g):
        return left(~g).t

#    for g in G:
#        print(left(g))
#        print()
#
#    return
#
#    def right(g):
#        R = zeros2(n,n)
#        for i in range(n):
#            h = perms[i]*g
#            j = lookup[h]
#            R[i,j] = 1
#        return Matrix(R)

    for g in G:
      for h in G:
        L = left(g)
        R = right(h)
        assert L*R == R*L



if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))






