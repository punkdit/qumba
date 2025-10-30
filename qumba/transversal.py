#!/usr/bin/env python
"""
_looking for transversal logical clifford operations
"""

from functools import reduce
from operator import add, matmul, mul
from random import shuffle

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, ForAll, PbEq

from qumba.qcode import QCode, SymplecticSpace, fromstr, shortstr, strop
from qumba.matrix import Matrix, scalar
from qumba import csscode
from qumba.action import mulclose, Group, Perm, mulclose_find, mulclose_hom
from qumba.util import allperms, cross
from qumba import equ
from qumba import construct 
from qumba import autos
from qumba import lin
from qumba.unwrap import unwrap, Cover
from qumba.argv import argv
from qumba.umatrix import UMatrix
from qumba.smap import SMap


def find_transversal(*codes, constant=False, verbose=True):
    solver = Solver()
    Add = solver.add

    m = len(codes)
    n = codes[0].n
    code = reduce(add, codes)
    H = code.H
    L = code.L
    k = code.k

    space = SymplecticSpace(m)
    Fm = space.F

    if not constant:
        items = []
        for i in range(n):
            U = UMatrix.unknown(2*m, 2*m)
            Add(U.t*Fm*U == Fm) # quadratic constraint
            items.append(U)
        U = reduce(UMatrix.direct_sum, items)
        U0 = None

    else:
        U0 = UMatrix.unknown(2*m, 2*m)
        Add(U0.t*Fm*U0 == Fm) # quadratic constraint
        U = reduce(UMatrix.direct_sum, [U0]*n)

    perm = numpy.array(list(range(n*m)))
    perm.shape = (m, n)
    perm = perm.transpose().copy()
    perm.shape = m*n
    perm = list(perm)
    P = code.space.get_perm(perm)
    U = P.t * U * P

    HU = H * U.t
    LU = L * U.t
    F = code.space.F
    R = HU * F * L.t
    Add(R==0) # linear constraint
    R = HU * F * H.t
    Add(R==0) # linear constraint

    E, D = code.get_encoder(), code.get_decoder()
    LU = D*U*E
    LU = LU[-2*k:, -2*k:]
    #I = Matrix.identity(2*k)
    #Add(LU!=I)

    found = set()
    gen = set()
    fgen = set()
    count = 0
    while 1:
        count += 1
        result = solver.check()
        if result != z3.sat:
            break
        #if count%100==0:
        #    print(".", end="", flush=True)
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = code.apply(M)
        assert dode.is_equiv(code)
        #L = dode.get_logical(code)
        L = LU.get_interp(model)
        #print(LU)
        #print(LU.get_interp(model))
        #print(L)
        #assert L == LU.get_interp(model) # whoops.. not the same..
        if L not in found:
            yield M, L
            found.add(L)
            #Add(LU != L) # slows things down..
        gen.add(M)
        Add(U != M)
        if verbose:
            print("mulclose...", end="", flush=True)
        G = mulclose(gen, verbose=verbose)
        if verbose:
            print("done")
        for g in G:
            if g not in fgen:
                #if U0 is not None:
                #    Add(U0 != g[:2*m, :2*m]) # doesn't work..
                #else:
                Add(U != g)
                fgen.add(g)
        if verbose:
            print("gen:", len(gen), "fgen:", len(fgen))


def find_local_cliffords(tgt, src=None, constant=False, verbose=True):
    #print("find_local_clifford")
    if src is None:
        src = tgt

    solver = Solver()
    Add = solver.add

    assert tgt.n == src.n
    assert tgt.k == src.k
    m = 1
    n = src.n

    space = SymplecticSpace(m)
    Fm = space.F

    if not constant:
        items = []
        for i in range(n):
            U = UMatrix.unknown(2*m, 2*m)
            Add(U.t*Fm*U == Fm) # quadratic constraint
            items.append(U)
        U = reduce(UMatrix.direct_sum, items)
        U0 = None

    else:
        U0 = UMatrix.unknown(2*m, 2*m)
        Add(U0.t*Fm*U0 == Fm) # quadratic constraint
        U = reduce(UMatrix.direct_sum, [U0]*n)

    HU = src.H * U.t
    LU = src.L * U.t
    F = src.space.F
    if tgt.k:
        R = HU * F * tgt.L.t
        Add(R==0) # linear constraint
    R = HU * F * tgt.H.t
    Add(R==0) # linear constraint

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = src.apply(M)
        assert dode.is_equiv(tgt)
        yield M

        Add(U != M)


def get_local_clifford(tgt, src, constant=False, verbose=False):
    for M in find_local_cliffords(tgt, src, constant, verbose):
        return M


def is_local_clifford_equiv(tgt, src, constant=False, verbose=False):
    for M in find_local_cliffords(tgt, src, constant, verbose):
        return True
    return False


def find_isomorphisms(code, dode=None):

    # find all automorphism permutations

    if dode is None:
        dode = code

    if code.n != dode.n or code.k != dode.k:
        return

    if code.d is not None and dode.d is not None and code.d != dode.d:
        return

    n = code.n
    nn = 2*n
    m = code.m
    space = code.space
    F = space.F

    solver = Solver()
    Add = solver.add

    # permutation matrix
    P = UMatrix.unknown(n, n)

    # symplectic permutation matrix
    A = numpy.zeros((nn,nn), dtype=object)
    for i in range(n):
      for j in range(n):
        A[2*i,2*j] = P[i,j]
        A[2*i+1,2*j+1] = P[i,j]
    P2 = UMatrix(A)

    for i in range(n):
      for j in range(n):
        rhs = reduce(And, [P[i,k]==0 for k in range(n) if k!=j])
        Add( Or(P[i,j]==0, rhs) )
        rhs = reduce(And, [P[k,j]==0 for k in range(n) if k!=i])
        Add( Or(P[i,j]==0, rhs) )

    for i in range(n):
        Add( reduce(Or, [P[i,j]!=0 for j in range(n)]) )

    #I = Matrix.identity(n, n)
    #Add(P * P.t == I)

    #H = code.H
    #H1 = H * P2

    R = UMatrix.unknown(m, m)
    #Add( R*H == H1 )
    Add( R*code.H == dode.H*P2 )

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        p = P.get_interp(model)
        p2 = P2.get_interp(model)

        assert (p2*code).is_equiv(dode)

        yield p2

        Add(P != p) # could also add all generators

find_autos = find_isomorphisms

def find_lw_css(code):
    "find low-weight row vectors in the stabilizers"
    code = code.to_css()

    H = code.Hx
    hx = list(find_lw(code.Hx))
    hz = list(find_lw(code.Hz))
    #print("find_lw_css:")
    #for h in hx:
    #    print(h)
    return hx, hz


def find_lw(H, w=None):
    "find low-weight row vectors in the row-span of H"
    m, n = H.shape

    if w is None:
        ws = H.sum(1)
        w = ws.min()

    solver = Solver()
    Add = solver.add

    u = UMatrix.unknown(1, m)
    
    H = Matrix(H)
    #print(H)

    uH = u*H
    #print(uH)

    #Add(Sum([If(uH[0,i].get(),1,0) for i in range(n)])==w)

    Add( PbEq([(uH[0,i].get(), True) for i in range(n)], w) )

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        v = u.get_interp(model)

        vH = v*H
        assert vH.sum() == w
        yield vH

        Add(u != v)


def find_wenum(H, wx, wz, wy):
    "find weight=(wx,wz,wy) row vectors in the row-span of H"
    m, nn = H.shape
    assert nn%2 == 0
    n = nn//2

    solver = Solver()
    Add = solver.add

    u = UMatrix.unknown(1, m)
    H = Matrix(H)
    uH = u*H
    uH = uH[0, :]
    #Add(Sum([If(uH[0,i].get(),1,0) for i in range(n)])==w)
    #Add( PbEq([(uH[0,i].get(), True) for i in range(n)], w) )

    row = [uH[i].get() for i in range(nn)]
    xs = [And(row[2*i], Not(row[2*i+1])) for i in range(n)]
    zs = [And(Not(row[2*i]), (row[2*i+1])) for i in range(n)]
    ys = [And((row[2*i]), (row[2*i+1])) for i in range(n)]
    Add(PbEq([(item,True) for item in xs], wx))
    Add(PbEq([(item,True) for item in zs], wz))
    Add(PbEq([(item,True) for item in ys], wy))

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        v = u.get_interp(model)

        vH = v*H
        yield vH

        Add(u != v)




def find_isomorphisms_css(code, dode=None, ffinv=False):

    # find all automorphism permutations

    if dode is None:
        dode = code

    if code.n != dode.n or code.k != dode.k:
        return

    if code.d is not None and dode.d is not None and code.d != dode.d:
        return

    code = code.to_css()
    dode = dode.to_css()

    n = code.n
    mx = code.mx
    mz = code.mz

    if code.mx != dode.mx or code.mz != dode.mz:
        return

    solver = Solver()
    Add = solver.add

    # permutation matrix
    P = UMatrix.unknown(n, n)

    if ffinv:
        Add( P == P.t )

    #for i in range(n):
    #    Add(Sum([If(P[i,j].get(),1,0) for j in range(n)])==1)

    for i in range(n):
      for j in range(n):
        rhs = reduce(And, [P[i,k]==0 for k in range(n) if k!=j])
        Add( Or(P[i,j]==0, rhs) )
        rhs = reduce(And, [P[k,j]==0 for k in range(n) if k!=i])
        Add( Or(P[i,j]==0, rhs) )

    for i in range(n):
        Add( reduce(Or, [P[i,j]!=0 for j in range(n)]) )

    #I = Matrix.identity(n, n)
    #Add(P * P.t == I)

    #H = code.H
    #H1 = H * P2

    Rx = UMatrix.unknown(mx, mx)
    Rz = UMatrix.unknown(mz, mz)

    Hx = Matrix(code.Hx)
    Hz = Matrix(code.Hz)
    Jx = Matrix(dode.Hx)
    Jz = Matrix(dode.Hz)

    Add( Rx*Hx == Jx*P )
    Add( Rz*Hz == Jz*P )

    gen = []
    found = set()

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        g = P.get_interp(model)

        #assert (g*code).is_equiv(dode) # ... fix..

        #print(p)
        gen.append(g)

        yield g
        Add(P != g) # could instead add generated perms, see below->

#        G = mulclose(gen, verbose=False)
#        for g in G:
#            if g not in found:
#                yield g
#                #print("\\", end='', flush=True)
#                Add(P!=g)
#                #print("/", end='', flush=True)
#                found.add(g)
#


def find_isomorphisms_selfdual(code, dode=None, ffinv=False):

    # find all automorphism permutations

    if dode is None:
        dode = code

    if code.n != dode.n or code.k != dode.k:
        return

    if code.d is not None and dode.d is not None and code.d != dode.d:
        return

    code = code.to_css()
    dode = dode.to_css()

    n = code.n
    mx = code.mx
    mz = code.mz

    if code.mx != dode.mx or code.mz != dode.mz:
        return

    assert mx==mz

    solver = Solver()
    Add = solver.add

    # permutation matrix
    P = UMatrix.unknown(n, n)

    if ffinv:
        Add( P == P.t )

    #for i in range(n):
    #    Add(Sum([If(P[i,j].get(),1,0) for j in range(n)])==1)

    for i in range(n):
      for j in range(n):
        rhs = reduce(And, [P[i,k]==0 for k in range(n) if k!=j])
        Add( Or(P[i,j]==0, rhs) )
        rhs = reduce(And, [P[k,j]==0 for k in range(n) if k!=i])
        Add( Or(P[i,j]==0, rhs) )

    for i in range(n):
        Add( reduce(Or, [P[i,j]!=0 for j in range(n)]) )

    #I = Matrix.identity(n, n)
    #Add(P * P.t == I)

    #H = code.H
    #H1 = H * P2

    Rx = UMatrix.unknown(mx, mx)
    Hx = Matrix(code.Hx)
    Jx = Matrix(dode.Hx)
    Add( Rx*Hx == Jx*P )

    gen = []
    found = set()

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        g = P.get_interp(model)

        #assert (g*code).is_equiv(dode) # ... fix..

        #print(p)
        gen.append(g)

        yield g
        Add(P != g) # could instead add generated perms, see below->


def find_autos_lc(code):

    # find all local clifford automorphism permutations

    n = code.n
    nn = 2*n
    space = code.space
    F = space.F

    solver = Solver()
    Add = solver.add

    U = UMatrix.unknown(nn, nn)

    for i in range(nn):
        Add(Sum([If(U[i,j].get(),1,0) for j in range(nn)])<=2)

    I = Matrix.identity(n, n)

    Add(U.t*F*U == F) # quadratic constraint

    H = code.H
    m = code.m
    H1 = H * U.t

    # U preserves the stabilizer group
    R = UMatrix.unknown(m, m)
    Add( R*H == H1 )

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        u = U.get_interp(model)

        assert (u*code).is_equiv(code)

        yield u

        Add(U != u) # could also add all generators




def main():
    if argv.code == (4,2,2):
        code = QCode.fromstr("XXXX ZZZZ")
    elif argv.code == (5,1,3):
        code = construct.get_513()
    #elif argv.code == (4,1,2):
    #    code = QCode.fromstr("XYZI IXYZ ZIXY")
    elif argv.code == (4,1,2):
        code = QCode.fromstr("XXZZ ZZXX XZXZ")
    elif argv.code == (6,2,2):
        code = QCode.fromstr("XXXIXI ZZIZIZ IYZXII IIYYYY")
    elif argv.code == (6,3,2):
        code = QCode.fromstr("XXXXXX ZZZZII IIIIZZ")
    elif argv.code == (7,1,3) and argv.css:
        code = construct.get_713()
    elif argv.code == (7,1,3):
        code = QCode.fromstr("""
        XIXIIII IZIIZII XXIXXII ZIZZIZI IIIYZXX IIIIIZZ
        """)
        #print(code.get_params())
        #return
    elif argv.code == (10,2,3):
        code = construct.get_10_2_3()
    elif argv.code == (10,1,4):
        code = QCode.fromstr("""
        XZ.Z.X.ZZ.
        .Y.ZZY..ZZ
        ..YZZY.X..
        .ZZY.YZ..Z
        .Z..XYZY..
        .ZZ.ZZXXZZ
        ..ZZZZZ.XZ
        .ZZZZ..ZZX
        ZZZZZZ....
        """)
    elif argv.code == (13,1,5):
        n = 13
        check = "ZXIIIIIIXZIII"
        checks = [''.join(check[(i+j)%13] for i in range(n)) for j in range(n-1)]
        code = QCode.fromstr(' '.join(checks))
        print(code.get_params())
    elif argv.code == "YY":
        code = QCode.fromstr("YY", None, "Y. ZZ")
    elif argv.code == "YYY":
        code = QCode.fromstr("YYY")
    elif argv.code:
        code = QCode.fromstr(argv.code)
    else:
        return

    print(code.longstr())

    #for N in [1, 2, 3, 4, 5, 6]:
    for N in range(1, argv.get("N", 4)+1):
        count = 0
        gen = []
        arg = [code]*N
        print("N =", N)
        for M,L in find_transversal(*arg, constant=argv.constant):
            print(L)
            gen.append(L)
            count += 1
        print("gen:", len(gen))
        G = mulclose(gen)
        print("|G| =", len(G))
        print()

        make_gap("transversal_%d.gap"%(N,), gen)


def make_gap(name, gen):
    print("make_gap", name)
    f = open(name, "w")
    names = []
    for i,M in enumerate(gen):
        name = "M%d"%i
        print("%s := %s;"%(name,M.gap()), file=f)
        names.append(name)
    print("G := Group([%s]);"%(','.join(names)), file=f)
    print("Order(G);", file=f)
    f.close()


def main_autos_lc():
    if argv.code == (5,1,3):
        #code = construct.get_513()
        code = QCode.fromstr(
        "XZZXI IXZZX XIXZZ ZXIXZ", Ls="XXXXX ZZZZZ")
    elif argv.code == (13,1,5):
        n = 13
        check = "XXZZ.Z...Z.ZZ"
        checks = [''.join(check[(i+j)%n] for i in range(n)) for j in range(n-1)]
        code = QCode.fromstr(' '.join(checks))
    else:
        return

    print(code)
    print(code.longstr())
    assert code.is_gf4()

#    N, perms = code.get_autos()
#
#    for perm in perms:
#        P = code.space.get_perm(perm)
#        dode = P*code
#        assert code.is_equiv(dode)
#        #print(dode.get_logical(code))
#        #print()

    space = code.space
    H, S = space.H, space.S
    op = reduce(mul, [S(i) for i in range(code.n)])
    dode = op*code
    assert not dode.is_equiv(code)

    if code.n == 5:
        iso = code.get_isomorphism(dode)
        p = space.get_perm(iso)
        eode = p*dode
        assert eode.is_equiv(code)
        print(eode.get_logical(code))

    #return

    print("find_autos_lc:")

    gen = []
    for u in find_autos_lc(code):
        assert (u*code).is_equiv(code)
        gen.append(u)
        print('.', end='', flush=True)
        G = mulclose(gen, verbose=True) #, maxsize=10000)
        #if len(G) == 660:
        #    break
        #for g in G:
        #    assert (g*code).is_equiv(code)
        #print("yes")
    print()

    print(len(G))



def find_clifford_stab():
    #code = construct.get_513()
    n, k = 4, 2
    space = SymplecticSpace(n)
    I = space.get_identity()
    code = QCode.from_encoder(I, k=k)

    n = code.n
    m = code.m
    H = code.H
    Ht = H.t

    F = code.space.F

    gen = []

    solver = Solver()
    Add = solver.add

    U = UMatrix.unknown(2*n, 2*n)
    Add(U.t*F*U == F) # quadratic constraint
    #Add(U*Ht == Ht)

    V = UMatrix.unknown(m, m)
    Vi = UMatrix.unknown(m, m)
    I = Matrix.identity(m, m)
    Add(V*Vi==I)

    Add(U*Ht*V == Ht)
    
    while 1:
    #for _ in range(10):
        result = solver.check()
        if result != z3.sat:
            print("result:", result)
            break
    
        model = solver.model()
        u = U.get_interp(model)
        assert u.t*F*u == F
        gen.append(u)
        #print(u)
        #print()

        v = V.get_interp(model)
        #vi = Vi.get_interp(model)
        assert u*Ht*v == Ht

        Add(U != u)
        
        dode = u*code
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        #print(L)
        G = mulclose(gen, verbose=True)
        G = list(G)
        #shuffle(G)
        #for g in G[:1000]:
        #    Add(U!=g) # woah
        #if len(G) == 4608:
        #    break
        #if len(G) == 4128768:
        #    break
        #if len(G) == 8847360:
        #    break
        del G

    #make_gap("clifford_stab_42", gen)



def test_513():
    code = construct.get_513()

    #print(code.longstr())
    p = [1,2,3,4,0]
    lc_ops = set()
    for g in find_local_cliffords(code, code):
        if g.is_identity():
            continue
        dode = code.apply(g)
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        lc_ops.add(L)
        break

    space = code.space
    for p in allperms(list(range(5))):
        if p==(0,1,2,3,4):
            continue
        P = space.get_perm(p)
        dode = code.apply_perm(p)
        for g in find_local_cliffords(code, dode):
            break
        else:
            assert 0
        #print(g)
        eode = dode.apply(g)
        assert eode.is_equiv(code)
        L = eode.get_logical(code)
        if L in lc_ops:
            continue
        gp = g*P
        dode = code.apply(gp)
        assert dode.is_equiv(code)
        lc_ops.add(L)
        break
        
    G = mulclose(lc_ops)
    assert len(G)==6
    assert len(lc_ops) == 2

    N = 4
    gen = []
    I = SymplecticSpace(1).get_identity()
    for g in lc_ops:
        op = reduce(Matrix.direct_sum, [g]+[I]*(N-1))
        gen.append(op)

    count = 0
    arg = [code]*N
    src = reduce(add, arg)
    print(src)
    print("N =", N)
    for M,L in find_transversal(*arg, constant=argv.constant):
        gen.append(L)
        tgt = src.apply(M)
        assert tgt.is_equiv(src)
        count += 1
    print()
    print("gen:", len(gen))
    for g in gen:
        print(g, g.shape)
    G = mulclose(gen, verbose=True)
    print("|G| =", len(G))
    print()

    f = open("generate_513_%d.gap"%N, "w")
    names = []
    for i,M in enumerate(gen):
        name = "M%d"%i
        print("%s := %s;"%(name,M.gap()), file=f)
        names.append(name)
    print("G := Group([%s]);"%(','.join(names)), file=f)
    print("Order(G);", file=f)
    f.close()


def test_513_cover():
    code = construct.get_513()

    #print(code.longstr())
    p = [1,2,3,4,0]
    lc_ops = set()
    physical = set()
    for g in find_local_cliffords(code, code):
        if g.is_identity():
            continue
        dode = code.apply(g)
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        lc_ops.add(L)
        physical.add(g)
        break

    space = code.space
    for p in allperms(list(range(5))):
        if p==(0,1,2,3,4):
            continue
        P = space.get_perm(p)
        dode = code.apply_perm(p)
        for g in find_local_cliffords(code, dode):
            break
        else:
            assert 0
        #print(g)
        eode = dode.apply(g)
        assert eode.is_equiv(code)
        L = eode.get_logical(code)
        if L in lc_ops:
            continue
        gp = g*P
        dode = code.apply(gp)
        assert dode.is_equiv(code)
        physical.add(gp)
        lc_ops.add(L)
        break
        
    G = mulclose(lc_ops)
    assert len(G)==6
    assert len(lc_ops) == 2

    cover = Cover.frombase(code)
    total = cover.total

    gen = set()
    for g in physical:
        g = cover.lift(g)
        gen.add(total.get_logical(g*total))

    g = cover.get_ZX()
    gen.add(total.get_logical(g*total))
    g = cover.get_CZ()
    gen.add(total.get_logical(g*total))

    G = mulclose(gen)
    assert len(G) == 36


def test_833():
    # See:
    # https://arxiv.org/abs/quant-ph/9702029"
    code = construct.get_833()
    space = code.space
    I = space.get_identity()
    n = code.n
    N, perms = code.get_autos()
    dode = code.apply_perm(perms[0])
    assert dode.is_equiv(code)
    #print(dode.get_logical(code))

    Sn = Group.symmetric(n)
    G, gen = set(), []
    for g in Sn:
        perm = [g[i] for i in range(n)]
        P = space.get_perm(perm)
        #print(P, perm)
        dode = code.apply(P)
        if code.is_equiv(dode):
            h = I
        else:
            for h in find_local_cliffords(code, dode):
                break
            else:
                continue
        eode = dode.apply(h)
        assert eode.is_equiv(code)
        l = eode.get_logical(code)
        if l not in G:
            gen.append(l)
            G = mulclose(gen)
        #print(l, len(G), len(gen))
        #print()
    # takes about 30 minutes to get here
    assert len(G) == 168

        
def test_412_gottesman():
    # See:
    # https://arxiv.org/abs/quant-ph/9702029

    s = SymplecticSpace(4)
    CX = s.CX
    ops = [CX(int(i),int(j)) for (i,j) in "20 10 01 02 21 31 23 12".split()]
    g = reduce(mul, ops)
    print(g)

    for perm in allperms(range(4)):
        P = s.get_perm(perm)
        #print(P.t * g * P == g, perm)

    c = construct.get_412()
    code = c+c+c+c
    perm = [j*4 + i for i in range(4) for j in range(4)]
    code = code.apply_perm(perm)

    g4 = reduce(Matrix.direct_sum, [g]*4)
    dode = code.apply(g4)
    assert code.is_equiv(dode)
    assert code.get_logical(dode) == g # nice

    gen = [CX(i,j) for i in range(4) for j in range(4) if i!=j]
    h = mulclose_find(gen, g)
    assert len(h.name) == 6
    assert h==g

    name = ('CX(0,1)', 'CX(3,0)', 'CX(0,2)', 'CX(1,3)', 'CX(2,1)', 'CX(1,0)')

    dode = code
    for name in reversed(name):
        g1 = s.get_expr(name)
        g4 = reduce(Matrix.direct_sum, [g1]*4)
        dode = dode.apply(g4)
        assert dode.distance() >= 2
    assert dode.is_equiv(code)


def test_513_gottesman():
    # See Fig. 1.
    # https://arxiv.org/abs/quant-ph/9702029

    s = SymplecticSpace(3)
    S, H, CX, CZ = s.S, s.H, s.CX, s.CZ
    SHS = lambda i :S(i)*H(i)*S(i)
    HS = lambda i :H(i)*S(i)
    SH = lambda i :S(i)*H(i)

    # T_3 gate
    g = CX(2,0)*CX(2,1)*CX(1,2)*CX(0,2)*SHS(0)*SHS(1)*CZ(0,1)*CX(0,1)*CX(1,0)*CX(0,1)

    c = construct.get_513()
    code = c+c+c

    perm = [j*5 + i for i in range(5) for j in range(3)]
    #print(perm)
    code = code.apply_perm(perm)

    g5 = reduce(Matrix.direct_sum, [g]*5)
    dode = code.apply(g5)
    assert code.is_equiv(dode)
    assert code.get_logical(dode) == g # nice

    gen  = [S(0),S(1),S(2)]
    gen += [H(0),H(1),H(2)]
    #gen += [SH(0),SH(1),SH(2)]
    #gen += [HS(0),HS(1),HS(2)]
    #gen += [SHS(0),SHS(1),SHS(2)]
    gen += [CX(i,j) for i in range(3) for j in range(3) if i!=j]
    gen += [CZ(i,j) for i in range(3) for j in range(i+1,3)]
    h = mulclose_find(gen, g)
    print(h)
    print(h.name)

    name = ('CX(0,1)', 'H(0)', 'S(0)', 'CZ(0,2)', 'CX(1,0)', 'CX(1,2)', 'H(1)', 'CX(1,0)')
    dode = code
    for name in reversed(name):
        g1 = s.get_expr(name)
        g5 = reduce(Matrix.direct_sum, [g1]*5)
        dode = dode.apply(g5)
        assert dode.distance() == 3
    assert dode.is_equiv(code)




def get_412_transversal():

    N = argv.get("N", 4)

    # N*[[4,1,2]] cliffords
    code = construct.get_412()
    arg = tuple([code]*N)
    logical = []
    #physical = []
    for M,L in find_transversal(*arg, constant=argv.constant):
        print(L)
        logical.append(L)
        #physical.append(M)
    print("logical:", len(logical))
    G = mulclose(logical)
    print("|G| =", len(G))

    # single [[4,1,2]] cliffords
    single = [] 
    n = code.n
    space = code.space
    src = code
    G = Group.symmetric(src.n)
    gen = set()
    for g in G:
        perm = [g[i] for i in range(n)]
        P = space.get_perm(perm)
        tgt = src.apply_perm(perm)
        for M in find_local_cliffords(src, tgt):
            code = tgt.apply(M)
            assert code.is_equiv(src)
            dode = src.apply(M*P)
            assert code.is_equiv(dode)
            L = code.get_logical(src)
            if L not in gen:
                #print(P)
                print(M, g)
                print(L)
                gen.add(L)
                #single.append(M*P)
                single.append(L)
    G = mulclose(gen)
    print(len(G))

    I = SymplecticSpace(1).get_identity()
    for M in single:
        for i in range(N):
            op = [I]*N
            op[i] = M
            op = reduce(Matrix.direct_sum, op)
            #physical.append(op)
            print(op)
            logical.append(op)

    if 0:
        G = mulclose(logical, verbose=True)
        print(len(G))
        return


    f = open("generate_%d.gap"%N, "w")
    names = []
    for i,M in enumerate(logical):
        name = "M%d"%i
        print("%s := %s;"%(name,M.gap()), file=f)
        names.append(name)
    print("G := Group([%s]);"%(','.join(names)), file=f)
    print("Order(G);", file=f)
    f.close()

    #G = mulclose(physical, verbose=True)
    #print("|G| =", len(G))



#def get_codes(n, k, d):
#    from bruhat.sp_pascal import i_grassmannian
#    perm = []
#    for i in range(n):
#        perm.append(i)
#        perm.append(2*n - i - 1)
#    found = []
#    for _,H in i_grassmannian(n, n-k):
#        H = H[:, perm]
#        H = Matrix(H)
#        code = QCode(H, check=False)
#        if code.get_distance() < d:
#            #print("x", end='', flush=True)
#            continue
#        found.append(code)
#    return found


def all_codes():
    from bruhat.sp_pascal import i_grassmannian

    #n, k, d = 4, 1, 2
    #n, k, d = 5, 1, 3
    n, k, d = argv.get("code", (4,1,2))
    constant = argv.get("constant", True)

    space = SymplecticSpace(n)
    gen = []
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(2*n - i - 1)
        gen.append(space.get_S(i))
        gen.append(space.get_H(i))
    found = []

    if 0:
        _Cliff = mulclose(gen)
        print("|_Cliff| =", len(_Cliff))
        code = construct.get_513()
        for g in _Cliff:
            dode = code.apply(g)
            found.append(dode)
        print(len(found))
        hom = equ.quotient_rep(found, QCode.is_equiv)
        found = list(set(hom.values()))
        print(len(found))
        return

    F = space.F
    count = 0
    #found = []
    for _,H in i_grassmannian(n, n-k):
        H = H[:, perm] # reshuffle to qumba symplectic
        H = Matrix(H)
        count += 1
        code = QCode(H, check=False)
        if code.get_distance() < d:
            #print("x", end='', flush=True)
            continue

        items = list(find_transversal(code, constant=constant, verbose=False))
        gen = [item[1] for item in items]
        G = mulclose(gen)
        if len(G) > 3:
            print()
            print(code.H)
            print("|G| =", len(G))

        elif len(G) == 3:
            print("[3]", end='', flush=True)
            found.append(code)

        elif len(G) == 2:
            print("[2]", end='', flush=True)
            found.append(code)

        #else:
        #    print(".", end='', flush=True)

        #if len(found) > 6:
        #    break

    print()
    print("count", count)
    print("found", len(found))
    #print(len([code for code in found if not code.is_css()]))

    hom = equ.quotient_rep(found, autos.is_iso)
    found = list(set(hom.values()))
    print("hom:", len(found))

    return found


def test_local_clifford():
    print("test_local_clifford")
    if argv.code == (4,2,2):
        code = QCode.fromstr("XXXX ZZZZ")
    elif argv.code == (5,1,3):
        code = construct.get_513()
    elif argv.code == (4,1,2):
        code = QCode.fromstr("XYZI IXYZ ZIXY")
    elif argv.code == (6,2,2):
        code = QCode.fromstr("XXXIXI ZZIZIZ IYZXII IIYYYY")
    elif argv.code == (8,2,3):
        code = QCode.fromstr("""
            X...YZZZ
            .X.ZYX.X
            .ZX.YYZX
            .ZZYZXZZ
            ..Z...YX
            ZZZZZZZZ
        """)
    elif argv.code == (8,3,2):
        code = QCode.fromstr("""
            XXXXIIII
            ZIZIZIZI
            IIYYYYII
            IZIZIZIZ
            IIIIXXXX
        """)

    print(code.get_params())

    n = code.n
    src = code
    G = Group.symmetric(src.n)
    count = 0
    gen = set()
    for g in G:
        #if g.is_identity():
        #    continue
        perm = [g[i] for i in range(n)]
        tgt = src.apply_perm(perm)
        for M in find_local_cliffords(src, tgt):
            code = tgt.apply(M)
            assert code.is_equiv(src)
            L = code.get_logical(src)
            if L not in gen:
                print(M, g)
                print(L)
                gen.add(L)
            count += 1
    print(count)
    G = mulclose(gen)
    print("|G| =", len(G))


def find_clifford(code, pairs, constant=False, verbose=True):
    solver = Solver()
    Add = solver.add

    m = 2
    Fm = SymplecticSpace(m).F
    n = code.n // m
    space = code.space

    U0 = UMatrix.unknown(2*m, 2*m)
    Add(U0.t*Fm*U0 == Fm) # quadratic constraint
    U = reduce(UMatrix.direct_sum, [U0]*n)
    perm = reduce(add, pairs)
    P = space.get_perm(perm)
    U = P.t*U*P

    HU = code.H * U.t
    LU = code.L * U.t
    F = code.space.F
    R = HU * F * code.L.t
    Add(R==0) # linear constraint
    R = HU * F * code.H.t
    Add(R==0) # linear constraint

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = code.apply(M)
        assert dode.is_equiv(code)
        yield M

        Add(U != M)



def main_unwrap():

    for idx in range(31):
        code = construct.get_513(idx)
        n = code.n
    
        dode = unwrap(code)
        pairs = [(i, i+n) for i in range(n)]
        perm = [i+n for i in range(n)] + list(range(n))
    
        eode = dode.apply_perm(perm)
        eode = eode.apply_H()
        assert eode.is_equiv(dode)
    
        eode = dode
        for (i,j) in pairs:
            eode = eode.apply_CZ(i, j)
        assert eode.is_equiv(dode)
    
        count = 0
        gen = []
        for M in find_cliffords(dode, pairs):
            count += 1
            #print(M)
            eode = dode.apply(M)
            assert eode.is_equiv(dode)
            L = eode.get_logical(dode)
            #print(L)
            gen.append(L)
        print(count, end=" ")
        G = mulclose(gen)
        assert count == len(G)

        if not argv.autos:
            print()
            continue

        A = autos.get_autos(dode)
        #print(len(A), end=" ")

        for perm in A:
            #P = dode.space.get_perm(perm)
            eode = dode.apply_perm(perm)
            assert eode.is_equiv(dode)
            L = eode.get_logical(dode)
            #print(L)
            gen.append(L)
        #    print(perm)
        #continue
        G = mulclose(gen)
        print(len(G))


def test_412():
    space = SymplecticSpace(1)
    I = space.get_identity()
    H = space.get_H()
    S = space.get_S()
    names = {
        I : "I",
        S : "S",
        H : "H",
        S*H : r"(S\cdot H)",
        H*S : r"(H\cdot S)",
        H*S*H : r"(H\cdot S\cdot H)",
    }

    code = QCode.fromstr("XYZI IXYZ ZIXY")
    space = code.space
    n = code.n
    G = Group.symmetric(n)
    perms = [tuple(g[i] for i in range(n)) for g in G]
    perms.sort()
    for perm in perms:
        print(tuple(i+1 for i in perm), "&", end=" ")
        dode = code.apply_perm(perm)
        if dode.is_equiv(code):
            eode = dode
            U = space.get_identity()
        else:
            U = None
            found = list(find_local_cliffords(code, dode))
            assert len(found) == 1
            U = found[0]
            eode = dode.apply(U)
        assert eode.is_equiv(code)
        name = [names[U[i:i+2, i:i+2]] for i in range(0,space.nn,2)]
        name = r"".join(name)
        print(name, "&", end=" ")
        L = eode.get_logical(code)
        print(names[L], end=" ")
        print(r"\\")

    E = code.get_encoder()
    print(E)
    space = code.space
    print(space.get_name(E))


def find_perm_gate(code, perm):
    print("find_perm_gate", perm)
    n = code.n
    from qumba.clifford import Clifford, mulclose_names
    dode = code.apply_perm(perm)
    #print(dode.longstr())

    eode = dode
    space = code.space
    G = space.local_clifford_group()
    assert len(G) == 6**n
    found = None
    for g in G:
        eode = dode.apply(g)
        if eode.is_equiv(code):
            assert found is None
            found = g
    #else:
    #    assert 0
    lc_name = found.name
    del g
    #print("local clifford:", lc_name)

    circuit = lc_name + space.get_P(*perm).name
    eode = code.apply(code.space.get_expr(circuit))
    assert eode.is_equiv(code)

    H = code.H
    n = code.n
    c = Clifford(n)
    P = code.get_projector()
    assert P*P == P
    
    p = c.get_P(*perm)
    gate = c.get_expr(lc_name, rev=True)
    gate = p*gate
    if gate*P == P*gate:
        print("found gate")
        return circuit

    assert gate == c.get_expr(circuit, rev=True)
    
    names = c.pauli_group(False)
    found = None
    for g,name in names.items():
        #print(g, name)
        op = gate*g
        if(op*P == P*op):
            found = name
            break
    assert found
    print("pauli correction", found)
        
    circuit += found
    op = c.get_expr(circuit, rev=True)
    assert op*P == P*op
    return circuit


def test_412_clifford():
    from qumba.clifford import Clifford, green, red, half
    from huygens.zx import Circuit, Canvas
    code = QCode.fromstr("XYZI IXYZ ZIXY")
#    E = code.get_encoder()
#    D = code.get_decoder()
#    cvs = code.space.render(E)
#    cvs.writePDFfile("encoder.pdf")
#    assert E*D == code.space.get_identity()
#    assert D*E == code.space.get_identity()
#
#    E = code.space.translate_clifford(E)
#    D = code.space.translate_clifford(D)
#    DE = D*E
#    ED = E*D
#    c = Clifford(code.n)
##    for g in c.pauli_group(1):
##        if g*D == E.d:
##            break
##    else:
##        assert 0
##    print(g.name)
#
#    D = E.d
#    assert E*E.d == c.get_identity()
#
#    P = code.get_projector()
#    lhs = P
#    print(lhs)
#    print(lhs.shape, lhs.M.rank())
#
#    #rr = red(1,0)*red(0,1)
#    ##rr = green(1,0)*green(0,1)
#    #rhs = rr@rr@rr@Clifford(1).get_identity()
#    #print(rhs)
#    #print(rhs.shape, rhs.rank())
#    #print(P)
#    return

    clifford = Clifford(code.n)
    I = Clifford(1).get_identity()
    E = code.get_clifford_encoder()
    P = code.get_projector()
    D = E.d
    lhs = D * P * E
    rhs = [red(1,0)*red(0,1) for i in range(code.n)]
    rhs[-1] = I
    rhs = reduce(matmul, rhs)
    assert rhs == (2**code.m)*lhs

    # disc, prep
    prep = reduce(matmul, [red(1,0)]*code.m + [I]*code.k)
    disc = reduce(matmul, [red(0,1)]*code.m + [I]*code.k)

    S4 = Group.symmetric(code.n)
    perms = [[g[i] for i in range(code.n)] for g in S4]
    perms.sort()

    found = []
    for perm in perms:
    #for g in S4:
    #    perm = [g[i] for i in range(code.n)]
    
        print()
        circuit = find_perm_gate(code, perm)
        g = Clifford(code.n).get_expr(circuit, rev=True)
        assert g*P == P*g

        g = disc * D * g * E * prep
        g = (half**code.m)*g
        #print(g, g.rank())
        assert g.rank() == 2

        c = Clifford(1)
        gen = [c.get_identity(), c.X(), c.Z(), c.Y(), c.H(0), c.S(0), c.wI()**2]
        g = mulclose_find(gen, g)
        assert g is not None
        print("logical:", g.name)
        
        c = Circuit(code.n)
        cvs = c.render_expr(circuit, width=6, height=4)
        name = "circuit_%s.pdf"%(''.join(str(i) for i in perm))
        print(name)
        cvs.writePDFfile(name)
        found.append((cvs, perm))
        #if len(found)>1:
        #    break

    print("found:", len(found))
    cvs = Canvas()
    y = 0.
    for c, perm in found:
        cvs.insert(0, y, c)
        x = c.get_bound_box().width
        cvs.text(x, y, "%s"%(perm))
        y -= 1.5*c.get_bound_box().height
    cvs.writePDFfile("circuit.pdf")


def test_822_state_prep():
    # FAIL FAIL FAIL
    row_weight = argv.get("row_weight", 3)
    diagonal = argv.get("diagonal", False)

    code = QCode.fromstr("""
    XX...XX.
    .XX...XX
    ..XXX..X
    .ZZ.ZZ..
    ..ZZ.ZZ.
    Z..Z..ZZ
    """, Ls="""
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """)

    print(code)

    n = code.n
    nn = 2*n
    space = code.space
    F = space.F

    solver = Solver()
    Add = solver.add
    U = UMatrix.unknown(nn, nn)
    if diagonal:
        for i in range(nn):
            U[i,i] = Const(1)

    Add(U.t*F*U == F) # U symplectic

    #for perm in perms:
    #    P = space.get_perm(perm)
    #    U1 = P.t*U*P
    #    Add(U==U1)

    U0 = U[:, :2]

    R = code.H * F * U0
    Add(R==0) # linear constraint

    if row_weight is not None:
        for i in range(nn):
            Add(Sum([If(U[i,j].get(),1,0) for j in range(nn)])<=row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        print(M)
        return

        Add(U != M)



def test_hexagons():
    "how do dehn twists act on hexagons?"
    from qumba.lin import (
        parse, rank, array2, zeros2, 
        shortstr, linear_independent, intersect,
        dot2,
    )
    lookup = {}
    rows = cols = 6
    n = rows*cols
    for i in range(rows):
      for j in range(cols):
        lookup[i,j] = len(lookup)
    n = len(lookup)
    for i in range(rows):
      for j in range(cols):
        for di in (-rows, 0, rows):
          for dj in (-cols, 0, cols):
            lookup[i+di,j+dj] = lookup[i,j]

    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    #print(perm)
    P = zeros2((n, n))
    for (i,j) in enumerate(perm):
        P[i,j] = 1

    # black squares
    print("H =")
    H = []
    for i in range(rows):
      for j in range(cols):
        if (i+j)%2:
            continue
        op = [0]*n
        op[lookup[i,j]] = 1
        op[lookup[i+1,j]] = 1
        op[lookup[i,j+1]] = 1
        op[lookup[i+1,j+1]] = 1
        H.append(op)
    H = array2(H)
    print(shortstr(H), H.shape, rank(H))
    H = linear_independent(H)
    HP = dot2(H,P)
    print("H^HP =", rank(intersect(H, HP)))
    print()

    # right hexagons
    K = []
    for i in range(rows):
      for j in range(cols):
        if (i+j)%2:
            continue
        op = [0]*n
        op[lookup[i,j]] = 1
        op[lookup[i+1,j]] = 1
        op[lookup[i,j+1]] = 1
        op[lookup[i+1,j+2]] = 1
        op[lookup[i+2,j+1]] = 1
        op[lookup[i+2,j+2]] = 1
        K.append(op)
    K = array2(K)
    print("K =")
    print(shortstr(K), K.shape, rank(K))
    K = linear_independent(K)
    HK = intersect(H, K)
    print("H^K =", rank(HK))
    print("H^KP =", rank(intersect(H,dot2(K,P))))
    print("HP^KP =", rank(intersect(HP,dot2(K,P))))

    print()

    m = len(H)
    J = []
    for i in range(m):
      for j in range(m):
        op = (H[i] + H[j])%2
        if op.sum() == 6:
            J.append(op)
    J = array2(J)
    J = linear_independent(J)
    print("J =")
    print(shortstr(J), J.shape)
    JK = intersect(J, K)
    print("J^K =", rank(JK))
    print()


def search_gate(code, dode, *perms, row_weight=None, diagonal=False):
    solver = Solver()
    Add = solver.add

    n = code.n
    nn = 2*n
    space = code.space
    F = space.F

    U = UMatrix.unknown(nn, nn)
    if diagonal:
        for i in range(nn):
            U[i,i] = Const(1)

    Add(U.t*F*U == F) # U symplectic

    for perm in perms:
        P = space.get_perm(perm)
        U1 = P.t*U*P
        Add(U==U1)

    HU = code.H * U.t
    LU = code.L * U.t
    R = HU * F * dode.L.t
    Add(R==0) # linear constraint
    R = HU * F * dode.H.t
    Add(R==0) # linear constraint

    if row_weight is not None:
        for i in range(nn):
            Add(Sum([If(U[i,j].get(),1,0) for j in range(nn)])<=row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        eode = code.apply(M)
        assert dode.is_equiv(eode)
        yield M

        Add(U != M)


def old_test_dehn():
    lookup = {}
    rows, cols = 4, 4
    for i in range(rows):
      for j in range(cols):
        lookup[i,j] = len(lookup)
    n = len(lookup)
    for i in range(rows):
      for j in range(cols):
        for di in (-rows, 0, rows):
          for dj in (-cols, 0, cols):
            lookup[i+di,j+dj] = lookup[i,j]

    facess = [[], []]
    parity = 0
    for i in range(rows):
      for j in range(cols):
        face = [lookup[i,j] 
            for (i,j) in [(i,j), (i+1,j), (i+1,j+1), (i,j+1)]]
        facess[parity].append(face)
        parity = (parity+1)%2
      parity = (parity+1)%2
    #print(facess, n)
    stabs = []
    for (faces, op) in zip(facess, list('XZ')):
      for face in faces:
        stab = ['I']*n
        for idx in face:
            stab[idx] = op
        stab = fromstr(''.join(stab))
        #print(stab)
        stabs.append(stab)
    #stabs = numpy.array(stabs)
    stabs = numpy.concatenate(stabs)
    code = QCode(A=stabs)
    print(code)
    print(strop(code.H))

    return

    #for zx in code.find_zx_dualities():
    #    print(zx)
    #return

    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    dode = code.apply_perm(perm)
    #print(dode.is_equiv(code))
    #for M in find_local_cliffords(dode, code):
    #    print(M)
    print(dode)
    print(strop(dode.H))
    print(code.is_equiv(dode))


    hperm = {}
    vperm = {}
    for i in range(rows):
      for j in range(cols):
        hperm[lookup[i,j]] = lookup[i+2,j]
        vperm[lookup[i,j]] = lookup[i,j+2]
    hperm = [hperm[i] for i in range(n)]
    vperm = [vperm[i] for i in range(n)]
    print(hperm)
    print(vperm)
    
    assert code.apply_perm(vperm).is_equiv(code)
    assert code.apply_perm(hperm).is_equiv(code)
    assert dode.apply_perm(vperm).is_equiv(dode)
    assert dode.apply_perm(hperm).is_equiv(dode)

    if 0:
        #return
    
        H = dode.H.intersect(code.H)
        print(code.H.shape, "-->", H.shape)

        eode = QCode(H)
        #from distance import distance_z3
        #distance_z3(eode)
        #print(eode)
        #print(eode.longstr())
        #return
    
        M = dode.get_encoder() * code.get_decoder()
        #print(shortstr(M.A))
    
        op = dode.get_logical(code)
        s = SymplecticSpace(2)
        gen = [s.get_CNOT(1,0), s.get_CZ(), 
            s.get_S(0), s.get_H(0), s.get_S(1), s.get_H(1)]
        G = mulclose(gen)
        G = list(G)
        print(G[0].name)
        #print(op in mulclose(gen))
        idx = G.index(op)
        print(G[idx].name)
        #print(SymplecticSpace(2).get_name(op))
    
        return M

    row_weight = argv.row_weight
    diagonal = argv.diagonal
    for M in search_gate(code, dode, hperm, vperm, row_weight=row_weight, diagonal=diagonal):
        print(M)
        print(code.space.get_name(M))
        #break


def test_16_2_4():
    "dehn twists on [[16,2,4]]"
    l = 4
    n = l**2

    lookup = {}
    for row in range(l):
      for col in range(l):
        lookup[row, col] = len(lookup)
    #print(lookup)

    stabs = []
    for row in range(l):
      for col in range(l):
        bit = "X" if (row+col)%2==0 else "Z"
        stab = ["I"]*n
        #print(row, col, bit)
        stab[lookup[row,col]] = bit
        stab[lookup[row,(col+1)%l]] = bit
        stab[lookup[(row+1)%l,(col+1)%l]] = bit
        stab[lookup[(row+1)%l,col]] = bit
        stab = "".join(stab)
        stabs.append(stab)
    stabs = stabs[:-2]
    H = '\n'.join(stabs)
    space = SymplecticSpace(n)
    H = space.fromstr(H)
    assert space.is_isotropic(H)

    code = QCode(H)
    print(code)

    if 0:
        G = space.get_identity()
        for row in range(l):
          for col in range(l):
            if (row+col)%2==0:
                i = lookup[row, col]
                j = lookup[(row+1)%l, col]
                op = space.CX(j,i)
                #print(i,j)
                assert op*G == G*op
                G = op*G
        
        dode = G*code
        print(dode.is_equiv(code))

    perm = {}
    for row in range(l):
      for col in range(l):
        src = lookup[row, col]
        tgt = lookup[(row + col)%l, col]
        perm[src] = tgt
    perm = [perm[i] for i in range(n)]
    G = space.get_perm(perm)
    dode = G*code
    print(dode.is_equiv(code))

    print(code.H.shape)

    H0 = code.H.intersect(dode.H)
    print(H0, H0.shape)

    #eode = QCode(H0)
    #print(eode)
    #print(eode.longstr())

    from qumba.css import CSS

    src = dode.to_css()
    tgt = code.to_css()

    src = CSS(src.Hx, src.Hz)
    tgt = CSS(tgt.Hx, tgt.Hz)
    print(src)
    print(tgt)

    mx = src.mx
    mz = src.mz

    solver = Solver()
    Add = solver.add

    U = UMatrix.unknown(n, n)
    V = UMatrix.unknown(n, n)
    I = Matrix.identity(n)

    Add( U*V == I )

    Mx = UMatrix.unknown(mx, mx)
    Mz = UMatrix.unknown(mz, mz)

    Add( src.Hx * U.t == Mx * tgt.Hx )
    Add( src.Hz * V == Mz * tgt.Hz )

    if argv.diagonal:
        for i in range(n):
            Add( U[i,i] == 1 )

    # locality ?
    coords = [(row,col) for row in range(l) for col in range(l)]
#    for u in coords:
#      for v in coords:
#        nbd = [((u[0]+i)%l, (u[1]+j)%l) for i in [-1,0,1] for j in [-1,0,1]]
#        i = lookup[u]

    def get_shift(dx, dy):
        perm = []
        for u in coords:
            v = (u[0]+dx)%l, (u[1]+dy)%l
            perm.append(lookup[v])
        P = Matrix.get_perm(perm)
        return P

    P = get_shift(2,0)
    Add( P * U * P.t == U )
    P = get_shift(0,2)
    Add( P * U * P.t == U )

    row_weight = argv.get("row_weight")
    if row_weight is not None:
        for i in range(n):
            Add(Sum([If(U[i,j].get(),1,0) for j in range(n)])<=row_weight)

    count = 0
    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            print(result)
            return
    
        model = solver.model()
        U0 = U.get_interp(model)
        V0 = V.get_interp(model)
        Add(U != U0)
    
        weight = U0.sum()
        if weight-n > 8:
            continue

        print()
        print(U0, U0.shape, U0.sum()-n)

        count += 1
        #break
        assert U0*V0 == I
    
        E = U0 << V0.t
        # block order --> ziporder
        perm = []
        for i in range(n):
            perm.append(i)
            perm.append(i+n)
        E = E[perm, :]
        E = E[:, perm]
    
        space = SymplecticSpace(n)
        assert space.is_symplectic(E)
    
        #assert (E*code).is_equiv(dode)
        assert (E*dode).is_equiv(code)
    
    print("found:", count)


def test_dehn():
    "dehn twists on toric codes"
    l = 4

    css = construct.toric(l, l)
    css.bz_distance()
    print(css)

    n = css.n
    lookup = css.lookup

    code = css.to_qcode()

    space = code.space
    G = space.get_identity()
    for r in range(l):
      for c in range(l):
        idx = lookup[r,c,0]
        jdx = lookup[r,c,1]
        op = space.CX(idx, jdx)
        print((idx,jdx), end=" ")
        assert op*G == G*op
        G = op*G
    print()

    dode = G*code
    #dode = code
    print(dode.is_equiv(code))

    other = dode.to_css()
    other.bz_distance()
    print(other)

    #return

    from qumba.css import CSS

    src = dode.to_css()
    tgt = code.to_css()

    src = CSS(src.Hx, src.Hz)
    tgt = CSS(tgt.Hx, tgt.Hz)
    print(src)
    print(tgt)

    mx = src.mx
    mz = src.mz

    solver = Solver()
    Add = solver.add

    U = UMatrix.unknown(n, n)
    #V = UMatrix.unknown(n, n)
    V = U.t
    I = Matrix.identity(n)

    Add( U*V == I )

    Mx = UMatrix.unknown(mx, mx)
    Mz = UMatrix.unknown(mz, mz)

    Add( src.Hx * U.t == Mx * tgt.Hx )
    Add( src.Hz * V == Mz * tgt.Hz )

    row_weight = argv.get("row_weight", 1)
    if row_weight is not None:
        print("row_weight:", row_weight)
        for i in range(n):
            Add(Sum([If(U[i,j].get(),1,0) for j in range(n)])<=row_weight)

    gen = set()

    count = 0
    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            print(result)
            break
    
        model = solver.model()
        U0 = U.get_interp(model)
        V0 = V.get_interp(model)
        Add(U != U0)
    
        count += 1
        assert U0*V0 == I
    
        E = U0 << V0.t
        # block order --> ziporder
        perm = []
        for i in range(n):
            perm.append(i)
            perm.append(i+n)
        E = E[perm, :]
        E = E[:, perm]
    
        space = SymplecticSpace(n)
        assert space.is_symplectic(E)
    
        eode = E*dode
        assert eode.is_equiv(code)
        l = eode.get_logical(code)
        if l not in gen:
            gen.add(l)
            print(l)
    
    print("found:", count)
    print("gen:", len(gen))
    G = mulclose(gen)
    print("logicals:", len(G))


def test_all_412():
    "generate all 412 codes & _look for local clifford & perm gates"
    n, k, d = 4, 1, 2
    G = Group.symmetric(n)
    perms = [[g[i] for i in range(n)] for g in G]
    count = 0
    for code in construct.all_codes(n, k, d):
        #print(code)
        found = 0
        for perm in perms:
            dode = code.apply_perm(perm)
            if dode.is_equiv(code):
                found += 1
            elif is_local_clifford_equiv(code, dode):
                found += 1
        print("%2d"%found, end=' ', flush=True)
        if count%32==0:
            print()
        count += 1
    print(count)

    
def find_all_perm_lc_412():
    n, k, d = 4, 1, 2
    G = Group.symmetric(n)
    perms = [[g[i] for i in range(n)] for g in G]
    codes = []
    for code in construct.all_codes(n, k, d):
        codes.append(code)
        print('.',end='',flush=True)
        #if len(codes)>30:
        #    break
    print()
    print(len(codes))
    def lc_perm(code, dode):
        for perm in perms:
            eode = dode.apply_perm(perm)
            if eode.is_equiv(code):
                return True
            elif is_local_clifford_equiv(code, eode):
                return True
    hom = equ.quotient_rep(codes, lc_perm)
    found = list(set(hom.values()))
    print("equs:", len(found))


def sp_find(items):
    n = 2
    space = SymplecticSpace(n)

    nn = 2*n
    F = space.F

    solver = Solver()
    Add = solver.add
    U = UMatrix.unknown(nn, nn)
    Add(U.t*F*U == F) # U symplectic

    parse = space.parse
    pairs = []
    for item in items:
        src, tgt = item.split()
        src = parse(src).t
        tgt = parse(tgt).t
        #print(src, tgt)
        pairs.append((src, tgt))

        Add(U * src == tgt)
        
    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("unsat")
            return

        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        yield M
        #print(M)
        #print()
        for src, tgt in pairs:
            assert (M*src == tgt)
        Add(U != M)


def test_braid():
    n = 2
    space = SymplecticSpace(n)

    CX, S, H = space.CX, space.S, space.H
    gen = [CX(0,1), CX(1,0), S(0), S(1), H(0), H(1)]
    G = mulclose(gen)
    G = list(G)
    print(len(G))

    found = set()
    #rhs = ["YI YZ", "IY ZY", "ZX IX", "XZ XI"]
    rhs = ["YI YZ", "IY ZY", "XZ IX", "ZX XI"]
    for R in sp_find(rhs):
        lhs = "XI XZ,IX ZX,ZY YI,YZ IY".split(',')
        for L in sp_find(lhs):
            U = L*R
            found.add(U)

    for g in found:
        print(g)
        g = G[G.index(g)]
        print(g.name)

    
def find_qcode():
    # generalize find_css to non-css codes

    params = argv.code
    if params is not None:
        n, k, d = params
        m = n-k
    else:
        n = argv.get("n", 5)
        k = argv.get("k", 2)
        d = argv.get("d", 2) 
        m = n-k

    print("code: [[%d, %d, %d]]"%(n, k, d))

    space = SymplecticSpace(n)
    F = space.F

    solver = Solver()
    Add = solver.add

    nn = 2*n

    # symplectic/unitary encoder map
    E = UMatrix.unknown(nn, nn) 

    H = E[:2*m:2, :]
    L = E[2*m:, :]

    # just check single qubit errors
    for i in range(n):
        for e in [(1,0), (0,1), (1,1)]:
            err = lin.zeros2(nn,1)
            err[2*i:2*i+2, 0] = e
            check = H*Matrix(err)
            Add(Or(*[check[j,0].get() for j in range(m)]))

    assert d==2, "not implemented" # XX TODO

    Add( E.t * F * E == F )

    count = 0
    while 1:
        result = solver.check()
        if str(result) != "sat":
            #print(result)
            break
    
        model = solver.model()
        _E = E.get_interp(model)
        _H = H.get_interp(model)
    
        code = QCode.from_encoder(_E.t, k=k)
        print(code)
        print(code.longstr())
        assert code.get_encoder() == _E.t

        Add( H != _H )

        # this needs to be ForAll U: U*H != _H
        #U = UMatrix.unknown(m, m)
        #Add( U*H != _H )

        count += 1
        #break # <-----------

        #print(".", end="", flush=True)
    #print()

    print("found:", count)


def find_code_autos():
    # modify find_qcode above to allow for certain automorphisms...

    params = argv.code
    if params is not None:
        n, k, d = params
        m = n-k
    else:
        n = argv.get("n", 5)
        k = argv.get("k", 2)
        d = argv.get("d", 2) 
        m = n-k

    l = argv.get("l", 1)
    assert n%l == 0, "blocks must divide n"

    print("code: [[%d, %d, %d]]"%(n, k, d))

    space = SymplecticSpace(n)
    F = space.F

    solver = Solver()
    Add = solver.add

    nn = 2*n

    #H = UMatrix.unknown(m, nn)
    #T = UMatrix.unknown(m, nn)
    #Lx = UMatrix.unknown(k, nn)
    #Lz = UMatrix.unknown(k, nn)

    # symplectic/unitary encoder map
    E = UMatrix.unknown(nn, nn) 

    H = E[:2*m:2, :]
    L = E[2*m:, :]

    assert d==2, "not implemented"

    # just check single qubit errors
    for i in range(n):
        for e in [(1,0), (0,1), (1,1)]:
            err = lin.zeros2(nn,1)
            err[2*i:2*i+2, 0] = e
            check = H*Matrix(err)
            Add(Or(*[check[j,0].get() for j in range(m)]))

    perms = []
    idxs = list(range(n//l))
    idxs[:2] = [1,0] # swap
    perms.append(idxs)
    perms.append([(i+1)%(n//l) for i in range(n//l)])

    #perm = space.SWAP(0,1)
    #perms.append(perm)
    #perm = space.P(*[(i+1)%n for i in range(n)])
    #perms.append(perm)

    ops = []

    for perm in perms:
        perm = reduce(add,
            [[i + block*(n//l) for i in perm] for block in range(l)])
        print("perm:", perm)
        assert len(perm)==n
        assert len(perm)==len(set(perm))
        op = space.P(*perm).t
        ops.append(op)
        H1 = H * op
        if 1:
            U = UMatrix.unknown(m, m)
            Add( U*H1 == H )
        else:
            # on small codes (n=4,5) this is much slower, 
            # haven't tried larger codes:
            Add( H*F*H1.t == 0 )
            Add( L*F*H1.t == 0 )

    Add( E.t * F * E == F )

    count = 0
    while 1:
        result = solver.check()
        if str(result) != "sat":
            #print(result)
            break
    
        model = solver.model()
        _E = E.get_interp(model)
        #print("E:")
        #print(_E)
    
        _H = H.get_interp(model)
        #print("H:")
        #print(_H)
    
        code = QCode.from_encoder(_E.t, k=k)
        print(code)
        print(code.longstr())
        assert code.get_encoder() == _E.t

        gen = set()
        for op in ops:
            dode = op*code
            assert dode.is_equiv(code)
            logop = dode.get_logical(code)
            gen.add(logop)
        gen = list(gen)
        G = mulclose(gen, verbose=True)
        print("logical action:", len(G))

        #Add( E != _E )
        Add( H != _H )

        # this needs to be ForAll U: U*H != _H
        #U = UMatrix.unknown(m, m)
        #Add( U*H != _H )

        count += 1
        #break # <-----------

        #print(".", end="", flush=True)
    #print()

    print("found:", count)


def gethom_Sp2(idx=0):

    # construct an explicit isomorphism: Sp(4,2) = S_6
    # (not shown: gap torture session)

    from qumba.symplectic import uturn_to_zip
    U = uturn_to_zip(2)

    space = SymplecticSpace(2)
    a = Matrix.parse("""
    1 . 1 1
    1 . . 1
    . 1 . 1
    1 1 1 1
    """)
    b = Matrix.parse("""
    . . 1 .
    1 . . .
    . . . 1
    . 1 . .
    """)

    a = U.t*a*U
    b = U.t*b*U
    assert space.is_symplectic(a)
    assert space.is_symplectic(b)

    items = list(range(6))

    # a -> (1,2,6,3)
    s = Perm.fromcycles([(0,1,5,2)], items)
    # b -> (1,5)(2,4,3,6)
    t = Perm.fromcycles([(0,4), (1,3,2,5)], items)
    hom = mulclose_hom([a,b], [s,t])

    assert len(hom) == 720

    hom = {v:k for (k,v) in hom.items()}
    S6 = mulclose([s,t])
    S6 = list(S6)
    #for g in S6:
    #    print(g)
    #    print(hom[g])

    #gap> aut := AutomorphismGroup(H);
    #gap> GeneratorsOfGroup(aut);
    #[ ^(1,2,3,4,5,6), ^(1,2), 
    #    [ (5,6), (1,2,3,4,5) ]
    #    -> [(1,2)(3,5)(4,6), (1,2,3,4,5) ] ]
    # gap> Order(aut);
    # 1440

    lgen = [
        Perm.fromcycles([(4,5)], items),
        Perm.fromcycles([(0,1,2,3,4)], items)]
    rgen = [
        Perm.fromcycles([(0,1),(2,4),(3,5)], items),
        Perm.fromcycles([(0,1,2,3,4)], items)]

    outer = mulclose_hom(lgen, rgen)
    assert len(outer) == 720

    if idx == 0:
        return [s,t], hom
    elif idx == 1:
        s = outer[s]
        t = outer[t]
        return [s,t], hom
    assert 0

    
def find_code_Sp2():
    # modify find_code_autos above ...

    l = argv.get("l", 1)
    k = 2
    n = 6*l
    m = n-k
    d = 2

    idx = argv.get("idx", 0)

    print("code: [[%d, %d, %d]]"%(n, k, d))

    space = SymplecticSpace(n)
    Fn = space.F
    Fk = SymplecticSpace(k).F

    solver = Solver()
    Add = solver.add

    nn = 2*n

    # symplectic/unitary encoder map
    E = UMatrix.unknown(nn, nn) 

    H = E[:2*m:2, :]
    L = E[2*m:, :]

    assert d==2, "not implemented"

    # just check single qubit errors
    for i in range(n):
        for e in [(1,0), (0,1), (1,1)]:
            err = lin.zeros2(nn,1)
            err[2*i:2*i+2, 0] = e
            check = H*Matrix(err)
            Add(Or(*[check[j,0].get() for j in range(m)]))

    gen, hom = gethom_Sp2(idx)
    #G = mulclose(gen)
    #for a in G:
    #  for b in G:
    #    assert hom[a]*hom[b] == hom[a*b]
    #print("is hom: yes")

    ops = []
    for g in gen:
        tgt = hom[g] # target Sp_2
        perm = [g[i] for i in range(6)]
        print(perm)
        perm = reduce(add,
            [[i + block*(n//l) for i in perm] for block in range(l)])
        print(perm)
        assert len(perm)==n
        assert len(perm)==len(set(perm))
        op = space.P(*perm).t
        ops.append(op)
        H1 = H * op
        U = UMatrix.unknown(m, m)
        Add( U*H1 == H ) # preserve the stabilizer space

        L1 = L * op
        Add( Fk * L * Fn * L1.t == tgt )
        print(tgt)

    Add( E.t * Fn * E == Fn )

    #return

    count = 0
    while 1:
        result = solver.check()
        if str(result) != "sat":
            #print(result)
            break
    
        model = solver.model()
        _E = E.get_interp(model)
        #print("E:")
        #print(_E)
    
        _H = H.get_interp(model)
        #print("H:")
        #print(_H)
    
        code = QCode.from_encoder(_E.t, k=k)
        print(code)
        print(code.longstr())
        assert code.get_encoder() == _E.t

        gen = set()
        for op in ops:
            dode = op*code
            assert dode.is_equiv(code)
            logop = dode.get_logical(code)
            gen.add(logop)
        gen = list(gen)
        G = mulclose(gen, verbose=True)
        print("logical action:", len(G))

        #Add( E != _E )
        Add( H != _H )

        # this needs to be ForAll U: U*H != _H
        #U = UMatrix.unknown(m, m)
        #Add( U*H != _H )

        count += 1
        #break # <-----------

        #print(".", end="", flush=True)
    #print()

    print("found:", count)


    


def find_css():
    # see also previous version: csscode.find_z3
    params = argv.code
    if params is not None:
        n, k, d = params
        mx = (n-k)//2
        mz = n-k-mx
    else:
        mx = argv.get("mx", 3)
        mz = argv.get("mz", 3)
        n = argv.get("n", mx+mz+1)
        k = n-mx-mz
        assert k>0
    
        d = argv.get("d", 3) 

    print("code: [[%d, %d, %d]]"%(n, k, d))

    solver = Solver()
    Add = solver.add

    Hx = UMatrix.unknown(mx, n)
    Hz = UMatrix.unknown(mz, n)
    Tx = UMatrix.unknown(mx, n)
    Tz = UMatrix.unknown(mz, n)
    Lx = UMatrix.unknown(k, n)
    Lz = UMatrix.unknown(k, n)

    if argv.normal:
        print("normal")
        #print( type(Hx[:,mx]) )
        Add( Hx[:, :mx] == Matrix.identity(mx) )
        Add( Hz[:, :mz] == Matrix.identity(mz) )
        #Add( Tx[:, :mx] == Matrix.identity(mx) )
        #Add( Tx[:, mx:] == Matrix.zeros((mx, n-mx)) )
        #Add( Tz[:, :mz] == Matrix.identity(mz) )
        #Add( Tz[:, mz:] == Matrix.zeros((mz, n-mz)) )

    Add( Hx*Tz.t == Matrix.identity(mx) ) # Hx full rank
    Add( Hx*Hz.t == Matrix.zeros((mx, mz)) ) # Hx/Hz commuting
    Add( Hx*Lz.t == Matrix.zeros((mx, k)) )

    Add( Tx*Tz.t == Matrix.zeros((mx, mz)) )
    Add( Tx*Hz.t == Matrix.identity(mx) ) # Hz full rank
    Add( Tx*Lz.t == Matrix.zeros((mx, k)) )

    Add( Lx*Tz.t == Matrix.zeros((mx, mz)) )
    Add( Lx*Hz.t == Matrix.zeros((mx, mx)) )
    Add( Lx*Lz.t == Matrix.identity(k) )

    #Rx = UMatrix.unknown(n, mx)
    #Rz = UMatrix.unknown(n, mz)

    #Add( Hx*Rx == Matrix.identity(mx, mx) )
    #Add( Hz*Rz == Matrix.identity(mz, mz) )

    if argv.selfdual:
        print("selfdual")
        assert mx==mz
        Add(Hx==Hz)
        Add(Tx==Tz)
        Add(Lx==Lz)

    if k > 0:
        A = UMatrix.unknown(1, mx+k)
        u = A[:, :mx]
        v = A[:, mx:]
    
        op = u*Hx + v*Lx
        lhs = reduce(Or, [v[0,i].get() for i in range(k)])
        rhs = Sum([If(op[0,i].get(),1,0) for i in range(n)]) >= d
        Add(ForAll([A[0,i].v for i in range(mx+k)], If(lhs, rhs, True)))
    
        A = UMatrix.unknown(1, mz+k)
        u = A[:, :mz]
        v = A[:, mz:]
    
        op = u*Hz + v*Lz
        lhs = reduce(Or, [v[0,i].get() for i in range(k)])
        rhs = Sum([If(op[0,i].get(),1,0) for i in range(n)]) >= d
        Add(ForAll([A[0,i].v for i in range(mz+k)], If(lhs, rhs, True)))

    else:
        u = UMatrix.unknown(1, mx)
        for v in numpy.ndindex((2,)*k):
            v = numpy.array(v)
            v.shape = (1,k)
            v = Matrix(v)
            l = v*Lx


    result = solver.check()
    if str(result) != "sat":
        print(result)
        return

    model = solver.model()
    Hx = Hx.get_interp(model)
    Hz = Hz.get_interp(model)
    Tx = Tx.get_interp(model)
    Tz = Tz.get_interp(model)
    Lx = Lx.get_interp(model)
    Lz = Lz.get_interp(model)

    print("Hx:")
    print(Hx)

    print("Hz:")
    print(Hz)

    from qumba.csscode import CSSCode
    #code = CSSCode(Hx=Hx.A, Hz=Hz.A, Lx=Lx.A, Lz=Lz.A, Tx=Tx.A, Tz=Tz.A)
    code = CSSCode(Hx=Hx.A, Hz=Hz.A)

    d_x, d_z = code.distance()
    print(code, d_x, d_z)

    print(code.longstr())


def is_graph_state(code):
    if isinstance(code, QCode):
        assert code.k == 0, "not a state"
        #print(code.longstr(False))
        #print()
        n = code.n
        H = code.H
    else:
        assert isinstance(code, Matrix)
        H = code
        n, nn = H.shape
        assert 2*n==nn

    H = list(H.rowspan())
    shuffle(H)
    H = Matrix(H)
    H = H.reshape(2**n, 2*n)
    s = strop(H)
    stabs = s.split("\n")
    idxs = set()
    A = lin.zeros2(n,n)
    for stab in stabs:
        if stab.count("X") != 1:
            continue
        if "Y" in stab:
            continue
        #print(stab)
        idx = stab.index("X")
        if idx in idxs:
            return
        idxs.add(idx)
        for i,s in enumerate(stab):
            if s=='Z':
                A[idx,i] = 1
    if len(idxs) != n:
        return
    A = Matrix(A)
    #print(A)
    if A==A.t:
        return A



def find_lagrangian():
    n = 4
    code = QCode.fromstr(""" XZZZ ZZXZ ZXZZ ZZZX """)
    code = QCode.fromstr(""" XIZI ZIXZ IXIZ IZZX """) # U shape
    pode = QCode.fromstr(""" XZZI ZXIZ ZIXZ IZZX """)
    #pode = QCode.fromstr(""" XZZZ ZXII ZIXI ZIIX """)

    assert is_graph_state(pode) is not None

    H = pode.H.normal_form()
    assert QCode(H).is_equiv(pode)

    found = set()
    for code in construct.all_codes(4,0,0):
        H = code.H.normal_form()
        dode = QCode(H)
        found.add(dode)
        assert dode.is_equiv(code)
    print(H.shape)
    print(len(found))

    space = code.space
    gen = [space.S(i) for i in range(n)]
    gen += [space.H(i) for i in range(n)]
    G = mulclose(gen)
    print("|G| =", len(G))

    orbits = []
    remain = set(found)
    graphs = set()
    while remain:
        code = remain.pop()
        orbit = [code]
        for g in G:
            dode = g*code
            H = dode.H.normal_form()
            eode = QCode(H)
            assert eode.k == 0
            assert eode.is_equiv(dode)
            assert eode in found
            if eode in remain:
                remain.remove(eode)
                orbit.append(eode)
        orbits.append( orbit )
    orbits.sort(key = len)

    for orbit in orbits:
        col = 0
        smap = SMap()
        for code in orbit:
            A = is_graph_state(code)
            if A is not None:
                graphs.add(A)
                smap[0, col] = str(A)
                col += n+1
        print(len(orbit))
        print(smap)
        print()

    print("graphs:", len(graphs))

    return


    found = [code]
    for g in G:
        dode = g*code
        if dode.is_equiv(pode):
            print("*")
        for eode in found:
            if eode.is_equiv(dode):
                break
        else:
            found.append(dode)
            print("[%d]"%(len(found)), end='', flush=True)
    print(len(found))


def graph_state_orbits():

    n = argv.get("n", 6)

    # first we build all the graph states
    S = numpy.empty(shape=(n,n), dtype=object)
    S[:] = '.'
    I,X,Z = ".XZ"

    def shortstr(S):
        return ("\n".join(''.join(row) for row in S))

    idxs = [(i,j) for i in range(n) for j in range(i+1,n)]
    assert len(idxs) == n*(n-1)//2
    N = len(idxs)

    graphs = set()
    for bits in numpy.ndindex((2,)*N):
        S[:] = I
        for i in range(n):
            S[i,i] = X

        for (i,bit) in enumerate(bits):
            if bit==0:
                continue
            j,k = idxs[i]
            S[j,k] = Z
            S[k,j] = Z

        s = shortstr(S)
        #print(s)
        #code = QCode.fromstr(s)
        #H = code.H.normal_form()
        H = fromstr(s)
        H = Matrix(H).normal_form()
        #print(H)
        graphs.add(H)

    assert len(graphs) == 2**N
    print("graphs:", len(graphs))

    space = SymplecticSpace(n)
    gens = [space.S(i) for i in range(n)]
    gens += [space.H(i) for i in range(n)]

    #graphs = list(graphs)
    orbits = []
    while graphs:
        H = graphs.pop()
        orbit = {H}
        bdy = [H]
        #A = is_graph_state(H)
        #print(A)
        while bdy:
            _bdy = []
            for H in bdy:
              #code = QCode(H)
              for g in gens:
                J = H*g.t
                #dode = g*code
                #J = dode.H
                #assert J == H*g.t
                J = J.normal_form()
                if J in orbit:
                    continue
                #B = is_graph_state(J) 
                if J in graphs:
                    graphs.remove(J)
                _bdy.append(J)
                orbit.add(J)
            bdy = _bdy
        orbits.append(orbit)
        print(len(orbit), end=' ', flush=True)
    print()

    print("found %d orbits" % (len(orbits)))
    orbits.sort(key=len)
    counts = ([len(orbit) for orbit in orbits])
    print(counts, "==", sum(counts))


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




