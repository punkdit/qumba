#!/usr/bin/env python

#import numpy
from qumba.lin import zeros2, identity2

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.umatrix import UMatrix, Solver, PbEq, PbLe, PbGe
from qumba.cyclic import get_cyclic_perms
from qumba.qcode import QCode, SymplecticSpace
from qumba.action import mulclose
from qumba.transversal import find_lw
from qumba import construct


#class Algebra:
#    def __init__(self, mul, unit):
#        self.mul = mul
#        self.unit = unit


def main_totient():
    # this sequence is Euler totient function phi(n) = |GL(1,Z/n)|
    # https://oeis.org/A000010
    #n = argv.get("n", 12)
    for n in range(2, 20):
        count = len(list(find_homs(n, True)))
        gens = get_cyclic_perms(n)
        #print("gens:", len(gens))
        assert count == len(gens)
        print(n, len(gens))


def main_weak():
    n = argv.get("n", 7)

    A = list(find_homs(n, False, verbose=True))
    print(len(A))

    A = set(A)
    for a in A:
      for b in A:
        assert a*b in A

    n2 = n**2
    vs = [v.reshape(n2) for v in A]
    vs = Matrix(vs)
    print(len(vs), vs.rank())


def find_homs(n, strict, rw=None, verbose=False):
    #print("find_homs(%s)"%n)
    n2 = n**2

    mul = zeros2(n, n2)
    for i in range(n):
      for j in range(n):
        mul[(i+j)%n, i + n*j] = 1 # Z/n group algebra
    mul = Matrix(mul)
    #print(mul)

    unit = [0]*n
    unit[0] = 1
    unit = Matrix(unit).reshape(n,1)

    I = Matrix.get_identity(n)

    swap = zeros2(n2, n2)
    for i in range(n):
      for j in range(n):
        swap[j + n*i, i + n*j] = 1
    swap = Matrix(swap)

    # unital
    assert mul * (unit@I) == I
    assert mul * (I@unit) == I

    # assoc
    assert mul*(mul@I) == mul*(I@mul)

    # comm
    assert mul*swap == mul

    solver = Solver()
    add = solver.add

    P = UMatrix.unknown(n,n)

    add( P*unit == unit )
    add( mul*(P@P) == P*mul )

    if strict:
        add( P*P.t == I )
        for i in range(n):
            add( PbEq([(P[i,j].get(),True) for j in range(n)], 1) )
    else:
        # isomorphism
        U = UMatrix.unknown(n,n)
        add(P*U == I)
        #add(U*P == I)

    if rw is not None:
        for i in range(n):
            add( PbLe([(P[i,j].get(),True) for j in range(n)], rw) )

    #if verbose:
    #    print("solver", end='', flush=True)
    count = 0
    items = []
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
        p = P.get_interp(model)
        add(P != p)

        yield p
        if verbose:
            #print(".", end="", flush=True)
            print(p)
            print()
        count += 1
    if verbose:
        print()


def css_to_sp(U, V=None):
    nn = 2*len(U)
    if V is None:
        V = (~U).t
    U2 = zeros2(nn,nn)
    U2[0:nn:2, 0:nn:2] = U
    U2[1:nn:2, 1:nn:2] = V
    U2 = Matrix(U2)
    return U2




def main_cyclic():
    from qumba.solve import zeros2

    code = QCode.fromstr("""
    X..XX.X.XXXX...
    ..XX.X.XXXX...X
    .XX.X.XXXX...X.
    XX.X.XXXX...X..
    Z..ZZ.Z.ZZZZ...
    ..ZZ.Z.ZZZZ...Z
    .ZZ.Z.ZZZZ...Z.
    ZZ.Z.ZZZZ...Z..
    """)
    #code = construct.get_713()

    assert code.is_cyclic()
    n = code.n
    nn = 2*n
    space = SymplecticSpace(n)
    F = space.F

    count = 0
    for U in find_homs(n, False, 1):
        count += 1
        print()
        print(U)
        U2 = css_to_sp(U)
        assert space.is_symplectic(U2)
        dode = U2*code
        assert dode.is_css()
        assert dode.is_cyclic()
        e = dode.is_equiv(code)
        d = dode.to_css().bz_distance()
        print(dode, e, d)
        #print("/."[e], end="", flush=True)
    print()
    print(count)
    #gens = get_cyclic_perms(n)
    #print(len(gens))

    return

    solver = Solver()
    Add = solver.add

    css = code.to_css()
    mx, mz = css.mx, css.mz
    Hx = Matrix(css.Hx)
    Hz = Matrix(css.Hz)

    U = UMatrix.unknown(n,n)
    V = UMatrix.unknown(n,n)
    Add(U*V.t == Matrix.get_identity(n)) # V == (U^-1)^T

    Mx = UMatrix.unknown(mx,mx)
    #Mxi = UMatrix.unknown(mx,mx)
    #Add(Mx*Mxi == Matrix.get_identity(mx))
    Add(Hx*U.t == Mx*Hx)

    Mz = UMatrix.unknown(mz,mz)
    #Mzi = UMatrix.unknown(mz,mz)
    #Add(Mz*Mzi == Matrix.get_identity(mz))
    Add(Hz*V.t == Mz*Hz)

    P = Matrix.perm([(i+1)%n for i in range(n)])
    Add(P*U == U*P)

    print("solver...")

    count = 0
    gen = []
    while 1:
        result = solver.check()
        result = str(result)
        if result == "unsat":
            break
        elif result == "unknown":
            return
    
        model = solver.model()
        _U = U.get_interp(model)
        _V = V.get_interp(model)
    
        print(_U)
        print()
        gen.append(_U)
        Add(U != _U)
        count += 1

        U2 = css_to_sp(_U, _V)
    
        assert space.is_symplectic(U2)
        #print(U2)
    
        dode = U2*code
        assert dode.is_equiv(code)
        print(".", end="", flush=True)


    print("count:", count)

    G = mulclose(gen, verbose=True)
    print(len(G))


def main_20():
    H = Matrix.parse("""
    1........1.11.1111..
    .1.......1..11..1111
    ..1.......11.11.111.
    ...1......1.1111.1.1
    ....1....11.1...1111
    .....1...1.1..1111.1
    ......1...11.1.1111.
    .......1.1.1......1.
    ........1.1.11111..1
    """)

    print(H.shape)
    m, n = H.shape
    assert (H*H.t).sum() == 0

    solver = Solver()
    add = solver.add

    U = UMatrix.unknown(n,n)
    for i in range(n):
        add( PbEq([(U[i,j].get(),True) for j in range(n)], 1) )
        add( U[i,i] == False ) # fixed-point free
    #for i in range(n):
    #    #add( PbGe([(U[i,j].get(),True) for j in range(n)], 1) )
    #    add( PbLe([(U[i,j].get(),True) for j in range(n)], 3) )
    #    add( PbLe([(U[j,i].get(),True) for j in range(n)], 3) )
    #    add( U[i,i] == True )

    I = Matrix.get_identity(n)
    #add(U*U.t == I)
    add(U*U == I) # involution

    #Ui = UMatrix.unknown(n,n)
    #add(U*Ui == I)

    Im = Matrix.get_identity(m)
    Um = UMatrix.unknown(m,m)
    Vm = UMatrix.unknown(m,m)
    add(Um*Vm == Im)
    
    add(H*U.t == Um*H)
    #add(H*Ui == Vm*H)

    code = QCode.build_css(H,H)
    space = code.space

    print(code.is_cyclic())

    ops = [space.get_S(), space.get_H()]
    assert len(mulclose(ops)) == 6

    #CX = space.CX(6,3)
    #A = CX[::2, ::2]
    #print(A)
    #assert CX == css_to_sp(A)
    #return

    def get_pairs(U):
        pairs = []
        for i in range(n):
          for j in range(i+1,n):
            if U[i,j]:
                assert U[j,i]
                pairs.append((i,j))
        return pairs

    gen = set()
    for g in ops:
        dode = g*code
        l = dode.get_logical(code)
        gen.add(l)
        print(l, l.shape)
        print(SymplecticSpace(2).get_name(l))

    #print(code.longstr())

    count = 0
    while 1:
        
        result = solver.check()
        result = str(result)
        if result == "unsat":
            break
        elif result == "unknown":
            return
    
        model = solver.model()
        U1 = U.get_interp(model)
        add(U != U1)

        #print(U1)
        #print()

        #U1 = U1 + I
        #if U1.rank() != n:
        #    print("/")
        #    continue
        #print("+")

        E = css_to_sp(U1)
        assert space.is_symplectic(E)

        V1 = Um.get_interp(model)
        assert ~V1*V1 == Im

        dode = E*code
        #print(dode.longstr())
        assert dode.is_equiv(code)

        l = dode.get_logical(code)
        if l not in gen:
            gen.add(l)
            G = mulclose(gen)
            print(len(G))

        count += 1
        #print(".", end="", flush=True)
        pairs = get_pairs(U1)
        print(pairs)
        op = space.get_identity()
        for (i,j) in pairs:
            op = op*space.CZ(i,j)
            #op = op*space.CX(i,j)
            #op = op*space.SWAP(i,j)
        dode = op*code
        assert dode.is_equiv(code)

        l = dode.get_logical(code)
        if l not in gen:
            gen.add(l)
            G = mulclose(gen)
            print(len(G))
            return

        #break

    print()

    print(count)
    



def main_lw():
    # [[63,51,3]]
    code = QCode.fromstr("""
    XXXIIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIII
    ZZZIIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIII
    XXIIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIX
    ZZIIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZ
    XIIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXX
    ZIIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZ
    IIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXXX
    IIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZZ
    IIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXXXI
    IIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZZI
    IIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXXXII
    IIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZZII
    """)
    n = code.n
    css = code.to_css()
    Hx = Matrix(css.Hx)
    Hz = Matrix(css.Hz)
    Lx = Matrix(css.Lx)
    Lz = Matrix(css.Lz)
    wenum = Hx.get_wenum()
    #print([(idx,wenum[idx]) for idx in range(64)])

    perms = get_cyclic_perms(n)
    print("perms:", len(perms))
    G = []
    for g in perms:
        g = Matrix(g[::2, ::2])
        H1 = Hx*g
        if H1.t.solve(Hx.t) is not None:
            G.append(g)
        #L1 = Lx*g
        #if L1.t.solve(Lx.t) is not None:
        #    G.append(g)
    print(len(G))
    for g in G:
      for h in G:
        assert g*h in G
    g = Matrix.perm([(i+1)%n for i in range(n)])
    G = mulclose([g]+G)
    print("|G| =", len(G))

    HL = Hx.concatenate(Lx)

    count = 0
    logops = []
    rows = []
    for l in find_lw(HL, 3):
        l = l[0]
        if l[0]:
            print(l)
            rows.append(l)
            #print(Lz*l, Hz*l)
        logops.append(l)
        assert (Hz * l).sum() == 0
        assert (Lz * l).sum() != 0
        count += 1
    L = Matrix(logops)
    print(L.shape)
    #print(L.A.sum(0))
    #print(L.A.sum(1))

    #for i in range(1, n):
    #    for row in rows:
    #        if row[i]:
    #            print(row)
    #return

    print(L.rank())
    print((Hx.concatenate(L)).rank() - len(Hx))
    assert Hx.rank() == len(Hx)

    remain = set(logops)
    orbits = []
    while remain:
        #print(len(remain))
        op = iter(remain).__next__()
        orbit = []
        for g in G:
            u = op*g
            if u in remain:
                remain.remove(u)
                orbit.append(u)
        orbits.append(orbit)

    orbits.sort(key = len)
    for orbit in orbits:
        print("orbit:", len(orbit))
        L1 = Matrix(orbit)
        print("rank:", (Hx.concatenate(L1)).rank() - len(Hx))

    o = orbits[0] + orbits[1]
    L1 = Matrix(o)
    print("rank 21+63:", (Hx.concatenate(L1)).rank() - len(Hx))
    o = orbits[0] + orbits[1] + orbits[2]
    L1 = Matrix(o)
    print("rank 21+63+189:", (Hx.concatenate(L1)).rank() - len(Hx))


def main_cnot():
    n, m = argv.get("n", 3), argv.get("m", 2)
    
    A = zeros2(m, n)
    for i in range(m):
        A[i, n-m+i] = 1
    A = Matrix(A)

    gen = []
    for i in range(n):
      for j in range(n):
        if i<=j:
            continue
        g = identity2(n)
        g[i,j] = 1
        g = Matrix(g)
        gen.append(g)

    found = set([A])
    bdy = list(found)

    while bdy:
        #print(bin(len(bdy)), end=",", flush=True)
        s = bin(len(bdy))
        s = s.replace("0b","")
        s = s.replace("0",".")
        print(s.rjust(30))
        _bdy = []
        for g in gen:
            for A in bdy:
                B = A*g
                B = B.normal_form()
                if B not in found:
                    _bdy.append(B)
                    found.add(B)
        bdy = _bdy
    print()
    print(len(found))
    found = list(found)
    found.sort(key=str)
    #for g in found:
    #    print(g, g.shape)



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




