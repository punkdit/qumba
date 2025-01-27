#!/usr/bin/env python

#import numpy
from qumba.solve import zeros2

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.umatrix import UMatrix, Solver, PbEq
from qumba.cyclic import get_cyclic_perms
from qumba.qcode import QCode, SymplecticSpace
from qumba.action import mulclose
from qumba.transversal import find_lw
from qumba import construct


class Algebra:
    def __init__(self, mul, unit):
        self.mul = mul
        self.unit = unit


def main():
    # this sequence is Euler totient function phi(n) = |GL(1,Z/n)|
    # https://oeis.org/A000010
    #n = argv.get("n", 12)
    for n in range(2, 20):
        count = find(n)
        gens = get_cyclic_perms(n)
        #print("gens:", len(gens))
        assert count == len(gens)
        print(n, len(gens))


def main_weak():
    n = argv.get("n", 7)

    A = find(n, False, True)
    print(len(A))

    A = set(A)
    for a in A:
      for b in A:
        assert a*b in A

    n2 = n**2
    vs = [v.reshape(n2) for v in A]
    vs = Matrix(vs)
    print(len(vs), vs.rank())


def find(n, strict=True, verbose=False):
    #print("find(%s)"%n)
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

        if not strict:
            items.append(p)
        if verbose:
            #print(".", end="", flush=True)
            print(p)
            print()
        count += 1
    if verbose:
        print()

    if not strict:
        return items
    return count




def test_cyclic():
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
    code = construct.get_713()
    assert code.is_cyclic()
    n = code.n
    nn = 2*n
    space = SymplecticSpace(n)
    F = space.F

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
        if str(result) != "sat":
            break
    
        model = solver.model()
        _U = U.get_interp(model)
        _V = V.get_interp(model)
    
        #print(_U)
        #print()
        gen.append(_U)
        Add(U != _U)
        count += 1
    
        U2 = zeros2(nn,nn)
        U2[0:nn:2, 0:nn:2] = _U
        U2[1:nn:2, 1:nn:2] = _V
        U2 = Matrix(U2)
    
        assert space.is_symplectic(U2)
        #print(U2)
    
        dode = U2*code
        assert dode.is_equiv(code)
        print(".", end="", flush=True)


    print("count:", count)

    G = mulclose(gen, verbose=True)
    print(len(G))




def test_lw():
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
    H = Matrix(css.Hx)
    Lx = Matrix(css.Lx)
    wenum = H.get_wenum()
    #print([(idx,wenum[idx]) for idx in range(64)])

    perms = get_cyclic_perms(n)
    print(len(perms))
    G = []
    for g in perms:
        g = Matrix(g[::2, ::2])
        H1 = H*g
        if H1.t.solve(H.t) is not None:
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
    print(len(G))

    HL = H.concatenate(Lx)

    count = 0
    logops = []
    for l in find_lw(HL, 3):
        if l[0,0]:
            print(l)
        logops.append(l[0])
        count += 1
    print(count)
    L = Matrix(logops)
    print(L.shape)
    #print(L.A.sum(0))
    #print(L.A.sum(1))
    print(L.rank())
    print((H.concatenate(L)).rank() - len(H))
    assert H.rank() == len(H)

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
        print("rank:", (H.concatenate(L1)).rank() - len(H))

    o = orbits[0] + orbits[1]
    L1 = Matrix(o)
    print("rank 21+63:", (H.concatenate(L1)).rank() - len(H))
    o = orbits[0] + orbits[1] + orbits[2]
    L1 = Matrix(o)
    print("rank 21+63+189:", (H.concatenate(L1)).rank() - len(H))




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




