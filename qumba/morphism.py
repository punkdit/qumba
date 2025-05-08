#!/usr/bin/env python

"""
logical gates as module morphisms

"""


from bruhat.gset import Perm, Group

from qumba import construct, cyclic, autos
from qumba.argv import argv
from qumba.lin import zeros2
from qumba.matrix import Matrix
from qumba.umatrix import UMatrix, Solver, Not, Or, And
from qumba.qcode import QCode


def group_algebra(G):
    N = len(G)
    mul = zeros2(N, N*N)
    for i,g in enumerate(G):
      for j,h in enumerate(G):
        k = G.lookup[g*h]
        mul[k, i + N*j] = 1
    #print(shortstr(mul), mul.shape)
    mul = Matrix(mul)
    unit = zeros2(N, 1)
    i = G.lookup[G.identity]
    unit[i] = 1
    unit = Matrix(unit)

    i_N = Matrix.identity(N)

    assert mul * (unit @ i_N) == i_N
    assert mul * (i_N @ unit) == i_N
    assert mul * (i_N @ mul) == mul * (mul @ i_N)

    return i_N, unit, mul
    

def algebra_morphisms(i_N, unit, mul):
    """
    find permutation morphisms of the algebra
    """

    solver = Solver()
    add = solver.add

    N = len(i_N)
    U = UMatrix.get_perm(solver, N)

    #for g in G:
    #    add( U*g == g*U )
    add( U*mul == mul*(U@U) )
    add( U*unit == unit )

    found = []
    while 1:

        result = solver.check()
        if str(result) != "sat":
            break
        
        model = solver.model()
        u = U.get_interp(model)
    
        #print(u)
        yield u

        found.append(u)
        add(U != u)

        #print(".", end='', flush=True)
    #print()

    #print("algebra_morphisms:", len(found))

    #for g in found:
    #    print(g.order(), end=" ")
    #print()

    #A = mulclose(found)
    #print(len(A))


def main():

    #code = construct.get_toric(1, 3)
    #code = construct.get_513()

    n = 17
    for code in cyclic.all_cyclic_css(n):
        print(code)
        #N, gens = code.get_autos()
        #print(N)
        result = autos.get_autos_css(code)
        print(result)
    return

    print("code:", code)

    space = code.space
    n = code.n
    nn = 2*n
    H = code.H
    m = len(H)

    N, gens = code.get_autos()

    #G = mulclose([Matrix.get_perm(g) for g in gens])
    #assert len(G) == N
    print("code auts:", N)

    G = Group.generate([Perm(g) for g in gens])

    print(G)

    i_N, unit, mul = group_algebra(G)

    # act on codespace
    act = zeros2(nn, N, nn)
    for i,g in enumerate(G):
        u = space.get_perm(g)
        act[:, i, :] = u
    act.shape = (nn, N*nn)
    act = Matrix(act)

    i_nn = Matrix.identity(nn)
    assert act * (unit @ i_nn) == i_nn
    assert act * (mul @ i_nn) == act * (i_N @ act)

    i_m = Matrix.identity(m)
    J = H.pseudo_inverse()
    assert H*J == i_m

    # H is a module morphism
    bact = H * act * (i_N @ J)
    assert H*act == bact*(i_N@H)

    F = space.F
    for m in algebra_morphisms(i_N, unit, mul):

        solver = Solver()
        add = solver.add
    
        U = UMatrix.get_perm(solver, nn)
        add(U.t * F * U == F) # symplectic permutation

        add( U * act == act * (m @ U) )
        
        result = solver.check()
        if str(result) != "sat":
            break
        
        model = solver.model()
        u = U.get_interp(model)
    
        #print()
        #print(u, u.shape)

        H1 = H*u
        dode = QCode(H1)
        print(dode, dode.is_equiv(code))

        for g in gens:
            g = space.get_perm(g)
            print("\t", (g*dode).is_equiv(dode))





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


