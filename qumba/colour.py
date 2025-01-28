#!/usr/bin/env python

from functools import reduce

import numpy

from bruhat.qcode import Geometry, get_adj, Group

from qumba.argv import argv
from qumba.qcode import QCode, strop
from qumba.csscode import distance_z3
from qumba.symplectic import SymplecticSpace
from qumba.lin import shortstr, linear_independent, dot2, solve
from qumba.action import mulclose_find
from qumba.transversal import Solver, UMatrix, And, Or, one, Sum, If


def make_colour():
    print("make_colour")
    key = argv.get("key", (3, 8))
    idx = argv.get("idx", 13)

    print()
    geometry = Geometry(key, idx, True)
    #graph = geometry.build_graph(desc)
    G = geometry.G
    print("|G| = %d, idx = %d" % (len(G), idx))

    faces = geometry.get_cosets([0,1,1])
    edges = geometry.get_cosets([1,0,1])
    verts = geometry.get_cosets([1,1,0])
    print("faces=%d, edges=%d, verts=%d"%(len(faces), len(edges), len(verts)))

    A = get_adj(faces, verts)
    #from bruhat.hecke import colour
    #colour(A)
    A1 = linear_independent(A)
    print(shortstr(A1))
    print()
    code = QCode.build_css(A1, A1, d=(4 if idx in [10,13] else None))
    print(code)
    #d = distance_z3(code.to_css())
    #print("d =", d)

    G = geometry.G
    gens = G.gens
    f, e, v = gens # face, edge, vert
    H = Group.generate([f*e, f*v, e*v])
    assert len(G) % len(H) == 0
    index = len(G) // len(H)
    assert index in [1, 2]
    if index == 1:
        print("non-orientable")
    elif index == 2:
        print("orientable")

    g = f*e*v*e*f # colour symmetry (XXX is one generator enough?)
    H = Group.generate([e, v, g])
    print("|H| =", len(H))
    index = len(G)//len(H)
    print("|G:H| =", index)

    if index != 3:
        print("non-3_colourable")
        return

    if len(G) > 1000:
        return

    colours = G.left_cosets(H)
    B = get_adj(faces, colours)
    #print(shortstr(B), B.shape)
    #print("colour:", B.sum(0), B.sum(1))

    # every edge is related to two colours
    CE = get_adj(colours, edges)
    #print("CE:")
    #print(shortstr(CE), CE.shape)
    #print(CE.sum(0), CE.sum(1))

    # inverse gives every edge a unique colour
    CE = 1 - CE
    print("CE:")
    print(shortstr(CE), CE.shape)

    RED, BLUE, GREEN = 0, 1, 2
    red_edges = [idx for idx in range(len(edges)) if CE[RED, idx]]
    blue_edges = [idx for idx in range(len(edges)) if CE[BLUE, idx]]
    print("red_edges:", red_edges)
    print("blue_edges:", blue_edges)

    r_faces = [idx for idx in range(len(faces)) if B[idx, RED]]
    bg_faces = [idx for idx in range(len(faces)) if not B[idx, RED]]

    b_faces = [idx for idx in range(len(faces)) if B[idx, BLUE]]
    rg_faces = [idx for idx in range(len(faces)) if not B[idx, BLUE]]
    return

    F = len(faces)
    greens = [i for i in range(F) if B[i,0]]
    print(greens)

    R = A[greens, :]
    print(shortstr(R))

    return

    # -------------------------------------------------

    solver = Solver()
    add = solver.add

#    items = []
#    for idx in greens:
#        jdxs, = numpy.where(A[idx, :])
#        print(jdxs)
#        N = len(jdxs)
#
#        I = UMatrix(numpy.identity(N, dtype=int))
#    
#        UX = UMatrix.unknown(N, N)
#        UZ = UMatrix.unknown(N, N)
#        add(UX.t * UZ == I) # symplectic
#        items += [UX, UZ]

    n = code.n
    I = UMatrix(numpy.identity(n, dtype=int))
    UX = UMatrix.unknown(n, n)
    UZ = UMatrix.unknown(n, n)
    add(UX.t * UZ == I) # symplectic
    #add(UX == I)

    select = UMatrix.unknown(2, n) # each qubit is in one of two codes
    for i in range(n):
        #add(Or(select[0,i], select[1,i]))
        add(select[0,i]+select[1,i] == one)
    #add(select[0].sum() == n//2)
    assert n%2==0
    add(Sum([If(select[0,j].get(),1,0) for j in range(n)])==n//2)

    jdxss = []
    for idx in greens:
        jdxs = [j for j in range(n) if A[idx, j]]
        jdxss.append(jdxs)
    print(jdxss)
    for i in range(n):
        for jdxs in jdxss:
            if i in jdxs:
                break
        else:
            assert 0
        for j in range(n):
            if j in jdxs:
                pass # OK
#            else:
#                add(UX[i,j] == 0)
#                #add(UX[j,i] == 0)
#                add(UZ[i,j] == 0)
#                #add(UZ[j,i] == 0)

    A = UMatrix(A)
    AX = A*UX.t
    AZ = A*UZ.t

    for AA in [AX, AZ]:
      #break #FAIL
      for row in range(len(AA)):
        add(
            Or(*[
            reduce(And, [If(AA[row, col]==1, select[k, col]==1, False) for col in range(n)])
            for k in [0,1]])
        )
        

    result = solver.check()
    assert str(result) == "sat"

    model = solver.model()

    UX = UX.get_interp(model)
    UZ = UZ.get_interp(model)
    select = select.get_interp(model)

    #idxs = reduce(lambda a,b:a+b, jdxss)
    #UX = UX[idxs, :][:,idxs]
    #print(shortstr(UX), UX.shape)
    #return


    HX = dot2(A1, UX.transpose())
    HZ = dot2(A1, UZ.transpose())

    print()
    print(shortstr(HX), HX.shape)
    print()
    print(shortstr(HZ), HZ.shape)

    assert dot2(HX, HZ.transpose()).sum() == 0 # commutes

    print()
    print(select)



def find_encoder():
    n = 6
    nn = 2*n

    space = SymplecticSpace(n)
    lhs = space.parse("""
    ZZ....
    .ZZ...
    ..ZZ..
    ...ZZ.
    ....ZZ
    XX....
    .XX...
    ..XX..
    ...XX.
    ....XX
    """)
    #print(lhs, lhs.shape)

    rhs = space.parse("""
    .Z....
    ..Z...
    ...Z..
    ....Z.
    .....Z
    X.X...
    .X.X..
    ..X.X.
    ...X.X
    X...X.
    """)
    #print(rhs, rhs.shape)
    N = 9
    lhs = lhs[:N]
    rhs = rhs[:N]

    U = lhs.solve(rhs)
    assert U is not None
    #print(U, U.shape)
    assert (lhs*U == rhs)

    # -----------------------------------

    code = QCode.fromstr("""
    ZZZZZZ
    XX....
    .XX...
    ..XX..
    ...XX.
    ....XX
    """)

    U = code.get_encoder()

    r = (lhs * U)
    #print(r, r.shape)
    #dode = QCode(H = r[5:])
    #print(dode.longstr())
    #return

    # -----------------------------------

    solver = Solver()
    add = solver.add

    F = UMatrix(space.F)
    L = UMatrix(lhs)
    R = UMatrix(rhs)
    U = UMatrix.unknown(nn, nn)

    add(U.t * F * U == F)
    #add(U*L.t == R.t)
    add(L*U == R)

    found = set()
    while 1:
        result = solver.check()
        #assert str(result) == "sat"
        if str(result) != "sat":
            break
        model = solver.model()

        U0 = U.get_interp(model)
        found.add(U0)
        add(U != U0)
        #print(U0)
        code = QCode.from_encoder(U0, k=0)
        s = code.longstr()
        #print()

        h = strop(code.H)
        if 'Z' in h or 'Y' in h:
            continue
        #print(s)
        #print()

    print(len(found))

    # Jordan-Wigner ?!??!
    """
    H =
    X.....
    XX....
    .XX...
    ..XX..
    ...XX.
    ....XX
    T =
    ZZZZZZ
    .ZZZZZ
    ..ZZZZ
    ...ZZZ
    ....ZZ
    .....Z
    """

    code = QCode.fromstr(
    """
    X.....
    XX....
    .XX...
    ..XX..
    ...XX.
    ....XX
    """,
    """
    ZZZZZZ
    .ZZZZZ
    ..ZZZZ
    ...ZZZ
    ....ZZ
    .....Z
    """)
    print(code.longstr())
    E = code.get_encoder()
    print(E)

    CX = space.CX
    gen = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    print(len(gen))

    g = mulclose_find(gen, E, verbose=True)
    print(g.name)



if __name__ == "__main__":

    from time import time
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





