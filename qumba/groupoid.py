#!/usr/bin/env python
"""
black box fault-tolerance
"""

from random import shuffle, choice, randint
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul
from math import ceil

import numpy

from qumba.lin import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, zeros2, solve2, normal_form)
#from qumba.matrix import Matrix
#from qumba.distance import distance_z3
#from qumba.autos import get_isos, is_iso
#from qumba.action import Perm, mulclose_find, mulclose
#from qumba import csscode 
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr, Matrix
from qumba import construct
from qumba.util import choose
from qumba.argv import argv
from qumba.transversal import find_transversal

from qumba.umatrix import UMatrix, Solver


def find_avoid(Hx, Lx, d):
    mx = len(Hx)
    k = len(Lx)

    avoid = []
    for idx in numpy.ndindex((2,)*k):
        idx = numpy.array(idx)
        if idx.sum()==0:
            continue
        l = dot2(idx, Lx)
        for jdx in numpy.ndindex((2,)*mx):
            jdx = numpy.array(jdx)
            lh = (l + dot2(jdx, Hx)) % 2
            if lh.sum() == d:
                avoid.append(lh)

    idxs = set()
    s = int(ceil(d/2))
    for v in avoid:
        idx = list(numpy.where(v)[0])
        #print(v, idx)
        assert len(idx) == d
        for ii in choose(idx, s):
            #print("\t", ii)
            idxs.add(ii)
    idxs = list(idxs)
    idxs.sort()

    return idxs


def get_avoid(code):
    css = code.to_css()
    x_avoid = find_avoid(css.Hx, css.Lx, code.d)
    z_avoid = find_avoid(css.Hz, css.Lz, code.d)
    return x_avoid, z_avoid


def test_avoid():

    code = construct.get_css((15,5,3))
    code = code.to_qcode()

    #code = construct.get_toric(3,3) # ??
    #code = construct.get_surface(3,3) # [[9,1,3]] unsat
    #code = construct.get_10_2_3() # sat
    code.get_distance()
    print(code)

    x_avoid, z_avoid = get_avoid(code)
    print(x_avoid)
    print(z_avoid)

    for M in find_gates(code, x_avoid, z_avoid):
        #print(M)
        pass
    return


def find_gates(code, x_avoid, z_avoid):
    solver = Solver()
    Add = solver.add

    H = code.H
    L = code.L
    k = code.k
    m = code.m
    n = code.n
    nn = 2*n

    Fn = code.space.F

    U = code.space.get_identity()
    CX = code.space.CX
    for _ in range(100):
        a = randint(0,n-2)
        b = randint(a+1,n-1)
        U = CX(a,b)*U

        a = randint(0,n-2)
        b = randint(a+1,n-1)
        U = CX(b,a)*U

    T = zeros2(nn,nn)
    for i in range(nn):
      for j in range(nn):
        if (i+j)%2:
            assert U[i,j] == 0
            T[i,j] = 1
        #else:
        #    assert U[i,j] == 0

    #print(U)
    #print()
    #print(shortstr(T))
    #print()

    space = SymplecticSpace(m)
    Fm = space.F

    U = UMatrix.unknown(2*n, 2*n)
    Add(U.t*Fn*U == Fn) # quadratic constraint

    J = UMatrix.unknown(m, m)
    Add(H*U.t == J*H)

    # restrict to CX circuit
    for i in range(nn):
      for j in range(nn):
        if (i+j)%2:
            Add(U[i,j] == 0)

    #X1 = Matrix(fromstr("X" + "I"*8)).t
    #print(M*X1)

    for idx in x_avoid:
        #print(idx)
        assert len(idx) == 2, "TODO"
        for i in range(n):
            Add(U[2*idx[0], 2*i] * U[2*idx[1], 2*i] == 0) # x-type
    for idx in z_avoid:
        #print(idx)
        assert len(idx) == 2, "TODO"
        for i in range(n):
            Add(U[2*idx[0]+1, 2*i+1] * U[2*idx[1]+1, 2*i+1] == 0) # z-type

    sk = SymplecticSpace(k)
    Fk = sk.F
    #LL = Fk*L*Fn*U*L.t
    LL = Fk*L*U.t*Fn*L.t
    I = sk.get_identity()
    Add(LL != I)

    found = set()
    gen = set()
    fgen = set()
    count = 0
    while 1:
        count += 1
        result = solver.check()
        if str(result) != "sat":
            break
        #if count%100==0:
        #    print(".", end="", flush=True)
        

        model = solver.model()
        M = U.get_interp(model)
        assert M.t*Fn*M == Fn

        dode = code.apply(M)
        assert dode.is_equiv(code)
        logop = dode.get_logical(code)
        #if logop.is_identity():
        #    pass
        #    print(".", end='', flush=True)
        #    #continue
        #else:
        #    print()
        #    print(logop)
        print(logop)
        print()

        yield M

        #print()
        #print(Fk*(L*M.t)*Fn*L.t)
        #print()
        print(M)
        print()

        #Add(U != M)
        Add(LL != logop)

        #if count>2:
        #    break

    print()




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





