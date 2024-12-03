#!/usr/bin/env python

from random import shuffle
from functools import reduce
from operator import add

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span, array2)
from qumba.qcode import QCode, SymplecticSpace, get_weight, fromstr
from qumba.argv import argv


def search_distance_z3(code, d):
    import z3
    from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

    if code.k == 0:
        return 
    if code.m == 0:
        return 1
    H, L = code.H, code.L

    #print("search_distance: d=%d"%d)

    m, nn = H.shape
    k, nn1 = L.shape
    assert nn==nn1
    n = nn//2

    solver = Solver()
    add = solver.add
    v = [Bool("v%d"%i) for i in range(nn)]

    term = Sum([If(Or(v[2*i], v[2*i+1]),1,0) for i in range(n)]) == d
    add(term)

    def check(op):
        terms = []
        for j in range(n):
            x, z = op[2*j:2*j+2]
            vx, vz = v[2*j:2*j+2]
            if x and z:
                terms.append(Xor(vx, vz))
            elif x:
                terms.append(vz)
            elif z:
                terms.append(vx)
        #assert len(terms)>1, len(terms)
        return reduce(Xor, terms)

    # parity checks
    for i in range(m):
        add(Not(check(H[i])))

    # non-trivial logical
    term = reduce(Or, [check(L[i]) for i in range(k)])
    add(term)

    result = solver.check()
    #print(result)
    if result != z3.sat:
        return 

    model = solver.model()
    v = [model.evaluate(v[i]) for i in range(nn)]
    v = [int(eval(str(vi))) for vi in v]
    v = array2(v)

    F = code.space.F
    u = dot2(code.H, F, v)
    assert u.sum() == 0, "bug bug... try updating z3?"
    u = dot2(code.L, F, v)
    assert u.sum() != 0, "bug bug... try updating z3?"
    assert get_weight(v) == d, "bug bug... try updating z3?"
    return v


def distance_z3(code, verbose=False):

    if code.k == 0:
        return code.n

    d = 1
    while 1:
        if search_distance_z3(code, d) is not None:
            if code.d is None:
                code.d = d
            return d
        if verbose:
            print("d >", d)
        d += 1


def distance_z3_lb(code, max_d, verbose=False):
    "is the distance > max_d ?"

    if code.k == 0:
        return code.n > max_d # ?

    d = 1
    while d <= max_d:
        if search_distance_z3(code, d) is not None:
            if code.d is None:
                code.d = d
            return False
        if verbose:
            print("d >", d)
        d += 1

    return True


def distance_meetup(code, max_m=None, verbose=False):
    w = logop_meetup(code, max_m, verbose)
    if w is not None:
        return get_weight(w)


def logop_meetup(code, max_m=None, verbose=False):
    from qumba.util import choose

    n = code.n
    if max_m is None:
        max_m = 1+n//2
    nn = 2*n
    items = list(range(n))

    H = code.H.A
    L = code.L.A

    lookup = {}
    v = numpy.zeros((nn,), dtype=numpy.int8)
    #print(v)
    m = 1
    while m <= max_m:
      if verbose: print("m =", m)
      for idxs in choose(items, m):
        for bits in numpy.ndindex((3,)*m):
            v[:] = 0
            for i,idx in enumerate(idxs):
                tgt = v[2*idx:2*idx+2]
                bit = bits[i]
                if bit==0:
                    tgt[:] = [1,0] # X
                elif bit==1:
                    tgt[:] = [0,1] # Z
                elif bit==2:
                    tgt[:] = [1,1] # Y
                else:
                    assert 0
            s = dot2(H, v)
            key = s.tobytes()
            u = lookup.get(key)
            if u is None:
                #print(".", end="")
                lookup[key] = v.copy()
            else:
                #print("*", end="")
                w = (u+v)%2
                assert dot2(H, w).sum() == 0
                if dot2(L, w).sum():
                    if verbose: print("lookup size:", len(lookup))
                    #print("found")
                    return w
      #print()
      m += 1

    #return 2*m+1



def test():
    from qumba.matrix import Matrix
    from qumba.cyclic import get_code
    from qumba.qcode import strop

    #code = QCode.fromstr("""
    #""")

    #code = get_code((13,1,5))
    code = get_code()

    print(code)

    w = distance_meetup(code)
    d = get_weight(w)
    code.d = d
    print(code)
    w = Matrix(w)
    #print(w)
    print(strop(w))







if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

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


