#!/usr/bin/env python

from random import shuffle
from functools import reduce
from operator import add

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    zeros2, rank, rand2, pseudo_inverse, kernel, direct_sum, span, array2)
from qumba.qcode import QCode, SymplecticSpace, get_weight, fromstr
from qumba.csscode import CSSCode, find_zx_duality
from qumba.argv import argv


def search_distance_z3(code, d):
    import z3
    from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

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
        assert len(terms)>1
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
    return True


def distance_z3(code):

    d = 1
    while 1:
        if search_distance_z3(code, d):
            return d
        d += 1







