#!/usr/bin/env python


import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
import pickle

import numpy

from qumba import solve
solve.int_scalar = numpy.int32 # qupy.solve
from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba import csscode, construct
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
from qumba import transversal
from qumba import clifford, matrix
from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
from qumba.syntax import Syntax
from qumba.circuit import (Circuit, measure, barrier, send, vdump, variance,
    parsevec, strvec, find_state, get_inverse, load_batch, send_idxs)
from qumba.circuit_css import css_encoder, process



def gate_search(space, gates, target):
    gates = list(gates)
    A = space.get_identity()
    found = []
    while A != target:
        if len(found) > 100:
            return
        count = (A+target).sum()
        #print(count, end=' ')
        shuffle(gates)
        for g in gates:
            B = g*A # right to left
            if (B+target).sum() <= count:
                break
        else:
            #print("X ", end='', flush=True)
            return None
        found.insert(0, g)
        A = B
    #print("^", end='')
    return found


def find_encoder(space, H, gates):
    print("find_encoder")
    print(H)


def test():

    #code = construct.get_golay(23)
    code = construct.get_toric(2,2)
    n = code.n

    H = code.H
    print(code)
    print(code.longstr())

    space = code.space
    E = code.get_encoder()
    #print(E)

    name = space.get_name(E)
    print(name, len(name))

    code = code.to_css()
    #name = css_encoder(code.Hz)
    #print(name, len(name))

    E1 = space.get_expr(name)

    print(E == E1)

#    dode = QCode.from_encoder(E1, k=1)
#    assert dode.is_css()
#    dode = dode.to_css()
#    dode.bz_distance()
#    print(dode)

    CX, H = space.CX, space.H
    E1 = E #* H(3)*H(4)*H(5)

    gates = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    #gates += [H(i) for i in range(n)]


    find_encoder(space, code.H, gates)

    
    

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



