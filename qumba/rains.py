#!/usr/bin/env python

"""
try to implement section "Linear codes" in:
    https://arxiv.org/abs/quant-ph/9703048
    Nonbinary quantum codes
    Eric M. Rains

"""

from functools import reduce
from operator import add, matmul, mul

import numpy

from qumba.qcode import QCode, SymplecticSpace, Matrix, fromstr, shortstr, strop
from qumba.action import mulclose, Group, mulclose_find
from qumba.util import allperms, all_subsets
from qumba import equ
from qumba import construct
from qumba import autos
from qumba.unwrap import unwrap
from qumba.argv import argv


def span(G):
    G = list(G)
    N = len(G)
    algebra = set()
    for bits in numpy.ndindex((2,)*N):
        A = Matrix([[0,0],[0,0]])
        for i,bit in enumerate(bits):
            if bit:
                A = A + G[i]
        algebra.add(A)
    algebra = list(algebra)
    algebra.sort()
    return algebra


def main():

    space = SymplecticSpace(1)
    I = space.I()
    S = space.S()
    H = space.H()

    G = mulclose([S,H])
    G = list(G)
    assert len(G) == 6

    algebra = span(G)
    assert len(algebra) == 16

#    for H in [
#        mulclose([S]),
#        mulclose([H]),
#    ]:
        
    for gen in all_subsets(G):
        if not gen:
            continue
        H = mulclose(gen)
        #if len(H) == 6:
        #    continue
        #print(len(H), end=' ')
        algebra = span(H)
        print(len(H), len(algebra))





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




