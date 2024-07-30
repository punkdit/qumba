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


def algclose(gen, verbose=False, maxsize=None):
    els = set(gen)
    bdy = list(els)
    changed = True
    while bdy:
        if verbose:
            print(len(els), end=" ", flush=True)
        _bdy = []
        for A in gen:
            for B in bdy:
                for C in [A*B, A+B]:
                  if C not in els:
                    els.add(C)
                    _bdy.append(C)
                    if maxsize and len(els)>=maxsize:
                        return els
        bdy = _bdy
    if verbose:
        print()
    return els



def main_1():

    space = SymplecticSpace(1)
    I = space.I()
    S = space.S()
    H = space.H()

    G = mulclose([S,H])
    G = list(G)
    assert len(G) == 6

    algebra = span(G)
    assert len(algebra) == 16

    found = set()
    for gen in all_subsets(G):
        if not gen:
            continue
#        H = algclose(gen)
#        H = list(H)
#        H.sort()
#        H = tuple(H)
        algebra = algclose(gen)
        algebra = list(algebra)
        algebra.sort()
        algebra = tuple(algebra)
        found.add(algebra)
        #print(len(H), len(algebra))

    print("found:", len(found))
    for A in found:
        print("|A| = ", len(A))
        for a in A:
            print(a)
            print()

class Algebra(object):
    def __init__(self, items):
        items = list(items)
        items.sort()
        self.items = tuple(items)
    def __eq__(self, other):
        return self.items == other.items
    def __hash__(self):
        return hash(self.items)
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]
    @classmethod
    def generate(cls, gen):
        A = algclose(gen)
        return cls(A)


def main():
    n = 2
    nn = 2*n
    space = SymplecticSpace(n)
    I = space.I()
    S, H = space.S, space.H

    gen = [S(0),S(1),H(0),H(1),space.CX(0,1)]
    G = mulclose(gen)
    print(len(G))

    A = Algebra.generate(gen)
    assert len(A) == 2**16

    zero = Matrix.zeros((nn,nn))

#    found = set()
#    for a in A:
#      for b in A:
#        B = Algebra.generate([a,b])
#        found.add(B)
#    print(len(found))





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




