#!/usr/bin/env python

"""
try to implement section "Linear codes" in:
    https://arxiv.org/abs/quant-ph/9703048
    Nonbinary quantum codes
    Eric M. Rains

"""

from functools import reduce, cache
from operator import add, matmul, mul

import numpy

from qumba.qcode import QCode, SymplecticSpace, Matrix, fromstr, shortstr, strop
from qumba.matrix import scalar
from qumba.action import mulclose, Group, mulclose_find
from qumba.util import allperms, all_subsets
from qumba import equ
from qumba import construct
from qumba import autos
from qumba.smap import SMap
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


def generate(gen, verbose=False, maxsize=None):
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


class Algebra(object):
    def __init__(self, items):
        items = list(items)
        items.sort(key = lambda a:(str(a).count('1'), a))
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
        A = generate(gen)
        return cls(A)

    def dump(algebra):
        smap = SMap()
        for i,a in enumerate(algebra):
            smap[0, 4*i] = str(a)
        print(smap)
        print()



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

    print("algebra:")
    Algebra(algebra).dump()
    #return

    for g in G:
        assert g*~g == I

    count = 0
    found = set()
    for gen in all_subsets(algebra):
        count += 1
        if algebra[0] not in gen:
            continue

        if I not in gen:
            continue

        A = Algebra.generate(gen)
        found.add(A)

        #if len(found) > 2:
        #    break

    J = H
    conj = lambda a : J*a.t*J

    print("found:", len(found), "count:", count)
    found = list(found)
    found.sort(key = lambda A : (len(A), tuple(A)))
    for A in found:
        print("|A| = ", len(A))
        A.dump()

        for a in A:
            assert conj(a) in A
            for b in A:
                assert a+b in A
                assert a*b in A
                assert conj(a*b) == conj(b)*conj(a)

    for A in found:
        sig = ['.']*len(found)
        for g in G:
            B = Algebra([g*a*~g for a in A])
            i = found.index(B)
            sig[i] = "*"
        print(''.join(sig))

    print()

    m, n = 2, 4
    space = SymplecticSpace(n)

    sigs = set()
    for C in space.grassmannian(m):
        C = C * uturn_to_zip(n)
        C2 = C.reshape(m, n, 2)
        Ct = C.t
        sig = []
        for algebra in found:
            for a in algebra:
                D = C2*a
                D = D.reshape(m, 2*n)
                u = Ct.solve(D.t)
                if u is None:
                    sig.append(".")
                    break
            else:
                sig.append("*")
        sig = ''.join(sig)
        sigs.add(''.join(sig))

        if sig == "*......*....":
            code = QCode(C)
            print(code)
            assert code.is_gf4()
            #print(code.longstr())
            print(strop(code.H))
            print()

    sigs = list(sigs)
    sigs.sort()
    for sig in sigs:
        print(sig)
    

@cache
def uturn_to_zip(n):
    nn = 2*n
    U = numpy.zeros((nn, nn), dtype=scalar)
    #print(U)
    for i in range(n):
        U[2*i, i] = 1
        #print(U)
        U[2*i+1, 2*n-i-1] = 1
        #print(U)
    U = Matrix(U)
    return U.t


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




