#!/usr/bin/env python

"""
Bruhat-Tits building & Bruhat decomposition 

ported from: 
https://github.com/punkdit/bruhat/blob/master/bruhat/algebraic.py

"""



from random import shuffle, choice
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

from qumba.solve import (shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2,
    kernel, span)
from qumba.solve import int_scalar as scalar
from qumba.action import mulclose
from qumba.matrix import Matrix, DEFAULT_P
from qumba.argv import argv

def isprime(n):
    assert 0<n<10 
    return n in [2, 3, 5, 7]


# https://math.stackexchange.com/questions/34271/
# order-of-general-and-special-linear-groups-over-finite-fields

def order_gl(n, q):
    order = 1
    for i in range(n):
        order *= (q**n - q**i)
    return order

def order_sl(n, q):
    order = order_gl(n, q)
    assert order % (q-1) == 0
    return order//(q-1)


def order_sp(n, q):
    # n = 2*m
    assert n%2==0
    m = n//2
    N = q**(m**2)
    for i in range(m):
        N *= (q**(2*(i+1)) - 1)
    return N

assert order_sp(2, 2)==6     # 3!
assert order_sp(4, 2)==720   # 6!




class Algebraic(object):
    "Algebraic Group"
    def __init__(self, gen, order=None, p=DEFAULT_P, G=None, verbose=False, **kw):
        self.__dict__.update(kw)
        self.gen = list(gen)
        if G is not None:
            assert order is None or order==len(G)
            order = len(G)
        self.order = order
        self.G = G # elements
        self.p = p
        assert gen
        A = gen[0]
        self.n = len(A)
        assert p == A.p
        self.I = Matrix.identity(self.n, p)
        self.verbose = verbose

    def get_elements(self):
        if self.G is None:
            I = self.I
            G = mulclose(self.gen, maxsize=self.order, verbose=self.verbose)
            G.remove(I)
            G.add(I)
            G = list(G)
            self.G = G
            self.order = len(self.G)
        return self.G

    def sample(self):
        gen = self.gen
        N = 10*len(gen)
        A = choice(gen)
        for i in range(N):
            A = A*choice(gen)
        return A

    def __len__(self):
        if self.order is None:
            self.get_elements()
        return self.order

    def __getitem__(self, idx):
        if self.G is None:
            self.get_elements()
        return self.G[idx]

    def __contains__(self, g):
        return g in self.get_elements()

    def __eq__(self, other):
        return set(self.get_elements()) == set(other.get_elements())

    @classmethod
    def Sp_4_2(cls, F, **kw):
        A = numpy.array([[1,0,1,1],[1,0,0,1],[0,1,0,1],[1,1,1,1]], dtype=scalar)
        B = numpy.array([[0,0,1,0],[1,0,0,0],[0,0,0,1],[0,1,0,0]], dtype=scalar)
        gen = [Matrix(A, 2), Matrix(B, 2)]
        return Sp(gen, 720, p=2, invariant_form=F, **kw)

    @classmethod
    def Sp(cls, n, p=DEFAULT_P, **kw):
        gen = []
        assert n%2 == 0

        assert isprime(p)
        F = numpy.zeros((n, n), dtype=scalar)
        for i in range(n//2):
            F[i, n-i-1] = 1
            F[i + n//2, n//2-i-1] = p-1
        F = Matrix(F, p)

        for i in range(1, p):
            vals = set((i**k)%p for k in range(p+1))
            if len(vals)==p-1:
                fgen = i # generates GL(1, p)
                break
        else:
            assert 0
        for i in range(1, p):
            if (i*fgen)%p == 1:
                ifgen = i
                break
        else:
            assert 0

        if n==2:
            G = Sp.SL(2, p, invariant_form=F, **kw)
            return G
        if n==4 and p==2:
            return Sp.Sp_4_2(F, **kw)

        if p==2:
            m = n//2
            A = numpy.zeros((n, n), dtype=scalar)
            B = numpy.zeros((n, n), dtype=scalar)
            for i in range(n):
                A[i, i] = 1
            A[0, m-1] = 1
            A[0, n-1] = 1
            A[m, n-1] = 1
            for i in range(m-1):
                B[i+1, i] = 1
                B[i+m, i+m+1] = 1
            B[0, m] = 1
            B[n-1, m-1] = 1
            gen = [Matrix(A, 2), Matrix(B, 2)]
        else:
            m = n//2
            A = numpy.zeros((n, n), dtype=scalar)
            B = numpy.zeros((n, n), dtype=scalar)
            for i in range(n):
                A[i, i] = 1
            A[0, 0] = fgen
            A[n-1, n-1] = ifgen

            for i in range(m-1):
                B[i+1, i] = 1
                B[m+i, m+i+1] = 1
            B[0, 0] = 1
            B[0, m] = 1
            B[n-2, m-1] = 1
            B[n-1, m-1] = ifgen

            gen = [Matrix(A, p), Matrix(B, p)]

        G = Sp(gen, order_sp(n, p), p=p, invariant_form=F, **kw)
        return G






class GL(Algebraic):

    def get_weyl(self):
        n = self.n
        items = list(range(n))
        gen = []
        for ii in range(n-1):
            perm = list(range(n))
            perm[ii:ii+2] = perm[ii+1], perm[ii]
            M = Matrix.perm(perm, self.p, name="w%d"%ii)
            gen.append(M)
            #print(M.name)
            #print(M.shortstr())
        return Algebraic(gen, p=self.p)


class Sp(Algebraic):

    def get_pairs(self):
        n = self.n//2
        pairs = [(i, 2*n-i-1) for i in range(n)]
        return pairs

    def get_blocks(self, M):
        pairs = self.get_pairs()
        A = M.A[:, [pair[0] for pair in pairs]]
        B = M.A[:, [pair[1] for pair in pairs]]
        #A = Matrix(A, self.p)
        #B = Matrix(B, self.p)
        return A, B

    def from_blocks(self, A, B):
        n = self.n//2
        m = len(A)
        assert A.shape == (m, n)
        assert B.shape == (m, n)
        M = numpy.zeros((m, 2*n))
        pairs = self.get_pairs()
        M[:, [pair[0] for pair in pairs]] = A
        M[:, [pair[1] for pair in pairs]] = B
        #M = numpy.concatenate((A, B), axis=1)
        #assert M.shape == (m, 2*n)
        #print(M)
        M = Matrix(M, self.p)
        return M

    def is_isotropic(self, M):
        F = self.invariant_form
        MM = M*F*M.transpose()
        return MM.is_zero()

    def get_weyl(self):
        nn = self.n
        pairs = self.get_pairs()
        k = len(pairs)
        gen = []
        for ii in range(k-1):
            idxs = list(range(k))
            idxs[ii:ii+2] = idxs[ii+1], idxs[ii]
            qairs = [pairs[idx] for idx in idxs]
            src = reduce(add, pairs)
            tgt = reduce(add, qairs)
            #print(src, "-->", tgt)
            perm = list(range(nn))
            for i, j in zip(src, tgt):
                perm[i] = j
            #print('\t', perm)
            M = Matrix.perm(perm, self.p, name="w%d"%ii)
            gen.append(M)
            #print(M.name)
            #print(M.shortstr())
        perm = list(range(nn))
        a, b = pairs[-1]
        perm[a], perm[b] = perm[b], perm[a]
        M = Matrix.perm(perm, self.p, name="w%d"%(k-1))
        gen.append(M)
        #print(M.name)
        #print(M.shortstr())
        #return self.get_all_weyl()
        #print(len(gen))
        #print(len(mulclose(gen)))
        return Algebraic(gen, p=self.p)

    def get_all_weyl(self):
        nn = self.n
        pairs = self.get_pairs()
        k = len(pairs)
        I = Matrix.identity(nn, self.p)
        flips = []
        for (a, b) in pairs:
            perm = list(range(nn))
            perm[a], perm[b] = b, a
            M = Matrix.perm(perm)
            flips.append((I, M))
        W = []
        for qairs in allperms(pairs):
            perm = [None]*nn
            for src, tgt in zip(pairs, qairs):
                perm[src[0]] = tgt[0]
                perm[src[1]] = tgt[1]
            assert None not in perm
            M = Matrix.perm(perm, self.p)
            for items in cross(flips):
                N = reduce(mul, items)
                W.append(M*N)
        return W

    def get_borel(self):
        F = self.invariant_form
        pairs = self.get_pairs()
        gen = []
        lookup = {}
        for (i, j) in pairs:
            assert i<j
            A = numpy.identity(self.n, dtype=scalar)
            A[i, j] = 1
            M = Matrix(A, self.p, name="E%d%d"%(j,i))
            gen.append(M)
            lookup[j, i] = M

        k = len(pairs)
        for i in range(k):
          for j in range(i+1, k):

            a, b = pairs[i]
            c, d = pairs[j]
            assert d < b
            assert a < c

            A = numpy.identity(self.n, dtype=scalar)
            A[a, c] = 1
            A[d, b] = self.p-1

            M = Matrix(A, self.p, name="E%d%d"%(b,d))
            gen.append(M)
            lookup[b, d] = M # b>d
            assert (c, a) not in lookup
            lookup[c, a] = M

            # we don't need these gen's but we need the lookup
            c, d = d, c
            assert d < b
            assert a < c
            A = numpy.identity(self.n, dtype=scalar)
            A[a, c] = 1
            A[d, b] = 1

            M = Matrix(A, self.p, name="E%d%d"%(b,d))
            gen.append(M)
            lookup[b, d] = M
            assert (c, a) not in lookup
            lookup[c, a] = M

        for M in gen:
            assert M*F*M.transpose() == F

        B = Algebraic(gen, p=self.p)
        B.lookup = lookup
        return B


class Building(object):
    def __init__(self, G):
        self.G = G
        self.nn = G.n
        self.W = G.get_weyl()
        self.B = G.get_borel()

    def decompose(self, g):
        nn = self.nn
        n = nn//2
        lookup = self.B.lookup
        b1 = b2 = self.G.I
        src = nn-1
        pivots = [] # cols
        while src >= 0:
            for col in range(nn):
                if g[src, col]:
                    break
            else:
                assert 0
            pivots.append((src, col))
            tgt = src-1
            while tgt >= 0:
                if g[tgt, col]:
                    g = lookup[src, tgt]*g
                    b1 = lookup[src, tgt]*b1
                tgt -= 1
            src -= 1

        for row, src in pivots:
            for tgt in range(src+1, nn):
                if g[row, tgt]==0:
                    continue
                b = lookup.get((tgt, src))
                if b is not None:
                    g = g*b
                    b2 = b2*b
        return b1, b2






def test_building():
    n = argv.get("n", 3)
    nn = 2*n
    p = argv.get("p", 2)
    G = Algebraic.Sp(nn, p)
    I = G.I
    F = G.invariant_form
    N = len(G)
    building = Building(G)

    print("|G| =", N)

    W = building.W
    print("|W| =", len(W))

    B = building.B
    print("|B| =", len(B))

    if argv.slow:
        found = {}
        for b1 in B:
         for w in W:
          b1w = b1*w
          for b2 in B:
            g = b1w*b2
            path = found.get(g)
            if path is None:
                found[g] = b1, w, b2
                continue
            b11, ww, b22 = path
            lhs = (len(b1.name), len(b2.name))
            rhs = (len(b11.name), len(b22.name))
            if lhs < rhs:
                found[g] = b1, w, b2
        assert len(found) == len(G)
    else:
        found = None

    lookup = B.lookup
    for trial in range(3):
        print()
        g = G.sample()
        print(g)
        b1, b2 = building.decompose(g)
        w = b1*g*b2
        assert b1 in B
        assert b2 in B
        assert w in W
        print("b1 =")
        print(b1)
        print("w =")
        print(w)
        print("b2 =")
        print(b2)
        if found is not None:
            assert w == found[g][1]


def main():
    pass


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








