#!/usr/bin/env python3

"""
matrix groups over Z/pZ.

"""


from random import shuffle, choice
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul
from math import prod

import numpy

from qumba.solve import (shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2,
    kernel, span, pseudo_inverse, rank, row_reduce, linear_independent)
from qumba.solve import int_scalar as scalar
from qumba import solve
from qumba.action import mulclose
from qumba.decode.network import TensorNetwork


DEFAULT_P = 2 # qubits


def flatten(H):
    if H is not None and len(H.shape)==3:
        H = H.view()
        m, n, _ = H.shape
        H.shape = m, 2*n
    return H


def complement(H):
    H = flatten(H)
    H = row_reduce(H)
    m, nn = H.shape
    #print(shortstr(H))
    pivots = []
    row = col = 0
    while row < m:
        while col < nn and H[row, col] == 0:
            #print(row, col, H[row, col])
            pivots.append(col)
            col += 1
        row += 1
        col += 1
    while col < nn:
        pivots.append(col)
        col += 1
    W = zeros2(len(pivots), nn)
    for i, ii in enumerate(pivots):
        W[i, ii] = 1
    #print()
    return W




class Matrix(object):
    def __init__(self, A, p=DEFAULT_P, shape=None, name="?"):
        if type(A) == list or type(A) == tuple:
            A = numpy.array(A, dtype=scalar)
        elif isinstance(A, Matrix):
            A, p = A.A, A.p
        elif isinstance(A, numpy.ndarray):
            A = A.astype(scalar) # will always make a copy
        else:
            raise TypeError( "whats this: %s"%(type(A)) )
        A = flatten(A)
        if shape is not None:
            A.shape = shape
        self.A = A
        assert int(p) == p
        assert p>=0
        self.p = p
        if p>0:
            self.A %= p
        self.key = (self.p, self.A.tobytes())
        self._hash = hash(self.key)
        self.shape = A.shape
        #assert name != "?"
        assert name != ""
        if type(name) is str:
            name = name,
        self.name = name

    @classmethod
    def promote(cls, item, p=DEFAULT_P, name="?"):
        if item is None:
            return None
        if isinstance(item, Matrix):
            return item
        return Matrix(item, p, name=name)

    def reshape(self, *shape):
        A = self.A
        A = A.reshape(*shape)
        return Matrix(A)

    @classmethod
    def perm(cls, items, p=DEFAULT_P, name=None):
        n = len(items)
        A = numpy.zeros((n, n), dtype=scalar)
        for i, ii in enumerate(items):
            A[ii, i] = 1
        if name is None:
            name = "P"+str(tuple(items))
        return Matrix(A, p, name=name)

    def to_perm(self):
        A = self.A
        perm = []
        for row in A:
            idx = numpy.where(row)[0]
            assert len(idx) == 1, "not a perm"
            perm.append(idx[0])
        return perm

    @classmethod
    def identity(cls, n, p=DEFAULT_P):
        A = numpy.identity(n, dtype=scalar)
        return Matrix(A, p, name="I")

    def shortstr(self):
        return str(self.A).replace("0", ".")
        # XXX broken:
        #s = shortstr(self.A)
        #lines = s.split()
        #lines = [" ["+line+"]" for line in lines]
        #lines[0] = "["+lines[0][1:]
        #lines[-1] = lines[-1] + "]"
        #return '\n'.join(lines)
    __str__ = shortstr

    def latex(self):
        A = self.A
        rows = ['&'.join('.1'[i] for i in row) for row in A]
        rows = r'\\'.join(rows)
        rows = r"\begin{bmatrix}%s\end{bmatrix}"%rows
        return rows

    def __repr__(self):
        return "Matrix(%s)"%str(self.A)

    def shortstr(self):
        return shortstr(self.A)

    def __hash__(self):
        return self._hash

    def is_zero(self):
        return self.A.sum() == 0

    def __len__(self):
        return len(self.A)

    def __eq__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        return self.key == other.key

    def __ne__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        return self.key != other.key

    def __lt__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        return self.key < other.key

    def __add__(self, other):
        assert self.p == other.p
        A = self.A + other.A
        return Matrix(A, self.p)

    def __sub__(self, other):
        assert self.p == other.p
        assert self.shape == other.shape
        A = self.A - other.A
        return Matrix(A, self.p)

    def __neg__(self):
        A = -self.A
        return Matrix(A, self.p)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            assert self.p == other.p
            A = numpy.dot(self.A, other.A)
            if A.shape == ():
                return A%self.p
            return Matrix(A, self.p, name=self.name+other.name)
        else:
            return NotImplemented

    def __rmul__(self, r):
        A = r*self.A
        return Matrix(A, self.p)

    def __matmul__(self, other):
        A = numpy.kron(self.A, other.A)
        return Matrix(A)

    def direct_sum(self, other):
        "direct_sum"
        A = direct_sum(self.A, other.A)
        return Matrix(A, self.p)
    #__add__ = direct_sum # ??
    #__lshift__ = direct_sum # ???

    def __getitem__(self, idx):
        A = self.A[idx]
        #print("__getitem__", idx, type(A))
        if type(A) is scalar:
            return A
        return Matrix(A, self.p)

    def transpose(self):
        A = self.A
        name = self.name
        names = []
        for n in reversed(name):
            assert len(n), name
            if n.endswith(".t"):
                names.append(n[:-2])
            elif n[0] == "H":
                names.append(n) # symmetric
            else:
                names.append(n+".t")
        #name = tuple(n[:-2] if n.endswith(".t") else n+".t" for n in reversed(name))
        name = tuple(names)
        return Matrix(A.transpose(), self.p, None, name)

    @property
    def t(self):
        return self.transpose()

    def sum(self, axis=None):
        A = self.A
        B = A.sum(axis=axis, dtype=numpy.int64)
        if axis is None:
            return B
        return Matrix(B) # ??

    def kernel(self):
        K = kernel(self.A)
        K = Matrix(K, self.p)
        return K

    def pseudo_inverse(self):
        A = pseudo_inverse(self.A)
        return Matrix(A)

    def where(self):
        return list(zip(*numpy.where(self.A))) # list ?

    def concatenate(self, *others, axis=0):
        A = numpy.concatenate((self.A,)+tuple(other.A for other in others), axis=axis)
        return Matrix(A, self.p)

    def max(self):
        return self.A.max()

    def min(self):
        return self.A.min()

    def copy(self):
        return Matrix(self.A, self.p)

    def rowspan(self):
        m, n = self.A.shape
        for u in span(self.A):
            u.shape = 1, n
            u = Matrix(u)
            yield u

    def rank(self):
        return rank(self.A)

    def row_reduce(self):
        A = row_reduce(self.A)
        return Matrix(A)

    def linear_independent(self):
        A = linear_independent(self.A)
        return Matrix(A)

    def get_projector(A):
        "project onto the colspace"
        P = A*A.pseudo_inverse()
        return P

#    def reshape(self, shape):
#        A = self.A.view()
#        A.shape = shape
#        return Matrix(A)

    def to_spider(self, scalar=int, verbose=True):
        from qumba.decode import network
        scalar = numpy.int8
        network.scalar = scalar
        from qumba.decode.network import green, red, TensorNetwork
        S = self.A
        m, n = S.shape
        net = TensorNetwork()
    
        for j in range(n):
            A = green(S[:, j].sum(), 1, scalar)
            links = [(i, j) for i in range(m) if S[i, j]] + [("*", j)]
            net.append(A, links)
    
        for i in range(m):
            A = red(1, S[i, :].sum(), scalar)
            links = [(i, "*")] + [(i, j) for j in range(n) if S[i, j]]
            net.append(A, links)
    
        idxs = list(zip(*numpy.where(S)))
        while idxs and len(net) > 1:
            #print("net:")
            #for (A, links) in zip(net.As, net.linkss):
            #    print("\t", links, len(links), len(set(links)))
            size = {link : 1 for link in net.get_links()}
            for (A, links) in zip(net.As, net.linkss):
                s = reduce(mul, A.shape)
                for link in links:
                    size[link] *= s
            idxs.sort( key = lambda link : size.get(link, 0) )
            if not idxs:
                break
            link = idxs.pop()
            #print(size)
            #print("contract at", link, size.get(link, 0))
            pair = []
            for (i, (A, links)) in enumerate(zip(net.As, net.linkss)):
                if link in links:
                    pair.append(i)
                if len(pair) == 2:
                    break
            else:
            #    print("no pair left")
                break
            net.contract_pair(*pair)
        #print("done contract")
    
        assert len(net)
        A = reduce((lambda a,b:numpy.tensordot(a,b,axes=([],[]))), net.As)
        links = reduce(add, net.linkss)
        #print(A.shape, links)
        assert len(A.shape) == len(links)
        E = numpy.zeros((2**m, 2**n), dtype=int)
        tgt = [(i, "*") for i in range(m)]+[("*", j) for j in range(n)]
        #for (A, links) in net:
        #print(A.shape, links)
        idxs = tuple(tgt.index(link) for link in links)
        #print("idxs:", idxs)
        jdxs = [None]*len(idxs)
        for i,idx in enumerate(idxs):
            jdxs[idx] = i
        #print("jdxs:", jdxs)
        E = A.transpose(jdxs)
        E = E.astype(int)
        E = E.reshape(2**m, 2**n)
        return E


def pushout(j, k, j1=None, k1=None):
    assert j.shape[1] == k.shape[1]
    J = j.A
    K = k.A
    if j1 is not None:
        J1 = j1.A
        K1 = k1.A
        JJ, KK, F = solve.pushout(J, K, J1, K1)
        jj = Matrix(JJ)
        kk = Matrix(KK)
        f = Matrix(F)
        return jj, kk, f

    else:
        JJ, KK = solve.pushout(J, K)
        jj = Matrix(JJ)
        kk = Matrix(KK)
        return jj, kk


def pullback(j, k, j1=None, k1=None):
    assert j.shape[0] == k.shape[0]
    if j1 is not None:
        jj, kk, f = pushout(j.t, k.t, j1.t, k1.t)
        return jj.t, kk.t, f.t

    else:
        jj, kk = pushout(j.t, k.t)
        return jj.t, kk.t




def test():
    I = Matrix.identity(5)
    assert I*I == I




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


