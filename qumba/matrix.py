#!/usr/bin/env python3

"""
matrix groups over Z/pZ.

"""


from random import shuffle, choice
from functools import reduce
from operator import add, mul
from math import prod

import numpy

from qumba.default_p import DEFAULT_P
from qumba.lin import (shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2,
    kernel, span, pseudo_inverse, rank, row_reduce, linear_independent, rand2, parse,
    normal_form,)
from qumba.lin import int_scalar as scalar
from qumba import lin
from qumba.action import mulclose
from qumba.decode.network import TensorNetwork


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
        if shape is not None:
            A.shape = shape
        self.A = A
        assert int(p) == p
        assert p>=0
        self.p = p
        if p>0:
            self.A %= p
        self.shape = A.shape
        self.key = (self.p, self.shape, self.A.tobytes())
        self._hash = hash(self.key)
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

    @classmethod
    def parse(cls, desc):
        A = parse(desc)
        return cls(A)

    def reshape(self, *shape):
        A = self.A
        A = A.reshape(*shape)
        return Matrix(A)

    @classmethod
    def einsum(cls, desc, *args):
        A = numpy.einsum(desc, *[M.A for M in args])
        return Matrix(A).reshape(*A.shape)

    @classmethod
    def perm(cls, items, p=DEFAULT_P, name=None):
        n = len(items)
        A = numpy.zeros((n, n), dtype=scalar)
        for i, ii in enumerate(items):
            A[ii, i] = 1
        if name is None:
            name = "P"+str(tuple(items))
        return Matrix(A, p, name=name)
    get_perm = perm

    @classmethod
    def rand(cls, m, n):
        A = rand2(m, n)
        return Matrix(A)

    def to_perm(self):
        A = self.A
        perm = []
        for row in A:
            idx = numpy.where(row)[0]
            assert len(idx) == 1, "not a perm"
            perm.append(int(idx[0]))
        return perm

    @classmethod
    def identity(cls, n, p=DEFAULT_P):
        A = numpy.identity(n, dtype=scalar)
        return Matrix(A, p, name="I")
    get_identity = identity

    @classmethod
    def zeros(cls, shape, p=DEFAULT_P):
        A = numpy.zeros(shape, dtype=scalar)
        return Matrix(A, p, name="0")

    def __str__(self):
        if len(self.shape) <= 2:
            return shortstr(self.A)
        else:
            return str(self.A)
        #return str(self.A).replace("0", ".")
        # XXX broken:
        #s = shortstr(self.A)
        #lines = s.split()
        #lines = [" ["+line+"]" for line in lines]
        #lines[0] = "["+lines[0][1:]
        #lines[-1] = lines[-1] + "]"
        #return '\n'.join(lines)

    def latex(self):
        A = self.A
        rows = ['&'.join('.123456'[i] for i in row) for row in A]
        rows = r'\\'.join(rows)
        rows = r"\begin{bmatrix}%s\end{bmatrix}"%rows
        return rows

    def gap(self):
        m, n = self.shape
        matrix = []
        for row in self.A:
            line = []
            for x in row:
                line.append('Z(2)' if x else '0*Z(2)')
            line = "[%s]"%(','.join(line))
            matrix.append(line)
        matrix = "[%s]"%(','.join(matrix))
        return str(matrix)

    def __repr__(self):
        return "Matrix(%s)"%str(self.A)

    def shortstr(self):
        return shortstr(self.A)

    def __hash__(self):
        return self._hash

    def is_zero(self):
        return self.A.sum() == 0

    def is_identity(self):
        return self == Matrix.identity(len(self))

    def order(self):
        n = 1
        A = self
        I = Matrix.identity(len(self))
        while A != I:
            n += 1
            A *= self
        return n

    def __len__(self):
        return len(self.A)

    def __eq__(self, other):
        assert self.p == other.p
        #assert self.shape == other.shape # too strict
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
        if isinstance(other, Matrix):
            A = numpy.kron(self.A, other.A)
            return Matrix(A, self.p)
        else:
            return NotImplemented

    def __pow__(self, n):
        assert n>=0
        A = Matrix.identity(len(self))
        for i in range(n):
            A = self*A
        return A

    def direct_sum(self, other):
        "direct_sum"
        other = Matrix.promote(other)
        A = direct_sum(self.A, other.A)
        return Matrix(A, self.p)
    __lshift__ = direct_sum # ???
    #__add__ = direct_sum # ??

    def __getitem__(self, idx):
        A = self.A[idx]
        #print("__getitem__", idx, type(A))
        if type(A) is scalar:
            return A
        return Matrix(A, self.p)

    def transpose(self, *arg):
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
        return Matrix(A.transpose(*arg), self.p, None, name)

    @property
    def t(self):
        return self.transpose()

    def sum(self, axis=None):
        A = self.A
        B = A.sum(axis=axis, dtype=numpy.int64)
        #if axis is None:
        #    return B
        #return Matrix(B) # ??
        return B

    def kernel(self):
        K = kernel(self.A)
        K = Matrix(K, self.p)
        return K

    def pseudo_inverse(self):
        A = pseudo_inverse(self.A)
        return Matrix(A)
    __invert__ = pseudo_inverse

    def solve(self, other):
        A = lin.solve(self.A, other.A)
        return Matrix(A) if A is not None else None

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
        assert self.p == 2, "not implemented"
        return rank(self.A)

    def row_reduce(self):
        assert self.p == 2, "not implemented"
        A = row_reduce(self.A)
        return Matrix(A)

    def normal_form(self, truncate=True):
        assert self.p == 2, "not implemented"
        A = normal_form(self.A, truncate)
        return Matrix(A)

    def linear_independent(self):
        assert self.p == 2, "not implemented"
        A = linear_independent(self.A)
        return Matrix(A)

    def intersect(self, other):
        assert self.p == 2, "not implemented"
        A = intersect(self.A, other.A)
        return Matrix(A)

    def get_projector(A):
        "project onto the colspace"
        P = A*A.pseudo_inverse()
        return P

    def puncture(A, i):
        A0 = A[:, :i]
        A1 = A[:, i+1:]
        return A0.concatenate(A1, axis=1)

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
            #print("_contract at", link, size.get(link, 0))
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
        #print("done _contract")
    
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

    def get_wenum(H):
        m, n = H.shape
        assert m < 32, ("%s is too big ??" % (2**m))
        A = H.A
        wenum = {i:0 for i in range(n+1)}
        for bits in numpy.ndindex((2,)*m):
            #v = Matrix(bits)*H
            v = numpy.dot(bits, A)%2
            wenum[v.sum()] += 1
        return tuple(wenum[i] for i in range(n+1))


    def get_pivots(H):
        "return pivots of a normal_form"
        m, nn = H.shape
        pivots = []
        row = col = 0
        while row < m:
            while col < nn and H[row, col] == 0:
                col += 1
            if col < nn:
                pivots.append((row,col))
            row += 1
            col += 1
        return pivots

    def get_components(H):
        return get_components(H)


def get_components(H): # dumb & slow XXX
    #print("get_components")
    H = H.normal_form()

    found = set()
    m, n = H.shape
    for r0 in range(m):
        h = H[r0]
        cols = set(c for c in range(n) if H[r0,c])
        done = False
        while not done:
            #print("cols", cols)
            done = True
            for col in list(cols):
                for r1 in range(m):
                    if H[r1,col] == 0:
                        continue
                    for c in range(n):
                        if H[r1, c] and c not in cols:
                            cols.add(c)
                            done = False
        #print("done")
        cmp = list(cols)
        cmp.sort()
        cmp = tuple(cmp)
        if cmp not in found:
            H1 = H[:, list(cmp)]
            H1 = H1.row_reduce()
            yield H1
            found.add(cmp)
        
    




def pushout(j, k, j1=None, k1=None):
    assert j.shape[1] == k.shape[1]
    J = j.A
    K = k.A
    if j1 is not None:
        J1 = j1.A
        K1 = k1.A
        JJ, KK, F = lin.pushout(J, K, J1, K1)
        jj = Matrix(JJ)
        kk = Matrix(KK)
        f = Matrix(F)
        return jj, kk, f

    else:
        JJ, KK = lin.pushout(J, K)
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


