#!/usr/bin/env python
"""
stabilizer codes & symplectic geometry
"""

import string, os
from random import randint, choice, random
from time import sleep, time
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add
import json

import numpy
from numpy import alltrue, zeros, dot, concatenate

from qumba import solve 
from qumba.solve import (
    array2, zeros2, shortstr, dot2, solve2, linear_independent, row_reduce, kernel,
    span, intersect, rank, enum2, shortstrx, identity2, eq2, pseudo_inverse, direct_sum)
from qumba.argv import argv
from qumba.csscode import CSSCode
from qumba.smap import SMap


def parse(s):
    for c in "XZY":
        s = s.replace(c, '1')
    s = s.replace("I", "0")
    for c in " [],":
        s = s.replace(c, '')
    return solve.parse(s)


def fromstr(Hs):
    if type(Hs) is str:
        stabs = Hs.split()
    else:
        stabs = list(Hs)
    H = []
    I, X, Z, Y = [0,0], [1,0], [0,1], [1,1]
    lookup = {'X':X, 'Z':Z, 'Y':Y, 'I':I, '.':I}
    for stab in stabs:
        row = [lookup[c] for c in stab]
        H.append(row)
    H = array2(H)
    H = flatten(H)
    return H


def css_to_symplectic(Lx, Lz):
    assert Lz.shape == Lx.shape
    m, n = Lx.shape
    L = zeros2(2*m, n, 2)
    L[0::2, :, 0] = Lx
    L[1::2, :, 1] = Lz
    return L


def css_to_isotropic(Hx, Hz):
    mx, n = Hx.shape
    mz, n1 = Hz.shape
    assert n==n1
    H = zeros2(mx+mz, n, 2)
    H[:mx, :, 0] = Hx
    H[mx:, :, 1] = Hz
    return H


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

def get_weight_slow(v):
    count = 0
    for i in range(len(v)//2):
      if v[2*i] or v[2*i+1]:
        count += 1
    c1 = get_weight_fast(v)
    if count != c1:
        print("v =")
        print(v)
        print(count, c1)
        assert 0
    return count

def get_weight_fast(v): # not much faster for n=18
    n = len(v)//2
    v.shape = n,2
    w = v[:,0] + v[:,1]
    v.shape = 2*n,
    return numpy.count_nonzero(w)

get_weight = get_weight_fast


def monte_carlo(H, v, p=0.5, trials=10000):
    H = H.view()
    if len(H.shape) == 3:
        m, n, o = H.shape
        assert o==2
        nn = 2*n
        H.shape = m, nn
    else:
        m, nn = H.shape
    assert v.shape == (nn,), v.shape
    d0 = get_weight_fast(v)
    #print("[",d0, end=",", flush=True)
    p0 = p**d0
    #randint = numpy.random.randint
    for trial in range(trials):
        #u = randint(2, size=m)
        #h = dot2(u, H)
        i = randint(0, m-1)
        h = H[i]
        w = (v+h)%2
        d1 = get_weight_fast(w)
        p1 = p**d1
        a = random()
        if (p0/p1) < a:
            v = w
            d0 = d1
            p0 = p**d0
            #print(d0, end=",", flush=True)
    #print("]")
    return d0

def strop(H):
    assert len(H.shape) == 2, H.shape
    smap = SMap()
    m, nn = H.shape
    for i in range(m):
      for j in range(nn//2):
        x, z = H[i, 2*j:2*j+2]
        c = '.'
        if x and z:
            c = 'Y'
        elif x:
            c = 'X'
        elif z:
            c = 'Z'
        smap[i,j] = c
    return str(smap)

@cache
def symplectic_form(n):
    F = zeros2(2*n, 2*n)
    for i in range(n):
        F[2*i:2*i+2, 2*i:2*i+2] = [[0,1],[1,0]]
    return F


class SymplecticSpace(object):
    def __init__(self, n):
        assert 0<=n
        self.n = n
        self.F = symplectic_form(n)

    def is_symplectic(self, M):
        nn = 2*self.n
        F = self.F
        assert M.shape == (nn, nn)
        return eq2(F, dot2(M, F, M.transpose()))

    def get_perm(self, f):
        n, nn = self.n, 2*self.n
        assert len(f) == n
        assert set([f[i] for i in range(n)]) == set(range(n))
        A = zeros2(nn, nn)
        for i in range(n):
            A[2*i, 2*f[i]] = 1
            A[2*i+1, 2*f[i]+1] = 1
        assert self.is_symplectic(A)
        return A

    def get(self, M, idx=None):
        M = array2(M)
        assert M.shape == (2,2)
        n = self.n
        A = identity2(2*n)
        idxs = list(range(n)) if idx is None else [idx]
        for i in idxs:
            A[2*i:2*i+2, 2*i:2*i+2] = M
        return A.transpose()

    def get_H(self, idx=None):
        # swap X<-->Z on bit idx
        H = array2([[0,1],[1,0]])
        return self.get(H, idx)

    def get_S(self, idx=None):
        # swap X<-->Y
        S = array2([[1,1],[0,1]])
        return self.get(S, idx)

    def get_SH(self, idx=None):
        # X-->Z-->Y-->X 
        SH = array2([[0,1],[1,1]])
        return self.get(SH, idx)

    def get_CZ(self, idx, jdx):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*idx, 2*jdx+1] = 1
        A[2*jdx, 2*idx+1] = 1
        return A.transpose()

    def get_CNOT(self, idx, jdx):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*jdx+1, 2*idx+1] = 1
        A[2*idx, 2*jdx] = 1
        return A.transpose()


class QCode(object):
    def __init__(self, H, T=None, L=None, J=None, d=None, check=True):
        assert len(H)==0 or H.max() <= 1
        H = flatten(H) # stabilizers
        m, nn = H.shape
        assert nn%2 == 0
        n = nn//2
        T = flatten(T)
        L = flatten(L)
        J = flatten(J)
        self.H = H # stabilizers
        self.T = T # destabilizers
        self.L = L # logicals
        self.J = J # gauge generators
        self.m = m
        self.n = n
        self.nn = nn
        self.k = n - m # number of logicals
        self.kk = 2*self.k
        self.d = d
        self.shape = m, n
        self.space = SymplecticSpace(n)
        if L is not None:
            assert L.shape == (2*self.k, nn)
        if T is not None:
            assert T.shape == (self.m, nn)
        if check:
            self.check()

    def __eq__(self, other):
        assert isinstance(other, QCode)
        eq = lambda A, B : (A is None and B is None) or A is not None and B is not None and eq2(A, B)
        return (
            eq(self.H, other.H) and eq(self.T, other.T) and
            eq(self.L, other.L) and eq(self.J, other.J))

    def __hash__(self):
        A = self.get_symplectic()
        key = A.tobytes(), self.m
        return hash(key)

    def __add__(self, other):
        H = direct_sum(self.H, other.H)
        T = direct_sum(self.T, other.T)
        L = direct_sum(self.L, other.L)
        return QCode(H, T, L)

    def __rmul__(self, count):
        assert type(count) is int
        assert count>=0
        code = reduce(add, [self]*count)
        return code

    @classmethod
    def build_css(cls, Hx, Hz, Tx=None, Tz=None, Lx=None, Lz=None, Jx=None, Jz=None, **kw):
        H = css_to_isotropic(Hx, Hz)
        #T = css_to_isotropic(Tx, Tz) if Tx is not None else None # FAIL
        T = None
        L = css_to_symplectic(Lx, Lz) if Lx is not None else None
        J = css_to_symplectic(Jx, Jz) if Jx is not None else None
        return QCode(H, T, L, J, **kw)

    @classmethod
    def build_gauge(cls, J, **kw):
        m, n, _ = J.shape
        J1 = J.copy()
        J1[:, :, :] = J1[:, :, [1,0]]
        J1.shape = (m, 2*n)
        K = kernel(J1)
        #print("K:", K.shape)
        #print(shortstr(K))
        J = J.view()
        J.shape = m, 2*n
        H = intersect(J, K)
        #print("H:", H.shape)
        #H.shape = len(H), n, 2
        #J.shape = len(J), n, 2
        code = QCode(H, None, None, J, **kw)
        return code

    @classmethod
    def build_gauge_css(cls, Jx, Jz, **kw):
        J = css_to_isotropic(Jx, Jz)
        code = cls.build_gauge(J, **kw)
        return code

    @classmethod
    def fromstr(cls, Hs, Ts=None, Ls=None, Js=None, check=True, **kw):
        H = fromstr(Hs) if Hs is not None else None
        T = fromstr(Ts) if Ts is not None else None
        L = fromstr(Ls) if Ls is not None else None
        J = fromstr(Js) if Js is not None else None
        if H is None and J is not None:
            code = QCode.build_gauge(J, T=T, L=L)
        else:
            code = QCode(H, T, L, J, check=check, **kw)
        return code

    def __str__(self):
        d = self.d if self.d is not None else '?'
        return "[[%s, %s, %s]]"%(self.n, self.k, d)

    def longstr(self):
        s = "H =\n%s"%strop(self.H)
        T = self.T
        if T is not None and len(T):
            s += "\nT =\n%s"%strop(T)
        L = self.L
        if L is not None and len(L):
            s += "\nL =\n%s"%strop(L)
        return s

    def shortstr(self):
        H = self.H.view()
        H.shape = (self.m, 2*self.n)
        return shortstr(H)

    def check(self):
        H = self.H
        m, n = self.shape
        H1 = dot2(H, symplectic_form(n))
        R = dot2(H, H1.transpose())
        if R.sum() != 0:
            print(shortstr(R))
            assert 0, "not isotropic"
        L = self.get_logops()
        R = dot2(L, H1.transpose())
        if R.sum() != 0:
            print("R:")
            print(shortstr(R))
            assert 0
        R = dot2(L, symplectic_form(n), L.transpose())
        if not eq2(R, symplectic_form(self.k)):
            assert 0
        T = self.get_destabilizers()
        HT = array2(list(zip(H, T)))
        HT.shape = 2*m, 2*n
        A = numpy.concatenate((HT, L))
        F = dot2(A, symplectic_form(n), A.transpose())
        assert eq2(F, symplectic_form(n))

    @property
    def deepH(self):
        H = self.H.view()
        m, n = self.shape
        H.shape = m, n, 2
        return H

    @property
    def deepL(self):
        L = self.L.view()
        L.shape = 2*self.k, self.n, 2
        return L

    @property
    def deepT(self):
        T = self.T.view()
        T.shape = self.m, self.n, 2
        return T

#    def dual(self):
#        D = array2([[0,1],[1,0]])
#        H = dot(self.H, D) % 2
#        return QCode(H)

#    def overlap(self, other):
#        F = symplectic_form(self.n)
#        A = dot2(dot2(self.L, F), other.L.transpose())
#        #A = dot2(self.L, other.L.transpose())
#        A = dot2(symplectic_form(self.k), A)
#        return A

    def equiv(self, other):
        H1, H2 = self.H.transpose(), other.H.transpose()
        U = solve2(H1, H2)
        if U is None:
            return False
        U = solve2(H2, H1)
        if U is None:
            return False
        return True

    def to_css(self):
        H = self.H
        #print(shortstr(H))
        #print(self.space.F)
        m, nn = H.shape
        Hx = H[:, 0:nn:2]
        Hz = H[:, 1:nn:2]
        idxs = numpy.where(Hx.sum(1))[0]
        jdxs = numpy.where(Hz.sum(1))[0]
        #print(idxs, jdxs)
        assert Hx[jdxs, :].sum() == 0, "not in css form"
        assert Hz[idxs, :].sum() == 0, "not in css form"
        Hx = Hx[idxs, :]
        Hz = Hz[jdxs, :]
        code = CSSCode(Hx=Hx, Hz=Hz)
        return code

    def to_qcode(self):
        return self

    def get_graph(self):
        "encode into a pynauty Graph"
        from pynauty import Graph
        H = self.H
        rows = [v for v in span(H) if v.sum()]
        V = array2(rows)
        #print(V.sum(0)) # not much help..
        m, nn = V.shape
        n = nn//2
        g = Graph(nn+m, True) # bits + checks
        for bit in range(nn):
            checks = [nn+check for check in range(m) if V[check, bit]]
            g.connect_vertex(bit, checks)
        for bit in range(n):
            g.connect_vertex(2*bit, [2*bit+1])
        #g.set_vertex_coloring([
        #    set(i for i in range(nn)), 
        #    set(range(nn, m+nn))
        #])
        g.set_vertex_coloring([
            set(2*i for i in range(n)), 
            set(2*i+1 for i in range(n)), 
            set(range(nn, m+nn))
        ])
        return g

    def get_autos(self):
        assert self.m <= 20, ("um... %s is too big ??"%(self.m))
        from pynauty import Graph, autgrp
        g = self.get_graph()
        aut = autgrp(g)
        gen = aut[0]
        N = int(aut[1])
        n = self.n
        perms = []
        for perm in gen:
            for bit in range(n):
                assert perm[2*bit]%2 == 0
                assert perm[2*bit+1]%2 == 1
                assert perm[2*bit]+1 == perm[2*bit+1]
            perm = [perm[2*bit]//2 for bit in range(n)]
            perms.append(perm)
        return N, perms

    def get_isomorphism(self, other):
        import pynauty
        lhs = self.get_graph()
        #print(lhs)
        rhs = other.get_graph()
        #print(rhs)
        if not pynauty.isomorphic(lhs, rhs):
            return None
        # See https://github.com/pdobsan/pynauty/issues/31
        f = pynauty.canon_label(lhs) # lhs--f-->C
        g = pynauty.canon_label(rhs) # rhs--g-->C
        iso = [None]*len(f)
        for i in range(len(f)):
            iso[f[i]] = g[i]
        n = self.n
        for bit in range(n):
            assert iso[2*bit]%2 == 0
            assert iso[2*bit+1]%2 == 1
            assert iso[2*bit]+1 == iso[2*bit+1]
        iso = [iso[2*bit]//2 for bit in range(n)]
        return iso

    def is_isomorphic(self, other):
        return self.get_isomorphism(other) is not None

    def apply_perm(self, f):
        n = self.n
        H = self.deepH[:, [f[i] for i in range(n)], :].copy()
        L = self.deepL
        if L is not None:
            L = L[:, [f[i] for i in range(n)], :].copy()
        T = self.deepT
        if T is not None:
            T = T[:, [f[i] for i in range(n)], :].copy()
        return QCode(H, T, L)
    permute = apply_perm

    def apply(self, idx, gate):
        if idx is None:
            code = self
            for idx in range(self.n):
                code = code.apply(idx, gate) # <--- recurse
            return code
        H = self.deepH.copy()
        H[:, idx] = dot(H[:, idx], gate) % 2
        T = self.T
        if T is not None:
            T = self.deepT.copy()
            T[:, idx] = dot(T[:, idx], gate) % 2
        L = self.L
        if L is not None:
            L = self.deepL.copy()
            L[:, idx] = dot(L[:, idx], gate) % 2
        return QCode(H, T, L)

    def apply_H(self, idx=None):
        # swap X<-->Z on bit idx
        H = array2([[0,1],[1,0]])
        return self.apply(idx, H)
    get_dual = apply_H
    H = apply_H

    def apply_S(self, idx=None):
        # swap X<-->Y
        S = array2([[1,1],[0,1]])
        return self.apply(idx, S)
    S = apply_S

    def apply_SH(self, idx=None):
        # X-->Z-->Y-->X 
        SH = array2([[0,1],[1,1]])
        return self.apply(idx, SH)
    SH = apply_SH

    def apply_CZ(self, idx, jdx):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*idx, 2*jdx+1] = 1
        A[2*jdx, 2*idx+1] = 1
        H = dot2(self.H, A)
        T = dot2(self.T, A)
        L = dot2(self.L, A)
        return QCode(H, T, L)
    CZ = apply_CZ

    def apply_CNOT(self, idx, jdx):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*jdx+1, 2*idx+1] = 1
        A[2*idx, 2*jdx] = 1
        #print("apply_CNOT")
        #print(A)
        H = dot2(self.H, A)
        T = dot2(self.T, A)
        L = dot2(self.L, A)
        return QCode(H, T, L)
    CNOT = apply_CNOT

    def row_reduce(self):
        H = self.H.copy()
        m, n = self.shape
        H.shape = m, 2*n
        H = row_reduce(H)
        m, nn = H.shape
        return QCode(H)

    def get_logops(self):
        if self.L is not None:
            return self.L
        m, n = self.shape
        k = self.k
        H = self.H
        L = []
        for i in range(k):
            M = dot2(H, symplectic_form(n))
            K = kernel(M) # XXX do this incrementally for more speed XXX
            J = dot2(K, symplectic_form(n), K.transpose())
            for (row, col) in zip(*numpy.where(J)):
                break
            lx = K[row:row+1]
            lz = K[col:col+1]
            L.append(lx)
            L.append(lz)
            H = numpy.concatenate((H, lx, lz))
    
        L = array2(L)
        L.shape = (2*k, 2*n)
    
        M = dot2(L, symplectic_form(n), L.transpose())
        assert eq2(M, symplectic_form(k))
        self.L = L
        return L

    def get_destabilizers(self):
        if self.T is not None:
            return self.T
        m, n = self.shape
        H = self.H
        L = self.get_logops()
        HL = numpy.concatenate((H, L))
        T = []
        for i in range(m):
            A = dot2(HL, symplectic_form(n))
            #B = numpy.concatenate((identity2(m), zeros2(2*self.k+i, m)))
            B = zeros2(len(A), 1)
            B[i] = 1
            #print("A =")
            #print(A)
            tt = solve2(A, B) # XXX do this incrementally for more speed XXX
            assert tt is not None
            t = tt.transpose()
            t.shape = (1, 2*n)
            T.append(t)
            HL = numpy.concatenate((HL, t))
        T = array2(T)
        T.shape = (m, 2*n)
        self.T = T

        HT = array2(list(zip(H, T)))
        HT.shape = 2*m, 2*n
        A = numpy.concatenate((HT, L))
        M = dot2(A, symplectic_form(n), A.transpose())
        assert eq2(M, symplectic_form(n))

        return T

    def build(self):
        self.get_logops()
        self.get_destabilizers()

    def get_all_logops(self):
        L = self.get_logops()
        H = self.H
        HL = numpy.concatenate((H, L))
        return HL

    def get_distance(self, max_mk=22):
        L = self.get_logops()
        kk = len(L)
        H = self.H
        m = len(H)
        HL = numpy.concatenate((H, L))
        mk = len(HL)
        if mk > max_mk:
            return None
        d = self.n
        for w in numpy.ndindex((2,)*mk):
            u, v = w[:m], w[m:]
            if sum(v) == 0:
                continue
            v = dot2(w, HL)
            count = get_weight(v) # destructive
            if count:
                d = min(count, d)
        if kk:
            self.d = d
        return d

    def get_params(self, max_mk=22):
        d = self.get_distance(max_mk)
        return self.n, self.kk//2, d

    def bound_distance(self):
        L = self.get_logops()
        L = L.copy()
        H = self.H
        kk = len(L)
        #H = self.H
        d = self.n
        for u in L:
            d = min(d, u.sum())
            w = monte_carlo(H, u)
            d = min(w, d)
        return d

    #@cache # needs __hash__
    def get_encoder(self, inverse=False):
        self.build()
        H, T, L = self.H, self.T, self.L
        HT = array2(list(zip(H, T)))
        m, n = self.shape
        HT.shape = 2*m, 2*n
        M = numpy.concatenate((HT, L))
        Mt = M.transpose()
        if inverse:
            F = self.space.F
            return dot2(F, M, F)
        else:
            return Mt
    get_symplectic = get_encoder

    def get_decoder(self):
        return self.get_encoder(True)

    @classmethod
    def from_encoder(cls, M, m=None, k=None, **kw):
        nn = len(M)
        assert M.shape == (nn, nn)
        assert nn%2 == 0
        if k is not None:
            m = nn//2 - k
        if m is None:
            m = nn//2
        assert 0<=2*m<=nn
        M = M.transpose()
        HT = M[:2*m, :]
        L = M[2*m:, :]
        H = HT[0::2, :]
        T = HT[1::2, :]
        code = QCode(H, T, L, **kw)
        return code
    from_symplectic = from_encoder

    @classmethod
    def trivial(cls, n):
        return cls.from_symplectic(identity2(2*n))

    def __lshift__(left, right):
        assert left.n == right.n
        L = left.get_symplectic() 
        R = right.get_symplectic()
        E = dot2(L, R)
        code = QCode.from_symplectic(E, k=right.k)
        return code

    @classmethod
    def load_codetables(cls):
        f = open("codetables.txt")
        for line in f:
            record = json.loads(line)
            n = record['n']
            k = record['k']
            stabs = record["stabs"]
            code = QCode.fromstr(stabs, d=record["d"])
            yield code

    def is_css(self):
        "check for transversal CNOT"
        n = self.n
        src = self + self
        tgt = src
        for i in range(n):
            tgt = tgt.apply_CNOT(i, n+i)
        return src.equiv(tgt)

    def is_selfdual(self):
        tgt = self.apply_H()
        return self.equiv(tgt)

    def get_logical(self, other):
        L = dot2(self.get_decoder(), other.get_encoder())
        L = L[-self.kk:, -self.kk:]
        return L

    def get_overlap(self, other):
        items = []
        for i in range(self.k+1):
            lhs = concatenate((self.H, self.L[:2*i:2]))
            rhs = concatenate((other.H, other.L[:2*i:2]))
            J = intersect(lhs, rhs)
            items.append(len(J))
        return tuple(items)





