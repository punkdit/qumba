#!/usr/bin/env python
"""
stabilizer codes & symplectic geometry
"""

import string, os
from random import randint, choice, random
from time import sleep, time
from functools import reduce
from operator import add, lshift
import json

import numpy, sys
numpy.set_printoptions(threshold=sys.maxsize)
from numpy import zeros, dot, concatenate

from qumba import lin 
from qumba.lin import (
    array2, zeros2, shortstr, dot2, solve2, linear_independent, row_reduce, kernel,
    span, intersect, rank, enum2, shortstrx, identity2, eq2, pseudo_inverse)
from qumba.matrix import Matrix, flatten
from qumba.symplectic import SymplecticSpace, symplectic_form
from qumba.argv import argv
from qumba.smap import SMap
from qumba.util import cache



def parse(s):
    for c in "XZY":
        s = s.replace(c, '1')
    s = s.replace("I", "0")
    for c in "[],":
        s = s.replace(c, '')
    return lin.parse(s)


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
    L = flatten(L)
    return L


def css_to_isotropic(Hx, Hz):
    mx, n = Hx.shape
    mz, n1 = Hz.shape
    assert n==n1
    H = zeros2(mx+mz, n, 2)
    H[:mx, :, 0] = Hx
    H[mx:, :, 1] = Hz
    return H


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
    if isinstance(v, Matrix):
        v = v.A
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

def strop(H, ichar=".", sep="\n"):
    assert len(H.shape) in (1,2), H.shape
    if len(H.shape) == 1:
        shape = (1,)+H.shape
        H = H.reshape(*shape)
    smap = SMap()
    m, nn = H.shape
    for i in range(m):
      for j in range(nn//2):
        x, z = H[i, 2*j:2*j+2]
        c = ichar
        if x and z:
            c = 'Y'
        elif x:
            c = 'X'
        elif z:
            c = 'Z'
        smap[i,j] = c
    return str(smap).replace("\n", sep)


class QCode(object):
    def __init__(self, H=None, T=None, L=None, J=None, A=None, 
            d=None, d_lower_bound=None, d_upper_bound=None,
            name=None, desc="", check=True, **attrs
        ):
        A = Matrix.promote(A)
        if H is None:
            assert A is not None
            H = A.linear_independent()
        else:
            H = flatten(H) # stabilizers
            H = Matrix.promote(H)
        assert len(H)==0 or H.max() <= 1
        m, nn = H.shape
        assert nn%2 == 0
        n = nn//2
        T = Matrix.promote(T)
        L = Matrix.promote(L)
        J = Matrix.promote(J)
        self.H = H # stabilizers
        self.A = A # redundant stabilizers
        self.T = T # destabilizers
        self.L = L # logicals
        self.J = J # gauge generators
        self.m = m
        self.n = n
        self.nn = nn
        self.k = n - m # number of logicals
        self.kk = 2*self.k
        self.d_lower_bound = d_lower_bound or 1
        self.d_upper_bound = d_upper_bound or n
        if d is not None:
            self.d = d
        self.shape = m, n
        self.space = SymplecticSpace(n)
        if L is not None:
            assert L.shape == (2*self.k, nn), str((L.shape , (2*self.k, nn)))
        if T is not None:
            assert T.shape == (self.m, nn)
        if check:
            self.check()
            if self.d is None and self.n < 10:
                self.get_distance()
        self.name = name
        self.desc = desc
        self.attrs = attrs # serialize these attrs

        #if argv.store_db:
        #    from qumba import db
        #    db.add(self)

    def get_d(self):
        if self.d_lower_bound == self.d_upper_bound:
            return self.d_lower_bound

    def set_d(self, d):
        d = int(d)
        assert type(d) is int
        self.d_lower_bound = d
        self.d_upper_bound = d

    d = property(get_d, set_d)

    def __getattr__(self, name):
        if name in self.attrs:
            return self.attrs[name]
        raise AttributeError

    def __eq__(self, other):
        assert isinstance(other, QCode)
        # XXX assert None's agree before we check, because of __hash__ below
        eq = lambda A, B : (A is None and B is None) or A is not None and B is not None and A==B
        return (
            eq(self.H, other.H) and eq(self.T, other.T) and
            eq(self.L, other.L) and eq(self.J, other.J))

    def __hash__(self):
        self.build()
        key = self.longstr()
        return hash(key)

    def __add__(self, other):
        H = self.H.direct_sum(other.H)
        T = self.T.direct_sum(other.T)
        L = self.L.direct_sum(other.L)
        return QCode(H, T, L)

    def __rmul__(self, r):
        if type(r) is int:
            assert r>=0
            code = reduce(add, [self]*r)
        elif type(r) is Matrix:
            E = self.get_encoder()
            E = r*E
            code = QCode.from_encoder(E, self.m)
        else:
            raise TypeError
        return code

    @classmethod
    def build_css(cls, Hx, Hz, Tx=None, Tz=None, Lx=None, Lz=None, Jx=None, Jz=None, **kw):
        H = css_to_isotropic(Hx, Hz)
        #T = css_to_isotropic(Tx, Tz) if Tx is not None else None # FAIL
        T = None
        L = css_to_symplectic(Lx, Lz) if Lx is not None else None
        J = css_to_symplectic(Jx, Jz) if Jx is not None else None
        code = QCode(H, T, L, J, **kw)
        code.get_destabilizers()
        return code

    @classmethod
    def build_gauge(cls, J, **kw):
        m, n, _ = J.shape
        J1 = J.copy()
        J1[:, :, :] = J1[:, :, [1,0]]
        J1.shape = (m, 2*n)
        K = kernel(J1)
        J = J.view()
        J.shape = m, 2*n
        H = intersect(J, K)
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
            code = QCode.build_gauge(J, T=T, L=L, check=check, **kw)
        else:
            code = QCode(H, T, L, J, check=check, **kw)
        return code

    def __str__(self):
        #if self.name is not None:
            #return self.name #???
        d = (self.d if self.d is not None 
            else '%d<=d<=%d'%(self.d_lower_bound, self.d_upper_bound))
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
        assert H.rank() == len(H)
        m, n = self.shape
        F = self.space.F # the symplectic form
        R = H * F * H.t
        if R.sum() != 0:
            print("R =")
            print(shortstr(R))
            assert 0, "not isotropic"
        L = self.get_logops()
        R = L*F*H.t
        if R.sum() != 0:
            print("R:")
            print(shortstr(R))
            assert 0
        R = L*F*L.t
        if R != symplectic_form(self.k):
            assert 0
        T = self.get_destabilizers()
        HT = array2(list(zip(H.A, T.A)))
        HT.shape = 2*m, 2*n
        HT = Matrix(HT)
        A = HT.concatenate(L)
        assert A*F*A.t == F

    @property
    def deepH(self):
        H = self.H.A.view()
        m, n = self.shape
        H.shape = m, n, 2
        return H

    @property
    def deepL(self):
        L = self.L.A.view()
        L.shape = 2*self.k, self.n, 2
        return L

    @property
    def deepT(self):
        T = self.T.A.view()
        T.shape = self.m, self.n, 2
        return T

    def is_equiv(self, other):
        assert isinstance(other, QCode)
        assert self.n == other.n, "wut"
        if self.k != other.k:
            return False
        H1, H2 = self.H.t, other.H.t
        U = solve2(H1.A, H2.A)
        if U is None:
            return False
        U = solve2(H2.A, H1.A)
        if U is None:
            return False
        return True

    @cache 
    def to_css(self):
        from qumba.csscode import CSSCode
        H = self.H
        m, nn = H.shape
        Hx = H[:, 0:nn:2].A
        Hz = H[:, 1:nn:2].A
        idxs = numpy.where(Hx.sum(1))[0]
        jdxs = numpy.where(Hz.sum(1))[0]
        #assert Hx[jdxs, :].sum() == 0, "not in css form"
        #assert Hz[idxs, :].sum() == 0, "not in css form"
        Hx = Hx[idxs, :]
        Hz = Hz[jdxs, :]
        code = CSSCode(Hx=Hx, Hz=Hz)
        return code

    def to_qcode(self):
        return self

    #@cache
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
        rhs = other.get_graph()
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

    def find_zx_dualities(code):
        from qumba import csscode
        from qumba.action import Perm
        n = code.n
        A = strop(code.A).split()
        for op in A:
            xop = 'X' in op
            zop = 'Z' in op
            assert not (xop and zop), "not CSS form"
        Ax = '\n'.join([op for op in A if 'X' in op])
        Az = '\n'.join([op for op in A if 'Z' in op])
        Ax = parse(Ax)
        Az = parse(Az)
        duality = csscode.find_zx_duality(Ax, Az)
        items = list(range(n))
        duality = Perm.promote(duality, items)
        #print(duality)
        perms = csscode.find_autos(Ax, Az)
        G = [Perm.promote(g, items) for g in perms]
        #print("|G| =", len(G))
        I = Perm(items, items)
        zxs = []
        for g in G:
            zx = g*duality
            if zx*zx != I:
                continue
            for i in items:
                if zx[i] == i:
                    break
            else:
                zxs.append(zx)
        return zxs

    def apply(self, M):
        assert isinstance(M, Matrix)
        self.build()
        Mt = M.t
        H = self.H * Mt
        T = self.T * Mt
        L = self.L * Mt
        return QCode(H, T, L)

    def apply_perm(self, f):
        M = self.space.get_perm(f)
        return self.apply(M)
    permute = apply_perm

    def apply_swap(self, i, j):
        perm = list(range(self.n))
        perm[i], perm[j] = perm[j], perm[i]
        return self.apply_perm(perm)
    swap = apply_swap

    def apply_H(self, idx=None):
        # swap X<-->Z on bit idx
        M = self.space.get_H(idx)
        return self.apply(M)
    get_dual = apply_H
    H = apply_H

    def apply_S(self, idx=None):
        # swap X<-->Y
        M = self.space.get_S(idx)
        return self.apply(M)
    S = apply_S

    def apply_SH(self, idx=None):
        # X-->Z-->Y-->X 
        M = self.space.get_SH(idx)
        return self.apply(M)
    SH = apply_SH

    def apply_CZ(self, idx, jdx):
        assert idx != jdx
        M = self.space.get_CZ(idx, jdx)
        return self.apply(M)
    CZ = apply_CZ

    def apply_CNOT(self, idx, jdx):
        assert idx != jdx
        M = self.space.get_CNOT(idx, jdx)
        return self.apply(M)
    apply_CX = apply_CNOT
    CNOT = apply_CNOT
    CX = CNOT


#    def row_reduce(self):
#        H = self.H.copy()
#        m, n = self.shape
#        H.shape = m, 2*n
#        H = row_reduce(H)
#        m, nn = H.shape
#        return QCode(H)

    def normal_form(self):
        "row reduce X's for Clifford encoder"
        H = self.H
        assert isinstance(H, Matrix)
        H = H.A.copy()
        #print("row_reduce", H.shape)
        m, nn = H.shape
        n = nn//2
        #print("="*n)
        #print(strop(H))
        #print("="*n)
    
        i = j = 0
        while i < m and j < n:
    
            # first find a nonzero X entry in this col
            for i1 in range(i, m):
                if H[i1, 2*j]:
                    break
            else:
                j += 1 # move to the next col
                continue # <----------- continue ------------
    
            if i != i1:
                # swap rows
                h = H[i, :].copy()
                H[i, :] = H[i1, :]
                H[i1, :] = h
    
            assert H[i, 2*j]
            for i1 in range(i+1, m):
                if H[i1, 2*j]:
                    H[i1, :] += H[i, :]
                    H[i1, :] %= 2
    
            i += 1
            j += 1
    
        #print("="*n)
        #print(strop(H))
        #print("="*n)
        #H = Matrix(H)
        code = QCode(H)
        return code

    def get_logops(self):
        if self.L is not None:
            return self.L
        m, n = self.shape
        F = self.space.F
        k = self.k
        H = self.H
        L = []
        for i in range(k):
            M = H*F
            K = M.kernel()
            J = K*F*K.t
            for (row, col) in J.where():
                break
            lx = K[row:row+1]
            lz = K[col:col+1]
            L.append(lx.A)
            L.append(lz.A)
            H = H.concatenate(lx, lz)
    
        L = array2(L)
        L.shape = (2*k, 2*n)
        L = Matrix(L)
    
        M = L*F*L.t
        assert M==symplectic_form(k)
        self.L = L
        self.bound_d()
        return L

    def bound_d(self):
        L = self.L
        #if L is None:
        #    L = self.get_logops()
        for i in range(self.k):
            l = L[i:i+1,:].transpose()
            w = get_weight(l.copy()) # destructive
            if w < self.d_upper_bound:
                self.d_upper_bound = w
        n = self.n
        nn = 2*n
        H = self.H
        W = H.A.sum(0)
        #print(W, W.shape)
        W.shape = (n, 2)
        #print(W, W.shape, numpy.min(W))

        if self.d is not None:
            return

        if numpy.min(W) > 0 and self.d_lower_bound<2:
            self.d_lower_bound = 2

        if self.n > 100:
            return

        if self.k == 0:
            return

        from qumba.distance import distance_meetup
        max_m = argv.get("max_m", 3)
        d = distance_meetup(self, max_m)
        if d is None:
            self.d_lower_bound = 2*max_m + 1
        else:
            self.d = d

    def get_destabilizers(self):
        if self.T is not None:
            return self.T
        m, n = self.shape
        H = self.H
        L = self.get_logops()
        HL = H.concatenate(L)
        F = self.space.F
        T = []
        for i in range(m):
            A = HL*F
            B = zeros2(len(A), 1)
            B[i] = 1
            tt = solve2(A.A, B) # XXX do this incrementally for more speed XXX
            assert tt is not None, strop(H)
            t = tt.transpose()
            t.shape = (1, 2*n)
            T.append(t)
            HL = HL.concatenate(Matrix(t))
        T = array2(T)
        T.shape = (m, 2*n)
        T = Matrix(T)
        self.T = T

        # one last check...
        HT = array2(list(zip(H.A, T.A)))
        HT.shape = 2*m, 2*n
        A = numpy.concatenate((HT, L.A))
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
        HL = numpy.concatenate((H.A, L.A))
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

    def distance(self, method=None, verbose=False):
        if method == "z3":
            from qumba.distance import distance_z3
            return distance_z3(self, verbose)
        else:
            return self.get_distance()

    def get_params(self, max_mk=22):
        d = self.d or self.get_distance(max_mk)
        return self.n, self.kk//2, d

    def bound_distance(self):
        L = self.get_logops()
        L = L.copy()
        H = self.H
        kk = len(L)
        d = self.n
        for u in L:
            d = min(d, u.sum())
            w = monte_carlo(H, u)
            d = min(w, d)
        return d

    #@cache # needs __hash__
    def get_encoder(self, inverse=False):
        """
            Return symplectic encoder matrix E.
            Note: first m qubits are in |+>,|-> basis, 
            because these are the eigenvectors of X operator.
            TODO: Is this natural or should we Hadamard these ????? ARGHHH
        """
        self.build()
        H, T, L = self.H, self.T, self.L
        HT = array2(list(zip(H.A, T.A)))
        m, n = self.shape
        HT.shape = 2*m, 2*n
        M = numpy.concatenate((HT, L.A))
        M = Matrix(M)
        if inverse:
            F = self.space.F
            return F*M*F
        else:
            return M.t
    get_symplectic = get_encoder

    def get_clifford_encoder(code, verbose=False):
        from qumba.clifford import Clifford, red, green, Matrix
        code = code.normal_form()
        n = code.n
        c = Clifford(n)
        CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    
        E = I = c.get_identity()
        for src,h in enumerate(code.H):
            op = strop(h).replace('.', 'I')
            ctrl = op[src]
            if verbose:
                print(src, op, ctrl, end=":  ")
            if ctrl=='I':
                E0 = I
            elif ctrl in 'XZY':
                E0 = H(src)
                if verbose:
                    print("H(%d)"%src, end=" ")
            else:
                assert 0, ctrl
    
            for tgt,opi in enumerate(op):
                if tgt==src:
                    continue
                if opi=='I':
                    pass
                elif opi=='X':
                    E0 = CX(src,tgt)*E0
                    if verbose:
                        print("CX(%d,%d)"%(src,tgt), end=" ")
                elif opi=='Z':
                    E0 = CZ(src,tgt)*E0
                    if verbose:
                        print("CZ(%d,%d)"%(src,tgt), end=" ")
                elif opi=='Y':
                    E0 = CY(src,tgt)*E0
                    if verbose:
                        print("CY(%d,%d)"%(src,tgt), end=" ")
                else:
                    assert 0, opi
    
            if ctrl in 'XI':
                pass
            elif ctrl == 'Z':
                E0 = H(src)*E0
                if verbose:
                    print("H(%d)"%src, end=" ")
            elif ctrl == 'Y':
                E0 = S(src)*E0
                if verbose:
                    print("S(%d)"%src, end=" ")
            else:
                assert 0, ctrl
            if verbose:
                print()
            E = E * E0
        return E

    def get_encoder_name(code, verbose=False):
        from qumba.syntax import Syntax
        s = Syntax()
        code = code.normal_form()
        n = code.n
        CX, CY, CZ, H, S = s.CX, s.CY, s.CZ, s.H, s.S
    
        E = I = s.get_identity()
        for src,h in enumerate(code.H):
            op = strop(h).replace('.', 'I')
            ctrl = op[src]
            if verbose:
                print(src, op, ctrl, end=":  ")
            if ctrl=='I':
                E0 = I
            elif ctrl in 'XZY':
                E0 = H(src)
                if verbose:
                    print("H(%d)"%src, end=" ")
            else:
                assert 0, ctrl
    
            for tgt,opi in enumerate(op):
                if tgt==src:
                    continue
                if opi=='I':
                    pass
                elif opi=='X':
                    E0 = CX(src,tgt)*E0
                    if verbose:
                        print("CX(%d,%d)"%(src,tgt), end=" ")
                elif opi=='Z':
                    E0 = CZ(src,tgt)*E0
                    if verbose:
                        print("CZ(%d,%d)"%(src,tgt), end=" ")
                elif opi=='Y':
                    E0 = CY(src,tgt)*E0
                    if verbose:
                        print("CY(%d,%d)"%(src,tgt), end=" ")
                else:
                    assert 0, opi
    
            if ctrl in 'XI':
                pass
            elif ctrl == 'Z':
                E0 = H(src)*E0
                if verbose:
                    print("H(%d)"%src, end=" ")
            elif ctrl == 'Y':
                E0 = S(src)*E0
                if verbose:
                    print("S(%d)"%src, end=" ")
            else:
                assert 0, ctrl
            if verbose:
                print()
            E = E * E0
        return E

    def get_decoder(self):
        return self.get_encoder(True)

    def get_isometry(self, inverse=False):
        E = self.get_encoder(inverse)
        if inverse:
            E = E[-self.kk:, :]
        else:
            E = E[:, -self.kk:]
        return E

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
        E = L*R
        code = QCode.from_symplectic(E, k=right.k)
        return code

    def concat(self, other):
        #lhs = reduce(add, [self]*other.n)
        #lhs = lhs.get_encoder()
        E = self.get_encoder()
        lhs = reduce(lshift, [E]*other.n)
        I = SymplecticSpace(other.n).get_identity()
        rhs = reduce(lshift, [I]*self.m + [other.get_encoder()]*self.k)
        #idxs = [None]*(len(lhs)//2)
        #count = 0
        idxs = []
        for i in range(self.m):
          for j in range(other.n):
            idxs.append( j*self.n + i )
        for i in range(self.k):
          for j in range(other.n):
            idxs.append( j*self.n + (self.m + i) )
        assert len(set(idxs)) == len(idxs)
        P = SymplecticSpace(len(idxs)).get_perm(idxs).t
        E = lhs * P * rhs
        idxs = list(range(other.n*self.m))
        k = len(idxs)
        for i in range(self.k):
          for j in range(other.m):
            idxs.append( k + i*other.n + j )
        for i in range(self.k):
          for j in range(other.k):
            idxs.append( k + i*other.n + other.m + j )
        assert len(set(idxs)) == len(idxs)
        P = SymplecticSpace(len(idxs)).get_perm(idxs).t
        E = E * P
        code = QCode.from_encoder(E, k=self.k*other.k)
        return code

    @classmethod
    def load_codetables(cls):
        f = open("codetables.txt")
        for line in f:
            record = json.loads(line)
            n = record['n']
            k = record['k']
            stabs = record["stabs"]
            code = QCode.fromstr(stabs, d=record["d"], desc="codetables") 
            assert code.d is not None, (record["d"], str(code), id(code))
            yield code

    @cache
    def is_css(self):
        H = self.H.A
        n = self.n
        xs = zeros2(1, 2*n)
        xs[0, ::2] = 1
        Hx = H*xs
        if solve2(H.transpose(), Hx.transpose()) is None:
            css = False
        else:
            zs = zeros2(1, 2*n)
            zs[0, ::2] = 1
            Hz = H*zs
            css = solve2(H.transpose(), Hz.transpose()) is not None
        #assert css == self.is_css_slow()
        self.attrs["css"] = css
        return css

    def is_css_slow(self):
        "check for transversal CNOT"
        n = self.n
        src = self + self
        tgt = src
        for i in range(n):
            tgt = tgt.apply_CNOT(i, n+i)
        css = src.is_equiv(tgt)
        self.attrs["css"] = css
        return css

    @cache
    def is_gf4(self):
        "check for transversal SH"
        n = self.n
        tgt = self
        for i in range(n):
            tgt = tgt.apply_H(i)
            tgt = tgt.apply_S(i)
        gf4 = self.is_equiv(tgt)
        self.attrs["gf4"] = gf4
        return gf4

    @cache
    def is_selfdual(self):
        tgt = self.apply_H()
        sd = self.is_equiv(tgt)
        self.attrs["selfdual"] = sd
        return sd

    def get_tp(self):
        gf4 = self.is_gf4()
        css = self.is_css()
        sd = self.is_selfdual()
        if css and sd:
            tp = "selfdualcss"
        elif css:
            tp = "css"
        elif gf4:
            tp = "gf4"
        elif sd:
            tp = "selfdual"
        else:
            tp = "none"
        self.attrs["tp"] = tp
        return tp

    @cache
    def is_cyclic(self):
        n = self.n
        perm = [(i+1)%n for i in range(n)]
        code = self.space.get_perm(perm)*self
        return code.is_equiv(self)

    def old_get_logical(self, other, check=False):
        if check:
            assert self.is_equiv(other)
        L = self.get_decoder() * other.get_encoder()
        L = L[-self.kk:, -self.kk:]
        return L

    def get_logical(self, other, check=False):
        if check:
            assert self.is_equiv(other)
        Fn = self.space.F
        Fk = SymplecticSpace(self.k).F
        U = Fk*self.L * Fn * other.L.t
        if check:
            L = self.old_get_logical(other)
            assert U == L
        return U

    def get_overlap(self, other):
        assert 0, "fix me"
        items = []
        for i in range(self.k+1):
            lhs = concatenate((self.H, self.L[:2*i:2]))
            rhs = concatenate((other.H, other.L[:2*i:2]))
            J = intersect(lhs, rhs)
            items.append(len(J))
        return tuple(items)

    #@cache
    def get_projector(self):
        "get Clifford projector onto codepsace"
        from qumba.clifford import Clifford, half
        H = self.H
        n = self.n
        assert n < 16, "wup.. too big ?"
        c = Clifford(n)
        I = c.get_identity()
        P = I
        for u in H:
            desc = strop(u)
            g = c.get_pauli(desc)
            P *= half * (I + g)
        return P

    # much slower than get_projector
#    def get_average_operator(self):
#        "get Clifford projector onto codepsace"
#        from qumba.clifford import Clifford, half
#        H = self.H
#        n = self.n
#        assert n < 16, "wup.. too big ?"
#        c = Clifford(n)
#        P = None
#        for u in H.rowspan():
#            desc = strop(u)
#            g = c.get_pauli(desc)
#            P = g if P is None else P+g
#        return P


    def get_components(self):
        print("get_components")
        H = self.H
        print(H.normal_form())
        




