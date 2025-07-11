#!/usr/bin/env python

"""

An implementation of the projective Clifford group.
These have orders: 24, 11520, 92897280, ...

The idea is to store (g,alpha) in class Element.
This is a F_2 symplectic matrix g, together with
a dict alpha:V-->Z/4 of phases. V here is the
collection of F_2 vectors. alpha satisfies an
equation that we check in Element.check below.

Reference:
The Weil representation in characteristic two
Shamgar Gurevich,∗, Ronny Hadani (2012)
section 0.2

See also: 
bruhat/heisenberg.py
qumba/group_cohomology.py

"""

from random import choice

from functools import lru_cache
cache = lru_cache(maxsize=None)

import numpy

from bruhat.gset import Perm, Group, mulclose, mulclose_hom
#from bruhat.repr_sage import dixon_irr

from qumba.argv import argv
from qumba.smap import SMap
from qumba.lin import zeros2, array2
from qumba.matrix import Matrix
from qumba.symplectic import SymplecticSpace


def beta(u, v):
    # beta is a 2-cocycle for constructing a "Heisenberg group"
    # as a central extension:
    # Z_4 >---> H(V) -->> V
    assert u.shape == v.shape
    u0 = u[::2, :]
    v0 = v[1::2, :]
    uv = u0.t*v0
    assert uv.shape == (1,1)
    result = 2*int(uv[0,0])
    return result


class AffineSpace:
    def __init__(self, n, basis, zero, V):
        self.space = SymplecticSpace(n)
        self.n = n
        self.zero = zero
        self.basis = basis
        self.V = V

    def find_all(self, g):
        # alpha : V --> Z/4Z
        nn = 2*self.n
        zero = self.zero
        basis = self.basis
        for bits in numpy.ndindex((4,)*nn):
            alpha = {zero:0}
            for i,bit in enumerate(bits):
                alpha[basis[i]] = bit
            element = Element.build(self, g, alpha)
            if element is not None:
                yield element

    @cache
    def get_identity(self):
        g = self.space.get_identity()
        alpha = {v:0 for v in self.basis}
        e = Element.build(self, g, alpha)
        assert e is not None
        e.check()
        return e

    def S(self, i=0):
        assert i is not None
        g = self.space.S(i)
        alpha = {v:0 for v in self.basis}
        alpha[self.basis[2*i]] = 1 # or 3 ?
        e = Element.build(self, g, alpha)
        assert e is not None
        e.check()
        return e

    def H(self, i=None):
        g = self.space.H(i)
        alpha = {v:0 for v in self.basis}
        e = Element.build(self, g, alpha)
        assert e is not None
        e.check()
        return e

    def CX(self, i=0, j=1):
        g = self.space.CX(i, j)
        alpha = {v:0 for v in self.basis}
        e = Element.build(self, g, alpha)
        assert e is not None
        e.check()
        return e

    def CZ(self, i=0, j=1):
        g = self.space.CZ(i, j)
        alpha = {v:0 for v in self.basis}
        e = Element.build(self, g, alpha)
        assert e is not None
        e.check()
        return e

    def CY(self, i=0, j=1):
        g = self.space.CY(i, j)
        alpha = {v:0 for v in self.basis}
        e = Element.build(self, g, alpha)
        assert e is not None
        e.check()
        return e




class Element:
    def __init__(self, affine, g, alpha):
        self.affine = affine
        self.g = g
        self.alpha = alpha

    @classmethod
    def build(self, affine, g, alpha):
        "build from values on basis"
        basis = affine.basis
        alpha = dict(alpha) # paranoid
        nn = 2*affine.n
        while len(alpha) < 2**nn:
            for u in list(alpha.keys()):
              for v in basis:
                uv = u+v
                value = (beta(g*u,g*v) - beta(u,v) + alpha[u] + alpha[v])%4
                if uv in alpha and alpha[uv] != value:
                    return
                elif uv in alpha:
                    continue
                assert u in alpha
                assert v in alpha
                alpha[uv] = value
        return Element(affine, g, alpha)

    def check(self):
        affine, g, alpha = self.affine, self.g, self.alpha
        V = affine.V
        for u in V:
            for v in V:
                lhs = (alpha[u+v] - alpha[u] - alpha[v]) % 4
                rhs = (beta(g*u,g*v) - beta(u,v))%4
                assert lhs==rhs

    def __eq__(self, other):
        return self.g==other.g and self.alpha==other.alpha

    def __str__(self):
        smap = SMap()
        smap[0,0] = str(self.g)
        nn = 2*self.affine.n
        #s = str(self.alpha).replace("\n", "")
        s = ''.join(str(self.alpha[v]) for v in self.affine.basis)
        smap[nn-1, nn+1] = "["+s+"]"
        return str(smap)
    __repr__ = __str__

    def __hash__(self):
        return hash(str(self))

    def __mul__(self, other): # hotspot 10%
        assert isinstance(other, Element)
        assert self.affine is other.affine
        V = self.affine.V
        g, a = self.g, self.alpha
        h, b = other.g, other.alpha
        gh = g*h
        ab = {v:(a[h*v]+b[v])%4 for v in V}
        return Element(self.affine, gh, ab)

    def __call__(self, v, phase=0):
        return self.g*v, (phase + self.alpha[v])%4


def vec(bits):
    nn = len(bits)
    return Matrix(array2(bits)).reshape(nn,1)

def get(n):

    nn = 2*n

    space = SymplecticSpace(n)
    F = space.F # omega

    V = []
    for idxs in numpy.ndindex((2,)*nn):
        v = vec(idxs)
        V.append(v)
    basis = []
    for i in range(nn):
        bits = [0]*nn
        bits[i] = 1
        basis.append(vec(bits))
    assert len(basis) == nn
    zero = vec((0,)*nn)

    omega = {}
    for u in V:
      for v in V:
        w = (u.t * F * v)
        assert w.shape == (1,1)
        w = int(w[0,0])
        omega[u,v] = w

    # check that
    # beta(u,v) - beta(v,u) = 2*omega(u,v) in 2Z/4Z
    for u in V:
      for v in V:
        lhs = (beta(u,v) - beta(v,u)) % 4
        rhs = 2*omega[u,v]
        #print("%s:%s"%(lhs, rhs), end=" ")
        assert lhs==rhs

    # todo: construct H(V) using beta & check it..

    return AffineSpace(n, basis, zero, V)



def test_ASp(n):

    nn = 2*n
    affine = get(n)

    space = SymplecticSpace(n)

    S, H, CX = space.S, space.H, space.CX
    gen = []
    for i in range(n):
        gen.append(S(i))
        gen.append(H(i))
        for j in range(n):
            if i!=j:
                gen.append(CX(i,j))

    G = mulclose(gen)
    print("|Sp| =", len(G))

    ASp = [] # here goes..
    for g in G:
        for e in affine.find_all(g):
            ASp.append(e)
        print(".", end="", flush=True)
    print()

    print("|ASp| =", len(ASp))

    V = affine.V
    for trial in range(100):
        e = choice(ASp)
        e.check()

        #print(e)

        f = choice(ASp)
        fe = f*e
        fe.check()
        assert fe in ASp

        g = choice(ASp)
        lhs = fe*g
        rhs = f*(e*g)
        assert lhs == rhs

        for v in V:
            assert e(v)[0] in V
            for phase in [0,1,2,3]:
                assert fe(v, phase) == f(*e(v, phase))

#    print("trivial action:")
#    for e in ASp:
#        for v in V:
#            if e(v,0) != (v,0):
#                break
#        else:
#            print(e)

    if n==1:
        N = len(ASp)
        lookup = {e:i for (i,e) in enumerate(ASp)}
        perms = []
        for e in ASp:
            perm = [lookup[e*g] for g in ASp]
            assert len(set(perm)) == N
            perms.append(Perm(perm))
            print(".", end="", flush=True)
        print()
        G = Group(perms)
        print(G)
        G.do_check()

    X = [(v, phase) for v in V for phase in range(4)]
    print("|X| =", len(X))

    N = len(X)
    lookup = {vp:i for (i,vp) in enumerate(X)}
    perms = []
    for e in ASp:
        perm = [lookup[e(*vp)] for vp in X]
        assert len(set(perm)) == N
        perms.append(Perm(perm))
        #print(".", end="", flush=True)
    #print()
    G = Group(perms)
    #print(G)
    #G.do_check()

    print()


    

def test_hom(n):
    nn = 2*n
    affine = get(n)
    S, H, CX, CZ = affine.S, affine.H, affine.CX, affine.CZ
    agen = []
    for i in range(n):
        agen.append(S(i))
        agen.append(H(i))
        for j in range(n):
            if i!=j:
                agen.append(CX(i,j))
            if i<j:
                agen.append(CZ(i,j))

    from qumba.clifford import Clifford
    c = Clifford(n)
    S, H, CX, CZ = c.S, c.H, c.CX, c.CZ
    bgen = []
    for i in range(n):
        bgen.append(S(i))
        bgen.append(H(i))
        for j in range(n):
            if i!=j:
                bgen.append(CX(i,j))
            if i<j:
                bgen.append(CZ(i,j))

    #G = mulclose(gen, verbose=True)
    #print(len(G))

    hom = mulclose_hom(bgen, agen, verbose=True)

    def getrand(gen, N=10):
        g = gen[0]
        for trial in range(N):
            g = g*choice(gen)
        return g

    I = c.I
    for trial in range(100):
        g = getrand(bgen)
        h = getrand(bgen)
        assert hom[g]*hom[h] == hom[g*h]

    # kernel
    w = c.wI()
    for n in range(8):
        assert hom[w**n] == affine.get_identity()





def test():
    test_ASp(1)
    #test_ASp(2) # sloow

    test_hom(1)
    #test_hom(2) # sloow
    


if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "test"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))


