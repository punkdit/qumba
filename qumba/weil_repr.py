#!/usr/bin/env python

"""
The Weil representation in characteristic two
Shamgar Gurevich,∗, Ronny Hadani
2012
section 0.2

see also: 
bruhat/heisenberg.py
qumba/group_cohomology.py

"""

from random import choice

import numpy

from bruhat.gset import Perm, Group, mulclose, mulclose_hom
#from bruhat.repr_sage import dixon_irr

from qumba.argv import argv
from qumba.smap import SMap
from qumba.umatrix import Not, And, Var, UMatrix, Solver
from qumba.lin import zeros2, array2
from qumba.matrix import Matrix
from qumba.symplectic import SymplecticSpace


def beta(u, v):
    # beta is a 2-cocycle for constructing the Heisenberg group
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



class Element:
    def __init__(self, affine, g, alpha):
        self.affine = affine
        self.g = g
        self.alpha = alpha

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

    def __mul__(self, other):
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



def get(n):

    nn = 2*n

    space = SymplecticSpace(n)
    F = space.F # omega

    V = []
    basis = []
    for idxs in numpy.ndindex((2,)*nn):
        v = Matrix(array2(idxs)).reshape(nn, 1)
        V.append(v)
        if v.sum() == 1:
            basis.append(v)
        elif v.sum() == 0:
            zero = v
    assert len(basis) == nn

    omega = {}
    for u in V:
      for v in V:
        w = (u.t * F * v)
        assert w.shape == (1,1)
        w = int(w[0,0])
        omega[u,v] = w

    # beta is a 2-cocycle for constructing the Heisenberg group
    # as a central extension:
    # Z_4 >---> H(V) -->> V

#    beta = {}
#    for u in V:
#      u0 = u[::2, :]
#      #print(u.t, u.shape, u0.t, u0.shape)
#      for v in V:
#        v0 = v[1::2, :]
#        uv = u0.t*v0
#        beta[u,v] = 2*int(uv[0,0])
#        #print("\t", v.t, v.shape, v0.t, v0.shape, "=", uv, uv.shape)

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



def build(n):

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

    zero, basis = affine.zero, affine.basis
    
    ASp = [] # here goes..
    for g in G:
        # alpha : V --> Z/4Z
        for bits in numpy.ndindex((4,)*nn):
            alpha = {zero:0}
            for i,bit in enumerate(bits):
                alpha[basis[i]] = bit
            succ = True
            gen = set(alpha.keys())
            while succ and len(gen) < 2**nn:
                for u in list(gen):
                  for v in basis:
                    uv = u+v
                    value = (beta(g*u,g*v) - beta(u,v) + alpha[u] + alpha[v])%4
                    if uv in alpha and alpha[uv] != value:
                        succ = False
                        break
                    elif uv in alpha:
                        continue
                    assert u in alpha
                    assert v in alpha
                    alpha[uv] = value
                    gen.add(uv)
                  if not succ:
                    break
            if succ:
                element = Element(affine, g, alpha)
                ASp.append(element)
                #print([alpha[basis[i]] for i in range(nn)])
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
    print(G)
    #G.do_check()


    
def test():
    build(1)
    build(2)
    




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


