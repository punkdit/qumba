#!/usr/bin/env python3

"""
Symplectic spaces and their symplectic transformations

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


@cache
def symplectic_form(n, p=DEFAULT_P):
    F = zeros2(2*n, 2*n)
    for i in range(n):
        F[2*i:2*i+2, 2*i:2*i+2] = [[0,1],[p-1,0]]
    F = Matrix(F, p, name="F")
    return F


class SymplecticSpace(object):
    def __init__(self, n, p=DEFAULT_P):
        assert 0<=n
        self.n = n
        self.nn = 2*n
        self.p = p
        self.F = symplectic_form(n, p)

    def __lshift__(self, other):
        assert isinstance(other, SymplecticSpace)
        assert other.p == self.p
        return SymplecticSpace(self.n + other.n, self.p)
    __add__ = __lshift__

    def is_symplectic(self, M):
        assert isinstance(M, Matrix)
        nn = 2*self.n
        F = self.F
        assert M.shape == (nn, nn)
        return F == M*F*M.transpose()

    def get_identity(self):
        A = identity2(self.nn)
        M = Matrix(A, self.p, name=())
        return M

    def get_perm(self, f):
        n, nn = self.n, 2*self.n
        assert len(f) == n
        assert set([f[i] for i in range(n)]) == set(range(n))
        name = "P(%s)"%(",".join(str(i) for i in f))
        A = zeros2(nn, nn)
        for i in range(n):
            A[2*i, 2*f[i]] = 1
            A[2*i+1, 2*f[i]+1] = 1
        M = Matrix(A, self.p, None, name)
        assert self.is_symplectic(M)
        return M

    def get(self, M, idx=None, name="?"):
        assert M.shape == (2,2)
        assert isinstance(M, Matrix)
        n = self.n
        A = identity2(2*n)
        idxs = list(range(n)) if idx is None else [idx]
        for i in idxs:
            A[2*i:2*i+2, 2*i:2*i+2] = M.A
        A = A.transpose()
        return Matrix(A, self.p, None, name)

    def invert(self, M):
        F = self.F
        return F * M.t * F

    def get_H(self, idx=None):
        # swap X<-->Z on bit idx
        H = Matrix([[0,1],[1,0]], name="H")
        name = "H(%s)"%idx
        return self.get(H, idx, name)

    def get_S(self, idx=None):
        # swap X<-->Y
        S = Matrix([[1,1],[0,1]], name="S")
        name = "S(%s)"%idx
        return self.get(S, idx, name)

    def get_SH(self, idx=None):
        # X-->Z-->Y-->X 
        SH = Matrix([[0,1],[1,1]], name="SH")
        name = "SH(%s)"%idx
        return self.get(SH, idx, name)

    def get_CZ(self, idx=0, jdx=1):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*idx, 2*jdx+1] = 1
        A[2*jdx, 2*idx+1] = 1
        A = A.transpose()
        name="CZ(%d,%d)"%(idx,jdx)
        return Matrix(A, self.p, None, name)

    def get_CNOT(self, idx=0, jdx=1):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*jdx+1, 2*idx+1] = 1
        A[2*idx, 2*jdx] = 1
        A = A.transpose()
        name="CNOT(%d,%d)"%(idx,jdx)
        return Matrix(A, self.p, None, name)

    @cache
    def get_borel(self, verbose=False):
        # _generate the Borel group
        n = self.n
        gen = []
        for i in range(n):
            gen.append(self.get_S(i))
            for j in range(i+1, n):
                gen.append(self.get_CNOT(i, j))
                gen.append(self.get_CZ(i, j))
    
        G = mulclose(gen, verbose=verbose)
        return G

    @cache
    def get_weyl(self, verbose=False):
        # _generate the Weyl group
        n = self.n
        I = self.get_identity()
        gen = [I]
        for i in range(n):
            gen.append(self.get_H(i))
            for j in range(i+1, n):
                perm = list(range(n))
                perm[i], perm[j] = perm[j], perm[i]
                gen.append(self.get_perm(perm))
    
        G = mulclose(gen, verbose=verbose)
        return G

    @cache
    def get_sp_gen(self):
        n = self.n
        gen = []
        for i in range(n):
            gen.append(self.get_S(i))
            gen.append(self.get_H(i))
            for j in range(i+1, n):
                gen.append(self.get_CZ(i, j))
        return gen

    @cache
    def get_sp(self, verbose=False):
        gen = self.get_sp_gen()
        G = mulclose(gen, verbose=verbose)
        return G

    def sample_sp(self):
        gen = self.get_sp_gen()
        A = self.get_identity()
        for i in range(10*self.n):
            A = choice(gen) * A
        return A
    
    def bruhat_decompose(space, g):
        n = space.n
        W = space.get_weyl()
        I = space.get_identity()
        print(g)

    def bruhat_decompose_fail(space, g): # FAIL FAIL
        n = space.n
        W = space.get_weyl()
        I = space.get_identity()
        print(g)
        pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
        left, right = [], []
        # right Borel
        for (i,j) in pairs:
            h = space.get_CNOT(i,j)
            gh = g*h
            if gh.sum() < g.sum():
                right.append(h)
                g = gh
                print(g)
            
        for (i,j) in pairs:
            h = space.get_CZ(i,j)
            gh = g*h
            if gh.sum() < g.sum():
                right.append(h)
                g = gh
                print(g)
    
        for i in range(n):
            h = space.get_S(i)
            gh = g*h
            print("try:", h.name)
            print(gh, "?")
            if gh.sum() < g.sum():
            #if (gh+I).sum() < (g+I).sum():
                right.append(h)
                g = gh
                print(g)
            else:
                print("no")
        del gh, h
    
        # left Borel
        for i in range(n):
            h = space.get_S(i)
            hg = h*g
            if hg.sum() < g.sum():
            #if (hg+I).sum() < (g+I).sum():
                left.append(h)
                g = hg
                print(g)
    
        for (i,j) in pairs:
            h = space.get_CZ(i,j)
            hg = h*g
            if hg.sum() < g.sum():
                left.append(h)
                g = hg
                print(g)
    
        for (i,j) in pairs:
            h = space.get_CNOT(i,j)
            hg = h*g
            if hg.sum() < g.sum():
                left.append(h)
                g = hg
                print(g)
            
        left = list(reversed(left))
        assert(g in W)
        


def test():

    space = SymplecticSpace(2)
    I = space.get_identity()
    CZ = space.get_CZ(0, 1)
    HH = space.get_H()
    S = space.get_perm([1, 0])

    G = mulclose([CZ, HH])
    assert S in G
    assert len(G) == 12

    space = space + space
    gen = [g.direct_sum(I) for g in G]+[I.direct_sum(g) for g in G]
    gen.append(space.get_perm([2,3,0,1]))
    gen.append(space.get_CNOT(0, 2) * space.get_CNOT(1, 3))
    #gen.append(space.get_CZ(0, 2) * space.get_CZ(1, 3))
    G = mulclose(gen)
    assert len(G) == 46080
    

def test_bruhat():

    n = 2
    space = SymplecticSpace(n)
    I = space.get_identity()

    B = space.get_borel()
    print("B", len(B))

    W = space.get_weyl()
    print("W", len(W))

    Sp = space.get_sp()
    print("Sp", len(Sp))
    Sp = list(Sp)

    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]

    # compute a normal form...
    G_S = mulclose([I]+[space.get_S(i) for i in range(n)])
    G_CZ = mulclose([I]+[space.get_CZ(i,j) for (i,j) in pairs])
    G_CNOT = mulclose([I]+[space.get_CNOT(i,j) for (i,j) in pairs])
    right = [s*cz*cnot for s in G_S for cz in G_CZ for cnot in G_CNOT]
    left = [cnot*cz*s for s in G_S for cz in G_CZ for cnot in G_CNOT]
    normal = {l*w*r for l in left for w in W for r in right}
    print("normal:", len(normal))
    #for g in normal:
    #    print(g.name)

    #g = choice(Sp)
    #Sp = list(Sp)
    Sp = list(normal)
    Sp.sort(key = str)
    normal = dict((g,g) for g in normal)

    print("F =")
    print(space.F)


    E = {}
    idxs = [3, 2, 1, 0]
    E[1,3] = space.get_CNOT(0, 1) # 1<--3
    E[2,0] = E[1,3]
    E[2,3] = space.get_S(1).transpose() # 2<--3
    E[0,1] = space.get_S(0).transpose() # 0<--1
    #E[0,3] = space.get_H(0) * space.get_CNOT(0, 1) #* space.get_H(0) # 0<--3
    E[0,3] = space.get_CNOT() * space.get_S(1).transpose() * space.get_S(0).transpose() * space.get_CNOT() * space.get_S(0).transpose() # hack this... wtf
    E[2,1] = E[0,3]
    for key in E.keys():
        print(key, "=", E[key][key]) # should be 1
    assert E[1,3][2,0] == 1
    assert E[0,3][2,1] == 1

    if 0:
        gen = list(E.values())
        B = mulclose(gen)
        print(len(B))
        for g in B:
            if g[0,3] == 1 and g[2,1] == 1:
                print(g, g.name)
        return

    decompose_attempt_0(space, Sp, W, E)

def decompose_attempt_1(space, Sp, W, E):
    n = space.n
    nn = 2*n
    #g = Sp[3]
    B = list(E.values())
    for g in Sp:
        print()
        print("="*79)
        print(g.name)
        print(g)

        for h in B:
            hg = h*g
            if (hg).sum() < g.sum():
                g = hg
        for h in B:
            gh = g*h
            if (gh).sum() < g.sum():
                g = gh
        print(g)
        print(g in W)



def decompose_attempt_0(space, Sp, W, E):
    n = space.n
    nn = 2*n
    #g = Sp[3]
    for g in Sp:
        print()
        print("="*79)
        print(g.name)
        print(g)
        row = nn-1
        #rows = [3, 1, 0, 2]
        pivots = []
        print("row elimination ================")
        cols = list(range(nn))
        while row>=0:
        #for irow, row in enumerate(rows):
            print("row =", row)
            print(g)
            #for col in range(nn):
            for col in cols:
                if g[row, col]:
                    break
            else:
                assert 0
            pivots.append((row, col))
            cols.remove(col)
            print("pivot:", pivots[-1])
            row1 = row-1
            while row1 >= 0:
            #for row1 in rows[irow+1:]:
                if g[row1, col] and E.get((row1, row)) is not None:
                    print("reduce", row1, "<---", row)
                    op = E[row1, row]
                    g = op*g
                    print(g)
                row1 -= 1
            row -= 1
        print("pivots:", pivots)
        print("col elimination ================")
        for (row, col) in pivots:
            for col1 in range(col+1, nn):
                if g[row, col1]:
                    print("reduce", col, "<---", col1)
                if g[row, col1] and E.get((col1, col)) is not None:
                    print("reduce", col, "<---", col1)
                    op = E[col1, col]
                    g = g*op
                    print(g)
    
        if not g in W:
            print("FAIL:")
            print("w =")
            #print(space.get_H(0) * space.get_perm([1,0]) * space.get_H(0))
            print(space.get_perm([1,0])*space.get_H(0))
        assert g in W

#    for g in Sp[:5]:
#        print()
#        print(normal[g].name)
#        space.bruhat_decompose(g)


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


