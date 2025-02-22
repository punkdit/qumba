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

from qumba.lin import (shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2,
    kernel, span)
from qumba.lin import int_scalar as scalar
from qumba.action import mulclose, mulclose_find
from qumba.matrix import Matrix, DEFAULT_P, pullback


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
        #self.I = self.get_identity()

    def __lshift__(self, other):
        assert isinstance(other, SymplecticSpace)
        assert other.p == self.p
        return SymplecticSpace(self.n + other.n, self.p)
    __add__ = __lshift__

    def is_isotropic(self, H):
        assert isinstance(H, Matrix)
        nn = 2*self.n
        F = self.F
        return (H*F*H.t).sum() == 0

    def is_symplectic(self, M):
        assert isinstance(M, Matrix)
        nn = 2*self.n
        F = self.F
        assert M.shape == (nn, nn)
        return F == M*F*M.transpose()

    def parse(self, s, name="?"):
        assert type(s) is str, repr(s)
        rows = s.strip().split()
        m = len(rows)
        n = len(rows[0])
        assert n==self.n, "do we care?"
        A = zeros2(m, 2*n)
        for i,row in enumerate(rows):
            assert len(row) == n
            for j,c in enumerate(row):
                if c=='X' or c=='Y':
                    A[i, 2*j] = 1
                if c=='Z' or c=='Y':
                    A[i, 2*j+1] = 1
        return Matrix(A, self.p, name=name)
    fromstr = parse

    def get_identity(self):
        A = identity2(self.nn)
        M = Matrix(A, self.p, name=())
        return M
    I = get_identity # ??

    def get_perm(self, f):
        n, nn = self.n, 2*self.n
        assert len(f) == n, "len(%s) != %s"%(f, n)
        assert set([f[i] for i in range(n)]) == set(range(n))
        name = "P(%s)"%(",".join(str(i) for i in f))
        A = zeros2(nn, nn)
        for i in range(n):
            A[2*i, 2*f[i]] = 1
            A[2*i+1, 2*f[i]+1] = 1
        M = Matrix(A, self.p, None, name)
        assert self.is_symplectic(M)
        return M

    def get_P(self, *args):
        return self.get_perm(args)
    P = get_P

    def get_SWAP(self, i=0, j=1):
        assert i!=j
        f = list(range(self.n))
        f[i], f[j] = f[j], f[i]
        return self.get_perm(f)
    SWAP = get_SWAP

    def get(self, M, idx=None, name="?"):
        assert M.shape == (2,2)
        assert isinstance(M, Matrix)
        n = self.n
        A = identity2(2*n)
        idxs = list(range(n)) if idx is None else [idx]
        name = tuple("%s(%d)"%(name,i) for i in idxs)
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
        #name = "H(%s)"%idx
        return self.get(H, idx, "H")
    H = get_H

    def get_S(self, idx=None):
        # swap X<-->Y
        S = Matrix([[1,1],[0,1]], name="S")
        #name = "S(%s)"%idx
        return self.get(S, idx, "S")
    S = get_S

    def get_SH(self, idx=None):
        # X-->Z-->Y-->X 
        return self.S(idx)*self.H(idx)

    def get_CZ(self, idx=0, jdx=1):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*idx, 2*jdx+1] = 1
        A[2*jdx, 2*idx+1] = 1
        A = A.transpose()
        name="CZ(%d,%d)"%(idx,jdx)
        return Matrix(A, self.p, None, name)
    CZ = get_CZ

    def get_CX(self, idx=0, jdx=1):
        assert idx != jdx
        n = self.n
        A = identity2(2*n)
        A[2*jdx+1, 2*idx+1] = 1
        A[2*idx, 2*jdx] = 1
        A = A.transpose()
        name="CX(%d,%d)"%(idx,jdx)
        return Matrix(A, self.p, None, name)
    get_CNOT = get_CX
    CX = get_CX

    def get_CY(self, idx=0, jdx=1):
        assert idx != jdx
        g = self.CX(idx, jdx)
        s = self.S(jdx)
        return s*g*s # s==s^-1 here

    def get_expr(self, expr):
        if expr == ():
            op = self.get_identity()
        elif type(expr) is tuple:
            op = reduce(mul, [self.get_expr(e) for e in expr]) # recurse
        else:
            expr = "self.get_"+expr
            op = eval(expr, {"self":self})
        return op

    @cache
    def get_borel(self, verbose=False):
        # _generate the Borel group
        n = self.n
        gen = []
        for i in range(n):
            gen.append(self.get_S(i))
            for j in range(i+1, n):
                gen.append(self.get_CX(i, j))
                gen.append(self.get_CZ(i, j))
    
        G = mulclose(gen, verbose=verbose)
        return G

    def get_weyl_gen(self):
        n = self.n
        I = self.get_identity()
        gen = [I]
        for i in range(n):
            gen.append(self.get_H(i))
            for j in range(i+1, n):
                perm = list(range(n))
                perm[i], perm[j] = perm[j], perm[i]
                gen.append(self.get_perm(perm))
        return gen
    
    @cache
    def get_weyl(self, verbose=False):
        # _generate the Weyl group
        gen = self.get_weyl_gen()
        G = mulclose(gen, verbose=verbose)
        return G

    def find_weyl(self, w):
        f = w.to_perm()
        src = [(2*i, 2*i+1) for i in range(self.n)]
        tgt = list(zip(f[::2], f[1::2]))
        perm = [] # qubit permutation
        flips = [] # Hadamard
        for (i,j) in tgt:
            if (i,j) in src:
                idx = src.index((i,j))
            else:
                assert (j,i) in src
                idx = src.index((j,i))
                flips.append(self.get_H(idx))
            perm.append(idx)
        if perm == list(range(self.n)):
            w0 = self.get_identity()
        else:
            w0 = self.get_perm(perm)
        w1 = reduce(mul, flips) if flips else self.get_identity()
        assert w0*w1 == w
        w = w0*w1
        return w

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

    @cache
    def get_building(self):
        building = Building(self)
        return building

    def decompose(self, sop):
        building = self.get_building()
        l, w, r = building.decompose(sop)
        return l, w, r

    def get_name(self, sop):
        l, w, r = self.decompose(sop)
        return l.name + w.name + r.name

    def render_expr(self, name, *args, **kw):
        from huygens.zx import Circuit
        c = Circuit(self.n)
        cvs = c.render_expr(name, *args, **kw)
        return cvs

    def render(self, sop, *args, **kw):
        l, w, r = self.decompose(sop)
        name = l.name + w.name + r.name
        cvs = self.render_expr(name, *args, **kw)
        return cvs

    def translate_clifford(self, sop, verbose=False):
        """
        _translate symplectic matrix sop to 2**n by 2**n clifford unitaries
        """
    
        l, w, r = self.decompose(sop)
        if verbose:
            print("translate_clifford:")
            print("\t", l.name)
            print("\t", w.name)
            print("\t", r.name)
    
        from qumba.clifford import Clifford
        cliff = Clifford(self.n)
        l = cliff.get_expr(l.name)
        r = cliff.get_expr(r.name)
        w = cliff.get_expr(w.name)
    
        if 0:
            print(w)
            f = w.to_perm()
            from qumba.matrix import Matrix
            p = Matrix.perm(f)
            assert w == p
            print(p.name) # no it's not a perm, it's got H and perm, FAIL
            w = cliff.get_expr(p.name)
            print(w)
    
        # clifford unitary
        cop = l*w*r
    
        return cop

    @cache
    def get_majorana_modes(self):
        F = self.F
        n, nn = self.n, self.nn
        Xs, Zs = [], []
        for i in range(n):
            Xs.append(2*i)
            #Zs.append(n + n-i-1) # ziporder
            Zs.append(2*i+1)
        modes = []
        for i in range(n):
            u = numpy.zeros((nn,), dtype=scalar)
            v = numpy.zeros((nn,), dtype=scalar)
            for j in range(i-1):
                u[Zs[j]] = 1
                v[Zs[j]] = 1
            u[Xs[i]] = 1
            v[Xs[i]] = 1
            v[Zs[i]] = 1
            #u.shape = 1,nn
            #v.shape = 1,nn
            u, v = Matrix(u), Matrix(v)
            #print()
            #print(u)
            #print(v)
            #print(F*v.transpose())
            uv = u*F*v.transpose()
            #print(uv)
            assert uv == 1, uv
            modes.append(u)
            modes.append(v)
        modes = numpy.array(modes, dtype=scalar)
        modes = Matrix(modes) 
        modes_i = modes.pseudo_inverse()
        return modes_i

    def is_majorana(self, H):
        modes_i = self.get_majorana_modes()
        U = modes_i * H.transpose()
        U2 = U.sum(0).A % 2
        return U2.sum() == 0

    @cache
    def local_clifford_group(self):
        assert self.n < 10, "too big?"
        gen = [self.get_identity()]
        for i in range(self.n):
            gen.append(self.get_S(i))
            gen.append(self.get_H(i))
        G = mulclose(gen)
        return G

    def grassmannian(self, k):
        from qumba.sp_pascal import grassmannian
        V = uturn_to_zip(self.n)
        for U in grassmannian(self.n, k):
            A = Matrix(U.A)
            yield A*V


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




class Building(object):
    """
        Provide an interface to qumba.building _which uses a
        different symplectic form (the uturn form).
    """
    def __init__(self, space):
        from qumba.building import Algebraic, Building
        n = space.n
        uturn = Algebraic.Sp(2*n)
        building = Building(uturn)
        I = space.get_identity()
        #B = space.get_borel()
        #if space.n <= 6:
            #W = space.get_weyl()
            #W = dict((w,w) for w in W)
        #else:
            #W = None
        ops = [space.get_S(i) for i in range(n)]
        ops += [space.get_CX(i, j) for i in range(n) for j in range(i+1,n)]
        ops += [space.get_CZ(i, j) #*space.get_H(i)*space.get_H(j)
            for i in range(n) for j in range(i+1,n)]
    
        # send "E(i,j)" names --> S, CX, CZ in symplectic ziporder
        rename = {"I":I}
        lookup = uturn.get_borel().lookup # these are the "E(i,j)" borel's
        for key in lookup.keys():
            M = lookup[key]
            name = M.name[0]
            assert name[0] == "E"
    
            M = uturn.to_ziporder(M)
            assert space.is_symplectic(M)
    
            #Mi = space.invert(M)
            #print(key, lookup[key].name,
            #    ops[ops.index(M.t)].name[0], ops[ops.index(Mi.t)].name[0])
            assert M.t in ops
            rename[name] = ops[ops.index(M.t)]
            #print(name, "-->", rename[name].name)
        self.n = n
        self.uturn = uturn
        self.building = building
        self.space = space
        #self.W = W
        self.rename = rename
    
    def convert(self, op): 
        " uturn borel --> zip borel (transpose) "
        #print(op.name)
        space = self.space
        uturn = self.uturn
        rename = self.rename
        assert uturn.is_symplectic(op)
        op = reduce(mul, [rename[name].transpose() for name in op.name])
        assert space.is_symplectic(op)
        return op

    def invert(self, op):
        """ for inverse just reverse the order of the generators,
        because these are self-inverse """
        space = self.space
        uturn = self.uturn
        assert space.is_symplectic(op)
        name = tuple(reversed(op.name))
        op = space.get_expr(name)
        return op

    def decompose(self, _g):

        g = _g.t # work with transpose

        space = self.space
        n = self.n
        uturn = self.uturn
        assert space.is_symplectic(g)
        g1 = uturn.from_ziporder(g)
        gi = uturn.invert(g1)
        I = space.get_identity()
        assert g1 * gi == I
        left, right = self.building.decompose(g1)
        w = left*g1*right # Weyl element
 
        w = uturn.to_ziporder(w)
        assert space.is_symplectic(w)

        w = space.find_weyl(w)
 
        l = uturn.to_ziporder(left)
        r = uturn.to_ziporder(right)
        assert l * g * r == w
 
        left = self.convert(left)
        right = self.convert(right)
        assert left == l
        assert right == r
 
        for op in [left, right]:
            assert (op == space.get_expr(op.name))
        #if self.W is not None:
        #    assert (w == space.get_expr(w.name))
 
        li = self.invert(left)
        ri = self.invert(right)
        assert g == li * w * ri

        return ri.t, w.t, li.t # undo transpose



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
    gen.append(space.get_CX(0, 2) * space.get_CX(1, 3))
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

    n = 2
    space = SymplecticSpace(n)
    building = Building(space)
    for trial in range(10):
        g = space.sample_sp()
        l, w, r = building.decompose(g)
        print(g)
        print(l.name)
        print("\t", w.name)
        print("\t", r.name)
        assert g == l*w*r
        for op in [l, w, r]:
            assert op == space.get_expr(op.name)



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


