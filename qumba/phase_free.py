#!/usr/bin/env python

"""
"""


import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add, mul
from random import random, randint, choice

import z3

import numpy

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.qcode import strop, QCode
from qumba import construct 
from qumba.matrix_sage import Matrix
from qumba.clifford import Clifford, w8, w4, r2, ir2
from qumba import clifford 
from qumba.action import mulclose
from qumba.lin import shortstr


class Atom:
    def __init__(self, space, m, n, left=None, right=None):
        self.space = space
        self.m = m
        self.n = n
        self.shape = (m, n)
        if left is None:
            left = tuple(space.get_var() for i in range(m))
        if right is None:
            right = tuple(space.get_var() for i in range(n))
        self.left = left
        self.right = right

    def __str__(self):
        return "%s(%d, %d)"%(self.__class__.__name__, self.m, self.n)
    __repr__ = __str__

    def __mul__(self, other):
        assert self.n == other.m
        assert self.space is other.space
        return HBox(self.space, [self, other])

    def __matmul__(self, other):
        assert self.space is other.space
        return VBox(self.space, [self, other])

    def constrain(self):
        return []

    def get(self):
        return self.space.solve(self)

    def __eq__(self, other):
        assert self.space is other.space
        return self.get() == other.get()


class Compound(Atom):
    def __init__(self, space, m, n, atoms, left, right):
        Atom.__init__(self, space, m, n, left, right)
        self.atoms = list(atoms)

    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, str(self.atoms))
    __repr__ = __str__

    def constrain(self):
        for atom in self.atoms:
            for term in atom.constrain():
                yield term


class HBox(Compound):
    def __init__(self, space, atoms):
        assert len(atoms)
        m = atoms[0].m
        n = atoms[-1].n
        left = atoms[0].left
        right = atoms[-1].right
        Compound.__init__(self, space, m, n, atoms, left, right)

    def constrain(self):
        for term in Compound.constrain(self):
            yield term
        atoms = self.atoms
        N = len(atoms)
        for i in range(N-1):
            left = atoms[i].right
            right = atoms[i+1].left
            for (l,r) in zip(left, right):
                yield (l==r)


class VBox(Compound):
    def __init__(self, space, atoms):
        m = sum([atom.m for atom in atoms], 0)
        n = sum([atom.n for atom in atoms], 0)
        left = reduce(add, [atom.left for atom in atoms])
        right = reduce(add, [atom.right for atom in atoms])
        Compound.__init__(self, space, m, n, atoms, left, right)


class Green(Atom):
    def constrain(self):
        # all variables agree
        vs = self.left + self.right
        term = z3.Or(z3.And(*vs), z3.And(*[z3.Not(v) for v in vs])) # use == instead?
        yield term


class Red(Atom):
    def constrain(self):
        # enforce a parity constraint
        vs = self.left + self.right
        N = len(vs)
        terms = []
        for bits in numpy.ndindex((2,)*N):
            if sum(bits)%2:
                continue
            term = z3.And(*[vs[i] if bits[i] else z3.Not(vs[i]) for i in range(N)])
            terms.append(term)
        term = z3.Or(*terms)
        yield term


class Perm(Atom):
    def __init__(self, space, idxs):
        m = len(idxs)
        assert len(set(idxs)) == m, idxs
        assert set(idxs) == set(range(m)), idxs
        Atom.__init__(self, space, m, m)
        self.idxs = list(idxs)
        left = self.left
        right = [self.right[i] for i in idxs]
        self.term = z3.And(*[l==r for (l,r) in zip(left, right)])

    def constrain(self):
        yield self.term


class Space:
    def __init__(self):
        self.idx = 0
        self.vs = []

    def get_var(self):
        name = "v%d"%self.idx
        self.idx += 1
        v = z3.Bool(name)
        self.vs.append(v)
        return v

    def green(self, m, n):
        return Green(self, m, n)

    def red(self, m, n):
        return Red(self, m, n)

    def perm(self, *idxs):
        return Perm(self, idxs)

    def solve(self, op):
        solver = z3.Solver()

        for term in op.constrain():
            solver.add(term)

        left = op.left
        right = op.right
        A = numpy.zeros((2**op.m, 2**op.n), dtype=int)
    
        while 1:
            result = solver.check()
            if result != z3.sat:
                break
            model = solver.model()

            values = {}
            term = []
            for v in self.vs:
                val = model.get_interp(v)
                if val is None:
                    continue
                values[v] = val
                term.append(v==val)
            idx = 0
            for bit,v in enumerate(left):
                if values[v]:
                    idx += 2**bit
            jdx = 0
            for bit,v in enumerate(right):
                if values[v]:
                    jdx += 2**bit
            A[idx, jdx] = 1
            solver.add(z3.Not(z3.And(*term)))
            #yield values
        A = Matrix(clifford.K, A)
        return A
    



def test():

    space = Space()
    g = space.green
    r = space.red
   
    g_gg = g(1, 2)
    gg_g = g(2, 1)
    r_rr = r(1, 2)
    rr_r = r(2, 1)
    rr_rr = r(2, 2)

    op = gg_g*g_gg
    op = op * rr_rr
    print(op)

    A = space.solve(r_rr)
    print(A)

    assert rr_r*r_rr == rr_rr



def get_decoder(code):
    assert code.is_css()
    n = code.n
    space = Clifford(n)

    E = code.get_clifford_encoder()
    print(E.name)

    return

    css = code.to_css()
    Hx = css.Hx
    mx = Hx.shape[0]
    Hz = css.Hz
    mz = Hz.shape[0]

    print("Hx")
    print(shortstr(Hx))
    print("Hz")
    print(shortstr(Hz))

    K = sage.CyclotomicField(4)
    _g = Matrix(K, [[1,1]])
    _r = Matrix(K, [[1,0]])
    I = Matrix.identity(K, 2)
    lhs = [_g]*mx + [_r]*mz + [I]
    print(len(lhs))
    lhs = reduce(matmul, lhs)
    print(lhs.shape)

    for i in range(mx):
        for j in range(n):
            if Hx[i, j] == 0:
                continue
            #op = space.CX(i, tgt)


def get_projector(n):

    rr_r = r2*red(2,1)
    op = reduce(matmul, [rr_r]*n)

    perm = [2*i for i in range(n)]+[2*i+1 for i in range(n)]
    perm = Clifford(2*n).get_P(*perm)
    perm = perm.t

    op = perm*op
    lhs = Clifford(n).get_identity() @ green(0, n)
    op = lhs*op
    return op



def test_projector():

    n = 3
    #code = construct.get_713()
    #Ed = get_decoder(code)

    print(green(1,1))
    print(green(1,2))
    print(green(1,3))
    

    P = get_projector(n)
    print(P)
    print(P*P==2*P)

    space = Clifford(n)
    In = space.get_identity()
    Q = In + space.get_pauli("X"*n)
    print(Q)
    print(P==Q)

    return

    op = 4*red(1,n)
    print(op, op.shape)

    for i,bits in enumerate(numpy.ndindex((2,)*n)):
        if op[0,i]:
            print(i, bits)



if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    fn = eval(name)

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            fn()
        profiler.print()

    else:
        fn()


    t = time() - start_time
    print("\nOK! finished in %.3f seconds\n"%t)


