#!/usr/bin/env python

"""
Evaluate phase free ZX-diagrams using a SAT solver .
"""


import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add, mul
from random import random, randint, choice, shuffle

import z3

import numpy

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.qcode import strop, QCode
from qumba import construct 
from qumba.matrix_sage import Matrix
from qumba.clifford import Clifford, w8, w4, r2, ir2, half
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
        #left = self.left
        left = [self.left[i] for i in idxs]
        #right = [self.right[i] for i in idxs]
        right = self.right
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

    def get_identity(self, n):
        return Perm(self, list(range(n)))

    def fuse(self, lop, rop, pairs):
        #print("fuse", pairs)
        pairs = list(pairs)
        pairs.sort()
        for (i,j) in pairs:
            assert 0<=i<lop.n, (i, lop)
            assert 0<=j<rop.m, (j, rop)
            # etc
        N = len(pairs)
        M = lop.n+rop.m-N
        perm = {i:None for i in range(M)}
        lookup = dict(pairs)
        jdx = 0
        for idx in range(lop.n):
            if idx in lookup:
                perm[idx] = lookup[idx]+lop.n-N
            else:
                perm[idx] = jdx
                jdx += 1
        #print(perm)
        assert jdx == lop.n - N
        values = set(lookup.values())
        jdx = 0
        for idx in range(rop.m):
            if idx in values:
                pass
            else:
                assert perm[jdx+lop.n] is None
                perm[jdx+lop.n] = idx+lop.n-N
                jdx += 1
        perm = {j:i for (i,j) in perm.items()} # reversed
        perm = [perm[i] for i in range(M)]
        #print("perm:", perm)
        lhs = lop @ self.get_identity(rop.m-N)
        #print("\t", rop.m-N, lop.n-N)
        perm = self.perm(*perm)
        rhs = self.get_identity(lop.n-N) @ rop
        return lhs * perm * rhs

    def CX(self, n, ctrl, tgt):
        assert ctrl != tgt
        ident = lambda n=1: self.get_identity(n)
        if tgt < ctrl:
            tgt, ctrl = ctrl, tgt
            lop = self.red(1,2)
            rop = self.green(2,1)
        else:
            lop = self.green(1,2)
            rop = self.red(2,1)
        lhs = ident(ctrl) @ lop @ ident(n-ctrl-1)
        rhs = ident(tgt) @ rop @ ident(n-tgt-1)
        assert lhs.n == rhs.m
        #print("CX", n, ctrl, tgt)
        #print(lhs.shape, rhs.shape)
        assert ctrl < tgt
        pairs = [(i,i) for i in range(ctrl+1)]
        pairs.append((ctrl+1, tgt))
        pairs += [(i+1,i) for i in range(ctrl+1, tgt)]
        pairs += [(i+1,i+1) for i in range(tgt, n)]
        #print("pairs:", pairs)
        assert len(pairs) == lhs.n 
        op = self.fuse(lhs, rhs, pairs)
        return op

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
            for bit,v in enumerate(reversed(left)):
                if values[v]:
                    idx += 2**bit
            jdx = 0
            for bit,v in enumerate(reversed(right)):
                if values[v]:
                    jdx += 2**bit
            assert A[idx, jdx] == 0, "wut"
            A[idx, jdx] = 1
            solver.add(z3.Not(z3.And(*term)))
            #yield values
        A = Matrix(clifford.K, A)
        return A
    


def XXXget_projector(n):

    rr_r = r2*red(2,1)
    op = reduce(matmul, [rr_r]*n)

    perm = [2*i for i in range(n)]+[2*i+1 for i in range(n)]
    perm = Clifford(2*n).get_P(*perm)
    perm = perm.t

    op = perm*op
    lhs = Clifford(n).get_identity() @ green(0, n)
    op = lhs*op
    return op


def get_projector(code):
    assert code.is_css()
    n = code.n
    assert n < 12

    css = code.to_css()
    Hx = css.Hx
    mx = Hx.shape[0]
    Hz = css.Hz
    mz = Hz.shape[0]

    space = Space()
    green = space.green
    red = space.red
    perm = space.perm
    ident = space.get_identity

    P = ident(n)

    for (H, rmeth, gmeth) in [
        (Hx, red, green),
        (Hz, green, red),
    ]:
      for h in H:
        rhs = [rmeth(2,1) if h[i] else ident(1) for i in range(n)]
        rhs = reduce(matmul, rhs)
        lhs = gmeth(0, h.sum())

        pairs = []
        idx = 0
        for j in range(n):
            if h[j] == 0:
                continue
            pairs.append((idx, j+idx))
            idx += 1
        #print("pairs:", pairs)
        op = space.fuse(lhs, rhs, pairs)
        assert op.shape == (n, n)

        P = op*P

    P = P.get()
    P = (half**len(Hx))*P

    return P





def test():

    space = Space()
    get_identity = space.get_identity
    g = space.green
    r = space.red
    fuse = space.fuse
   
    # XXX cannot re-use these guys more than once 
    # in the same operator XXX
    I = g(1, 1)
    g_gg = g(1, 2)
    gg_g = g(2, 1)
    r_rr = r(1, 2)
    rr_r = r(2, 1)
    rr_rr = r(2, 2)

    op = gg_g*g_gg
    op = op * rr_rr

    assert rr_r*r_rr == rr_rr
    assert g_gg*gg_g == I
    assert I == r(1, 1)

    for trial in range(5):
        n = 4
        idxs = list(range(n))
        shuffle(idxs)
        P = space.perm(*idxs)
        lhs = P.get()
        rhs = Clifford(n).get_P(*idxs)
        assert lhs == rhs

    assert fuse(gg_g, r_rr, []) == gg_g@r_rr
    assert fuse(gg_g, r_rr, [(0,0)]) == gg_g*r_rr

    lhs = fuse(g_gg, rr_r, [(1,0)])
    lhs = lhs.get()
    assert lhs == Clifford(2).CX(0,1)

    op = (g_gg@I).get() * (I@rr_r).get()
    assert lhs == op

    lhs = g_gg @ get_identity(2)
    P = space.perm(0,2,1,3)
    rhs = get_identity(2) @ rr_r
    op = lhs*P*rhs
    A = op.get()
    assert A == Clifford(3).CX(0,2)

    B = space.CX(3, 0, 2)
    B = B.get()
    assert A == B

    n = 4
    for i in range(n):
      for j in range(i+1,n):
        lhs = space.CX(n, i, j)
        lhs = lhs.get()
        if lhs != Clifford(n).CX(i, j):
            print(lhs)
            assert 0

        lhs = space.CX(n, j, i)
        lhs = lhs.get()
        if lhs != Clifford(n).CX(j, i):
            print(lhs)
            assert 0

    for code in [
        construct.get_422(),
        QCode.fromstr("XXII IIXX ZZZZ"),
        QCode.fromstr("ZZII IIZZ XXXX"),
        construct.get_713(),
    ]:
        P = get_projector(code)
        Q = code.get_projector()
        assert P==Q





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


