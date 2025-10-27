#!/usr/bin/env python
"""
sat / smt solving for unknown F_2 matrices, etc.
"""

from math import prod
from functools import reduce
from operator import add, matmul, mul
from random import shuffle

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, ForAll, PbEq, PbLe, PbGe

from qumba.qcode import QCode, SymplecticSpace, fromstr, shortstr, strop
from qumba.matrix import Matrix, scalar
from qumba import csscode
from qumba.action import mulclose, Group, Perm, mulclose_find
from qumba.util import allperms, cache
from qumba import equ
from qumba import construct 
from qumba.argv import argv


class Expr(object):
    @classmethod
    def promote(cls, item):
        if isinstance(item, Expr):
            return item
        return Const(item)
    def is_zero(self):
        return False
    def is_one(self):
        return False
#    def __add__(self, other):
#        other = self.promote(other)
#        return Add(self, other)
    def __radd__(self, other):
        other = self.promote(other)
        return other.__add__(self)
#    def __mul__(self, other):
#        other = self.promote(other)
#        return Mul(self, other)
    def __rmul__(self, other):
        other = self.promote(other)
        return other.__mul__(self)
    def __eq__(self, other):
        other = self.promote(other)
        return self.get() == other.get()
    def __ne__(self, other):
        other = self.promote(other)
        return self.get() != other.get()
    def __add__(self, other):
        other = self.promote(other)
        if self.is_zero():
            expr = other
        elif other.is_zero():
            expr = self
        else:
            expr = Add(self, other)
        return expr
    def __mul__(self, other):
        other = self.promote(other)
        if self.is_one():
            expr = other
        elif self.is_zero():
            expr = self
        elif other.is_one():
            expr = self
        elif other.is_zero():
            expr = other
        else:
            expr = Mul(self, other)
        return expr


class Const(Expr):
    def __init__(self, value):
        assert value == 0 or value == 1
        #print("Const", value, type(value), type(value==1))
        self.value = value
    def get(self):
        return bool(self.value == 1)
    def get_interp(self, model):
        return bool(self.value == 1)
    def __str__(self):
        return str(self.value)
    __repr__ = __str__
    def is_zero(self):
        return self.value == 0
    def is_one(self):
        return self.value == 1

class Var(Expr):
    count = 0
    def __init__(self, name=None):
        if name is None:
            name = "v%d"%self.count
        self.v = Bool(name)
        self.name = name
        Var.count += 1
    def get(self):
        return self.v
    def get_interp(self, model):
        value = model.get_interp(self.v)
        return bool(value)
    def __str__(self):
        return self.name
    __repr__ = __str__


class Add(Expr):
    def __init__(self, a, b):
        self.items = (a, b)
    def get(self):
        a, b = self.items
        return Xor(a.get(), b.get())
    def get_interp(self, model):
        a, b = self.items
        a, b = a.get_interp(model), b.get_interp(model)
        return a != b
    def __str__(self):
        return "(%s+%s)"%self.items
    __repr__ = __str__

class Mul(Add):
    def get(self):
        a, b = self.items
        return And(a.get(), b.get())
    def get_interp(self, model):
        a, b = self.items
        a, b = a.get_interp(model), b.get_interp(model)
        return a and b
    def __str__(self):
        return "%s*%s"%self.items
    __repr__ = __str__

zero = Const(0)
one = Const(1)

class UMatrix(object):
    def __init__(self, A=None, shape=None):
        if A is None:
            A = numpy.empty(shape, dtype=object)
            A[:] = zero
        if isinstance(A, Matrix):
            A = A.A
        if isinstance(A, numpy.ndarray) and A.dtype is numpy.int8:
            A = A.astype(int)
        self.A = A = numpy.array(A, dtype=object)
        assert shape is None or shape == A.shape
        self.shape = A.shape

    def __getitem__(self, idx):
        value = self.A[idx]
        if isinstance(value, numpy.ndarray):
            value = UMatrix(value)
        return value

    def __setitem__(self, idx, value):
        self.A[idx] = value

    def __len__(self):
        return len(self.A)

#    @classmethod
#    def promote(self, item):
#        if isinstance(item, UMatrix):
#            return item

    def sum(self):
        return self.A.sum()

    def __add__(self, other):
        A = self.A + other.A
        return UMatrix(A)

    def __mul__(self, other):
        A = numpy.dot(self.A, other.A)
        return UMatrix(A)

    def __matmul__(self, other):
        A = numpy.kron(self.A, other.A)
        return UMatrix(A)

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            A = numpy.dot(other.A, self.A)
        else:
            assert 0, other
        return UMatrix(A)

    def __rmatmul__(self, other):
        if isinstance(other, Matrix):
            A = numpy.kron(other.A, self.A)
        else:
            assert 0, other
        return UMatrix(A)

    def prod(self, other):
        A = self.A * other.A
        return UMatrix(A)

    def __ne__(self, other):
        terms = []
        #print("__ne__")
        for idx in numpy.ndindex(self.shape):
            if isinstance(other, (UMatrix, Matrix)):
                rhs = other[idx]
            else:
                rhs = other
            term = self[idx] != rhs
            #print("\t", idx, self[idx], rhs, type(term), )
            if not isinstance(term, numpy.bool_): # hmm, edge cases? empty terms ?
                terms.append(term)
        term = reduce(Or, terms)
        return term

    def __eq__(self, other):
        #other = UMatrix.promote(other)
        terms = []
        for idx in numpy.ndindex(self.shape):
            if isinstance(other, (UMatrix, Matrix)):
                rhs = other[idx]
            else:
                rhs = other
            #print("rhs:", rhs, type(rhs))
            terms.append(self[idx] == rhs)
            #print("done")
        term = reduce(And, terms)
        return term

    @property
    def t(self):
        A = self.A.transpose()
        return UMatrix(A)

    @classmethod
    def unknown(cls, *shape):
        A = numpy.empty(shape, dtype=object)
        for idx in numpy.ndindex(shape):
            A[idx] = Var()
        return UMatrix(A)

    @classmethod
    def get_perm(cls, solver, n):
        add = solver.add
        I = Matrix.identity(n)
        U = UMatrix.unknown(n, n)
        for i in range(n):
            for j in range(n):
                term = And(*[Not(U[i,k].get()) for k in range(n) if k!=j])
                term = Or(Not(U[i,j].get()), term)
                #print(term)
                add(term)
            term = Or(*[U[i,k].get() for k in range(n)])
            add(term)
        add( U*U.t == I )
        return U

    @classmethod
    def identity(cls, n):
        A = numpy.identity(n, dtype=int)
        return UMatrix(A)

    def __str__(self):
        return str(self.A)

    def direct_sum(self, other):
        m, n = self.shape
        shape = (self.shape[0]+other.shape[0], self.shape[1]+other.shape[1])
        M = UMatrix(shape=shape)
        M.A[:m, :n] = self.A
        M.A[m:, n:] = other.A
        return M
    __lshift__ = direct_sum

    def get_interp(self, model):
        shape = self.shape
        A = numpy.zeros(shape, dtype=int)
        for idx in numpy.ndindex(shape):
            value = self[idx]
            if isinstance(value, Expr):
                value = value.get_interp(model)
                assert type(value) is bool
                #if type(value) != bool:
                #    try:
                #        value = model.get_interp(value)
                #    except:
                #        print("value =", value, "type(value) =", type(value))
                #        raise
            A[idx] = bool(value)
        A = Matrix(A)
        return A


def orthogonal_order(n): 
    return (1 << (n//2)**2)*prod((1 << i)-1 for i in range(2, 2*((n-1)//2)+1, 2))
        

def gen_orthogonal(m):
    N = orthogonal_order(m)
    print(N)

    solver = Solver()
    add = solver.add

    Im = Matrix.identity(m)
    H = UMatrix.unknown(m,m)
    add(H*H.t == Im)

    found = set()
    gen = []

    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
        g = H.get_interp(model)
        yield g
        add(H != g)
        found.add(g)

        gen.append(g)
        G = mulclose(gen, verbose=True)
        #print(len(G))
        #bdy = []
        for g in G:
            if g in found:
                continue
            yield g
            #bdy.append(g)
            found.add(g)

        if len(G) == N:
            break
        del G

        # do we care...
        #for g in bdy:
        #    add(H != g)
        #    found.add(g)


def test_orthogonal():
    m = argv.get("m", 5)
    N = orthogonal_order(m)
    print(N)

    u = Matrix([1]*m).t

    count = 0
    for g in gen_orthogonal(m):
        count += 1
        assert g*u == u
        #print((g*u).t)
    assert count==N

    return
    
        
def test_selfdual():

    n = argv.get("n", 6)
    m = n//2

    Im = Matrix.identity(m)

    found = {}
    count = 0
    for g in gen_orthogonal(m):

        h = Im.concatenate(g, axis=1)
        assert (h*h.t).sum() == 0

        #key = get_wenum(h)
        key = h.get_wenum()
        if key not in found:
            print(key)
            found[key] = 1
        else:
            found[key] += 1

        count += 1

    print()
    for key in found.keys():
        print(key, "found:", (found[key]))
    print("count:", count)


def old_test_selfdual():

    n = argv.get("n", 6)
    m = n//2

    solver = Solver()
    add = solver.add

    H = UMatrix.unknown(m,n)
    H[:, :m] = Matrix.identity(m)
    add(H*H.t == 0)

    found = {}

    count = 0
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
        h = H.get_interp(model)
        key = get_wenum(h)
        if key not in found:
            print(key)
            found[key] = [h]
        else:
            found[key].append(h)

        add(H != h)
        count += 1

    for key in found.keys():
        print(key, "found:", len(found[key]))
    found = [len(hs) for hs in found.values()]
    print("count:", count, found)
    # counts are:
    # 1, 2, 6, 48, 720, 23040
    # https://oeis.org/A003053


def test_bijection():
    from qumba.lin import zeros2

    n = argv.get("n", 3)
    k = argv.get("k", 0)
    nn = 2*n

    count = 0
    found = set()
    for code in construct.all_codes(n, k, 1):
        count += 1
        H = code.H
        #print()
        #print(H)
        J = zeros2(n, nn+1)
        J[:, :nn] = H
        M = Matrix(J)
        #print("-"*(nn+1))
        #print(M)
        #print("-"*(nn+1))
        MMt = M*M.t
        #print(MMt)
        found.add(MMt)

        for i in range(n):
            if MMt[i,i]:
                J[i,nn] = 1
        M = Matrix(J)
        #print("-"*(nn+1))
        #print(M)
        #print("-"*(nn+1))
        MMt = M*M.t
        #print(MMt)
        
    print(count)
    print("MMt's:", len(found))
    for op in found:
        #print(op)
        assert op==op.t
        #print()



def test():
    solver = Solver()
    add = solver.add

    u = Var()
    v = Var()
    add((u+v) == 0)
    add((u*v) == 1)
    result = solver.check()
    assert str(result) == "sat"
    model = solver.model()
    assert model.evaluate(v.get())
    assert model.evaluate(u.get())


    solver = Solver()
    add = solver.add

    A = UMatrix.unknown(2,3)
    B = UMatrix.unknown(3,2)
    AB = A*B
    Z = UMatrix(shape=(2,2))
    add(AB==Z)
    result = solver.check()
    assert str(result) == "sat"

    # ------------------------

    code = QCode.fromstr("XYZI IXYZ ZIXY")
    #code = QCode.fromstr("XXXX ZZZZ XXII")
    n = code.n
    nn = 2*n

    solver = Solver()
    add = solver.add

    # find a logical operator
    H = code.H
    Ht = H.transpose()
    M = UMatrix.unknown(1, nn)
    add(M*Ht == 0)
    add(M != 0)

    result = solver.check()
    assert str(result) == "sat"

    model = solver.model()
    M = M.get_interp(model)

    # ------------------------

    print("done.")




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






