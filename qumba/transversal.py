#!/usr/bin/env python

from functools import reduce

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

from qumba.qcode import QCode
from qumba.argv import argv


class Expr(object):
    @classmethod
    def promote(cls, item):
        if isinstance(item, Expr):
            return item
        return Const(item)
    def __add__(self, other):
        other = self.promote(other)
        return Add(self, other)
    def __mul__(self, other):
        other = self.promote(other)
        return Mul(self, other)
    def __eq__(self, other):
        other = self.promote(other)
        return self.get() == other.get()


class Const(Expr):
    def __init__(self, value):
        assert value in [0, 1]
        self.value = value
    def get(self):
        return (self.value == 1)
    def __str__(self):
        return str(self.value)
    __repr__ = __str__

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
    def __str__(self):
        return self.name
    __repr__ = __str__


class Add(Expr):
    def __init__(self, a, b):
        self.items = (a, b)
    def get(self):
        a, b = self.items
        return Xor(a.get(), b.get())
    def __str__(self):
        return "(%s+%s)"%self.items
    __repr__ = __str__

class Mul(Add):
    def get(self):
        a, b = self.items
        return And(a.get(), b.get())
    def __str__(self):
        return "%s*%s"%self.items
    __repr__ = __str__

zero = Const(0)
one = Const(1)

class Matrix(object):
    def __init__(self, A=None, shape=None):
        if A is None:
            A = numpy.empty(shape, dtype=object)
            A[:] = zero
        self.A = numpy.array(A, dtype=object)
        assert shape is None or shape == A.shape
        self.shape = A.shape

    def __getitem__(self, idx):
        return self.A[idx]

#    @classmethod
#    def promote(self, item):
#        if isinstance(item, Matrix):
#            return item

    def __mul__(self, other):
        A = numpy.dot(self.A, other.A)
        return Matrix(A)

    def __ne__(self, other):
        terms = []
        for idx in numpy.ndindex(self.shape):
            if isinstance(other, Matrix):
                rhs = other[idx]
            else:
                rhs = other
            terms.append(self[idx] != rhs)
        term = reduce(Or, terms)
        return term

    def __eq__(self, other):
        #other = Matrix.promote(other)
        terms = []
        for idx in numpy.ndindex(self.shape):
            if isinstance(other, Matrix):
                rhs = other[idx]
            else:
                rhs = other
            terms.append(self[idx] == rhs)
        term = reduce(And, terms)
        return term

    @classmethod
    def unknown(cls, *shape):
        A = numpy.empty(shape, dtype=object)
        for idx in numpy.ndindex(shape):
            A[idx] = Var()
        return Matrix(A)

    def __str__(self):
        return str(self.A)

    def direct_sum(self, other):
        m, n = self.shape
        shape = (self.shape[0]+other.shape[0], self.shape[1]+other.shape[1])
        M = Matrix(shape=shape)
        M.A[:m, :n] = self.A
        M.A[m:, n:] = other.A
        return M

    #def get_interp(self, model):
        

def find(code):
    n = code.n
    


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

    A = Matrix.unknown(2,3)
    B = Matrix.unknown(3,2)
    AB = A*B
    Z = Matrix(shape=(2,2))
    add(AB==Z)
    result = solver.check()
    assert str(result) == "sat"


def main():
    test()

    solver = Solver()
    add = solver.add

    code = QCode.fromstr("XYZI IXYZ ZIXY")
    #print(code.get_params())
    n = code.n
    nn = 2*n

    #E = code.get_encoder()
    #print(E)

    H = code.H
    Ht = H.transpose()
    print(Ht)

    M = Matrix.unknown(1, nn)
    add(M*Ht == 0)
    add(M != 0)

    result = solver.check()
    assert str(result) == "sat"

    model = solver.model()
    for k in dir(model):
        print(k)
    for v in model.decls():
        print(v, model.get_interp(v))





if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))






