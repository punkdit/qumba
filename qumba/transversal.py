#!/usr/bin/env python

from functools import reduce

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

from qumba.qcode import QCode, SymplecticSpace, Matrix
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
        assert value in [0, 1]
        #print("Const", value, type(value), type(value==1))
        self.value = value
    def get(self):
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

class UMatrix(object):
    def __init__(self, A=None, shape=None):
        if A is None:
            A = numpy.empty(shape, dtype=object)
            A[:] = zero
        self.A = numpy.array(A, dtype=object)
        assert shape is None or shape == A.shape
        self.shape = A.shape

    def __getitem__(self, idx):
        value = self.A[idx]
        if isinstance(value, numpy.ndarray):
            value = UMatrix(value)
        return value

#    @classmethod
#    def promote(self, item):
#        if isinstance(item, UMatrix):
#            return item

    def __mul__(self, other):
        A = numpy.dot(self.A, other.A)
        return UMatrix(A)

    def __rmul__(self, other):
        if isinstance(other, Matrix):
            A = numpy.dot(other.A, self.A)
        else:
            assert 0, other
        return UMatrix(A)

    def __ne__(self, other):
        terms = []
        #print("__ne__")
        for idx in numpy.ndindex(self.shape):
            if isinstance(other, (UMatrix, Matrix)):
                rhs = other[idx]
            else:
                rhs = other
            #print("\t", idx, self[idx], rhs)
            terms.append(self[idx] != rhs)
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

    def __str__(self):
        return str(self.A)

    def direct_sum(self, other):
        m, n = self.shape
        shape = (self.shape[0]+other.shape[0], self.shape[1]+other.shape[1])
        M = UMatrix(shape=shape)
        M.A[:m, :n] = self.A
        M.A[m:, n:] = other.A
        return M

    def get_interp(self, model):
        shape = self.shape
        A = numpy.zeros(shape, dtype=int)
        for idx in numpy.ndindex(shape):
            value = self[idx]
            if isinstance(value, Expr):
                value = value.get()
                #print(value, type(value))
                if type(value) != bool:
                    value = model.get_interp(value)
            A[idx] = bool(value)
        A = Matrix(A)
        return A
        
        

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

    A = UMatrix.unknown(2,3)
    B = UMatrix.unknown(3,2)
    AB = A*B
    Z = UMatrix(shape=(2,2))
    add(AB==Z)
    result = solver.check()
    assert str(result) == "sat"


def main():
    test()

    code = QCode.fromstr("XYZI IXYZ ZIXY")
    #code = QCode.fromstr("XXXX ZZZZ XXII")
    n = code.n
    nn = 2*n

    # ------------------------

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

    solver = Solver()
    add = solver.add

    c2 = code + code
    #c2 = code + code.apply_perm([1,2,3,0])
    #c2 = code + code.apply_perm([2,3,1,0])
    #print(c2.longstr())
    
    space = SymplecticSpace(2)
    F2 = space.F
    CZ = space.get_CZ() # target gate
    M = Matrix.identity(12).direct_sum(CZ)
    E, D = c2.get_encoder(), c2.get_decoder()
    L = E*M*D
    tgt = c2.apply(L)
    assert tgt.is_equiv(c2)
    
    H = c2.H
    L = c2.L

    if 1:
        items = []
        for i in range(n):
            U = UMatrix.unknown(4, 4)
            add(U.t*F2*U == F2) # quadratic constraint
            items.append(U)
        U = reduce(UMatrix.direct_sum, items)
        U0 = None

    else:
        U0 = UMatrix.unknown(4, 4)
        add(U0.t*F2*U0 == F2) # quadratic constraint
        U = reduce(UMatrix.direct_sum, [U0]*n)

    P = c2.space.get_perm([0,4,1,5,2,6,3,7])
    U = P.t * U * P
    #print(U)

    HU = H * U.t
    LU = L * U.t
    F = c2.space.F
    R = HU * F * L.t
    #print(R)
    add(R==0) # linear constraint
    R = HU * F * H.t
    #print(R)
    add(R==0) # linear constraint

    E, D = c2.get_encoder(), c2.get_decoder()
    LU = D*U*E
    LU = LU[-4:, -4:]
    #print(LU)
    #return
    I = Matrix.identity(4)
    add(LU!=I)
    #add(L == space.get_CNOT())

    found = set()
    count = 0
    while 1:
    #for i in range(100):
        count += 1
        result = solver.check()
        if result != z3.sat:
            break
        if count%100==0:
            print(".", end="", flush=True)
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = c2.apply(M)
        assert dode.is_equiv(c2)
        L = dode.get_logical(c2)
        if L not in found:
            print()
            print(dode.longstr())
            print((D*M*E)[-4:, -4:])
            print("M=")
            print(M)
            if U0 is not None:
                print("U0=")
                print(U0.get_interp(model))
            print("L=")
            print(L)
            found.add(L)
            add(LU != L)
            #break
    
        add(U != M)

    print("\ndone.")



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






