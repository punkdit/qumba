#!/usr/bin/env python

from math import sin, cos, pi
from functools import reduce
from operator import matmul

import numpy
from numpy import array
from scipy import optimize 


from bruhat.comonoid import dot, tensor, System
from bruhat import comonoid 

from qumba import construct
from qumba.argv import argv
from qumba.smap import SMap


"""
Here we are doing the complex algebra over the reals.

(a + ib) * (c + id)
= a*c - b*d + i(b*c + a*d)
= [a*c-b*d  b*c+a*d]

so the multiply matrix is:
   ac ad bc bd
  [1, 0, 0, -1], --> real
  [0, 1, 1,  0]] --> imag

"""




def eq(a, b):
    return numpy.allclose(a, b)

def test_root():
    
    one = array([1, 0])
    imag = array([0, 1])
    
    # complex multiply
    M = array([
        [1, 0, 0, -1],
        [0, 1, 1,  0]])
    
    def mul(a, b):
        return dot(M, tensor(a, b))
    
    def conj(a):
        assert a.shape == (2,)
        b = a.copy()
        b[1] *= -1
        return b
    
    
    assert eq( dot(M, tensor(imag, imag) ), -one )

    # find a fifth root of unity

    system = System(2)
    z = system.unknown((2,))
    print(z)

    z2 = mul(z,z)
    z3 = mul(z2,z)
    z4 = mul(z3,z)
    z5 = mul(z4,z)

    print(z5.shape)

    system.add(z5, one)
    system.add(mul(z, conj(z)), one)

    #v = system.get_root(x0=[2,3])
    items = system.root()
    v = items[0]

    print("solution:")
    print(v, type(v))

    theta = 2*pi / 5
    for i in range(5):
        print(sin(i*theta), cos(i*theta))
    

EPSILON = 1e-8

def snum(a):
    if not isinstance(a, numpy.number):
        return a
    for v in [-1, 0, 1]:
        if abs(a-v) < EPSILON:
            return v
    return a

class Complex:
    def __init__(self, a, b=0):
        self.a = a
        self.b = b
    def __str__(self):
        a = snum(self.a)
        b = snum(self.b)
        return "(%s+i*%s)"%(a, b)
    #def __repr__(self):
    #    return "Complex(%s,%s)"%(self.a, self.b)
    __repr__ = __str__
    def __eq__(self, other):
        #return self.a==other.a and self.b==other.b
        err = (self.a-other.a)**2 + (self.b-other.b)**2
        return err<EPSILON
    def __add__(self, other):
        return Complex(self.a+other.a, self.b+other.b)
    def __sub__(self, other):
        return Complex(self.a-other.a, self.b-other.b)
    def __neg__(self):
        return Complex(-self.a, -self.b)
    def __mul__(self, other):
        a = self.a*other.a - self.b*other.b
        b = self.b*other.a + self.a*other.b
        return Complex(a, b)
    def __pow__(self, n):
        if n==0:
            return Complex(1,0)
        a = self
        while n>1:
            a = a*self
            n -= 1
        return a
    def conj(self):
        return Complex(self.a, -self.b)
    @classmethod
    def promote(cls, item):
        if isinstance(item, Complex):
            return item
        return Complex(item)
    def subs(self, values):
        #print("Complex", self, values)
        a = self.a.subs(values)
        b = self.b.subs(values)
        return Complex(a, b)


class Matrix:
    def __init__(self, A):
        A = numpy.array(A, dtype=object)
        for idx in numpy.ndindex(A.shape):
            A[idx] = Complex.promote(A[idx])
        self.A = A
        self.shape = A.shape

    def __str__(self):
        #s = smap
        return "Matrix(\n%s)"%str(self.A)
    __repr__ = __str__

    def copy(self):
        return Matrix(self.A.copy())

    def __eq__(self, other):
        return numpy.all(self.A == other.A)

    @classmethod
    def identity(self, n):
        A = numpy.identity(n)
        return Matrix(A)

    def __mul__(self, other):
        assert isinstance(other, Matrix)
        A = numpy.dot(self.A, other.A)
        return Matrix(A)

    def __matmul__(self, other):
        A = numpy.kron(self.A, other.A)
        return Matrix(A)

    def __getitem__(self, idx):
        return self.A[idx]

    def __setitem__(self, idx, value):
        self.A[idx] = Complex.promote(value)

    @property
    def d(self):
        A = self.A.transpose()
        for idx in numpy.ndindex(A.shape):
            A[idx] = A[idx].conj()
        A = Matrix(A)
        return A
        
    


class Solver(comonoid.System):
    def __init__(self):
        comonoid.System.__init__(self, 2)

#    def unknown(self, shape, name='v'):
#        A = numpy.empty(shape, dtype=object)
#        for idx in numpy.ndindex(shape):
#            A[idx] = self.get_var(name)
#        self.items.append(A)
#        return A

    def unknown(self, shape, name='v'):
        A = numpy.empty(shape, dtype=object)
        for idx in numpy.ndindex(shape):
            A[idx] = self.scalar(name)
        A = Matrix(A)
        self.items.append(A)
        return A

    def scalar(self, name="z"):
        a = self.get_var(name+"a")
        b = self.get_var(name+"b")
        return Complex(a, b)

    def add_scalar(self, lhs, rhs=Complex(0)):
        eqs = self.eqs
        z = lhs-rhs
        assert isinstance(z, Complex)
        if z.a != 0:
            eqs.append(z.a)
        if z.b != 0:
            eqs.append(z.b)

    def add(self, lhs, rhs):
        assert isinstance(lhs, Matrix)
        assert isinstance(rhs, Matrix)
        assert rhs.shape == lhs.shape
        eqs = self.eqs
        for idx in numpy.ndindex(lhs.shape):
            self.add_scalar(lhs[idx], rhs[idx])
        #print(len(eqs), "eqs")

    def _solve(self):
        n = len(self.vs)
        x0 = list(numpy.random.normal(size=n))
        v = self.get_root(x0=x0)
        #items = solver.root()
        #v = items[0]
        return v

    def subs(self, x):
        return x

    def subs(self, values=None, x=None, dtype=float):
        if values is None:
            values = dict((str(v), xi) for (v, xi) in zip(self.vs, x))
        items = [A.copy() for A in self.items]
        for i, A in enumerate(items):
            for idx in numpy.ndindex(A.shape):
                A[idx] = A[idx].subs(values)
        #items = [A.astype(dtype) for A in items]
        return items

    def solve(self, *args, **kw):
        items = self.root(*args, **kw)
        return items



def test():
    # find a fifth root of unity

    one = Complex(1,0)

    solver = Solver()
    z = solver.scalar()
    print(z)

    solver.add_scalar(z**5, one)
    solver.add_scalar(z*z.conj(), one) # _redundant

    if 0:
        print("solution:")
        print(v, type(v))
    
        theta = 2*pi / 5
        for i in range(5):
            print(sin(i*theta), cos(i*theta))


    solver = Solver()

    I = Matrix.identity(2)
    U = solver.unknown((2,2))
    solver.add( U*U.d , I )
    solver.add( U*U*U , I )

    UU = U@U
    assert UU.shape == (4,4)

    items = solver.solve(trials=100)
    u = items[0]

    #print("identity:", u==I)
    #print(u)
    uuu = u*u*u
    assert (uuu==I)
    assert u*u.d == I
    



def main_find():

    code = construct.get_513() # doesn't find the SH ... hmm..
    #code = construct.get_713() # too big
    #code = construct.get_422() # finds the Hadamard
    #code = construct.get_412()
    print(code)
    #print(code.longstr())

    N = 2**code.n
    M = 2**code.m
    P = M*code.get_projector()

    A = numpy.zeros((N,N), dtype=object)
    for idx in numpy.ndindex((N,N)):
        val = P.M[idx]
        z = val.complex_embedding()
        a = float(z.real())
        b = float(z.imag())
        A[idx] = Complex(a, b)
        #try:
        #    A[idx] = float(P.M[idx])
        #except:
        #    print(P.M[idx])
        #    raise
    P = Matrix(A)
    assert P.shape == (N,N)

    found = set()
    while 1:
        solver = Solver()
    
        I = Matrix.identity(2)
        U = solver.unknown((2,2))
        solver.add( U*U.d , I )
        #solver.add(U*U*U, I)
    
        g = reduce(matmul, [U]*code.n)
        #print(g.shape)
        assert g.shape == (N, N)
    
        solver.add( P*g , g*P )
    
        items = solver.solve(trials=100, jac=False)
        u = items[0]
    
        s = str(u)
        if s in found:
            continue
        found.add(s)
        print(s)
        print("identity:", u==I)
    


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






