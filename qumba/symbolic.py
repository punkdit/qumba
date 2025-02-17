#!/usr/bin/env python

from math import sin, cos, pi
from functools import reduce
from operator import matmul

import numpy
from numpy import array
from scipy import optimize 


from bruhat.comonoid import dot, tensor, System
from bruhat.sympy import Const

from qumba import construct
from qumba.qcode import QCode
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
    z = system.get_unknown((2,))
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
    return "%.6f"%a

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
        
    


class Solver(System):
    def __init__(self):
        System.__init__(self, 2)

#    def get_unknown(self, shape, name='v'):
#        A = numpy.empty(shape, dtype=object)
#        for idx in numpy.ndindex(shape):
#            A[idx] = self.get_var(name)
#        self.items.append(A)
#        return A

    def get_scalar(self, name="z"):
        a = self.get_var(name+"a")
        b = self.get_var(name+"b")
        return Complex(a, b)

    def get_unknown(self, shape, name='v'):
        A = numpy.empty(shape, dtype=object)
        for idx in numpy.ndindex(shape):
            A[idx] = self.get_scalar(name)
        A = Matrix(A)
        self.items.append(A)
        return A

    def get_phase(self, name='v'):
        A = numpy.empty((2,2), dtype=object)
        A[:] = Complex(Const(0), Const(0))
        A[0,0] = Complex(Const(1), Const(0))
        A[1,1] = self.get_scalar(name)
        A = Matrix(A)
        self.items.append(A)
        return A

    def add_scalar(self, lhs, rhs=Complex(0)):
        eqs = self.eqs
        z = lhs-rhs
        assert isinstance(z, Complex)
        if not z.a.is_zero():
            eqs.append(z.a)
        if not z.b.is_zero():
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
    z = solver.get_scalar()
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
    U = solver.get_unknown((2,2))
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
    

def get_projector(code):

    print("get_projector")
    N = 2**code.n
    M = 2**code.m
    #P = M*code.get_projector()
    P = code.get_average_operator()

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

    return P


#def find_transversal(code):


def main_find():

    if argv.code==(5,1,3):
        code = construct.get_513() # doesn't find the SH ... hmm..
    elif argv.code==(4,2,2):
        code = construct.get_422() # finds the Hadamard
    elif argv.code==(4,1,2):
        code = construct.get_412()
    elif argv.code==(7,1,3):
        code = construct.get_713() 
    elif argv.code==(8,3,2):
        code = construct.get_832()
    elif argv.code==(10,1,4):
        code = QCode.fromstr("""
        YYZIZIIZIZ
        ZYYZIZIIZI
        IZYYZIZIIZ
        ZIZYYZIZII
        IZIZYYZIZI
        IIZIZYYZIZ
        ZIIZIZYYZI
        IZIIZIZYYZ
        ZIZIIZIZYY
        """)
    else:
        return

    print(code)
    #print(code.longstr())
    P = get_projector(code)

    return

    found = set()
    solver = Solver()

    I = Matrix.identity(2)

    print("g")
    if 0:
        U = solver.get_phase()
        g = reduce(matmul, [U]*code.n)
    elif 0:
        U = solver.get_unknown((2,2))
        solver.add( U*U.d , I )
        #solver.add(U*U*U, I)
        g = reduce(matmul, [U]*code.n)
    else:
        ops = [solver.get_phase() for i in range(code.n)]
        g = reduce(matmul, ops)

    print("solver.add")
    solver.add( P*g , g*P )

    print("solve")
    #for eq in solver.eqs:
    #    print(eq)
    ##f = solver.py_func(verbose=True)
    #return
    
    while 1:
        items = solver.solve(trials=100, jac=False)

        s = str([u[1,1] for u in items])
    
        if s in found:
            continue
        found.add(s)
        print(s)
    


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






