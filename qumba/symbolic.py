#!/usr/bin/env python
"""
encoding equations as polynomials & rootfinding.

"""

from math import sin, cos, pi, atan
from functools import reduce
from operator import matmul

import numpy
from numpy import array
from scipy import optimize 


from qumba import construct
from qumba.qcode import QCode
from qumba.cyclic import get_cyclic_perms
from qumba.argv import argv
from qumba.smap import SMap
from qumba.action import mulclose



EPSILON = 1e-8
class Poly:
    "multivariate polynomials over the integers"
    def __init__(self, rank, cs):
        self.cs = dict(cs) # map tuple -> coeff
        for k,v in cs.items():
            assert v != 0, self # <----------- TODO: REMOVE ME FOR SPEED TODO 
        self.rank = rank
    @classmethod
    def get_var(cls, rank, idx):
        assert 0<=idx<rank
        key = [0]*rank
        key[idx] = 1
        return cls(rank, {tuple(key):1})
    @classmethod
    def const(cls, rank, r):
        if r == 0:
            return cls(rank, {})
        return cls(rank, {(0,)*rank:r})
    #def __str__(self):
    #    return str(self.cs)
    def __str__(self):
        cs = self.cs
        items = []
        for key,value in cs.items():
            term = []
            for (idx,i) in enumerate(key):
                if i==0:
                    continue
                if i==1:
                    term.append("v%d"%idx)
                else:
                    term.append("v%d**%d"%(idx,i))
            assert value != 0
            term = "*".join(term)
            if not term:
                items.append(str(value))
            elif value==1:
                items.append(term)
            else:
                items.append(str(value)+"*"+term)
        s = "+".join(items) or "0"
        if len(items)>1:
            s = "("+s+")"
        return s
    __repr__ = __str__
    def subs(self, ns):
        #print("subs", ns)
        vals = []
        for k,v in ns.items():
            assert k[0]=="v"
            idx = int(k[1:])
            if idx>=len(vals):
                vals += [None]*(idx-len(vals)+1)
            vals[idx] = v
        #print(vals)
        r = 0.
        for (k,v) in self.cs.items():
            s = 1.
            for (idx,i) in enumerate(k):
                if i==0:
                    continue
                s *= vals[idx]**i
            r += v*s
        return r
    def __eq__(self, other):
        if not isinstance(other, Poly):
            other = Poly.const(self.rank, rhs)
        zero = 0
        for k,v in self.cs.items():
            if other.cs.get(k, zero) != v:
                return False
        for k,v in other.cs.items():
            if self.cs.get(k, zero) != v:
                return False
        return True
    def is_zero(self):
        for k,v in self.cs.items():
            assert v != 0, self
            return False
        return True
    def __add__(lhs, rhs):
        if not isinstance(rhs, Poly):
            rhs = Poly.const(lhs.rank, rhs)
        cs = dict(lhs.cs)
        zero = 0
        for k,v in rhs.cs.items():
            v = cs.get(k, zero) + v
            if v == 0 and cs.get(k) is not None:
                del cs[k]
            elif v != 0:
                cs[k] = v
        return Poly(lhs.rank, cs)
    def __sub__(lhs, rhs):
        if not isinstance(rhs, Poly):
            rhs = Poly.const(lhs.rank, rhs)
        cs = dict(lhs.cs)
        zero = 0
        for k,v in rhs.cs.items():
            v = cs.get(k, zero) - v
            if v == 0 and cs.get(k) is not None:
                del cs[k]
            elif v != 0:
                cs[k] = v
        return Poly(lhs.rank, cs)
    def __neg__(self):
        cs = {k:-v for (k,v) in self.cs.items()}
        return Poly(self.rank, cs)
    def __mul__(lhs, rhs):
        if type(rhs) == float:
            #assert 0, "%s"%rhs
            assert abs(rhs-int(round(rhs))) < EPSILON, str(rhs)
            rhs = int(round(rhs))
        if type(rhs) == int:
            return lhs.__rmul__(rhs)
        cs = {}
        for l,u in lhs.cs.items():
          for r,v in rhs.cs.items():
            k = tuple((a+b) for (a,b) in zip(l,r))
            #cs[k] = u*v
            v = cs.get(k, 0) + u*v
            if v == 0 and cs.get(k) is not None:
                del cs[k]
            elif v != 0:
                cs[k] = v
        return Poly(lhs.rank, cs)
    def __rmul__(self, r):
        r = int(r)
        if r==0:
            return Poly(self.rank, {})
        cs = {k:r*v for (k,v) in self.cs.items()}
        return Poly(self.rank, cs)
    def __pow__(self, n):
        assert n>0
        a = self
        while n>1:
            a = a*self
            n -= 1
        return a
    


def snum(a):
    if not isinstance(a, numpy.number):
        return a
    for v in [-1, 0, 1]:
        if abs(a-v) < EPSILON:
            return v
    return "%.6f"%a

class Complex:
    def __init__(self, a, b):
        assert isinstance(a, (Poly, int, float)), (type(a), a)
        assert isinstance(b, (Poly, int, float)), (type(b), b)
        assert type(a) is type(b)
        self.a = a
        self.b = b
    def __str__(self):
        a = snum(self.a)
        b = snum(self.b)
        s = "<%s+i*%s>"%(a, b)
        if s.endswith("+i*0>"):
            s = s[:-len("+i*0>")] + ">"
        elif s.startswith("<0+"):
            s = "<" + s[len("<0+"):]
        return s
    #def __repr__(self):
    #    return "Complex(%s,%s)"%(self.a, self.b)
    __repr__ = __str__
    def __eq__(self, other):
        #assert type(self.a) == type(other.a), (type(self.a), type(other.a))
        #if isinstance(other, (int, Poly)):
        #    return self.b==0 and self.a==other
        if isinstance(other, (int,float)):
            err = (self.a-other)**2 + self.b**2
            #print("err:", err)
            return err<EPSILON
        assert isinstance(other, Complex)
        if type(self.a) != type(other.a):
            return str(self) == str(other) # hack this
        if type(self.a) is Poly: # argh.. use inheritance ??
            return self.a==other.a and self.b==other.b
        err = (self.a-other.a)**2 + (self.b-other.b)**2
        return err<EPSILON
    def __add__(self, other):
        if not isinstance(other, Complex):
            return NotImplemented
        return Complex(self.a+other.a, self.b+other.b)
    def __sub__(self, other):
        if not isinstance(other, Complex):
            return NotImplemented
        return Complex(self.a-other.a, self.b-other.b)
    def __neg__(self):
        return Complex(-self.a, -self.b)
    def __mul__(self, other):
        if not isinstance(other, Complex):
            return NotImplemented
        a = self.a*other.a - self.b*other.b
        b = self.b*other.a + self.a*other.b
        return Complex(a, b)
    def __rmul__(self, r):
        a = r*self.a 
        b = r*self.b
        return Complex(a, b)
    def __rtruediv__(self, r):
        assert 0, "check this"
        c = self.a
        d = self.b
        r = c**2 + d**2
        return Complex((c+d)/r, (c-d)/r)
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
        return Complex(item, 0)
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
        s = "Matrix(\n%s)"%str(self.A)
        s = s.replace("\n", "")
        return s
    __repr__ = __str__

    def copy(self):
        return Matrix(self.A.copy())

    def __eq__(self, other):
        return numpy.all(self.A == other.A)

    def __hash__(self):
        return hash(str(self))

    @classmethod
    def identity(self, n):
        A = numpy.identity(n)
        return Matrix(A)

    def __mul__(self, other):
        assert isinstance(other, Matrix)
        A = numpy.dot(self.A, other.A)
        return Matrix(A)

    def __rmul__(self, r):
        A = r*self.A
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
        A = A.copy()
        for idx in numpy.ndindex(A.shape):
            A[idx] = A[idx].conj()
        A = Matrix(A)
        return A
        
    def __pow__(self, n):
        assert n>0
        a = self
        while n>1:
            a = a*self
            n -= 1
        return a

    def det(self):
        A = self.A
        assert A.shape == (2,2)
        return A[0,0]*A[1,1] - A[1,0]*A[0,1]

    
# modified from bruhat.comonoid
class Solver:
    def __init__(self, rank):
        self.eqs = []
        self.idx = 0
        self.vs = [Poly.get_var(rank,i) for i in range(rank)]
        self.items = [] # list of array's
        self.rank = rank
        self.zero = self.const(0)
        self.one = self.const(1)

    def check(self):
        assert self.idx == self.rank, (self.idx, self.rank)

    def get_var(self, stem='v'):
        v = self.vs[self.idx]
        self.idx += 1
        return v

    def const(self, r):
        return Poly.const(self.rank, r)

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

    def get_const(self, real, imag=0):
        real = self.const(real)
        imag = self.const(imag)
        return Complex(real, imag)

    def get_zero(self, d=2):
        zero = self.get_const(0)
        I = [[zero for i in range(d)] for i in range(d)]
        I = Matrix(I)
        return I

    def promote(self, M):
        M = numpy.array(M)
        d = len(M)
        A = self.get_zero(d)
        for i in range(d):
          for j in range(d):
            u = M[i,j]
            if isinstance(u, (int, numpy.number)):
                A[i,j] = self.get_const(u)
            elif isinstance(u, Complex):
                A[i,j] = u
            else:
                assert 0, (u, type(u))
        return A

    def X(self):
        return self.promote([[0,1],[1,0]])

    def Z(self):
        return self.promote([[1,0],[0,-1]])

    def S(self):
        return self.promote([[1,0],[0,Complex(0,1)]])

    def get_identity(self, d=2):
        zero = self.get_const(0)
        one = self.get_const(1)
        I = [[zero for i in range(d)] for i in range(d)]
        for i in range(d):
            I[i][i] = one
        I = Matrix(I)
        return I

    def get_phase(self, name='v'):
        A = numpy.empty((2,2), dtype=object)
        zero, one = self.zero, self.one
        A[:] = Complex(zero, zero)
        A[0,0] = Complex(one, zero)
        A[1,1] = self.get_scalar(name)
        A = Matrix(A)
        self.items.append(A)
        return A

    def add_scalar(self, lhs, rhs):
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

    def py_func(self, verbose=False):
        arg = "".join(str(v)+"," for v in self.vs)
        #lines = ["def f(%s):"%arg]
        lines = ["def f(x):"]
        lines.append("  %s = x" % (arg,))
        lines.append("  value = [")
        for eq in self.eqs:
            lines.append("    %s,"%(eq,))
        lines.append("  ]")
        lines.append("  return value")
        code = '\n'.join(lines)
        code = code.replace(" 1.0*", " ")
        if verbose:
            print(code)
        ns = {}
        exec(code, ns, ns)
        return ns['f']

    def root(self, trials=1, scale=1., method="lm",  # only lm works...
            tol=1e-6, maxiter=1000, jac=False, debug=False, guess=1, verbose=0):
        from scipy.optimize import root
        n = len(self.vs)
        f = self.py_func()
        if jac:
            jac = self.py_jac()
        eol = ''
        for trial in range(trials):
            best = None
            r = None
            for _ in range(guess):
                x0 = numpy.random.normal(size=n)*scale
                fx0 = f(x0)
                value = sum(abs(y) for y in fx0)
                if best is None or value < r:
                    best = x0
                    r = value
                    if guess>1:
                        print("[%.3f]"%r, end="", flush=True)
            if guess>1:
                print()
            x0 = best
            #if verbose:
            #    print("root: ", end="", flush=True); eol='\n'
            solution = root(f, x0, method=method, jac=jac, tol=tol, options={"maxiter":maxiter})
            if not solution.success and verbose:
                print(".", end='', flush=True)
                eol = '\n'

            x = solution.x
            fx = f(x)
            for y in fx:
                if abs(y) > 1e-4:
                    break
            else:
                break
            if verbose:
                print("X", end='', flush=True)
                eol = '\n'
        else:
            print()
            return None
        print(eol, end='')
        self.solution = solution
        x = solution.x
        fx = f(x)
        for y in fx:
            if abs(y) > 1e-4:
                print("System.root FAIL: %s != 0.0" % y)
                return None
        if debug:
            print("x = ", x)
            print("f(x) =", (' '.join("%.3f"%xi for xi in f(x))))
            df = jac(x)
            print("df(x) =", str(df)[:1000]+"...")
            #sol = self.get_root(list(x))
            #print("sol:", sol)
            #x = sol
        values = self.subs(x=x)
        return values

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



def test_poly():

    #one = Complex(1)
    #i = Complex(0,1)

    rank = 3
    one = Poly(rank, {(0,0,0):1})
    zero = Poly(rank, {})

    assert one == Poly.const(3,1)

    a = Poly(rank, {(1,0,0):1})
    b = Poly(rank, {(0,1,0):1})
    c = Poly(rank, {(0,0,1):1})

    assert str(a) == "v0", str(a)
    assert str(a+b) == "(v0+v1)", str(a+b)

    assert c == Poly.get_var(3, 2)

    assert one*one == one
    assert one*a == a
    assert a+b == b+a
    assert a+b != b
    assert a+a == Poly(rank, {(1,0,0):2})
    assert a*a == Poly(rank, {(2,0,0):1})
    assert (a+b)*a == a*a + b*a
    assert (a+b)*(b+c) == a*b + a*c + b*b + b*c

    assert 3*a == a+a+a

    assert (a+b) * (a-b) == a*a - b*b
    assert (a+b)*(a+b) == a*a + 2*a*b + b*b


def test_root():
    # find a fifth root of unity

    solver = Solver(2)
    #z = solver.get_scalar()
    #print(z)
    #print(z**5)
    z = solver.get_unknown((1,1))

    one = solver.get_const(1)
    I = solver.get_identity(1)

    solver.add(z**5, I)
    solver.add(z*z.d, I) # _redundant

    for i in range(10):
        items = solver.solve(trials=100)
        z = items[0]
        u = z.d
        #print(z)
        #print(u)
        #print(z*u, z**5)

    if 0:
        theta = 2*pi / 5
        for i in range(5):
            print(sin(i*theta), cos(i*theta))

    # ----------------------------------

    solver = Solver(8)
    I = solver.get_identity(2)
    U = solver.get_unknown((2,2))
    solver.add( U*U.d , I )
    solver.add( U*U*U , I )

    UU = U@U
    assert UU.shape == (4,4)

    items = solver.solve(trials=100)
    u = items[0]

    uuu = u*u*u
    assert str(uuu) == str(I)

    # ----------------------------------

    solver = Solver(8)
    I = solver.get_identity(2)
    U = solver.get_unknown((2,2))
    solver.add( U*U.d , I )

    X = solver.X()
    Z = solver.Z()

    Pauli = mulclose([X,Z])
    print(len(Pauli))

    print(X == X)
    print(X)
    print(U*X*U.d)

    

#    items = solver.solve(trials=100)
#    u = items[0]

    

def get_projector(code):

    print("get_projector")
    N = 2**code.n
    M = 2**code.m
    P = M*code.get_projector()
    #P = code.get_average_operator()

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


def main():

    if argv.code==(5,1,3):
        code = construct.get_513() # doesn't find the SH ... hmm..
    elif argv.code==(4,2,2):
        code = construct.get_422() # finds the Hadamard
    elif argv.code==(4,1,2):
        code = construct.get_412()
    elif argv.code==(6,1,2):
        code = QCode.fromstr("""
        XX.ZZ.
        .XX.ZZ
        Z.XX.Z
        ZZ.XX.
        .ZZ.XX
        """) # cyclic
#        code = QCode.fromstr("""
#        XXXXII
#        IIXXXX
#        ZZZZII
#        IIZZZZ
#        IYIYIY
#        """)
    elif argv.code==(6,2,2):
        #code = construct.get_622()
        code = QCode.fromstr("""
        XXXXII
        IIXXXX
        ZZZZII
        IIZZZZ
        """)
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
    elif argv.code:
        code = QCode.fromstr(argv.code)
    else:
        return

    print(code)

    Q = P = get_projector(code)

    if argv.cyclic:
        assert code.is_cyclic()
        perms = get_cyclic_perms(code.n)
        for g in perms:
            if g.is_identity():
                continue
            dode = g*code
            print("is_equiv:", dode.is_equiv(code))
        Q = get_projector(dode)

    found = set()

    print("g")
    if argv.constphase:
        solver = Solver(2)
        U = solver.get_phase()
        g = reduce(matmul, [U]*code.n)
    elif argv.constdiag:
        solver = Solver(8)
        I = solver.get_identity()
        U = solver.get_unknown((2,2))
        solver.add( U*U.d , I )
        solver.add_scalar(U.det(), Complex(1,0)) # SU(2)
        g = reduce(matmul, [U]*code.n)
    elif argv.diag:
        solver = Solver(8 * code.n)
        I = solver.get_identity()
        ops = []
        for i in range(code.n):
            U = solver.get_unknown((2,2))
            solver.add( U*U.d , I )
            solver.add_scalar(U.det(), Complex(1,0)) # SU(2)
            ops.append(U)
        g = reduce(matmul, ops)
    elif argv.phase:
        solver = Solver(2*code.n)
        ops = [solver.get_phase() for i in range(code.n)]
        g = reduce(matmul, ops)
    else:
        return

    solver.check()

    #print(g)
    #return

    print("lhs, rhs")
    lhs = Q*g
    rhs = g*P
    print("solver.add")
    solver.add( lhs, rhs )

    eqs = solver.eqs
    print("vs:", solver.rank)
    print("eqs:", len(solver.eqs))
    if 0:
        n = len(eqs)
        for i in range(n):
          #print(eqs[i])
          for j in range(i+1,n):
            if eqs[i] == eqs[j]:
                print("*", end="", flush=True)
            else:
                assert str(eqs[i]) != str(eqs[j])
          #print("/", end="", flush=True)
        #print()
        #return

    print("solve")
    #for eq in solver.eqs:
    #    print(eq)
    ##f = solver.py_func(verbose=True)
    #return
    
    while 1:
        items = solver.solve(trials=100)
        if items is None:
            print("fail.")
            continue

        #s = str([u[1,1] for u in items])
        s = str([u for u in items])
    
        if s in found:
            continue
        found.add(s)
        print(s, len(found))

        #break
    

def test_clifford():
    solver = Solver(2+8)

    I = solver.get_identity()
    X = solver.X()
    Z = solver.Z()
    S = solver.S()
    #print(X)
    #print(S)
    assert S*S == Z

    #ir2 = solver.get_var()
    #one = solver.get_const(1)
    #solver.add_scalar( 2*(ir2 ** 2), one )

    ir2 = solver.get_unknown((1,1))
    H = ir2[0,0] * Matrix([[1,1],[1,-1]])
    #print(H*S*H)

    T = solver.get_unknown((2,2))
    solver.add(T*T, S)
    solver.add(T*T.d, I)

    one = solver.get_const(1)
    I = solver.get_identity(1)

    solver.add(2*(ir2**2), I)

    #print(solver.eqs)

    solver.check()
    items = solver.solve(trials=10)
    ir2, T = items
    #print(T)
    #print(T*T)

    assert T*T == S


def get_pairs():

    from qumba.clifford import get_cliff1, I
    Cliff1 = get_cliff1()

    pairs = []
    refls = [g for g in Cliff1 if g*g==I]
    count = 0
    n = len(refls)
    for i in range(n):
      g = refls[i]
      for j in range(i+1,n):
        h = refls[j]
        if g*h == -h*g:
            G = mulclose([g,h])
            assert len(G) == 8
            pairs.append((g.name,h.name))
            count += 1
    pairs.sort()
    return pairs


def main_clifford():

    pairs = get_pairs()
    print(len(pairs)) # 48

    def get_op(name):
        u = I
        for n in name:
            v = {"S":S, "H":H, "w2I":Complex(0,1)*I}[n]
            u = v*u
        return u

    for (g,h) in pairs:
        
        solver = Solver(2+8)
    
        one = solver.get_identity(1)
        I = solver.get_identity()
        X = solver.X()
        Z = solver.Z()
        S = solver.S()
        assert S*S == Z
    
        # Hadamard
        ir2 = solver.get_unknown((1,1))
        H = ir2[0,0] * Matrix([[1,1],[1,-1]])
        solver.add(2*(ir2**2), one)
    
        print(g, h)
        g = get_op(g)
        h = get_op(h)
    
        # Cliff^3
        U = solver.get_unknown((2,2))
        solver.add(U*U.d, I) # unitary
        solver.add(U*X*U.d, g)
        solver.add(U*Z*U.d, h)
        solver.add_scalar(U.det(), Complex(1,0)) # SU(2)
    
        solver.check()

        for _ in range(1):
            items = solver.solve(trials=10)
            if items is None:
                assert 0
                continue
            ir2, U = items
            assert(U*U.d == I)
            z = U[0,1]
            u = U[0,0]
            if z!=0:
                continue
            theta = atan(u.b/u.a)
            print(theta / (pi) )
            r = Complex(cos(theta), sin(-theta))
            print(U)
            U = -r*U
            print(U)
            print(U*U)
            print()

        #return


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






