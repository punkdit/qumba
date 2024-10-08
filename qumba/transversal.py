#!/usr/bin/env python
"""
_looking for transversal logical clifford operations
"""

from functools import reduce
from operator import add, matmul, mul
from random import shuffle

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, ForAll

from qumba.qcode import QCode, SymplecticSpace, Matrix, fromstr, shortstr, strop
from qumba.action import mulclose, Group, Perm, mulclose_find
from qumba.util import allperms
from qumba import equ
from qumba import construct 
from qumba import autos
from qumba.unwrap import unwrap, Cover
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


def find_transversal(*codes, constant=False, verbose=True):
    solver = Solver()
    Add = solver.add

    m = len(codes)
    n = codes[0].n
    code = reduce(add, codes)
    H = code.H
    L = code.L
    k = code.k

    space = SymplecticSpace(m)
    Fm = space.F

    if not constant:
        items = []
        for i in range(n):
            U = UMatrix.unknown(2*m, 2*m)
            Add(U.t*Fm*U == Fm) # quadratic constraint
            items.append(U)
        U = reduce(UMatrix.direct_sum, items)
        U0 = None

    else:
        U0 = UMatrix.unknown(2*m, 2*m)
        Add(U0.t*Fm*U0 == Fm) # quadratic constraint
        U = reduce(UMatrix.direct_sum, [U0]*n)

    perm = numpy.array(list(range(n*m)))
    perm.shape = (m, n)
    perm = perm.transpose().copy()
    perm.shape = m*n
    perm = list(perm)
    P = code.space.get_perm(perm)
    U = P.t * U * P

    HU = H * U.t
    LU = L * U.t
    F = code.space.F
    R = HU * F * L.t
    Add(R==0) # linear constraint
    R = HU * F * H.t
    Add(R==0) # linear constraint

    E, D = code.get_encoder(), code.get_decoder()
    LU = D*U*E
    LU = LU[-2*k:, -2*k:]
    #I = Matrix.identity(2*k)
    #Add(LU!=I)

    found = set()
    gen = set()
    fgen = set()
    count = 0
    while 1:
        count += 1
        result = solver.check()
        if result != z3.sat:
            break
        #if count%100==0:
        #    print(".", end="", flush=True)
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = code.apply(M)
        assert dode.is_equiv(code)
        #L = dode.get_logical(code)
        L = LU.get_interp(model)
        #print(LU)
        #print(LU.get_interp(model))
        #print(L)
        #assert L == LU.get_interp(model) # whoops.. not the same..
        if L not in found:
            yield M, L
            found.add(L)
            #Add(LU != L) # slows things down..
        gen.add(M)
        Add(U != M)
        if verbose:
            print("mulclose...", end="", flush=True)
        G = mulclose(gen, verbose=verbose)
        if verbose:
            print("done")
        for g in G:
            if g not in fgen:
                #if U0 is not None:
                #    Add(U0 != g[:2*m, :2*m]) # doesn't work..
                #else:
                Add(U != g)
                fgen.add(g)
        if verbose:
            print("gen:", len(gen), "fgen:", len(fgen))


def find_local_cliffords(tgt, src=None, constant=False, verbose=True):
    #print("find_local_clifford")
    if src is None:
        src = tgt

    solver = Solver()
    Add = solver.add

    assert tgt.n == src.n
    assert tgt.k == src.k
    m = 1
    n = src.n

    space = SymplecticSpace(m)
    Fm = space.F

    if not constant:
        items = []
        for i in range(n):
            U = UMatrix.unknown(2*m, 2*m)
            Add(U.t*Fm*U == Fm) # quadratic constraint
            items.append(U)
        U = reduce(UMatrix.direct_sum, items)
        U0 = None

    else:
        U0 = UMatrix.unknown(2*m, 2*m)
        Add(U0.t*Fm*U0 == Fm) # quadratic constraint
        U = reduce(UMatrix.direct_sum, [U0]*n)

    HU = src.H * U.t
    LU = src.L * U.t
    F = src.space.F
    if tgt.k:
        R = HU * F * tgt.L.t
        Add(R==0) # linear constraint
    R = HU * F * tgt.H.t
    Add(R==0) # linear constraint

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = src.apply(M)
        assert dode.is_equiv(tgt)
        yield M

        Add(U != M)


def get_local_clifford(tgt, src, constant=False, verbose=False):
    for M in find_local_cliffords(tgt, src, constant, verbose):
        return M


def is_local_clifford_equiv(tgt, src, constant=False, verbose=False):
    for M in find_local_cliffords(tgt, src, constant, verbose):
        return True
    return False


def find_isomorphisms(code, dode=None):

    # find all automorphism permutations

    if dode is None:
        dode = code

    if code.n != dode.n or code.k != dode.k:
        return

    n = code.n
    nn = 2*n
    m = code.m
    space = code.space
    F = space.F

    solver = Solver()
    Add = solver.add

    # permutation matrix
    P = UMatrix.unknown(n, n)

    # symplectic permutation matrix
    A = numpy.zeros((nn,nn), dtype=object)
    for i in range(n):
      for j in range(n):
        A[2*i,2*j] = P[i,j]
        A[2*i+1,2*j+1] = P[i,j]
    P2 = UMatrix(A)

    for i in range(n):
      for j in range(n):
        rhs = reduce(And, [P[i,k]==0 for k in range(n) if k!=j])
        Add( Or(P[i,j]==0, rhs) )
        rhs = reduce(And, [P[k,j]==0 for k in range(n) if k!=i])
        Add( Or(P[i,j]==0, rhs) )

    for i in range(n):
        Add( reduce(Or, [P[i,j]!=0 for j in range(n)]) )

    #I = Matrix.identity(n, n)
    #Add(P * P.t == I)

    #H = code.H
    #H1 = H * P2

    R = UMatrix.unknown(m, m)
    #Add( R*H == H1 )
    Add( R*code.H == dode.H*P2 )

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        p = P.get_interp(model)
        p2 = P2.get_interp(model)

        assert (p2*code).is_equiv(dode)

        yield p2

        Add(P != p) # could also add all generators

find_autos = find_isomorphisms

def find_isomorphisms_css(code, dode=None):

    # find all automorphism permutations

    if dode is None:
        dode = code

    if code.n != dode.n or code.k != dode.k:
        return

    code = code.to_css()
    dode = dode.to_css()

    n = code.n
    mx = code.mx
    mz = code.mz

    if code.mx != dode.mx or code.mz != dode.mz:
        return

    solver = Solver()
    Add = solver.add

    # permutation matrix
    P = UMatrix.unknown(n, n)

    #for i in range(n):
    #    Add(Sum([If(P[i,j].get(),1,0) for j in range(n)])==1)

    for i in range(n):
      for j in range(n):
        rhs = reduce(And, [P[i,k]==0 for k in range(n) if k!=j])
        Add( Or(P[i,j]==0, rhs) )
        rhs = reduce(And, [P[k,j]==0 for k in range(n) if k!=i])
        Add( Or(P[i,j]==0, rhs) )

    for i in range(n):
        Add( reduce(Or, [P[i,j]!=0 for j in range(n)]) )

    #I = Matrix.identity(n, n)
    #Add(P * P.t == I)

    #H = code.H
    #H1 = H * P2

    Rx = UMatrix.unknown(mx, mx)
    Rz = UMatrix.unknown(mz, mz)

    Hx = Matrix(code.Hx)
    Hz = Matrix(code.Hz)
    Jx = Matrix(dode.Hx)
    Jz = Matrix(dode.Hz)

    Add( Rx*Hx == Jx*P )
    Add( Rz*Hz == Jz*P )

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        p = P.get_interp(model)

        #assert (p*code).is_equiv(dode) # ... fix..

        #print(p)
        yield p

        Add(P != p) # could also add all generated perms


def find_autos_lc(code):

    # find all local clifford automorphism permutations

    n = code.n
    nn = 2*n
    space = code.space
    F = space.F

    solver = Solver()
    Add = solver.add

    U = UMatrix.unknown(nn, nn)

    for i in range(nn):
        Add(Sum([If(U[i,j].get(),1,0) for j in range(nn)])<=2)

    I = Matrix.identity(n, n)

    Add(U.t*F*U == F) # quadratic constraint

    H = code.H
    m = code.m
    H1 = H * U.t

    # U preserves the stabilizer group
    R = UMatrix.unknown(m, m)
    Add( R*H == H1 )

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        u = U.get_interp(model)

        assert (u*code).is_equiv(code)

        yield u

        Add(U != u) # could also add all generators




def main():
    test()

    if argv.code == (4,2,2):
        code = QCode.fromstr("XXXX ZZZZ")
    elif argv.code == (5,1,3):
        code = construct.get_513()
    #elif argv.code == (4,1,2):
    #    code = QCode.fromstr("XYZI IXYZ ZIXY")
    elif argv.code == (4,1,2):
        code = QCode.fromstr("XXZZ ZZXX XZXZ")
    elif argv.code == (6,2,2):
        code = QCode.fromstr("XXXIXI ZZIZIZ IYZXII IIYYYY")
    elif argv.code == (6,3,2):
        code = QCode.fromstr("XXXXXX ZZZZII IIIIZZ")
    elif argv.code == (7,1,3) and argv.css:
        code = construct.get_713()
    elif argv.code == (7,1,3):
        code = QCode.fromstr("""
        XIXIIII IZIIZII XXIXXII ZIZZIZI IIIYZXX IIIIIZZ
        """)
        #print(code.get_params())
        #return
    elif argv.code == (10,2,3):
        code = construct.get_10_2_3()
    elif argv.code == (10,1,4):
        code = QCode.fromstr("""
        XZ.Z.X.ZZ.
        .Y.ZZY..ZZ
        ..YZZY.X..
        .ZZY.YZ..Z
        .Z..XYZY..
        .ZZ.ZZXXZZ
        ..ZZZZZ.XZ
        .ZZZZ..ZZX
        ZZZZZZ....
        """)
    elif argv.code == (13,1,5):
        n = 13
        check = "ZXIIIIIIXZIII"
        checks = [''.join(check[(i+j)%13] for i in range(n)) for j in range(n-1)]
        code = QCode.fromstr(' '.join(checks))
        print(code.get_params())
    elif argv.code == "YY":
        code = QCode.fromstr("YY", None, "Y. ZZ")
    elif argv.code == "YYY":
        code = QCode.fromstr("YYY")
    elif argv.code:
        code = QCode.fromstr(argv.code)
    else:
        return

    print(code.longstr())

    #for N in [1, 2, 3, 4, 5, 6]:
    for N in range(1, argv.get("N", 4)+1):
        count = 0
        gen = []
        arg = [code]*N
        print("N =", N)
        for M,L in find_transversal(*arg, constant=argv.constant):
            print(L)
            gen.append(L)
            count += 1
        print("gen:", len(gen))
        G = mulclose(gen)
        print("|G| =", len(G))
        print()

        make_gap("transversal_%d.gap"%(N,), gen)


def make_gap(name, gen):
    print("make_gap", name)
    f = open(name, "w")
    names = []
    for i,M in enumerate(gen):
        name = "M%d"%i
        print("%s := %s;"%(name,M.gap()), file=f)
        names.append(name)
    print("G := Group([%s]);"%(','.join(names)), file=f)
    print("Order(G);", file=f)
    f.close()


def main_autos_lc():
    if argv.code == (5,1,3):
        #code = construct.get_513()
        code = QCode.fromstr(
        "XZZXI IXZZX XIXZZ ZXIXZ", Ls="XXXXX ZZZZZ")
    elif argv.code == (13,1,5):
        n = 13
        check = "XXZZ.Z...Z.ZZ"
        checks = [''.join(check[(i+j)%n] for i in range(n)) for j in range(n-1)]
        code = QCode.fromstr(' '.join(checks))
    else:
        return

    print(code)
    print(code.longstr())
    assert code.is_gf4_linear()

#    N, perms = code.get_autos()
#
#    for perm in perms:
#        P = code.space.get_perm(perm)
#        dode = P*code
#        assert code.is_equiv(dode)
#        #print(dode.get_logical(code))
#        #print()

    space = code.space
    H, S = space.H, space.S
    op = reduce(mul, [S(i) for i in range(code.n)])
    dode = op*code
    assert not dode.is_equiv(code)

    if code.n == 5:
        iso = code.get_isomorphism(dode)
        p = space.get_perm(iso)
        eode = p*dode
        assert eode.is_equiv(code)
        print(eode.get_logical(code))

    #return

    print("find_autos_lc:")

    gen = []
    for u in find_autos_lc(code):
        assert (u*code).is_equiv(code)
        gen.append(u)
        print('.', end='', flush=True)
        G = mulclose(gen, verbose=True) #, maxsize=10000)
        #if len(G) == 660:
        #    break
        #for g in G:
        #    assert (g*code).is_equiv(code)
        #print("yes")
    print()

    print(len(G))



def find_clifford_stab():
    #code = construct.get_513()
    n, k = 4, 2
    space = SymplecticSpace(n)
    I = space.get_identity()
    code = QCode.from_encoder(I, k=k)

    n = code.n
    m = code.m
    H = code.H
    Ht = H.t

    F = code.space.F

    gen = []

    solver = Solver()
    Add = solver.add

    U = UMatrix.unknown(2*n, 2*n)
    Add(U.t*F*U == F) # quadratic constraint
    #Add(U*Ht == Ht)

    V = UMatrix.unknown(m, m)
    Vi = UMatrix.unknown(m, m)
    I = Matrix.identity(m, m)
    Add(V*Vi==I)

    Add(U*Ht*V == Ht)
    
    while 1:
    #for _ in range(10):
        result = solver.check()
        if result != z3.sat:
            print("result:", result)
            break
    
        model = solver.model()
        u = U.get_interp(model)
        assert u.t*F*u == F
        gen.append(u)
        #print(u)
        #print()

        v = V.get_interp(model)
        #vi = Vi.get_interp(model)
        assert u*Ht*v == Ht

        Add(U != u)
        
        dode = u*code
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        #print(L)
        G = mulclose(gen, verbose=True)
        G = list(G)
        #shuffle(G)
        #for g in G[:1000]:
        #    Add(U!=g) # woah
        #if len(G) == 4608:
        #    break
        #if len(G) == 4128768:
        #    break
        #if len(G) == 8847360:
        #    break
        del G

    #make_gap("clifford_stab_42", gen)



def test_513():
    code = construct.get_513()

    #print(code.longstr())
    p = [1,2,3,4,0]
    lc_ops = set()
    for g in find_local_cliffords(code, code):
        if g.is_identity():
            continue
        dode = code.apply(g)
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        lc_ops.add(L)
        break

    space = code.space
    for p in allperms(list(range(5))):
        if p==(0,1,2,3,4):
            continue
        P = space.get_perm(p)
        dode = code.apply_perm(p)
        for g in find_local_cliffords(code, dode):
            break
        else:
            assert 0
        #print(g)
        eode = dode.apply(g)
        assert eode.is_equiv(code)
        L = eode.get_logical(code)
        if L in lc_ops:
            continue
        gp = g*P
        dode = code.apply(gp)
        assert dode.is_equiv(code)
        lc_ops.add(L)
        break
        
    G = mulclose(lc_ops)
    assert len(G)==6
    assert len(lc_ops) == 2

    N = 4
    gen = []
    I = SymplecticSpace(1).get_identity()
    for g in lc_ops:
        op = reduce(Matrix.direct_sum, [g]+[I]*(N-1))
        gen.append(op)

    count = 0
    arg = [code]*N
    src = reduce(add, arg)
    print(src)
    print("N =", N)
    for M,L in find_transversal(*arg, constant=argv.constant):
        gen.append(L)
        tgt = src.apply(M)
        assert tgt.is_equiv(src)
        count += 1
    print()
    print("gen:", len(gen))
    for g in gen:
        print(g, g.shape)
    G = mulclose(gen, verbose=True)
    print("|G| =", len(G))
    print()

    f = open("generate_513_%d.gap"%N, "w")
    names = []
    for i,M in enumerate(gen):
        name = "M%d"%i
        print("%s := %s;"%(name,M.gap()), file=f)
        names.append(name)
    print("G := Group([%s]);"%(','.join(names)), file=f)
    print("Order(G);", file=f)
    f.close()


def test_513_cover():
    code = construct.get_513()

    #print(code.longstr())
    p = [1,2,3,4,0]
    lc_ops = set()
    physical = set()
    for g in find_local_cliffords(code, code):
        if g.is_identity():
            continue
        dode = code.apply(g)
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        lc_ops.add(L)
        physical.add(g)
        break

    space = code.space
    for p in allperms(list(range(5))):
        if p==(0,1,2,3,4):
            continue
        P = space.get_perm(p)
        dode = code.apply_perm(p)
        for g in find_local_cliffords(code, dode):
            break
        else:
            assert 0
        #print(g)
        eode = dode.apply(g)
        assert eode.is_equiv(code)
        L = eode.get_logical(code)
        if L in lc_ops:
            continue
        gp = g*P
        dode = code.apply(gp)
        assert dode.is_equiv(code)
        physical.add(gp)
        lc_ops.add(L)
        break
        
    G = mulclose(lc_ops)
    assert len(G)==6
    assert len(lc_ops) == 2

    cover = Cover.frombase(code)
    total = cover.total

    gen = set()
    for g in physical:
        g = cover.lift(g)
        gen.add(total.get_logical(g*total))

    g = cover.get_ZX()
    gen.add(total.get_logical(g*total))
    g = cover.get_CZ()
    gen.add(total.get_logical(g*total))

    G = mulclose(gen)
    assert len(G) == 36


def test_833():
    # See:
    # https://arxiv.org/abs/quant-ph/9702029"
    code = construct.get_833()
    space = code.space
    I = space.get_identity()
    n = code.n
    N, perms = code.get_autos()
    dode = code.apply_perm(perms[0])
    assert dode.is_equiv(code)
    #print(dode.get_logical(code))

    Sn = Group.symmetric(n)
    G, gen = set(), []
    for g in Sn:
        perm = [g[i] for i in range(n)]
        P = space.get_perm(perm)
        #print(P, perm)
        dode = code.apply(P)
        if code.is_equiv(dode):
            h = I
        else:
            for h in find_local_cliffords(code, dode):
                break
            else:
                continue
        eode = dode.apply(h)
        assert eode.is_equiv(code)
        l = eode.get_logical(code)
        if l not in G:
            gen.append(l)
            G = mulclose(gen)
        #print(l, len(G), len(gen))
        #print()
    # takes about 30 minutes to get here
    assert len(G) == 168

        
def test_412_gottesman():
    # See:
    # https://arxiv.org/abs/quant-ph/9702029

    s = SymplecticSpace(4)
    CX = s.CX
    ops = [CX(int(i),int(j)) for (i,j) in "20 10 01 02 21 31 23 12".split()]
    g = reduce(mul, ops)
    print(g)

    for perm in allperms(range(4)):
        P = s.get_perm(perm)
        #print(P.t * g * P == g, perm)

    c = construct.get_412()
    code = c+c+c+c
    perm = [j*4 + i for i in range(4) for j in range(4)]
    code = code.apply_perm(perm)

    g4 = reduce(Matrix.direct_sum, [g]*4)
    dode = code.apply(g4)
    assert code.is_equiv(dode)
    assert code.get_logical(dode) == g # nice

    gen = [CX(i,j) for i in range(4) for j in range(4) if i!=j]
    h = mulclose_find(gen, g)
    assert len(h.name) == 6
    assert h==g

    name = ('CX(0,1)', 'CX(3,0)', 'CX(0,2)', 'CX(1,3)', 'CX(2,1)', 'CX(1,0)')

    dode = code
    for name in reversed(name):
        g1 = s.get_expr(name)
        g4 = reduce(Matrix.direct_sum, [g1]*4)
        dode = dode.apply(g4)
        assert dode.distance() >= 2
    assert dode.is_equiv(code)


def test_513_gottesman():
    # See Fig. 1.
    # https://arxiv.org/abs/quant-ph/9702029

    s = SymplecticSpace(3)
    S, H, CX, CZ = s.S, s.H, s.CX, s.CZ
    SHS = lambda i :S(i)*H(i)*S(i)
    HS = lambda i :H(i)*S(i)
    SH = lambda i :S(i)*H(i)

    # T_3 gate
    g = CX(2,0)*CX(2,1)*CX(1,2)*CX(0,2)*SHS(0)*SHS(1)*CZ(0,1)*CX(0,1)*CX(1,0)*CX(0,1)

    c = construct.get_513()
    code = c+c+c

    perm = [j*5 + i for i in range(5) for j in range(3)]
    #print(perm)
    code = code.apply_perm(perm)

    g5 = reduce(Matrix.direct_sum, [g]*5)
    dode = code.apply(g5)
    assert code.is_equiv(dode)
    assert code.get_logical(dode) == g # nice

    gen  = [S(0),S(1),S(2)]
    gen += [H(0),H(1),H(2)]
    #gen += [SH(0),SH(1),SH(2)]
    #gen += [HS(0),HS(1),HS(2)]
    #gen += [SHS(0),SHS(1),SHS(2)]
    gen += [CX(i,j) for i in range(3) for j in range(3) if i!=j]
    gen += [CZ(i,j) for i in range(3) for j in range(i+1,3)]
    h = mulclose_find(gen, g)
    print(h)
    print(h.name)

    name = ('CX(0,1)', 'H(0)', 'S(0)', 'CZ(0,2)', 'CX(1,0)', 'CX(1,2)', 'H(1)', 'CX(1,0)')
    dode = code
    for name in reversed(name):
        g1 = s.get_expr(name)
        g5 = reduce(Matrix.direct_sum, [g1]*5)
        dode = dode.apply(g5)
        assert dode.distance() == 3
    assert dode.is_equiv(code)




def get_412_transversal():

    N = argv.get("N", 4)

    # N*[[4,1,2]] cliffords
    code = construct.get_412()
    arg = tuple([code]*N)
    logical = []
    #physical = []
    for M,L in find_transversal(*arg, constant=argv.constant):
        print(L)
        logical.append(L)
        #physical.append(M)
    print("logical:", len(logical))
    G = mulclose(logical)
    print("|G| =", len(G))

    # single [[4,1,2]] cliffords
    single = [] 
    n = code.n
    space = code.space
    src = code
    G = Group.symmetric(src.n)
    gen = set()
    for g in G:
        perm = [g[i] for i in range(n)]
        P = space.get_perm(perm)
        tgt = src.apply_perm(perm)
        for M in find_local_cliffords(src, tgt):
            code = tgt.apply(M)
            assert code.is_equiv(src)
            dode = src.apply(M*P)
            assert code.is_equiv(dode)
            L = code.get_logical(src)
            if L not in gen:
                #print(P)
                print(M, g)
                print(L)
                gen.add(L)
                #single.append(M*P)
                single.append(L)
    G = mulclose(gen)
    print(len(G))

    I = SymplecticSpace(1).get_identity()
    for M in single:
        for i in range(N):
            op = [I]*N
            op[i] = M
            op = reduce(Matrix.direct_sum, op)
            #physical.append(op)
            print(op)
            logical.append(op)

    if 0:
        G = mulclose(logical, verbose=True)
        print(len(G))
        return


    f = open("generate_%d.gap"%N, "w")
    names = []
    for i,M in enumerate(logical):
        name = "M%d"%i
        print("%s := %s;"%(name,M.gap()), file=f)
        names.append(name)
    print("G := Group([%s]);"%(','.join(names)), file=f)
    print("Order(G);", file=f)
    f.close()

    #G = mulclose(physical, verbose=True)
    #print("|G| =", len(G))



#def get_codes(n, k, d):
#    from bruhat.sp_pascal import i_grassmannian
#    perm = []
#    for i in range(n):
#        perm.append(i)
#        perm.append(2*n - i - 1)
#    found = []
#    for _,H in i_grassmannian(n, n-k):
#        H = H[:, perm]
#        H = Matrix(H)
#        code = QCode(H, check=False)
#        if code.get_distance() < d:
#            #print("x", end='', flush=True)
#            continue
#        found.append(code)
#    return found


def all_codes():
    from bruhat.sp_pascal import i_grassmannian

    #n, k, d = 4, 1, 2
    #n, k, d = 5, 1, 3
    n, k, d = argv.get("code", (4,1,2))
    constant = argv.get("constant", True)

    space = SymplecticSpace(n)
    gen = []
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(2*n - i - 1)
        gen.append(space.get_S(i))
        gen.append(space.get_H(i))
    found = []

    if 0:
        _Cliff = mulclose(gen)
        print("|_Cliff| =", len(_Cliff))
        code = construct.get_513()
        for g in _Cliff:
            dode = code.apply(g)
            found.append(dode)
        print(len(found))
        hom = equ.quotient_rep(found, QCode.is_equiv)
        found = list(set(hom.values()))
        print(len(found))
        return

    F = space.F
    count = 0
    #found = []
    for _,H in i_grassmannian(n, n-k):
        H = H[:, perm] # reshuffle to qumba symplectic
        H = Matrix(H)
        count += 1
        code = QCode(H, check=False)
        if code.get_distance() < d:
            #print("x", end='', flush=True)
            continue

        items = list(find_transversal(code, constant=constant, verbose=False))
        gen = [item[1] for item in items]
        G = mulclose(gen)
        if len(G) > 3:
            print()
            print(code.H)
            print("|G| =", len(G))

        elif len(G) == 3:
            print("[3]", end='', flush=True)
            found.append(code)

        elif len(G) == 2:
            print("[2]", end='', flush=True)
            found.append(code)

        #else:
        #    print(".", end='', flush=True)

        #if len(found) > 6:
        #    break

    print()
    print("count", count)
    print("found", len(found))
    #print(len([code for code in found if not code.is_css()]))

    hom = equ.quotient_rep(found, autos.is_iso)
    found = list(set(hom.values()))
    print("hom:", len(found))

    return found


def test_local_clifford():
    print("test_local_clifford")
    if argv.code == (4,2,2):
        code = QCode.fromstr("XXXX ZZZZ")
    elif argv.code == (5,1,3):
        code = construct.get_513()
    elif argv.code == (4,1,2):
        code = QCode.fromstr("XYZI IXYZ ZIXY")
    elif argv.code == (6,2,2):
        code = QCode.fromstr("XXXIXI ZZIZIZ IYZXII IIYYYY")
    elif argv.code == (8,2,3):
        code = QCode.fromstr("""
            X...YZZZ
            .X.ZYX.X
            .ZX.YYZX
            .ZZYZXZZ
            ..Z...YX
            ZZZZZZZZ
        """)
    elif argv.code == (8,3,2):
        code = QCode.fromstr("""
            XXXXIIII
            ZIZIZIZI
            IIYYYYII
            IZIZIZIZ
            IIIIXXXX
        """)

    print(code.get_params())

    n = code.n
    src = code
    G = Group.symmetric(src.n)
    count = 0
    gen = set()
    for g in G:
        #if g.is_identity():
        #    continue
        perm = [g[i] for i in range(n)]
        tgt = src.apply_perm(perm)
        for M in find_local_cliffords(src, tgt):
            code = tgt.apply(M)
            assert code.is_equiv(src)
            L = code.get_logical(src)
            if L not in gen:
                print(M, g)
                print(L)
                gen.add(L)
            count += 1
    print(count)
    G = mulclose(gen)
    print("|G| =", len(G))


def find_clifford(code, pairs, constant=False, verbose=True):
    solver = Solver()
    Add = solver.add

    m = 2
    Fm = SymplecticSpace(m).F
    n = code.n // m
    space = code.space

    U0 = UMatrix.unknown(2*m, 2*m)
    Add(U0.t*Fm*U0 == Fm) # quadratic constraint
    U = reduce(UMatrix.direct_sum, [U0]*n)
    perm = reduce(add, pairs)
    P = space.get_perm(perm)
    U = P.t*U*P

    HU = code.H * U.t
    LU = code.L * U.t
    F = code.space.F
    R = HU * F * code.L.t
    Add(R==0) # linear constraint
    R = HU * F * code.H.t
    Add(R==0) # linear constraint

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        dode = code.apply(M)
        assert dode.is_equiv(code)
        yield M

        Add(U != M)



def main_unwrap():

    for idx in range(31):
        code = construct.get_513(idx)
        n = code.n
    
        dode = unwrap(code)
        pairs = [(i, i+n) for i in range(n)]
        perm = [i+n for i in range(n)] + list(range(n))
    
        eode = dode.apply_perm(perm)
        eode = eode.apply_H()
        assert eode.is_equiv(dode)
    
        eode = dode
        for (i,j) in pairs:
            eode = eode.apply_CZ(i, j)
        assert eode.is_equiv(dode)
    
        count = 0
        gen = []
        for M in find_cliffords(dode, pairs):
            count += 1
            #print(M)
            eode = dode.apply(M)
            assert eode.is_equiv(dode)
            L = eode.get_logical(dode)
            #print(L)
            gen.append(L)
        print(count, end=" ")
        G = mulclose(gen)
        assert count == len(G)

        if not argv.autos:
            print()
            continue

        A = autos.get_autos(dode)
        #print(len(A), end=" ")

        for perm in A:
            #P = dode.space.get_perm(perm)
            eode = dode.apply_perm(perm)
            assert eode.is_equiv(dode)
            L = eode.get_logical(dode)
            #print(L)
            gen.append(L)
        #    print(perm)
        #continue
        G = mulclose(gen)
        print(len(G))


def test_412():
    space = SymplecticSpace(1)
    I = space.get_identity()
    H = space.get_H()
    S = space.get_S()
    names = {
        I : "I",
        S : "S",
        H : "H",
        S*H : r"(S\cdot H)",
        H*S : r"(H\cdot S)",
        H*S*H : r"(H\cdot S\cdot H)",
    }

    code = QCode.fromstr("XYZI IXYZ ZIXY")
    space = code.space
    n = code.n
    G = Group.symmetric(n)
    perms = [tuple(g[i] for i in range(n)) for g in G]
    perms.sort()
    for perm in perms:
        print(tuple(i+1 for i in perm), "&", end=" ")
        dode = code.apply_perm(perm)
        if dode.is_equiv(code):
            eode = dode
            U = space.get_identity()
        else:
            U = None
            found = list(find_local_cliffords(code, dode))
            assert len(found) == 1
            U = found[0]
            eode = dode.apply(U)
        assert eode.is_equiv(code)
        name = [names[U[i:i+2, i:i+2]] for i in range(0,space.nn,2)]
        name = r"".join(name)
        print(name, "&", end=" ")
        L = eode.get_logical(code)
        print(names[L], end=" ")
        print(r"\\")

    E = code.get_encoder()
    print(E)
    space = code.space
    print(space.get_name(E))


def find_perm_gate(code, perm):
    print("find_perm_gate", perm)
    n = code.n
    from qumba.clifford import Clifford, mulclose_names
    dode = code.apply_perm(perm)
    #print(dode.longstr())

    eode = dode
    space = code.space
    G = space.local_clifford_group()
    assert len(G) == 6**n
    found = None
    for g in G:
        eode = dode.apply(g)
        if eode.is_equiv(code):
            assert found is None
            found = g
    #else:
    #    assert 0
    lc_name = found.name
    del g
    #print("local clifford:", lc_name)

    circuit = lc_name + space.get_P(*perm).name
    eode = code.apply(code.space.get_expr(circuit))
    assert eode.is_equiv(code)

    H = code.H
    n = code.n
    c = Clifford(n)
    P = code.get_projector()
    assert P*P == P
    
    p = c.get_P(*perm)
    gate = c.get_expr(lc_name, rev=True)
    gate = p*gate
    if gate*P == P*gate:
        print("found gate")
        return circuit

    assert gate == c.get_expr(circuit, rev=True)
    
    names = c.pauli_group(False)
    found = None
    for g,name in names.items():
        #print(g, name)
        op = gate*g
        if(op*P == P*op):
            found = name
            break
    assert found
    print("pauli correction", found)
        
    circuit += found
    op = c.get_expr(circuit, rev=True)
    assert op*P == P*op
    return circuit


def test_412_clifford():
    from qumba.clifford import Clifford, green, red, half
    from huygens.zx import Circuit, Canvas
    code = QCode.fromstr("XYZI IXYZ ZIXY")
#    E = code.get_encoder()
#    D = code.get_decoder()
#    cvs = code.space.render(E)
#    cvs.writePDFfile("encoder.pdf")
#    assert E*D == code.space.get_identity()
#    assert D*E == code.space.get_identity()
#
#    E = code.space.translate_clifford(E)
#    D = code.space.translate_clifford(D)
#    DE = D*E
#    ED = E*D
#    c = Clifford(code.n)
##    for g in c.pauli_group(1):
##        if g*D == E.d:
##            break
##    else:
##        assert 0
##    print(g.name)
#
#    D = E.d
#    assert E*E.d == c.get_identity()
#
#    P = code.get_projector()
#    lhs = P
#    print(lhs)
#    print(lhs.shape, lhs.M.rank())
#
#    #rr = red(1,0)*red(0,1)
#    ##rr = green(1,0)*green(0,1)
#    #rhs = rr@rr@rr@Clifford(1).get_identity()
#    #print(rhs)
#    #print(rhs.shape, rhs.rank())
#    #print(P)
#    return

    clifford = Clifford(code.n)
    I = Clifford(1).get_identity()
    E = code.get_clifford_encoder()
    P = code.get_projector()
    D = E.d
    lhs = D * P * E
    rhs = [red(1,0)*red(0,1) for i in range(code.n)]
    rhs[-1] = I
    rhs = reduce(matmul, rhs)
    assert rhs == (2**code.m)*lhs

    # disc, prep
    prep = reduce(matmul, [red(1,0)]*code.m + [I]*code.k)
    disc = reduce(matmul, [red(0,1)]*code.m + [I]*code.k)

    S4 = Group.symmetric(code.n)
    perms = [[g[i] for i in range(code.n)] for g in S4]
    perms.sort()

    found = []
    for perm in perms:
    #for g in S4:
    #    perm = [g[i] for i in range(code.n)]
    
        print()
        circuit = find_perm_gate(code, perm)
        g = Clifford(code.n).get_expr(circuit, rev=True)
        assert g*P == P*g

        g = disc * D * g * E * prep
        g = (half**code.m)*g
        #print(g, g.rank())
        assert g.rank() == 2

        c = Clifford(1)
        gen = [c.get_identity(), c.X(), c.Z(), c.Y(), c.H(0), c.S(0), c.wI()**2]
        g = mulclose_find(gen, g)
        assert g is not None
        print("logical:", g.name)
        
        c = Circuit(code.n)
        cvs = c.render_expr(circuit, width=6, height=4)
        name = "circuit_%s.pdf"%(''.join(str(i) for i in perm))
        print(name)
        cvs.writePDFfile(name)
        found.append((cvs, perm))
        #if len(found)>1:
        #    break

    print("found:", len(found))
    cvs = Canvas()
    y = 0.
    for c, perm in found:
        cvs.insert(0, y, c)
        x = c.get_bound_box().width
        cvs.text(x, y, "%s"%(perm))
        y -= 1.5*c.get_bound_box().height
    cvs.writePDFfile("circuit.pdf")


def test_822_state_prep():
    # FAIL FAIL FAIL
    row_weight = argv.get("row_weight", 3)
    diagonal = argv.get("diagonal", False)

    code = QCode.fromstr("""
    XX...XX.
    .XX...XX
    ..XXX..X
    .ZZ.ZZ..
    ..ZZ.ZZ.
    Z..Z..ZZ
    """, Ls="""
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """)

    print(code)

    n = code.n
    nn = 2*n
    space = code.space
    F = space.F

    solver = Solver()
    Add = solver.add
    U = UMatrix.unknown(nn, nn)
    if diagonal:
        for i in range(nn):
            U[i,i] = Const(1)

    Add(U.t*F*U == F) # U symplectic

    #for perm in perms:
    #    P = space.get_perm(perm)
    #    U1 = P.t*U*P
    #    Add(U==U1)

    U0 = U[:, :2]

    R = code.H * F * U0
    Add(R==0) # linear constraint

    if row_weight is not None:
        for i in range(nn):
            Add(Sum([If(U[i,j].get(),1,0) for j in range(nn)])<=row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        print(M)
        return

        Add(U != M)



def test_hexagons():
    "how do dehn twists act on hexagons?"
    from qumba.solve import (
        parse, rank, array2, zeros2, 
        shortstr, linear_independent, intersect,
        dot2,
    )
    lookup = {}
    rows = cols = 6
    n = rows*cols
    for i in range(rows):
      for j in range(cols):
        lookup[i,j] = len(lookup)
    n = len(lookup)
    for i in range(rows):
      for j in range(cols):
        for di in (-rows, 0, rows):
          for dj in (-cols, 0, cols):
            lookup[i+di,j+dj] = lookup[i,j]

    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    #print(perm)
    P = zeros2((n, n))
    for (i,j) in enumerate(perm):
        P[i,j] = 1

    # black squares
    print("H =")
    H = []
    for i in range(rows):
      for j in range(cols):
        if (i+j)%2:
            continue
        op = [0]*n
        op[lookup[i,j]] = 1
        op[lookup[i+1,j]] = 1
        op[lookup[i,j+1]] = 1
        op[lookup[i+1,j+1]] = 1
        H.append(op)
    H = array2(H)
    print(shortstr(H), H.shape, rank(H))
    H = linear_independent(H)
    HP = dot2(H,P)
    print("H^HP =", rank(intersect(H, HP)))
    print()

    # right hexagons
    K = []
    for i in range(rows):
      for j in range(cols):
        if (i+j)%2:
            continue
        op = [0]*n
        op[lookup[i,j]] = 1
        op[lookup[i+1,j]] = 1
        op[lookup[i,j+1]] = 1
        op[lookup[i+1,j+2]] = 1
        op[lookup[i+2,j+1]] = 1
        op[lookup[i+2,j+2]] = 1
        K.append(op)
    K = array2(K)
    print("K =")
    print(shortstr(K), K.shape, rank(K))
    K = linear_independent(K)
    HK = intersect(H, K)
    print("H^K =", rank(HK))
    print("H^KP =", rank(intersect(H,dot2(K,P))))
    print("HP^KP =", rank(intersect(HP,dot2(K,P))))

    print()

    m = len(H)
    J = []
    for i in range(m):
      for j in range(m):
        op = (H[i] + H[j])%2
        if op.sum() == 6:
            J.append(op)
    J = array2(J)
    J = linear_independent(J)
    print("J =")
    print(shortstr(J), J.shape)
    JK = intersect(J, K)
    print("J^K =", rank(JK))
    print()


def search_gate(code, dode, *perms, row_weight=None, diagonal=False):
    solver = Solver()
    Add = solver.add

    n = code.n
    nn = 2*n
    space = code.space
    F = space.F

    U = UMatrix.unknown(nn, nn)
    if diagonal:
        for i in range(nn):
            U[i,i] = Const(1)

    Add(U.t*F*U == F) # U symplectic

    for perm in perms:
        P = space.get_perm(perm)
        U1 = P.t*U*P
        Add(U==U1)

    HU = code.H * U.t
    LU = code.L * U.t
    R = HU * F * dode.L.t
    Add(R==0) # linear constraint
    R = HU * F * dode.H.t
    Add(R==0) # linear constraint

    if row_weight is not None:
        for i in range(nn):
            Add(Sum([If(U[i,j].get(),1,0) for j in range(nn)])<=row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("result:", result)
            break
    
        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        eode = code.apply(M)
        assert dode.is_equiv(eode)
        yield M

        Add(U != M)


def test_dehn():
    lookup = {}
    rows, cols = 4, 4
    for i in range(rows):
      for j in range(cols):
        lookup[i,j] = len(lookup)
    n = len(lookup)
    for i in range(rows):
      for j in range(cols):
        for di in (-rows, 0, rows):
          for dj in (-cols, 0, cols):
            lookup[i+di,j+dj] = lookup[i,j]

    facess = [[], []]
    parity = 0
    for i in range(rows):
      for j in range(cols):
        face = [lookup[i,j] 
            for (i,j) in [(i,j), (i+1,j), (i+1,j+1), (i,j+1)]]
        facess[parity].append(face)
        parity = (parity+1)%2
      parity = (parity+1)%2
    #print(facess, n)
    stabs = []
    for (faces, op) in zip(facess, list('XZ')):
      for face in faces:
        stab = ['I']*n
        for idx in face:
            stab[idx] = op
        stab = fromstr(''.join(stab))
        #print(stab)
        stabs.append(stab)
    #stabs = numpy.array(stabs)
    stabs = numpy.concatenate(stabs)
    code = QCode(A=stabs)
    print(code)
    print(strop(code.H))

    return

    #for zx in code.find_zx_dualities():
    #    print(zx)
    #return

    perm = []
    for i in range(rows):
        p = list(range(i*cols, (i+1)*cols))
        p = [p[(idx+i)%cols] for idx in range(cols)]
        perm += p
    dode = code.apply_perm(perm)
    #print(dode.is_equiv(code))
    #for M in find_local_cliffords(dode, code):
    #    print(M)
    print(dode)
    print(strop(dode.H))
    print(code.is_equiv(dode))


    hperm = {}
    vperm = {}
    for i in range(rows):
      for j in range(cols):
        hperm[lookup[i,j]] = lookup[i+2,j]
        vperm[lookup[i,j]] = lookup[i,j+2]
    hperm = [hperm[i] for i in range(n)]
    vperm = [vperm[i] for i in range(n)]
    print(hperm)
    print(vperm)
    
    assert code.apply_perm(vperm).is_equiv(code)
    assert code.apply_perm(hperm).is_equiv(code)
    assert dode.apply_perm(vperm).is_equiv(dode)
    assert dode.apply_perm(hperm).is_equiv(dode)

    if 0:
        #return
    
        H = dode.H.intersect(code.H)
        print(code.H.shape, "-->", H.shape)

        eode = QCode(H)
        #from distance import distance_z3
        #distance_z3(eode)
        #print(eode)
        #print(eode.longstr())
        #return
    
        M = dode.get_encoder() * code.get_decoder()
        #print(shortstr(M.A))
    
        op = dode.get_logical(code)
        s = SymplecticSpace(2)
        gen = [s.get_CNOT(1,0), s.get_CZ(), 
            s.get_S(0), s.get_H(0), s.get_S(1), s.get_H(1)]
        G = mulclose(gen)
        G = list(G)
        print(G[0].name)
        #print(op in mulclose(gen))
        idx = G.index(op)
        print(G[idx].name)
        #print(SymplecticSpace(2).get_name(op))
    
        return M

    row_weight = argv.row_weight
    diagonal = argv.diagonal
    for M in search_gate(code, dode, hperm, vperm, row_weight=row_weight, diagonal=diagonal):
        print(M)
        print(code.space.get_name(M))
        #break
    

def test_all_412():
    "generate all 412 codes & look for local clifford & perm gates"
    n, k, d = 4, 1, 2
    G = Group.symmetric(n)
    perms = [[g[i] for i in range(n)] for g in G]
    count = 0
    for code in construct.all_codes(n, k, d):
        #print(code)
        found = 0
        for perm in perms:
            dode = code.apply_perm(perm)
            if dode.is_equiv(code):
                found += 1
            elif is_local_clifford_equiv(code, dode):
                found += 1
        print("%2d"%found, end=' ', flush=True)
        if count%32==0:
            print()
        count += 1
    print(count)

    
def find_all_perm_lc_412():
    n, k, d = 4, 1, 2
    G = Group.symmetric(n)
    perms = [[g[i] for i in range(n)] for g in G]
    codes = []
    for code in construct.all_codes(n, k, d):
        codes.append(code)
        print('.',end='',flush=True)
        #if len(codes)>30:
        #    break
    print()
    print(len(codes))
    def lc_perm(code, dode):
        for perm in perms:
            eode = dode.apply_perm(perm)
            if eode.is_equiv(code):
                return True
            elif is_local_clifford_equiv(code, eode):
                return True
    hom = equ.quotient_rep(codes, lc_perm)
    found = list(set(hom.values()))
    print("equs:", len(found))


def sp_find(items):
    n = 2
    space = SymplecticSpace(n)

    nn = 2*n
    F = space.F

    solver = Solver()
    Add = solver.add
    U = UMatrix.unknown(nn, nn)
    Add(U.t*F*U == F) # U symplectic

    parse = space.parse
    pairs = []
    for item in items:
        src, tgt = item.split()
        src = parse(src).t
        tgt = parse(tgt).t
        #print(src, tgt)
        pairs.append((src, tgt))

        Add(U * src == tgt)
        
    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("unsat")
            return

        model = solver.model()
        M = U.get_interp(model)
        assert M.t*F*M == F
    
        yield M
        #print(M)
        #print()
        for src, tgt in pairs:
            assert (M*src == tgt)
        Add(U != M)


def test_braid():
    n = 2
    space = SymplecticSpace(n)

    CX, S, H = space.CX, space.S, space.H
    gen = [CX(0,1), CX(1,0), S(0), S(1), H(0), H(1)]
    G = mulclose(gen)
    G = list(G)
    print(len(G))

    found = set()
    #rhs = ["YI YZ", "IY ZY", "ZX IX", "XZ XI"]
    rhs = ["YI YZ", "IY ZY", "XZ IX", "ZX XI"]
    for R in sp_find(rhs):
        lhs = "XI XZ,IX ZX,ZY YI,YZ IY".split(',')
        for L in sp_find(lhs):
            U = L*R
            found.add(U)

    for g in found:
        print(g)
        g = G[G.index(g)]
        print(g.name)


def find_equivariant(X):
    from qumba.csscode import CSSCode, distance_z3

    G = X.G
    n = len(X)
    print("find_equivariant", n)

    perms = []
    for g in G:
        send = X(g)
        items = send.items # Coset's
        lookup = dict((v,k) for (k,v) in enumerate(items))
        send = {i:lookup[send[items[i]]] for i in range(len(items))}
        assert len(send) == n
        perms.append(send)

    solver = Solver()
    Add = solver.add

    hx = UMatrix.unknown(n)
    hz = UMatrix.unknown(n)

    # we have a transitive G action, so we can fix the 0 bit:
    hx[0] = 1
    hz[0] = 1

    Hx = [[hx[g[i]] for i in range(n)] for g in perms]
    Hx = UMatrix(Hx)

    c = Hx * hz.t
    Add(c==0)

    weight = lambda h : Sum([If(h[i].get(),1,0) for i in range(1, n)])+1
    row_weight = 10
    Add(weight(hx) < row_weight)
    Add(weight(hz) < row_weight)

    while 1:
        result = solver.check()
        if result != z3.sat:
            #print("unsat")
            return
    
        model = solver.model()
        _hx = hx.get_interp(model)
        _hz = hz.get_interp(model)

        #Add(hx != _hx)
        #Add(hz != _hz)
    
        #hx = Hx[0]
        #print("hx:", _hx)
        #print("hz:", _hz)
    
        Hx = Matrix([[_hx[g[i]] for i in range(n)] for g in perms])
        for _hx in Hx:
            Add(hx != _hx)
        Hx = Hx.linear_independent()
    
        Hz = Matrix([[_hz[g[i]] for i in range(n)] for g in perms])
        for _hz in Hz:
            Add(hz != _hz)
        Hz = Hz.linear_independent()
    
        code = CSSCode(Hx=Hx.A, Hz=Hz.A)
        if code.k == 0:
            continue

        yield code


def prune(Xs):
    i = 0
    while i < len(Xs):
        j = i+1
        while j < len(Xs) and i < len(Xs):
            if Xs[i].isomorphic(Xs[j]):
                Xs.pop(j)
            else:
                j += 1
        i += 1


def test_equivariant():
    # find equivariant CSS codes

    from qumba.csscode import CSSCode, distance_z3

    Hs = None
    Xs = None
    n = None

    # gap> LoadPackage( "AtlasRep", false );
    # gap> G := AtlasGroup("L2(11)");
    # Group([ (2,10)(3,4)(5,9)(6,7), (1,2,11)(3,5,10)(6,8,9) ])

    if argv.cyclic:
        n = argv.get("n", 10)
        G = Group.cyclic(n)

    elif argv.dihedral:
        n = argv.get("n", 10)
        G = Group.dihedral(n)

    elif argv.symmetric:
        m = argv.get("m", 4)
        G = Group.symmetric(m)

    elif argv.coxeter_bc:
        m = argv.get("m", 3)
        G = Group.coxeter_bc(m)
        n = len(G)

    elif argv.GL32:
        n = 7
        items = list(range(1, n+1))
        a = Perm.fromcycles([(1,), (2,), (7,), (3,5), (4,6)], items)
        b = Perm.fromcycles([(1,6,3), (5,), (2,4,7)], items)
        G = Group.generate([a,b])
        assert len(G) == 168

    elif argv.L2_8:
        n = 9
        items = list(range(1, n+1))
        a = Perm.fromcycles([ (1,2),(3,4),(6,7),(8,9) ], items)
        b = Perm.fromcycles([ (1,3,2),(4,5,6),(7,8,9) ], items)
        G = Group.generate([a,b])
        assert len(G) == 504
        X = G.tautological_action() # No solution
        X = X*X
        Xs = X.get_components()

    elif argv.L2_11:
        n = 11
        items = list(range(1, n+1))
        a = Perm.fromcycles([ (2,10),(3,4),(5,9),(6,7), ], items)
        b = Perm.fromcycles([(1,2,11),(3,5,10),(6,8,9) ], items)
        G = Group.generate([a,b])
        assert len(G) == 660
        X = G.tautological_action() # No solution
        X = X*X
        Xs = X.get_components()

    elif argv.L2_13:
        n = 14
        items = list(range(1, n+1))
        a = Perm.fromcycles([(1,12),(2,6),(3,4),(7,11),(9,10),(13,14)], items)
        b = Perm.fromcycles([ (1,6,11),(2,4,5),(7,8,10),(12,14,13) ], items)
        G = Group.generate([a,b])
        assert len(G) == 1092, len(G)
        X = G.tautological_action() # No solution
        X = X*X
        Xs = X.get_components()

    elif argv.L2_16:
        n = 17
        items = list(range(1, n+1))
        a = Perm.fromcycles([ (1,2),(3,4),(5,6),(7,9),(10,11),(12,13),(14,16),(15,17), ], items)
        b = Perm.fromcycles([ (1,3,2),(4,5,7),(6,8,10),(11,12,14),(13,15,17) ], items)
        G = Group.generate([a,b])
        assert len(G) == 4080
        X = G.tautological_action() # No solution
        Xs = [X]

    elif argv.M11:
        n = 11
        items = list(range(1, n+1))
        a = Perm.fromcycles([(2,10),(4,11),(5,7),(8,9)], items)
        b = Perm.fromcycles([(1,4,3,8),(2,5,6,9)], items)
        G = Group.generate([a,b])
        assert len(G) == 7920
        X = G.tautological_action()
        Xs = [X]

    elif argv.M20:
        n = 20
        items = list(range(1, n+1))
        a = Perm.fromcycles(
            [(1,2,4,3),(5,11,7,12),(6,13),(8,14),(9,15,10,16),(17,19,20,18)], items)
        b = Perm.fromcycles(
            [(2,5,6),(3,7,8),(4,9,10),(11,17,12),(13,16,18),(14,15,19)], items)
        G = Group.generate([a,b], verbose=True)
        assert len(G) == 960
        X = G.tautological_action()
        Xs = [X]

    elif argv.M21:
        n = 21
        items = list(range(1, n+1))
        a = Perm.fromcycles([(1,2),(4,6),(5,7),(8,12),(9,14),(10,15),(11,17),(13,19)], items)
        b = Perm.fromcycles([(2,3,5,4),(6,8,13,9),(7,10,16,11),(12,18),(14,20,21,15),(17,19)], items)
        G = Group.generate([a,b], verbose=True)
        assert len(G) == 20160
        X = G.tautological_action()
        Xs = [X]

    else:
        m = argv.get("m", 4)
        G = Group.alternating(m)

    n = argv.get("n", n or len(G))
    print("|G| =", len(G))

    if Xs is None:
        #Hs = [H for H in G.conjugacy_subgroups() if len(H)<len(G)]
        Hs = [H for H in G.subgroups() if len(H)<len(G)]
        print("indexes:")
        print('\t', [len(G)//len(H) for H in Hs])
        Xs = [G.action_subgroup(H) for H in Hs if len(G)//len(H) == n]

    print("|Xs| =", len(Xs))

    if argv.prune and len(Xs)>1:
        print("prune...")
        prune(Xs)
        print("|Xs| =", len(Xs))

#    for H in Hs:
#        if len(G)//len(H) != n:
#            continue
#        print("|H| =", len(H))
#        X = G.action_subgroup(H)
    show = argv.show

    for X in Xs:
        for code in find_equivariant(X):
            d = distance_z3(code)
            if d < 3:
                continue
            print("[[%d, %d, %d]]"%(code.n, code.k, d))
            #print(code)
            #print("distance =", d)
            #print(code.longstr())
            if (code.n, code.k, d) == show:
                print("Hx =")
                print(code.Hx)
                print("Hz =")
                print(code.Hz)
                print()
    

def find_css():
    # see also previous version: csscode.find_z3
    params = argv.code
    if params is not None:
        n, k, d = params
        mx = (n-k)//2
        mz = n-k-mx
    else:
        mx = argv.get("mx", 3)
        mz = argv.get("mz", 3)
        n = argv.get("n", mx+mz+1)
        k = n-mx-mz
        assert k>0
    
        d = argv.get("d", 3) 

    print("code: [[%d, %d, %d]]"%(n, k, d))

    solver = Solver()
    Add = solver.add

    Hx = UMatrix.unknown(mx, n)
    Hz = UMatrix.unknown(mz, n)
    Tx = UMatrix.unknown(mx, n)
    Tz = UMatrix.unknown(mz, n)
    Lx = UMatrix.unknown(k, n)
    Lz = UMatrix.unknown(k, n)

    if argv.normal:
        print("normal")
        #print( type(Hx[:,mx]) )
        Add( Hx[:, :mx] == Matrix.identity(mx) )
        Add( Hz[:, :mz] == Matrix.identity(mz) )
        #Add( Tx[:, :mx] == Matrix.identity(mx) )
        #Add( Tx[:, mx:] == Matrix.zeros((mx, n-mx)) )
        #Add( Tz[:, :mz] == Matrix.identity(mz) )
        #Add( Tz[:, mz:] == Matrix.zeros((mz, n-mz)) )

    Add( Hx*Tz.t == Matrix.identity(mx) ) # Hx full rank
    Add( Hx*Hz.t == Matrix.zeros((mx, mz)) ) # Hx/Hz commuting
    Add( Hx*Lz.t == Matrix.zeros((mx, k)) )

    Add( Tx*Tz.t == Matrix.zeros((mx, mz)) )
    Add( Tx*Hz.t == Matrix.identity(mx) ) # Hz full rank
    Add( Tx*Lz.t == Matrix.zeros((mx, k)) )

    Add( Lx*Tz.t == Matrix.zeros((mx, mz)) )
    Add( Lx*Hz.t == Matrix.zeros((mx, mx)) )
    Add( Lx*Lz.t == Matrix.identity(k) )

    #Rx = UMatrix.unknown(n, mx)
    #Rz = UMatrix.unknown(n, mz)

    #Add( Hx*Rx == Matrix.identity(mx, mx) )
    #Add( Hz*Rz == Matrix.identity(mz, mz) )

    if argv.selfdual:
        print("selfdual")
        assert mx==mz
        Add(Hx==Hz)
        Add(Tx==Tz)
        Add(Lx==Lz)

    if k > 0:
        A = UMatrix.unknown(1, mx+k)
        u = A[:, :mx]
        v = A[:, mx:]
    
        op = u*Hx + v*Lx
        lhs = reduce(Or, [v[0,i].get() for i in range(k)])
        rhs = Sum([If(op[0,i].get(),1,0) for i in range(n)]) >= d
        Add(ForAll([A[0,i].v for i in range(mx+k)], If(lhs, rhs, True)))
    
        A = UMatrix.unknown(1, mz+k)
        u = A[:, :mz]
        v = A[:, mz:]
    
        op = u*Hz + v*Lz
        lhs = reduce(Or, [v[0,i].get() for i in range(k)])
        rhs = Sum([If(op[0,i].get(),1,0) for i in range(n)]) >= d
        Add(ForAll([A[0,i].v for i in range(mz+k)], If(lhs, rhs, True)))

    else:
        u = UMatrix.unknown(1, mx)
        for v in numpy.ndindex((2,)*k):
            v = numpy.array(v)
            v.shape = (1,k)
            v = Matrix(v)
            l = v*Lx


    result = solver.check()
    if str(result) != "sat":
        print(result)
        return

    model = solver.model()
    Hx = Hx.get_interp(model)
    Hz = Hz.get_interp(model)
    Tx = Tx.get_interp(model)
    Tz = Tz.get_interp(model)
    Lx = Lx.get_interp(model)
    Lz = Lz.get_interp(model)

    print("Hx:")
    print(Hx)

    print("Hz:")
    print(Hz)

    from qumba.csscode import CSSCode
    #code = CSSCode(Hx=Hx.A, Hz=Hz.A, Lx=Lx.A, Lz=Lz.A, Tx=Tx.A, Tz=Tz.A)
    code = CSSCode(Hx=Hx.A, Hz=Hz.A)

    d_x, d_z = code.distance()
    print(code, d_x, d_z)

    print(code.longstr())



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






