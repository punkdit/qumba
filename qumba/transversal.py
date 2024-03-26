#!/usr/bin/env python
"""
_looking for transversal logical clifford operations
"""

from functools import reduce
from operator import add, matmul

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

from qumba.qcode import QCode, SymplecticSpace, Matrix, fromstr, shortstr, strop
from qumba.action import mulclose, Group, mulclose_find
from qumba import equ
from qumba import construct 
from qumba import autos
from qumba.unwrap import unwrap
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
        self.A = numpy.array(A, dtype=object)
        assert shape is None or shape == A.shape
        self.shape = A.shape

    def __getitem__(self, idx):
        value = self.A[idx]
        if isinstance(value, numpy.ndarray):
            value = UMatrix(value)
        return value

    def __setitem__(self, idx, value):
        self.A[idx] = value

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


def find_local_clifford(tgt, src, constant=False, verbose=True):
    #print("find_local_clifford")
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

def is_local_clifford_equiv(tgt, src, constant=False, verbose=False):
    for M in find_local_clifford(tgt, src, constant, verbose):
        return True
    return False


def main():
    test()

    if argv.code == (4,2,2):
        code = QCode.fromstr("XXXX ZZZZ")
    elif argv.code == (5,1,3):
        code = construct.get_513()
    elif argv.code == (4,1,2):
        code = QCode.fromstr("XYZI IXYZ ZIXY")
    elif argv.code == (6,2,2):
        code = QCode.fromstr("XXXIXI ZZIZIZ IYZXII IIYYYY")
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
    else:
        return

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


def get_412_transversal_4():

    N = 4

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
        for M in find_local_clifford(src, tgt):
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


    f = open("generate.gap", "w")
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
        for M in find_local_clifford(src, tgt):
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
        for M in find_clifford(dode, pairs):
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
            found = list(find_local_clifford(code, dode))
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
    #for M in find_local_clifford(dode, code):
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






