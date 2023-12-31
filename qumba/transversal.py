#!/usr/bin/env python
"""
looking for transversal logical clifford operations
"""

from functools import reduce
from operator import add

import numpy

import z3
from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

from qumba.qcode import QCode, SymplecticSpace, Matrix
from qumba.action import mulclose, Group
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
        assert value in [0, 1]
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


def get_codes(n, k, d):
    from bruhat.sp_pascal import i_grassmannian
    perm = []
    for i in range(n):
        perm.append(i)
        perm.append(2*n - i - 1)
    found = []
    for _,H in i_grassmannian(n, n-k):
        H = H[:, perm]
        H = Matrix(H)
        code = QCode(H, check=False)
        if code.get_distance() < d:
            #print("x", end='', flush=True)
            continue
        found.append(code)
    return found


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
        Cliff = mulclose(gen)
        print("|Cliff| =", len(Cliff))
        code = construct.get_513()
        for g in Cliff:
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
        H = H[:, perm]
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






