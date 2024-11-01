#!/usr/bin/env python

"""
See:
    https://arxiv.org/abs/quant-ph/9703048
    Nonbinary quantum codes
    Eric M. Rains

"""

from functools import reduce
from operator import add, matmul, mul

import numpy

from qumba.qcode import QCode, SymplecticSpace, Matrix, fromstr, shortstr, strop
from qumba import construct
from qumba.matrix import scalar
from qumba.action import mulclose, Group, mulclose_find
from qumba.smap import SMap
from qumba.argv import argv
from qumba.umatrix import UMatrix, Solver, Or
from qumba.util import all_subsets



def span(G):
    G = list(G)
    N = len(G)
    algebra = set()
    for bits in numpy.ndindex((2,)*N):
        A = Matrix([[0,0],[0,0]])
        for i,bit in enumerate(bits):
            if bit:
                A = A + G[i]
        algebra.add(A)
    algebra = list(algebra)
    algebra.sort()
    return algebra


# broken...
#def generate0(gen, verbose=False, maxsize=None):
#    els = set(gen)
#    bdy = list(els)
#    changed = True
#    while bdy:
#        if verbose:
#            print(len(els), end=" ", flush=True)
#        _bdy = []
#        for A in gen:
#            for B in bdy:
#                for C in [A*B, A+B]:
#                  if C not in els:
#                    els.add(C)
#                    _bdy.append(C)
#                    if maxsize and len(els)>=maxsize:
#                        return els
#        bdy = _bdy
#    if verbose:
#        print()
#    return els


def generate(gen, verbose=False, maxsize=None):
    els = set(gen)
    changed = True
    while changed:
        changed = False
        if verbose:
            print(len(els), end=" ", flush=True)
        items = list(els)
        for A in items:
            for B in items:
                for C in [A*B, A+B]:
                  if C not in els:
                    els.add(C)
                    changed = True
                    if maxsize and len(els)>=maxsize:
                        return els
    if verbose:
        print()
    return els


class Algebra(object):
    def __init__(self, items):
        items = list(items)
        items.sort(key = lambda a:(str(a).count('1'), a))
        self.items = tuple(items)

    def __eq__(self, other):
        return self.items == other.items

    def __hash__(self):
        return hash(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    @classmethod
    def generate(cls, gen):
        A = generate(gen)
        return cls(A)

    def dump(algebra):
        smap = SMap()
        for i,a in enumerate(algebra):
            smap[0, 4*i] = str(a)
        print(smap)
        print()


def get_algebra():

    space = SymplecticSpace(1)
    I = space.I()
    S = space.S()
    H = space.H()

    G = mulclose([S,H])
    G = list(G)
    assert len(G) == 6

    for g in G:
        assert g*~g == I
    algebra = span(G)
    assert len(algebra) == 16

    return Algebra(algebra)


def main_rains():
    space = SymplecticSpace(1)
    I = space.I()
    S = space.S()
    H = space.H()

    G = mulclose([S,H])
    G = list(G)
    assert len(G) == 6

    algebra = get_algebra()

    unital = argv.get("unital", True)

    _count = 0
    found = set()
    for gen in all_subsets(algebra):
        _count += 1
        if algebra[0] not in gen:
            continue

        if unital and I not in gen:
            continue

        A = Algebra.generate(gen)
        found.add(A)

        #if len(found) > 2:
        #    break

    J = H
    conj = lambda a : J*a.t*J

    print("found:", len(found), "_count:", _count)
    found = list(found)
    found.sort(key = lambda A : (len(A), tuple(A)))
    for A in found:
        print("|A| = ", len(A))

        comm = True
        for a in A:
            assert conj(a) in A
            for b in A:
                assert a+b in A
                assert a*b in A
                assert conj(a*b) == conj(b)*conj(a)
                if a*b != b*a:
                    comm = False
        print("comm:", comm)
        A.dump()


    for A in found:
        sig = ['.']*len(found)
        for g in G:
            B = Algebra([g*a*~g for a in A])
            i = found.index(B)
            sig[i] = "*"
        print(''.join(sig))

    print()

    m, n = 2, 4
    space = SymplecticSpace(n)

    sigs = set()
    for C in space.grassmannian(m):
        C2 = C.reshape(m, n, 2)
        Ct = C.t
        sig = []
        for algebra in found:
            for a in algebra:
                D = C2*a
                D = D.reshape(m, 2*n)
                u = Ct.solve(D.t)
                if u is None:
                    sig.append(".")
                    break
            else:
                sig.append("*")
        sig = ''.join(sig)
        sigs.add(''.join(sig))

        if sig == "*......*...." and 0:
            code = QCode(C)
            print(code)
            assert code.is_gf4()
            #print(code.longstr())
            print(strop(code.H))
            print()

    sigs = list(sigs)
    sigs.sort()
    for sig in sigs:
        print(sig)
    

def main_2():
    n = 2
    nn = 2*n
    space = SymplecticSpace(n)
    I = space.I()
    S, H = space.S, space.H

    gen = [S(0),S(1),H(0),H(1),space.CX(0,1)]
    G = mulclose(gen)
    print(len(G))

    A = Algebra.generate(gen)
    assert len(A) == 2**16

    zero = Matrix.zeros((nn,nn))

#    found = set()
#    for a in A:
#      for b in A:
#        B = Algebra.generate([a,b])
#        found.add(B)
#    print(len(found))




def find_algebras(dim):

    mul = UMatrix.unknown(dim, dim*dim)
    unit = UMatrix.unknown(dim, 1)
    I = UMatrix.identity(dim)

    solver = Solver()
    Add = solver.add

    # unital
    Add(mul*(unit@I) == I)
    Add(mul*(I@unit) == I)

    # Assoc
    Add(mul*(I@mul) == mul*(mul@I))

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        a = mul.get_interp(model)
        i = unit.get_interp(model)

        yield (i, a)

        Add( Or( (mul!=a) , (unit!=i) ) )


def find_add_algebras(dim):

    mul = UMatrix.unknown(dim, dim+dim)
    unit = UMatrix.unknown(dim, 0)
    I = UMatrix.identity(dim)

    solver = Solver()
    Add = solver.add

    # unital
    Add(mul*(unit<<I) == I)
    Add(mul*(I<<unit) == I)

    # Assoc
    Add(mul*(I<<mul) == mul*(mul<<I))

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        a = mul.get_interp(model)
        i = unit.get_interp(model)

        yield (i, a)

        Add( Or( (mul!=a) )) #, (unit!=i) ) )


def main_add():
    dim = 3
    _count = 0
    for (unit, mul) in find_add_algebras(dim):
        _count += 1
    # cocartesian => unique algebras
    assert _count == 1
        



def find_modules(dim, unit, mul):

    n = len(unit)
    act = UMatrix.unknown(dim, n*dim)

    Id = UMatrix.identity(dim)
    In = UMatrix.identity(n)

    solver = Solver()
    Add = solver.add

    # unital
    Add(act*(unit@Id) == Id)

    # Assoc
    Add(act*(In@act) == act*(mul@Id))

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        a = act.get_interp(model)

        yield a

        Add( act != a )


def find_coalgebras(dim):

    comul = UMatrix.unknown(dim*dim, dim)
    counit = UMatrix.unknown(1, dim)
    I = UMatrix.identity(dim)

    solver = Solver()
    Add = solver.add

    Add((counit@I)*comul == I)
    Add((I@counit)*comul == I)
    Add((I@comul)*comul == (comul@I)*comul)

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        _counit = counit.get_interp(model)
        _comul = comul.get_interp(model)

        yield (_counit, _comul)

        Add( Or( (comul!=_comul) , (counit!=_counit) ) )


def find_comodules(dim, counit, comul):

    _, n = counit.shape
    coact = UMatrix.unknown(n*dim, dim)

    Id = UMatrix.identity(dim)
    In = UMatrix.identity(n)

    solver = Solver()
    Add = solver.add

    # counital
    Add((counit@Id)*coact == Id)

    # coassoc
    Add((In@coact)*coact == (comul@Id)*coact)

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        _coact = coact.get_interp(model)

        yield _coact

        Add( coact != _coact )




def main_algebras():

    dim = 2
    _count = 0
    for (unit, mul) in find_algebras(dim):
        print("unit =")
        print(unit)
        print("mul =")
        print(mul)
#        dount = 0
#        for act in find_modules(2, unit, mul):
#            #print("act =")
#            #print(act)
#            dount += 1
#        print("[%d]"%dount, end="", flush=True)
        _count += 1
    print()
    print("found:", _count)


def get_swap(d):
    swap = numpy.zeros((d*d, d*d), dtype=scalar)
    #for i in range(d*d):
    for i in range(d):
      for j in range(d):
        row = d*i + j
        col = d*j + i
        swap[row, col] = 1
    #return UMatrix(swap)
    return swap

#print(get_swap(2))
#print(get_swap(3))

#assert str(get_swap(2)) == str(UMatrix([
#    [1,0,0,0],
#    [0,0,1,0],
#    [0,1,0,0],
#    [0,0,0,1],
#]))

def find_scfa(dim):
    "special Commutative frobenius algebras"

    I = UMatrix.identity(dim)

    swap = UMatrix(get_swap(dim))

    mul = UMatrix.unknown(dim, dim*dim)
    unit = UMatrix.unknown(dim, 1)
    comul = UMatrix.unknown(dim*dim, dim)
    counit = UMatrix.unknown(1, dim)

    solver = Solver()
    Add = solver.add

    # unital
    Add(mul*(unit@I) == I)
    Add(mul*(I@unit) == I)

    # Assoc
    Add(mul*(I@mul) == mul*(mul@I))

    # comm
    Add(mul*swap == mul)

    Add((counit@I)*comul == I)
    Add((I@counit)*comul == I)
    Add((I@comul)*comul == (comul@I)*comul)
    Add(swap*comul == comul)

    # frobenius
    Add( comul*mul == (I@mul)*(comul@I) )
    Add( comul*mul == (mul@I)*(I@comul) )
    # special
    Add( mul*comul == I )

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        _mul = mul.get_interp(model)
        _unit = unit.get_interp(model)
        _comul = comul.get_interp(model)
        _counit = counit.get_interp(model)

        yield SCFA(dim, _unit, _mul, _counit, _comul)

        Add( Or( (mul!=_mul) , (unit!=_unit) ) )


def all_vectors(dim, nonzero=True):
    vs = []
    for idxs in numpy.ndindex((2,)*dim):
        v = numpy.array(idxs)
        v = Matrix(v)
        v = v.reshape(dim, 1)
        if not nonzero or v.sum():
            vs.append(v)
    return vs


class SCFA(object):
    def __init__(self, dim, unit, mul, counit, comul):
        assert unit.shape == (dim, 1)
        assert mul.shape == (dim, dim*dim)
        assert counit.shape == (1, dim)
        assert comul.shape == (dim*dim, dim)
        self.dim = dim
        self.unit = unit
        self.mul = mul
        self.counit = counit
        self.comul = comul
        vs = all_vectors(dim)
        self.copyable = [v for v in vs if comul*v==v@v]
        self.key = (dim, unit, mul, counit, comul)
        self.I = Matrix.identity(dim)

    def __eq__(self, other):
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def act(self, g):
        unit, mul, counit, comul = self.unit, self.mul, self.counit, self.comul
        gi = ~g
        unit = g*unit
        mul = g*mul*(gi@gi)
        counit = counit*gi
        comul = (g@g)*comul*gi
        return SCFA(self.dim, unit, mul, counit, comul)

    def __str__(self):
        s = SMap()
        dim = self.dim
        i = 0
        s[0,i] = str(self.mul); i += dim*dim+1
        s[0,i] = str(self.unit); i += 2
        s[0,i] = str(self.comul); i += dim+1
        s[0,i] = str(self.counit); i += dim+1
        s[0,i] = "(%s)"%(len(self.copyable))
        return str(s)

    def __matmul__(A, B):
        assert A.dim==B.dim
        d = A.dim
        unit = A.unit@B.unit
        counit = A.counit@B.counit
        swap = Matrix(get_swap(d))
        perm = (A.I @ swap @ B.I) 
        mul = (A.mul @ B.mul) * perm
        comul = perm * (A.comul@B.comul)
        return SCFA(d*d, unit, mul, counit, comul)
        

    @classmethod
    def matrix(cls, d):
        "d*d matrix algebra"
        dd = d*d
        I = Matrix.identity(d)
        cup = I.reshape(dd, 1)
        cap = I.reshape(1, dd)
        mul = I@cap@I
        comul = I@cup@I
        return SCFA(dd, cup, mul, cap, comul)


def main_matrix():
    d = argv.get("d", 2)
    vs = all_vectors(d)
    vs = [v.reshape(d,1) for v in vs]
    I = Matrix.identity(d)
    As = list(find_scfa(d))
    As.sort(key = lambda A: (len(A.copyable), A.mul))

    M = SCFA.matrix(d)
    print(M)
    print()
    dd = d**2

    I = Matrix.identity(dd)
    unit, mul, counit, comul = M.unit, M.mul, M.counit, M.comul
    assert comul*mul == (mul@I)*(I@comul)
    assert comul*mul == (I@mul)*(comul@I)
    #print(mul*comul) # zero

    #for u in all_vectors(dd, False):
    #    print(mul * (u@I))
    #    print()

    #for u in all_vectors(d, False):
    #  for v in all_vectors(d, False):
    #    print(u @ v.t)
    #    print()

    A, B = As[0], As[3]

    I = Matrix.identity(d)
    ops = []
    for A in As:
      for u in all_vectors(d):
        a = A.mul*(u@I)
        ops.append(a)

    G = mulclose(ops)
    print(len(G))
    for g in G:
        print(g)
        print()

    return

    A = As[0] @ As[3]
    #print(A)

    solver = Solver()
    Add = solver.add

    g = UMatrix.unknown(dd, dd)
    gi = UMatrix.unknown(dd, dd)
    I = Matrix.identity(dd)
    Add(g*gi==I)

    #print(g*M.unit)
    #print(UMatrix(A.unit.A.astype(int)))
    #print(UMatrix(A.unit))

    Add(UMatrix(A.unit) == g*M.unit )
    Add(UMatrix(A.mul) == g*M.mul*(gi@gi) )
    #Add(UMatrix(A.counit) == M.counit*gi )
    #Add(UMatrix(A.comul) == (g@g)*M.comul*gi )

    result = solver.check()
    assert str(result) == "unsat" # wup
    return

    model = solver.model()
    g = g.get_interp(model)

    print(g)




def main_orbit():
    dim = argv.get("dim", 3)
    vs = all_vectors(dim)
    vs = [v.reshape(dim,1) for v in vs]
    I = Matrix.identity(dim)
    As = list(find_scfa(dim))
    As.sort(key = lambda A: (len(A.copyable), A.mul))

    from qumba.building import Algebraic
    G = Algebraic.SL(dim)
    print(len(G))

    for A in As:
        B = A.act(I)
        assert B == A

    for A in As:
        H = []
        for g in G:
            B = A.act(g)
            #print(B)
            assert B in As
            if A.act(g) == A:
                H.append(g)
        print(len(H), end=" ")
    print()


def main_latex():
    dim = argv.get("dim", 2)
    vs = all_vectors(dim, False)
    I = Matrix.identity(dim)
    As = list(find_scfa(dim))
    As.sort(key = lambda A: (len(A.copyable), A.mul))

    print()

    """
    \newcommand\T{\rule{0pt}{5.6ex}}       % Top strut
    \newcommand\B{\rule[-4.2ex]{0pt}{0pt}} % Bottom strut
    """

    print(r"\hline")
    for A in As:
        ops = [A.mul, A.unit, A.comul, A.counit]
        items = [r"\Field_4" if len(A.copyable)==0 else r"\Field_2^2"]
        items += ["%s"%op.latex() for op in ops]
        items = ["$%s$"%item for item in items]
        row = ' & '.join(items)
        print(row, r'\T\B \\')
        print(r"\hline")

    print('\n'*3)

    for A in As:
        items = [r"\Field_4" if len(A.copyable)==0 else r"\Field_2^2"]
        ops = [A.mul*(v@I) for v in vs]
        #ops.sort(key = lambda A : (A==I))
        #ops.remove(I)
        #ops.insert(1, I)
        items += [m.latex() for m in ops]
        items = ["$%s$"%item for item in items]
        row = ' & '.join(items)
        print(row, r'\T\B \\')
        print(r"\hline")
    print()



def main():
    dim = argv.get("dim", 2)
    _count = 0
    freq = {}
    vs = all_vectors(dim)
    vs = [v.reshape(dim,1) for v in vs]
    I = Matrix.identity(dim)
    print(I, I.shape)
    for A in find_scfa(dim):
        _count += 1
        div = True
        for v in vs:
            m = A.mul*(v@I)
            if m*(~m) != I:
                div = False
                break
        #print(len(A.copyable), "*" if div else " ", end=" ")
        #if _count%16 == 0:
            #print(_count)
        key = len(A.copyable), div # this should class'ify up to dim=4
        freq[key] = freq.get(key, 0) + 1
        #print(A.mul.latex())
        #print()
        continue

        if 1:
            ms = list(find_modules(2, unit, mul))
            #print("modules:", len(items))
            cms = list(find_comodules(2, counit, comul))
            #print("comodules:", len(items))
            assert len(ms) == len(cms)
            print(len(ms), end="", flush=True)
            print("(c=%d)"%len(copyable), end=", ")
            freq[len(ms)] = freq.get(len(ms), 0) + 1
        else:
            print('.', end='', flush=True)
            if _count%60==0:
                print(_count//60)
    print()
    print("found:", _count)
    print(freq)


def main_bialgebra():
    vs = all_vectors(2)
    swap = Matrix(get_swap(2))
    I = Matrix.identity(2)
    H = Matrix([[0,1],[1,0]])
    S = Matrix([[1,0],[1,1]])

    algebras = list(find_scfa(2))
    algebras.sort(key = lambda A: (len(A.copyable), A.mul))

    # adjoint structure (zig-zag)
    cup = I.reshape(4,1)
    cap = cup.t
    for A in algebras:
        cup = A.comul*A.unit
        cap = A.counit*A.mul
        #print(A)
        #print(cap, len(A.copyable))
        #print()
        assert (I@cap)*(cup@I) == I
        assert (cap@I)*(I@cup) == I

    #return
        
    g_gg = Matrix([
        [1,0,0,0],
        [0,0,0,1],
    ]) # green mul

    r_rr = Matrix([
        [1,0,0,0],
        [0,1,1,1],
    ]) # red mul

    b_bb = Matrix([
        [1,1,1,0],
        [0,0,0,1],
    ]) # blue mul

    for A in algebras:
        print(A)

    A = algebras[0]
    for v in vs:
        lhs = A.comul*v
        for u in vs:
          for w in vs:
            rhs = v@u
            #print( lhs==rhs, end=" ")
        #print()
        #print(lhs)
        #print("=/=")
        #print(rhs)
        #print()

    for A in algebras:
        cup = A.comul*A.unit
        cap = A.counit*A.mul
        #print(A)
        #print(cap, len(A.copyable))
        #print()
        assert (I@cap)*(cup@I) == I
        assert (cap@I)*(I@cup) == I

    #return

    green, = [A for A in algebras if A.mul == g_gg]
    red, = [A for A in algebras if A.mul == r_rr]
    blue, = [A for A in algebras if A.mul == b_bb]

    g_ = green.unit
    _g = green.counit
    gg_g = green.comul

    r_ = red.unit
    _r = red.counit
    rr_r = red.comul

    b_ = blue.unit
    _b = blue.counit
    bb_b = blue.comul

    assert rr_r != bb_b
    
    #return

    # bialgebra
    assert gg_g * r_ == r_@r_ 
    assert gg_g * b_ == b_@b_ 
    assert rr_r * g_ == g_@g_
    assert rr_r * b_ == b_@b_
    assert bb_b * g_ == g_@g_
    assert bb_b * r_ == r_@r_

    lhs = (gg_g * r_rr)
    rhs = (r_rr@r_rr) * (I@swap@I) * (gg_g@gg_g)
    assert lhs == rhs

    lhs = (gg_g * b_bb)
    rhs = (b_bb@b_bb) * (I@swap@I) * (gg_g@gg_g)
    assert lhs == rhs

    lhs = (rr_r * b_bb)
    rhs = (b_bb@b_bb) * (I@swap@I) * (rr_r@rr_r)
    assert lhs == rhs

    assert r_*_r == Matrix([[0,1],[0,0]])
    assert r_*_b == Matrix([[1,0],[0,0]])
    assert b_*_b == Matrix([[0,0],[1,0]])
    assert b_*_r == Matrix([[0,0],[0,1]])

    assert g_*_b == Matrix([[1,0],[1,0]])
    assert g_*_r == Matrix([[0,1],[0,1]])
    assert r_*_g == Matrix([[1,1],[0,0]])
    assert b_*_g == Matrix([[0,0],[1,1]])

    #print(g_gg * (g_@I))

    G = mulclose([S,H])
    assert len(G) == 6
    for u in G:
        muls = [g_gg, r_rr, b_bb]
        lhs = [u*mul for mul in muls]
        rhs = [mul*(u@u) for mul in muls]
        idxs = [lhs.index(mul) for mul in rhs]
        assert set(idxs) == {0, 1, 2}

    assert H*H==I
    assert H*g_gg == g_gg*(H@H)

    # facet 
    u = H*S
    #for mul in [g_gg, r_rr, b_bb]:
    #    print(  u*g_gg*(u@u) == mul )

    u = S*H*S
    assert u*u == I
    assert u*b_bb*(u@u) == g_gg
    assert u*r_rr*(u@u) == r_rr

#    gen = (
#        list(G) + [swap] + 
#        [g_gg, gg_g, g_, _g] +
#        [r_rr, rr_r, r_, _r])
#        #[b_bb, bb_b, b_, _b, swap])

    #gen = list(G) + [swap] # + vs
    gen = []
#    for A in algebras:
    for A in [red, green, blue]:
        gen += [A.unit, A.mul, A.counit, A.comul]
    print()
    print()
    print()
    for A in algebras:
        if len(A.copyable) == 2:
            continue
        print("_"*79)
        print(A)
        for v in vs:
            print(A.mul * (v@I) )
            print()


    #return

    for mul in [g_gg, r_rr, b_bb]:
      for comul in [gg_g, rr_r, bb_b]:
        m = (I@mul)*(comul@I)
        #print(m)
        #print()

    #return
    
    U = Matrix([
        [1,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,1,0,0],
    ])

    a = U@I
    b = I@U
    print( a*b*a == b*a*b )

    U1 = Matrix([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,0,1],
        [0,0,1,0],
    ])
    #search = [b_bb, bb_b, b_, _b]
    #search = [U]
    #monoidal_search(gen, [S], 5)


def monoidal_search(gen, search, maxdim=4):

    found = set(gen)
    bdy = list(found)
    while bdy:

        _bdy = []
        for a in bdy:
          for b in gen:
            items = [a@b, b@a]
            if a.shape[1] == b.shape[0]:
                items.append(a*b)
            if a.shape[0] == b.shape[1]:
                items.append(b*a)
            for c in items:
                if c.shape[0]*c.shape[1] <= 2**maxdim and c not in found:
                    found.add(c)
                    _bdy.append(c)
        bdy = _bdy
        print("[%d:%d]"%(len(found), len(bdy)), end="", flush=True)
        #if U in found:
        #    break
        for a in _bdy:
            assert len(a.shape) == 2
        for U in search:
            if U not in found:
                break
        else:
            break

    print([U in found for U in search])


def main_spider():
    vs = all_vectors(2)
    swap = Matrix(get_swap(2))
    I = Matrix.identity(2)
    H = Matrix([[0,1],[1,0]])
    S = Matrix([[1,0],[1,1]])

    algebras = list(find_scfa(2))
    algebras.sort(key = lambda A: (len(A.copyable), A.mul))

    #A = algebras[0]
    #print(A)

    gen = [I]
#    for A in [algebras[0], algebras[1]]:
    for A in algebras:
        gen += [A.mul, A.unit, A.comul, A.counit]
    gen = set(gen)
    M = monoidal_gen(gen)
    print(len(M))

    #for g in M:
    #    print(g.shape, end=' ')
    k = 2
    I = Matrix.identity(2**k)

    M = [g for g in M if g.shape==(2**k, 2**k)]
    G = []
    for g in M:
        gi = ~g
        if gi is None:
            continue
        #print('/')
        #print(g*gi)
        if g*gi == I:
            G.append(g)
    G = mulclose(G)
    print("|G| =", len(G))


def monoidal_gen(gen, maxdim=4):

    found = set(gen)
    bdy = list(found)
    while bdy:

        _bdy = []
        for a in bdy:
          for b in gen:
            items = [a@b, b@a]
            if a.shape[1] == b.shape[0]:
                items.append(a*b)
            if a.shape[0] == b.shape[1]:
                items.append(b*a)
            for c in items:
                if c.shape[0]*c.shape[1] <= 2**maxdim and c not in found:
                    found.add(c)
                    _bdy.append(c)
        bdy = _bdy
    return found


def main_hopf():
    dim = argv.get("dim", 2)
    swap = Matrix(get_swap(dim))
    I = Matrix.identity(dim)

    algebras = list(find_scfa(dim))
    print("algebras:", len(algebras))

    algebras.sort(key = lambda A : len(A.copyable))

    algebras = [A for A in algebras if len(A.copyable)==dim]
    #algebras = [A for A in algebras if len(A.copyable)]
    names = {}
    for i,A in enumerate(algebras):
        n = len(A.copyable)
        stem = "abcdefgh"[n]
        names[i] = "%s%d"%(stem, i)

    f = open("hopf_%d.dot"%dim, "w")
    print("graph {", file=f)

    # adjoint structure (zig-zag)
    cup = I.reshape(dim*dim,1)
    cap = cup.t
    for i, A in enumerate(algebras):
        cup = A.comul*A.unit
        cap = A.counit*A.mul
        #print(A)
        #print(cap, len(A.copyable))
        #print()
        assert (I@cap)*(cup@I) == I
        assert (cap@I)*(I@cup) == I

        aa_a, a_aa = A.comul, A.mul
        _a, a_ = A.counit, A.unit
        for j, B in enumerate(algebras):
            if A==B:
                print(" ", end='')
                continue
            bb_b, b_bb = B.comul, B.mul
            _b, b_ = B.counit, B.unit

            if aa_a * b_ != b_@b_ or bb_b*a_ != a_@a_:
                print(".", end='')
                continue
            
            lhs = (aa_a * b_bb)
            rhs = (b_bb@b_bb) * (I@swap@I) * (aa_a@aa_a)
            if lhs != rhs:
                print(".", end='')
                continue

            lhs = (bb_b * a_aa)
            rhs = (a_aa@a_aa) * (I@swap@I) * (bb_b@bb_b)
            if lhs != rhs:
                print(".", end='')
                continue

            print("*", end='')
            if i<j:
                print("  %s -- %s;"%(names[i],names[j]), file=f)
        print()
    print("}", file=f)




def css_get_wenum(code):
    from qumba.solve import dot2
    css = code.to_css()
    Hz = css.Hz
    Lz = css.Lz
    wenum = {w:[] for w in range(code.n+1)}
    for ik in numpy.ndindex((2,)*css.k):
      if ik == (0,)*css.k:
        continue
      for imz in numpy.ndindex((2,)*css.mz):
        h = dot2(imz, css.Hz) + dot2(ik, css.Lz)
        h %= 2
        wenum[h.sum()].append(h)

    return wenum


def get_wenum(code):
    H = code.H
    m, nn = H.shape
    wenum = {i:0 for i in range(nn+1)}
    for idxs in numpy.ndindex((2,)*m):
        a = Matrix(idxs).reshape(1,m)
        v = a*H
        wenum[v.sum()] += 1 # not the real weight... fix?
    return [wenum[i] for i in range(nn+1)]


def main_unwrap():
    code = construct.get_513()
    #print(get_wenum(code))
    #code = construct.get_14_3_3()
    #code = construct.get_412()
    #code = construct.toric(2,2).to_qcode()
    #code = construct.get_512()
    print(code)
    H = code.H
    m, nn = H.shape
    n = nn//2
    H = H.reshape(m, n, 2)
    #print(H)

    swap = Matrix([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1],
    ])

    g = Matrix.get_perm([1,0,2,3])
    h = Matrix.get_perm([1,2,3,0])
    G = mulclose([g,h])
    assert len(G) == 24

    G = []

    A = get_algebra()
    for a in A:
      for b in A:
        U = numpy.zeros((4,4), dtype=scalar)
        U[:2,:2] = a
        U[:2,2:] = b
        U[2:,:2] = b
        U[2:,2:] = a
        U = Matrix(U)
        #print(U)
        #return
        G.append(U)
    #return

    assert len(G) == 256

    space = SymplecticSpace(nn)

    found = 0
    for U in G:
        U = U.reshape(2,2,2,2)
        #J = (H@U).reshape(m, n, 2, 2, 2, 2, 2)
    
        J = Matrix.einsum("mno,opqr", H, U)
        #J = J.transpose((0,2,1,3,4))
        J = J.transpose((0,2,1,4,3))
        J = J.reshape(2*m, 2*n, 2)
    
        #print(J)
        J = J.reshape(2*m, 4*n)

        if J.rank() != 2*m:
            continue

        if not space.is_isotropic(J):
            continue

        code = QCode(J)
    
        #if not code.is_css():
        #    continue

        print(code, "*" if code.is_css() else " ", "gf4" if code.is_gf4() else " ",
            "+" if code.is_selfdual() else " ")

#        if code.is_gf4():
#            #print(strop(code.H))
#            H0 = code.H.normal_form()
#            print(strop(H0))
        #print(U.reshape(4,4))
        found += 1

        #print(code.longstr())
        #wenum = get_wenum(code)
        #print(wenum)

    print("found:", found)




def main_modules():

    m_dim = 2 # module dim is 2 for qubits
    a_dim = 2 # algebra dim

    vs = all_vectors(a_dim)
    vs = [v.reshape(a_dim, 1) for v in vs]
    I = Matrix.identity(m_dim)

    found = set()
    for unit, mul in find_algebras(a_dim):
    #for A in find_scfa(a_dim):
        #unit, mul = A.unit, A.mul
        for act in find_modules(m_dim, unit, mul):
            lins = []
            #found.append((unit, mul, act))
            for v in vs:
                u = act*(v@I)
                if u.sum():
                    lins.append(u)
                #print(u,'\n')
            lins.sort()
            lins = tuple(lins)
            found.add(lins)
    print("modules:", len(found))
    #return
    found = list(found)
    found.sort(key = lambda a:(len(a), a))

    m, n = 2, 4
    space = SymplecticSpace(n)

    sigs = {}
    _count = 0
    for C in space.grassmannian(m):
        C2 = C.reshape(m, n, 2)
        #print(C2,'\n')
        Ct = C.t
        sig = []
        for algebra in found:
            for a in algebra:
                D = C2*a.t
                D = D.reshape(m, 2*n)
                u = Ct.solve(D.t)
                if u is None:
                    sig.append(".")
                    break
            else:
                sig.append("*")
        sig = ''.join(sig)
        sigs[sig] = sigs.get(sig, 0) + 1
        _count += 1
    print("codes:", _count)

    keys = list(sigs)
    keys.sort()
    for i,key in enumerate(keys):
        print(key, i, sigs[key])
    

def main_comodules():

    m_dim = 2 # comodule dim is 2 for qubits
    a_dim = 2 # algebra dim

    vs = all_vectors(a_dim)
    #vs = [v.reshape(a_dim, 1) for v in vs]
    I = Matrix.identity(m_dim)

    found = set()
    for counit, comul in find_coalgebras(a_dim):
        for coact in find_comodules(m_dim, counit, comul):
            lins = []
            #found.append((unit, mul, act))
            for v in vs:
                u = (v@I)*coact
                if u.sum():
                    lins.append(u)
                #print(u,'\n')
            lins.sort()
            lins = tuple(lins)
            found.add(lins)
    print("comodules:", len(found))
    found = list(found)
    found.sort(key = lambda a:(len(a), a))

    m, n = 2, 4
    space = SymplecticSpace(n)

    sigs = set()
    _count = 0
    for C in space.grassmannian(m):
        C2 = C.reshape(m, n, 2)
        #print(C2,'\n')
        Ct = C.t
        sig = []
        for algebra in found:
            for a in algebra:
                D = C2*a.t
                D = D.reshape(m, 2*n)
                u = Ct.solve(D.t)
                if u is None:
                    sig.append(".")
                    break
            else:
                sig.append("*")
        sig = ''.join(sig)
        sigs.add(''.join(sig))
        _count += 1
    print("codes:", _count)

    sigs = list(sigs)
    sigs.sort()
    for i,sig in enumerate(sigs):
        print(sig, i)
    








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



