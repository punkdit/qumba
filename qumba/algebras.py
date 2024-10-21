#!/usr/bin/env python

"""
See:
    https://arxiv.org/abs/quant-ph/9703048
    Nonbinary quantum codes
    Eric M. Rains

"""

from functools import reduce, cache
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


def generate(gen, verbose=False, maxsize=None):
    els = set(gen)
    bdy = list(els)
    changed = True
    while bdy:
        if verbose:
            print(len(els), end=" ", flush=True)
        _bdy = []
        for A in gen:
            for B in bdy:
                for C in [A*B, A+B]:
                  if C not in els:
                    els.add(C)
                    _bdy.append(C)
                    if maxsize and len(els)>=maxsize:
                        return els
        bdy = _bdy
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



def main_rains():

    space = SymplecticSpace(1)
    I = space.I()
    S = space.S()
    H = space.H()

    G = mulclose([S,H])
    G = list(G)
    assert len(G) == 6

    algebra = span(G)
    assert len(algebra) == 16

    print("algebra:")
    Algebra(algebra).dump()
    #return

    for g in G:
        assert g*~g == I

    count = 0
    found = set()
    for gen in all_subsets(algebra):
        count += 1
        if algebra[0] not in gen:
            continue

        if I not in gen:
            continue

        A = Algebra.generate(gen)
        found.add(A)

        #if len(found) > 2:
        #    break

    J = H
    conj = lambda a : J*a.t*J

    print("found:", len(found), "count:", count)
    found = list(found)
    found.sort(key = lambda A : (len(A), tuple(A)))
    for A in found:
        print("|A| = ", len(A))
        A.dump()

        for a in A:
            assert conj(a) in A
            for b in A:
                assert a+b in A
                assert a*b in A
                assert conj(a*b) == conj(b)*conj(a)

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

        if sig == "*......*....":
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
    count = 0
    for (unit, mul) in find_add_algebras(dim):
        count += 1
    # cocartesian => unique algebras
    assert count == 1
        



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
    count = 0
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
        count += 1
    print()
    print("found:", count)


def get_swap(d):
    swap = numpy.zeros((d*d, d*d), dtype=scalar)
    #for i in range(d*d):
    for i in range(d):
      for j in range(d):
        row = d*i + j
        col = d*j + i
        swap[row, col] = 1
    return UMatrix(swap)

#print(get_swap(2))
#print(get_swap(3))

assert str(get_swap(2)) == str(UMatrix([
    [1,0,0,0],
    [0,0,1,0],
    [0,1,0,0],
    [0,0,0,1],
]))

def find_scfa(dim):
    "special Commutative frobenius algebras"

    I = UMatrix.identity(dim)

    swap = get_swap(dim)

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

        yield (_unit, _mul, _counit, _comul)

        Add( Or( (mul!=_mul) , (unit!=_unit) ) )


def nonzero_vectors(dim):
    vs = []
    for idxs in numpy.ndindex((2,)*dim):
        v = numpy.array(idxs)
        v = Matrix(v)
        if v.sum():
            vs.append(v)
    return vs

def main():
    dim = argv.get("dim", 2)
    count = 0
    freq = {}
    vs = nonzero_vectors(dim)
    for (unit, mul, counit, comul) in find_scfa(dim):
        #print("unit =")
        #print(unit)
        #print("mul =")
        #print(mul)
        count += 1
        copyable = [v for v in vs if comul*v==v@v]

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
            if count%60==0:
                print(count//60)
    print()
    print("found:", count)
    print(freq)


def main_codes():

    code = construct.get_513()
    H = code.H
    m, nn = H.shape
    n = nn//2
    H = H.reshape(m, n, 2)
    print(H, H.shape)

    # try to tensor product a code with an algebra.. fail

    dim = 2
    count = 0
    vs = nonzero_vectors(dim)
    omega = Matrix([[0,1],[1,0]])
    I = Matrix.identity(2)
    zero = Matrix([[0,0],[0,0]])
    for (unit, mul) in find_algebras(dim):
        copyable = [v for v in vs if v*mul==v@v]
        #print("unit =")
        #print(unit)
        #print("mul =")
        #print(mul)
        #print("c =", len(copyable))
        comul = mul.t
        #special = mul * comul == I
        #cond = omega*mul == mul * (omega @ I)
        #cond = mul * (omega @ I) * comul == omega
        #cond = comul * omega * mul == omega @ zero
        op = comul*mul*(I@omega)*comul*mul
        #op = comul*mul*(omega@I)*comul*mul
        if op.sum():
            continue
        # H : (m, n, 2)
        op = (comul * mul).reshape(2, 2, 2, 2)
        #print(op.shape)
        IH = (I@H).reshape(2, 2, *H.shape)
        #print(IH.shape) # (2,2,m,n,2)  (i,j,m,n,k)
        H1 = Matrix.einsum("ijmnk,jkuv", IH, op)
        #print(H1.shape)
        H1 = H1.transpose((0,3,4,1,2))
        #print(H1.shape)
        H1 = H1.reshape(2*m, 4*n)
        #print(H1.shape)
        H1 = H1.linear_independent()
        code = QCode(H1)
        print(code)
        #print(code.longstr())
        count += 1
    print()
    print("found:", count)



def main_modules():

    m_dim = 2 # module dim is 2 for qubits
    a_dim = 2 # algebra dim

    vs = nonzero_vectors(a_dim)
    vs = [v.reshape(a_dim, 1) for v in vs]
    I = Matrix.identity(m_dim)

    found = set()
    for unit, mul in find_algebras(a_dim):
    #for unit, mul, _, _ in find_scfa(a_dim):
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

    sigs = set()
    count = 0
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
        count += 1
    print("codes:", count)

    sigs = list(sigs)
    sigs.sort()
    for i,sig in enumerate(sigs):
        print(sig, i)
    

def main_comodules():

    m_dim = 2 # comodule dim is 2 for qubits
    a_dim = 2 # algebra dim

    vs = nonzero_vectors(a_dim)
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
    count = 0
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
        count += 1
    print("codes:", count)

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



