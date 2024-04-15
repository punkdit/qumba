#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
import pickle

import numpy

from qumba import solve
solve.int_scalar = numpy.int32 # qupy.solve
from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba.autos import get_autos
from qumba import csscode, construct
from qumba.construct import get_422, get_513, golay, get_10_2_3, reed_muller
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.util import cross
from qumba.symplectic import Building
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
from qumba import clifford, matrix
from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
from qumba.syntax import Syntax
from qumba.circuit import Circuit, measure, barrier, send



def test_513():
    base = construct.get_513()
    n = base.n
    print(base)
    print(base.longstr())

    s = base.space
    c = Clifford(n)
    E = reduce(mul, [c.get_CZ(i, (i+1)%n) for i in range(n)])
    v0 = E*red(n, 0, 0)
    v1 = E*red(n, 0, 2)
    assert (v0.d*v1)[0][0] == 0 # orthogonal

    P = base.get_projector()
    assert P*P == P
    assert P*v0 == v0
    assert P*v1 == v1

    lx = -c.get_pauli("XXXXX") # arf, what is this minus sign ??
    lz = c.get_pauli("ZZZZZ")
    #ly = c.get_pauli("YYYYY")
    ly = lz*lx
    assert lz*v0 == v0
    assert lz*v1 == -v1
    assert lx*v0 == v1
    assert lx*v1 == v0

    S, H = c.S, c.H
    g = reduce(mul, [S(i)*H(i) for i in range(n)])
    assert g*P == P*g
    u = g*v0
    print(lz*u==u, lz*u==-u)
    print(lx*u==u, lx*u==-u)
    print(ly*u==u, ly*u==-u)


def qupy_513():
    base = construct.get_513()
    n = base.n
    c = Clifford(n)
    P = base.get_projector()
    E = base.get_clifford_encoder()
    ops = [red(1,0)]*n
    v0 = reduce(matmul, ops)
    #cdump(c.get_expr(E.name[-4:])*v0)
    ops[-1] = red(1,0,2)
    v1 = reduce(matmul, ops)
    v0 = E*v0
    #cdump(v0)
    assert P*v0 == v0
    v1 = E*v1
    assert P*v1 == v1

    lx = -c.get_pauli("XXXXX")
    lz = c.get_pauli("ZZZZZ")

    #v1 = lx*v0

    assert lz*v0 == v0
    assert lz*v1 == -v1
    assert lx*v0 == v1
    assert lx*v1 == v0
    #print(lx*v0 == v1, lx*v0 == -v1)
    #print(lx*v1 == v0, lx*v1 == -v0)

    # ----------- qupy -------------------------------------------- #
    from qupy.qumba import Space, Operator, Code, CSSCode, eq, scalar # uses reversed bit order XXX
    space = Space(n)
    H = base.H
    _code = Code(strop(H).replace(".", "I"))
    _code.check()
    v0 = numpy.zeros((2**n,), dtype=scalar)
    v0[0] = 4*(2**0.5)
    g = space.get_expr(E.name)
    v0 = g*v0
    #vdump(v0)
    P = _code.P
    assert eq(P*v0, v0)
    #return

    #print(E.name)
    cover = Cover.frombase(base)
    code = cover.total
    nn = code.n
    E1 = cover.get_expr(E.name)
    prep = E1.name
    for i in range(n):
        prep += ("H(%d)"%(i+n),)
    #print(prep) # FAIL

    if 0:
        E = code.get_encoder_name()
        prep = E.name # FAIL

    #c = Clifford(nn)
    #E = c.get_expr(prep)

    space = Space(nn)
    #E = space.get_expr(prep)
    #print(E)

    if 0:
        prep = ('CX(4,5)', 'CX(0,9)', 'CX(3,9)', 'CX(4,8)', 'CX(2,8)',
            'CX(3,7)', 'CX(1,7)', 'CX(2,6)', 'CX(0,6)', 'CX(1,5)') # FAIL
        v0 = numpy.zeros((2**n,), dtype=scalar)
        v0[0] = 1
        v0[-1] = 1
        H = Space(n).H
        g = reduce(mul, [H(i) for i in range(n)])
        v1 = g*v0
        vdump(v0)
        vdump(v1)
        v = numpy.kron(v1, v0)
        vdump(v)
        print(v.shape)
    
        g = space.get_expr(prep)
        v0 = g*v

    print(prep)
    g = space.get_expr(prep)
    v0 = numpy.zeros((2**nn,), dtype=scalar)
    v0[0] = 1
    #for i in range(8):
    #    v0 = space.H(i)*v0
    v0 = g*v0

    #Hs = strop(code.H).replace('.', 'I')
    #_code = Code(Hs)
    #_code.check()
    #P = _code.P

    css = code.to_css()
    css = CSSCode(css.Hz, css.Hx)
    P = ((1/2)**len(css.stabs))*css.P
    assert P*P == P

    print(eq(P*v0, v0)) # FAIL 

    H = code.H
    _code = Code(strop(H).replace(".", "I"))
    P = _code.P
    print(eq(P*v0, v0)) # FAIL

    return

    v = css.get_encoded(0)
    vdump(v)
    assert eq(P*v, v)

    v0 = numpy.zeros((2**nn,), dtype=scalar)
    v0[0] = 1 # ket |00...0>
    v0 = E*v0
    u = P*v0
    vdump(v0)
    vdump(u)

    return

    #print(code.longstr())
    css = code.to_css()
    Hz = css.Hz
    #print(Hz)

    circuit = Circuit(nn)
    c = measure + prep
    qasm = circuit.run_qasm(c)
    idxs = circuit.labels # final qubit permutation

    Hz = Hz[:, idxs]

    shots = argv.get("shots", 10)
    samps = send([qasm], shots=shots)
    print(samps)
    for v in samps:
        v = parse(v)
        print(v)
        syndrome = dot2(Hz, v.transpose())
        print(syndrome)


def clifford_513_unwrap():
    base = construct.get_513()
    n = base.n
    c = Clifford(n)
    P = base.get_projector()
    E = base.get_clifford_encoder()
    ops = [red(1,0)]*n
    v0 = reduce(matmul, ops)
    #cdump(c.get_expr(E.name[-4:])*v0)
    ops[-1] = red(1,0,2)
    v1 = reduce(matmul, ops)
    v0 = E*v0
    #cdump(v0)
    assert P*v0 == v0
    v1 = E*v1
    assert P*v1 == v1

    lx = -c.get_pauli("XXXXX")
    lz = c.get_pauli("ZZZZZ")

    #v1 = lx*v0

    assert lz*v0 == v0
    assert lz*v1 == -v1
    assert lx*v0 == v1
    assert lx*v1 == v0
    #print(lx*v0 == v1, lx*v0 == -v1)
    #print(lx*v1 == v0, lx*v1 == -v0)

    #print(E.name)
    cover = Cover.frombase(base)
    code = cover.total
    nn = code.n
    E1 = cover.get_expr(E.name)
    prep = E1.name
    for i in range(n):
        prep += ("H(%d)"%(i+n),)
    print(prep)

    c = Clifford(nn)
    ops = [red(1,0)]*nn
    v0 = reduce(matmul, ops)
    cdump(v0)

    for g in reversed(prep):
        print(g)
        g = c.get_expr(g)
        v0 = g*v0
    P = code.get_projector()
    print(P*v0==v0) # FAIL



def symplectic_10_2_3():
    base = construct.get_513()
    code = unwrap(base)
    print(code.longstr())
    #css = code.to_css()
    perms = get_autos(code)
    print(perms)
    swap = SymplecticSpace(2).get_SWAP()
    for perm in perms:
        dode = code.apply_perm(perm)
        assert dode.is_equiv(code)
        if dode.get_logical(code) == swap:
            print(perm)


def clifford_10_2_3():
    base = construct.get_513()
    code = unwrap(base)
    n = base.n
    nn = code.n
    #print(code.longstr())
    print(base)
    print(code)

    if 0:
        lhs = red(n, 0)
        rhs = reduce(matmul, [red(1,0)]*n)
        print(lhs == rhs)
        print(lhs)
        print(rhs)
    
        return

    P = code.get_projector()
    #assert P*P == P # SLOOOW

    cc = Clifford(nn)
    CX = cc.CX
    H = code.H
    H = strop(H).split()
    #for h in H:
    #    print(h)
    #    op = cc.get_pauli(h) # SLOOOW

    def unwrap_CZ(src, tgt, v):
        #return CX(2*src, 2*tgt+1) * (CX(2*src+1, 2*tgt) * v)
        return CX(src, tgt+n) * (CX(tgt, src+n) * v)

    #v = red(nn, 0)
    v = red(n, 0) @ green(n, 0)
    print("v", v.shape)
    for i in range(n):
        v = unwrap_CZ(i, (i+1)%n, v)
        print("i =", i)
    print(P*v == v)

    Z0 = cc.get_pauli("ZZZZZ.....")
    assert Z0*v == v
    
    X1 = cc.get_pauli("..XX.X....")
    assert X1*v == v

    # we have the |0+> state


def test_qupy():
    from qupy.qumba import Space, Operator, Code, CSSCode, eq, scalar # uses reversed bit order XXX
    n = 3
    space = Space(n)
    CX, X, Z = space.CX, space.X, space.Z
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        assert CX(i,j)*X(i) == X(i)*X(j)*CX(i,j)
        assert CX(i,j)*X(j) == X(j)*CX(i,j)
        assert CX(i,j)*Z(i) == Z(i)*CX(i,j)
        assert CX(i,j)*Z(j) == Z(i)*Z(j)*CX(i,j)

    code = construct.get_713()
    qupy_code(code)


#def get_dual(name):
    

class GL(object):
    def __init__(self, n):
        self.n = n
    def CX(self, i, j):
        n = self.n
        A = numpy.identity(n, dtype=int)
        A[i,j] = 1
        return matrix.Matrix(A, name="CX(%d,%d)"%(i,j))
    def I(self, i=0):
        n = self.n
        A = numpy.identity(n, dtype=int)
        return matrix.Matrix(A, name=())
    H = I # gobble this
    def get_expr(self, expr, rev=False):
        if expr == ():
            op = self.I
        elif type(expr) is tuple:
            if rev:
                expr = reversed(expr)
            op = reduce(mul, [self.get_expr(e) for e in expr]) # recurse
        else:
            expr = "self."+expr
            op = eval(expr, {"self":self})
        return op


def qasm_10_2_3():
    base = construct.get_513()
    code = unwrap(base)
    n, nn = base.n, code.n

    # prepare |0+> state

    s = Syntax()
    CX, H = s.CX, s.H
    def unwrap_CZ(src, tgt):
        return CX(src, tgt+n) * CX(tgt, src+n)

    g = s.get_identity()
    for i in range(n):
        g = unwrap_CZ(i, (i+1)%n)*g

    # uses a couple of GHZ states ... bad
    for i in range(1, n):
        g = g * CX(i, 0)
    for i in range(n+1, nn):
        g = g * CX(n, i)
    for i in range(1, n):
        g = g * H(i)
    g = g * H(n)

    g = reduce(mul, [H(i) for i in range(nn)])*g

    prep = g.name
    print(prep)

    if 0:
        gl = GL(nn)
        A = gl.get_expr(prep)
        print(A)
        print(A.name, len(A.name))
    
        best = len(A.name)
    
        CX = gl.CX
        gen = [CX(i,j) for i in range(nn) for j in range(nn) if i!=j]
    
        for trial in range(1000):
            B = A
            I = gl.I()
            count = str(B+I).count('1')
            tgt = I
            while B != I:
                shuffle(gen)
                for g in gen:
                    h = g*B
                    a = str(h + I).count('1')
                    if a < count:
                        #print(count)
                        count = a
                        B = h
                        tgt = tgt*g
                        break
                else:
                    assert 0
            if len(tgt.name) < best:
                print(tgt.name, len(tgt.name))
                best = len(tgt.name)
    
        found = ('CX(1,5)', 'CX(5,9)', 'CX(4,0)', 'CX(3,9)', 'CX(2,8)',
        'CX(1,7)', 'CX(5,6)', 'CX(2,0)', 'CX(5,8)', 'CX(4,5)',
        'CX(0,9)', 'CX(0,6)', 'CX(3,0)', 'CX(2,9)', 'CX(5,7)', 'CX(3,7)', 'CX(1,0)')
        return


    css = code.to_css()
    Hz = numpy.concatenate((css.Hz, css.Lz))
    n = code.n

    circuit = Circuit(n)
    c = measure + prep
    qasm = circuit.run_qasm(c)
    #print(qasm)
    idxs = circuit.labels # final qubit permutation

#    idxs = list(reversed(range(n))) # <--------- XXX  This is Hx 
#    Hz = Hz[:, idxs]
#    #print(Hz)

    shots = argv.get("shots", 10)
    samps = send([qasm], shots=shots, error_model=False)
    print(samps)
    for v in samps:
        v = parse(v)
        syndrome = dot2(Hz, v.transpose())
        print(v, syndrome.transpose())


def qupy_10_2_3():
    base = construct.get_513()
    code = unwrap(base)
    n = code.n

    Hx = code.to_css().Hx
    Hx = row_reduce(Hx)
    #print(shortstr(Hx))
    """
    0123456789
    1..1..11..
    .1..1..11.
    ..11..1111
    ...111.111
    """

    # shave off 2 CX gates:
    Hx[2] += Hx[3]
    Hx %= 2
    qupy_code(code)


def test_css():
    #code = construct.get_713()
    #code = construct.get_toric(3, 3)
    #code = construct.reed_muller() # [[16,6,4]]
    code = construct.get_toric(4,0) # [[16,2,4]]
    qupy_code(code)


def qupy_code(code):
    print(code)
    #print(code.longstr())
    n = code.n
    k = code.k
    Hx = code.to_css().Hx
    Hx = row_reduce(Hx)
    #print(shortstr(Hx))

    s = Syntax()
    CX, H, I = s.CX, s.H, s.get_identity()
    g = I
    
    idxs = []
    for row in range(len(Hx)):
        j0 = row
        while Hx[row, j0] == 0:
            j0 += 1
        assert Hx[row, j0]
        idxs.append(j0)
        for j in range(j0+1, n):
            if Hx[row, j]:
                g = g*CX(j0,j)
    #print(idxs)
    for i in idxs:
        g = g*H(i)
    print(g.name, len(g.name))

    #g = g*reduce(mul, [H(i) for i in range(n)])
    prep = g.name

    if n <= 10:
        from qupy.qumba import Space, Operator, Code, CSSCode, eq, scalar # uses reversed bit order XXX
    
        space = Space(n)
        
        g = space.get_expr(prep)
        v0 = numpy.zeros((2**n,), dtype=scalar)
        v0[0] = 1
        #v0[-1] = 1
        v0 = g*v0
        #vdump(v0)
    
        css = code.to_css()
        css = CSSCode(css.Hz, css.Hx)
        P = ((1/2)**len(css.stabs))*css.P
        assert P*P == P
    
        assert eq(P*v0, v0)
    
        u0 = numpy.zeros((2**n,), dtype=scalar)
        u0[0] = 4
        u0 = P*u0
        #assert eq(v0, u0)
    
        vdump(v0)
    
        Hs = strop(code.H).split()
        for h in Hs:
            #print(h)
            h = space.make_op(h)
            #vdump(h*v0)
            assert eq(h*v0, v0)

    #print(code.longstr())
    css = code.to_css()
    Hz = numpy.concatenate((css.Hz, css.Lz))
    #print(Hz)

    circuit = Circuit(n)
    c = measure + prep
    qasm = circuit.run_qasm(c)
    #print(qasm)
    idxs = circuit.labels # final qubit permutation

    idxs = list(reversed(range(n))) # <--------- AAAAAAAAARRRRRRRRGGGGGG 
    Hz = Hz[:, idxs]
    #print(Hz)

    shots = argv.get("shots", 1000)
    samps = send([qasm], shots=shots, error_model=True)
    #print(samps)
    fail = 0
    count = 0
    for v in samps:
        v = parse(v)
        check = dot2(v, Hz.transpose())
        syndrome = check[:,:-k]
        err = check[:, -k:]
        #print(v, syndrome, err)
        if syndrome.sum():
            count += 1
        if err.sum() and not syndrome.sum():
            fail += 1
    print("syndrome:", count)
    print("fail:", fail)




if __name__ == "__main__":

    from time import time
    start_time = time()

    print(argv)

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        from random import seed
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)

