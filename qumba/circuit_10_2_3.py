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
from qumba import csscode, construct
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
from qumba import transversal 
from qumba import clifford, matrix
from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
from qumba.syntax import Syntax
from qumba.circuit import (Circuit, measure, barrier, send, vdump, variance,
    parsevec, strvec, find_state, get_inverse, load_batch, send_idxs)
from qumba.circuit_css import css_encoder, process



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


def get_p0_prep():
    # prepare |0+> state
    base = construct.get_513()
    code = unwrap(base)
    n, nn = base.n, code.n

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
    return prep


def qasm_find_p0():
    base = construct.get_513()
    code = unwrap(base)
    n, nn = base.n, code.n

    # prepare |0+> state
    prep = get_p0_prep()
    print(prep)

    #c = Clifford(nn)
    #v = parsevec("0"*nn)
    #for e in reversed(prep):
    #    v = c.get_expr(prep) * v
    #print(strvec(v))

    sp = SymplecticSpace(nn)
    Ep0 = sp.get_expr(prep)
    print(Ep0)
    print()

    E = code.get_encoder(Ep0)
    Ei = sp.invert(E)

    EE = Ep0 * Ei # |0+> <--- |00>
    print(EE)
    print(sp.get_name(EE))



def qasm_10_2_3_p0():
    base = construct.get_513()
    code = unwrap(base)
    n, nn = base.n, code.n

    # prepare |0+> state
    prep = get_p0_prep()
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



def clifford_512_unwrap():

    """
    0Z   Z1
    |  Z  |
    .--2--.
    |  Z  |
    3Z   Z4
    """

    n = 5
    sp = SymplecticSpace(n)
    S, H, P = sp.S, sp.H, sp.P
    src = construct.get_512()
    #print(src.longstr())
    """
    ZZZ..
    ..ZZZ
    X.XX.
    .XX.X
    """

    # logical hadamard on the 512 surface code
    hadamard = P(3,0,2,4,1)*H(0)*H(1)*H(2)*H(3)*H(4)
    hsrc = src.apply(hadamard)
    assert hsrc.is_equiv(src)
    assert hsrc.get_logical(src) == SymplecticSpace(1).H()

    return

    M = H(0)*S(0)
    M = H(1)*S(1)*H(1) * M
    M = H(3)*S(3)*H(3) * M
    M = H(4)*S(4) * M

    tgt = QCode.fromstr("""
    XYZ..
    ..ZYX
    Y.XX.
    .XX.Y
    """)
    assert src.apply(M).is_equiv(tgt)

    cover = Cover.frombase(tgt)
    print(cover.get_expr(M.name).name)

    code = unwrap(construct.get_513())
    iso = code.get_isomorphism(cover.total)
    print(iso)



def clifford_512():
    code = construct.get_512()
    n = code.n
    c = Clifford(n)
    s = Syntax()
    css = code.to_css()
    Hx, Hz = css.Hx, css.Hz

    prep_0 = css_encoder(Hx)

    #prep_0 = css_encoder(Hz)
    #prep_0 = s.P(3,0,2,4,1).name + prep_0

    #prep_0 = css_encoder(Hz)
    #prep_0 = s.P(0,3,2,1,4).name + prep_0

    print(prep_0)

    v0 = parsevec("0"*n)
    E = c.get_expr(prep_0)
    v0 = E*v0

    P = code.get_projector()
    assert P*v0 == v0

    #print(code.longstr())
    lz = c.get_pauli("Z..Z.")
    lx = c.get_pauli("XX...")
    v1 = lx*v0
    assert lz*v0 == v0
    assert lz*v1 == -v1

    u0 = (1/r2)*(v0 + v1) # plus state
    assert lx*u0 == u0

    if 0:
        # find the |+> state
        H, CX = c.H, c.CX
        #src = H(0)*H(1)*parsevec("0"*n)
        src = parsevec("0"*n)
        gen = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
        gen += [H(i) for i in range(n)]
        name = find_state(u0, src, gen, verbose=True)
        print(name)

    H = s.H
    prep_p = (s.P(3,0,2,4,1)*H(0)*H(1)*H(2)*H(3)*H(4)).name + prep_0 # also works
    dec_p = get_inverse(prep_0) + (H(0)*H(1)*H(2)*H(3)*H(4)*s.P(1,4,2,0,3)).name 
    #prep_p = ('CX(0,1)', 'CX(1,2)', 'CX(1,3)', 'CX(3,4)', 'H(0)', 'H(1)', 'H(3)')
    #dec_p = get_inverse(prep_p)

    E = c.get_expr(prep_p)
    u0 = E*parsevec("0"*n)
    assert P*u0 == u0
    assert lx*u0 == u0
    print("plus state", prep_p)

    perm = [i+n for i in range(n)]+[i for i in range(n)]
    prep = prep_p + s.P(*perm).name + prep_0
    print(prep)

    # double
    code = code + code.apply_H()
    n = code.n

    print(strop(code.H))
    print()

    tgt = unwrap(construct.get_513())
    print(strop(tgt.H))
    return

    print(code.longstr())

    c = measure + barrier + dec_p + barrier + prep

    circuit = Circuit(n)
    qasm = circuit.run_qasm(c)
    #print(qasm)

    shots = argv.get("shots", 10000)
    samps = send([qasm], shots=shots, error_model=True)
    process(code, samps, circuit)


def build_10_2_3_HI_IH():
    """
                                                                  
      code       P                  M1
    [[10,2,3]]  --->  [[10,2,3]]  -----> [[5,1,2]]CSS+[[5,1,2]]CSS*
        |                 |                      |
        |                 |                      |
  cov_0 |           cov_1 |                cov_2 |
        v                 v                      v
        v                 v                      v
    [[5,1,3]]         [[5,1,2]]   ----->     [[5,1,2]]CSS  
                         c_512       M         c_surf                       
    """

    # this is our canonical [[10,2,3]] code
    cov_0 = Cover.frombase(construct.get_513())

    c_surf = construct.get_512() # CSS surface code
    d_surf = c_surf.get_dual()
    c_surf2 = c_surf + d_surf # CSS double cover
    cov_2 = Cover.frombase(c_surf)
    assert cov_2.total.is_equiv(c_surf2)

    # a non-CSS [[5,1,2]] code
    c_512 = QCode.fromstr("""
    YXZ..
    X.XY.
    .YX.X
    ..ZXY
    """)
    cov_1 = Cover.frombase(c_512) # unwrapped non-CSS [[5,1,2]]
    perm = cov_1.total.get_isomorphism(cov_0.total)
    assert cov_0.total.apply_perm(perm).is_equiv(cov_1.total)
    P = cov_0.total.space.get_perm(perm)
    assert cov_0.total.apply(P).is_equiv(cov_1.total)

    perm = cov_0.total.get_isomorphism(cov_1.total)
    Pi = cov_0.total.space.get_perm(perm)

    space = c_512.space
    M = transversal.get_local_clifford(c_surf, c_512)
    M.name = space.get_name(M)

    M1 = cov_1.get_expr(M.name)
    assert M1 == cov_1.lift(M) # lift has no name

    assert cov_1.total.apply(M1).is_equiv(cov_2.total)
    assert cov_0.total.apply(M1*P).is_equiv(cov_2.total)

    # construct logical H on c_surf, also works on d_surf
    space = c_surf.space
    H = space.H()
    assert H.name == ('H(0)', 'H(1)', 'H(2)', 'H(3)', 'H(4)')
    I = space.I()
    assert I.name == ()
    perm = c_surf.get_isomorphism(c_surf.get_dual())
    H = space.get_perm(perm) * H
    assert c_surf.apply(H).get_logical(c_surf, True) == SymplecticSpace(1).H()
    assert d_surf.apply(H).get_logical(d_surf, True) == SymplecticSpace(1).H()

    HI = H.direct_sum(I)
    HI.name = send_idxs(H.name, [i for i in range(space.n)], 2*space.n)

    IH = I.direct_sum(H)
    IH.name = send_idxs(H.name, [i+space.n for i in range(space.n)], 2*space.n)

    assert cov_2.total.apply(HI).is_equiv(cov_2.total)
    assert cov_2.total.apply(IH).is_equiv(cov_2.total)

    code = cov_0.total
    space = code.space
    invert = space.invert

    M1i = invert(M1)
    M1i.name = get_inverse(M1.name)
    assert M1i == space.get_expr(M1i.name)

    HI = Pi*M1i*HI*M1*P
    assert HI == space.get_expr(HI.name)

    IH = Pi*M1i*IH*M1*P
    assert IH == space.get_expr(IH.name)

    HI, IH = IH, HI # swapped
    s2 = SymplecticSpace(2)
    assert code.apply(HI).get_logical(code, True) == SymplecticSpace(2).H(0)
    assert code.apply(IH).get_logical(code, True) == SymplecticSpace(2).H(1)

    if 0:
        print(HI.name)
        c = measure + HI.name
        circuit = Circuit(space.n)
        qasm = circuit.run_qasm(c)
        print(qasm)

    return HI, IH



def build_10_2_3_HH():
    "logical HH gate "
    base = construct.get_513()
    code = unwrap(base)
    n = code.n

    lspace = SymplecticSpace(code.k)
    lHH = lspace.H(0) * lspace.H(1)

    dual = code.get_dual()
    perm = dual.get_isomorphism(code)
    dode = dual.apply_perm(perm)
    assert dode.is_equiv(code)
    assert dode.get_logical(code) == lHH
    
    n = code.n
    space = code.space
    HH = space.get_perm(perm) * reduce(mul, [space.H(i) for i in range(n)])
    dode = code.apply(HH)
    assert dode.is_equiv(code)
    assert dode.get_logical(code) == lHH

    return HH
    


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

    Hz = code.to_css().Hz
    Hz = row_reduce(Hz)

    if argv.dual:
        prep = css_encoder(Hz, True)
    else:
        prep = css_encoder(Hx)
    print(prep)

    s = Syntax()
    CX, CZ, SWAP, H, X, Z, I = s.CX, s.CZ, s.SWAP, s.H, s.X, s.Z, s.get_identity()
    g = I

    # XXX WARNING: qupy.qumba uses reversed bit order XXX
    from qupy.qumba import Space, Operator, Code, CSSCode, eq, scalar

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

    #vdump(v0)
    #vdump(P*v0)
    assert eq(P*v0, v0)

    u0 = numpy.zeros((2**n,), dtype=scalar)
    u0[0] = 4
    u0 = P*u0
    #assert eq(v0, u0)

    #vdump(v0)

    Hs = strop(code.H).split()
    for h in Hs:
        #print(h)
        h = space.make_op(h)
        #vdump(h*v0)
        assert eq(h*v0, v0)

    return

    HI, IH = build_10_2_3_HI_IH()
    HI = space.get_expr(HI.name) # FAIL
    v1 = HI*v0
    assert eq2(P*v1, v1)


def optimize_prep_0p():
    name = ('P(0,6,7,8,4,9,3,2,1,5)', 'CX(0,5)', 'CX(4,9)', 'P(5,1,2,3,4,0,6,7,8,9)',
    'P(0,6,2,3,4,5,1,7,8,9)', 'P(0,1,2,8,4,5,6,7,3,9)', 'P(0,1,2,3,9,5,6,7,8,4)',
    'CX(4,9)', 'CX(3,8)', 'CX(1,6)', 'CX(0,5)', 'P(0,3,2,1,4,5,6,7,8,9)',
    'H(0)', 'H(1)', 'H(2)', 'H(3)', 'H(4)', 'CX(0,5)', 'CX(1,6)',
    'CX(3,8)', 'CX(4,9)', 'P(0,1,2,3,9,5,6,7,8,4)', 'P(0,1,2,8,4,5,6,7,3,9)',
    'P(0,6,2,3,4,5,1,7,8,9)', 'P(5,1,2,3,4,0,6,7,8,9)', 'CX(4,9)',
    'CX(0,5)', 'P(0,8,7,6,4,9,1,2,3,5)', 'CX(0,3)', 'CX(0,6)',
    'CX(0,7)', 'CX(1,4)', 'CX(1,7)', 'CX(1,8)', 'CX(2,4)',
    'CX(2,5)', 'CX(2,6)', 'CX(3,4)', 'CX(3,5)', 'CX(3,7)',
    'CX(3,8)', 'CX(3,9)', 'H(0,)', 'H(1,)', 'H(2,)', 'H(3,)')
    cost = lambda name : (len([gate for gate in name if gate.startswith("C")]))
    print(cost(name))
    
    n = 10
    space = SymplecticSpace(n)
    CX = space.CX
    CZ = space.CZ
    SWAP = space.SWAP
    H = space.H

    E0 = space.get_expr(name)
    print(E0)

    #name = space.get_name(E0)
    #print(cost(name)) # 51 

    I = space.I()
    gates = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    gates += [CZ(i,j) for i in range(n) for j in range(i+1,n)]
    gates += [SWAP(i,j) for i in range(n) for j in range(i+1,n)]
    gates += [H(i) for i in range(n)]
    for g in gates:
        g.cost = cost(g.name)
    #    print(g.name, g.cost)
    #return

    done = False
    best = 18
    while not done:
    #for trial in range(10):
        E = E0
        weight = str(E+I).count('1')
        name = ()
        while 1:
            shuffle(gates)
            for g in gates:
                E1 = g*E
                w = str(E1+I).count('1')
                #if w < weight:
                if w < weight or w==weight and g.cost==0:
                    E = E1
                    name = g.name + name
                    weight = w
                    break
            else:
                break
        #print()
        #print(E, name, cost(name))
        print('.', end='', flush=True)
        if E==I and cost(name) < best:
            print()
            print(name)
            best = cost(name)
            print("cost =", best)

    # cost 25
    ('CX(3,1)', 'CZ(1,8)', 'CX(1,3)', 'H(1)', 'CX(1,8)',
    'CX(8,6)', 'CX(1,0)', 'CX(8,9)', 'H(2)', 'CX(2,5)', 'H(7)',
    'CX(8,4)', 'H(4)', 'CZ(2,4)', 'CZ(4,5)', 'CX(4,3)', 'CX(8,5)',
    'H(8)', 'CX(9,0)', 'CX(5,4)', 'CX(7,3)', 'CX(3,8)', 'CX(1,8)',
    'CX(7,0)', 'CZ(0,9)', 'CX(8,0)', 'CX(3,6)', 'CX(8,1)', 'CZ(8,9)', 'CX(2,6)')

    # cost 20
    ('H(7)', 'H(3)', 'H(2)', 'CX(3,9)', 'CZ(0,9)', 'P(0,1,7,3,4,5,6,2,8,9)',
    'P(0,1,3,2,4,5,6,7,8,9)', 'P(0,1,2,7,4,5,6,3,8,9)', 'CX(2,8)',
    'P(0,1,2,7,4,5,6,3,8,9)', 'CX(2,6)', 'P(0,1,7,3,4,5,6,2,8,9)',
    'CZ(3,9)', 'P(0,1,7,3,4,5,6,2,8,9)', 'CX(5,4)', 'CX(3,1)',
    'H(4)', 'P(0,1,2,3,7,5,6,4,8,9)', 'H(4)', 'CX(2,5)',
    'P(0,1,7,3,4,5,6,2,8,9)', 'CX(2,1)', 'H(4)', 'P(0,1,7,3,4,5,6,2,8,9)',
    'CX(4,5)', 'CX(3,0)', 'H(2)', 'P(0,1,2,7,4,5,6,3,8,9)',
    'CX(1,6)', 'CX(7,8)', 'P(0,1,2,7,4,5,6,3,8,9)', 'P(0,3,2,1,4,5,6,7,8,9)',
    'CX(9,0)', 'P(0,1,4,3,2,5,6,7,8,9)', 'P(0,4,2,3,1,5,6,7,8,9)',
    'CX(1,8)', 'CX(4,1)', 'CX(5,7)', 'CX(1,0)', 'CZ(1,9)',
    'CX(2,6)', 'P(0,1,2,3,7,5,6,4,8,9)')

def get_prep_0p():
    # cost 18
    name = ('H(3)', 'P(0,1,2,7,4,5,6,3,8,9)', 'H(3)', 'H(3)', 'H(3)',
    'H(7)', 'CX(5,4)', 'CX(3,1)', 'H(7)', 'P(0,1,2,7,4,5,6,3,8,9)',
    'P(0,1,7,3,4,5,6,2,8,9)', 'CZ(0,3)', 'H(7)', 'P(0,1,2,3,7,5,6,4,8,9)',
    'CX(9,8)', 'P(0,1,3,2,4,5,6,7,8,9)', 'H(2)', 'H(7)',
    'P(0,1,2,7,4,5,6,3,8,9)', 'P(0,1,2,7,4,5,6,3,8,9)', 'CX(4,5)',
    'CX(3,0)', 'H(2)', 'P(0,1,7,3,4,5,6,2,8,9)', 'CX(9,6)',
    'P(0,1,7,3,4,5,6,2,8,9)', 'P(0,1,9,3,4,5,6,7,8,2)', 'CX(9,5)',
    'CX(2,9)', 'H(2)', 'P(3,1,2,0,4,5,6,7,8,9)', 'P(2,1,0,3,4,5,6,7,8,9)',
    'CX(7,1)', 'CX(5,7)', 'CX(4,6)', 'H(1)', 'CX(9,8)', 'P(0,1,2,3,7,5,6,4,8,9)',
    'P(7,1,2,3,4,5,6,0,8,9)', 'CX(7,3)', 'P(0,1,2,7,4,5,6,3,8,9)',
    'CX(9,3)', 'P(0,1,7,3,4,5,6,2,8,9)', 'CX(9,6)', 'P(1,0,2,3,4,5,6,7,8,9)',
    'CX(2,8)', 'P(0,2,1,3,4,5,6,7,8,9)', 'P(3,1,2,0,4,5,6,7,8,9)',
    'H(3)', 'CX(3,6)')
    name = get_inverse(name)
    return name






def qasm_10_2_3():
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

    Hz = code.to_css().Hz
    Hz = row_reduce(Hz)

    prep_pp = css_encoder(Hz, True)
    prep_00 = css_encoder(Hx)

    #print("prep_00 = ", prep_00)
    #print("prep_pp = ", prep_pp)

    s = Syntax()
    CX, CZ, SWAP, H, X, Z, I = s.CX, s.CZ, s.SWAP, s.H, s.X, s.Z, s.get_identity()
    g = I

    L = """
    0123456789
    XXXXX.....
    ZZZZZ.....
    ..XX.X....
    .Z..ZZ....
    """
    X0 = X(0)*X(1)*X(2)*X(3)*X(4)
    Z0 = Z(0)*Z(1)*Z(2)*Z(3)*Z(4)
    X1 = X(2)*X(3)*X(5)
    Z1 = Z(1)*Z(4)*Z(5)
    X0, Z0, X1, Z1 = X0.name, Z0.name, X1.name, Z1.name

    Hn = tuple("H(%d)"%i for i in range(n))
    fini_pp = measure+Hn
    fini_00 = measure

    # see HH_10_2_3 above:
    HH = ('P(0,3,1,4,2,5,8,6,9,7)', 'H(0)', 'H(1)', 'H(2)',
        'H(3)', 'H(4)', 'H(5)', 'H(6)', 'H(7)', 'H(8)', 'H(9)')

    if argv.HH:
        fini_00, fini_pp = fini_pp + HH, fini_00 + HH
        code = code.get_dual()

    HI, IH = build_10_2_3_HI_IH()
    HI, IH = HI.name, IH.name

    cx = reduce(mul, [CX(i, i+5)*SWAP(i, i+5) for i in range(5)]).name
    cz = reduce(mul, [CZ(i, i+5) for i in range(5)]).name

    print("CX =", cx)
    print("CZ =", cz)

    if argv.spam_00:
        c = fini_00 + barrier + prep_00                     # 0.996   success rate
    elif argv.spam_11:
        c = fini_00 + barrier + X0+X1+prep_00               
    elif argv.state == (0,0): # X0,X1 
        c = fini_00 + barrier + cx + barrier + prep_00
    elif argv.state == (0,1):
        c = fini_00 + barrier + X0+X1 + barrier + cx + barrier + X1 + barrier + prep_00
    elif argv.state == (1,0):
        c = fini_00 + barrier + X1 + barrier + cx + barrier + X0 + barrier + prep_00
    elif argv.state == (1,1):
        c = fini_00 + barrier + X0 + barrier + cx + barrier + X0+X1 + barrier + prep_00

    elif argv.spam_pp:
        c = fini_pp + barrier + prep_pp
        code = code.get_dual() # switch code for decode below
    elif argv.spam_mm:
        c = fini_pp + barrier + Z0+Z1+prep_pp
        code = code.get_dual() 
    elif argv.state == "pp":  # Z0,Z1
        c = fini_pp + barrier + cx + barrier + prep_pp
        code = code.get_dual()
    elif argv.state == "pm": 
        c = fini_pp + barrier + Z0 + barrier + cx + barrier + Z1 + barrier + prep_pp
        code = code.get_dual()
    elif argv.state == "mp": 
        c = fini_pp + barrier + Z0+Z1 + barrier + cx + barrier + Z0 + barrier + prep_pp
        code = code.get_dual()
    elif argv.state == "mm": 
        c = fini_pp + barrier + Z1 + cx + barrier + Z0+Z1 + barrier + prep_pp
        code = code.get_dual()

    elif argv.spam_0p:
        #c = fini_00 + barrier + IH + barrier + get_prep_0p() # 0.95    success rate  ARGGHH!!
        #c = fini_00 + IH + barrier + IH + prep_00 # 0.97(0) success rate
        #c = fini_pp + barrier + HI + barrier + IH + prep_00 # 0.96(8)
        #code = code.get_dual()
        c = fini_00 + barrier + IH + barrier + IH + prep_00 # 0.974 success rate
    elif argv.state == "0p": # X0,Z1
        c = fini_00 + barrier + IH + barrier + cz + barrier + IH + prep_00 # 0.96(7)
    elif argv.state == "1p":
        c = (fini_00 + barrier + IH + barrier 
            + Z1+X0 + barrier + cz + barrier + X0 + 
            barrier + IH + prep_00) # 0.965
    elif argv.state == "0m":
        c = (fini_00 + barrier + IH + barrier 
            + Z1 + barrier + cz + barrier + Z1 + 
            barrier + IH + prep_00) 
    elif argv.state == "1m":
        c = (fini_00 + barrier + IH + barrier 
            + X0 + barrier + cz + barrier + X0+Z1 + 
            barrier + IH + prep_00) 
    else:
        assert 0

    print(c)

    circuit = Circuit(n)
    qasm = circuit.run_qasm(c)
    #print(qasm)

    if argv.batch:
        samps = load_batch()

    else:
        shots = argv.get("shots", 1000)
        samps = send([qasm], shots=shots, error_model=True)

    process(code, samps, circuit)



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

