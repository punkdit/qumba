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
from qumba.circuit import parsevec, Circuit, send, get_inverse, measure, barrier, variance, vdump, load







def test_822_clifford_unwrap_encoder():
    base = QCode.fromstr("XYZI IXYZ ZIXY")
    print(base)

    tgt = unwrap(base)

    # fix the logicals:
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
    assert code.is_equiv(tgt)
    print(code)
    n = code.n

    fibers = [(i, i+base.n) for i in range(base.n)]
    print("fibers:", fibers)

    # 412 state prep for logical |0>
    prep = ('Z(0)', 'X(0)', 'H(0)', 'CX(0,3)', 'CY(1,2)', 'H(2)', 'CY(0,1)', 'H(0)', 'H(1)')

    cover = Cover(base, code, fibers)

    # unwrap 412 state prep... ?
    E = cover.get_expr(prep)
    prep = E.name
    for (i,j) in fibers:
        Hj = ("H(%d)"%(j,),)
        prep = prep + Hj
    print("prep:", prep)
    #return

    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z
    get_perm = c.get_P
    
    v0 = parsevec("0"*n)
    #E = c.get_expr(prep)
    for g in reversed(prep):
        g = c.get_expr(g)
        v0 = g*v0
    #print(v0)

    for op in """
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """.strip().split():
        print(op)
        g = c.get_pauli(op)
        v1 = g*v0
        #u0 = v0+v1
        #u1 = v0-v1
        print(g*v0 == v0)

    # we have prepared the |+0> state doh !

    P = code.get_projector()
    assert P*v0 == v0

def qupy_822():
    # from test_822_clifford_unwrap_encoder

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
    n = code.n

    prep = ('P(4,1,2,3,0,5,6,7)', 'CX(0,3)', 'CX(7,4)', 'CX(2,6)',
    'CX(1,2)', 'CX(6,5)', 'CX(2,6)', 'P(0,1,6,3,4,5,2,7)',
    'CX(1,5)', 'CX(0,1)', 'CX(5,4)', 'CX(1,5)', 'P(4,1,2,3,0,5,6,7)',
    'P(0,5,2,3,4,1,6,7)', 'H(4)', 'H(5)', 'H(6)', 'H(7)')

    from qupy.qumba import Space, Operator, Code, CSSCode, eq, scalar # uses reversed bit order XXX
    space = Space(n)
    E = space.get_expr(prep)
    print(E)

    X, Z, CX, CZ, H, SWAP = space.X, space.Z, space.CX, space.CZ, space.H, space.make_swap
    I = space.I
    assert CX(0,1)*CX(1,0)*CX(0,1) == SWAP(0,1)
    assert H(3)*H(3) == I
    assert H(3)*Z(3)*H(3) == X(3)

    #Hs = strop(code.H).replace('.', 'I')
    #_code = Code(Hs)
    #_code.check()
    #P = _code.P

    css = code.to_css()
    css = CSSCode(css.Hz, css.Hx)
    P = ((1/2)**len(css.stabs))*css.P
    assert P*P == P
    v = css.get_encoded(0)
    vdump(v)
    assert eq(P*v, v)

    v0 = numpy.zeros((2**n,), dtype=scalar)
    v0[0] = 1 # ket |00...0>
    v0 = E*v0
    u = P*v0
    vdump(v0)
    vdump(u)

    assert eq(v0, u) # WORKS



def test_822_clifford():
    base = QCode.fromstr("XYZI IXYZ ZIXY")
    print(base)

    tgt = unwrap(base)

    # fix the logicals:
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
    assert code.is_equiv(tgt)
    print(code)
    n = code.n

    # row reduced X stabilizer/logops
    S0 = parse("""
    X......X
    .X..X...
    XX...XX.
    .XX...XX
    ..XXX..X
    """)
    SX = parse("""
    X......X
    .X..X...
    ..X.X.XX
    ...X..X.
    ....XXXX
    """)

    assert shortstr(SX) == shortstr(row_reduce(S0))
    

    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    I = c.get_identity()
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z
    get_perm = c.get_P
    cx, cz = CX, CZ

    if 0:
        fibers = [(i, i+base.n) for i in range(base.n)]
        print("fibers:", fibers)
        cover = Cover(base, code, fibers)
        expr = tuple("S(0) H(0) S(0) S(1) H(1) H(2) S(2) S(3) P(0,2,1,3) X(0) X(2)".split())
        gate = cover.get_expr(expr).name
        print("gate:", gate)
        E = c.get_expr(gate)
        print(E1 == E)

    if argv.prep_00:
        # row reduced Z stabilizer/logops
        SZ = """
        01234567
        Z....Z..
        .ZZ.ZZ..
        ..ZZ.ZZ.
        ...ZZ...
        ....ZZZZ
        """

        # prepare |00> state
        g =   cx(5,0) # src,tgt
        g = g*cx(2,1)*cx(4,1)*cx(5,1)
        g = g*cx(3,2)*cx(5,2)*cx(6,2)
        g = g*cx(4,3)
        g = g*cx(5,4)*cx(6,4)*cx(7,4)
        g = g*H(5)*H(6)*H(7)

    elif argv.prep_pp:
        ("""
        01234567
        X......X
        .X..X...
        ..X.X.XX
        ...X..X.
        ....XXXX
        """)
        g =   cx(0,7)
        g = g*cx(1,4)
        g = g*cx(2,4)*cx(2,6)*cx(2,7)
        g = g*cx(3,6)
        g = g*cx(4,5)*cx(4,6)*cx(4,7)
        g = g*H(0)*H(1)*H(2)*H(3)*H(4)

    else:
        return

    print("prep:", g.name)
    #return

    # logical's
    LX0 = X(0)*X(7)
    LX1 = X(1)*X(4)
    LZ0 = Z(0)*Z(5)
    LZ1 = Z(3)*Z(4)

    gate = """
    P(0,4,2,6,1,5,3,7)
    CX(4,0)
    SWAP(1,5)
    CX(1,5)
    CX(2,6)
    SWAP(2,6)
    CX(3,7)
    """.strip().split()
    gate = tuple(reversed(gate))

    # See test.test_822
    if 0:
        # lifted 1,0,2,3
        gate = ('CX(7,3)', 'CX(2,6)', 'P(0,5,2,3,4,1,6,7)', 'CX(1,5)',
            'CX(0,4)', 'P(4,1,2,3,0,5,6,7)', 'P(1,0,2,3,5,4,6,7)')
    else:
        # lifted 0,2,1,3 is Dehn twist
        gate = ('CX(3,7)', 'P(0, 1, 6, 3, 4, 5, 2, 7)', 'CX(2,6)', 'CX(1,5)', 'P(0, 5, 2, 3, 4, 1, 6, 7)', 'CX(4,0)', 'P(0, 2, 1, 3, 4, 6, 5, 7)')
    print("gate:", gate)
    E = c.get_expr(gate)

    v0 = parsevec("00000000")
    v0 = g*v0 # |00>

    if argv.tom:
        #v0 = LX0 * v0 # |10>
        v0 = LX1 * v0 # |01>
        u0 = E*v0
        print( u0==v0 )
    
        for l in [I, LX0, LZ0]:
          for r in [I, LX1, LZ1]:
            op = l*r
            v1 = op*v0
            print(int(v1==v0), int(v1==-v0), op.name)

    # still in groundspace
    P = code.get_projector()
    assert P*E == E*P
    assert P*v0 == v0


def get_822():
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
    return code


def HH_822():
    "logical HH gate "
    code = get_822()
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

    assert HH.name == ('P(0,1,2,3,6,7,4,5)', 'H(0)', 'H(1)',
        'H(2)', 'H(3)', 'H(4)', 'H(5)', 'H(6)', 'H(7)')




def opt_822_prep():
    # from test_822_clifford:
    prep = (
        'CX(5,0)', 'CX(2,1)', 'CX(4,1)', 'CX(5,1)', 'CX(3,2)',
        'CX(5,2)', 'CX(6,2)', 'CX(4,3)', 'CX(5,4)', 'CX(6,4)', 'CX(7,4)', )
        #'H(5)', 'H(6)', 'H(7)')

    n = 8
    s = SymplecticSpace(n)
    H, CX = s.H, s.CX
    ops = [eval(name, {"CX":CX, "H":H}) for name in prep]
    tgt = reduce(mul, ops)
    print(tgt)

    gen = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    print(len(gen))

    metric = lambda g : str(g+tgt).count('1')

    best_count = 20
    while 1:
        g = s.get_identity()
        d = metric(g)
        while 1:
            done = True
            best = None
            best_d = d
            hgs = [h*g for h in gen]
            ds = [metric(hg) for hg in hgs]
            d0 = min(ds)
            if d0 >= d:
                break
            hgs = [hg for hg in hgs if metric(hg)==d0]
            g = choice(hgs)
            d = d0
    
        if d!=0:
            continue

        count = len(g.name)
        if count < best_count:
            print(g.name, count)
            best_count = count



def run_822_qasm():
    base = QCode.fromstr("XYZI IXYZ ZIXY")
    #print(base)

    tgt = unwrap(base)

    # fix the logicals:
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
    assert code.is_equiv(tgt)
    #print(code)

    HLx = parse("""
    XX...XX.
    .XX...XX
    ..XXX..X
    X......X
    .X..X...
    """)

    HLz = parse("""
    .ZZ.ZZ..
    ..ZZ.ZZ.
    Z..Z..ZZ
    Z....Z..
    ...ZZ...
    """)

    fibers = [(i, i+base.n) for i in range(base.n)]
    #print("fibers:", fibers)

    # from test_822_clifford:
    # prep |00> state
    prep_00 = ('CX(5,0)', 'CX(2,1)', 'CX(4,1)', 'CX(5,1)', 'CX(3,2)',
        'CX(5,2)', 'CX(6,2)', 'CX(4,3)', 'CX(5,4)', 'CX(6,4)',
        'CX(7,4)', 'H(5)', 'H(6)', 'H(7)')

    if argv.opt:
        # from opt_822_prep, not as good 
        prep_00 = ('CNOT(5,3)', 'CNOT(5,4)', 'CNOT(4,1)', 'CNOT(2,1)', 'CNOT(6,4)', 'CNOT(6,3)', 'CNOT(3,2)', 'CNOT(4,3)', 'CNOT(7,4)', 'CNOT(5,0)')
        prep_00 += ('H(5)', 'H(6)', 'H(7)')

    # from test_822_clifford
    # prepare |++> state
    prep_pp = ('CX(0,7)', 'CX(1,4)', 'CX(2,4)', 'CX(2,6)', 'CX(2,7)', 'CX(3,6)', 'CX(4,5)', 'CX(4,6)', 'CX(4,7)', 
        'H(0)', 'H(1)', 'H(2)', 'H(3)', 'H(4)')

    # lifted 0,2,1,3 from test_822_clifford is Dehn twist
    gate = ('CX(3,7)', 'P(0,1,6,3,4,5,2,7)', 'CX(2,6)', 'CX(1,5)',
        'P(0,5,2,3,4,1,6,7)', 'CX(4,0)','P(0,2,1,3,4,6,5,7)')

    """
    01234567
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """
    I = ()
    X0 = ("X(0)", "X(7)")
    X1 = ("X(1)", "X(4)")
    Z0 = ("Z(0)", "Z(5)")
    Z1 = ("Z(3)", "Z(4)")

    n = code.n
    circuit = Circuit(n)

    #c = measure + logical + decode + physical + barrier + prep_00
    #c = measure + X1 + barrier + (gate + barrier) + barrier + X1 + prep_00
    #c = measure + Z0 + barrier + (gate + barrier) + barrier + Z0 + prep_00

    # logical HH, see HH_822 above:
    HH = ('P(0,1,2,3,6,7,4,5)', 'H(0)', 'H(1)',
        'H(2)', 'H(3)', 'H(4)', 'H(5)', 'H(6)', 'H(7)')

    h8 = tuple("H(%d)"%i for i in range(n))

    fini_00 = measure
    fini_pp = measure + h8 

    if argv.HH:
        fini_00, fini_pp = fini_pp + HH, fini_00 + HH
        HLx, HLz = HLz, HLx

    if argv.spam:
        if argv.prep_00:
            c = fini_00 + barrier + prep_00 # SPAM
        elif argv.prep_pp:
            c = fini_pp + barrier + prep_pp # SPAM
        else:
            return
    elif argv.state == (0,0):
        c = fini_00 + barrier + gate + barrier + prep_00
    elif argv.state == (1,0):
        c = fini_00 + X0 + barrier + gate + barrier + X0 + prep_00
    elif argv.state == (0,1):
        c = fini_00 + X0+X1 + barrier + gate + barrier + X1 + prep_00
    elif argv.state == (1,1):
        c = fini_00 + X1 + barrier + gate + barrier + X0+X1 + prep_00
    elif argv.state == "pp":
        c = fini_pp + barrier + gate + barrier + prep_pp
    elif argv.state == "mp":
        c = fini_pp + Z0+Z1 + barrier + gate + barrier + Z0 + prep_pp
    elif argv.state == "pm":
        c = fini_pp + Z1 + barrier + gate + barrier + Z1 + prep_pp
    elif argv.state == "mm":
        c = fini_pp + Z0 + barrier + gate + barrier + Z0 + Z1 + prep_pp
    else:
        return

    print(c)

    qasms = []
    qasms.append(circuit.run_qasm(c))

    if argv.load:
        samps = load()

    else:
        shots = argv.get("shots", 1000)
        samps = send(qasms, shots=shots, error_model=True)
        #print(samps)

    idxs = circuit.labels # final qubit permutation

    if type(argv.state) is str or argv.prep_pp:
        #print("measure X syndromes")
        H = HLx
    else:
        #print("measure Z syndromes")
        assert type(argv.state) is tuple or argv.prep_00
        H = HLz

    H = H[:, idxs] # shuffle

    #print(H)
    succ = 0
    fail = 0
    for i,v in enumerate(samps):
        v = parse(v)
        u = dot2(H, v.transpose())
        if u.sum() == 0:
            succ += 1
        elif u[:3].sum() == 0:
            fail += 1
        if argv.show:
            print(shortstr(u.transpose()), end=" ")
            if (i+1)%20==0:
                print()
    #print()
    #get_syndrome(samps)

    print("samps:", len(samps))
    if not samps:
        return

    print("succ: ", succ)
    print("err:  ", len(samps)-succ-fail)
    print("fail: ", fail)
    shots = fail+succ
    if shots:
        p = (1 - fail / (fail+succ))
        print("p   = %.6f" % p)
        print("var = %.6f" % variance(p, shots))


def get_syndrome(S):
    from pecos import BinArray #This is just a cute function is PECOS that reverses results so you look at them in a more human readable direction
    
    #Simple processing code
    #Calculate logical operator, calculate syndromes
    #Post-select on non-trivial syndromes
    #Calculate logical fidelity
    suc1 = 0
    suc2 = 0
    suc3 = 0
    suc_count = 0
    for j in range(len(S)):
        raw_out = numpy.array([int(i) for i in BinArray(S[j])])
        S1 = raw_out[1]^raw_out[2]^raw_out[4]^raw_out[5]
        S2 = raw_out[2]^raw_out[3]^raw_out[5]^raw_out[6]
        S3 = raw_out[0]^raw_out[3]^raw_out[7]^raw_out[6]
        S4 = raw_out[0]^raw_out[1]^raw_out[7]^raw_out[4]
    
    
        log_1 = raw_out[3]^raw_out[4]
        log_2 = raw_out[0]^raw_out[5]
        log_3 = raw_out[2]^raw_out[7]
        if S1 == 0 & S2 == 0 & S3 == 0 & S4==0:
            #print('succes!')
            suc_count +=1
            if log_1 == 0:
                suc1 += 1
    
            if log_2 == 0:
                suc2 += 1
            if log_3 == 0:
                suc3 += 1
    
    print(suc1/suc_count)
    print(suc2/suc_count)
    print(suc3/suc_count)
    


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

