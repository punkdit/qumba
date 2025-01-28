#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
import pickle

import numpy

from qumba import lin
lin.int_scalar = numpy.int32 # qupy.lin
from qumba.lin import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba.autos import get_autos
from qumba import csscode, construct
from qumba.construct import get_422, get_513, get_golay, get_10_2_3, reed_muller
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
from qumba.circuit import parsevec, Circuit, send, get_inverse, measure, barrier, variance, load


def get_protocol():
    protocol = {}
    # logical S gate
    physical = (
        'S(0)', 'H(0)', 'S(0)', 
        'S(1)', 'H(1)', 
        'H(2)', 'S(2)', 
        'S(3)', 
        'P(0, 2, 1, 3)', 'X(0)', 'X(2)')
    logical = ("S(3).d",) # inverse logical 
    #protocol.append((physical, logical))
    protocol["S(0)"] = (physical, logical)

    # logical X
    physical = ('Z(0)', 'X(1)',)
    logical = ("X(3)",) # inverse logical 
    protocol["X(0)"] = (physical, logical)
    #protocol.append((physical, logical))

    # logical Z
    physical = ('Z(1)', 'X(2)',)
    logical = ("Z(3)",) # inverse logical 
    protocol["Z(0)"] = (physical, logical)
    #protocol.append((physical, logical))

    # logical H gate
    #physical = ("Z(1)", "X(2)", "P(3,0,1,2)") # XXX the paper uses P(0321)
    #logical = ("H(3)" ) # inverse logical 
    #physical = ("H(0)", "H(1)", "H(2)", "H(3)", "X(0)", "X(2)", "P(0,3,2,1)")
    #logical = ("H(3)", "Z(3)", ) # inverse logical 
    physical = ("Z(1)", "X(2)", "H(0)", "H(1)", "H(2)", "H(3)", "X(0)", "X(2)", "P(0,3,2,1)")
    logical = ("H(3)",) # inverse logical 
    protocol["H(0)"] = (physical, logical)
    #protocol.append((physical, logical))
    return protocol


def magic_412():
    "Can we do magic state prep on the [[4,1,2]] ?"

    def get_eig(u, g):
        "return eigenval if u is eigvec of g, None otherwise"
        r = (u.d*u)[0][0]
        val = (u.d * g * u)[0][0]
        val = val / r
        if g*u == val*u:
            return val
        return None

    c = Clifford(1)
    H, S, Z, X, Y = c.H(), c.S(), c.Z(), c.X(), c.Y()
    pauli = mulclose([X, Z, Y])
    assert len(pauli) == 16

    g = S*H*X
    h = H*S
    assert g*h == h*g
    #for val, vec, dim in g.eigenvectors():
    #    print(vec)
    #    print(val, ",", get_eig(vec, h))

    for val, vec, dim in S.eigenvectors():
        print(vec, val)

    
    return

    #ops = [g, h, S*H*S*X, S]
    ops = [Z*H, H]
    for g in ops:
      for h in ops:
        print(g*h==h*g, end=' ')
      print()
    return

    if 0:
        # the ZH magic state is not magic, it's a Y state
        g = Z*H
        evs = g.eigenvectors()
        for val, vec, dim in evs:
            print(val)
        #assert Z*H*vec==vec
        print(vec)
        vec.d * g*vec
        print(get_eig(vec, g))
        for g in pauli:
            if g*vec == vec:
                print("found +1")
                print(g.name)
                print(g)
        return

    cliff = mulclose([H, S])
    assert len(cliff) == 192
    #for g in cliff:
    #    if g.order() == 4:
    #        print(g.name)
    #return

    #evs = H.eigenvectors() # this gives a ZH state, which is not magic
    #evs = S.eigenvectors() # this gives a ZH state, which is not magic
    #evs = (S*H*S).eigenvectors()
    #evs = (S*H).eigenvectors() # needs argv.degree=24
    vs = []
    for g in cliff:
        evs = g.eigenvectors()
        for val, vec, dim in evs:
            if dim != 1:
                continue
            vs.append(vec)
    print("vs:", len(vs))
    #assert H*vec==vec
    #assert S*vec==vec
    #v = vec@vec@vec@vec
    #vs = [vec@vec@vec@vec for vec in vs]
    SS = S*S
    SSS = SS*S
    vs = [vec@(S*vec)@(SS*vec)@(SSS*vec) for vec in vs]
    
    code = construct.get_412()
    n = code.n
    c = Clifford(n)

    P = code.get_projector()
    #u = P*v
    #print(u)
    us = [P*v for v in vs]

    protocol = get_protocol()
    lh = c.get_expr(protocol["H(0)"][0])
    lh.name = ("H(0)",)
    assert P*lh == lh*P
    ls = c.get_expr(protocol["S(0)"][0])
    ls.name = ("S(0)",)
    assert P*ls == ls*P
    lx = c.get_expr(protocol["X(0)"][0])
    assert P*lx == lx*P
    lz = c.get_expr(protocol["Z(0)"][0])
    assert P*lz == lz*P

    #print(u.d * lh * u) == 0
    
    lhs = -w4*lz*P
    rhs = ls*ls*P
    assert lhs == rhs

    if 0:
        g = lz*lh
        assert g*u == w4*u
        val = get_eig(u, g)
        assert val == w4
    
        g = ls*ls*lh
        assert g*u == u

    #return

    G = mulclose([lh, ls])
    assert lx in G
    #G = [lh, ls, lh*ls, ls*lh, lh*ls*lh, ls*lh*ls]
    print("|G| =", len(G))

    for u in us:
        for idx,g in enumerate(G):
            val = get_eig(u, g)
            if val is None:
                continue
            #if val is not None:
                #print(idx, end=" ")
            #if val==1:
            #    continue
            lg = Clifford(1).get_expr(g.name)
            print(g.name)
            print(lg, val)
            if g.name == ('S(0)', 'S(0)', 'H(0)'):
                #print(".", end='', flush=True)
                break
        else:
            print("found!\n")
            break

    print("done")



def test_412():
    code = construct.get_412()
    n = code.n
    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z
    get_perm = c.get_P
    #E = get_encoder(code)
    EE = code.get_clifford_encoder()
    P = code.get_projector()

    E2 = CZ(2,0)*CY(2,3)*H(2)
    E1 = CY(1,2)*CZ(1,3)*H(1)
    E0 = CY(0,1)*CZ(0,2)*H(0)
    E = E0*E1*E2 # WORKS

    assert  EE == E

    # pick the correct logical basis
    E = E*S(3)*H(3)*S(3).d

    print(E.name)

    r0 = red(1,0,0)
    r1 = red(1,0,2)
    v0 = r0@r0@r0@r0
    v1 = r0@r0@r0@r1
    u0 = E*v0
    u1 = E*v1
    assert P*u0 == u0
    assert P*u1 == u1
    u = (r2*u1)
    
    # look for logical zero and logical one

    X, Y, Z = c.X, c.Y, c.Z
    Lx = Z(0) * X(1)
    Lz = Z(1) * X(2)
    Ly = w4 * Lx * Lz
    assert Lx * Lz == -Lz * Lx
    assert Lx * Ly == -Ly * Lx
    assert Lz * Ly == -Ly * Lz
    assert Lx*P == P*Lx
    assert Lz*P == P*Lz
    assert Ly*P == P*Ly

    dot = lambda l,r : (l.d*r)[0][0]

    # Use codespace projector to generate vectors in the codespace:

    states = []

    # logical |+>,|->
    v0 = 2*P*parsevec("0000 + -1*0001") # +1 eigenvector of Lx
    v1 = Lz*v0 # -1 eigenvector of Lx
    assert Lx*v0 == v0
    assert Lx*v1 == -v1
    assert dot(v0,v1) == 0
    states.append([v0,v1])

    # logical |+i>,|-i>
    v0 = 2*P*parsevec("0000") # +1 eigenvector of Ly 
    v1 = Lx*v0 # -1 eigenvector of Ly
    assert Ly*v0 == v0
    assert Ly*v1 == -v1
    assert dot(v0,v1) == 0
    states.append([v0,v1])

    # logical |0>,|1>
    v0 = 2*P*parsevec("0000 + i*0001") # +1 eigenvector of Lz
    v1 = Lx*v0 # -1 eigenvector of Lz
    assert Lz*v0 == v0
    assert Lz*v1 == -v1
    assert dot(v0,v1)==0
    assert v0 == u0
    states.append([v0,v1])

    if argv.find_state:
        v0 = states[argv.i][argv.j]
        # search for logical |0> state prep 
        gen = [op(i) for i in range(n) for op in [X,Y,Z,S,H]]
        gen += [CZ(i,j) for i in range(n) for j in range(i)]
        gen += [op(i,j) for i in range(n) for j in range(n) for op in [CX,CY] if i!=j]
        name = find_state(v0, parsevec("0000"), gen, maxsize = 100000, verbose = True)
        print(name)

    # logical |0> state prep 
    v0, v1 = states[2] # |0>,|1>
    prep = ('Z(0)', 'X(0)', 'H(0)', 'CX(0,3)', 'CY(1,2)', 'H(2)', 'CY(0,1)', 'H(0)', 'H(1)')
    U = c.get_expr(prep)
    u = U*parsevec("0000")
    assert u == v0

    #M = (half**4)*clifford.Matrix(K,[[dot(u0,v0),dot(u0,v1)],[dot(u1,v0),dot(u1,v1)]])
    #print(M)

    if 0:
        # logical H
        L = get_perm(1,2,3,0)
        assert L*Lx*L.d == Lz # H action
    elif 1:
        L = get_perm(0,3,2,1)
        L = H(0)*H(1)*H(2)*H(3)*X(0)*X(2)*L
        lz = L*Lx*L.d
        lx = L*Lz*L.d
        assert lz*Lx == -Lx*lz
        assert lz*Lz == Lz*lz
        assert lx*Lx == Lx*lx
        assert lx*Lz == -Lz*lx
        #assert L*Lx*L.d == Lz # H action
    elif 0:
        # logical X*Z
        L = get_perm(1,0,3,2)*H(0)*H(1)*H(2)*H(3)*X(0)*X(2)
    else:
        # logical S
        L = SHS(0)*SH(1)*HS(2)*S(3)
        L = L*get_perm(0,2,1,3)
        L = L*X(0)*X(2)
        print(L.name)

    c1 = Clifford(1)
    I,X,Y,Z,S,H = c1.get_identity(), c1.X(), c1.Y(), c1.Z(), c1.S(), c1.H()
    phases = []
    for i in range(8):
        phase = (w8**i)*I
        phase.name = ("(1^{%d/8})"%i,)
        phases.append(phase)
    gen = phases + [X, Z, Y, S, S.d, H]
    #g = mulclose_find(gen, M)
    #print(g.name)
    #return

    G = mulclose([S, H])
    assert len(G) == 192

    def getlogop(L):
        basis = [v0, v1]
        M = []
        for l in basis:
          row = []
          for r in basis:
            u = l.d * L * r
            row.append(u[0][0])
          M.append(row)
        return clifford.Matrix(K, M)
    
    #print("Lx =")
    lx = (half**4)*getlogop(Lx)
    assert lx == X
    #print(lx)
    #print("Lz =")
    #print(getlogop(Lz))

    assert P*L==L*P

    # this is the encoded logical
    l = (half**4)*getlogop(L)
    print("l =")
    print(l)
    #print("l^2 =")
    #gen = [(w8**i)*I for i in range(8)] + [X, Z, Y, S, S.d, H]
    # now we find a name for the encoded logical
    g = mulclose_find(gen, l)
    print("g =")
    print(g, "name =", g.name)

    return

    assert g==l
    assert g**2 == Y

    assert g*X == Z*g
    assert g*Y == Y*g

    assert (S*H)**3 == w8*I
    assert S*H*S.d*H*S.d*H == (w8**7)*Z
    assert g == (w8**7)*Z*H
    print(H)
    print(Z*H)
    print((w8**7)*Z*H)
    return

    G = mulclose([X, Z, Y, S, S.d, H])
    print("|Cliff(1)|=", len(G))
    for g in G:
        if g*X==X*g and g*Z==Z*g and g*Y==Y*g:
            print(g.name)
    
    return

    c1 = Clifford(1)
    H, S = c1.H(), c1.S()
    print(H)
    print(H*H)

def gen_412():
    c = Clifford(1)
    gen = [c.X(), c.Z(), c.S(), c.H()]
    G = mulclose(gen)
    assert len(G) == 192
    #print(len(G))
    names = [g.name for g in G]
    #for g in G:
    #    print(g.name)
    return names


def run_412_qasm():
    circuit = Circuit(4)
    encode = ('CY(0,1)', 'CZ(0,2)', 'H(0)', 'CY(1,2)', 'CZ(1,3)',
        'H(1)', 'CZ(2,0)', 'CY(2,3)', 'H(2)', 'S(3)', 'H(3)',
        'S(3).d')
    decode = get_inverse(encode)

    # state prep for logical |0>
    prep = ('Z(0)', 'X(0)', 'H(0)', 'CX(0,3)', 'CY(1,2)', 'H(2)', 'CY(0,1)', 'H(0)', 'H(1)')

    protocol = {}
    #physical = ("P(1,2,3,0)",)
    #logical = ("Z(3)", "H(3)") # inverse logical
    #protocol.append((physical, logical))

    # logical S gate
    physical = (
        'S(0)', 'H(0)', 'S(0)', 
        'S(1)', 'H(1)', 
        'H(2)', 'S(2)', 
        'S(3)', 
        'P(0, 2, 1, 3)', 'X(0)', 'X(2)')
    logical = ("S(3).d",) # inverse logical 
    #protocol.append((physical, logical))
    protocol["S(0)"] = (physical, logical)

    # logical X
    physical = ('Z(0)', 'X(1)',)
    logical = ("X(3)",) # inverse logical 
    protocol["X(0)"] = (physical, logical)
    #protocol.append((physical, logical))

    # logical Z
    physical = ('Z(1)', 'X(2)',)
    logical = ("Z(3)",) # inverse logical 
    protocol["Z(0)"] = (physical, logical)
    #protocol.append((physical, logical))

    # logical H gate
    #physical = ("Z(1)", "X(2)", "P(3,0,1,2)") # XXX the paper uses P(0321)
    #logical = ("H(3)" ) # inverse logical 
    #physical = ("H(0)", "H(1)", "H(2)", "H(3)", "X(0)", "X(2)", "P(0,3,2,1)")
    #logical = ("H(3)", "Z(3)", ) # inverse logical 
    physical = ("Z(1)", "X(2)", "H(0)", "H(1)", "H(2)", "H(3)", "X(0)", "X(2)", "P(0,3,2,1)")
    logical = ("H(3)",) # inverse logical 
    protocol["H(0)"] = (physical, logical)
    #protocol.append((physical, logical))
    del physical, logical

    names = gen_412()
    #names = [nam for nam in names if len(nam)==1]

    N = argv.get("N", 4) # circuit depth
    trials = argv.get("trials", 4)
    print("circuit depth:", N)
    print("trials:", trials)

    if argv.nobarrier:
        global barrier
        barrier = ()

    lx = protocol["X(0)"][1] + ("COMMENT('X(0)')",)

    qasms = []
    for trial in range(trials):
        physical = ()
        logical = ()
        if argv.lx:
            logical = lx
        #print("protocol:")
        for i in range(N):
            name = choice(names)
            p, l = (), ()
            for nami in name:
                p = p + protocol[nami][0]
                l = protocol[nami][1] + l
            #print("name:", name)
            #print("\tl:", l)
            #print(p)
            
            physical = barrier + p + physical
            logical = logical + l
    
        # left <---<---< right 
        if argv.encode:
            #print("encode")
            c = measure + logical + decode + physical + barrier + encode
    
        else:
            #print("prep")
            c = measure + logical + decode + physical + barrier + prep
    
        qasm = circuit.run_qasm(c)
        #print(qasm[-60:])
        if argv.show:
            print(qasm)
        qasms.append(qasm)
    #return

    if argv.dump:
        for qasm in qasms:
            print("\n// qasm job")
            print(qasm)
            print("// end qasm\n")

    else:

        kw = {}
        #if not argv.get("leakage", False):
        #    kw['p1_emission_ratio'] = 0
        #    kw['p2_emission_ratio'] = 0
    
        shots = argv.get("shots", 100)
        samps = send(qasms, shots=shots, N=N, 
            simulator="state-vector", 
            memory_errors=argv.get("memory_errors", False),
            leak2depolar = argv.get("leak2depolar", False),
            error_model=argv.get("error_model", True),
            reorder=True, # uses fix_qubit_order
            **kw,
        )
        #print(samps)
        process_412(samps) #, circuit.labels)

def process_412(samps):
    print("samps:", len(samps))
    if not samps:
        return
    succ=samps.count('0000')
    fail=samps.count('0001')
    #print("labels:", labels)
    #for c in "0001 0010 0100 1000".split():
    for c in "1000 0100 0010 0001".split():
        print(samps.count(c), end=' ')
    print()
    print("succ: ", succ)
    print("err:  ", len(samps)-succ-fail)
    print("fail: ", fail)
    n = fail+succ
    if n:
        p = (1 - fail / n)
        print("p   = %.6f" % p) 
        print("var = %.6f" % variance(p, n))
    else:
        print("p = %.6f" % 1.)


def load_412():
    samps = load(reorder=True)
    process_412(samps)




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

