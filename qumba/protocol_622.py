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
#from qumba.circuit import parsevec, Circuit, send, get_inverse, measure, barrier, variance, vdump, load


def test_ec():
    code = QCode.fromstr("XXI IXX ZZZ")
    #code = QCode.fromstr("XXXX ZZZZ")
    #code = construct.get_513()

    n = code.n
    P = code.get_projector()
    assert P*P == P
    assert P.rank() == 2**code.k

    c = Clifford(n)
    N = 2**n

    for k in range(N):
        if P[0,k] != 0:
            break
    else:
        assert 0

    u = P[0,k]

    errors = [c.X(i) for i in range(n)]
    errors += [c.Z(i) for i in range(n)]
    errors += [c.S(i) for i in range(n)]
    errors += [c.S(i)*c.H(i) for i in range(n)]

    for E in errors:
        lhs = P*E*P
        v = (lhs[0,k]/u) 
        rhs = v * P
        assert lhs==rhs
        print(v, end=" ", flush=True)

    return

    for Ei in errors:
      for Ej in errors:
        if Ei==Ej:
            continue
        lhs = P*Ei.d * Ej *P
        v = (lhs[0,k]/u) 
        rhs = v * P
        assert lhs==rhs
        print(v, end=" ", flush=True)



def test():

    code = QCode.fromstr("XXXXII IIXXXX ZZZZII IIZZZZ")
    print(code.longstr())

#    E = code.get_encoder()
#    print(E)
#
#    H = (E.t)[:8:2, :]
#    code = QCode(H)
#    print(code)
#    print(code.longstr())

    P = code.get_projector()
    assert P*P == P
    assert P.rank() == 2**code.k

    c6 = Clifford(6)

    stabs = "XXXXII IIXXXX ZZZZII IIZZZZ".split()
    for e in stabs:
        stab = c6.get_pauli(e)
        assert stab*P == P

    logops = "XX....  Z.Z.Z.  ZZ....  .XX.X.".split()
    for e in logops:
        op = c6.get_pauli(e)
        assert op*P == P*op

    I = red(1,1)
    _r = red(0,1,2) # TODO: add phase here & Pauli correction below
    _g = green(0,1) # TODO: add phase here & Pauli correction below

    op = reduce(matmul, [_r,_g,I,I,I,I])
    Q = op*P
    print(Q.rank())
    print(Q.shape)

    c4 = Clifford(4)
    for e in "XXXX ZZZZ".split():
        stab = c4.get_pauli(e)
        print(stab*Q == Q)

    lop = c4.get_pauli("XXII")
    rop = c6.get_pauli("IIIIXX")
    print(lop*Q == Q*rop)

    lop = c4.get_pauli("XIXI")
    rop = c6.get_pauli("IXXIXI")
    print(lop*Q == Q*rop)

    lop = c4.get_pauli("ZZII")
    rop = c6.get_pauli("ZZIIII")
    print(lop*Q == Q*rop)

    lop = c4.get_pauli("ZIZI")
    rop = c6.get_pauli("ZIZIZI")
    print(lop*Q == Q*rop, lop*Q == -Q*rop)

    Q = Q*Q.d
    assert Q*Q == Q
    stab = c4.get_pauli("XXXX")
    print(stab*Q == Q)
    stab = c4.get_pauli("ZZZZ")
    print(stab*Q == Q)

    return

    for s in cross(["IZX"]*4):
        lop = c4.get_pauli(s)
        if(-lop*Q == Q*rop):
            print("\t", s)

    return

    for s in cross(["IZ"]*6):
        s = ''.join(s)
        rop = c6.get_pauli(s)
        if(lop*Q == Q*rop):
            print("\t", s)

    #print(QCode.fromstr("XXXX ZZZZ").longstr())







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

