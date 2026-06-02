#!/usr/bin/env python

"""

Analyse diagonal gates using phase polynomials

"""


import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add, mul, lshift
from random import random, randint, choice, shuffle

import z3

import numpy

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.qcode import strop, QCode, SymplecticSpace
from qumba.csscode import CSSCode
from qumba import construct 
from qumba.matrix import Matrix

#from qumba.matrix_sage import Matrix
#from qumba.clifford import Clifford, w8, w4, r2, ir2, half
#from qumba import clifford 
#from qumba.action import mulclose
#from qumba.lin import shortstr
#from qumba.distill import PauliDistill


class Op:
    def __init__(self, items={}, level=1):
        items = dict(items)
        assert 0<=level
        self.items = items
        self.level = level

    def __str__(self):
        return "Op(%s, %s)"%(self.items, self.level)

    def __eq__(self, other):
        assert self.level == other.level
        return self.items == other.items

    def __mul__(self, other):
        assert self.level == other.level
        level = self.level
        N = 2**level
        items = dict(self.items)
        for idxs,v in other.items.items():
            if idxs in items:
                v = (items[idxs] + v) % N
                if v == 0:
                    del items[idxs]
                else:
                    items[idxs] = v
            else:
                items[idxs] = v
        return Op(items, level)

    def __pow__(self, n):
        op = Op({}, self.level)
        for i in range(n):
            op = op*self
        return op

    def act(self, Hx):
        level = self.level
        N = 2**level
        items = self.items
        m, n = Hx.shape
        #for v in Hx.span():
        for u in numpy.ndindex((2,)*m):
            u = Matrix(u)
            v = u*Hx
            a = 0
            for (idxs,r) in items.items():
                m = 1
                for idx in idxs:
                    m *= v[idx]
                a += r*m
            a %= N
            yield u, a
    


class Diag:
    def __init__(self, n, level=1):
        self.n = n
        self.level = level

    def get_identity(self):
        return Op({}, self.level)

    def op(self, i, colevel=1):
        level = self.level
        assert level >= colevel
        N = 2**(level-colevel)
        items = {(i,):N}
        return Op(items, level)

    def cop(self, i, j, colevel=1):
        assert i!=j
        level = self.level
        assert level >= colevel
        if j<i:
            i, j = j, i
        N = 2**(level-colevel)
        items = {(i,j):N}
        return Op(items, level)

    def ccop(self, i, j, k, colevel=1):
        level = self.level
        assert level >= colevel
        idxs = [i, j, k]
        idxs.sort()
        i, j, k = idxs
        assert i!=j and j!=k
        N = 2**(level-colevel)
        items = {(i,j,k):N}
        return Op(items, level)

    def Z(self, i):
        return self.op(i, 1)

    def CZ(self, i, j):
        return self.cop(i, j, 1)

    def CCZ(self, i, j, k):
        return self.ccop(i, j, k, 1)

    def S(self, i):
        return self.op(i, 2)

    def CS(self, i, j):
        return self.cop(i, j, 2)

    def CCS(self, i, j, k):
        return self.ccop(i, j, k, 2)

    def T(self, i):
        return self.op(i, 3)

    def CT(self, i, j):
        return self.cop(i, j, 3)

    def CCT(self, i, j, k):
        return self.ccop(i, j, k, 3)

    def R(self, i):
        return self.op(i, 4)

    # etc...


def test():

    n = 3
    space = Diag(n, 2)
    Z = space.Z
    CZ = space.CZ
    CCZ = space.CCZ
    I = space.get_identity()

    Hx = Matrix([[1,0,0],[0,1,0],[0,0,1]])
    print(Hx)
    g = Z(2)*CZ(0,2)*CZ(1,2)*CCZ(0,1,2)
    g.act(Hx)

    #return

    code = construct.get_713()
    code = construct.get_10_2_3()

    n = code.n
    space = Diag(n, 2)
    Z = space.Z
    S = space.S
    CS = space.CS
    CZ = space.CZ
    I = space.get_identity()
    Z2 = Z(2)
    Z3 = Z(3)

    assert Z2*Z2 == I
    assert Z2*Z3 != I
    assert Z2*Z3 == Z3*Z2

    assert CZ(2,0) == CZ(0,2)
    s = str(CZ(2,0) * Z(3) * S(2))

    g = S(2)
    assert g**0 == I
    assert g != I
    assert g*g != I
    assert g*g*g == g**3
    assert g**3 != I
    assert g**4 == I

    css = code.to_css()
    Hx = Matrix(css.Hx)
    g = reduce(mul, [S(i) for i in range(n)])
    g.act(Hx)

    # ---------------------------

    code = construct.get_832()
    """
    L =
    XX..XX..
    .Z.Z....
    .X.X.X.X
    ZZ......
    XXXX....
    .Z...Z..

    Lx =
    XX..XX..
    .X.X.X.X
    XXXX....
    =
    .XX.X..X
    """

    n = code.n
    space = Diag(n, 3)
    Z = space.Z
    S = space.S
    T = space.T
    CS = space.CS
    CZ = space.CZ
    I = space.get_identity()

    print(code.longstr())
    css = code.to_css()
    Hx = Matrix(css.Hx)
    Lx = Matrix(css.Lx)
    HLx = Hx.concatenate(Lx)
    ops = [None]*n
    idxs = [1,2,4,7]
    for i in range(n):
        if i in idxs:
            ops[i] = T(i)**7
        else:
            ops[i] = T(i)
    for g in ops:
        print(g)
    g = reduce(mul, ops)
    print(g)
    g.act(HLx)



def test_unwrap():

    from qumba.unwrap import Cover

    code = construct.get_513()
    code = construct.get_412()
    cover = Cover.frombase(code)

    code = cover.total
    css = code.to_css()

    space = Diag(css.n)
    Z = space.Z
    CZ = space.CZ
    I = space.get_identity()

    op = I
    for (i,j) in cover.fibers:
        op = op*CZ(i,j)

    print(code)
    print(code.longstr())

    Hx = Matrix(css.Hx)
    Lx = Matrix(css.Lx)
    A = Lx.concatenate(Hx)
    A = Hx
    for (v,a) in op.act(A):
        print(v, v*A,a)

    # -------------------

    from qumba.symplectic import SymplecticSpace
    space = SymplecticSpace(css.n)
    CZ = space.CZ

    op = space.get_identity()
    for (i,j) in cover.fibers:
        op = op*CZ(i,j)

    dode = op*code
    print(dode)
    print(dode.longstr())
    print(dode.is_equiv(code))

    return

    # -------------------

    from qumba.clifford import Clifford

    P = code.get_projector()
    print(P.shape)
    assert P*P == P

    space = Clifford(css.n)
    CZ = space.CZ

    op = space.get_identity()
    for (i,j) in cover.fibers:
        op = op*CZ(i,j)

    print(op*P == P*op)



def main_tensor():
    H = Matrix([[1,1]])
    Ht = H.t

    m, n = H.shape
    concatenate = Matrix.concatenate
    identity = Matrix.identity
    zeros = Matrix.zeros

    I1 = identity(1)
    I2 = identity(2)

    H0 = concatenate(H@I2@I1, I2@H@I1, I2@I2@Ht)

    H1 = concatenate(
        concatenate(I1@H@I1, H@I1@I1, zeros((1, 8)), axis=1),
        concatenate(I1@I2@Ht, zeros((4,2)), H@I2@I2, axis=1),
        concatenate(zeros((4,2)), I2@I1@Ht, I2@H@I2, axis=1))

    assert (H1*H0).is_zero()

    H2 = concatenate(I1@I1@Ht, I1@H@I2, H@I1@I2, axis=1)
    assert (H2*H1).is_zero()

    Hx = H0.t
    Hz = H1
    css = CSSCode(Hx=Hx.A, Hz=Hz.A)
    css.bz_distance()
    print(css)

    dual = css.get_dual()
    code = css+css+css
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)

    return

    Hx = H1.t
    Hz = H2
    css = CSSCode(Hx=Hx.A, Hz=Hz.A)
    css.bz_distance()
    print(css)
    print(css.longstr())


def parse(n, stabs):
    import numpy
    #print(stabs)
    stabs = stabs.replace(" ", "")
    stabs = stabs.replace("\n", "")
    stabs = stabs.split(",")
    x_ops = []
    z_ops = []
    for stab in stabs:
        if "X" in stab:
            s = stab.replace("X", " ")
            ops = x_ops
        elif "Z" in stab:
            s = stab.replace("Z", " ")
            ops = z_ops
        else:
            assert 0
        idxs = s.split(" ")
        idxs = [int(i)-1 for i in idxs if len(i)]
        #print(idxs)
        op = [0]*n
        for i in idxs:
            op[i] = 1
        ops.append(op)
    Hx = numpy.array(x_ops)
    Hz = numpy.array(z_ops)
    #print(Hx)
    css = CSSCode(Hx=Hx, Hz=Hz)
    css.bz_distance()
    return css
        
    



def main_vasmer():
    # https://arxiv.org/abs/1801.04255
    # Appendix A
    n = 12
    stabs = """
    X5X6X7X8X9X10X11X12,
    X1X3X5, X2X4X7,
    Z6Z9, Z6Z10,
    Z8Z11, Z8Z12,
    Z1Z5Z6, Z2Z6Z7,
    Z4Z7Z8, Z3Z5Z8"""

    C0 = parse(n, stabs)
    print(C0)
    #print(C0.longstr())

    stabs = """
    X1X5X6X9, X2X6X7X10,
    X3X5X8X11, X4X7X8X12,
    Z1Z3Z5, Z1Z2Z6,
    Z2Z4Z7, Z3Z4Z8,
    Z5Z9Z11, Z6Z9Z10,
    Z7Z10Z12"""

    C1 = parse(n, stabs)
    print(C1)
    #print(C1.longstr())

    stabs = """
    X1X2X3X4X5X6X7X8,
    X6X9X10, X8X11X12,
    Z1Z5, Z3Z5,
    Z2Z7, Z4Z7,
    Z5Z8Z11, Z5Z6Z9,
    Z6Z7Z10, Z7Z8Z12"""

    C2 = parse(n, stabs)
    print(C2)
    #print(C2.longstr())

    code = C0+C1+C2
    code.bz_distance()

    print(code)


    n = code.n
    space = Diag(n)

    Z = space.Z
    CZ = space.CZ
    CCZ = space.CCZ
    I = space.get_identity()

    op = I
    for i in range(12):
        op = op*CCZ(i, 12+i, 24+i)

    Hx = Matrix(code.Hx)
    Hz = Matrix(code.Hz)
    print(Hx, Hx.shape)
    print(op)

    Hzt = Hz.t
    print("act:")
    for v,a in op.act(Hx):
        #if v.sum() and Hzt.solve(v) is not None:
        #    print(v, "found")
        #    break
        if a==0:
            continue
        print(v) #, Hzt.solve(v) is not None)
    print("done")






def test_vasmer():
    # https://arxiv.org/abs/1801.04255
    # Appendix A
    n = 12
    stabs = """
    X5X6X7X8X9X10X11X12,
    X1X3X5, X2X4X7,
    Z6Z9, Z6Z10,
    Z8Z11, Z8Z12,
    Z1Z5Z6, Z2Z6Z7,
    Z4Z7Z8, Z3Z5Z8"""

    C0 = parse(n, stabs)
    print(C0)
    #print(C0.longstr())

    stabs = """
    X1X5X6X9, X2X6X7X10,
    X3X5X8X11, X4X7X8X12,
    Z1Z3Z5, Z1Z2Z6,
    Z2Z4Z7, Z3Z4Z8,
    Z5Z9Z11, Z6Z9Z10,
    Z7Z10Z12"""

    C1 = parse(n, stabs)
    print(C1)
    #print(C1.longstr())

    stabs = """
    X1X2X3X4X5X6X7X8,
    X6X9X10, X8X11X12,
    Z1Z5, Z3Z5,
    Z2Z7, Z4Z7,
    Z5Z8Z11, Z5Z6Z9,
    Z6Z7Z10, Z7Z8Z12"""

    C2 = parse(n, stabs)
    print(C2)
    #print(C2.longstr())

    cube = construct.get_832()
    print(cube)

    right = C0+C1+C2
    right = right.to_qcode()
    print(right)

    Er = right.get_encoder()
    #code = QCode.from_encoder(Er, k=3)


    Er = SymplecticSpace(cube.m * n).get_identity() << Er
    print(Er.shape)
    #print(Er)

    if 0:
        code = QCode.from_encoder(Er, k=3)
        print(code)
        code = code.to_css()
        code.bz_distance()
        print(code)

    #return

    E = cube.get_encoder()

    El = reduce(lshift, [E]*n)
    print(El.shape)

    idxs = []
    for i in range(n):
      for j in range(cube.m):
        idxs.append(cube.n*i + j)

    N = cube.m*n
    for i in range(cube.k):
      for j in range(n):
        idxs.append(cube.n*j + cube.m + i)

    #print(idxs)

    assert len(set(idxs)) == len(idxs)
    assert set(idxs) == set(range(len(idxs)))

    assert len(idxs)*2 == len(El)
    assert len(idxs) == n * cube.n
    P = SymplecticSpace(n*cube.n).get_perm(idxs).t
    E = El * P * Er
    code = QCode.from_encoder(E, k=3)
    d = code.distance("z3")
    print(code, d)
    
    #print(code.longstr())

    #return

    code = code.to_css()
    code.bz_distance()
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)

    

if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    fn = eval(name)

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            fn()
        profiler.print()

    else:
        fn()


    t = time() - start_time
    print("\nOK! finished in %.3f seconds\n"%t)


