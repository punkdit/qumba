#!/usr/bin/env python

"""

Analyse diagonal gates using phase polynomials

"""


import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add, mul
from random import random, randint, choice, shuffle

import z3

import numpy

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.qcode import strop, QCode
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
        for v in Hx.span():
            a = 0
            for (idxs,r) in items.items():
                m = 1
                for idx in idxs:
                    m *= v[idx]
                a += r*m
            a %= N
            print(v, a)
    


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


