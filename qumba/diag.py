#!/usr/bin/env python

"""
diagonal unitaries with integer entries..

"""

from random import choice, randint
from operator import mul, matmul, add
from functools import reduce
#from functools import cache
from functools import lru_cache
cache = lru_cache(maxsize=None)

import numpy

from qumba.clifford import Clifford, Matrix, K
from qumba.lin import zeros2, identity2
from qumba.action import mulclose, mulclose_names, mulclose_find
from qumba.util import cross, allperms
from qumba.argv import argv


class Op:
    def __init__(self, op):
        self.op = op
        self.N = len(op)
        self.s = str(self.op).replace(" ", "")
    def __str__(self):
        return self.s
    __repr__ = __str__
    def __hash__(self):
        return hash(self.s)
    def __eq__(self, other):
        assert isinstance(other, Op)
        assert self.N == other.N
        return numpy.all(self.op == other.op)
    def __mul__(self, other):
        op = self.op * other.op
        return Op(op)


class Diagonal:
    def __init__(self, n):
        self.n = n
        self.N = 2**n

    def get_identity(self):
        N = self.N
        op = numpy.zeros((N,), dtype=int)
        op[:] = 1
        return Op(op)

    def Z(self, idx):
        assert 0<=idx<self.n
        N = self.N
        op = numpy.zeros((N,), dtype=int)
        for (i,bits) in enumerate(numpy.ndindex((2,)*self.n)):
            op[i] = 1 - 2*bits[idx] 
        return Op(op)

    def CZ(self, idx, jdx):
        assert 0<=idx<self.n
        assert 0<=jdx<self.n
        assert idx != jdx
        N = self.N
        op = numpy.zeros((N,), dtype=int)
        for (i,bits) in enumerate(numpy.ndindex((2,)*self.n)):
            op[i] = 1 - 2*bits[idx]*bits[jdx]
        return Op(op)

    def CCZ(self, idx, jdx, kdx):
        assert 0<=idx<self.n
        assert 0<=jdx<self.n
        assert 0<=kdx<self.n
        assert len({idx,jdx,kdx})==3
        N = self.N
        op = numpy.zeros((N,), dtype=int)
        for (i,bits) in enumerate(numpy.ndindex((2,)*self.n)):
            op[i] = 1 - 2*bits[idx]*bits[jdx]*bits[kdx]
        return Op(op)
        
    def to_matrix(self, op):
        N = self.N
        op = op.op
        assert len(op) == N
        A = numpy.zeros((N, N), dtype=int)
        for i in range(N):
            A[i,i] = op[i]
        M = Matrix(K, A)
        return M

    def parse(self, desc):
        desc = desc.strip()
        assert "\n" not in desc
        if " " in desc:
            # recurse
            ops = [self.parse(desci) for desci in desc.split(" ")]
            op = reduce(mul, ops)
            return op
        #print("parse %r"%desc)
        i0 = desc.index("[")
        stem = desc[:i0]
        tail = desc[i0:]
        op = self.get_identity()
        while tail:
            i0 = tail.index("]")
            spec, tail = tail[:i0+1], tail[i0+1:]
            #print("%r^%r"%(spec, tail))
            spec = eval(spec)
            #print(stem, spec)
            op = op * getattr(self, stem)(*spec)
            #print(op)
        return op



def test_ccz():
    n = 3
    clifford = Clifford(n)
    space = Diagonal(n)

    for i in range(n):
        lhs = space.to_matrix(space.Z(i))
        assert lhs == clifford.Z(i)
        for j in range(n):
            if j==i:
                continue
            lhs = space.CZ(i,j)
            #print(lhs)
            lhs = space.to_matrix(lhs)
            rhs = clifford.CZ(i,j)
            #print(rhs)
            assert lhs==rhs
#            for k in range(j+1, n):
#                lhs = space.CCZ(i,j,k)
#                lhs = space.to_matrix(lhs)
#                rhs = clifford.CCZ(i,j,k)
#                #print(rhs)
#                assert lhs==rhs

    lhs = space.CCZ(0,1,2)
#    assert numpy.all(lhs == space.parse("CCZ[0,1,2]"))
    assert lhs == space.parse("CCZ[0,1,2]")

    space = Diagonal(5)
    lhs = space.CZ(2,3)*space.CZ(0,2)*space.CCZ(0,1,3)
#    assert numpy.all(lhs == space.parse("CZ[2,3][0,2] CCZ[0,1,3]"))
    assert lhs == space.parse("CZ[2,3][0,2] CCZ[0,1,3]")


    n = 9
    # output from CSSLO 
    desc = """

    CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8] 
    CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8] 
    CZ[1,2][3,5][3,8][4,5][4,8][5,7][5,8][6,8][7,8] CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8] 
    CZ[1,3][1,5][2,6][2,8][5,8] CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8] 
    CZ[2,3][2,6][2,8][3,5][4,5][5,7][6,8] CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8] 
    """.strip().split("\n")

    desc = """
Z[0]
Z[1]
Z[2]
Z[3]
Z[4]
Z[5]
Z[6]
Z[7]
Z[8]
CZ[0,1]
CZ[0,4][2,4][2,6][3,4][3,5][3,7][4,5][4,6][4,7][5,6][5,7][6,7]
CZ[0,5][1,5][2,6][3,8][6,8][7,8]
CZ[0,6][1,4][1,5][1,7][2,4][2,8][3,4][3,6][4,5][4,6][4,7][4,8][5,6][6,7]
CZ[0,7][1,4][1,7][2,4][2,6][3,4][3,5][3,7][4,5][4,6][4,7][5,6][5,7][5,8][6,7][6,8]
CZ[0,8][2,6][2,8]
CZ[0,2][3,5][3,8][4,5][4,8][5,7][5,8][6,8][7,8] 
CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8]
CZ[0,3][1,5][2,8][3,8][7,8] 
CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8]
CZ[1,2][3,5][3,8][4,5][4,8][5,7][5,8][6,8][7,8] CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8]
CZ[1,3][1,5][2,6][2,8][5,8] CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8]
CZ[2,3][2,6][2,8][3,5][4,5][5,7][6,8] CCZ[0,1,3][0,1,5][0,2,3][0,2,5][0,2,6][0,2,7][0,2,8][0,3,5][0,3,8][0,4,5][0,4,8][0,5,7][0,6,8][0,7,8][1,2,3][1,3,5][1,4,5][1,5,7][2,3,5][2,3,6][2,3,7][2,4,8][2,5,7][3,4,8][3,5,8][3,6,8][4,6,8][4,7,8][5,6,8][5,7,8][6,7,8]
CZ[2,5][2,6]
CZ[2,7][2,8][3,8][4,8][5,8][7,8]
    """.strip().split("\n")

    desc = [d for d in desc if d[0]!="Z"]
    desc = [d for d in desc if "CCZ" in d]

    diag = Diagonal(n)
    tgt = set()
    for i in range(n):
      for j in range(i+1,n):
        for k in range(j+1,n):
            op = diag.CCZ(i,j,k)
            tgt.add(op)
    gen = []
    for item in desc:
        item = item.split()
        for desci in item:
            if desci.startswith("CCZ"):
                print(desci)
                op = diag.parse(desci)
                gen.append(op)

    gen = set(gen)
    print("gen:", len(gen))
    return

    G = mulclose(gen, verbose=True)
    print(len(G))

    G = set(G)
    for g in tgt:
        print(int(g in G), end=" ", flush=True)
    print()


    

if __name__ == "__main__":

    from time import time
    start_time = time()

    _seed = argv.get("seed")
    if _seed is not None:
        from random import seed
        print("seed(%s)"%_seed)
        seed(_seed)

    profile = argv.profile
    fn = argv.next() or "test"

    print("%s()"%fn)

    if profile:
        import cProfile as profile
        profile.run("%s()"%fn)

    else:
        fn = eval(fn)
        fn()

    print("\nOK: finished in %.3f seconds"%(time() - start_time))
    print()

