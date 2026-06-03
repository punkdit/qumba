#!/usr/bin/env python

"""
Here we implement 
_phased lagrangian relations (see lagrel.py etc.)
A kind of stabilizer tableaux.

FAIL !


"""

from random import choice, randint
from operator import mul, matmul, add
from functools import reduce
from functools import cache

import numpy

from qumba.argv import argv
from qumba.action import mulclose
from qumba.matrix import Matrix, pullback
from qumba.symplectic import symplectic_form
from qumba.smap import SMap
from qumba.qcode import strop


zeros = lambda a,b : Matrix.zeros((a,b))


def normalize(left, right, phase):
    #print("normalize")
    assert left.shape[0] == right.shape[0]
    m, n = left.shape[1], right.shape[1]
    A = left.concatenate(right, axis=1)
    A = A.normal_form()
    left = A[:, :m]
    right = A[:, m:]
    return left, right, phase




class Tab:
    def __init__(self, left, right=None, phase=None):
        left = Matrix.promote(left)
        if right is None:
            right = Matrix.identity(left.shape[0])
        else:
            right = Matrix.promote(right)
        assert left.shape[0] == right.shape[0], \
            "%s %s"%(left.shape, right.shape)
        if phase is None:
            phase = Matrix.zeros((len(left),)) # yes..?
        left, right, phase = normalize(left, right, phase)
        rank = left.shape[0]
        self.left = left
        self.right = right
        self.tgt = left.shape[1]
        self.src = right.shape[1]
        self.rank = rank
        self.A = left.concatenate(right, axis=1)
        #B = self._left.concatenate(self._right, axis=1)
        #AB = self.A.intersect(B)
        #assert len(AB) == rank # yes
        self.shape = (rank, self.tgt, self.src)
        self.phase = phase
        self.check()

    def __repr__(self):
        left, right, phase = self.left, self.right, self.phase
        left = str(left.A).replace("\n", "")
        right = str(right.A).replace("\n", "")
        phase = str(phase.A).replace("\n", "")
        return "Tab(%s, %s, %s)"%(left, right, phase)

    def __str__(self):
        left, right, phase = self.left, self.right, self.phase
        A = self.A
        smap = SMap()
        c = 1
        smap[0,c] = strop(left)
        w = left.shape[1] // 2
        for i in range(left.shape[0]):
            smap[i,c+w] = "|"
            s = strop(A[i])
            pi = (phase[i] + s.count("Y")) % 4
            assert pi in [0,2]
            smap[i,0] = "+-"[pi//2]
        smap[0,c+w+1] = strop(right)
        return str(smap)

    @classmethod
    def get_identity(cls, n):
        I = Matrix.identity(2*n)
        return cls(I, I)

    def _check(self):
        assert self.is_lagrangian()
        A = self.A
        phase = self.phase
        for i in range(A.shape[0]):
            s = strop(A[i])
            pi = (phase[i] + s.count("Y")) % 4
            assert pi in [0,2], pi

    def check(self):
        try:
            self._check()
        except:
            print("!"*79)
            print("Tab.check: FAIL")
            print(repr(self))
            raise

    def is_lagrangian(self):
        A = self.A
        m, nn = A.shape
        assert nn%2 == 0
        assert nn//2 == m, A.shape
        F = symplectic_form(m)
        # assert isotropic
        At = A.transpose()
        AFA = A * F * At
        return AFA.sum() == 0




def test():

    I = Tab.get_identity(1)
    H = Tab([[0,1],[1,0]])
    S = Tab([[1,0],[1,1]], None, [0,3]) 

    from qumba import pauli
    X, Z = pauli.X, pauli.Z
    XZ = X*Z
    #print(X*Z, Z*X)
    print(XZ*XZ)



def test_dense():

    from qumba.clifford import Clifford

    c2 = Clifford(2)
    II = c2.I
    XI = c2.X(0)
    IX = c2.X(1)
    ZI = c2.Z(0)
    IZ = c2.Z(1)
    wI = c2.wI()

    Pauli = mulclose([wI*wI, XI, IX, ZI, IZ])
    assert len(Pauli) == 64, len(Pauli)

    assert c2 is Clifford(2)

    SI = c2.S(0)
    IS = c2.S(1)
    HI = c2.H(0)
    IH = c2.H(1)
    CZ = c2.CZ(0, 1)

    C2 = mulclose([SI, IS, HI, IH, CZ], maxsize=None, verbose=True) # slow
    #assert len(C2) == 92160, len(C2)
    print()

    # The action of the clifford group by conjugation
    # on the Pauli group.
    # Find the Clifford stabilizer of a (Pauli) stabilizer state.

    state = [ZI, IZ, ZI*IZ] # |00>
    found = []
    for g in C2:
        for p in state[:2]:
            if (~g)*p*g not in state:
                break
        else:
            found.append(g)
    print("stab:", len(found))
    assert len(found) == 1536 # 1536 == 92160 // 60








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

