#!/usr/bin/env python
"""

Matrix's with columns _indexed by integer sets

"""

from functools import cache, reduce
from operator import add

import numpy

from bruhat.matroid import Matroid, SpMatroid, mask_le
from bruhat.action import mulclose, get_orbits
from bruhat.algebraic import qchoose_2

from qumba import construct 
from qumba.symplectic import SymplecticSpace
from qumba.qcode import QCode
from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.lax import lc_orbits, Lower, Upper
from qumba.util import all_subsets
from qumba.smap import SMap
from qumba.matroid import detect_classical


class IMatrix:
    def __init__(self, H, idxs=None):
        assert isinstance(H, Matrix), H.__class__
        H = H.normal_form()
        m, n = H.shape
        if idxs is None:
            idxs = list(range(n))
        assert len(idxs) == n
        assert len(set(idxs)) == len(idxs) # uniq
        self.H = H
        self.shape = (m, n)
        self.idxs = tuple(idxs)

    def __str__(self):
        smap = SMap()
        H = self.H
        idxs = self.idxs
        for (i,idx) in enumerate(idxs):
            smap[0,i] = str(idx%10)
        smap[1,0] = str(H)
        smap = str(smap)
        return smap

    def __eq__(self, other):
        other = IMatrix.promote(other)
        return self.idxs==other.idxs and other.H==self.H
        #return str(self) == str(other)

    def __hash__(self):
        key = self.H, self.idxs
        return hash(key)

    def __getattr__(self, name):
        attr = getattr(self.H, name)
        return attr

    @classmethod
    def promote(cls, item):
        if isinstance(item, IMatrix):
            return item
        if isinstance(item, numpy.ndarray):
            item = Matrix(item)
        m, n = item.shape
        idxs = list(range(n))
        return IMatrix(item, idxs)

    @classmethod
    def parse(cls, s):
        H = Matrix.parse(s)
        return cls.promote(H)

    @classmethod
    def rand(cls, m, n):
        H = Matrix.rand(m, n)
        return cls.promote(H)

    @cache
    def dual(self):
        K = self.H.kernel()
        K = IMatrix(K, self.idxs)
        return K

    def get_wenum(self):
        wenum = self.H.get_wenum()
        return wenum

    def __getitem__(self, key):
        row, col = key
        H = self.H[key]
        assert 0

    def normal_form(self):
        H = self.H.normal_form()
        return IMatrix(H, self.idxs)

    def restrict(self, idxs):
        # restrict = delete = puncture
#        print("restrict", idxs)
#        print(self)
        if type(idxs) is int:
            idxs = [idxs]
        H = self.H
        m, n = H.shape
        for i in idxs:
            assert i in self.idxs
        cols = [self.idxs.index(i) for i in idxs]
        J = H[:, cols]
        J = J.normal_form()
        J = IMatrix(J, idxs)
#        print("=")
#        print(J, J.shape)
        return J
    delete = restrict
    
    def contract(self, idxs):
#        print("contract", idxs)
#        print(self)
        H = self.H
        m, n = H.shape
        if type(idxs) is int:
            idxs = [idxs]
        for i in idxs:
            assert i in self.idxs
        jdxs = [i for i in self.idxs if i not in idxs]
        col_jdxs = [self.idxs.index(i) for i in jdxs]
        col_idxs = [self.idxs.index(i) for i in idxs]
        A = H[:, col_jdxs]
        #print("contract", jdxs)
        #print(A, A.shape)
        K = A.t.kernel()
        #print(K, K.shape)
        KH = K*H
        KH = KH[:, col_idxs]
        KH = KH.normal_form()
        KH = IMatrix(KH, idxs)
#        print("=")
#        print(KH, KH.shape)
        return KH

    def is_loop(self, idx):
        assert idx in self.idxs
        i = self.idxs.index(idx)
        return self.H[:, i].sum() == 0

    def is_coloop(self, idx):
        assert idx in self.idxs
        return self.dual().is_loop(idx)

    def _tutte(self, x, y, depth=0):
        m, n = self.shape
        if n == 0:
            return 1
        i = self.idxs[0]
        #print(" "*depth+ "_tutte", i)
        #smap = SMap()
        #smap[0,depth] = str(self)
        #print(smap)
        idxs = list(self.idxs)
        idxs.remove(i)
        if self.is_loop(i):
            #print(" "*depth+ "is_loop")
            lhs = self.delete(idxs)
            p = y*lhs._tutte(x, y, depth+1)
        elif self.is_coloop(i):
            #print(" "*depth+ "is_coloop")
            rhs = self.contract(idxs)
            p = x*rhs._tutte(x, y, depth+1)
        else:
            #print(" "*depth+ "else")
            lhs = self.delete(idxs)
            rhs = self.contract(idxs)
            lhs = lhs._tutte(x, y, depth+1)
            rhs = rhs._tutte(x, y, depth+1)
            p = lhs+rhs
        return p

#    @classmethod
#    def get_xy(self):
#        ring = element.Z
#        zero = Poly({}, ring)
#        one = Poly({():1}, ring)
#        x = Poly("x", ring)
#        y = Poly("y", ring)
#        return x, y

    def get_tutte(self):
        from sage import all_cmdline as sage
        R = sage.PolynomialRing(sage.ZZ, list("xy"))
        x, y, = R.gens()
        p = self._tutte(x, y)
        return p

    def get_matroid(self):
        return detect_classical(self.H)




def build_pascal(m, n):
    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, list("xy"))
    x, y = R.gens()

    print("\nbuild_pascal", m, n)
    k = n-m
    found = {}
    for H in qchoose_2(n, m):
        H = IMatrix.promote(H)
        p = H._tutte(x, y)
        found.setdefault(p, []).append(H)
        M = H.get_matroid()
        q = M.get_tutte()
        if type(q) is int:
            continue
        q = eval(q.python_str())
        assert (p==q), (p,q)
        #print(M)
        assert p(x=1, y=1) == len(M.get_basis())
        assert p(x=2, y=1) == len(M.masks)

    #return

    #print("found:", len(found))
    items = []
    for p,Hs in found.items():
        s = str(sage.latex(p))
        items.append(s)
        print(len(Hs), ":", str(p).rjust(40), "\t", end="")
        smap = SMap()
        col = 0
        for H in Hs:
            smap[0, col] = str(H.H)
            col += n+1
        #print(smap)
        for H in Hs:
            wenum = H.get_wenum()
        print(wenum, end=" ")
        if type(p) is int:
            continue
        #w = (x-y)**k * y**(n-k) * p(x=x/y, y=(x+y)/(x-y)) # wenum of G = H.dual()
        w = (x-y)**(n-k) * y**k * p(x=y,y=x)(x=x/y, y=(x+y)/(x-y))
        print(w)
        #print([i for (i,e) in p(x=1+x, y=y-1)])
    return items


def test_pascal():

    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, list("xy"))
    x, y = R.gens()

    #p = x**4 + x**3 * y
    #print(sage.factor(x**5+2*x**4-2*x**2-x))

    #return

    N = 6
    table = {}
    for n in range(N):
      for m in range(n+1):
        result = build_pascal(m, n)
        table[n,m] = result

    #return

    for n in range(N):
        #row = [table[n,m] for m in range(n+1)]
        k = max(len(table[n,m]) for m in range(n+1))
        for m in range(n+1):
            cell = table[n,m]
            table[n,m] = cell + [""]*(k-len(cell))
            assert len(table[n,m]) == k, (k, table[n,m])
        print(r"\hline")
        for row in zip(*[table[n,m] for m in range(n+1)]):
            row = reduce(add, [('',item) for item in row])
            row = ('',)*(4-n) + row
            #print(row, len(row))
            print(" & ".join(row), r"\\", "%", len(row))
        

    


def main():
    m, n = 3, 5

    for trial in range(20):
        H = IMatrix.rand(m, n)
        print(H)

        J = H.dual()
        #print(J)
        assert J.dual() == H

        p = H.get_tutte()
        q = J.get_tutte()
        print(p)
        print(p(x=1))
        print(p(y=1))
        print(H.get_wenum())
        #print(q)
        R = p.parent()
        x, y = R.gens()
        assert q(x=y, y=x) == p


def test_coassoc():

    m, n = 4, 7
    idxs = [0,1]
    jdxs = [2,4]
    kdxs = [3,5,6]

    for trial in range(20):

        H = IMatrix.rand(m, n)
        H = H.normal_form()
        print(H)

        smap = SMap()
        smap[0,0] = str(H)
    
        AB = H.restrict(idxs+jdxs)
        A0 = AB.restrict(idxs)
        B0 = AB.contract(jdxs)
        C0 = H.contract(kdxs)

        A1 = H.restrict(idxs)
        #_n = A1.shape[1]
        BC = H.contract(jdxs+kdxs)
        #print("BC =")
        #print(BC)
        #_jdxs = [j-_n for j in jdxs]
        #_kdxs = [k-_n for k in kdxs]
        B1 = BC.restrict(jdxs)
        C1 = BC.contract(kdxs)

        print(A0==A1, B0==B1, C0==C1)
        assert A0==A1
        assert B0==B1
        assert C0==C1


def test_contract():

    #m, n = 2, 5
    #H = IMatrix.rand(m, n)
    H = IMatrix.parse("""
    1011
    0111
    """)
    print(H)
    print()

    H = H.normal_form()
    print(H)
    print()

    J = H.contract([2,3])
    print(J, J.shape)
    print()



def test_delete_contract():
    from bruhat.matroid import find_lin

    n, m = 5, 3
    found = set()
    for H in qchoose_2(n, m):
        H = Matrix(H)
        M = detect_classical(H)
        if M in found:
            continue
        found.add(M)
    
        M0 = M.delete(n-1)
        M1 = M.contract(n-1)
        if M0 is None or M1 is None:
            continue

        print(M)
        print(H)

        A = H[:, n-1:]
        print(A, A.shape)
        K = A.t.kernel()
        print(K, K.shape)
        KH = K*H
        print(KH, KH.shape)
        KH = KH[:, :n-1].normal_form()
        print(KH, KH.shape)

        print(M0)
        H0 = iter(find_lin(M0)).__next__().A
        #for H0 in find_lin(M0):
        #    print(H0)
        H0 = Matrix(H0)
        print(H0)
        H00 = H[:, :n-1].row_reduce()
        assert(H0==H00)
        #print(H00)
        print(M1)

        H1 = iter(find_lin(M1)).__next__().A
        H1 = Matrix(H1)
        #for H1 in find_lin(M1):
        #    print(H1)
        print(H1, H1==KH)
        assert H1==KH

        print()

        #if len(found) > 40:
        #    break
    

def test_wenum():

    ns = argv.get("n")
    if ns is not None:
        ns = [ns]
    else:
        ns = range(8)
    for n in ns:
      print("n=%d: "%n, end=' ', flush=True)
      for m in range(n+1):
        found = set()
        for H in qchoose_2(n, m):
            H = Matrix(H)
            H1 = IMatrix(H)
            #p = H.get_wenum()
            p = H1.get_tutte()
            found.add(p)
        print(len(found), end=' ', flush=True)
      print()


def test_random_wenum():
    oeis = [
        [1,                                                       ],
        [1,   1,                                                  ],
        [1,   2,   1,                                             ],
        [1,   3,   3,   1,                                        ],
        [1,   4,   6,   4,    1,                                  ],
        [1,   5,  10,  10,    5,    1,                            ],
        [1,   6,  16,  22,   16,    6,    1,                      ],
        [1,   7,  23,  43,   43,   23,    7,    1,                ],
        [1,   8,  32,  77,  106,   77,   32,    8,    1,          ],
        [1,   9,  43, 131,  240,  240,  131,   43,    9,   1,     ],
        [1,  10,  56, 213,  516,  705,  516,  213,   56,  10,   1,],
        [1,  11,  71, 333, 1060, 1988, 1988, 1060,  333,  71,  11,   1],
    ]

    for n in range(12):
      print("n=%d: "%n, end='', flush=True)
      for m in range(n//2 + 1):
      #for m in range(n+1):
        count = oeis[n][m]
        found = set()
        while len(found) < count:
            while 1:
                H = IMatrix.rand(m, n)
                if H.rank() == m:
                    break
            H = H.normal_form()
            #print(H)
            p = H.get_tutte()
            found.add(p)
        print(len(found), end=' ', flush=True)

      print()


def test_tutte():
    A = IMatrix.parse("""
    1000101
    0100110
    0010011
    0001011
    """)
    print(A)
    print(A.get_tutte())
    B = IMatrix.parse("""
    1000100
    0100111
    0010011
    000101a
    """.replace("a", "1"))
    print(B)
    print(B.get_tutte())



if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        import cProfile as profile
        profile.run("%s()"%name)

    elif argv.pyinstrument:
        fn = eval(name)
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            fn()
        profiler.print()

    else:
        fn = eval(name)
        fn()


    t = time() - start_time
    print("\nOK! finished in %.3f seconds\n"%t)




