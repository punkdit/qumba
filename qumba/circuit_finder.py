#!/usr/bin/env python

"""
previous version: circuit_golay.py

"""


import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice, seed
from operator import add, matmul, mul
from functools import reduce, cache
import pickle

import numpy

from qumba import solve
#solve.int_scalar = numpy.int32 # qupy.solve
from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce, zeros2)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.smap import SMap
from qumba.argv import argv
from qumba.matrix import Matrix
from qumba import construct


# ----------------------------------------------------------------------------
    

def get_span(H):
    n = H.shape[1]
    wenum = {i:[] for i in range(n+1)}
    for u in H.rowspan():
        d = u.sum()
        wenum[d].append(u)
    for i in range(1, n+1):
        if wenum[i]:
            break
    vs = [v.A[0,:] for v in wenum[i]]
    hx = Matrix(vs)
    return hx


def get_GL(n):
    # These generate GL(n)
    # the (block) symplectic matrix for A in GL is:
    # [[A,0],[0,A.t]]
    I = Matrix.identity(n)
    gates = []
    names = {}
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        A = identity2(n)
        A[i,j] = 1
        A = Matrix(A)
        gates.append(A)
        assert A*A == I
        names[A] = "CX(%d,%d)"%(i,j)
        #print(A, "\n")
        #print(~A, "\n\n")
    return names

# ----------------------------------------------------------------------------


class Graph:
    def __init__(self, Ux, Uz, parent=None, ggt=None):
        self.Ux = Ux
        self.Uz = Uz
        self.parent = parent
        self.ggt = ggt
        if parent is not None:
            self.size = parent.size + 1
        else:
            self.size = 0
        self.children = {}
        self.scores = []
        self.score = None

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        #print("__getitem__", i, self.size)
        if i >= self.size:
            raise IndexError
        c = self
        while i>0:
            c = c.parent
            i -= 1
        return c

    def prune(self, depth):
        assert depth >= 0
        if depth == 0:
            self.scores = []
            self.children = {}
        else:
            for child in self.children.values():
                child.prune(depth - 1)

    @cache
    def get_overlap(self, Sx, Sz): # hotspot 
        Ux, Uz = self.Ux, self.Uz
        score = 0
        for u in Ux:
            score += (u+Sx).A.sum(1).min()
        for u in Uz:
            score += (u+Sz).A.sum(1).min()
        return score

    def act(self, ggt):
        tree = self.children.get(ggt)
        if tree is not None:
            #print("!", end="")
            return tree
        g, gt = ggt
        Ux, Uz = self.Ux*gt, self.Uz*g
        graph = Graph(Ux, Uz, self, ggt)
        self.children[ggt] = graph
        return graph

    def mark_end(self, score):
        tree = self.parent
        while tree is not None:
            tree.scores.append(score)
            tree = tree.parent

    def playout(self, trials, gates, Sx, Sz):
        tree = self

        count = 0
        while count < trials:
            score = tree.get_overlap(Sx, Sz)
            if score == 0:
                print("success!")
                return tree
    
            for _ in range(len(gates)):
                ggt = choice(gates)
                child = tree.act(ggt)
                s = child.get_overlap(Sx, Sz)
                if s < score:
                    print(s, end=" ")
                    tree = child
                    break
            else:
                print(".", end=" ", flush=True)
                print(score)
                tree.mark_end(score)
                tree = self
                count += 1
    
            assert len(tree) < 1000



def back_search(n, Hx, Hz):

    names = get_GL(n)
    gates = [(g,g.t) for g in names]
    I = Matrix.identity(n)

    mx, mz = len(Hx), len(Hz)
    Ux = I[:mx, :]
    Uz = I[mx:mx+mz, :]

    # reverse
    Ux, Uz, Hx, Hz = Hx, Hz, Ux, Uz

    Sx = get_span(Hx)
    Sz = get_span(Hz)
    root = Graph(Ux, Uz)

    tree = root

    sols = []
    for trial in range(argv.get("trials",100)):
        found = tree.playout(1, gates, Sx, Sz)
        if found is not None:
            sols.append(found)
            print("found:", len(found), "best:", min(len(s) for s in sols))
            print()
    
    assert sols
    sols.sort(key = len)
    print([len(s) for s in sols])
    tree = sols[0]
    
    result = [names[c.ggt[0]] for c in tree]
    return list(reversed(result))



def test():

    if argv.code == (7,1,3):
        code = construct.get_713()

    elif argv.code == (10,2,3):
        code = construct.get_10_2_3()

    elif argv.code == (16,6,4):
        code = construct.reed_muller() # fail

    elif argv.code == (15, 7, 3):
        code = QCode.fromstr("""
        XXXX.X.XX..X...
        XXX.X.XX..X...X
        XX.X.XX..X...XX
        X.X.XX..X...XXX
        ZZZZ.Z.ZZ..Z...
        ZZZ.Z.ZZ..Z...Z
        ZZ.Z.ZZ..Z...ZZ
        Z.Z.ZZ..Z...ZZZ
        """) # [[15, 7, 3]] # fail
    
    elif argv.code == (12, 2, 4):
        code = QCode.fromstr("""
        XXX....X.X.X
        X.XX..X.XX..
        XX..XXX....X
        ..XX.XXX...X
        XX..X...XXX.
        ZZ..Z..ZZ..Z
        Z.Z.....ZZZZ
        ....ZZ.ZZZZ.
        .ZZZ...Z..ZZ
        ..Z.ZZZ...ZZ
        """) # [[12, 2, 4]] # fail
    elif argv.code == (23, 1, 7):
        code = construct.get_golay(23)
    else:
        return

    n = code.n
    css = code.to_css()
    Hx = Matrix(css.Hx)
    Hz = Matrix(css.Hz)

    print("Hx")
    print(Hx)
    print("Hz")
    print(Hz)

    #greedy_search(n, Hx, Hz)
    #monte_search(n, Hx, Hz)
    names = back_search(n, Hx, Hz)

    print()
    if names is None:
        return

    print()
    print(names, len(names))

    space = SymplecticSpace(n)
    H = space.H
    expr = tuple(names)

    mx, mz = css.mx, css.mz
    E = reduce(mul, [H(i) for i in range(mx, mx+mz)])
    E = reduce(mul, [H(i) for i in range(mx)])
    E = space.get_expr(expr) * E

    dode = QCode.from_encoder(E, k=code.k)
    dode.distance()
    print(dode)
#    print(dode.longstr())
#    print()
#
#    print(code.longstr())
#    print()
#    print(code.H * space.F * dode.H.t)

    print("is_equiv:", dode.is_equiv(code))
    



# ['CX(14,20)', 'CX(12,13)', 'CX(14,15)', 'CX(14,11)', 'CX(16,19)', 'CX(22,19)', 'CX(21,18)', 'CX(18,14)', 'CX(13,22)', 'CX(19,13)', 'CX(17,16)', 'CX(12,16)', 'CX(12,18)', 'CX(16,13)', 'CX(16,21)', 'CX(22,12)', 'CX(15,11)', 'CX(17,11)', 'CX(14,17)', 'CX(19,22)', 'CX(21,19)', 'CX(17,12)', 'CX(15,21)', 'CX(19,21)', 'CX(13,18)', 'CX(20,12)', 'CX(20,16)', 'CX(15,13)', 'CX(11,19)', 'CX(16,15)', 'CX(19,14)', 'CX(16,20)', 'CX(16,22)', 'CX(22,12)', 'CX(11,12)', 'CX(17,11)', 'CX(22,12)', 'CX(15,21)', 'CX(19,20)', 'CX(14,19)', 'CX(20,10)', 'CX(20,11)', 'CX(13,16)', 'CX(22,17)', 'CX(18,22)', 'CX(21,7)', 'CX(1,19)', 'CX(12,8)', 'CX(12,13)', 'CX(9,19)', 'CX(20,7)', 'CX(18,0)', 'CX(11,2)', 'CX(20,21)', 'CX(13,9)', 'CX(16,22)', 'CX(17,11)', 'CX(20,19)', 'CX(2,17)', 'CX(14,4)', 'CX(11,6)', 'CX(11,15)', 'CX(20,15)', 'CX(16,2)', 'CX(15,1)', 'CX(22,3)', 'CX(11,17)', 'CX(19,22)', 'CX(16,5)'] 69

    

if __name__ == "__main__":

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

    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)



