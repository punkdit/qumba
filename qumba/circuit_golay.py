#!/usr/bin/env python


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



def gate_search(space, gates, target):
    gates = list(gates)
    A = space.get_identity()
    found = []
    while A != target:
        if len(found) > 100:
            return
        count = (A+target).sum()
        #print(count, end=' ')
        shuffle(gates)
        for g in gates:
            B = g*A # right to left
            if (B+target).sum() <= count:
                break
        else:
            #print("X ", end='', flush=True)
            return None
        found.insert(0, g)
        A = B
    #print("^", end='')
    return found

def test_1():

    #code = construct.get_golay(23)
    code = construct.get_toric(2,2)
    n = code.n

    H = code.H
    print(code)
    print(code.longstr())

    space = code.space
    E = code.get_encoder()
    #print(E)

    name = space.get_name(E)
    print(name, len(name))

    code = code.to_css()
    #name = css_encoder(code.Hz)
    #print(name, len(name))

    E1 = space.get_expr(name)

    print(E == E1)

#    dode = QCode.from_encoder(E1, k=1)
#    assert dode.is_css()
#    dode = dode.to_css()
#    dode.bz_distance()
#    print(dode)

    CX, H = space.CX, space.H
    E1 = E #* H(3)*H(4)*H(5)

    gates = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    #gates += [H(i) for i in range(n)]


    find_encoder(space, code.H, gates)

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

class Circuit:
    def __init__(self, ux, uz, g, parent=None):
        self.ux = ux
        self.uz = uz
        self.g = g
        self.parent = parent
        if parent is not None:
            parent.children.append(self)
            self.size = parent.size + 1
        else:
            self.size = 0
        self.children = []

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



def greedy_search(n, Hx, Hz):
    mx, mz = len(Hx), len(Hz)

    names = get_GL(n)
    gates = list(names)
    I = Matrix.identity(n)

    Sx = get_span(Hx)
    Sz = get_span(Hz)

    def get_overlap(Ux, Uz):
        score = [int((u+Sx).A.sum(1).min()) for u in Ux]+[int((u+Sz).A.sum(1).min()) for u in Uz]
        w = sum(score)
        mean = w // len(score)
        var = sum( [(i-mean)**2 for i in score] )
        return score,var

    trial = 0
    while 1:
        trial += 1

        Ux = I[:mx, :]
        Uz = I[mx:mx+mz, :]
        circuit = Circuit(Ux, Uz, I)

        while Ux.t.solve(Hx.t) is None or Uz.t.solve(Hz.t) is None:
            score, var = get_overlap(Ux, Uz)

            w = sum(score) + var
            shuffle(gates)
            for g in gates:
                ux = (g * Ux.t).t
                uz = (g.t * Uz.t).t
                s,var = get_overlap(ux, uz) # slow !
                if sum(s) < sum(score):
                    break
            else:
                print("fail")
                break 
            print(sum(s), end=" ")
            Ux = ux
            Uz = uz
            #circuit.append(g)
            circuit = Circuit(Ux, Uz, g, circuit)

            if len(circuit) > n**2:
                print("too big")
                break
        else:
            print("\nsuccess!", trial)
            print([names[g] for g in circuit], len(circuit))
            break


def monte_search(n, Hx, Hz):
    mx, mz = len(Hx), len(Hz)

    names = get_GL(n)
    gates = list(names)
    I = Matrix.identity(n)

    Sx = get_span(Hx)
    Sz = get_span(Hz)

    def get_overlap(Ux, Uz):
        score = [int((u+Sx).A.sum(1).min()) for u in Ux]+[int((u+Sz).A.sum(1).min()) for u in Uz]
        w = sum(score)
        mean = w // len(score)
        var = sum( [(i-mean)**2 for i in score] )
        return score,var

    Ux = I[:mx, :]
    Uz = I[mx:mx+mz, :]
    circuit = Circuit(Ux, Uz, I)

    while Ux.t.solve(Hx.t) is None or Uz.t.solve(Hz.t) is None:
        score, var = get_overlap(Ux, Uz)

        for trial in range(len(gates)):
            g = choice(gates)
            ux = (g * Ux.t).t
            uz = (g.t * Uz.t).t
            s,var = get_overlap(ux, uz) # slow !
            if sum(s) < sum(score):
                print(sum(s), end=" ")
                circuit = Circuit(ux, uz, g, circuit) # push
                break
        else:
            N = len(circuit)
            circuit = circuit[randint(1, N-1)]
            print(".", end="", flush=True)

        Ux = circuit.ux
        Uz = circuit.uz

    else:
        print("\nsuccess!")
        print([names[c.g] for c in circuit], len(circuit))

# ----------------------------------------------------------------------------

class Tree:
    def __init__(self, Ux, Uz, ggt, parent=None):
        self.Ux = Ux
        self.Uz = Uz
        self.ggt = ggt
        self.parent = parent
        if parent is not None:
            parent.children[ggt] = self
            self.size = parent.size + 1
        else:
            self.size = 0
        self.children = {}

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

    def act(self, ggt):
        tree = self.children.get(ggt)
        if tree is not None:
            #print("!", end="")
            return tree
        g, gt = ggt
        Ux, Uz = self.Ux*gt, self.Uz*g
        return Tree(Ux, Uz, ggt, self)


def tree_search(n, Hx, Hz):
    mx, mz = len(Hx), len(Hz)
    Hxt, Hzt = Hx.t, Hz.t

    names = get_GL(n)
    gates = [(g,g.t) for g in names]
    I = Matrix.identity(n)

    Sx = get_span(Hx)
    Sz = get_span(Hz)

    #@cache
    def get_overlap(Ux, Uz):
        score = [int((u+Sx).A.sum(1).min()) for u in Ux]+[int((u+Sz).A.sum(1).min()) for u in Uz]
        return sum(score)

    Ux = I[:mx, :]
    Uz = I[mx:mx+mz, :]
    tree = Tree(Ux, Uz, I)

    print("warmup...")
    for ggt in gates:
        child = tree.act(ggt)
        for ggt in gates:
            child.act(ggt)

    while tree.Ux.t.solve(Hxt) is None or tree.Uz.t.solve(Hzt) is None:
        score = get_overlap(tree.Ux, tree.Uz)

        for trial in range(len(gates)):
            ggt = choice(gates)
            child = tree.act(ggt)
            s = get_overlap(child.Ux, child.Uz) # slow !
            if s < score:
                print(s, end=" ")
                #tree = Tree(ux, uz, g, tree) # push
                tree = child
                break
        else:
            #print(s, end=" ")
            N = len(tree)
            #tree = tree[randint(1, N-1)]
            tree = tree[N-1]
            #print("[%d]"%len(tree), end=" ", flush=True)
            print(".", end=" ", flush=True)

        assert len(tree) < n**2

    else:
        return [names[c.ggt[0]] for c in tree]


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
    def get_overlap(self, Sx, Sz):
        Ux, Uz = self.Ux, self.Uz
        score = sum([int((u+Sx).A.sum(1).min()) for u in Ux]+[int((u+Sz).A.sum(1).min()) for u in Uz])
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



def graph_search(n, Hx, Hz):
    mx, mz = len(Hx), len(Hz)
    Hxt, Hzt = Hx.t, Hz.t

    names = get_GL(n)
    gates = [(g,g.t) for g in names]
    I = Matrix.identity(n)

    Sx = get_span(Hx)
    Sz = get_span(Hz)

    Ux = I[:mx, :]
    Uz = I[mx:mx+mz, :]
    tree = Graph(Ux, Uz)
    tree.get_overlap(Sx, Sz)

    print("warmup...")
    for ggt in gates:
        child = tree.act(ggt)
        child.get_overlap(Sx, Sz)
        for ggt in gates:
            t = child.act(ggt)
            t.get_overlap(Sx, Sz)

    while 1:
        score = tree.get_overlap(Sx, Sz)
        if score == 0:
            break

        for trial in range(len(gates)):
            ggt = choice(gates)
            child = tree.act(ggt)
            s = child.get_overlap(Sx, Sz)
            if s < score:
                print(s, end=" ")
                tree = child
                break
        else:
            #print(s, end=" ")
            N = len(tree)
            #tree = tree[randint(1, N-1)]
            tree = tree[N-1]
            #print("[%d]"%len(tree), end=" ", flush=True)
            print(".", end=" ", flush=True)

        assert len(tree) < n**2

    return [names[c.ggt[0]] for c in tree]


def directed_search(n, Hx, Hz):

    names = get_GL(n)
    gates = [(g,g.t) for g in names]
    I = Matrix.identity(n)

    mx, mz = len(Hx), len(Hz)
    Ux = I[:mx, :]
    Uz = I[mx:mx+mz, :]

    if argv.reverse:
        Ux, Uz, Hx, Hz = Hx, Hz, Ux, Uz

    Sx = get_span(Hx)
    Sz = get_span(Hz)
    root = Graph(Ux, Uz)

    tree = root

    while 1:
        found = tree.playout(100, gates, Sx, Sz)
        if found is not None:
            tree = found
            break
    
        children = [child for child in tree.children.values() if child.scores]
        print("children:", len(children))
        if len(children) < 10:
            print("backtrack..")
            tree = root
            root.prune(4)
            continue
    
        children.sort( key = lambda child : -sum(child.scores)/len(child.scores) )
        tree = choice(children[:len(children)//2])
    
    result = [names[c.ggt[0]] for c in tree]
    return list(reversed(result))


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



