#!/usr/bin/env python


import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
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


class Circuit:
    def __init__(self, ux, uz, g, parent=None):
        self.ux = ux
        self.uz = uz
        self.g = g
        self.parent = parent
        if parent is not None:
            self.size = parent.size + 1
        else:
            self.size = 0

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
        return c.g



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

    #for trial in range(1000):
    trial = 0
    while 1:
        trial += 1

        Ux = I[:mx, :]
        Uz = I[mx:mx+mz, :]
        circuit = Circuit(Ux, Uz, I)

        #circuit = []
        while Ux.t.solve(Hx.t) is None or Uz.t.solve(Hz.t) is None:
            #print()
            #print(Ux, "\n"+'-'*n)
            #print(Uz, "\n"+'-'*n)
    
            score, var = get_overlap(Ux, Uz)

            w = sum(score) + var
            #print((w,var), end=' ')
    
            #best = []
            shuffle(gates)
            for g in gates:
                ux = (g * Ux.t).t
                uz = (g.t * Uz.t).t
                s,var = get_overlap(ux, uz) # slow !
                #if sum(s) < sum(score):
                #    best.append( (ux, uz, g, s, var) )
                #if max(s) == max(score) and sum(s) < sum(score):
                #    break
                if sum(s) < sum(score):
                    #best.append((ux, uz, g))
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
                #print()
                #print(Ux,'\n------')
                #print(Uz,'\n')
                print("too big")
                break
        else:
            print("\nsuccess!", trial)
            print([names[g] for g in circuit], len(circuit))
            break



def test():
    code = construct.get_713()

    code = construct.get_10_2_3()
    # ['CX(6,9)', 'CX(7,1)', 'CX(5,3)', 'CX(8,7)', 'CX(4,8)',
    # 'CX(9,3)', 'CX(7,0)', 'CX(1,9)', 'CX(6,2)', 'CX(4,6)',
    # 'CX(6,0)', 'CX(3,0)', 'CX(5,4)', 'CX(5,7)', 'CX(5,0)'] 15

    _code = construct.reed_muller() # fail

    _code = QCode.fromstr("""
    XXXX.X.XX..X...
    XXX.X.XX..X...X
    XX.X.XX..X...XX
    X.X.XX..X...XXX
    ZZZZ.Z.ZZ..Z...
    ZZZ.Z.ZZ..Z...Z
    ZZ.Z.ZZ..Z...ZZ
    Z.Z.ZZ..Z...ZZZ
    """) # [[15, 7, 3]] # fail

    _code = QCode.fromstr("""
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

    n = code.n
    code = code.to_css()
    Hx = Matrix(code.Hx)
    Hz = Matrix(code.Hz)

    print("find_encoder")
    print("Hx")
    print(Hx)
    print("Hz")
    print(Hz)

    greedy_search(n, Hx, Hz)
    

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



