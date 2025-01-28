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

from qumba import lin
#lin.int_scalar = numpy.int32 # qupy.lin
from qumba.lin import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce, zeros2, solve2)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.smap import SMap
from qumba.argv import argv
from qumba.matrix import Matrix
from qumba import construct

from qumba.circuit import parsevec, Circuit, send, get_inverse, measure, barrier, variance, vdump, load
from qumba.circuit_css import css_encoder, process





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

    def get_name(name):
        for ggt in gates:
            if names[ggt[0]] == name:
                return ggt

    if argv.pre:
        root = root.act(get_name("CX(0,1)"))
        root = root.act(get_name("CX(2,3)"))

    tree = root

    sols = []
    for trial in range(argv.get("trials",100)):
        found = tree.playout(1, gates, Sx, Sz)
        if found is not None:
            sols.append(found)
            print("found:", len(found), "best:", min(len(s) for s in sols))
            print()

#    for trial in range(argv.get("trials",100)):
#        found = tree.playout(10, gates, Sx, Sz)
#        if found is None:
#            continue
#        sols.append(found)
#
#        print("found:", len(found))
#        print()
#
#        #t = found[len(found)//2]
#        t = found[len(found)//3]
#        for _ in range(10):
#            t2 = t.playout(10, gates, Sx, Sz)
#
#            if t2 is not None:
#                sols.append(t2)
#                print("found:", len(t2))
#
#        print("best:", min(len(s) for s in sols))
#        print()
#
#        tree.prune(2)

    
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
    print("is_equiv:", dode.is_equiv(code))
    

def get_prep(code, dual=False):
    css = code.to_css()
    # 67
    prep = ['CX(13,12)', 'CX(13,14)', 'CX(14,20)', 'CX(17,19)',
    'CX(19,14)', 'CX(18,14)', 'CX(20,18)', 'CX(16,17)', 'CX(12,18)',
    'CX(22,21)', 'CX(22,15)', 'CX(15,14)', 'CX(11,21)', 'CX(14,20)',
    'CX(20,12)', 'CX(16,13)', 'CX(18,13)', 'CX(20,21)', 'CX(14,17)',
    'CX(21,16)', 'CX(13,22)', 'CX(20,15)', 'CX(15,14)', 'CX(17,11)',
    'CX(16,12)', 'CX(21,11)', 'CX(12,14)', 'CX(22,19)', 'CX(12,19)',
    'CX(14,7)', 'CX(12,20)', 'CX(15,13)', 'CX(20,4)', 'CX(18,13)',
    'CX(13,21)', 'CX(17,22)', 'CX(13,10)', 'CX(11,20)', 'CX(11,22)',
    'CX(18,19)', 'CX(11,21)', 'CX(19,16)', 'CX(21,1)', 'CX(22,14)',
    'CX(12,16)', 'CX(18,17)', 'CX(14,15)', 'CX(17,5)', 'CX(20,12)',
    'CX(16,13)', 'CX(14,2)', 'CX(16,10)', 'CX(1,21)', 'CX(14,22)',
    'CX(12,3)', 'CX(19,18)', 'CX(20,14)', 'CX(16,8)', 'CX(18,9)',
    'CX(20,2)', 'CX(19,15)', 'CX(17,1)', 'CX(20,19)', 'CX(15,0)',
    'CX(19,22)', 'CX(11,6)', 'CX(11,4)'] 

    # 65
    prep65 = ['CX(15,18)', 'CX(20,21)', 'CX(21,16)', 'CX(16,17)',
    'CX(19,13)', 'CX(18,17)', 'CX(22,20)', 'CX(15,21)', 'CX(14,11)',
    'CX(18,22)', 'CX(17,0)', 'CX(11,15)', 'CX(13,22)', 'CX(20,18)',
    'CX(22,20)', 'CX(16,14)', 'CX(20,13)', 'CX(16,13)', 'CX(15,16)',
    'CX(22,13)', 'CX(17,16)', 'CX(13,12)', 'CX(16,13)', 'CX(12,21)',
    'CX(14,19)', 'CX(12,13)', 'CX(16,18)', 'CX(19,15)', 'CX(14,20)',
    'CX(18,19)', 'CX(18,1)', 'CX(21,5)', 'CX(16,12)', 'CX(17,13)',
    'CX(17,12)', 'CX(21,17)', 'CX(12,15)', 'CX(21,0)', 'CX(22,12)',
    'CX(20,11)', 'CX(21,18)', 'CX(20,14)', 'CX(15,11)', 'CX(17,14)',
    'CX(12,15)', 'CX(18,12)', 'CX(19,10)', 'CX(11,4)', 'CX(20,3)',
    'CX(11,14)', 'CX(12,8)', 'CX(19,22)', 'CX(22,2)', 'CX(19,8)',
    'CX(7,22)', 'CX(13,7)', 'CX(21,1)', 'CX(19,12)', 'CX(16,15)',
    'CX(18,16)', 'CX(14,9)', 'CX(13,2)', 'CX(11,16)', 'CX(16,22)',
    'CX(15,6)']

    if dual:
        prep += ["H(%d)"%i for i in range(css.mx,css.mx+css.mz)]
    else:
        prep += ["H(%d)"%i for i in range(css.mx)]
    prep = tuple(prep)
    return prep



def get_code():
    if argv.golay:
        tgt = construct.get_golay(23)
        prep = get_prep(tgt, True)
    elif argv.steane:
        tgt = construct.get_713()
        css = tgt.to_css()
        prep = ['CX(3,5)', 'CX(5,2)', 'CX(5,4)', 'CX(5,6)', 'CX(6,0)',
            'CX(4,3)', 'CX(3,6)', 'CX(6,2)', 'CX(4,1)']
        prep += ["H(%d)"%i for i in range(css.mx,css.mx+css.mz)]
        prep = tuple(prep)
    else:
        assert 0
    return tgt, prep



def strong_sim():
    #from strong_clifford.simulator import Simulator
    from strong_clifford.simulator_symbolic_phases import Simulator

    tgt, prep = get_code()

    css = tgt.to_css()
    n_gates = len(prep)
    n = tgt.n

    print(n, n_gates)

    sim = Simulator(n, n_gates=n_gates, draw=True)

    def H(i):
        #print("H", i)
        sim.h(i)
        sim.pauli_error_1(i)
    def CX(i, j):
        #print("CX", i, j)
        sim.cnot(i, j)
        sim.pauli_error_2((i, j))

    for g in reversed(prep):
        exec(str(g))

    print(sim.draw_span())
    print()
    #print(tgt.longstr())

    for i in range(n):
        sim.measure(i)

    p0 = 1e-3
    std_dev_scale = 0.1
    
    #numpy.random.seed(0)
    #p_base_1q = numpy.random.normal(p0, std_dev_scale * p0, 3)
    p_base_1q = [3e-5]*3
    errors_1q = numpy.hstack((1 - numpy.sum(p_base_1q), p_base_1q))
    #p_base_2q = numpy.random.normal(p0, std_dev_scale * p0, 15)
    p_base_2q = [1e-3]*15
    errors_2q = numpy.hstack((1 - numpy.sum(p_base_2q), p_base_2q))

    print(errors_1q)
    print(errors_2q)

    #print(shortstr(css.Hz))

    #   For example, the steane code.
    #   After measurement we have 7 bits of data u_x.
    #   You apply the parity check matrix H_z to get 3 bits of syndrome data v=H_z*u_x.
    #   Then you apply the destabilizers T_x, which gives w_x = u_x + v*T_x.
    #   The decoder needs to take v and predict the logical which is L_z*w_x.
    #   To build a lookup table you run this experiment in simulation
    #   many times, and record for each v how many times each
    #   logical operator L_z*w_x is seen. Then when decoding
    #   you pick the one that was seen in training the most: argmax.

    N = argv.get("N", 128)
    u_x = sim.get_samples(errors_1q, errors_2q, measurement_results=None, shots=N)
    u_x = numpy.array(u_x)
    #print(f"Samples:")
    #print(shortstr(u_x), u_x.shape)
    #print()

    v_syn = dot2(u_x, css.Hz.transpose())
    #print(shortstr(v_syn), v_syn.shape)
    #print()
    v_x = dot2(v_syn, css.Tx)

    #print(shortstr(v_x), v_x.shape)

    w_x = (u_x + v_x) % 2
    #print(shortstr(w_x), w_x.shape)
    #print()

    #print(css.Lx.transpose())
    #print(shortstr(w_x.transpose()))

    LxHx = numpy.concatenate((css.Lx, css.Hx))

    A = solve2(LxHx.transpose(), w_x.transpose())
    assert A is not None
    A = A[:css.k, :] # logical correction
    #print(shortstr(A), A.shape)
    #print()
    #print(v_syn.shape)

    lookup = {}
    for i in range(N):
        key = v_syn[i].tobytes()
        a = tuple(A[:, i])
        lookup.setdefault(key, []).append(a)
        #print(shortstr(v_syn[i]), a)

    print("lookup:", len(lookup))
    succ = 0
    for k,v in lookup.items():
        vc = list(set(v))
        if len(vc) > 1:
            a, b = vc
            va, vb = v.count(a), v.count(b)
            #print("%d(%d,%d)"%(len(v), va, vb), a, b, end=" ")
            #print("*")
            vc = max(va, vb)
            succ += vc
        else:
            succ += len(v)
    print()
    print("error:", 1 - succ / N)




def sim_hugr():
    
    from hugr import tys
    from hugr.build.tracked_dfg import TrackedDfg
    from hugr.package import Package
    from hugr.std.float import FLOAT_T, FloatVal
    from hugr.std.logic import Not
    
    from hugr.tests.conftest import CX, QUANTUM_EXT, H, Measure, Rz, validate


def sim():

    tgt = construct.get_golay(23)
    css = tgt.to_css()
    n = tgt.n
    mx, mz = css.mx, css.mz

    prep = get_prep(tgt)
    
    space = SymplecticSpace(n)
    H = space.H

    #E = reduce(mul, [H(i) for i in range(mx)])
    E = space.get_expr(prep) 

    code = QCode.from_encoder(E, k=tgt.k)
    code.distance()
    print(code)
    #assert code.is_equiv(tgt)
    #print(code.longstr())


    c = measure + barrier + prep
    circuit = Circuit(n)
    qasm = circuit.run_qasm(c)
    if argv.showqasm:
        print(qasm)
        return


    shots = argv.get("shots", 1000)
    samps = send([qasm], shots=shots, error_model=True)
    process(code, samps, circuit)



    

    

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



