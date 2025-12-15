#!/usr/bin/env python

"""
see also: graph_states.py
see also: lax.py

"""

import os
from random import shuffle
from math import sin, cos, pi
from functools import cache, reduce
from operator import add

import numpy

from qumba.action import mulclose
from qumba.symplectic import SymplecticSpace
from qumba.matrix import Matrix
from qumba.qcode import QCode, fromstr, strop
from qumba.smap import SMap
from qumba import lin
from qumba.argv import argv
from qumba.util import binomial, factorial, choose
from qumba import construct


from huygens import config
config(text="pdflatex")
from huygens.namespace import Canvas, path, grey, black, red, white, st_thick





class Matroid:
    def __init__(self, n, masks):
        "n:number of elements, masks: the independent sets as bit-vectors"
        self.n = n
        masks = list(masks)
        masks.sort()
        masks = tuple(masks)
        self.masks = masks
        self.items = set(masks)
        self.key = (n, masks)
        self.rank = max(sum(a) for a in masks)

    def le(self, a, b):
        for i in range(self.n):
            if a[i]>b[i]:
                return False
        return True

    #def mul(self, a, b):
    #    return tuple(ai

    def restrict(self, mask):
        assert len(mask) == self.n
        masks = {m for m in self.masks if self.le(m, mask)}
        return Matroid(self.n, masks)

    def all_masks(self):
        return list(numpy.ndindex((2,)*self.n))

    def __str__(self):
        masks = [{i for (i,ii) in enumerate(mask) if ii} for mask in self.masks]
        return "Matroid(%d, %s)"%(self.n, masks)
    __repr__ = __str__

    def check(self):
        # check the Matroid axioms
        n = self.n
        items = self.items

        # (1) the empty set is independent
        assert (0,)*n in items

        # (2) independence is down-closed
        for a in self.all_masks():
            for b in items:
                if self.le(a, b):
                    assert a in items

        # (3) independent set exchange 
        for a in items:
          for b in items:
            if sum(a) <= sum(b):
                continue
            # A has more elements than B
            for i in range(n):
                if a[i] and not b[i]:
                    break
            else:
                assert 0


    def __hash__(self):
        return hash(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __le__(self, other):
        assert self.n == other.n
        return self.items.issubset(other.items)


def all_matroids(n):

    from qumba.umatrix import Solver, UMatrix, Var, And, Or, If
    bits = list(numpy.ndindex((2,)*n))

    # is this subset an independent set?
    vs = []
    lookup = {}
    for a in bits:
        name = "v"+''.join(str(i) for i in a)
        v = Var(name)
        vs.append(v)
        lookup[a] = v

    solver = Solver()
    Add = solver.add

    def le(a, b):
        for i in range(n):
            if a[i]>b[i]:
                return False
        return True

    pairs = [(a,b) for a in bits for b in bits]
    for a in bits:
        if sum(a)==0:
            Add(lookup[a] == 1)

    for (a,b) in pairs:
        if le(a,b):
            Add( If(lookup[b].get(), lookup[a].get(), True) )

    for (a,b) in pairs:
        if sum(a) <= sum(b):
            continue
        terms = []
        for i in range(n):
            if a[i] and not b[i]:
                c = list(b)
                c[i] = 1
                c = tuple(c)
                terms.append(lookup[c].get())
        assert terms
        Add( If((lookup[a] * lookup[b]).get(), Or(*terms), True) )
    

    count = 0
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
    
        sol = {a:lookup[a].get_interp(model) for a in bits}
        found = []
        for (k,v) in sol.items():
            if v:
                #found.append( {i for i in range(n) if k[i]} )
                found.append(k)
        #print(found)
        M = Matroid(n, found)
        M.check()
        yield M

        Add(Or(*[(lookup[a] != sol[a]) for a in bits]))
        count += 1
    #print("all_matroids(%d) = %d" % (n, count))


def test_rank():

    n = 3

    from qumba.graph_states import render_func
    cvs = Canvas()

    x = y = 0
    matroids = list(all_matroids(n))
    matroids.sort(key = lambda m:(m.rank, len(m.masks)))
    for idx,M in enumerate(matroids):
        print(M)

        f = []
        for mask in M.all_masks():
            R = M.restrict(mask)
            R.check()
            #print("\t%s : %d"%(mask, R.rank))
            f.append(R.rank)
        fg = render_func(n, f)
        cvs.insert(x, y, fg)
        #cvs.text(x, y, str(M))
        for mi,mask in enumerate(M.masks):
            cvs.text(x, y-0.5*mi, str(mask))
        x += fg.get_bound_box().width

    cvs.writePDFfile("matroid_%d.pdf"%n)

    
def test_matroids():

    # https://oeis.org/A058669
    for n in range(7):
        found = {i:[] for i in range(n+1)}
        for M in all_matroids(n):
            #print(M, M.rank)
            found[M.rank].append(M)
    
        print(n, [len(found[i]) for i in range(n+1)])

    # https://oeis.org/A058673
    assert len(list(all_matroids(0))) == 1
    assert len(list(all_matroids(1))) == 2
    assert len(list(all_matroids(2))) == 5
    assert len(list(all_matroids(3))) == 16
    assert len(list(all_matroids(4))) == 68
    assert len(list(all_matroids(5))) == 406
    #assert len(list(all_matroids(6))) == 3807
    #assert len(list(all_matroids(7))) == 75164

    




if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    print("%s()"%fn)

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))


