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

from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver

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

    def rankfunc(self):
        #masks = [tuple(mask) for mask in self.all_masks()]
        func = {mask:self.restrict(mask).rank for mask in self.all_masks()}
        return func

    def __str__(self):
        masks = [{i for (i,ii) in enumerate(mask) if ii} for mask in self.masks]
        return "Matroid(%d, %s)"%(self.n, masks)
    __repr__ = __str__

    def __rmul__(self, g):
        #print("__rmul__")
        masks = [tuple(m[g[i]] for i in range(self.n)) for m in self.masks]
        return Matroid(self.n, masks)

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

    bits = list(numpy.ndindex((2,)*n))

    # is this subset an independent set?
    vs = []
    lookup = {}
    for a in bits:
        name = "v"+''.join(str(i) for i in a)
        v = Bool(name)
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
            Add(lookup[a])

    for (a,b) in pairs:
        if le(a,b):
            Add( If(lookup[b], lookup[a], True) )

    for (a,b) in pairs:
        if sum(a) <= sum(b):
            continue
        terms = []
        for i in range(n):
            if a[i] and not b[i]:
                c = list(b)
                c[i] = 1
                c = tuple(c)
                terms.append(lookup[c])
        assert terms
        Add( If(And(lookup[a], lookup[b]), Or(*terms), True) )
    
    count = 0
    matroids = set()
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
    
        sol = {a:model.get_interp(lookup[a]) for a in bits}
        found = []
        for (k,v) in sol.items():
            if v:
                found.append(k)
        M = Matroid(n, found)
        M.check()
        assert M not in matroids
        matroids.add(M)
        yield M

        term = []
        for a in bits:
            if sol[a]:
                term.append(Not(lookup[a]))
            else:
                term.append((lookup[a]))
        term = Or(*term)
        Add(term)
        count += 1
    #print("all_matroids(%d) = %d" % (n, count))


def all_sp_matroids(n):
    # See: Theorem 3.8.1 in [Borovik,Gelfand,White]

    nn = 2*n
    # We use ziporder: (2*i,2*i+1) are the symplectic pairs

    # all admissable sets as bitvectors on 2*n elements
    bits = []
    for vec in numpy.ndindex((3,)*n):
        mask = [0]*nn
        for i in range(n):
            mask[2*i:2*i+2] = [[0,0], [0,1], [1,0]][vec[i]]
        mask = tuple(mask)
        bits.append(mask)
        
    # is this subset an independent set?
    lookup = {} # variables
    for a in bits:
        name = "v"+''.join(str(i) for i in a)
        v = Bool(name)
        lookup[a] = v

    solver = Solver()
    Add = solver.add
    #def Add(term):
    #    print("Add: %s"%(term))
    #    solver.add(term)

    def le(a, b):
        for i in range(nn):
            if a[i]>b[i]:
                return False
        return True

    pairs = [(a,b) for a in bits for b in bits]
    for a in bits:
        if sum(a)==0:
            Add(lookup[a]) # empty set

    # down-closed
    for (a,b) in pairs:
        if a!=b and le(a,b):
            Add( If(lookup[b], lookup[a], True) )

    star = lambda i : [i+1, i-1][i%2]
    for i in range(nn):
        assert 0<=star(i)<nn

    #print("symplectic exchange:")
    for (X,Y) in pairs:
        #print("pair:", X, Y)
        if sum(X) >= sum(Y):
            continue
        # |X| < |Y|
        src = And(lookup[X], lookup[Y]) # "if X and Y are independent sets"
        terms = []
        for i in range(nn):
            if Y[i] and not X[i]:
                Z = list(X)
                Z[i] = 1
                Z = tuple(Z)
                if Z in lookup: # Z admissable
                    terms.append(lookup[Z])
        assert terms
        Z = tuple(max(i,j) for (i,j) in zip(X,Y)) # Z = XuY
        if Z not in lookup:
            #print("XuY inadmissable", X, Y, Z)
            #items = []
            for i in range(nn):
                if Z[i]:
                    continue
                Xi = list(X)
                Xi[i] = 1
                Xi = tuple(Xi)
                if Xi not in lookup:
                    continue
                Yi = [0]*nn
                for j in range(n):
                    if X[j] and not Y[star(j)]:
                        Yi[j] = 1
                assert Yi[i] == 0
                Yi[i] = 1
                Yi = tuple(Yi)
                assert Yi in lookup
                t = And(lookup[Xi], lookup[Yi])
                #print("\t", t)
                terms.append(t)
            #terms.append
        Add( If(src, Or(*terms), True) )

    #return
    
    count = 0
    matroids = set()
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
    
        sol = {a:model.get_interp(lookup[a]) for a in bits}
        found = []
        for (k,v) in sol.items():
            if v:
                found.append(k)
        M = Matroid(nn, found)
        M.check()
        assert M not in matroids
        matroids.add(M)
        yield M

        term = []
        for a in bits:
            if sol[a]:
                term.append(Not(lookup[a]))
            else:
                term.append((lookup[a]))
        term = Or(*term)
        Add(term)
        count += 1
    #print("all_matroids(%d) = %d" % (n, count))


def all_orbits(n):

    from bruhat.gset import Group, Perm

    G = Group.symmetric(n)

    remain = set(all_matroids(n))

    orbits = []
    while remain:
        m = remain.pop()
        orbit = {g*m for g in G}
        remain.difference_update(orbit)
        orbit = list(orbit)
        orbits.append(orbit)
    return orbits


def test_sp():
    for n in [1,2,3,4]:
        count = 0
        found = {i:[] for i in range(n+1)}
        for M in all_sp_matroids(n):
            #print(M.rank, end=' ')
            found[M.rank].append(M)
        print(n, [len(found[i]) for i in range(n+1)])

    
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


