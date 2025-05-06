#!/usr/bin/env python

"""
brute-force some 2-cocycle's

see:
    https://math.stanford.edu/~conrad/210BPage/handouts/gpext.pdf
    

"""

import numpy

from bruhat.gset import Perm, Group
from bruhat.repr_sage import dixon_irr

from qumba.argv import argv
from qumba.umatrix import Not, And, Var, UMatrix, Solver
from qumba.lin import zeros2




def find_cocycles(G):
    print("find_cocycles", G)

    n = len(G)
    lookup = {g:i for (i,g) in enumerate(G)}
    Cocycle = {(g,h):Var() for g in G for h in G}

    solver = Solver()
    add = solver.add
    for g in G:
     for g1 in G:
      for g2 in G:
        u = Cocycle[g1,g2] + Cocycle[g*g1, g2] + Cocycle[g,g1*g2] + Cocycle[g,g1]
        add(u==0)

    print("\tsolver.check:")
    count = 0
    while 1:
        if str(solver.check()) != "sat":
            break

        model = solver.model()

        result = numpy.zeros((n,n), dtype=int)
        cocycle = {}
        term = []
        for (gh,v) in Cocycle.items():
            g, h = gh
            value = v.get_interp(model)
            value = int(value)
            cocycle[g,h] = value
            result[lookup[g],lookup[h]] = value
            term.append( v == value )
        yield cocycle

        add(Not(And(*term)))

        #break
        count += 1
    print("distinct cocycles:", count)

        
    
def test_extend():

    Z2 = Group.cyclic(2)
    G = Z2*Z2

    G = Group.alternating(4)

    M = [0,1] # the module

    MG = [(m,g) for m in M for g in G]
    n = len(MG)
    lookup = {(m,g):i for (i,(m,g)) in enumerate(MG)}
    found = {}
    for cocycle in find_cocycles(G):
        #mul = {}
        mul = numpy.zeros((n, n), dtype=int)
        for (m,g) in MG:
          for (m1,g1) in MG:
            m2 = (m + m1 + cocycle[g,g1]) % 2
            g2 = g*g1
            #mul[(m,g),(m1,g1)] = (m2,g2)
            mul[lookup[m,g],lookup[m1,g1]] = lookup[m2,g2]
        #print(mul)
        G = Group.from_table(mul)
        G.do_check()

        item = [g.order() for g in G]
        item.sort()
        item = tuple(item)
        found[item] = G
    #print(len(found))
    for item in found:
        print(item)
        G = found[item]
        table = dixon_irr(G)
        print(table)
        print()
        
        

def symplectic_form(n):
    F = zeros2(2*n, 2*n)
    for i in range(n):
        F[2*i:2*i+2, 2*i:2*i+2] = [[0,1],[1,0]]
    return F


def test_weil():
    """
    The Weil representation in characteristic two
    Shamgar Gurevicha,∗, Ronny Hadani
    2012
    section 0.2

    see also: bruhat/heisenberg.py

    """

    n = 1
    nn = 2*n

    omega = symplectic_form(n)
    print(omega)

    # beta(u,v) - beta(v,u) = 2*omega(u,v) in 2Z/4Z






if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "test"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))




