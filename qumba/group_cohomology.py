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
from qumba.smap import SMap
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

    # redundant but helps..
    e = G.identity
    for g in G:
        add(Cocycle[e,g] == Cocycle[e,e])

    print("solver.check()")
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


def find_gens(G):
    gens = []
    for a in G:
      for b in G:
        if a.order()==2 and b.order()==2 and len(Group.generate([a,b]))==len(G):
            return [a,b]
    assert 0

    
def test_extend():

    name = argv.next()

    gens = []
    Z2 = Group.cyclic(2)
    if name == "Z2Z2Z2":
        G = Z2*Z2*Z2
    elif name == "Z2Z2":
        G = Z2*Z2
        #gens = find_gens(G)
        gens = [a for a in G if a.order() == 2]
    elif name == "S4":
        G = Group.symmetric(4)
    elif name == "A4":
        G = Group.alternating(4)
    elif name == "D8":
        G = Group.dihedral(8)
    else:
        print("name=?")
        return

    M = [0,1] # the module

    MG = [(m,g) for m in M for g in G]
    n = len(MG)
    lookup = {(m,g):i for (i,(m,g)) in enumerate(MG)}
    found = {}
    src = G
    pairs = [(g,h) for g in G for h in G]
    uniq = set()
    for i, cocycle in enumerate(find_cocycles(src)):
        item = tuple(cocycle[pair] for pair in pairs)
        #print(i, item)
        assert item not in uniq
        uniq.add(item)
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
        G.cocycle = dict(cocycle)
        key = [g.order() for g in G]
        key.sort()
        key = tuple(key)
        #found.setdefault(key, []).append(G)
        if key in found:
            found[key].append(G)
            continue
        found[key] = [G]
        #found[key] = G
        print(key)
        table = dixon_irr(G)
        print(table)
        table.check_complete()
        print()

    if len(src) > 4:
        return

    for key,items in found.items():
        print(key, len(items))
        #for G in items:
        #    table = dixon_irr(G)
        #    print(table)
        #print()
        smap = SMap()
        col = 0
        for G in items:
            cocycle = G.cocycle
            for i,a in enumerate(src):
              for j,b in enumerate(src):
                s = ".1"[cocycle[a,b]]
                smap[i,j+col] = s
            col += len(gens)+2

        print(smap)

        
        

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




