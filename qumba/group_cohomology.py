#!/usr/bin/env python

"""
brute-force some 2-cocycle's

see:
    https://math.stanford.edu/~conrad/210BPage/handouts/gpext.pdf
    

"""

from random import choice

import numpy

from bruhat.gset import Perm, Group, mulclose, mulclose_hom
from bruhat.repr_sage import dixon_irr

from qumba.argv import argv
from qumba.smap import SMap
from qumba.umatrix import Not, And, Var, UMatrix, Solver
from qumba.lin import zeros2, array2
from qumba.matrix import Matrix




def find_cocycles(G, dim, action=(lambda g,m:m)):
    print("find_cocycles", G)

    n = len(G)
    lookup = {g:i for (i,g) in enumerate(G)}
#    Cocycle = {(g,h):[Var() for i in range(dim)] for g in G for h in G}
    Cocycle = {(g,h):UMatrix.unknown(dim,1) for g in G for h in G}

    solver = Solver()
    add = solver.add
    for g in G:
     for g1 in G:
      for g2 in G:
        u = (action(g,Cocycle[g1,g2]) + Cocycle[g*g1, g2]
            + Cocycle[g,g1*g2] + Cocycle[g,g1])
        add(u==0)
        #for i in range(dim):
            #u = (action(g,Cocycle[g1,g2][i]) + Cocycle[g*g1, g2][i]
            #    + Cocycle[g,g1*g2][i] + Cocycle[g,g1][i])
            #add(u==0)

    # redundant but helps..
    e = G.identity
    for g in G:
        add(Cocycle[e,g] == Cocycle[e,e])
        #for i in range(dim):
        #    add(Cocycle[e,g][i] == Cocycle[e,e][i])

    print("solver.check()")
    count = 0
    while 1:
        if str(solver.check()) != "sat":
            break

        model = solver.model()

        #result = numpy.zeros((n,n,dim), dtype=int)
        cocycle = {}
        term = []
        for (gh,u) in Cocycle.items():
            g, h = gh
            #values = [v.get_interp(model) for v in vs]
            #values = [int(value) for value in values]
            #values = Matrix(values)
            m = u.get_interp(model)
            cocycle[g,h] = m
            #result[lookup[g],lookup[h]] = values
            #for i in range(dim):
            #    term.append( vs[i] == values[i] )
            #print( u != m )
            term.append( u == m )
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

    action = lambda g,m:m

    gens = []
    Z2 = Group.cyclic(2)
    if name == "Z2Z2Z2":
        G = Z2*Z2*Z2
    elif name == "Z2Z2":
        G = Z2*Z2
        #gens = find_gens(G)
        gens = [a for a in G if a.order() == 2]
    elif name == "S3":
        G = Group.symmetric(3)
    elif name == "S4":
        G = Group.symmetric(4)
    elif name == "S5":
        G = Group.symmetric(5)
    elif name == "S6":
        G = Group.symmetric(6)
    elif name == "A4":
        G = Group.alternating(4)
    elif name == "D8":
        G = Group.dihedral(8)
    elif name == "Cliff1":
        G = Group.symmetric(3)
    else:
        print("name %r not found"%name)
        return

    # the module
    dim = argv.get("dim", 1)
    M = [] # the module
    for bits in numpy.ndindex( (2,)*dim ):
        v = Matrix(bits)
        v = v.reshape(dim, 1)
        M.append(v)

    MG = [(m,g) for m in M for g in G]
    n = len(MG)

    lookup = {(m,g):i for (i,(m,g)) in enumerate(MG)}
    found = {}
    src = G
    pairs = [(g,h) for g in G for h in G]
    uniq = set()
    for i, cocycle in enumerate(find_cocycles(src, dim, action)):
        item = tuple(cocycle[pair] for pair in pairs)
        #print(i, item)
        assert item not in uniq
        uniq.add(item)
        #mul = {}
        mul = numpy.zeros((n, n), dtype=int)
        for (m,g) in MG:
          for (m1,g1) in MG:
            m2 = m + m1 + cocycle[g,g1]
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
        print(G)
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


def test_clifford():

    G = Group.symmetric(3)

    # the module
    dim = 2
    M = [] # the module
    for bits in numpy.ndindex( (2,)*dim ):
        v = Matrix(bits)
        v = v.reshape(dim, 1)
        M.append(v)

    MG = [(m,g) for m in M for g in G]
    n = len(MG)

    lgen = [Perm([1,0,2]), Perm([0,2,1])]
    rgen = [Matrix([[1,0],[1,1]]), Matrix([[0,1],[1,0]])]
    hom = mulclose_hom(lgen, rgen)

    def action(g, m):
        m = hom[g]*m
        return m

    lookup = {(m,g):i for (i,(m,g)) in enumerate(MG)}
    found = {}
    src = G
    pairs = [(g,h) for g in G for h in G]
    uniq = set()
    for i, cocycle in enumerate(find_cocycles(src, dim, action)):
        item = tuple(cocycle[pair] for pair in pairs)
        #print(i, item)
        assert item not in uniq
        uniq.add(item)
        #mul = {}
        mul = numpy.zeros((n, n), dtype=int)
        for (m,g) in MG:
          for (m1,g1) in MG:
            m2 = m + action(g,m1) + cocycle[g,g1]
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
        print(G)
        print(key)
        table = dixon_irr(G)
        print(table)
        table.check_complete()
        print()
        #break


    for key,items in found.items():
        print(key, len(items))
        continue
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
                u = cocycle[a,b]
                #print(u, u.shape)
                idx, jdx = u[0,0], u[1,0]
                s = ".1"[idx] + ".1"[jdx]
                smap[i,3*j+col] = s
            col = smap.cols + 3

        print(smap)

        
        

def test_weil():
    """
    The Weil representation in characteristic two
    Shamgar Gurevicha,∗, Ronny Hadani
    2012
    section 0.2

    see also: bruhat/heisenberg.py

    """

    n = argv.get("n", 1)
    nn = 2*n

    from qumba.symplectic import SymplecticSpace
    space = SymplecticSpace(n)
    F = space.F # omega

    V = []
    basis = []
    for idxs in numpy.ndindex((2,)*nn):
        v = Matrix(array2(idxs)).reshape(nn, 1)
        V.append(v)
        if v.sum() == 1:
            basis.append(v)
        elif v.sum() == 0:
            zero = v
    assert len(basis) == nn

    omega = {}
    for u in V:
      for v in V:
        w = (u.t * F * v)
        assert w.shape == (1,1)
        w = int(w[0,0])
        omega[u,v] = w

    # beta is a 2-cocycle for constructing the Heisenberg group
    # as a central extension:
    # Z_4 >---> H(V) -->> V

    beta = {}
    for u in V:
      u0 = u[::2, :]
      #print(u.t, u.shape, u0.t, u0.shape)
      for v in V:
        v0 = v[1::2, :]
        uv = u0.t*v0
        beta[u,v] = 2*int(uv[0,0])
        #print("\t", v.t, v.shape, v0.t, v0.shape, "=", uv, uv.shape)

    # check that
    # beta(u,v) - beta(v,u) = 2*omega(u,v) in 2Z/4Z
    for u in V:
      for v in V:
        lhs = (beta[u,v] - beta[v,u]) % 4
        rhs = 2*omega[u,v]
        #print("%s:%s"%(lhs, rhs), end=" ")
        assert lhs==rhs

    # todo: construct H(V) using beta & check it..

    S, H, CX = space.S, space.H, space.CX
    gen = []
    for i in range(n):
        gen.append(S(i))
        gen.append(H(i))
        for j in range(n):
            if i!=j:
                gen.append(CX(i,j))

    G = mulclose(gen)
    print(len(G))
    
    ASp = [] # here goes..
    for g in G:
        # alpha : V --> Z/4Z
        for bits in numpy.ndindex((4,)*nn):
            alpha = {zero:0}
            for i,bit in enumerate(bits):
                alpha[basis[i]] = bit
            succ = True
            gen = set(alpha.keys())
            while succ and len(gen) < 2**nn:
                for u in list(gen):
                  for v in basis:
                    uv = u+v
                    value = (beta[g*u,g*v] - beta[u,v] + alpha[u] + alpha[v])%4
                    if uv in alpha and alpha[uv] != value:
                        succ = False
                        break
                    elif uv in alpha:
                        continue
                    assert u in alpha
                    assert v in alpha
                    alpha[uv] = value
                    gen.add(uv)
                  if not succ:
                    break
            if succ:
                ASp.append((g, alpha))
                #print([alpha[basis[i]] for i in range(nn)])
        print(".", end="", flush=True)
    print()

    print(len(ASp))
    #for (g,alpha) in ASp:
    for trial in range(100):
        (g, alpha) = choice(ASp)
        for u in V:
          for v in V:
            lhs = (alpha[u+v] - alpha[u] - alpha[v]) % 4
            rhs = (beta[g*u,g*v] - beta[u,v])%4
            assert lhs==rhs

    for trial in range(100):
        (g, a) = choice(ASp)
        (h, b) = choice(ASp)

        gh = g*h
        ab = {v:(a[h*v]+b[v])%4 for v in V}

        for u in V:
          for v in V:
            lhs = (ab[u+v] - ab[u] - ab[v]) % 4
            rhs = (beta[gh*u,gh*v] - beta[u,v])%4
            assert lhs==rhs

        assert (gh,ab) in ASp
        


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




