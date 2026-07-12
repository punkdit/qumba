#!/usr/bin/env python

"""
F_q Pauli and Cliff'ord group



"""

from operator import mul, matmul, add
from functools import reduce
from functools import cache

from sage.all_cmdline import (FiniteField, CyclotomicField, latex, block_diagonal_matrix,
    PolynomialRing)
from sage import all_cmdline as sage

from bruhat.gset import Perm, Group

from qumba.action import mulclose, mulclose_names, mulclose_find
from qumba.matrix_sage import Matrix
from qumba.argv import argv


def test():

    q = argv.get("q", 7)

    Q = q*4
    K = CyclotomicField(Q)
    print(K)

    z, = K.gens()
    wq = z**4
    J = z**q
    assert J**2 == -1

    if q==3:
        rq = 2*z - z**3
    elif q==5:
        rq = 1 + 2*z**4 - 2*z**6
    elif q==7:
        rq = z + z**3 - z**5 + z**9 - z**11 - z**13
    else:
        fail
    assert rq**2 == q

    one = K.one()
    for i in range(1,q):
        assert wq**i != one
    assert wq**q == one
    half = one//2
    irq = one/rq

    I = Matrix.identity(K, q)

    rows = []
    for i in range(q):
        row = [0]*q
        row[(i+1)%q] = 1
        rows.append(row)
    X = Matrix(K, rows)

    rows = [[0]*q for i in range(q)]
    for i in range(q):
        rows[i][i] = wq**i
    Z = Matrix(K, rows)

    Pauli = mulclose([X, Z])
    assert len(Pauli) == q**3

    found = set()
    for g in Pauli:
        if g==I:
            continue
        evs = g.eigenvectors()
        #print(g)
        for val,v,dim in evs:
            #print("\t", val, dim)
            #print(v, v.shape)
            if val!=1:
                continue
            assert dim==1
            r = (v.d * v)[0,0]
            if r==q:
                v = irq*v
            else:
                assert r==1
            #print(v.t, r)
            found.add(v)
    
    found = list(found)
    print("found:", len(found))

    #return

    lookup = {}
    for (i,v) in enumerate(found):
        for j in range(Q):
            w = (z**j)*v
            #if w in lookup:
            #    assert lookup[w] == i
            lookup[w] = i
    #for v in lookup:
    #    if v[0,0]==0 and v[1,0] ==0:
    #        print(v.t)

    def make_group(ops):
        gen = []
        for g in ops:
            perm = []
            for v in found:
                w = g*v
                if w not in lookup:
                    print(g)
                    print(v.t, "-->", w.t)
                    assert 0
                assert w in lookup
                j = lookup[w]
                perm.append(j)
            perm = Perm(perm)
            gen.append(perm)
        group = Group.generate(gen, verbose=True)
        return group

    pauli = make_group([X, Z])
    #print(pauli.structure_description())
    assert len(pauli) == q**2

    rows = [[wq**(i*j) for i in range(q)] for j in range(q)]
    H = irq*Matrix(K, rows)
    assert H.order() == 4
    assert (H.d)*H == I
    assert ~H == H.d

    for g in Pauli:
        assert (~H)*g*H in Pauli

    qhalf = 2**(q-2)
    assert (2*qhalf)%q == 1
    
    rows = [[0]*q for i in range(q)]
    for i in range(q):
        rows[i][i] = wq**(i*(i+1)*qhalf)
    S = Matrix(K, rows)

    for g in Pauli:
        assert (~S)*g*S in Pauli
    assert ~S == S.d
    
    cliff = make_group([X, S, H])
    print(len(cliff))
    print(cliff.structure_description())

    # yes, works:
    #for g in Cliff:
    #    for h in Pauli:
    #        assert (g.d)*h*g in Pauli
    assert type(Pauli) is set
    
    Cliff = mulclose([X, S, H], verbose=True)
    print(len(Cliff))

    print("-I in Cliff:", -I in Cliff)
    print("JI in Cliff:", J*I in Cliff)



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




