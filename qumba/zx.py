#!/usr/bin/env python

import pyzx as zx

from qumba.argv import argv



def mulclose(gen, verbose=False, maxsize=None):
    els = list(gen)
    bdy = list(els)
    changed = True
    while bdy:
        if verbose:
            print(len(els), end=" ", flush=True)
        _bdy = []
        for A in gen:
            for B in bdy:
                C = A*B
                if C not in els:
                    els.append(C)
                    _bdy.append(C)
                    if maxsize and len(els)>=maxsize:
                        return els
        bdy = _bdy
    if verbose:
        print()
    return els


class Circuit:
    def __init__(self, n, *args):
        c = zx.Circuit(n)
        for arg in args:
            #print("add_gate", arg)
            c.add_gate(*arg)
        self.c = c
        self.args = args
        self.n = n

    def __eq__(self, other):
        assert self.n == other.n
        c, d = self.c, other.c
        return c.verify_equality(d, up_to_global_phase=False)

    def __str__(self):
        return "Circuit(%d, %s)"%(self.n, self.args)
    __repr__ = __str__

    def __mul__(self, other):
        assert self.n == other.n
        args = self.args + other.args
        return Circuit(self.n, *args)




class Clifford:
    def __init__(self, n):
        self.n = n

    def I(self):
        return Circuit(self.n)

    def X(self, i=0):
        return Circuit(self.n, ("NOT", i))

    def Z(self, i=0):
        return Circuit(self.n, ("Z", i))

    def H(self, i=0):
        return Circuit(self.n, ("H", i))

    def S(self, i=0):
        return Circuit(self.n, ("S", i))

    def CNOT(self, i=0, j=1):
        return Circuit(self.n, ("CNOT", i, j))

    def CZ(self, i=0, j=1):
        return Circuit(self.n, ("CZ", i, j))




def test():
    c = Clifford(1)
    I, X, Z, H, S = c.I(), c.X(), c.Z(), c.H(), c.S()

    assert X != Z
    assert X*X == I

    G = mulclose([S, H], maxsize=20)
    for g in G:
        graph = g.c.to_graph()
        zx.to_clifford_normal_form_graph(graph)
        #print(graph)
        #print(dir(graph))
        #print(graph.__class__)
        #break

    #G = mulclose([S, H], verbose=True)
    #assert len(G) == 192

    c = Clifford(2)
    I, X0, Z0, H0, S0 = c.I(), c.X(0), c.Z(0), c.H(0), c.S(0)
    X1, Z1, H1, S1 = c.X(1), c.Z(1), c.H(1), c.S(1)
    CX, CZ = c.CNOT, c.CZ
    CNOT = c.CNOT(0, 1)

    assert CNOT != I
    assert CNOT*CNOT == I

    #G = mulclose([S0, S1, H0, H1, CNOT], verbose=True)
    #print(len(G))

    # from Bravyi, Maslov:
    lhs = CX(1,0)*H0*CX(1,0)
    rhs = CZ(0,1)*H0*CZ(0,1)*Z1
    assert lhs == rhs




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


