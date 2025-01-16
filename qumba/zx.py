#!/usr/bin/env python

import pyzx as zx

from qumba.argv import argv


class Circuit:
    def __init__(self, n, *args):
        c = zx.Circuit(n)
        for arg in args:
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


class Clifford:
    def __init__(self, n):
        self.n = n

    def X(self, i=0):
        return Circuit(self.n, ("NOT", i))

    def Z(self, i=0):
        return Circuit(self.n, ("Z", i))

    def H(self, i=0):
        return Circuit(self.n, ("H", i))

    def S(self, i=0):
        return Circuit(self.n, ("S", i))




def test():
    c = Clifford(1)
    X, Z, H, S = c.X(), c.Z(), c.H(), c.S()
    print(X)

    assert X != Z


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


