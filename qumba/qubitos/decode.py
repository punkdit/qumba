#!/usr/bin/env python

"""
experimenting with _simulating qec circuits .
previous version: wstim.py 

"""

if __name__ == "__main__":
    # much slower with threading so we disable this
    import os
    os.environ["OMP_NUM_THREADS"] = "1"


from functools import reduce
from operator import add, mul, matmul
from random import choice, seed

import numpy

from qumba.smap import SMap
from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba import construct
#from qumba import decode
from qumba import lin
from qumba.qcode import strop, SymplecticSpace
from qumba import dense 

from qumba.pauli import Pauli


class Decoder:
    def __init__(self, code):
        code = code.to_css()
        self.code = code

    def get_T(self, syndrome):
        # bitflip X-type errors, frustrate Hz checks, and produce Tx
        code = self.code
        n = code.n
        Hz = code.Hz
        Tx = code.Tx
        T = syndrome * Tx
        return T

    def decode(self, p, err_op, verbose=False, **kw):
        return None


class SimpleDecoder(Decoder):
    """
    Simple (&slow) optimal decoder that sums probabilities over cosets.
    """
    def __init__(self, code):
        Decoder.__init__(self, code)
        self.all_Lx = list(code.Lx.span())
        self.all_Hx = list(code.Hx.span())
        self.code = code

    def get_dist(self, p, T):
        "distribution over logical operators"
        code = self.code
        dist = []
        sr = 0.
        n = code.n
        for l_op in self.all_Lx:
            r = 0.
            T1 = l_op + T
            for s_op in self.all_Hx:
                T2 = s_op + T1
                d = T2.sum()
                r += (1-p)**(n-d)*p**d
            sr += r
            dist.append(r)
        dist = [r/sr for r in dist]
        return dist

    def decode(self, p, syndrome, verbose=False, **kw):
        T = self.get_T(syndrome)
        dist = self.get_dist(p, T)
        p1 = max(dist)
        idx = dist.index(p1)
        l_op = self.all_Lx[idx]
        op = l_op+T
        return op





if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

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

