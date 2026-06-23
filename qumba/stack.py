#!/usr/bin/env python

"""


"""


import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add, mul, lshift
from random import random, randint, choice, shuffle

import numpy

from qumba.argv import argv
from qumba.qcode import strop, QCode, SymplecticSpace
from qumba.csscode import CSSCode
from qumba import construct 
from qumba.matrix import Matrix
from qumba.lin import shortstr



def test():

    lookup = {}

    L = 3
    for i in range(L):
      for j in range(L):
        for k in range(L):
            print(i,j,k)


def test_14():
    code = construct.get_15_1_3()
    code = code.shorten(0)
    print(code)
    code = code.to_css()
    code.bz_distance()
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)

    

if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    fn = eval(name)

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            fn()
        profiler.print()

    else:
        fn()


    t = time() - start_time
    print("\nOK! finished in %.3f seconds\n"%t)


