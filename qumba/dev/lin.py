#!/usr/bin/env python

"""
linear operators, ...
"""

from functools import reduce, cache
from operator import add, matmul, mul

import numpy

from qumba.matrix import Matrix, scalar
from qumba.smap import SMap
from qumba.argv import argv


class Space(object):
    def __init__(self, n, name):
        assert type(n) is int
        self.n = n
        self.name = name

    def __str__(self):
        return self.name


def main():
    V = Space(2, "V")
    print(V)


if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))






