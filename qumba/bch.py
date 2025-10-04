#!/usr/bin/env python

"""

"""


import warnings
warnings.filterwarnings('ignore')



from random import shuffle
from functools import reduce
from operator import add, mul

import numpy


from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.qcode import QCode


def main_sd():
    # find selfdual quantum BCH codes
    # TODO: search over primitive_root argument of BCHCode  ???

    n = argv.get("n", 31)
    print("n =", n)

    d = argv.get("d", 3)

    found = set()
    for d in range(d, n+1):
      print("d =", d)
      for offset in range(1,n+1):
        C = sage.codes.BCHCode(sage.GF(2), n, d, offset=offset)

        vs = C.basis()
    
        vs = list(vs)
        H = Matrix(vs)
        HHt = H * H.t
        if HHt.max() != 0 or len(H)==0:
            continue

        code = QCode.build_css(H,H)
        if code.d < 3:
            print(".", end='', flush=True)
            continue
        key = str(C.weight_enumerator())
        print(code, "*" if key in found else "")

        if argv.store_db and key not in found:
            from qumba import db
            db.add(code)

        found.add(key)


def main():

    # find CSS codes based on BCH codes ... 

    n = argv.get("n", 31)
    print("n =", n)

    d = argv.get("d", 3)
    jump_size = argv.get("jump_size", 1)

    found = set()
    for d in range(d, n+1):
      print("\nd =", d)
      for offset in range(1,n+1):
       #for jump_size in range(1, n+1):
        try:
            C = sage.codes.BCHCode(sage.GF(2), n, d, offset=offset, jump_size=jump_size)
            key = str(C.weight_enumerator())
        except ValueError:
            continue
        if key in found:
            continue
        found.add(key)

        vs = C.basis()
        vs = list(vs)
        if not vs:
            continue

        H = Matrix(vs)
        K = H.kernel()
        #print(H.shape, K.shape)
        U = H.intersect(K)
        #print("\t", U.shape)
        if not len(U):
            print("/", end='', flush=True)
            continue

        assert (H*U.t).max() == 0

        #HHt = H * H.t
        #if HHt.max() != 0 or len(H)==0:
        #    continue

        code = QCode.build_css(H,U)
        if code.k == 0:
            print("0", end='', flush=True)
            continue
        if code.d < 3:
            print("d", end='', flush=True)
            continue

        desc = ("sd" if code.is_selfdual() else "")
        print()
        print(code, desc)




if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next()
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
        main()


    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")


