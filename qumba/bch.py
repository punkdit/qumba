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


def main():

    n = 49
    n = 85 # d=8 gives [85,53] BCH code

    found = []
    for d in range(1, n+1):
        C = sage.codes.BCHCode(sage.GF(2), n, d, offset=1)
        #if C in found:
        #    continue
        #found.append(C)
        print(C)

        vs = C.basis()
    
        vs = list(vs)
        H = Matrix(vs)
        HHt = H * H.t
        if HHt.max() == 0:
            print("found")
    

#        print(H, H.shape)
#        print()
#
#        #w = H.get_wenum()
#        #print(w)
#
#        K = H.kernel()
#        print(K, K.shape)
#
#        break
    


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


