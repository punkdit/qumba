#!/usr/bin/env python

"""
try to implement section "Linear codes" in:
    https://arxiv.org/abs/quant-ph/9703048
    Nonbinary quantum codes
    Eric M. Rains

"""

from functools import reduce, cache
from operator import add, matmul, mul

import numpy

from qumba.qcode import QCode, SymplecticSpace, Matrix, fromstr, shortstr, strop
from qumba.matrix import scalar
from qumba.action import mulclose, Group, mulclose_find
from qumba.util import allperms, all_subsets
from qumba import equ
from qumba import construct
from qumba import autos
from qumba.smap import SMap
from qumba.unwrap import unwrap
from qumba.argv import argv
from qumba.umatrix import UMatrix, Solver, Or

def find_algebras(dim):

    mul = UMatrix.unknown(dim, dim*dim)
    unit = UMatrix.unknown(dim, 1)
    I = UMatrix.identity(dim)

    solver = Solver()
    Add = solver.add

    # unital
    Add(mul*(unit@I) == I)
    Add(mul*(I@unit) == I)

    # assoc
    Add(mul*(I@mul) == mul*(mul@I))

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        a = mul.get_interp(model)
        i = unit.get_interp(model)

        yield (i, a)

        Add( Or( (mul!=a) , (unit!=i) ) )



def find_modules(dim, unit, mul):

    n = len(unit)
    act = UMatrix.unknown(dim, n*dim)

    Id = UMatrix.identity(dim)
    In = UMatrix.identity(n)

    solver = Solver()
    Add = solver.add

    # unital
    Add(act*(unit@Id) == Id)

    # assoc
    Add(act*(In@act) == act*(mul@Id))

    while 1:
        result = solver.check()
        if str(result) != "sat":
            return

        model = solver.model()
        a = act.get_interp(model)

        yield a

        Add( act != a )




def main_algebras():

    dim = 2
    count = 0
    for (unit, mul) in find_algebras(dim):
        #print("unit =")
        #print(unit)
        #print("mul =")
        #print(mul)
#        dount = 0
#        for act in find_modules(2, unit, mul):
#            #print("act =")
#            #print(act)
#            dount += 1
#        print("[%d]"%dount, end="", flush=True)
        count += 1
    print()
    print("found:", count)






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




