#!/usr/bin/env python

#import numpy
from qumba.solve import zeros2

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.umatrix import UMatrix, Solver, PbEq
from qumba.cyclic import get_cyclic_perms



class Algebra:
    def __init__(self, mul, unit):
        self.mul = mul
        self.unit = unit


def main():
    #n = argv.get("n", 12)
    for n in range(2, 20):
        count = find(n)
        gens = get_cyclic_perms(n)
        #print("gens:", len(gens))
        assert count == len(gens)
        print(n, len(gens))


def find(n):
    #print("find(%s)"%n)
    n2 = n**2

    mul = zeros2(n, n2)
    for i in range(n):
      for j in range(n):
        mul[(i+j)%n, i + n*j] = 1 # Z/n group algebra
    mul = Matrix(mul)
    #print(mul)

    unit = [0]*n
    unit[0] = 1
    unit = Matrix(unit).reshape(n,1)

    I = Matrix.get_identity(n)

    swap = zeros2(n2, n2)
    for i in range(n):
      for j in range(n):
        swap[j + n*i, i + n*j] = 1
    swap = Matrix(swap)

    # unital
    assert mul * (unit@I) == I
    assert mul * (I@unit) == I

    # assoc
    assert mul*(mul@I) == mul*(I@mul)

    # comm
    assert mul*swap == mul

    solver = Solver()
    add = solver.add

    P = UMatrix.unknown(n,n)

    add( P*unit == unit )
    add( mul*(P@P) == P*mul )
    add( P*P.t == I )
    for i in range(n):
        add( PbEq([(P[i,j].get(),True) for j in range(n)], 1) )

    #print("solver", end='', flush=True)
    count = 0
    while 1:
        result = solver.check()
        if str(result) != "sat":
            break

        model = solver.model()
        p = P.get_interp(model)
        add(P != p)
        #print(".", end="", flush=True)
        count += 1

    return count




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





