#!/usr/bin/env python

from math import sin, cos, pi

import numpy
from numpy import array

from scipy import optimize 


from bruhat.comonoid import System, dot, tensor

from qumba import construct
from qumba.argv import argv

"""
Here we are doing the complex algebra over the reals.

(a + ib) * (c + id)
= a*c - b*d + i(b*c + a*d)
= [a*c-b*d  b*c+a*d]

so the multiply matrix is:
   ac ad bc bd
  [1, 0, 0, -1], --> real
  [0, 1, 1,  0]] --> imag

"""

def eq(a, b):
    return numpy.allclose(a, b)


one = array([1, 0]).reshape(2,1)
imag = array([0, 1]).reshape(2,1)

# complex multiply
M = array([
    [1, 0, 0, -1],
    [0, 1, 1,  0]])

def mul(a, b):
    return dot(M, tensor(a, b))

assert eq( dot(M, tensor(imag, imag) ), -one )


def test():
    # find a fifth root of unity

    system = System(2)
    z = system.array(2,1)

    z2 = mul(z,z)
    z3 = mul(z2,z)
    z4 = mul(z3,z)
    z5 = mul(z4,z)

    system.add(z5, one)

    #v = system.get_root(x0=[2,3])
    items = system.root()
    v = items[0]

    print(v, type(v))

    theta = 2*pi / 5
    for i in range(5):
        print(sin(i*theta), cos(i*theta))
    



def main_find_1():

    def func(x):
        return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0, 0.5 * (x[1] - x[0])**3 + x[1]]

    sol = optimize.root(func, [0,0])
    print(sol.x)

    code = construct.get_513()
    print(code)

    P = code.get_projector()
    print(16*P)


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






