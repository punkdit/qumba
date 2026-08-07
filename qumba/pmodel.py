#!/usr/bin/env python

from functools import reduce
from operator import matmul, add
#import cmath

from sage import all_cmdline as sage

from qumba.matrix_sage import Matrix

import numpy
#from numpy import linalg
#from numpy import random
#from numpy import exp, pi, cos, arccos, sin

from qumba.argv import argv


class Table:
    def __init__(self, names, values):
        self.names = list(names)
        n = len(names)
        assert len(values) == 2**n
        self.values = list(values)
        self.n = n
        lookup = {}
        for i,bits in enumerate(numpy.ndindex((2,)*n)):
            lookup[bits] = values[i]
        self.lookup = lookup

    def __call__(self, *key):
        assert len(key) == self.n
        assert key in self.lookup
        return self.lookup[key]

    def __str__(self):
        return "Table(%s, %s)"%(self.names, self.lookup)
    __repr__ = __str__

    def sum(self):
        r = 0
        for v in self.lookup.values():
            r = r+v
        return r

#    def marginal(self, m_names):
#        values = []
#        for bits in enumerate(




def test():

    R = sage.PolynomialRing(sage.QQ, list("efr"))
    e, f, r = R.gens()

    one = R.one()
    half = one/2

    #px = Matrix(R, [[1-e, e], [e, 1-e]])
    #print(px)

    #py = Matrix(R, [[1-f, f], [f, 1-f]])
    #print(py)

    px = Table("x1 x0".split(), [1-e, e, e, 1-e])
    assert px(0,1) == e

    py = Table("y0 x0".split(), [1-f, f, f, 1-f])
    assert py(0, 0) == 1-f

    def get_alpha(i):
        if i == 0:
            # p(x0, y0)
            values = [half*py(0,0), half*py(0,1), half*py(1,0), half*py(1,1) ]
            return Table("x0 y0".split(), values) # <--- return

        assert i>0
        prev = get_alpha(i-1)
        names = ["x%d"%i] + ["y%d"%j for j in reversed(range(i+1))]
        values = []
        for xi in (0,1):
          for yis in numpy.ndindex((2,)*(i+1)): # yi,...,y0
            #print(xi, yis)
            yi = yis[0]
            v = 0
            for xi1 in (0,1):
                v += py(yi, xi) * px(xi, xi1) * prev(xi1, *(yis[1:]))
            values.append(v)
        #print(names, len(values))
        p = Table(names, values)
        
        return p

    a0 = get_alpha(0)
    assert a0.sum() == 1

    a1 = get_alpha(1)
    assert a1.sum() == 1

    ev = fv = 0.001
    alphas = []
    #bits = (0,)
    for i in range(6):
        ai = get_alpha(i)
        assert ai.sum() == 1
        alphas.append(ai)

        # find marginal over y's
        values = []
        for bits in numpy.ndindex((2,)*(i+1)):
            v = ai(*((0,)+bits)) + ai(*((1,)+bits))
            values.append(v)
        ys = Table(ai.names[1:], values)

        ybits = (0,)*i + (0,)
        print("i=%d"%i, ybits, ai.names)
        denom = ys(*ybits)
        #print("\t x%d=0:"%i, (ai(0,*ybits)/denom).subs(e=ev, f=fv))
        #print("\t x%d=1:"%i, (ai(1,*ybits)/denom).subs(e=ev, f=fv))
        poly = ai(1,*ybits)/denom
        print("\t x%d=1:"%i)
        top = poly.numerator()
        bot = poly.denominator()
        for weight,mono in top:
            print(mono.degree(), end=' ')
        print()

        bits = (0,) + bits

    #for i in [0,1]:
    #    print(a1(0,i,i).subs(e=ev, f=fv), "--", a1(1,i,i).subs(e=ev, f=fv))




if __name__ == "__main__":

#    numpy.set_printoptions(
#        precision=4, threshold=1024, suppress=True, 
#        formatter={'float': '{:0.4f}'.format}, linewidth=200)

    from random import seed
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
    print("\nOK! finished in %.3f seconds\n"%t)



