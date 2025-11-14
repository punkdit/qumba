#!/usr/bin/env python

from random import shuffle
from math import sin, cos, pi
from functools import cache, reduce
from operator import add

import numpy

from qumba.action import mulclose
from qumba.symplectic import SymplecticSpace
from qumba.matrix import Matrix
from qumba.qcode import QCode, fromstr, strop
from qumba.smap import SMap
from qumba import lin
from qumba.argv import argv
from qumba.util import binomial, factorial
from qumba import construct







class Functor:
    def __init__(self, H):
        assert isinstance(H, Matrix)
        m, nn = H.shape
        assert nn%2 == 0
        n = nn//2
        H = H.normal_form()
        self.H = H
        self.H2 = H.reshape(m, n, 2)
        self.m = m
        self.n = n
        self.nn = nn
        self.shape = m, nn

    def __eq__(self, other):
        return self.H == other.H

    def __hash__(self):
        return hash(self.H)

    def __str__(self):
        return (str(self.H) or "[]") + " " + str(self.shape)

    def upper(self, idxs):
        #H2 = self.H2[:, idxs, :]
        #H = H2.reshape(self.m, 2*self.n)
        mask = lin.zeros2(1,self.nn)
        mask[0,[2*i for i in idxs]] = 1
        mask[0,[2*i+1 for i in idxs]] = 1
        #print(mask)
        A = self.H.A * mask
        H = Matrix(A)
        return Functor(H)

    def lower(self, idxs):
        A = lin.zeros2(2*len(idxs), self.nn)
        I = lin.array2([[1,0],[0,1]])
        for i,ii in enumerate(idxs):
            A[2*i:2*i+2, 2*ii:2*ii+2] = I
        A = lin.intersect(self.H.A, A)
        H = Matrix(A)
        return Functor(H)




def main():

    code = construct.get_713()

    F = Functor(code.H)

    print(F)

    print()
    print(F.lower([1,2,3]))

    print()
    print(F.upper([1,2,3]))





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


