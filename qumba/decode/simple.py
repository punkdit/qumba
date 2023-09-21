#!/usr/bin/env python3

from random import *

import numpy
import numpy.random as ra



from qumba import solve
from qumba.solve import (
    pop2, zeros2, dot2, array2, eq2, rand2, binomial2,
    randexpo, shortstr, shortstrx)
from qumba.decode.dynamic import Tanner

from qumba.argv import argv

strop = solve.shortstr

def rowspan(H):
    m, n = H.shape
    vs = []
    for idxs in numpy.ndindex((2,)*m):
        vs.append(dot2(idxs, H))
    return vs


class Decoder(object):
    def __init__(self, code):
        self.code = code

    def get_T(self, err_op):
        # bitflip, x-type errors, hit Hz ops, produce Tx
        code = self.code
        n = code.n
        T = zeros2(n)
        Hz = code.Hz
        Tx = code.Tx
        m = Hz.shape[0]
        for i in range(m):
            if dot2(err_op, Hz[i]):
                T += Tx[i]
        T %= 2
        return T

    def decode(self, p, err_op, verbose=False, **kw):
        return None


class SimpleDecoder(Decoder): # XXX broken XXX
    def __init__(self, code):
        Decoder.__init__(self, code)
        self.all_Lx = rowspan(code.Lx)
        self.all_Hx = rowspan(code.Hx)
        self.code = code

    def get_dist(self, p, T):
        "distribution over logical operators"
        code = self.code
        dist = []
        sr = 0.
        n = code.n
        for l_op in self.all_Lx:
            #print "l_op:", shortstr(l_op)
            r = 0.
            T1 = l_op + T
            for s_op in self.all_Hx:
                T2 = (s_op + T1)%2
                d = T2.sum()
                #print shortstr(T2), d
                #print d,
                r += (1-p)**(n-d)*p**d
            #print
            sr += r
            dist.append(r)
        dist = [r//sr for r in dist]
        return dist

    def decode(self, p, err_op, verbose=False, **kw):
        #print "decode:"
        #print shortstr(err_op)
        T = self.get_T(err_op)
        #print shortstr(T)
        dist = self.get_dist(p, T)
        #print dist
        p1 = max(dist)
        idx = dist.index(p1)
        l_op = self.all_Lx[idx]
        #op = (l_op+T+err_op)%2
        op = (l_op+T)%2
        return op



