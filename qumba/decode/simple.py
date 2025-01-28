#!/usr/bin/env python3

from random import *

import numpy
import numpy.random as ra



from qumba import lin
from qumba.lin import (
    pop2, zeros2, dot2, array2, eq2, rand2, binomial2,
    randexpo, shortstr, shortstrx)
from qumba.decode.dynamic import Tanner

from qumba.argv import argv

strop = lin.shortstr

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
        # bitflip X-type errors, frustrate Hz checks, and produce Tx
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


class SimpleDecoder(Decoder):
    """
    Simple (&slow) optimal decoder that sums probabilities over cosets.
    """
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
            r = 0.
            T1 = l_op + T
            for s_op in self.all_Hx:
                T2 = (s_op + T1)%2
                d = T2.sum()
                r += (1-p)**(n-d)*p**d
            sr += r
            dist.append(r)
        dist = [r/sr for r in dist]
        return dist

    def decode(self, p, err_op, verbose=False, **kw):
        T = self.get_T(err_op)
        dist = self.get_dist(p, T)
        p1 = max(dist)
        idx = dist.index(p1)
        l_op = self.all_Lx[idx]
        op = (l_op+T)%2
        return op


class LookupDecoder(Decoder):
    def __init__(self, code):
        Decoder.__init__(self, code)
        code = self.code
        Hz = code.Hz
        m, n = Hz.shape
        lookup = {}

        # weight 0
        u = zeros2(n)
        v = dot2(Hz, u)
        lookup[str(v)] = u

        # weight 1
        for i in range(n):
            u = zeros2(n)
            u[i] = 1
            v = dot2(Hz, u)
            lookup[str(v)] = u

        # weight 2
        for i in range(n):
          for j in range(i+1, n):
            u = zeros2(n)
            u[i] = 1
            u[j] = 1
            v = dot2(Hz, u)
            key = str(v)
            if key not in lookup:
                lookup[key] = u

        # weight 3
        for i in range(n):
          for j in range(i+1, n):
           for k in range(j+1, n):
            u = zeros2(n)
            u[i] = 1
            u[j] = 1
            u[k] = 1
            v = dot2(Hz, u)
            key = str(v)
            if key not in lookup:
                lookup[key] = u

        self.lookup = lookup
        print("LookupDecoder: size=%d"%len(lookup))

    def decode(self, p, err_op, verbose=False, **kw):
        code = self.code
        Hz = code.Hz
        lookup = self.lookup
        v = dot2(Hz, err_op)
        #key = v.tobytes()
        key = str(v)
        if key in lookup:
            u = lookup[key]
        else:
            u = None

        #print()
        #print("decode", v, u)

        return u


class ChainDecoder(Decoder):
    def __init__(self, code, decoders):
        Decoder.__init__(self, code)
        self.decoders = decoders

    def decode(self, p, err_op, verbose=False, **kw):
        for decoder in self.decoders:
            u = decoder.decode(p, err_op, verbose)
            if u is not None:
                return u





