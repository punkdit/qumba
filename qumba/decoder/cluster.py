#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time

import numpy
import numpy.random as ra

from qumba import solve
from qumba.solve import shortstrx, shortstr, array2
from qumba.decoder import dynamic

def write(s):
    print(s, end="", flush=True)

dot = numpy.dot


class Cluster(object):
    def __init__(self, u0, u1=None, v=None):
        self.u0 = u0 # original syndrome
        self.u1 = u1 if u1 is not None else u0 # check neighbourhood
        self.v = v # bits neighbourhood

        # XXX paranoid:
        self.u0 = u0.copy()
        self.u1 = self.u1.copy()
        if v is not None:
            self.v = v.copy()

    def intersect(self, other):
#        u1 = self.u1 * other.u1
#        return u1.max() > 0
        v = self.v * other.v
        return v.max() > 0

    def grow(self, H, count=1):
        u1 = self.u1
        for i in range(count):
            v = dot(u1, H)
            u1 = dot(H, v)
        return Cluster(self.u0, u1, v)

    def union(self, other):
        u1 = self.u1 + other.u1
        assert (self.u0 * other.u0).max() == 0 # disjoint syndrome
        u0 = self.u0 + other.u0 # could use max
        v = self.v + other.v # could use max
        return Cluster(u0, u1, v)

    def solve(self, H, minimize=True, verbose=True):
        m, n = H.shape # rows, cols
        u1 = self.u1
        assert u1.shape == (m,)
        rows = [i for i in range(m) if u1[i]]
        #print rows
        H = H[rows, :]
        u0 = self.u0[rows]
        v = self.v
        cols = [i for i in range(n) if v[i]]
        #print cols
        H = H[:, cols]
        assert cols, self.u0
        #print shortstr(H)
        #print "u0:", shortstr(u0)
        if verbose: print("Cluster.solve:", H.shape)
        v = solve.solve(H, u0)
        if v is None:
            return
        if minimize:
          kern = solve.kernel(H)
          if len(kern):
            write("[w=%d, kern=%d, "%(v.sum(), len(list(kern))))
            graph = dynamic.Tanner(kern)
            #v = graph.minimize(v, target=30, verbose=True)
            v = graph.localmin(v, verbose=True)
            write("w=%d]"%v.sum())
        #u = dot(H, v)
        #assert numpy.abs(u-u0).sum() == 0
        v0 = numpy.zeros((n,), dtype=numpy.int32)
        v0[cols] = v
        #print "v0:", shortstr(v0)
        #write("[%d]"%v0.sum())
        return v0

    def minsolve(self, H, verbose):
        m, n = H.shape # rows, cols
        u1 = self.u1
        assert u1.shape == (m,)
        rows = [i for i in range(m) if u1[i]]
        #print rows
        H = H[rows, :]
        u0 = self.u0[rows]
        v = self.v
        cols = [i for i in range(n) if v[i]]
        #print cols
        H = H[:, cols]
        assert cols, self.u0
        #print shortstr(H)
        #print "u0:", shortstr(u0)
        if verbose: print("Cluster.solve:", H.shape)
        v = solve.solve(H, u0)


class CSSDecoder(object):

    def __init__(self, code, **kw):
        Hx, Hz = code.Hz, code.Hz
        mx, n = Hx.shape
        assert Hz.shape[1] == n
        mz, _ = Hz.shape

        self.Hx = Hx # detect Z errors
        self.Hz = Hz # detect X errors
        self.mx = mx
        self.mz = mz
        self.n = n
        self.d = 2
        self.__dict__.update(kw)

    def check(self, v):
        #print self.H.shape, err.shape
        u = dot(self.Hz, v) % self.d
        return u

    def is_stab_full(self, v, str=shortstr, verbose=False):
        Hx_t = self.Hx.transpose()
        #Hx_t = Hx_t.copy()
        #u = solve.search(Hx_t, v)
        #print "is_stab:", Hx_t.shape, v.shape,;sys.stdout.flush()
        u = solve.solve(Hx_t, v)
        #print u is not None
        if u is not None:
            #print "[%s]"%u.sum(),
            v1 = dot(Hx_t, u) % 2
            assert ((v+v1)%2).max() == 0
        return u is not None

    def is_stab_cluster(self, v, str=shortstr, verbose=False):
        if v.max() == 0:
            if verbose:
                print("is_stab: zero")
            return True
        if verbose:
            print("is_stab: v")
            print(str(v))
        v0 = v
        v = v.copy()
        Hx_t = self.Hx.transpose()
        c = Cluster(v)
        c = c.grow(Hx_t, 2) # hmmm... doesn't work when the code has an empty colum
        u = c.solve(Hx_t, verbose=False)
        if u is None:
            write('S')
            u = solve.solve(Hx_t, v, debug=False)
        if u is None:
            if verbose:
                print("is_stab: no solution found")
            write('N')
            return None
        v1 = dot(Hx_t, u) % 2
        if verbose:
            print("is_stab: solved")
        assert ((v0+v)%2).max() == 0 # v0==v
        assert ((v1+v)%2).max() == 0 # !!!
        return ((v1+v)%2).max() == 0 # v1==v ie. solution found
    #is_stab = is_stab_full
    is_stab = is_stab_cluster

    def decode(self, p, err, verbose=False, **kw):
        return


class ClusterCSSDecoder(CSSDecoder):

    minimize = False
    def decode(self, p, err, verbose=False, str=shortstr, **kw):
        s = self.check(err)
        if verbose:
            print("error:   \n%s" % str(err))
            #s1 = dot(self.Hz.transpose(), s)
            #s1 = str(s1).replace('1', '*')
            print("syndrome:\n%s" % s)
        #A = self.A
        mz, mx, n = self.mz, self.mx, self.n

        # use many clusters:
        assert len(s)==mz
        cs = []
        for i in range(mz):
            if s[i]:
                s1 = s.copy()
                s1[:] = 0
                s1[i] = 1
                #print str(s1)
                c = Cluster(s1)
                cs.append(c)

        v0 = numpy.zeros((n,), dtype=numpy.int32)
        while cs:
            cs1 = [c.grow(self.Hz) for c in cs]
            j = 0
            while j < len(cs1):
                cj = cs1[j]
                for k in range(j+1, len(cs1)):
                    ck = cs1[k]
                    if cj.intersect(ck):
                        cs1[j] = cj.union(ck)
                        cs1.pop(k)
                        break
                else:
                    j += 1

            cs = cs1
            #if len(cs)>1:
            #    print "interstections:"
            for c1 in cs:
              for c2 in cs:
                #print c1, c2
                if c1 is c2:
                    continue
                assert not c1.intersect(c2)
                #print str(c1.u1*c2.u1)

            if verbose and 0:
                print("clusters:", len(cs))
                for c in cs:
                    print(str(c.u0))
                    print(str(c.u1))
                    #print c.u1
                    print()

            cs1 = []
            for c in cs:
                v = c.solve(self.Hz, minimize=self.minimize, verbose=False)
                if v is None:
                    cs1.append(c)
                else:
                    #if (v0*v).max() > 0:
                    #    assert 0 # i guess this is possible...!!
                    #    #print "crap"
                    #    #return
                    v0 = (v+v0)%2
            # Yes this can happen.. just keep growing this cluster:
            #if len(cs)==1 and len(cs1)==1:
            #    c = cs[0]
            #    if c.u1.min() == 1:
            #        assert 0
            #        print "FAIL TO SOLVE"
            #        return None
            cs = cs1

            if len(cs)>1:
                write("(%s)"%len(cs))
            elif cs:
                write('!')

            #if len(cs)==1:
            #    sys.stdout.write('!');sys.stdout.flush()
            #    cs[0] = cs[0].grow(self.Hz, 10)
            #    #return None

            if verbose:
                print("solution:\n%s" % str(v0))

#        u = dot(self.Hz, v0) % 2
#        #assert u.max() < 2, u
#        if u.max() == 0:
#            return v0
        if verbose:
            print("result:")
            print(str((v0+err)%2))
        return v0


