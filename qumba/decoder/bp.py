#!/usr/bin/env python3

import sys, os
import subprocess
PIPE = subprocess.PIPE
DEVNULL = subprocess.DEVNULL

from math import *
from random import *

import numpy
import numpy.random as ra

from qumba.solve import shortstr, dot2, solve, array2, linear_independent, rank
from qumba.argv import argv


def save_alist(name, H, j=None, k=None):

    if j is None:
        # column weight
        j = H[:, 0].sum()

    if k is None:
        # row weight
        k = H[0, :].sum()

    m, n = H.shape # rows, cols
    f = open(name, 'w')
    print(n, m, file=f)
    print(j, k, file=f)

    for col in range(n):
        print( H[:, col].sum(), end=" ", file=f)
    print(file=f)
    for row in range(m):
        print( H[row, :].sum(), end=" ", file=f)
    print(file=f)

    for col in range(n):
        for row in range(m):
            if H[row, col]:
                print( row+1, end=" ", file=f)
        print(file=f)

    for row in range(m):
        for col in range(n):
            if H[row, col]:
                print(col+1, end=" ", file=f)
        print(file=f)
    f.close()


class RadfordNealBPDecoder(object):

    def __init__(self, code=None, H=None):
        if H is None:
            H = code.Hz
        self.H = H
        m, n = H.shape
        self.m = m # rows
        self.n = n # cols

        stem = 'tempcode_%.6d'%randint(0, 99999999)
        self.stem = stem
        save_alist(stem+'.alist', H)
        path = __file__
        assert path.endswith("/bp.py")
        path = path[:-len("bp.py")]
        path = path + "/LDPC-codes"
        r = os.system('%s/alist-to-pchk -t %s.alist %s.pchk' % (path, stem, stem))
        assert r==0
        self.path = path

    def __del__(self):
        stem = self.stem
        for ext in 'alist pchk out'.split():
            try:
                os.unlink("%s.%s"%(stem, ext))
            except OSError:
                pass

    def decode(self, p, err, max_iter=None, verbose=False, **kw):

        stem = self.stem
        if max_iter is None:
            max_iter = argv.get("maxiter", self.n)

        try:
            os.unlink('%s.out'%stem)
        except:
            pass

        cmd = '%s/decode %s.pchk - %s.out bsc %.4f prprp %d' % (
            self.path, stem, stem, p, max_iter)
        p = subprocess.Popen(cmd, shell=True,
            stdin=PIPE, stdout=DEVNULL,
            stderr=DEVNULL, close_fds=True)

        for x in err:
            data = ("%s\n"%x).encode() 
            p.stdin.write(data)

        p.stdin.close()
        p.wait()

        op = open('%s.out'%stem).read()

        op = [int(c) for c in op.strip()]
        syndrome = dot2(self.H, op)

        if syndrome.sum() == 0:
            return (err + op) % 2


def make_bigger(H, weight): 
    m, n = H.shape
    rows = []
    for i in range(m):
      for j in range(i+1, m):
        u = (H[i]+H[j])%2
        if u.sum() == weight:
            rows.append(u)
    #print("rows:", len(rows))
    #R = array2(rows)
    #print(rank(R), m)
    while 1:
        shuffle(rows)
        H1 = array2(rows)
        H1 = linear_independent(H1)
        while len(H1)<m:
            u = H[randint(0, m-1)]
            v = solve(H1.transpose(), u)
            if v is None:
                u.shape = (1, n)
                H1 = numpy.concatenate((H1, u))
                #print(len(H1), m)
        H1 = array2(H1)
        assert rank(H1) == m
        #print(H1.sum(1))
        print("/", end="", flush=True)
        yield H1


class RetryBPDecoder(object):
    def __init__(self, code):
        H = code.Hz
        Hs = make_bigger(H, 12)
        Hs = [Hs.__next__() for i in range(100)]
        print()
        self.Hs = Hs

    def decode(self, p, err, max_iter=None, verbose=False, **kw):
        for H in self.Hs:
            decoder = RadfordNealBPDecoder(H=H)
            op = decoder.decode(p, err)
            if op is not None:
                return op





