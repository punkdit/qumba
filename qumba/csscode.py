#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time

import numpy
import numpy.random as ra
from numpy.linalg import lstsq
from numpy import concatenate as cat
dot = numpy.dot

from qumba import solve
from qumba import qcode
from qumba.isomorph import Tanner, search
from qumba.solve import (
    shortstr, shortstrx, eq2, dot2, compose2, rand2,
    pop2, insert2, append2, array2, zeros2, identity2, rank)


    
    
def find_autos(Ax, Az):
    # find automorphisms of the XZ-Tanner graph
    ax, n = Ax.shape
    az, _ = Az.shape

    U = Tanner.build2(Ax, Az)
    V = Tanner.build2(Ax, Az)
    perms = []
    for g in search(U, V):
        # each perm acts on ax+az+n in that order
        assert len(g) == n+ax+az
        for i in range(ax):
            assert 0<=g[i]<ax
        for i in range(ax, ax+az):
            assert ax<=g[i]<ax+az
        g = [g[i]-ax-az for i in range(ax+az, ax+az+n)]
        perms.append(g)
    return perms


def find_zx_duality(Ax, Az):
    # Find a zx duality
    ax, n = Ax.shape
    az, _ = Az.shape
    U = Tanner.build2(Ax, Az)
    V = Tanner.build2(Az, Ax)
    for duality in search(U, V):
        break
    else:
        return None

    dual_x = [duality[i] for i in range(ax)]
    dual_z = [duality[i]-ax for i in range(ax,ax+az)]
    dual_n = [duality[i]-ax-az for i in range(ax+az,ax+az+n)]

    Ax1 = Ax[dual_z, :][:, dual_n]
    Az1 = Az[dual_x, :][:, dual_n]

    assert eq2(Ax1, Az)
    assert eq2(Az1, Ax)
    return dual_n





def sparsestr(A):
    idxs = numpy.where(A)[0]
    return "[[%s]]"%(' '.join(str(i) for i in idxs))


def check_conjugate(A, B):
    if A is None or B is None:
        return
    assert A.shape == B.shape
    I = numpy.identity(A.shape[0], dtype=numpy.int32)
    assert eq2(dot(A, B.transpose())%2, I)


def check_commute(A, B):
    if A is None or B is None:
        return
    C = dot2(A, B.transpose())
    assert C.sum() == 0 #, "\n%s"%shortstr(C)


def direct_sum(A, B):
    if A is None or B is None:
        return None
    m = A.shape[0] + B.shape[0] # rows
    n = A.shape[1] + B.shape[1] # cols
    C = zeros2(m, n)
    C[:A.shape[0], :A.shape[1]] = A
    C[A.shape[0]:, A.shape[1]:] = B
    return C


class CSSCode(object):

    def __init__(self,
            Lx=None, Lz=None, 
            Hx=None, Tz=None, 
            Hz=None, Tx=None,
            Gx=None, Gz=None, 
            build=True,
            check=True, verbose=False, logops_only=False):

        if Hx is None and Hz is not None:
            # This is a classical code
            Hx = zeros2(0, Hz.shape[1])

        self.Lx = Lx
        self.Lz = Lz
        self.Hx = Hx
        self.Tz = Tz
        self.Hz = Hz
        self.Tx = Tx
        self.Gx = Gx
        self.Gz = Gz

        if Hx is not None and len(Hx.shape)<2:
            Hx.shape = (0, Hz.shape[1])
        if Hz is not None and Hx is not None and Lz is not None and Gz is None:
            assert Hx.shape[0]+Hz.shape[0]+Lz.shape[0] == Hx.shape[1]
            assert Hx.shape[1] == Hz.shape[1] == Lz.shape[1], Lz.shape

        n = None
        if Gz is not None and Gx is not None:
            _, n = Gz.shape
            if build:
                self.build_from_gauge(check=check)
        #elif None in (Lx, Lz, Tx, Tz) and build:
        elif build and (Lx is None or Lz is None or Tx is None or Tz is None):
            self.build(check=check, logops_only=logops_only, verbose=verbose)
        elif Hz is not None and Hx is not None:
            _, n = Hz.shape
            self.k = n - Hz.shape[0] - Hx.shape[0]

        for op in [Lx, Lz, Hx, Tz, Hz, Tx]:
            if op is None:
                continue
            #print op.shape
            if n is not None and op.shape==(0,):
                op.shape = (0, n)
            n = op.shape[1]
        self.n = n
        if self.Hx is not None:
            self.mx = self.Hx.shape[0]
        if self.Hz is not None:
            self.mz = self.Hz.shape[0]
        if self.Lx is not None:
            self.k = self.Lx.shape[0]
        if self.Gx is not None:
            self.gx = self.Gx.shape[0]
        if self.Gz is not None:
            self.gz = self.Gz.shape[0]

        self.check = check
        self.do_check()

    def copy(self):
        Lx, Lz, Hx, Tz, Hz, Tx = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx)
        Lx = Lx.copy()
        Lz = Lz.copy()
        Hx = Hx.copy()
        Tz = Tz.copy()
        Hz = Hz.copy()
        Tx = Tx.copy()
        code = CSSCode(
            Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, 
            Hz=Hz, Tx=Tx, check=self.check)
        return code

    def __add__(self, other):
        "perform direct sum"
        Gx, Gz = direct_sum(self.Gx, other.Gx), direct_sum(self.Gz, other.Gz)
        Hx, Hz = direct_sum(self.Hx, other.Hx), direct_sum(self.Hz, other.Hz)
        Lx, Lz = direct_sum(self.Lx, other.Lx), direct_sum(self.Lz, other.Lz)
        Tx, Tz = direct_sum(self.Tx, other.Tx), direct_sum(self.Tz, other.Tz)
        code = CSSCode(
            Lx=Lx, Lz=Lz, Hx=Hx, Tz=Tz, 
            Hz=Hz, Tx=Tx, check=self.check)
        return code

    def __rmul__(self, r):
        assert type(r) is int
        code = self
        for i in range(r-1):
            code = self + code
        return code

    def __hash__(self):
        ss = []
        for H in [
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx]:
            if H is not None:
                ss.append(H.tostring())
        return hash(tuple(ss))

    def build_from_gauge(self, check=True, verbose=False):

        write("build_from_gauge:")

        Gx, Gz = self.Gx, self.Gz
        Hx, Hz = self.Hx, self.Hz
        Lx, Lz = self.Lx, self.Lz

        #print "build_stab"
        #print shortstr(Gx)
        #vs = solve.find_kernel(Gx)
        #vs = list(vs)
        #print "kernel Gx:", len(vs)
    
        n = Gx.shape[1]
    
        if Hz is None:
            A = dot2(Gx, Gz.transpose())
            vs = solve.find_kernel(A)
            vs = list(vs)
            #print "kernel GxGz^T:", len(vs)
            Hz = zeros2(len(vs), n)
            for i, v in enumerate(vs):
                Hz[i] = dot2(v.transpose(), Gz) 
        Hz = solve.linear_independent(Hz)
    
        if Hx is None:
            A = dot2(Gz, Gx.transpose())
            vs = solve.find_kernel(A)
            vs = list(vs)
            Hx = zeros2(len(vs), n)
            for i, v in enumerate(vs):
                Hx[i] = dot2(v.transpose(), Gx)
        Hx = solve.linear_independent(Hx)

        if check:
            check_commute(Hz, Hx)
            check_commute(Hz, Gx)
            check_commute(Hx, Gz)

        #Gxr = numpy.concatenate((Hx, Gx))
        #Gxr = solve.linear_independent(Gxr)
        #print(Hx.shape)
        #assert rank(Hx) == len(Hx)
        #assert eq2(Gxr[:len(Hx)], Hx)
        #Gxr = Gxr[len(Hx):]

        Px = solve.get_reductor(Hx).transpose()
        Gxr = dot2(Gx, Px)
        Gxr = solve.linear_independent(Gxr)
    
        #Gzr = numpy.concatenate((Hz, Gz))
        #Gzr = solve.linear_independent(Gzr)
        #assert eq2(Gzr[:len(Hz)], Hz)
        #Gzr = Gzr[len(Hz):]

        Pz = solve.get_reductor(Hz).transpose()
        Gzr = dot2(Gz, Pz)
        Gzr = solve.linear_independent(Gzr)
    
        if Lx is None:
            Lx = solve.find_logops(Gz, Hx)
        if Lz is None:
            Lz = solve.find_logops(Gx, Hz)

        write('\n')

        print("Gxr", Gxr.shape)
        print("Gzr", Gzr.shape)
        assert len(Gxr)==len(Gzr)
        kr = len(Gxr)
        V = dot2(Gxr, Gzr.transpose())
        U = solve.solve(V, identity2(kr))
        assert U is not None
        Gzr = dot2(U.transpose(), Gzr)

        if check:
            check_conjugate(Gxr, Gzr)
            check_commute(Hz, Gxr)
            check_commute(Hx, Gzr)
            check_commute(Lz, Gxr)
            check_commute(Lx, Gzr)

        assert len(Lx)+len(Hx)+len(Hz)+len(Gxr)==n

        self.Lx, self.Lz = Lx, Lz
        self.Hx, self.Hz = Hx, Hz
        self.Gxr, self.Gzr = Gxr, Gzr

    def build(self, logops_only=False, check=True, verbose=False):
    
        Hx, Hz = self.Hx, self.Hz
        Lx, Lz = self.Lx, self.Lz
        Tx, Tz = self.Tx, self.Tz

        if verbose:
            _write = write
        else:
            _write = lambda *args : None
    
        _write('li:')
        self.Hx = Hx = solve.linear_independent(Hx)
        self.Hz = Hz = solve.linear_independent(Hz)
    
        mz, n = Hz.shape
        mx, nx = Hx.shape
        assert n==nx
        assert mz+mx<=n, (mz, mx, n)
    
        _write('build:')
    
        if check:
            # check kernel of Hx contains image of Hz^t
            check_commute(Hx, Hz)
    
        if Lz is None:
            _write('find_logops(Lz):')
            Lz = solve.find_logops(Hx, Hz, verbose=verbose)
            #print shortstr(Lz)
            #_write(len(Lz))

        k = len(Lz)
        assert n-mz-mx==k, "_should be %d logops, found %d. Is Hx/z degenerate?"%(
            n-mx-mz, k)

        _write("n=%d, mx=%d, mz=%d, k=%d\n" % (n, mx, mz, k))
    
        # Find Lx --------------------------
        if Lx is None:
            _write('find_logops(Lx):')
            Lx = solve.find_logops(Hz, Hx, verbose=verbose)

        assert len(Lx)==k

        if check:
            check_commute(Lx, Hz)
            check_commute(Lz, Hx)


        U = dot2(Lz, Lx.transpose())
        I = identity2(k)
        A = solve.solve(U, I)
        assert A is not None, "problem with logops: %s"%(U,)
        #assert eq2(dot2(U, A), I)
        #assert eq2(dot2(Lz, Lx.transpose(), A), I)

        Lx = dot2(A.transpose(), Lx)

        if check:
            check_conjugate(Lz, Lx)

        if not logops_only:

            # Find Tz --------------------------
            _write('Find(Tz):')
            U = zeros2(mx+k, n)
            U[:mx] = Hx
            U[mx:] = Lx
            B = zeros2(mx+k, mx)
            B[:mx] = identity2(mx)
    
            Tz_t = solve.solve(U, B)
            Tz = Tz_t.transpose()
            assert len(Tz) == mx
    
            check_conjugate(Hx, Tz)
            check_commute(Lx, Tz)
    
            # Find Tx --------------------------
            _write('Find(Tx):')
            U = zeros2(n, n)
            U[:mz] = Hz
            U[mz:mz+k] = Lz
            U[mz+k:] = Tz
    
            B = zeros2(n, mz)
            B[:mz] = identity2(mz)
            Tx_t = solve.solve(U, B)
            Tx = Tx_t.transpose()
    
            _write('\n')
        
            if check:
                check_conjugate(Hz, Tx)
                check_commute(Lz, Tx)
                check_commute(Tz, Tx)

        self.k = k
        self.Lx = Lx
        self.Lz = Lz
        self.Tz = Tz
        self.Tx = Tx

    def do_check(self):
        if not self.check:
            return
        #write("checking...")
        Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz = (
            self.Lx, self.Lz, self.Hx,
            self.Tz, self.Hz, self.Tx,
            self.Gx, self.Gz)
        check_conjugate(Lx, Lz)
        check_conjugate(Hx, Tz)
        check_conjugate(Hz, Tx)
        #if Gx is not None and Gz is not None:
        #    check_conjugate(Gx, Gz)
        check_commute(Lx, Tz)
        check_commute(Lx, Hz)
        check_commute(Lz, Tx)
        check_commute(Lz, Hx)
        check_commute(Hx, Hz)
        check_commute(Tx, Tz)
        all_ops = [Lx, Lz, Hx, Tz, Hz, Tx]
        for ops in all_ops:
            if ops is not None:
                assert ops.shape[1] == self.n
        #write("done\n")

    _dual = None
    def get_dual(self, build=False, check=False):
        if self._dual:
            return self._dual
        code = CSSCode(
            self.Lz, self.Lx, 
            self.Hz, self.Tx, 
            self.Hx, self.Tz, self.Gz, self.Gx, 
            build, self.check or check)
        self._dual = code
        return code

    def to_qcode(self):
        return qcode.QCode.build_css(self.Hx, self.Hz, self.Lx, self.Lz)

    def __repr__(self):
        Lx = len(self.Lx) if self.Lx is not None else None
        Lz = len(self.Lz) if self.Lz is not None else None
        Hx = len(self.Hx) if self.Hx is not None else None
        Tz = len(self.Tz) if self.Tz is not None else None
        Hz = len(self.Hz) if self.Hz is not None else None
        Tx = len(self.Tx) if self.Tx is not None else None
        Gx = len(self.Gx) if self.Gx is not None else None
        Gz = len(self.Gz) if self.Gz is not None else None
        n = self.n
        #if Lx is None and Hz and Hx:
        #    Lx = "(%d)"%(n - Hx - Hz)
        #if Lz is None and Hz and Hx:
        #    Lz = "(%d)"%(n - Hx - Hz)
        return "CSSCode(n=%s, Lx:%s, Lz:%s, Hx:%s, Tz:%s, Hz:%s, Tx:%s, Gx:%s, Gz:%s)" % (
            n, Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz)

    def __str__(self):
        return "[[%d, %d]]"%(self.n, self.k)

    def save(self, name=None, stem=None):
        assert name or stem
        if stem:
            s = hex(abs(hash(self)))[2:]
            name = "%s_%s_%d_%d_%d.ldpc"%(stem, s, self.n, self.k, self.d)
            print("save", name)
        f = open(name, 'w')
        for name in 'Lx Lz Hx Tz Hz Tx Gx Gz'.split():
            value = getattr(self, name, None)
            if value is None:
                continue
            print("%s ="%name, file=f)
            print(shortstr(value), file=f)
        f.close()

    @classmethod
    def load(cls, name, build=True, check=False, rebuild=False):
        write("loading..")
        f = open(name)
        data = f.read()
        data = data.replace('.', '0')
        lines = data.split('\n')
        name = None
        items = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '=' in line:
                name = line.split('=')[0].strip()
                #print name
                rows = []
                items[name] = rows
            else:
                #line = list(int(x) for x in line)
                line = numpy.fromstring(line, dtype=numpy.uint8) - 48
                rows.append(line)
        #print items.keys()
        #print [len(x) for x in items.values()]
        kw = {}
        for key in list(items.keys()):
            #print items[key]
            value = array2(items[key])
            kw[key] = value
        write("done\n")
        if rebuild:
            for op in 'Lx Lz Tx Tz'.split():
                kw[op] = None
        code = cls(build=build, check=check, **kw)
        return code

    def longstr(self):
        lines = [
            "CSSCode:",
            "Lx:Lz =", shortstrx(self.Lx, self.Lz),
            "Hx:Tz =", shortstrx(self.Hx, self.Tz),
            "Tx:Hz =", shortstrx(self.Tx, self.Hz)
        ]
        return '\n'.join(lines)

    def weightstr(self, logops=False):
        lines = [
            "Lx:%s" % [op.sum() for op in self.Lx] 
                if self.Lx is not None else "Lx:?",
            "Lz:%s" % [op.sum() for op in self.Lz] 
                if self.Lz is not None else "Lz:?",
            "Hx:%s" % [op.sum() for op in self.Hx] 
                if self.Hx is not None else "Hx:?",
            "Tz:%s" % [op.sum() for op in self.Tz] 
                if self.Tz is not None else "Tz:?",
            "Hz:%s" % [op.sum() for op in self.Hz] 
                if self.Hz is not None else "Hz:?",
            "Tx:%s" % [op.sum() for op in self.Tx] 
                if self.Tx is not None else "Tx:?"]
        if logops:
            lines = lines[:2]
        return '\n'.join(lines)

    def weightsummary(self):
        lines = []
        for name in 'Lx Lz Hx Tz Hz Tx'.split():
            ops = getattr(self, name)
            if ops is None:
                continue
            m, n = ops.shape
            rweights = [ops[i].sum() for i in range(m)]
            cweights = [ops[:, i].sum() for i in range(n)]
            if rweights:
                lines.append("%s(%d:%.0f:%d, %d:%.0f:%d)"%(
                    name, 
                    min(rweights), 1.*sum(rweights)/len(rweights), max(rweights),
                    min(cweights), 1.*sum(cweights)/len(cweights), max(cweights),
                ))
            else:
                lines.append("%s()"%name)
        return '\n'.join(lines)

    def apply_perm(self, f):
        n = self.n
        cols = [f[i] for i in range(n)]
        #print("cols:", cols)
        ops = [self.Lx, self.Lz, self.Hx, self.Tz, self.Hz, self.Tx]
        ops = [op[:, cols] for op in ops]
        Lx, Lz, Hx, Tz, Hz, Tx = ops
        Gx, Gz = self.Gx, self.Gz
        Gx = Gx[:, cols] if Gx is not None else None
        Gz = Gz[:, cols] if Gz is not None else None
        code = CSSCode(Lx, Lz, Hx, Tz, Hz, Tx, Gx, Gz)
        return code

    def x_distance(self, min_d=1):
        Lx = [v for v in solve.span(self.Lx) if v.sum()]
        dx = self.n
        for u in solve.span(self.Hx):
            for v in Lx:
                w = (u+v)%2
                d = w.sum()
                if 0<d<dx:
                    dx = d
            if dx==min_d:
                return dx
        return dx

    def z_distance(self, min_d=1):
        Lz = [v for v in solve.span(self.Lz) if v.sum()]
        dz = self.n
        for u in solve.span(self.Hz):
            for v in Lz:
                w = (u+v)%2
                d = w.sum()
                if 0<d<dz:
                    dz = d
            if dz==min_d:
                return dz
        return dz

    def distance(self, min_d=1):
        dx = self.x_distance(min_d)
        dz = self.z_distance(min_d)
        return dx, dz

    def find_autos(self):
        return find_autos(self.Hx, self.Hz)

    def find_zx_duality(self):
        return find_zx_duality(self.Hx, self.Hz)


