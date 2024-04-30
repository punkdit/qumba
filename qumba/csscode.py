#!/usr/bin/env python3

import sys, os
from math import *
from random import *
import time
from functools import reduce

import numpy
import numpy.random as ra
from numpy.linalg import lstsq
from numpy import concatenate as cat
dot = numpy.dot

from qumba import solve
from qumba import qcode
from qumba.qcode import QCode, SymplecticSpace
from qumba.isomorph import Tanner, search
from qumba.solve import (
    shortstr, shortstrx, eq2, dot2, compose2, rand2,
    pop2, insert2, append2, array2, zeros2, identity2, rank, linear_independent)
from qumba.action import Perm
from qumba.argv import argv



def fixed(f):
    return [i for i in range(len(f)) if f[i]==i]

def is_identity(f):
    for i in range(len(f)):
        if f[i] != i:
            return False
    return True

def mul(f, g):
    return [f[g[i]] for i in range(len(g))]
        

def find_logicals(Ax, Az):

    Hx = linear_independent(Ax)
    Hz = linear_independent(Az)
    code = QCode.build_css(Hx, Hz)
    print(code)
    space = code.space

    perms = find_autos(Ax, Az)
    print("perms:", len(perms))

    duality = find_zx_duality(Ax, Az)

    dode = code.apply_perm(duality)
    dode = dode.apply_H()
    assert code.is_equiv(dode)

    n = code.n
    kk = 2*code.k
    K = SymplecticSpace(code.k)
    M = code.get_encoder()
    Mi = space.F * M.t * space.F
    #Mi = dot2(space.F, M.transpose(), space.F)
    #I = identity2(code.nn)
    I = space.get_identity()
    #assert eq2(dot2(M, Mi), I)
    #assert eq2(dot2(Mi, M), I)
    assert M*Mi == I
    assert Mi*M == I

    gens = []
    #A = dot2(space.get_H(), space.get_perm(duality))
    A = space.get_H() * space.get_perm(duality)
    gens.append(A)

    for f in perms:
        A = space.get_perm(f)
        gens.append(A)

    for f in perms:
        # zx duality
        zx = mul(duality, f)
        #print("fixed:", len(fixed(zx)), "involution" if is_identity(mul(zx,zx)) else "")

    for f in perms:
        # zx duality
        zx = mul(duality, f)
        if not is_identity(mul(zx, zx)) or len(fixed(zx))%2 != 0:
            continue
        # XXX there's more conditions to check

        A = I
        remain = set(range(n))
        for i in fixed(zx):
            #A = dot2(space.get_S(i), A)
            A = space.get_S(i) * A
            remain.remove(i)
        for i in range(n):
            if i not in remain:
                continue
            j = zx[i]
            assert zx[j] == i
            remain.remove(i)
            remain.remove(j)
            #A = dot2(space.get_CZ(i, j), A)
            A = space.get_CZ(i, j) * A
        gens.append(A)
        #break # sometimes we only need one of these ...

    print("gens:", len(gens))

    logicals = []
    found = set()
    for A in gens:
        dode = QCode.from_encoder(dot2(A, M), code.m)
        assert dode.is_equiv(code)
        #MiAM = dot2(Mi, A, M)
        MiAM = Mi*A*M
        L = MiAM[-kk:, -kk:]
        assert K.is_symplectic(L)
        s = L.shortstr()
        if s not in found:
            logicals.append(L)
            found.add(s)
    print("logicals:", len(logicals))
    gens = logicals

    from sage.all_cmdline import GF, matrix, MatrixGroup
    field = GF(2)
    logicals = [matrix(field, kk, A.A.copy()) for A in logicals]
    G = MatrixGroup(logicals)
    print("|G| =", G.order())
    print("G =", G.structure_description())

    return gens


    
    
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
            Ax=None, Az=None,
            build=True, check=True, verbose=False, logops_only=False):

        if Hx is None and Hz is not None:
            # This is a classical code
            Hx = zeros2(0, Hz.shape[1])

        Ax = Hx if Ax is None else Ax
        Az = Hz if Az is None else Az
        if Hx is None:
            assert Ax is not None
            Hx = linear_independent(Ax)
        if Hz is None:
            assert Az is not None
            Hz = linear_independent(Az)

        self.Lx = Lx
        self.Lz = Lz
        self.Hx = Hx
        self.Tz = Tz
        self.Hz = Hz
        self.Tx = Tx
        self.Gx = Gx
        self.Gz = Gz
        self.Ax = Ax
        self.Az = Az

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
        #vs = solve.kernel(Gx)
        #vs = list(vs)
        #print "kernel Gx:", len(vs)
    
        n = Gx.shape[1]
    
        if Hz is None:
            A = dot2(Gx, Gz.transpose())
            vs = solve.kernel(A)
            vs = list(vs)
            #print "kernel GxGz^T:", len(vs)
            Hz = zeros2(len(vs), n)
            for i, v in enumerate(vs):
                Hz[i] = dot2(v.transpose(), Gz) 
        Hz = solve.linear_independent(Hz)
    
        if Hx is None:
            A = dot2(Gz, Gx.transpose())
            vs = solve.kernel(A)
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

    @classmethod
    def random(cls, n, mx, mz, distance=None, **kw):
        """
        http://arxiv.org/abs/quant-ph/9512032
        Quantum error-correcting codes exist
        with asymptotic rate:
        k/n = 1 - 2H(2t/n) where
        H(p) = -p log p - (1-p) log (1-p) and
        t = floor((d-1)/2).
        """
    
        while 1:
            k = n-mx-mz
            assert k>=0
    
            #print("rate:", 1.*k//n)
            #H = lambda p: -p*log(p) - (1-p)*log(1-p)
            #d = 56
            #print(1-2*H(1.*d//n)) # works!
    
            Hz = rand2(mz, n)
            #print(shortstr(Hz))
            if rank(Hz) < mz:
                continue
            kern = numpy.array(solve.kernel(Hz))
    
            Hx = zeros2(mx, n)
            for i in range(mx):
                v = rand2(1, n-mx)
                Hx[i] = dot2(v, kern)
            C = cls(Hx=Hx, Hz=Hz, **kw)
    
            if distance is None:
                break
            dx,dz = C.distance()
            if min(dx,dz) >= distance:
                break
            #print('.', flush=True, end='')
        #print() 
        return C
    
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
        return qcode.QCode.build_css(self.Hx, self.Hz, self.Tx, self.Tz, self.Lx, self.Lz)

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
            if dx<=min_d:
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
            if dz<=min_d:
                return dz
        return dz

    def distance(self, min_d=1):
        dx = self.x_distance(min_d)
        dz = self.z_distance(min_d)
        return dx, dz

    def find_autos(self):
        return find_autos(self.Ax, self.Az)

    def find_zx_duality(self):
        return find_zx_duality(self.Ax, self.Az)

    def find_zx_duality_for_cz(self):
        "find all the zx dualities that give rise to a cz gate"
        found = []
        duality = self.find_zx_duality()
        if duality is None:
            return found

        perms = self.find_autos()
        for f in perms:
            # zx duality
            zx = mul(duality, f)
            if not is_identity(mul(zx, zx)) or len(fixed(zx))%2 != 0:
                continue
            # XXX there's more conditions to check

            found.append(zx)

        return found

    def find_zx_dualities(code):
        n = code.n
        Ax, Az = code.Ax, code.Az
        duality = find_zx_duality(Ax, Az)
        items = list(range(n))
        duality = Perm.promote(duality, items)
        #print(duality)
        perms = find_autos(Ax, Az)
        G = [Perm.promote(g, items) for g in perms]
        #print("|G| =", len(G))
        I = Perm(items, items)
        zxs = []
        for g in G:
            zx = g*duality
            if zx*zx != I:
                continue
            for i in items:
                if zx[i] == i:
                    break
            else:
                zxs.append(zx)
        return zxs




def find_z3(n, mx, mz, d=None):
    print("find_z3", n, mx, mz, d)
    import z3
    from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, sat, ForAll
    from qumba.transversal import UMatrix

    solver = Solver()
    add = solver.add

    Hx = zeros2(mx, n)
    Hx[:, :mx] = identity2(mx)
    Hz = zeros2(mz, n)

    Hx = UMatrix.unknown(mx, n)
    Hx[:, :mx] = identity2(mx)
    Hz = UMatrix.unknown(mz, n)
    #Hz[:, :mz] = identity2(mz)
    Tx = UMatrix.unknown(mz, n)
    Tz = UMatrix.unknown(mx, n)

    k = n - mx - mz
    Lx = UMatrix.unknown(k, n)
    Lz = UMatrix.unknown(k, n)

    Ik = UMatrix(identity2(k))
    Imx = UMatrix(identity2(mx))
    Imz = UMatrix(identity2(mz))
    add(Hx*Hz.t == 0)
    add(Hx*Lz.t == 0)
    add(Hx*Tz.t == Imx)
    add(Hz*Lx.t == 0)
    add(Hz*Tx.t == Imz)
    add(Tx*Tz.t == 0)
    add(Lx*Tz.t == 0)
    add(Lz*Tx.t == 0)
    add(Lx*Lz.t == Ik)

    vz = UMatrix.unknown(1, n)
    t_parity = (Hx*vz.t == 0)
    t_logical = (Lx*vz.t != 0)
    t_distance = Sum([If(vz[0,i].v,1,0) for i in range(n)]) >= d
    term = If(And(t_parity, t_logical),t_distance,True)
    add(ForAll([vz[0,i].v for i in range(n)], term))

    vx = UMatrix.unknown(1, n)
    t_parity = (Hz*vx.t == 0)
    t_logical = (Lz*vx.t != 0)
    t_distance = Sum([If(vx[0,i].v,1,0) for i in range(n)]) >= d
    term = If(And(t_parity, t_logical),t_distance,True)
    add(ForAll([vx[0,i].v for i in range(n)], term))

    result = solver.check()
    assert str(result) == "sat"

    model = solver.model()
    Hx = Hx.get_interp(model)
    print("Hx =")
    print(Hx)
    Hz = Hz.get_interp(model)
    print("Hz =")
    print(Hz)
    Lz = Lz.get_interp(model)
    print("Lz =")
    print(Lz)
    Lx = Lx.get_interp(model)
    print("Lx =")
    print(Lx)
    Tz = Tz.get_interp(model)
    Tx = Tx.get_interp(model)

    code = CSSCode(Hx=Hx.A, Hz=Hz.A, Lx=Lx.A, Lz=Lz.A, Tx=Tx.A, Tz=Tz.A, check=True)
    #code = CSSCode(Hx=Hx.A, Hz=Hz.A, check=True)
    return code


def test_find():
    n = argv.get("n", 15)
    m = argv.get("m", 7)
    mx = argv.get("mx", m)
    mz = argv.get("mz", m)
    d = argv.get("d", 3)
    code = find_z3(n, mx, mz, d)
    print(code)
    print(code.distance())
    #print(code.longstr())


def distance_lower_bound_z3(Hx, Lx, d):
    from z3 import Bool, And, Or, Xor, Not, Implies, Sum, If, Solver, sat

    assert len(Lx)

    m, n = Hx.shape
    k, n1 = Lx.shape

    solver = Solver()
    add = solver.add
    v = [Bool("v%d"%i) for i in range(n)]

    term = Sum([If(v[i],1,0) for i in range(n)]) <= d
    add(term)

    def check(hx):
        terms = [v[j] for j in range(n) if hx[j]]
        if len(terms)>1:
            return reduce(Xor, terms)
        elif len(terms)==1:
            return terms[0]
        assert 0, "dead check"

    # parity checks
    for i in range(m):
        add(Not(check(Hx[i])))

    # non-trivial logical
    term = reduce(Or, [check(Lx[i]) for i in range(k)])
    add(term)

    result = solver.check()
    #print(result)
    if result != sat:
        return

    model = solver.model()
    v = [model.evaluate(v[i]) for i in range(n)]
    v = [int(eval(str(vi))) for vi in v]
    v = array2(v)

    u = dot2(Hx, v)
    assert u.sum() == 0, "bug bug... try updating z3?"
    u = dot2(Lx, v)
    assert u.sum() != 0, "bug bug... try updating z3?"
    assert v.sum() <= d, ("v.sum()==%d: bug bug... try updating z3?"%v.sum())
    return v


def distance_z3_css(code):

    if code.k == 0:
        return code.n

    d_x = 1
    while 1:
        v = distance_lower_bound_z3(code.Hz, code.Lz, d_x)
        if v is not None:
            break
        d_x += 1

    d_z = 1
    while d_z<d_x:
        v = distance_lower_bound_z3(code.Hx, code.Lx, d_z)
        if v is not None:
            break
        d_z += 1
    return d_x, d_z


def distance_z3(code):
    d_x, d_z = distance_z3_css(code)
    return min(d_x, d_z)



def test_distance():
    print("\ntest()")
    n = argv.get("n", 15)
    d = argv.get("d", 3)
    code = CSSCode.random(n, n//2, n//2, d, check=False)
    print(code)
    #print(code.longstr())
    d0 = distance_z3(code)
    print("d =", d0)
    return
    code = code.to_qcode()
    from qumba import distance 
    d1 = distance.distance_z3(code)
    #assert d0==d1
    print("d1 =", d1)
    print(code)
    #print(code.longstr())


if __name__ == "__main__":

    from time import time
    start_time = time()
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
    print("OK! finished in %.3f seconds\n"%t)





