#!/usr/bin/env python

"""

"""

import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add
from random import random, randint

import numpy
from scipy.optimize import root

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.smap import SMap
from qumba.qcode import strop, QCode
from qumba import construct 
from qumba.matrix_sage import Matrix
from qumba.clifford import Clifford, w4, r2, ir2
from qumba.action import mulclose
from qumba.dense import bitlog
from qumba.util import choose
from qumba.triorthogonal import is_morthogonal, strong_morthogonal
from qumba import pauli
from qumba import lin

EPSILON = 1e-6

def eq(a, b):
    return abs(a-b) < 1e-6

diff = sage.derivative

py_w8 = (2**(-1/2))*(1+1j)
assert eq(py_w8**2, 1j)

def simplify(f):
    #return f

    if not hasattr(f, "numerator"):
        return f

    top = f.numerator()
    bot = f.denominator()
    #print("simplify:", [ti for ti in top], [bi for bi in bot])

    factor = None
    for (c,e) in list(top)+list(bot):
        #print("int(c)", c)
        try:
            c = int(c)
        except TypeError:
            #print("simplify: TypeError")
            return f
        if factor is None:
            factor = c
        else:
            factor = gcd(factor, c)

    if factor == 0:
        return f

    top = top / factor
    bot = bot / factor

    assert f == top/bot

    f = top / bot

    return f


def pystr(u):
    s = str(u)
    s = s.replace("zeta8^2", "1j")
    #assert "zeta8" not in s, repr(s)
    s = s.replace("zeta8", "py_w8")
    s = s.replace("^", "**")
    return s

rnd = lambda radius=10: 2*radius*random() - radius

#latex = sage.latex

def latex(a):
    s = sage.latex(a)
    #s = s.replace(r"\cdot", "")
    s = s.replace(r"\cdot (", "(")
    s = s.replace(r"\cdot z", "z")
    #s = s.replace(r"\cdot", r"\times")
    s = s.replace("  ", " ")
    #s = s.replace(r"\zeta_{8}^{2}", "i") # ??
    s = s.replace("r_{2}", r"\sqrt{2}")
    return s

if argv.latex:
    mkstr = lambda x : sage.latex(x) + r" \\"
    right_arrow = r"&\mapsto "
else:
    mkstr = str
    right_arrow = chr(0x2192)

from qumba.clifford import K as base
one = base.one()
half = one/2

clifford = Clifford(1)
I = clifford.I
X = clifford.X()
Y = clifford.Y()
Z = clifford.Z()


class Meromorphic:
    R = sage.I.parent()
    Rz = sage.PolynomialRing(R, ["z"])
    K = sage.FractionField(Rz)
    z = K.gens()[0]

    def __init__(self, f):
        f = Meromorphic.K(f)
        self.f = f

    @classmethod
    def build(cls, a=1, b=0, c=0, d=1):
        z = Meromorphic.z
        assert a*d - b*c != 0
        f = (a*z + b) / (c*z + d)
        return cls(f)

    @classmethod
    def promote(cls, item):
        if isinstance(item, Meromorphic):
            return item
        return Meromorphic(item)

    def __str__(self):
        return "Meromorphic(%s)"%(self.f)
    __repr__ = __str__

    def __mul__(self, other):
        other = Meromorphic.promote(other)
        z = Meromorphic.z
        f = self.f.subs({z:other.f})
        return Meromorphic(f)

    def __eq__(self, other):
        other = Meromorphic.promote(other)
        return self.f == other.f

    def __hash__(self):
        return hash(self.f)

    def order(self):
        A = self
        z = Meromorphic.z
        I = Meromorphic(z)
        i = 1
        while A != z:
            A = A*self
            i += 1
        return i


def build_mobius():

    I = Meromorphic.build()
    assert I*I == I

    X = Meromorphic.build(0,1,1,0)
    assert X != I
    assert X*X == I

    Z = Meromorphic.build(-1,0,0,1)
    assert Z*Z == I
    assert Z*X == X*Z
    assert Z*X != I

    Pauli = mulclose([X,Z])
    assert len(Pauli) == 4

    H = Meromorphic.build(1,1,1,-1)
    assert H*H == I
    Meromorphic.H = H

    S = Meromorphic.build(-sage.I, 0, 0, 1) # i hope this is S not S dagger 
    assert S*S == Z
    Meromorphic.S = S

    Clifford = mulclose([S,H])
    assert len(Clifford) == 24

    Meromorphic.Clifford = Clifford
    Meromorphic.Pauli = Pauli

    z = Meromorphic.z
    f = Meromorphic((z**2 + 1)/(2*z)) # XXX rename Mobius as Meromorphic
    f = Meromorphic(z**2)

    for g in Clifford:
        gf = g*f

    F = S*H
    assert F.order() == 3

build_mobius()


def get_projector(code):
    H = strop(code.H, "I")
    assert code.k == 1

    n = code.n
    space = Clifford(n)
    I = space.I
    P = None
    for h in H.split():
        print("\t", h)
        s = space.get_pauli(h)
        s = I+s
        P = s if P is None else s*P

    M = 2**code.m
    P = (one/M)*P

    print("P*P == P ... ", end='', flush=True)
    assert P*P == P
    print("yes")
    assert P.rank() == 2**code.k

    if 0:
        assert P.conjugate()==P
        P = P.change_ring(sage.QQ)
    return P


def stereo(x,y,z):
    #assert abs(1-z) > EPSILON, str(z)
    X, Y = x/(1-z), y/(1-z)
    return X, Y

def istereo(X, Y):
    bot = 1+X**2+Y**2
    x = 2*X/bot
    y = 2*Y/bot
    z = (X**2+Y**2-1)/bot
    return (x,y,z)


def find_zeros(f0, g0, trials=None, nsols=None, verbose=False):

    K = f0.parent()
    X, Y = K.gens()

    #print(K)
    S = sage.PolynomialRing(sage.QQ, list("XY"))

    pyfunc = lambda u : eval("lambda X,Y: %s"%pystr(u))

    fn = f0.numerator()
    gn = g0.numerator()

    fn = S(fn)
    gn = S(gn)
    X, Y = S.gens()
    #print(sage.derivative(fn, S.gens()[0]))
    #return

    d = lambda f,u : pyfunc(sage.derivative(f,u))
    Jac = [[d(fn, X), d(fn, Y)], [d(gn, X), d(gn, Y)]]
    def jac(XY):
        X, Y = XY
        return [[Jac[j][i](X,Y) for i in [0,1]] for j in [0,1]]
    

    fn = pyfunc(fn)
    gn = pyfunc(gn)
    fd = pyfunc(f0.denominator())
    gd = pyfunc(g0.denominator())

    def fun(XY):
        X, Y = XY
        fv = fn(X,Y)
        gv = gn(X,Y)
        return fv, gv

    sols = []

    # try some well-known points..
    for (x,y,z) in [
        (1,0,0),
        (-1,0,0),
        (0,1,0),
        (0,-1,0),
        (0,0,-1),
        # (0,0,1), # singular
    ]:
        x0, y0 = stereo(x, y, z)
        #x0 += rnd(1e-12)
        #y0 += rnd(1e-12)
        #print(x,y,z, fn(x0,y0), gn(x0,y0), )
        if (abs(fn(x0,y0)) < EPSILON and abs(gn(x0,y0)) < EPSILON and \
            abs(fd(x0,y0)) > EPSILON and abs(gd(x0,y0)) > EPSILON):
            yield x0, y0
            sols.append((x0,y0))
    #return

    if verbose:
        print("find_zeros:")
        print("\t", fun([0,0]))
        print("\t", fd(0,0), gd(0,0))

    trial = 0
    while trials is None or trial < trials:
        trial += 1
        x0, y0 = rnd(), rnd()
        sol = root(fun, [x0, y0], #jac=jac, 
            method="hybr", tol=1e-6, 
            #options = {"xtol":1e-2, "maxfev":10000},
        ) # does jac even help?
        if verbose:
            print("x =", sol.x, sol.success)

        #if not sol.success:
        #    #print(sol)
        #    #print("...")
        #    continue
    
        X, Y = sol.x
        for (X1,Y1) in sols:
            r = abs(X1-X) + abs(Y1-Y)
            if r < 0.01:
                if verbose:
                    print("\tskip 1")
                break
        else:
            #print("nfev", sol.nfev)
            #print("find_zeros: %.4f,%.4f" %  (X, Y))
            #print("value:", sol.fun)
            #print("denom:", fd(X,Y), gd(X,Y))
            dx,dy = rnd(1e-4), rnd(1e-4)
            X1,Y1 = X+dx, Y+dy
            if abs(fd(X1,Y1)) < 1e-17 or abs(gd(X1,Y1)) < 1e-17:
                if verbose:
                    print("\tdiv by zero", abs(fd(X1,Y1)), abs(gd(X1,Y1)) )
                continue
            vals = fn(X1,Y1)/fd(X1,Y1), gn(X1,Y1)/gd(X1,Y1)
            #print("\t fun:", vals[0], vals[0])
            if abs(vals[0]) > 0.0001 or abs(vals[1]) > 0.0001:
                if verbose:
                    print("\tnon-zero: [%.4f, %.4f]" %(
                        abs(vals[0]), abs(vals[1])))
                continue
    
            #print("\t---> %.4f,%.4f" %  (X, Y))
            if not sol.success:
                print("Warning: sol.success == False", vals[0], vals[1])
            yield X, Y
            sols.append((X,Y))

            if nsols and len(sols) >= nsols:
                #print("nsols", len(sols))
                return

def vprint(verbose, *args):
    if verbose:
        print(*args)


def find_roots(f, verbose=False):

    vprint(verbose, "find_roots")
    vprint(verbose, "\t", f)

    top = f.numerator()
    bot = f.denominator()

    ring = sage.I.parent()
    R = sage.PolynomialRing(ring, ["z"])

    z = R.gens()[0]

    top = R(top)
    bot = R(bot)
    #vprint(verbose, "\t = %s(%s)/(%s)"%( stop, top, bot ))

    #ftop = sage.factor(top)
    #fbot = sage.factor(bot)
    #vprint(verbose, "\t = %s %s / %s"%( stop, ftop, fbot ))

    for val,m in top.roots(ring=sage.CIF):

        u = bot.subs({z:val})
        if u == 0:
            print("FAIL")
            assert 0

        cval = complex(val)
        #print("find_roots:", repr(val))
        yield val, cval, m



def pprint(f):

    print("f(z) =", f)

    top = f.numerator()
    bot = f.denominator()

    if "zeta8^2" in str(f) and "zeta8^2" in str(w4*f):
        print()
        print(r"\begin{align*}")
        s = r"f(z) &= \frac{%s}{%s} \\"%( latex(top), latex(bot) )
        s = s.replace(r"\zeta_{8}^{2}", "i")
        print(s)
        print(r"\end{align*}")
        print()
        return

    stop = ""
    if "zeta8^2" in str(top):
        top = -w4*top
        assert "zeta8" not in str(top)
        stop = "i "

    if "zeta8^2" in str(bot):
        bot = w4*bot
        assert "zeta8" not in str(bot)
        stop = "-i "+stop
        assert 0, "check me"

    R = sage.PolynomialRing(sage.QQ, "z".split())
    z = R.gens()[0]

    top = R(top)
    bot = R(bot)
    print()
    print(r"\begin{align*}")
    print(r"f(z) &= %s\frac{%s}{%s} \\"%( stop, latex(top), latex(bot) ))

    ftop = sage.factor(top)
    fbot = sage.factor(bot)
    print(r"    &= %s\frac{%s}{%s} \\"%( stop, latex(ftop), latex(fbot) ))
    print(r"\end{align*}")
    print()
    #print("%%    = %s(%s)/(%s)"%( stop, (top), (bot) ))
    #print("%%    = %s %s / %s"%( stop, ftop, fbot ))

    #top = f.numerator().factor()
    #bot = f.denominator().factor()
    #print("f = (%s) / (%s)" % (top, bot))



class Distill:

    f = None
    def __init__(self, n):
        self.n = n

    def get_variety(self):
        pass

    def find(self, trials=1000, nsols=1, top=True, verbose=False):
        sign = +1 if top else -1
        x, y, z, w = self.get_variety()

        #y = sign*y
        #z = sign*z
    
        R = sage.PolynomialRing(base, list("XY"))
        X, Y = R.gens()
        ix = 2*X/(1+X**2+Y**2)
        iy = sign*2*Y/(1+X**2+Y**2)
        iz = sign*(X**2+Y**2-1)/(1+X**2+Y**2)
        #print(ix, iy, iz)
    
        Kx,Ky,Kz,_ = x.parent().gens()
        u, v = stereo(x/w, y/w, z/w)
    
        u = u.subs({Kx:ix, Ky:iy, Kz:iz})
        v = v.subs({Kx:ix, Ky:iy, Kz:iz})
        #print("u =", u)
        #print("v =", v)

        S = sage.PolynomialRing(sage.QQ, list("XY"))
        S = sage.FractionField(S)
        u = S(u)
        v = S(v)

        #f = u.numerator()
        #g = v.numerator()
        #print("zeros:")
        #for (X1,Y1) in find_zeros(f, g, trials, nsols, verbose=verbose):
        #    print("\t", X1, Y1)
    
        jac = [
            [diff(u,X), diff(v,X)],
            [diff(u,Y), diff(v,Y)],
        ]
        assert jac[0][0] == jac[1][1], "Cauchy-Riemann fail"
        assert jac[1][0] == -jac[0][1], "Cauchy-Riemann fail"
        #print("Cauchy-Riemann: yes")
    
        f, g = jac[0]
        #print("f =", f)
        #print("g =", g)
    
        for (X,Y) in find_zeros(f, g, trials, nsols, verbose=verbose):
            x, y, z = istereo(X, Y)
            x, y, z = (x, sign*y, sign*z)
            yield x, y, z

    def build(self):
        if self.f is not None:
            return self.f

        x, y, z, w = self.get_variety()
        R = sage.PolynomialRing(base, list("XY"))
        X, Y = R.gens()
        ix = 2*X/(1+X**2+Y**2)
        iy = 2*Y/(1+X**2+Y**2)
        iz = (X**2+Y**2-1)/(1+X**2+Y**2)
        #print(ix, iy, iz)
    
        Kx,Ky,Kz,_ = x.parent().gens()
        u, v = stereo(x/w, y/w, z/w)
    
        u = u.subs({Kx:ix, Ky:iy, Kz:iz})
        v = v.subs({Kx:ix, Ky:iy, Kz:iz})
        #print("u =", u)
        #print("v =", v)

        T = sage.PolynomialRing(base, "z zb".split())
        T = sage.FractionField(T)
        z, zb = T.gens()

        real = half*(z+zb)
        imag = half*(-w4)*(z-zb)
        uz = u.subs({X:real,Y:imag}) 
        vz = v.subs({X:real,Y:imag}) 
        f = uz + w4*vz

        f = simplify(f)

        #assert f.subs({zb:-99}) == f, (f.subs({zb:1})-f.subs({zb:-2}))
        assert "zb" not in str(f), str(f)
        self.f = f

        return f

    def fast_find(self, top=True, verbose=False):
#        sign = +1 if top else -1
#        x, y, z, w = self.get_variety()
#
#        #y = sign*y
#        #z = sign*z
#    
#        R = sage.PolynomialRing(base, list("XY"))
#        X, Y = R.gens()
#        ix = 2*X/(1+X**2+Y**2)
#        iy = sign*2*Y/(1+X**2+Y**2)
#        iz = sign*(X**2+Y**2-1)/(1+X**2+Y**2)
#        #print(ix, iy, iz)
#    
#        Kx,Ky,Kz,_ = x.parent().gens()
#        u, v = stereo(x/w, y/w, z/w)
#    
#        u = u.subs({Kx:ix, Ky:iy, Kz:iz})
#        v = v.subs({Kx:ix, Ky:iy, Kz:iz})
#        #print("u =", u)
#        #print("v =", v)
#
#        T = sage.PolynomialRing(base, "z zb".split())
#        T = sage.FractionField(T)
#        z, zb = T.gens()
#
#        real = half*(z+zb)
#        imag = half*(-w4)*(z-zb)
#        uz = u.subs({X:real,Y:imag}) 
#        vz = v.subs({X:real,Y:imag}) 
#        f = uz + w4*vz
#
#        f = simplify(f)
#
#        if verbose:
#            pprint(f)
#        #assert f.subs({zb:1234}) == f
#        assert "zb" not in str(f)
#
#        pyfunc = lambda u : eval("lambda z: %s"%pystr(u))
#        py_f = pyfunc(f)
#        self.f = py_f

        assert top
        f = self.build()
        pyfunc = lambda u : eval("lambda z: %s"%pystr(u))
        py_f = pyfunc(f)
        self.py_f = py_f

        if not hasattr(f, "numerator"):
            return

        #print("f =", mkstr(f))

        top = (f.numerator())
        bot = (f.denominator())
        py_top = pyfunc(top)
        py_bot = pyfunc(bot)

        #print(top, "==", z*bot)
        #print("\t", top == z*bot)
        if "z" not in str(f).replace("zeta","") :
            return

        z = f.parent().gens()[0]
        try:
            df = diff(f, z)
        except TypeError:
            print("TypeError:", f)
            raise
        py_df = pyfunc(df)

        for val in [0,1,-1,1j,-1j]:
            try:
                fval = py_f(val)
            except ZeroDivisionError:
                continue
            result = eq(val, fval)
            result = "   FIX" if result else "NONFIX"
            #print(result, val, istereo(val.real, val.imag), fval)

        if df == 0:
            return

        gz = top.parent().gens()[0]
        tops = []
        bots = []
        t,b = top, bot
        while t != 0 or b != 0:
            tops.append(t)
            bots.append(b)
            t = diff(t, gz)
            b = diff(b, gz)

        py_tops = [pyfunc(t) for t in tops]
        py_bots = [pyfunc(b) for b in bots]

        for val, cval, m in find_roots(df):
            #print()
            #print("df(z)  =", py_df(cval))
            #print("\t", eq(py_dtop(cval)*py_bot(cval), py_dbot(cval)*py_top(cval)))
            #print([eq(pyt(cval), cval*pyb(cval)) for (pyt,pyb) in zip(py_tops, py_bots)])
            X = cval.real
            Y = cval.imag
            x, y, z = istereo(X, Y)
            #x, y, z = (x, sign*y, sign*z)
            x, y, z = (x, y, z)
            yield (x, y, z, m, val)
                


class GateDistill(Distill):
    def __init__(self, op):
        self.op = op

    @cache
    def get_projective_variety(self):
        op = self.op
        M,N = op.shape
        n = bitlog(N)
        assert M==2

        K = sage.PolynomialRing(base, list("xyzw"))
        Kx, Ky, Kz, Kw = K.gens()
    
        rho = half*(Kw*I + Kx*X + Ky*Y + Kz*Z)
        assert rho.trace() == Kw
    
        #print("rho^n ... ", end='', flush=True)
        rho = reduce(matmul, [rho]*n)
        print("rho.trace", rho.trace() )
        #print("P*rho*Pd ... ", end='', flush=True)
        sho = op*rho*op.d
        #print("trace ... ", end='', flush=True)

        scale = N//2
        assert N%2==0

        x = scale*(sho*X).trace()
        y = scale*(sho*Y).trace()
        z = scale*(sho*Z).trace()
        w = scale*sho.trace()
    
        print(r"\begin{align*}")
        print("x", right_arrow, mkstr(x))
        print("y", right_arrow, mkstr(y))
        print("z", right_arrow, mkstr(z))
        print("w", right_arrow, mkstr(w))
        print(r"\end{align*}")
    
        return x, y, z, w

    @cache
    def get_variety(self, projective=False):
        if projective:
            return self.get_projective_variety()

        op = self.op
        M,N = op.shape
        n = bitlog(N)
        assert M==2

        K = sage.PolynomialRing(base, list("xyzw"))
        Kx, Ky, Kz, Kw = K.gens()
    
        Kw = 1
        rho = half*(Kw*I + Kx*X + Ky*Y + Kz*Z)
        assert rho.trace() == Kw
    
        #print("rho^n ... ", end='', flush=True)
        rho = reduce(matmul, [rho]*n)
        assert rho.trace() == 1
        #print("P*rho*Pd ... ", end='', flush=True)
        sho = op*rho*op.d
        #print("trace ... ", end='', flush=True)
    
        x = (sho*X).trace()
        y = (sho*Y).trace()
        z = (sho*Z).trace()
        w = sho.trace()
    
        #print("x", right_arrow, x)
        #print("y", right_arrow, y)
        #print("z", right_arrow, z)
        #print("w", right_arrow, w) # div
        #print()

#        S = sage.PolynomialRing(sage.QQbar, list("xyzw"))
#        x = S(x)
#        y = S(y)
#        z = S(z)
#        w = S(w)
#    
#        print("x", right_arrow, x)
#        print("y", right_arrow, y)
#        print("z", right_arrow, z)
#        print("w", right_arrow, w) # div

        return x, y, z, w


class PauliDistill(Distill): # much faster than CodeDistill
    def __init__(self, code):
        assert code.k == 1
        Distill.__init__(self, code.n)
        self.code = code

    @cache
    def get_variety(self, projective=False, verbose=False):
        code = self.code
        n = code.n

        result = pauli.get_wenum(code)
        #print(result)

        if not projective:
            R = result[0].parent()
            w = R.gens()[3]
            result = [p.subs({w:1}) for p in result]

        x, y, z, w = result

        #verbose = True

        vprint(verbose, "x", right_arrow, x)
        vprint(verbose, "y", right_arrow, y)
        vprint(verbose, "z", right_arrow, z)
        vprint(verbose, "w", right_arrow, w) # div

        return x, y, z, w



class CodeDistill(Distill):
    def __init__(self, code):
        assert code.k == 1
        Distill.__init__(self, code.n)
        self.code = code

    @cache
    def get_variety(self, projective=False):
        code = self.code
        n = code.n
        space = Clifford(n)
        M = 2**code.m
    
        P = get_projector(code)
        Pd = P.d
    
        K = sage.PolynomialRing(base, list("xyzw"))
        Kx, Ky, Kz, Kw = K.gens()
    
        if not projective:
            Kw = 1
        rho = half*(Kw*I + Kx*X + Ky*Y + Kz*Z)
        assert rho.trace() == Kw
    
        #print("rho^n ... ", end='', flush=True)
        rho = reduce(matmul, [rho]*n)
        if not projective:
            assert rho.trace() == 1
        #print("P*rho*Pd ... ", end='', flush=True)
        rho = P*rho*Pd
        #print("trace ... ", end='', flush=True)
        div = rho.trace()
    
        L = strop(code.L, "I").split()
        LX = space.get_pauli(L[0])
        LZ = space.get_pauli(L[1])
        LY = w4*LX*LZ
    
        x = M*(rho*LX).trace()
        y = M*(rho*LY).trace()
        z = M*(rho*LZ).trace()
        w = M*div
    
        print(r"\begin{align*}")
        print("x", right_arrow, mkstr(x))
        print("y", right_arrow, mkstr(y))
        print("z", right_arrow, mkstr(z))
        print("w", right_arrow, mkstr(w)) # div
        print(r"\end{align*}")
    
        return x, y, z, w


class MultiDistill(Distill):
    def __init__(self, code):
        assert code.k == 1
        Distill.__init__(self, code.n)
        self.code = code

    def slow_xyzw(self, projective=False):
        code = self.code
        n = code.n
        space = Clifford(n)
        M = 2**code.m
    
        P = get_projector(code)
        Pd = P.d

        assert not projective
        Kw = 1
    
        gens = []
        for i in range(n):
            gens.append("x%d"%i)
            gens.append("y%d"%i)
            gens.append("z%d"%i)
        K = sage.PolynomialRing(base, gens)
        gens = K.gens()
        #print(gens)
    
        clifford = Clifford(1)
        I = clifford.I
        X = clifford.X()
        Y = clifford.Y()
        Z = clifford.Z()

        rhos = []
        for i in range(n):
            Kx, Ky, Kz = gens[3*i:3*i+3]
            rho = half*(Kw*I + Kx*X + Ky*Y + Kz*Z)
            assert rho.trace() == Kw
            #print(rho)
            rhos.append(rho)
    
        #print("rho^n ... ", end='', flush=True)
        rho = reduce(matmul, rhos)
        if not projective:
            assert rho.trace() == 1
        #print("P*rho*Pd ... ", end='', flush=True)
        rho = P*rho*Pd
        #print("trace ... ", end='', flush=True)
        div = rho.trace()
    
        L = strop(code.L, "I").split()
        LX = space.get_pauli(L[0])
        LZ = space.get_pauli(L[1])
        LY = w4*LX*LZ
    
        x = M*(rho*LX).trace()
        y = M*(rho*LY).trace()
        z = M*(rho*LZ).trace()
        w = M*div
        return x,y,z,w

    def fast_xyzw(self, projective=False):
        code = self.code
        n = self.n
        x,y,z,w = pauli.get_full_wenum(code)

        if projective:
            return x,y,z,w

        K = x.parent()
        gens = K.gens()

        hens = []
        for i in range(n):
            hens.append("x%d"%i)
            hens.append("y%d"%i)
            hens.append("z%d"%i)
        K = sage.PolynomialRing(base, hens)
        hens = K.gens()
        #print(gens)

        subs = {}
        for i in range(n):
            # x,y,z,w -> x,y,z,1
            subs[gens[4*i+1]] = hens[3*i+0] # x
            subs[gens[4*i+2]] = hens[3*i+1] # y
            subs[gens[4*i+3]] = hens[3*i+2] # z
            subs[gens[4*i+0]] = 1
        x,y,z,w = [p.subs(subs) for p in [x,y,z,w]]

        x,y,z,w = [K(p) for p in [x,y,z,w]]
        return x,y,z,w

#    def build(self, projective=False):
#        n = self.n
#
#        #x,y,z,w = self.slow_xyzw(projective)
#        x,y,z,w = self.fast_xyzw(projective)
#
#        #print("x", right_arrow, mkstr(x))
#        #print("y", right_arrow, mkstr(y))
#        #print("z", right_arrow, mkstr(z))
#        #print("w", right_arrow, mkstr(w)) # div
#
#        K = x.parent()
#        gens = K.gens()
#    
#        u, v = stereo(x/w, y/w, z/w)
#        #print("u =", u)
#        #print("v =", v)
#
#        hens = []
#        for i in range(n):
#            hens.append("X%d"%i) # real
#            hens.append("Y%d"%i) # imag
#        R = sage.PolynomialRing(base, hens)
#        hens = R.gens()
#        subs = {}
#        for i in range(n):
#            X, Y = hens[2*i:2*i+2]
#            ix = 2*X/(1+X**2+Y**2)
#            iy = 2*Y/(1+X**2+Y**2)
#            iz = (X**2+Y**2-1)/(1+X**2+Y**2)
#            subs[gens[3*i]] = ix
#            subs[gens[3*i+1]] = iy
#            subs[gens[3*i+2]] = iz
#
#        print("u.subs")
#        u = u.subs(subs)
#        print("v.subs")
#        v = v.subs(subs)
#
#        kens = []
#        for i in range(n):
#            kens.append("z%d"%i) # complex
#            kens.append("zb%d"%i) # complex conjugate
#        T = sage.PolynomialRing(base, kens)
#        T = sage.FractionField(T)
#        kens = T.gens()
#
#        subs = {}
#        for i in range(n):
#            z, zb = kens[2*i:2*i+2]
#            real = half*(z+zb)
#            imag = half*(-w4)*(z-zb)
#            subs[hens[2*i]] = real
#            subs[hens[2*i+1]] = imag
#        print("uz = u.subs")
#        uz = u.subs(subs)
#        print("vz = v.subs")
#        vz = v.subs(subs)
#        print("f = uz + w4*vz")
#        f = uz + w4*vz
#        assert f.subs({zb:1234}) == f
#        assert "zb" not in str(f)
#
#        f = simplify(f)
#
#        self.f = f
#        self.gens = [kens[2*i] for i in range(n)]
#
#        return f


    def build(self, projective=False):
        n = self.n

        #x,y,z,w = self.slow_xyzw(projective)
        x,y,z,w = self.fast_xyzw(projective)

        #print("x", right_arrow, mkstr(x))
        #print("y", right_arrow, mkstr(y))
        #print("z", right_arrow, mkstr(z))
        #print("w", right_arrow, mkstr(w)) # div

        K = x.parent()
        gens = K.gens()
    
        u, v = stereo(x/w, y/w, z/w)
        #print("u =", u)
        #print("v =", v)

        #hens = []
        #for i in range(n):
        #    hens.append("X%d"%i) # real
        #    hens.append("Y%d"%i) # imag
        #R = sage.PolynomialRing(base, hens)
        #hens = R.gens()

        kens = []
        for i in range(n):
            kens.append("z%d"%i) # complex
            kens.append("zb%d"%i) # complex conjugate
        base = sage.I.parent()
        w4 = sage.I
        T = sage.PolynomialRing(base, kens)
        T = sage.FractionField(T)
        kens = T.gens()

        hens = []
        for i in range(n):
            z, zb = kens[2*i:2*i+2]
            real = half*(z+zb)
            imag = half*(-w4)*(z-zb)
            hens.append(real)
            hens.append(imag)

        subs = {}
        for i in range(n):
            X, Y = hens[2*i:2*i+2]
            ix = 2*X/(1+X**2+Y**2)
            iy = 2*Y/(1+X**2+Y**2)
            iz = (X**2+Y**2-1)/(1+X**2+Y**2)
            subs[gens[3*i]] = ix
            subs[gens[3*i+1]] = iy
            subs[gens[3*i+2]] = iz

        #print("uz = u.subs")
        uz = u.subs(subs)
        #print("vz = v.subs")
        vz = v.subs(subs)
        #print("T(uz);T(vz)")
        uz = T(uz)
        vz = T(vz)
        #print("f = uz + w4*vz")
        f = uz + w4*vz
        #print("assert")
        #assert f.subs({zb:1234}) == f
        assert "zb" not in str(f)

        #print("simplify")
        f = simplify(f)

        self.f = f
        self.gens = [kens[2*i] for i in range(n)]

        return f


def test_stereo():
    def normalize(x,y,z):
        r = (x*x+y*y+z*z)**(1/2)
        x, y, z = (x/r,y/r,z/r)
        return x,y,z
    
    def rnd():
        x,y,z = numpy.random.normal(0, 1, 3)
        x, y, z = normalize(x,y,z)
        return (x,y,z)

    for _ in range(100):

        for sign in [+1, -1]:
    
            x,y,z = rnd()
            #print(x,y,z)
            u,v = stereo(x,sign*y,sign*z)
            x1, y1, z1 = istereo(u,v)
            #print(x1,y1,z1)
            assert abs(x-x1) < EPSILON, sign
            assert abs(y-sign*y1) < EPSILON, sign
            assert abs(z-sign*z1) < EPSILON, sign

test_stereo()


def test_mobius():

    R = sage.PolynomialRing(sage.QQ, list("wz"))
    w,z = R.gens()

    K = sage.FractionField(R)
    w = K(w)
    z = K(z)

    Hz = (z+1)//(z-1)
    Hw = (w+1)//(w-1)
    w = Hz.subs({z:(Hz*Hw)})

    print(w)

    mul = eval("lambda w,z : %s"%pystr(w))

    a,b,c = [rnd() for i in range(3)]
    print(mul(a,mul(b,c)), mul(mul(a,b), c))

    print( a, mul( 0, a ) )
    print( a, mul( 1, a ) )
    print( a, mul( -1, a ) )


def get_code(code=None, idx=0, verbose=True):
    idx = argv.get("idx", idx)
    params = argv.get("code", code)
    if params == (3,1,1):
        code = [
            QCode.fromstr("ZZZ XXI"),
        ][idx]
    if params == (4,1,2):
        code = [
            QCode.fromstr("YYZI IXXZ ZIYY"), # 0
            QCode.fromstr("XXXX ZZZZ YYII"), # 1
            QCode.fromstr("XYZI IXYZ ZIXY"), # 2
            QCode.fromstr("ZZZZ XXII IIXX", None, "XIXI ZZII"), # 3
        ][idx]
    if params == (5,1,1):
        code = [
        QCode.fromstr("""
        XZX.Z
        ZXX.Z
        ZZ.XX
        ZZZ..
        """),
        QCode.fromstr("""
        Y.XXY
        ZXXXZ
        ZZ.Z.
        ZZZ..
        """, None, """
        YYIII
        IXXXI"""),
        ][idx]

    if params == (5,1,2) and idx == 0:
        code = construct.get_512()
    if params == (5,1,2) and idx == 1:
        code = QCode.fromstr("""
        XIXIX
        IXXXI
        ZZZII
        IIZZZ
        """, None, "XXIII ZIIIZ")
        space = code.space
        #code = space.H(1)*space.H(3) * code
    if params == (5,1,3):
        #code = construct.get_513()
        #code = QCode.fromstr("XZZX.  .XZZX X.XZZ ZX.XZ", None, "ZXZII YZYII")
        code = QCode.fromstr("XZZX.  .XZZX X.XZZ ZX.XZ", None, "XXXXX ZZZZZ")
    if params == (6,1,2):
        code = QCode.fromstr("""
        X.ZZX.
        .X.ZXZ
        ZZX..X
        Z..X.X
        ZZ..Z.
        """)

    if params == (7,1,3):
        code = [
            construct.get_713(),
            QCode.fromstr("""
        XXIZIZI
        IXXIZIZ
        ZIXXIZI
        IZIXXIZ
        ZIZIXXI
        IZIZIXX""")][idx]

    if params == (8,1,3):
        code = QCode.fromstr("""
        YYZZIIZZ
        ZYYZZIIZ
        ZZYYZZII
        IZZYYZZI
        IIZZYYZZ
        ZIIZZYYZ
        ZZIIZZYY""", None)
    if params == (9,1,3) and idx == 0:
        code = construct.get_913() # Shor code
    if params == (9,1,3) and idx == 1:
        #code = construct.get_surface(3, 3)
        code = QCode.fromstr("""
        ZZ.ZZ....
        .XX.XX...
        ...XX.XX.
        ....ZZ.ZZ
        XX.......
        .......XX
        ...Z..Z..
        ..Z..Z...
        """, None, "X..X..X.. ZZZ......")

    if params == (16,1,4):
        #code = construct.get_surface(4, 4)
        code = QCode.fromstr("""
        ZZ..ZZ..........
        .XX..XX.........
        ..ZZ..ZZ........
        ....XX..XX......
        .....ZZ..ZZ.....
        ......XX..XX....
        ........ZZ..ZZ..
        .........XX..XX.
        ..........ZZ..ZZ
        XX..............
        ..XX............
        ............XX..
        ..............XX
        ....Z...Z.......
        .......Z...Z....
        """, None, "X...X...X...X... ZZZZ............")


    if params == (10,1,2):
        code = QCode.fromstr("XXIXXIIXII IXXXIXIIXI IIIXXXXIIX "
        "ZZIIIIIIZI IIIZZIIIZI IZIZIIIIIZ IIZIIZIIIZ IIIZIZIZII IIIIZIZZII",
        None, "XXXXXXXIII ZZZZZZZIII")

    if params == (10,1,3):
        code = QCode.fromstr("""
        XX..Z..Z..
        .XX..Z..Z.
        ..XX..Z..Z
        Z..XX..Z..
        .Z..XX..Z.
        ..Z..XX..Z
        Z..Z..XX..
        .Z..Z..XX.
        ..Z..Z..XX""", None, "YZZY......  ZZZ...X...")

    if params == (11,1,3):
        H = """
        XIIIIXXIIIX
        IXIIIIXIIXX
        IIXIIIXXIIX
        IIIXIIXIXIX
        IIIIXXXXXXI
        ZIIIIZZIIIZ
        IZIIIIZIIZZ
        IIZIIIZZIIZ
        IIIZIIZIZIZ
        IIIIZZZZZZI
        """ # [[11,1,3]] s.d.
        code = QCode.fromstr(H, None, "XXXXXXXXXXX ZZZZZZZZZZZ".split())
    if params == (13,1,5):
        H = """
        XXZZIZIIIZIZZ
        ZXXZZIZIIIZIZ
        ZZXXZZIZIIIZI
        IZZXXZZIZIIIZ
        ZIZZXXZZIZIII
        IZIZZXXZZIZII
        IIZIZZXXZZIZI
        IIIZIZZXXZZIZ
        ZIIIZIZZXXZZI
        IZIIIZIZZXXZZ
        ZIZIIIZIZZXXZ
        ZZIZIIIZIZZXX
        """ # [[13,1,5]] gf4
        code = QCode.fromstr(H, None, "X"*13 + " " + "Z"*13)

    if params == (15,1,3):
        code = construct.get_15_1_3()
        #n = code.n
        #space = code.space
        #for idx in [0,1,2,4,8]:
        #    code = space.H(n-1-idx)*code # ???

    if params == (17,1,5):
        code = QCode.fromstr("""
        XIIIIIIIIXIIXIIXI
        IXIIIIIIIXIIXIIIX
        IIXIIIIIIIIXIXXII
        IIIXIIIIXXIIXIIII
        IIIIXIIIXIXIXXXXX
        IIIIIXIIIIXXIXIII
        IIIIIIXIXXXIIXXXX
        IIIIIIIXIIXXIIXII
        ZIIIIIIIIZIIZIIZI
        IZIIIIIIIZIIZIIIZ
        IIZIIIIIIIIZIZZII
        IIIZIIIIZZIIZIIII
        IIIIZIIIZIZIZZZZZ
        IIIIIZIIIIZZIZIII
        IIIIIIZIZZZIIZZZZ
        IIIIIIIZIIZZIIZII
        """)

    if params == (19,1,5):
        code = QCode.fromstr("""
        XIIIIIIIIXXXXXIXIXI
        IXIIIIIIIIIXXIXXXXX
        IIXIIIIIIXXXIXIXIXX
        IIIXIIIIIXIXIIIXXXI
        IIIIXIIIIXIXIIXIIII
        IIIIIXIIIIIIIXIIXXI
        IIIIIIXIIXIIIIXXIII
        IIIIIIIXIXXIXXXIXIX
        IIIIIIIIXIXIIIIIXXI
        ZIIIIIIIIZZZZZIZIZI
        IZIIIIIIIIIZZIZZZZZ
        IIZIIIIIIZZZIZIZIZZ
        IIIZIIIIIZIZIIIZZZI
        IIIIZIIIIZIZIIZIIII
        IIIIIZIIIIIIIZIIZZI
        IIIIIIZIIZIIIIZZIII
        IIIIIIIZIZZIZZZIZIZ
        IIIIIIIIZIZIIIIIZZI
        """)
        

    if params == (23,1,7):
        code = construct.get_golay(23)

    if params == (31,1,3):
        code = QCode.fromstr("""
X.X.X.X.X.X.X.X.X.X.X.X.X.X.X.X
.XX..XX..XX..XX..XX..XX..XX..XX
...XXXX....XXXX....XXXX....XXXX
.......XXXXXXXX........XXXXXXXX
...............XXXXXXXXXXXXXXXX
.ZZZZ..........................
Z.ZZ.Z.........................
ZZ.Z..Z........................
.ZZ....ZZ......................
Z.Z....Z.Z.....................
ZZ.....Z..Z....................
ZZZZ...Z...Z...................
Z..Z...Z....Z..................
.Z.Z...Z.....Z.................
..ZZ...Z......Z................
.ZZ............ZZ..............
Z.Z............Z.Z.............
ZZ.............Z..Z............
ZZZZ...........Z...Z...........
Z..Z...........Z....Z..........
.Z.Z...........Z.....Z.........
..ZZ...........Z......Z........
ZZZ....Z.......Z.......Z.......
Z......Z.......Z........Z......
.Z.....Z.......Z.........Z.....
..Z....Z.......Z..........Z....
...Z...Z.......Z...........Z...
.ZZZ...Z.......Z............Z..
Z.ZZ...Z.......Z.............Z.
ZZ.Z...Z.......Z..............Z
    """, None, "X"*31+" "+"Z"*31)

    if params == (31,1,7) and idx==0:
        code = QCode.fromstr("""
        XIIXIIXIXIIIXXIXXIIIIIIIIIIIIII
        IIXIIXIXIIIXXIXXIIIIIIIIIIIIIIX
        IXIIXIXIIIXXIXXIIIIIIIIIIIIIIXI
        XIIXIXIIIXXIXXIIIIIIIIIIIIIIXII
        IIXIXIIIXXIXXIIIIIIIIIIIIIIXIIX
        IXIXIIIXXIXXIIIIIIIIIIIIIIXIIXI
        XIXIIIXXIXXIIIIIIIIIIIIIIXIIXII
        IXIIIXXIXXIIIIIIIIIIIIIIXIIXIIX
        XIIIXXIXXIIIIIIIIIIIIIIXIIXIIXI
        IIIXXIXXIIIIIIIIIIIIIIXIIXIIXIX
        IIXXIXXIIIIIIIIIIIIIIXIIXIIXIXI
        IXXIXXIIIIIIIIIIIIIIXIIXIIXIXII
        XXIXXIIIIIIIIIIIIIIXIIXIIXIXIII
        XIXXIIIIIIIIIIIIIIXIIXIIXIXIIIX
        IXXIIIIIIIIIIIIIIXIIXIIXIXIIIXX
        ZIIZIIZIZIIIZZIZZIIIIIIIIIIIIII
        IIZIIZIZIIIZZIZZIIIIIIIIIIIIIIZ
        IZIIZIZIIIZZIZZIIIIIIIIIIIIIIZI
        ZIIZIZIIIZZIZZIIIIIIIIIIIIIIZII
        IIZIZIIIZZIZZIIIIIIIIIIIIIIZIIZ
        IZIZIIIZZIZZIIIIIIIIIIIIIIZIIZI
        ZIZIIIZZIZZIIIIIIIIIIIIIIZIIZII
        IZIIIZZIZZIIIIIIIIIIIIIIZIIZIIZ
        ZIIIZZIZZIIIIIIIIIIIIIIZIIZIIZI
        IIIZZIZZIIIIIIIIIIIIIIZIIZIIZIZ
        IIZZIZZIIIIIIIIIIIIIIZIIZIIZIZI
        IZZIZZIIIIIIIIIIIIIIZIIZIIZIZII
        ZZIZZIIIIIIIIIIIIIIZIIZIIZIZIII
        ZIZZIIIIIIIIIIIIIIZIIZIIZIZIIIZ
        IZZIIIIIIIIIIIIIIZIIZIIZIZIIIZZ
        """, None, "X"*31+" "+"Z"*31)

    if params == (31,1,7) and idx==1:
        code = QCode.fromstr("""
        X.X.X.X.X.X.X.X.X.X.X.X.X.X.X.X
        .XX..XX..XX..XX..XX..XX..XX..XX
        ...XXXX....XXXX....XXXX....XXXX
        .......XXXXXXXX........XXXXXXXX
        ...............XXXXXXXXXXXXXXXX
        ..X...X...X...X...X...X...X...X
        ....X.X.....X.X.....X.X.....X.X
        ........X.X.X.X.........X.X.X.X
        ................X.X.X.X.X.X.X.X
        .....XX......XX......XX......XX
        .........XX..XX..........XX..XX
        .................XX..XX..XX..XX
        ...........XXXX............XXXX
        ...................XXXX....XXXX
        .......................XXXXXXXX
        Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.Z.Z
        .ZZ..ZZ..ZZ..ZZ..ZZ..ZZ..ZZ..ZZ
        ...ZZZZ....ZZZZ....ZZZZ....ZZZZ
        .......ZZZZZZZZ........ZZZZZZZZ
        ...............ZZZZZZZZZZZZZZZZ
        ..Z...Z...Z...Z...Z...Z...Z...Z
        ....Z.Z.....Z.Z.....Z.Z.....Z.Z
        ........Z.Z.Z.Z.........Z.Z.Z.Z
        ................Z.Z.Z.Z.Z.Z.Z.Z
        .....ZZ......ZZ......ZZ......ZZ
        .........ZZ..ZZ..........ZZ..ZZ
        .................ZZ..ZZ..ZZ..ZZ
        ...........ZZZZ............ZZZZ
        ...................ZZZZ....ZZZZ
        .......................ZZZZZZZZ
            """, None, "X"*31+" "+"Z"*31) # same as idx==1 ?!?
    
    if params == (47,1,11):
        code = QCode.fromstr("""
XXIIXIXIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIII
XIIXIXIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIX
IIXIXIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXX
IXIXIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXI
XIXIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXII
IXIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIX
XIIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXI
IIXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIX
IXIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXI
XIIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXII
IIXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIX
IXXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXI
XXIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXII
XIXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIX
IXXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXX
XXIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXI
XIIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIX
IIXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIXX
IXXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIXXI
XXIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIXXII
XIIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIXXIIX
IIIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIXXIIXX
IIXIIIIIIIIIIIIIIIIIIIIIIXXIIXIXIIXIIXXIXXIIXXI
ZZIIZIZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIII
ZIIZIZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZ
IIZIZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZ
IZIZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZI
ZIZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZII
IZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZ
ZIIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZI
IIZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZ
IZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZI
ZIIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZII
IIZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZ
IZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZI
ZZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZII
ZIZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZ
IZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZ
ZZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZI
ZIIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZ
IIZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZZ
IZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZZI
ZZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZZII
ZIIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZZIIZ
IIIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZZIIZZ
IIZIIIIIIIIIIIIIIIIIIIIIIZZIIZIZIIZIIZZIZZIIZZI
        """, None, "X"*47+" "+"Z"*47)
    if params == (49,1,9):
        code = QCode.fromstr("""
XXXIXIIIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIII
XXIXIIIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIX
XIXIIIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXX
IXIIIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXX
XIIIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXI
IIIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIX
IIIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXI
IIIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXII
IIIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIII
IIIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIII
IIIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIII
IIIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIII
IIXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIII
IXXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIII
XXXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIII
XXIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIX
XIXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXX
IXIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXX
XIIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXXI
IIXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXXIX
IXXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXXIXI
XXXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXXIXII
XXIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXXIXIIX
XIXIIIIIIIIIIIIIIIIIIIIIIIXXXIXIIIIIIIIIXXXIXIIXX
ZZZIZIIIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIII
ZZIZIIIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZ
ZIZIIIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZ
IZIIIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZ
ZIIIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZI
IIIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZ
IIIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZI
IIIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZII
IIIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIII
IIIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIII
IIIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIII
IIIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIII
IIZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIII
IZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIII
ZZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIII
ZZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZ
ZIZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZ
IZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZ
ZIIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZI
IIZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZIZ
IZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZIZI
ZZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZIZII
ZZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZIZIIZ
ZIZIIIIIIIIIIIIIIIIIIIIIIIZZZIZIIIIIIIIIZZZIZIIZZ
        """, None, "X"*49+" "+"Z"*49)

    if argv.concat:
        smap = SMap()
        lhs = QCode.fromstr("ZZI IZZ", None, "XXX ZII")
        smap[0,0] = (lhs.longstr(False))
        rhs = QCode.fromstr("XXI IXX", None, "XII ZZZ")
        smap[0,8] = (rhs.longstr(False))
        print(smap)
        #print(rhs.longstr())
        print( "lhs =", PauliDistill(lhs).build() )
        print( "rhs =", PauliDistill(rhs).build() )
        code = lhs.concat(rhs)

    if argv.stab:
        assert code is None
        code = QCode.fromstr(argv.stab, argv.destab, argv.logical)

    if argv.xy:
        space = code.space
        S = space.S()
        H = space.H()
        code = H*S*H * code

    assert code is not None

    if verbose:
        print(code)
        print(code.longstr(False))

    return code


def orbit():
    code = get_code()
    distill = PauliDistill(code)
    f = distill.build()

    print(f)
    print(latex(f))
    print(sage.factor(f.numerator()), "/", sage.factor(f.denominator()))

    best = None
    for g in Meromorphic.Clifford:
        gf = g*f
        s = str(gf.f)
        print("  ", s, len(s))
        if best is None or len(best)>len(s):
            best = s
        top = gf.f.numerator()
        bot = gf.f.denominator()
        #print("\t\t", sage.factor(top), "/", sage.factor(bot))
    print()
    print("best:", best)


def wenum():
    code = get_code()

    result = pauli.get_wenum(code)
    for p in result:
        pass

    ring = p.parent()
    x,y,z,w = ring.gens()
    for p in result:
        print(p)
        print(p.subs({x:x, y:0, z:0, w:1}))
        print()
    #print(sage.factor(p))

    return

    print()
    print(p.subs({x:1, y:1, z:1, w:1}))
    print(p.subs({x:z, y:1, z:1, w:1}))
    print(p.subs({x:1, y:z, z:1, w:1}))
    print(p.subs({x:1, y:1, z:z, w:1}))
    print(p.subs({x:1, y:1, z:1, w:z}))
    print(p.subs({x:z, y:z, z:1, w:1}))
    print(p.subs({x:z, y:1, z:z, w:1}))
    print(p.subs({x:z, y:1, z:1, w:z}))
    print(p.subs({x:1, y:z, z:z, w:1}))
    print(p.subs({x:1, y:z, z:1, w:z}))
    print(p.subs({x:1, y:1, z:z, w:z}))
    print(p.subs({x:z, y:z, z:z, w:1}))
    print(p.subs({x:z, y:z, z:1, w:z}))
    print(p.subs({x:z, y:1, z:z, w:z}))
    print(p.subs({x:1, y:z, z:z, w:z}))
    print(p.subs({x:z, y:z, z:z, w:z}))

    return

    distill = PauliDistill(code)
    f = distill.build()

    print(f)
    print(latex(f))
    print(sage.factor(f.numerator()), "/", sage.factor(f.denominator()))


def test_diff():

    code = get_code()
    distill = PauliDistill(code)
    f = distill.build()
    print("f =", f)

    z = f.parent().gens()[0]

    df = diff(f, z)
    print("df =", df)

def full_wenum():
    code = get_code()
    result = pauli.get_full_wenum(code)

def show_wenum():
    code = get_code()
    result = pauli.get_wenum(code)
    #print(result)
    x, y, z, w = result
    R = x.parent()
    x,y,z,w = (R.gens())
    for p in result:
        print(p, ":", p.subs({x:1,y:1,z:1,w:1}))


def dessin():

    #  stab="XXXX ZZII IIZZ" logical="XXII ZIZI"

    code = get_code()
    distill = PauliDistill(code)

    f = distill.build()
    print(f, right_arrow, end=" ")

    f = (f+1)/2
    print(f)

    print("f(z) == 0:")
    for z,zv,m in find_roots(f):
        print("\tz =", z, "degree", m)

    print("f(z) == 1:")
    for z,zv,m in find_roots(f-1):
        print("\tz =", z, "degree", m)


def multi():
    code = get_code()
    distill = MultiDistill(code)
    distill.build()
    f = distill.f
    gens = distill.gens
    print(f)

    R = sage.PolynomialRing(base, ["z"])
    F = sage.FractionField(R)
    z = F.gens()[0]
    fz = f.subs({g:z for g in gens})
    print(fz)
    dfz = diff(fz, z)


    df = [diff(f, z) for z in gens]

    params = code.n, code.k, code.d
    z0 = argv.get("z0")
    if z0 is not None:
        pass
    elif params == (5,1,3):
        z0 = 1.3660254037844386+1.3660254037844386j
    elif params == (5,1,2):
        z0 = 1 # df == 0
    elif params == (7,1,3):
        z0 = 0.
    else:
        z0 = 0.
    print("z0 =", z0)
    subs = {g:z0 for g in gens}
    df = [dfi.subs(subs) for dfi in df]
    total = 0
    for dfi in df:
        print(eq(dfi, 0), "df(z0,...,z0) =", dfi)
        total += dfi
    print("total =", total)

    print("df(z0) =", dfz.subs({z:z0}) )


def find_clifford():
    # what states do 2 qubit Clifford's distill?
    c = Clifford(2)
    Z, S, H, CX = c.Z, c.S, c.H, c.CX

    #gen = [S(0), S(1), H(0), H(1), CX()]
    gen = [Z(0), Z(1), H(0), H(1), CX()]
    G = mulclose(gen, verbose=True, maxsize=None)
    print(len(G))

    R = sage.I.parent()
    Rz = sage.PolynomialRing(R, ["z"])
    K = sage.FractionField(Rz)
    #z = K.gens()[0]
    z = Rz.gens()[0]

    meas = ir2*Matrix(base, [[1,1]])
    meas = Matrix(base, [[1,0]])
    for g in G:
        op = (meas@I)*g
        distill = GateDistill(op)
        f = distill.build()
        f = K(f)
        p, q = f.numerator(), f.denominator()
        r = p - z*q
        if r.degree() < 2:
            continue
        df = diff(p, z)
        if df == 0:
            continue
        dfs = [u for (u,m) in sage.factor(df)]
        rs = [u for (u,m) in sage.factor(r)]
        #print(rs, "df:", dfs)
        match = ([u for u in dfs if u in rs])
        if match:
            print(match, "\tf=%s, r=%s"%(f,r))



def test():

    code = None

    if argv.CH:
        CH = clifford.I << clifford.H()
        plus = ir2*Matrix(base, [[1,1]])
        print(plus)
        assert (plus*plus.d)[0,0] == 1
        op = (plus@I)*CH
        print(op)
        # TODO fix the problem with sage rings, etc. somehow...
        #assert 0, "TODO"
        distill = GateDistill(op)

    elif argv.CX:
        CX = Clifford(2).CX()
        plus = ir2*Matrix(base, [[1,1]])
        ket0 = Matrix(base, [[1,0]])
        op = (plus@I)*CX
        distill = GateDistill(op)

    elif argv.CX1:
        CX = Clifford(2).CX()
        plus = ir2*Matrix(base, [[1,1]])
        ket0 = Matrix(base, [[1,0]])
        assert (plus*plus.d)[0,0] == 1
        op = (I@ket0)*CX
        distill = GateDistill(op)

    elif argv.CZ:
        CZ = Clifford(2).CZ()
        plus = ir2*Matrix(base, [[1,1]])
        ket0 = Matrix(base, [[1,0]])
        ket = plus
        assert (ket*ket.d)[0,0] == 1

        n = argv.get("n", 2)
        if n==2:
            op = (ket@I)*CZ
        elif n==3:
            op = (ket@ket@I)*(CZ@I)*(I@CZ)*Clifford(3).CZ(0,2)
        print(op)
        distill = GateDistill(op)

    else:
        code = get_code()
        #distill = CodeDistill(code)
        distill = PauliDistill(code)

    trials = argv.get("trials", 10000)
    nsols = argv.get("nsols", 10)
    verbose = argv.get("verbose", False)

    if argv.projective:
        distill.get_variety(projective=True)
        return

    f = distill.build()
    print(f)

    for top in [True]:
        print("--")
        found = 0
        for (x,y,z,m,val) in distill.fast_find(top, verbose=verbose):
            val = complex(val)
            fval = distill.py_f(complex(val))
            print(r"& %d \times (%.8f, %.8f, %.8f) \\ %% %s --> %s"%(
                m, x, y, z, val, fval))
            print("\t\t%% --> %s "%(istereo(fval.real, fval.imag),))
            found += m
        print(r"\begin{align*}")
        print(r"\end{align*}")
        print("total:", found)
        #break

    return

    x, y, z, w = distill.get_variety()
    pyfunc = lambda u : eval("lambda x,y,z,w=1.0: %s"%pystr(u))
    fx = pyfunc(x)
    fy = pyfunc(y)
    fz = pyfunc(z)
    fw = pyfunc(w)
    def func(x,y,z):
        rx,ry,rz,rw = (
            fx(x,y,z),
            fy(x,y,z),
            fz(x,y,z),
            fw(x,y,z))
        return (rx/rw, ry/rw, rz/rw)

    def norm(x,y,z):
        return (x*x+y*y+z*z)**(1/2)

    
    for top in [True, False]:
      for (x,y,z) in distill.find(trials, nsols, top, verbose=verbose):
        alpha = 1 - 0.01
        x0, y0, z0 = alpha*x, alpha*y, alpha*z
        x1, y1, z1 = func(x0,y0,z0)
        r0 = norm(x0, y0, z0)
        r1 = norm(x1, y1, z1)
        squeeze = (1-r0) / (1-r1)
        #print("%.8f --> %.8f"%(1-r0, 1-r1))
        print("%.8f, %.8f, %.8f"%(x, y, z), "squeeze = %.2f"%squeeze)
      print("--")


def getkey(val):
    val = complex(val)
    r = val.real
    i = val.imag
    if abs(r) < 1e-8:
        r = 0.
    if abs(i) < 1e-8:
        i = 0.
    return "(%.6f+%.6fj)"%(r, i)


def all_cyclic(n, k, d):
    from qumba import cyclic 
    for code in cyclic.all_cyclic(n):
        if code.k != k:
            continue
        if code.d < d:
            continue
        yield code


def search_cyclic():

    n = argv.get("n", 4)
    d = argv.get("d", 1)
    k = 1
    
    for code in all_cyclic(n, k, d):
        print(code)
        print(code.longstr(False))

        distill = PauliDistill(code)
        f = distill.build()
        print(f)
        break



def search_fix():

    n = argv.get("n", 4)
    d = argv.get("d", 1)
    k = 1
    fn = construct.all_codes
    if argv.css:
        fn = construct.all_css
    if argv.cyclic:
        fn = all_cyclic
    print(fn)
    print((n,k,d))

    r2 = 2**0.5
    skip = [0,1,-1,1j,-1j]
    skip += [1/r2+1j/r2]
    skip += [-1/r2+1j/r2]
    skip += [-1/r2-1j/r2]
    skip += [1/r2-1j/r2]
    for u in [r2+1,-(r2+1), r2-1, 1-r2]:
        skip.append(u)
        skip.append(u*1j)
    skip = set(getkey(x) for x in skip)
    #skip = set()

    found = 0
    for code in fn(n,k,d):
        code.build()
        assert code.L is not None
        #print(code)
        found += 1
        distill = PauliDistill(code)
        try:
            for (x,y,z,m,val) in distill.fast_find():
                val = complex(val)
                fval = distill.py_f(complex(val))
                if not eq(val, fval): # <<-- look for fix points
                    continue
                if getkey(val) not in skip:
                    print(code)
                    print(code.longstr())
                    print("\t", m, "x", val, "-->", fval, flush=True)
        except ArithmeticError:
            print("ArithmeticError")
    print("found:", found)


def search_poly():

    n = argv.get("n", 4)
    d = argv.get("d", 1)
    k = 1
    fn = construct.all_codes
    if argv.css:
        fn = construct.all_css
    if argv.cyclic:
        fn = all_cyclic
    print(fn)
    print((n,k,d))

    found = set()
    count = 0
    for code in fn(n,k,d):
        if argv.no_Y and "Y" in code.longstr():
            continue
        count += 1
        code.build()
        assert code.L is not None
        distill = PauliDistill(code)
        try:
            f0 = distill.build()
        except:
            print("distill.build: Fail")
            print(code.longstr(False))
            continue
        for u in Meromorphic.Clifford:
            try:
                f = (u*f0).f
            except ZeroDivisionError:
                continue
            s = str(f).replace("zeta", "")
            bot = str(f.denominator())
            if "z" in bot or "I" in s:
                continue
            if f not in found:
                print(f, code)
                found.add(f)
            if s == "z + 1":
                print(code.longstr(False))

    print("count:", count)


def search_unitary():

    n = argv.get("n", 2)

    wmul = Matrix(base, [[1,0,0,0],[0,0,0,1]])
    bmul = Matrix(base, [[1,0,0,1],[0,1,1,0]])

    zs = ["z%d"%i for i in range(n)]
    K = sage.PolynomialRing(base, zs)
    zs = K.gens()
    vecs = [Matrix(K, [[zs[i]],[1]]) for i in range(n)]

    v = reduce(matmul, vecs)

    def normalize(v):
        r = v[len(v)-1, 0]
        if r == 0:
            return v
        v = (1/r)*v
        return v

    c = Clifford(n)
    gens = [c.S(i) for i in range(n)]
    gens += [c.H(i) for i in range(n)]
    gens += [c.CZ(i,j) for i in range(n) for j in range(i+1,n)]

    #cliff = mulclose(gens, verbose=True, maxsize=None)
    #print()

    #print(normalize(bmul*v))

    bdy = [v]
    found = set(bdy)
    seen = set()
    while bdy:
        _bdy = []
        for g in gens:
            for v in bdy:
                gv = g*v
                if gv in found:
                    continue
                _bdy.append(gv)
                found.add(gv)
                #gv = normalize(gv)
                #for u in [wmul*gv, bmul*gv]:
                #    f = normalize(u)[0,0]
                f = normalize(gv)[0,0]
                try:
                    f = f.subs({zs[i]:zs[0] for i in range(n)})
                except ZeroDivisionError:
                    continue
                s = str(f)
                if "zeta" in s:
                    continue
                if s in seen:
                    continue
                seen.add(s)
                top = (f.numerator()) 
                bot = f.denominator()
                if "z" not in str(top) or "z" not in str(bot):
                    print(f)

        bdy = _bdy
        print("[%s, %s]"%(len(found), len(bdy)))

    return

    for g in cliff:
      for w in [wmul*g*v, bmul*g*v]:
        w = normalize(w)
        f = w[0,0]
        if "zeta" in str(f):
            continue
        try:
            f = f.subs({zs[i]:zs[0] for i in range(n)})
        except ZeroDivisionError:
            continue
        if f in found:
            continue
        found.add(f)
        print(f)


def search_expr():
    R = sage.I.parent()
    base = sage.PolynomialRing(R, list("wz"))
    K = sage.FractionField(base)
    w,z = K.gens()

    p = w*z
    q = (w*z + 1) / (w+z)

    clifford = [K(f.f) for f in Meromorphic.Clifford]

    gen = [0, 1, -1, sage.I, -sage.I]
    gen = [K(g) for g in gen]
    gen += [p, q] + clifford

    pairs = [(g,h) for g in gen for h in gen]

    expr = {}

    def dump(f):
        
        print("==============")
        print("dump", f)
        while 1:
            item = expr.get(f)
            print(f, "<---", item)
            if item is None:
                break
            f = item[0]
        print("==============")

    found = set(gen)
    bdy = list(found)
    while bdy and len(found) < 20000:

        _bdy = []
        for g in bdy:
            s = str(g)
            for h,k in pairs:
                try:
                    f = g.subs({w:h, z:k})
                except ZeroDivisionError:
                    continue
                #assert f != 2, str(f)
                if f in found: 
                    continue
                found.add(f); 
                assert f not in expr, f
                expr[f] = (g, h, k)
                if f==z+1 or f==w+z:
                    dump(f)
                if len(str(f))>20: 
                    continue
                #if h==1 and k==1:
                #    print("-->", f)
                top = f.numerator()
                if top.degree() > 4:
                    continue
                bot = f.denominator()
                if bot.degree() > 4:
                    continue
                _bdy.append(f)
                #if "/" not in str(f):
                if bot == 1:
                    print("\t", f)

        bdy = _bdy
        print("[%d, %d]"%(len(found), len(bdy)))


def search_heap():
    R = sage.I.parent()
    base = sage.PolynomialRing(R, list("wz"))
    K = sage.FractionField(base)
    w,z = K.gens()
    one = K.one()

    p = w*z
    q = (w*z + 1) / (w+z)

    clifford = [K(f.f) for f in Meromorphic.Clifford]

    gen = [0, 1, -1, sage.I, -sage.I]
    gen = [K(g) for g in gen]
    gen += [p, q] + clifford

    gen = [g for g in gen if "I" not in str(g)] # Real only
    clifford = [g for g in clifford if "I" not in str(g)] # Real only

    pairs = [(g,h) for g in gen for h in gen]

    expr = {}

    def dump(f):
        
        print()
        print("==============")
        print("dump", f)
        while 1:
            item = expr.get(f)
            print(f, "<---", item)
            if item is None:
                break
            f = item[0]
        print("==============")
        print()

    target = [w+z, w-z, -w+z, -w-z]
    target += [1/g for g in target]
    target = set(target)
    target.add(z/(w+z))
    target.add(w/(w+z))
    print(target)
    for h in list(target):
      for g in clifford:
        k = g.subs({z:h})
        target.add(k)
    print(target)
    print()

    pystr = lambda f : str(f).replace("^", "**")

    found = set(gen)
    bdy = list(found)
    count = 0
    while bdy and len(found) < 50000:
        count += 1

        N = len(bdy)
        #bdy.sort(key = lambda f: -len(str(f)))
        bdy.sort(key = lambda f: -(f.numerator().degree() + f.denominator().degree() + str(f).count("/")))

        if N < 10 or 1:
            g = bdy.pop()
        else:
            idx = randint(N - 10, N - 1)
            g = bdy.pop(idx)

        #g = simplify(g)
        #g = K(g)
        print(str(g).replace(" ", ""), end=" ", flush=True)
        #assert g not in target # wahh??
        if g in target:
            dump(g)
            return

        sg = pystr(g)
        for h,k in pairs:
            #try:
            #    f0 = g.subs({w:h, z:k})
            #except ZeroDivisionError:
            #    continue
            s1 = sg.replace("z", "Z")
            s1 = s1.replace("w", "("+pystr(h)+")")
            s1 = s1.replace("Z", "("+pystr(k)+")")
            s1 = "one*"+s1
            s1 = s1.replace("/", "//")
            try:
                f = eval(s1, {"w":w, "z":z, "one":one})
            except ZeroDivisionError:
                continue
            assert "." not in str(f)

            try:
                f = K(simplify(f))
            except TypeError:
                pass

            if f in found: 
                #print("(found %s)"%f, end=" ")
                continue
            #print("\n(%s --> %s)"%(f, f1))
            found.add(f); 
            assert f not in expr, f
            #if f not in expr:
            expr[f] = (g, h, k)
            if f in target:
                dump(f)
                return
            #if len(str(f))>20: 
            #    continue
            #if h==1 and k==1:
            #    print("-->", f)
            #top = f.numerator()
            #bot = f.denominator()
            #if top.degree() + bot.degree() > 10:
            #    continue
            s = str(f)
            if "8" in s:
                continue
            if "w" in s and "z" in s:
                bdy.append(f)
            #if "/" not in str(f):
            #if bot == 1:
            #    print("\t", f)

        if len(found) % 1000 == 0:
            print("\n[%d, %d]"%(len(found), len(bdy)))

    print("\n[%d, %d]"%(len(found), len(bdy)))




def search_compose():

    n = argv.get("n", 4)
    k = 1
    d = argv.get("d", 1)

    fn = construct.all_codes
    if argv.css:
        fn = construct.all_css
    print(fn)
    print((n,k,d))

    r2 = 2**0.5
    skip = [0,1,-1,1j,-1j]
    skip += [1/r2+1j/r2]
    skip += [-1/r2+1j/r2]
    skip += [-1/r2-1j/r2]
    skip += [1/r2-1j/r2]
    for u in [r2+1,-(r2+1), r2-1, 1-r2]:
        skip.append(u)
        skip.append(u*1j)
    skip = set(getkey(x) for x in skip)

    src = {}
    tgt = {}
    verts = set()

    found = 0
    for code in fn(n,k,d):
        code.build()
        assert code.L is not None
        distill = PauliDistill(code)
        for (x,y,z,m,val) in distill.fast_find():
            val = complex(val)
            fval = distill.py_f(complex(val))

            s = getkey(val)
            t = getkey(fval)

            if s in skip or t in skip:
                continue

            verts.add(s)
            verts.add(t)

            src.setdefault(s, set()).add(t)
            tgt.setdefault(t, set()).add(s)

            print("%s --> %s"%(s, t))

            t1 = src.get(t)
            if t1:
                #print("%s --> %s --> %s"%(s, t, t1))
                found += 1

            s0 = tgt.get(s)
            if s0:
                #print("%s --> %s --> %s"%(s0, s, t))
                found += 1

        #if found>10:
        #    break

    names = {}
    for i,v in enumerate(verts):
        names[v] = "v%d"%i

    dot = "distill_%s%s%s.dot"%(n,k,d)
    print("writing", dot)
    f = open(dot, "w")
    print("digraph {", file=f)
    for s in verts:
        for t in src.get(s, []):
            if src.get(t):
                print("  %s -> %s;"%(names[s], names[t]), file=f)
    print("}", file=f)

    print("found:", found)



def test_rho():

    #base = sage.CyclotomicField(4)
    #w4 = base.gens()[0]

    K = sage.PolynomialRing(base, list("xyzw"))
    x, y, z, w = K.gens()

    rho = half * Matrix(K, [
        [w+z, x-w4*y], [x+w4*y, w-z]
    ])

    assert rho == half * ( w*I + x*X + y*Y + z*Z )

    print(rho)
    assert rho.trace() == w

    print( "tr(rho**2) =", (rho*rho).trace() )

    u = Matrix(K, [[w+w4*x,y+w4*z]])
    ud = Matrix(K, [[w-w4*x,y-w4*z]]).t
    print(u)

    print( u*ud )

    sho = ud @ u
    print(sho, sho.shape)
    print(sho.trace())


def test_ring():

    H = Clifford(1).H()
    r = H[0,0]

    H = (1/r)*H
    #print(H)

    R = sage.PolynomialRing(sage.ZZ, "u v".split())
    K = sage.FractionField(R)

    u, v = K.gens()

    mul = lambda u,v : K(u)*K(v)
    add = lambda u,v : (K(u) + v) / (1 + u*v)

    
    for trial in range(10):
        a, b, c, d = [randint(-10, 10) for i in range(4)]
        #print (add(add(a,b), c) , add(a, add(b, c)))
        assert (add(add(a,b), c) == add(a, add(b, c)))

        print( mul(a, add(b, c)) == add(mul(a, b), mul(a, c)))


def test_density():
    N = 4

    R = sage.PolynomialRing(base, "a0 b0 c0 d0 a1 b1 c1 d1".split())
    gens = R.gens()
    u0 = gens[:N]
    u1 = gens[N:]

    rows = []
    coords = "abcd"
    for i in range(N):
        #r = [coords[i] + "*" + coords[j] + "^" for j in range(N)]
        r = [u0[i] * u1[j] for j in range(N)]
        rows.append(r)
    density = Matrix(R, rows)

    print(density)

    c = Clifford(1)
    I, X, Z, Y = c.I, c.X(), c.Z(), c.Y()
    #Pauli = mulclose([X@I, Z@I, I@X, I@Z])
    pauli = [I, X, Y, Z]

    rows = [[((g@h) * density).trace() for g in pauli] for h in pauli]
    op = Matrix(R, rows)

    M = op.subs({u0[3]:1, u1[3]:0})
    print(M)


def search_clifford():
    
    #S = Matrix(base, [[1,0],[0,w4]])

    S = Clifford(1).S()
    Z = S*S
    H = Clifford(1).H()

    b_bb = Matrix(base, [
        [1,0,0,1], 
        [0,1,1,0],
    ])


    w_ww = Matrix(base, [
        [1,0,0,0], 
        [0,0,0,1],
    ])

    W = Matrix(base, [
        [0,1,1,0], 
        [0,0,0,1],
    ])

    assert W.norm() < 2
    cliff = mulclose([S,H])
    assert len(cliff)==192

    W = Matrix(base, [
        [1,1],
        [0,1],
    ])

    target = set(g*W for g in cliff)

    b_ = Matrix(base, [[1],[0]])
    #assert b_bb * (b_@I) == I
    #assert b_bb * (I@b_) == I
    
    w_ = Matrix(base, [[1],[1]])
    assert ( w_ww * (w_@I) ) == I
    assert ( w_ww * (I@w_) ) == I

    _b = b_.t
    _w = w_.t
    bb_b = b_bb.t
    ww_w = w_ww.t

    single = [I, S, H, _b, b_, _w, w_] #, b_bb, w_ww] #, bb_b, ww_w]
    pair = [a@b for a in single for b in single]

    gen = single + pair
    gen += [Clifford(2).CX()]

    found = set(gen)
    bdy = list(found)
    while bdy:

        _bdy = []

        for g in bdy:
          for h in gen:
            if g.shape[1] != h.shape[0]:
                continue
            op = g*h
            if op in found:
                continue
            found.add(op)
            if op in target:
                print("FOUND!!!\n\n\n"*4)
                return
            #if len(found) > 10000:
            #    return
            #if op.shape[0]*op.shape[1] > 4*4:
            #    continue
            if op.shape[0] > 4 or op.shape[1] > 4:
                continue
            if op.norm() > 2:
                continue
            if "1/4" in str(op):
                continue
            _bdy.append(op)
            if len(_bdy)%100==0:
                print(op)

        bdy = _bdy
        print(len(found), len(bdy))

    print("found:", len(found))


def test_wnode():

    base = sage.CyclotomicField(6)
    w = base.gens()[0]

    I = Matrix(base, [[1,0],[0,1]])
    X = Matrix(base, [[0,1],[1,0]])
    Z3 = Matrix(base, [[1,0],[0,w]])

    b_bb = Matrix(base, [
        [1,0,0,1], 
        [0,1,1,0],
    ])

    W = Matrix(base, [
        [0,1,1,0], 
        [0,0,0,1],
    ])

    cap = Matrix(base, [[1,0,0,1]])
    cup = cap.t

    assert Z3**6 == I

    op = (Z3@I) * cup
    op = I @ op @ I
    op = (b_bb @ b_bb) * op
    op = (Z3@Z3) * op
    op = b_bb * op
    op = X * op


    r = 1 // (2*w - 1)
    op = r*op
    
    print(op)
    assert op == W


def test_512():

    from qumba.symplectic import SymplecticSpace

    n = 5

    space = SymplecticSpace(n)
    CX = space.CX
    H = space.H

    E = CX(4,3) * CX(2,3) * CX(2,1) * CX(0,1) 
    E = CX(0,4) * E
    E = H(1)*H(3) * E

    E = ~E

    print(E)

    code = QCode.from_encoder(E, k=1)
    print(code)
    print(code.longstr())


def test_H():

    base = sage.CyclotomicField(24)
    w24 = base.gens()[0]
    R = sage.PolynomialRing(base, "z")
    K = sage.FractionField(R)
    z = K.gens()[0]

    w24 = K(w24)
    w6 = w24**4
    assert w6**3 == -1
    w8 = w24**3
    w4 = w24**6
    assert w4**2 == -1

    r2 = w8 + w8**7
    assert r2**2 == 2

    H = Meromorphic.H.f
    SH = (Meromorphic.S * Meromorphic.H).f

    H = K(H)
    SH = K(SH)

    assert( H.subs({z : 1+r2}) == 1+r2 )
    assert( H.subs({z : 1-r2}) == 1-r2 )


def test_surface():
    R = sage.PolynomialRing(sage.ZZ, "z")
    z = R.gens()[0]
    f = z**16 + 4*z**14 + 16*z**12 + 44*z**10 + 126 * z**8 + 44*z**6 + 16*z**4 + 4*z**2 + 1

    for val,m in f.roots(ring=sage.CIF):
        print(val, m)


def test_poly():
    R = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(R)
    z = R.gens()[0]

    # Golay code wenum
    f = (z**23 + 506*z**15 + 1288*z**11 + 253*z**7) // (253*z**16 + 1288*z**12 + 506*z**8 + 1)

    # [[17,1,5]] wenum
    #f = (z**17 + 17*z**13 + 187*z**9 + 51*z**5) // (51*z**12 + 187*z**8 + 17*z**4 + 1)

    # CZ 
    #f = (z**2 + z) / (z-1)

    print(f)

    find(f)

    df = diff(f, z)

    top = df.numerator()
    bot = df.denominator()

    top = (sage.factor(top))
    print(r"\frac{%s}{(%s)^2}"%(latex(top), latex(f.denominator())))


@cache
def get_bits(m):
    rows = numpy.array(list(numpy.ndindex((2,)*m)), dtype=int)
    return rows

def classical_wenum(H):
    m, n = H.shape

    if 0:
        wenum = [0]*(n+1)
        for bits in numpy.ndindex((2,)*m):
            v = numpy.dot(bits, H)%2
            w = v.sum()
            assert w%4 == 0, str(v)
            wenum[w] += 1
        #print("wenum", wenum)

    A = get_bits(m)
    AH = numpy.dot(A, H)%2
    w = AH.sum(axis=1)
    assert (w%4).sum() == 0
    w = list(w)
    assert w[0] == 0
    #print(w.count(8))
    wenum = [0]*(n+1)
    #print(len(w), "/", end='', flush=True)
    for i in w:
        wenum[i] += 1
    #print("\\")
    #print(wenum)

    #assert 0

    base = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(base)
    z = R.gens()[0]

    bot = 0
    top = 0
    for (i,w) in enumerate(wenum):
        bot += w*(z**i)
        top += w*(z**(n-i))

    f = top // bot
    return f




def build_selfdual(code):
    #print(code)

    rows = strop(code.H).split()
    rows = [r for r in rows if "X" in r]
    rows = [r.replace("X","1").replace(".","0") for r in rows]
    H = [[int(c) for c in row] for row in rows]
    H = numpy.array(H)
    #print(H, H.shape)

    f = classical_wenum(H)
    return f



def find(f):
    base = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(base)
    z = R.gens()[0]

    f = R(f)

    top = f.numerator()
    bot = f.denominator()

    print(f)

    print("fixed points:")
    g = base(top-z*bot)
    for val,m in (g).roots(ring=sage.CIF):
        print("\t(%d)"%m, val)

    df = diff(f, z)
    top = df.numerator()
    bot = df.denominator()
    print(r"f'(z) = \frac{%s}{(%s)^2}"%(latex(sage.factor(top)), latex(f.denominator())))

    print("stationary points:")
    for val,m in top.roots(ring=sage.CIF):
        print("\t(%d)"%m, val, "->", f.subs({z:val}))


def test_find():

    code = get_code()
    distill = PauliDistill(code)
    f = distill.build()
    find(f)


def test_selfdual():
    code = get_code(verbose=False)
    f = build_selfdual(code)
    find(f)


def test_rm():
    from qumba.matrix import Matrix

    r = argv.get("r", 1)
    m = argv.get("m", 4)
    code = construct.reed_muller(r, m)
    print(code)
    if code is None:
        return

    css = code.to_css()
    Hx = css.Hx
    print("is_morthogonal(2)", is_morthogonal(Hx,2))
    print("is_morthogonal(3)", is_morthogonal(Hx,3))

    n = code.n

    H = strop(code.H)
    print(H)
    print()
    rows = H.split()
    rows = [r[1:] for r in rows if "X" in r]
    #rows.remove("X"*n) # logop
    H = '\n'.join(rows)
    #H = fromstr(H)
    print(H)
    Hx = Matrix.parse(H)
    Hz = Hx.kernel()
    print(str(Hz).replace("1", "Z"))

    #return

    code = QCode.fromstr("""
X.X.X.X.X.X.X.X.X.X.X.X.X.X.X.X
.XX..XX..XX..XX..XX..XX..XX..XX
...XXXX....XXXX....XXXX....XXXX
.......XXXXXXXX........XXXXXXXX
...............XXXXXXXXXXXXXXXX
.ZZZZ..........................
Z.ZZ.Z.........................
ZZ.Z..Z........................
.ZZ....ZZ......................
Z.Z....Z.Z.....................
ZZ.....Z..Z....................
ZZZZ...Z...Z...................
Z..Z...Z....Z..................
.Z.Z...Z.....Z.................
..ZZ...Z......Z................
.ZZ............ZZ..............
Z.Z............Z.Z.............
ZZ.............Z..Z............
ZZZZ...........Z...Z...........
Z..Z...........Z....Z..........
.Z.Z...........Z.....Z.........
..ZZ...........Z......Z........
ZZZ....Z.......Z.......Z.......
Z......Z.......Z........Z......
.Z.....Z.......Z.........Z.....
..Z....Z.......Z..........Z....
...Z...Z.......Z...........Z...
.ZZZ...Z.......Z............Z..
Z.ZZ...Z.......Z.............Z.
ZZ.Z...Z.......Z..............Z
    """, None, "X"*31+" "+"Z"*31)

    print(code)
    #css = code.to_css()
    #print(css.bz_distance()) # (15,3)
    
    #f = build_selfdual(code)
    #find(f)

    #distill = PauliDistill(code)
    #f = distill.build()
    #print("f(z) =", f)


def gen_latex():
    code = get_code(verbose=False)

    print(code.longstr(False))
    print("is_selfdual", code.is_selfdual())
    if code.is_css():
        css = code.to_css()
        Hx = css.Hx
        #print(Hx)
        print("is_morthogonal(2)", is_morthogonal(Hx, 2))
        print("is_morthogonal(3)", is_morthogonal(Hx, 3))
        print("strong_morthogonal(2)", strong_morthogonal(Hx, 2))
        print("strong_morthogonal(3)", strong_morthogonal(Hx, 3))

    base = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(base)
    z = R.gens()[0]

    if code.is_css() and code.is_selfdual() and not argv.se: # and code.is_doublyeven():
        f = build_selfdual(code)
    else:
        f = PauliDistill(code).build()

    if argv.Z:
        f = -f

    if argv.H:
        f = (Meromorphic.H * f).f

    f = R(f)

    if argv.SH:
        assert 0
        S, H = Meromorphic.S, Meromorphic.H
        f = (S*H * f).f

    top = f.numerator()
    bot = f.denominator()

    r = sage.factor(top - z*bot)

    df = sage.diff(f, z)
    df_top = df.numerator()
    df_bot = df.denominator()

    print(r"\begin{align*}")
    print("f(z) &= %s \\\\" % latex(f))
    print("r(z) &= %s \\\\" % latex(r))

    if df_bot == bot**2:
        print(r"f'(z) &= \frac{%s}{%s^2}"%(
            latex(sage.factor(df_top)), (latex( sage.factor(bot) ))))
    else:
        print(r"f'(z) &= \frac{%s}{%s} %% df_bot != bot**2"%(
            latex(sage.factor(df_top)), (latex( sage.factor(df_bot) ))))
    print(r"\end{align*}")

    from qumba.clifford import K
    ring = sage.PolynomialRing(K, "z")
    df_bot = ring(df_bot)
    print("df_bot:", sage.factor(df_bot))
    


def test_binary():

    base = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(base)

    from qumba.selfdual import load
    #items = load.get_items("24-II.magma")
    #items = load.get_items("32-II.magma")
    items = load.get_items("40-II8.magma") # 16470

    print("items:", len(items))

    z = base.gens()[0]
    F4p = z**4 + 2*z**3 + 2*z**2 - 2*z + 1
    F4m = z**4 - 2*z**3 + 2*z**2 + 2*z + 1
    Pz = z

    z = R.gens()[0]
    for idx,item in enumerate(items):
        H = numpy.array(item)
        H = H[1:, 1:]
        #print(H, H.shape)
        #code = QCode.build_css(H, H)
        #assert code.is_selfdual()
        #if code.d < 7:
        #    continue

        #f = build_selfdual(code)
        f = classical_wenum(H)

        df = diff(f, z)
        top = df.numerator()
        bot = df.denominator()

        #print(top)
        #if top % F4p == 0:
        #    print("\t", top//F4p)

        #print(code, end=" ")

        print("%d"%idx, end="", flush=True)

        ftop = sage.factor(top)
        for factor,m in ftop:
            if factor == Pz:
                print("(Pz^%d)"%m, end="")
            if factor == F4p:
                print("(F4p^%d)"%m, end="")
            if factor == F4m:
                print("(F4m^%d)"%m, end="")

            #print(code)
            #print(f)
            #print(r"f'(z) = \frac{%s}{(%s)^2}"%(latex(sage.factor(top)), latex(f.denominator())))

        #find(f)
        #break

        print(" ", end='', flush=True)
        #break


def test_de():

    base = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(base)

    from qumba.selfdual import load
    #items = load.get_items("24-II.magma")
    #items = load.get_items("32-II.magma")
    #items = load.get_items("40-II8.magma") # 16470
    items = load.get_items("40-II-d4-p1.magma")

    print("items:", len(items))

    z = base.gens()[0]
    F4p = z**4 + 2*z**3 + 2*z**2 - 2*z + 1
    F4m = z**4 - 2*z**3 + 2*z**2 + 2*z + 1
    Pz = z

    items = reversed(items)

    z = R.gens()[0]
    for idx,item in enumerate(items):
        H = numpy.array(item)
        H = H[1:, 1:]
        f = classical_wenum(H)

        #print(f.numerator())

        df = diff(f, z)
        top = df.numerator()
        bot = df.denominator()

        ftop = sage.factor(top)

        print(("%d"%idx).ljust(6), end="")

        for factor,m in ftop:
            if "z" not in str(factor):
                continue
            #if factor.degree() <= 4:
            #    continue
            #print("\t( %s )^%d" % (factor, m))
            print("(%d^%d)"%(factor.degree(),m), end="")
        print()



def test_CH():

    x = sage.polygen(sage.ZZ, "x")
    base = sage.NumberField(x**2 - 2, "r2")
    r2 = base.gens()[0]
    ir2 = (1/r2)
    print(base)

    I = Matrix.identity(base, 2)
    H = Matrix(base, [[1,1],[1,-1]])
    H = ir2*H
    assert H*H == I

    CH = I<<H
    print(CH)

    plus = ir2*Matrix(base, [[1,1]])
    #print(plus)
    assert (plus*plus.d)[0,0] == 1
    op = (plus@I)*CH

    #post = Matrix(base, [[0,1]])
    #op = (post@I)*CH

    print(op)

    R = sage.PolynomialRing(base, "z")
    z = R.gens()[0]

    op = Matrix(R, op)
    print(op)

    u = Matrix(R, [z**2, z, z, 1]).t
    v = op*u
    #print(v, v.shape)
    top = v[0,0]
    bot = v[1,0]

    K = sage.FractionField(R)
    top, bot = K(top), K(bot)
    f = top / bot
    print(f)

    df = diff(f, z)

    r = sage.factor(top - z*bot)

    df = sage.diff(f, z)
    df_top = df.numerator()
    df_bot = df.denominator()

    print(r"\begin{align*}")
    print("f(z) &= %s \\\\" % latex(f))
    print("r(z) &= %s \\\\" % latex(r))

    if df_bot == bot**2:
        print(r"f'(z) &= \frac{%s}{%s^2}"%(
            latex(sage.factor(df_top)), (latex( sage.factor(bot) ))))
    else:
        print(r"f'(z) &= \frac{%s}{%s} %% df_bot != bot**2"%(
            latex(sage.factor(df_top)), (latex( sage.factor(df_bot) ))))
    print(r"\end{align*}")


def test_ctrl():

    base = sage.PolynomialRing(sage.ZZ, list("abcd"))
    print(base)
    a,b,c,d = base.gens()

    I = Matrix.identity(base, 2)
    A = Matrix(base, [[a,b],[c,d]])

    for _ in range(2):
        CA = I<<A
        print(CA)
    
        plus = Matrix(base, [[1,1]])
        op = (plus@I)*CA
    
        #post = Matrix(base, [[0,1]])
        #op = (post@I)*CA
    
        print(op)
    
        R = sage.PolynomialRing(base, "z")
        z = R.gens()[0]
    
        op = Matrix(R, op)
        print(op)
    
        u = Matrix(R, [z**2, z, z, 1]).t
        v = op*u
        #print(v, v.shape)
        top = v[0,0]
        bot = v[1,0]
    
        K = sage.FractionField(R)
        top, bot = K(top), K(bot)
        #a, b, c, d = K(a),K(b),K(c),K(d)

        f = top / bot
        print(f)
    
        df = diff(f, z)
    
        r = sage.factor(top - z*bot)
    
        df = sage.diff(f, z)
        df_top = df.numerator()
        df_bot = df.denominator()
    
        print(r"\begin{align*}")
        print("f(z) &= %s \\\\" % latex(f))
        print("r(z) &= %s \\\\" % latex(r))
    
        if df_bot == bot**2:
            print(r"f'(z) &= \frac{%s}{%s^2}"%(
                latex(sage.factor(df_top)), (latex( sage.factor(bot) ))))
        else:
            print(r"f'(z) &= \frac{%s}{%s} %% df_bot != bot**2"%(
                latex(sage.factor(df_top)), (latex( sage.factor(df_bot) ))))
        print(r"\end{align*}")
    
        char = (a-z)*(d-z) - b*c
        print("char:", latex(char))

        #top = top.subs({a:1,b:1,c:1,d:-1})
        #bot = bot.subs({a:1,b:1,c:1,d:-1})
        print()

        A = Matrix(base, [[1,1],[1,-1]])



def get_coords(v): # todo: could unroll all of this into py code
    m, n = v.shape
    coords = {}
    for idxs in choose(list(range(m)), n):
        w = v[idxs, :]
        a = w.determinant()
        if a != 0:
            #print("[%s:%s]"%(idxs,a), end='', flush=True)
            coords[idxs] = a
    return coords


def test_kl():

    code = construct.get_422()
    print(code)
    print(code.longstr(False))

    P = code.get_projector()
    space = Clifford(code.n)
    I = space.I

    assert P*P == P

    evecs = P.eigenvectors()
    for val,vec,d in evecs:
        if val == 1:
            assert d == 2**code.k
            break
    else:
        assert 0

    v = vec
    print("v =")
    print(v)

    print(get_coords(v))

    op = space.X(0)
    u = op*v
    
    print(get_coords(u))


def test_fgl():
    "test some formal group law identities"
    from sage.all_cmdline import (
        coth, tanh, sech, sinh, cosh, csch, 
            acoth, atanh, asech, asinh, acosh, acsch)

    epsilon = 1e-6
    eq = lambda a,b : abs(a-b)<epsilon
    F = lambda x,y : (x+y)/(1+x*y)
    G = lambda x,y : (1+x*y)/(x+y)
    H = lambda x,y : (x+y)/(1-x*y)
    for trial in range(100):
        x = 10*random() - 5
        y = 10*random() - 5
        assert eq(tanh(x+y), F(tanh(x), tanh(y)))
        assert eq(coth(x+y), G(coth(x), coth(y)))
        assert eq(tanh(x+y), -1j*H(1j*tanh(x), 1j*tanh(y)))

    base = sage.PolynomialRing(sage.ZZ, list("xy"))
    R = sage.FractionField(base)

    x,y = R.gens()
    G = (1+x*y)/(x+y)
    F = (x+y)/(1+x*y)
    dG = diff(G, x).subs({y:0})
    print(dG)
    dF = diff(F, x).subs({y:0})
    print(dF)


def test_513():
    R = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(R)
    z = R.gens()[0]

    F = z**4 + 2*z**3 + 2*z**2 - 2*z + 1
    print(F)

    p = (z**5 - 5*z) / (5*z**4 - 1)
    Q = F.subs({z:p})
    print(Q)
    top = Q.numerator()
    top = sage.factor(top)
    print(top)

    #Q = sage.factor(Q)
    #print(Q)



def test_distill():
    code = get_code(verbose=False)

    print(code.longstr(False))
    print("is_selfdual", code.is_selfdual())

    base = sage.PolynomialRing(sage.ZZ, "z")
    R = sage.FractionField(base)
    z = R.gens()[0]

    if code.is_css() and code.is_selfdual() and not argv.se: # and code.is_doublyeven():
        f = build_selfdual(code)
    else:
        f = PauliDistill(code).build()
    f = R(f)

    if argv.Z:
        f = -f

    top = f.numerator()
    bot = f.denominator()

    r = sage.factor(top - z*bot)

    df = sage.diff(f, z)
    df_top = df.numerator()
    df_bot = df.denominator()

    def factor(p):
        items = sage.factor(p)
        items = [p for (p,m) in items if "z" in str(p)]
        return items

    items = factor(df_top)
    #jtems = factor( 

    print(f)

    for p in items:
        print()
        print(p, "= 0")
        p = p.subs({z:f}) # wtf is this ???
        p = p.numerator()
        #print(p)
        print(sage.factor(p))

    return

    # This says something true, but trivial:
    # a root of p(z) is a root of q(p(z))-q(0). duh.
    for p in items:
        print(p, "= 0")
        print("-->")
        print("\tf(%s) = f(0)" % (p,))
        f1 = f.subs({z:p}) - f.subs({z:0})
        print("\t%s = 0" % (f1))
        top = f1.numerator()
        print("\t%s = 0" % top)
        print("\t%s = 0" % sage.factor(top))


def test_yspider():

    vs = []
    for val,vec,dim in Y.eigenvectors():
        vs.append(vec)
    pos, neg = vs

    mul = pos @ pos.d @ pos.d + neg @ neg.d @ neg.d
    #print(mul)


    base = sage.PolynomialRing(sage.QQ, list("uv"))
    R = sage.FractionField(base)
    u,v = R.gens()

    f = (u+v)/(1+u*v)
    print(f)
    #print(" ".join(dir(f)))

    def partial(i, j):
        df = f
        for _ in range(i):
            df = diff(df, u)
        for _ in range(j):
            df = diff(df, v)
        n = sage.factorial(i) * sage.factorial(j)
        #return n
        return df.subs({u:0, v:0}) #// n

    N = 5
    table = Matrix(sage.QQ, [[partial(i,j) for i in range(N)] for j in range(N)])
    print(table)


def dynamic():

    code = get_code((15,1,3))
    #code = get_code((7,1,3))

    print(code)

    f = PauliDistill(code).build()

    s = pystr(f)
    print(s)

    f = eval("lambda z:%s"%s)


    N = 10000
    R = 10
    items = []
    for _ in range(N):
        x = numpy.random.normal(0, R)
        y = numpy.random.normal(0, R)
        z = x + 1j*y
        items.append(z)

    from huygens.namespace import Canvas, path, black, red
    cvs = Canvas()
    p = path.rect(-R,-R,2*R,2*R)
    cvs.stroke(p)
    cvs.clip(p)
    #for z in items[::10]:
    #    cvs.fill(path.circle(z.real, z.imag, 0.2), [red.alpha(0.2)])

    for i in range(20):
        items = [f(z) for z in items]
        items = [z for z in items if abs(z) < 1e6]

    items = [z for z in items if abs(z) < 100]
    print("items:", len(items))
    print(items[:10])

    for z in items:
        cvs.fill(path.circle(z.real, z.imag, 0.1), [black.alpha(0.2)])

    for z in [1,-1,1j,-1j, 0]:
        z = complex(z)
        cvs.stroke(path.circle(z.real, z.imag, 0.2))

    cvs.writePDFfile("distill_code")





if __name__ == "__main__":

    from random import seed
    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    fn = eval(name)

    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            fn()
        profiler.print()

    else:
        fn()


    t = time() - start_time
    print("\nOK! finished in %.3f seconds\n"%t)



