#!/usr/bin/env python

"""

"""

import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add
from random import random

import numpy
from scipy.optimize import root

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.qcode import strop, QCode
from qumba import construct 
from qumba.matrix_sage import Matrix
from qumba.clifford import Clifford, w4, ir2
from qumba.action import mulclose
from qumba.dense import bitlog
from qumba import pauli

EPSILON = 1e-6

def eq(a, b):
    return abs(a-b) < 1e-6

diff = sage.derivative
latex = sage.latex

py_w8 = (2**(-1/2))*(1+1j)
assert eq(py_w8**2, 1j)

def simplify(f):
    #return f

    top = f.numerator()
    bot = f.denominator()
    #print("simplify:", [ti for ti in top], [bi for bi in bot])

    factor = None
    for (c,e) in list(top)+list(bot):
        #print("int(c)", c)
        try:
            c = int(c)
        except TypeError:
            print("simplify: TypeError")
            return f
        if factor is None:
            factor = c
        else:
            factor = gcd(factor, c)

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


class Mobius:
    R = sage.I.parent()
    Rz = sage.PolynomialRing(R, ["z"])
    K = sage.FractionField(Rz)
    z = K.gens()[0]

    def __init__(self, f):
        f = Mobius.K(f)
        self.f = f

    @classmethod
    def build(cls, a=1, b=0, c=0, d=1):
        z = Mobius.z
        assert a*d - b*c != 0
        f = (a*z + b) / (c*z + d)
        return cls(f)

    @classmethod
    def promote(cls, item):
        if isinstance(item, Mobius):
            return item
        return Mobius(item)

    def __str__(self):
        return "Mobius(%s)"%(self.f)
    __repr__ = __str__

    def __mul__(self, other):
        other = Mobius.promote(other)
        z = Mobius.z
        f = self.f.subs({z:other.f})
        return Mobius(f)

    def __eq__(self, other):
        other = Mobius.promote(other)
        return self.f == other.f

    def __hash__(self):
        return hash(self.f)


def build_mobius():

    I = Mobius.build()
    assert I*I == I

    X = Mobius.build(0,1,1,0)
    assert X != I
    assert X*X == I

    Z = Mobius.build(-1,0,0,1)
    assert Z*Z == I
    assert Z*X == X*Z
    assert Z*X != I

    Pauli = mulclose([X,Z])
    assert len(Pauli) == 4

    H = Mobius.build(1,1,1,-1)
    assert H*H == I

    S = Mobius.build(-sage.I, 0, 0, 1) # i hope this is S not S dagger 
    assert S*S == Z

    Clifford = mulclose([S,H])
    assert len(Clifford) == 24

    Mobius.Clifford = Clifford
    Mobius.Pauli = Pauli

    z = Mobius.z
    f = Mobius((z**2 + 1)/(2*z)) # XXX rename Mobius as Meromorphic
    f = Mobius(z**2)

    for g in Clifford:
        gf = g*f


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

        assert f.subs({zb:1234}) == f
        assert "zb" not in str(f)

        return f

    def fast_find(self, top=True, verbose=False):
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

        T = sage.PolynomialRing(base, "z zb".split())
        T = sage.FractionField(T)
        z, zb = T.gens()

        real = half*(z+zb)
        imag = half*(-w4)*(z-zb)
        uz = u.subs({X:real,Y:imag}) 
        vz = v.subs({X:real,Y:imag}) 
        f = uz + w4*vz

        f = simplify(f)

        if verbose:
            pprint(f)
        assert f.subs({zb:1234}) == f
        assert "zb" not in str(f)

        pyfunc = lambda u : eval("lambda z: %s"%pystr(u))
        py_f = pyfunc(f)
        self.f = py_f

        print("f =", mkstr(f))

        top = (f.numerator())
        bot = (f.denominator())
        py_top = pyfunc(top)
        py_bot = pyfunc(bot)

        #print(top, "==", z*bot)
        #print("\t", top == z*bot)
        df = diff(f, z)
        py_df = pyfunc(df)

        if top:
          for val in [0,1,-1,1j,-1j]:
            try:
              if eq(val, py_f(val)):
                print("FIXED:", val, istereo(val.real, val.imag), py_df(val))
            except ZeroDivisionError:
                pass

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
            x, y, z = (x, sign*y, sign*z)
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
    
        print("x", right_arrow, x)
        print("y", right_arrow, y)
        print("z", right_arrow, z)
        print("w", right_arrow, w) # div
        print()

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
        Distill.__init__(self, code.n)
        self.code = code

    def get_xyzw(self, projective=False):
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

    def build(self, projective=False):
        n = self.n

        #x,y,z,w = self.get_xyzw(projective)
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

        hens = []
        for i in range(n):
            hens.append("X%d"%i) # real
            hens.append("Y%d"%i) # imag
        R = sage.PolynomialRing(base, hens)
        hens = R.gens()
        subs = {}
        for i in range(n):
            X, Y = hens[2*i:2*i+2]
            ix = 2*X/(1+X**2+Y**2)
            iy = 2*Y/(1+X**2+Y**2)
            iz = (X**2+Y**2-1)/(1+X**2+Y**2)
            subs[gens[3*i]] = ix
            subs[gens[3*i+1]] = iy
            subs[gens[3*i+2]] = iz

        print("u.subs")
        u = u.subs(subs)
        print("v.subs")
        v = v.subs(subs)

        kens = []
        for i in range(n):
            kens.append("z%d"%i) # complex
            kens.append("zb%d"%i) # complex conjugate
        T = sage.PolynomialRing(base, kens)
        T = sage.FractionField(T)
        kens = T.gens()

        subs = {}
        for i in range(n):
            z, zb = kens[2*i:2*i+2]
            real = half*(z+zb)
            imag = half*(-w4)*(z-zb)
            subs[hens[2*i]] = real
            subs[hens[2*i+1]] = imag
        print("uz = u.subs")
        uz = u.subs(subs)
        print("vz = v.subs")
        vz = v.subs(subs)
        print("f = uz + w4*vz")
        f = uz + w4*vz
        assert f.subs({zb:1234}) == f
        assert "zb" not in str(f)

        f = simplify(f)

        self.f = f
        self.gens = [kens[2*i] for i in range(n)]

        return f


    def fast_build(self, projective=False):
        n = self.n

        #x,y,z,w = self.get_xyzw(projective)
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

        print("uz = u.subs")
        uz = u.subs(subs)
        print("vz = v.subs")
        vz = v.subs(subs)
        print("T(uz);T(vz)")
        uz = T(uz)
        vz = T(vz)
        print("f = uz + w4*vz")
        f = uz + w4*vz
        print("assert")
        #assert f.subs({zb:1234}) == f
        assert "zb" not in str(f)

        print("simplify")
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


def get_code():
    code = None
    idx = argv.get("idx", 0)
    params = argv.code
    if params == (3,1,1):
        code = QCode.fromstr("ZZZ XXI")
    if params == (4,1,2):
        code = [
            QCode.fromstr("YYZI IXXZ ZIYY"),
            QCode.fromstr("XXXX ZZZZ YYII"),
            QCode.fromstr("XYZI IXYZ ZIXY"),
        ][idx]
    if params == (5,1,1):
        code = QCode.fromstr("""
        XZX.Z
        ZXX.Z
        ZZ.XX
        ZZZ..
        """)
    if params == (5,1,2):
        code = construct.get_512()
    if params == (5,1,3):
        #code = construct.get_513()
        code = QCode.fromstr("XZZX.  .XZZX X.XZZ ZX.XZ", None, "ZXZII YZYII")
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
        ZZIIZZYY""")
    if params == (9,1,3) and idx == 0:
        code = construct.get_913() # Shor code
    if params == (9,1,3) and idx == 1:
        code = construct.get_surface(3, 3)

    if params == (16,1,4):
        code = construct.get_surface(4, 4)

    if params == (10,1,2):
        code = QCode.fromstr("XXIXXIIXII IXXXIXIIXI IIIXXXXIIX "
        "ZZIIIIIIZI IIIZZIIIZI IZIZIIIIIZ IIZIIZIIIZ IIIZIZIZII IIIIZIZZII")

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
        code = QCode.fromstr(H)
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
        code = QCode.fromstr(H)

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

    assert code is not None

    return code


def orbit():
    code = get_code()
    distill = PauliDistill(code)
    f = distill.build()

    print(f)

    for g in Mobius.Clifford:
        gf = g*f
        print("\t", gf.f)


def full_wenum():
    code = get_code()
    result = pauli.get_full_wenum(code)




def multi():
    code = get_code()
    distill = MultiDistill(code)

    distill.fast_build()

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
    if params == (5,1,3):
        z0 = 1.3660254037844386+1.3660254037844386j
    elif params == (5,1,2):
        z0 = 1 # df == 0
    elif params == (7,1,3):
        z0 = 0.
    else:
        z0 = argv.get("z0", 0)
    print("z0 =", z0)
    subs = {g:z0 for g in gens}
    df = [dfi.subs(subs) for dfi in df]
    total = 0
    for dfi in df:
        print(eq(dfi, 0), "df(z0,...,z0) =", dfi)
        total += dfi
    print("total =", total)

    print("df(z0) =", dfz.subs({z:z0}) )



def test():

    code = None
    #if argv.code:
    #    import sys
    #    print(repr(argv.code), sys.argv)
    #    code = QCode.fromstr(argv.code)
    if argv.code:
        code = get_code()
        #distill = CodeDistill(code)
        distill = PauliDistill(code)

    elif argv.stab:
        code = QCode.fromstr(argv.stab, argv.destab, argv.logical)
        distill = PauliDistill(code)

    elif argv.CH:
        CH = clifford.I << clifford.H()
        plus = ir2*Matrix(base, [[1,1]])
        print(plus)
        assert (plus*plus.d)[0,0] == 1
        op = (plus@I)*CH
        print(op)
        # TODO fix the problem with sage rings, etc. somehow...
        assert 0, "TODO"
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

    print(code)
    print(code.longstr())

    print("is_gf4:", code.is_gf4())
    print("is_css:", code.is_css())
    print("is_selfdual:", code.is_selfdual())

    trials = argv.get("trials", 10000)
    nsols = argv.get("nsols", 10)
    verbose = argv.get("verbose", False)

    if argv.projective:
        distill.get_variety(projective=True)
        return

    for top in [True, False]:
        print("--")
        found = 0
        for (x,y,z,m,val) in distill.fast_find(top, verbose=verbose):
            val = complex(val)
            fval = distill.f(complex(val))
            print(r"& %d \times (%.8f, %.8f, %.8f) \\ %% %s --> %s"%(
                m, x, y, z, val, fval))
            found += m
        print(r"\begin{align*}")
        print(r"\end{align*}")
        print("total:", found)
        break

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


def search_fix():

    n = argv.get("n", 4)
    d = argv.get("d", 1)
    k = 1
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

    found = 0
    for code in fn(n,k,d):
        code.build()
        assert code.L is not None
        found += 1
        distill = PauliDistill(code)
        try:
            for (x,y,z,m,val) in distill.fast_find():
                val = complex(val)
                fval = distill.f(complex(val))
                if not eq(val, fval): # <<-- look for fix points
                    continue
                if getkey(val) not in skip:
                    print(code)
                    print(code.longstr())
                    print("\t", m, "x", val, "-->", fval, flush=True)
        except ArithmeticError:
            print("ArithmeticError")
    print("found:", found)


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
            fval = distill.f(complex(val))

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
    print("OK! finished in %.3f seconds\n"%t)



