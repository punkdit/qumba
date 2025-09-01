#!/usr/bin/env python

from functools import reduce
from operator import matmul, add

from qumba.argv import argv

from qumba.qcode import strop, QCode
from qumba import construct 

from qumba.matrix_sage import Matrix
from qumba import clifford
from qumba.clifford import Clifford, w4

from sage import all_cmdline as sage

right_arrow = chr(0x2192)


def get_code():
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
    return code

def test():

    base = clifford.K
    one = base.one()
    half = one/2

    K = sage.PolynomialRing(base, list("xyzw"))
    Kx,Ky,Kz,Kw = K.gens()

    c = Clifford(1)
    I = c.I
    X = c.X()
    Y = c.Y()
    Z = c.Z()

    Kw = 1
    rho = half*(Kw*I + Kx*X + Ky*Y + Kz*Z)
    assert rho.trace() == Kw

    code = QCode.fromstr("ZZ")
    #code = construct.get_422() # no...
    #code = QCode.fromstr("XXXX ZZZZ ZZII")
    #code = QCode.fromstr("XXXXXX ZZZZZZ ZZZZII IIXXXX ZZIIII")
    #code = QCode.fromstr("YYZI IXXZ ZIYY") # [[4,1,2]]
    #code = construct.get_513()
    #code = construct.get_512()
    #code = construct.get_713()
    code = construct.get_913()
    #code = get_code() # too big..

    print(code)
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

    Pd = P.d

    #print("rho^n ... ", end='', flush=True)
    rho = reduce(matmul, [rho]*n)
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

    pystr = lambda u : str(u).replace("^", "**")
    pyfunc = lambda u : eval("lambda x,y,z: %s"%pystr(u))

    #x,y,z,w = Kx,Ky,Kz,1 # identity

    print("x", right_arrow, x)
    print("y", right_arrow, y)
    print("z", right_arrow, z)
    print("w", right_arrow, w) # div

    def stereo(x,y,z):
        return x/(1-z), y/(1-z)

    R = sage.PolynomialRing(base, list("XY"))
    X, Y = R.gens()
    ix = 2*X/(1+X**2+Y**2)
    iy = 2*Y/(1+X**2+Y**2)
    iz = (X**2+Y**2-1)/(1+X**2+Y**2)
    print(ix, iy, iz)

    u, v = stereo(x/w, y/w, z/w)
    u = u.subs({Kx:ix, Ky:iy, Kz:iz})
    v = v.subs({Kx:ix, Ky:iy, Kz:iz})
    print("u =", u)
    print("v =", v)

    print("Cauchy-Riemann:")
    print("\t", sage.derivative(u,X) == sage.derivative(v,Y))
    print("\t", sage.derivative(u,Y) == -sage.derivative(v,X))

    return

    fx = pyfunc(x/w)
    fy = pyfunc(y/w)
    fz = pyfunc(z/w)


    #F = sage.FractionField(K)
    #jac = Matrix(F, [[
    jac = [[pyfunc(sage.derivative(v,u))
        for v in [x/w,y/w,z/w]] 
        for u in [Kx,Ky,Kz]]

    from qumba.distill import normalize, norm, rnd
    import numpy

    EPSILON = 1e-6
    ir2 = 2**(-1/2)
    def ortho(base, vec):
        # make vec ortho to base, which is normalize'd
        base = numpy.array(base)
        assert abs(norm(*base)-1) < EPSILON
        vec = vec - base*numpy.dot(vec, base)
        return vec

    #print(ortho(numpy.array([ir2,ir2,0]), numpy.array([0,1,0])))
    #return

    dx = numpy.array([1,0,0])
    dy = numpy.array([0,1,0])
    dz = numpy.array([0,0,1])

#    def diff(u):
#        r = norm(*u)
#        total = 1000*abs(1-r) # penalize away from 1 
#        u = (1/r)*numpy.array(u) # normalize
#        v = numpy.array([fx(*u), fy(*u), fz(*u)])
#        r = norm(*v)
#        v = (1/r)*v # normalize
#        dv = numpy.array([[f(*u) for f in row] for row in jac])
#        #print(dv)
#        for d in [dx,dy,dz]:
#            d = ortho(u, d)
#            d_out = numpy.dot(dv, d)
#            d_out = ortho(v, d_out)
#            total += (d_out**2).sum()
#        return total

    def diff(u):
        r = norm(*u)
        #total = 1000*abs(1-r) # penalize away from 1 
        u = (1/r)*numpy.array(u) # normalize
        dv = numpy.array([[f(*u) for f in row] for row in jac])
        return (dv**2).sum()

    #diff(*rnd())
    u = normalize(1,1,1)
    u = numpy.array(u)
    #u = (ir2,ir2,0.1)
    u[0] += 0.1
    r = diff(u)

    from scipy.optimize import minimize

    method = "Nelder-Mead"
    for trial in range(20):
        x0 = rnd()
        #x0 = normalize(*[-0.28,  -0.9, -0.29])
        #print(diff(x0))
        res = minimize(diff, x0, 
            method=method, tol=1e-6, bounds=[(-1.2,1.2)]*3,
            options={"maxiter":10000, "maxfev":10000},
        )
        #if not res.success:
        #    print(res)
        #    continue
        x = res.x
        x = normalize(*x)
        v = numpy.array([fx(*x), fy(*x), fz(*x)])
        print("fun=%.4f, x=(%.4f, %.4f, %.4f), |x|=%.4f"%(
            res.fun, x[0], x[1], x[2], norm(*x)), "*" if not res.success else "")
        print("  -->", v)

    return



    a,b,c = 0,ir2,ir2
    u,v,w = (fx(a,b,c), fy(a,b,c), fz(a,b,c))




    #print(sage.macaulay2("3/77"))
    macaulay2 = sage.macaulay2

    #sub = Kw**n + Kx**n + Ky**n + Kz**n
    #print(sub)

#    R = macaulay2('QQ[w,x,y,z]', 'R')
#    #s = "map(R,R,matrix{{%s,%s,%s,%s}})"%(N*div , N*x , N*y , N*z )
#    s = """matrix{
#{-10*x*y*z^2+10*x*y*w^2, 10*x*y^2*z-10*x*z*w^2,-5*x^4-5*y^2*z^2+5*y^2*w^2+5*z^2*w^2, 10*x*y^2*w+10*x*z^2*w},
#{-5*y^4-5*x^2*z^2+5*x^2*w^2+5*z^2*w^2, 10*x^2*y*z-10*y*z*w^2, -10*x*y*z^2+10*x*y*w^2, 10*x^2*y*w+10*y*z^2*w},
#{-10*x^2*y*z+10*y*z*w^2, 5*x^2*y^2+5*z^4-5*x^2*w^2-5*y^2*w^2, -10*x*y^2*z+10*x*z*w^2, 10*x^2*z*w+10*y^2*z*w},
#{10*x^2*y*w+10*y*z^2*w, -10*x^2*z*w-10*y^2*z*w, 10*x*y^2*w+10*x*z^2*w, 5*x^2*y^2+5*x^2*z^2+5*y^2*z^2+5*w^4}
#    }
#    """.replace("\n", " ")
#    print(s)
#    f = macaulay2(s)
#    print(f)
#    print("kernel:")
#    print(f.kernel())


if __name__ == "__main__":

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
    print("OK! finished in %.3f seconds\n"%t)



