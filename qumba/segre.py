#!/usr/bin/env python

"""
Measurements should correspond to critical points of Morse (Morse-Bott ?)
functions.

"""

from random import choice
from operator import mul, matmul, add
from functools import reduce, cache

import numpy

from sage import all_cmdline as sage

from qumba.argv import argv
from qumba.matrix_sage import Matrix
from qumba.util import cross


def prod(a, b, c, d):
    return (a*c - b*d, b*c + a*d)


def main_2():
    # ----------------------------------------------------------------------

    D = 2
    vs = []
    for i in range(D):
        vs.append("a%d"%i)
        vs.append("b%d"%i)

    K = sage.QQ
    #R = sage.PolynomialRing(K, vs)
    R = sage.PolynomialRing(K, vs)
    vs = R.gens()
    P = sage.ProjectiveSpace(len(vs)-1, K, list(vs))

    affine = vs[:2]
    A = sage.AffineSpace(K, len(affine), affine)

    (a0, b0, a1, b1) = vs

    vket = Matrix(R, [a0, b0, a1, b1]).reshape(2*D, 1)
    vbra = Matrix(R, [a0, -b0, a1, -b1]).reshape(1, 2*D)

    # Affine chart
    a1, b1 = 1, 0
    vket = Matrix(R, [a0, b0, a1, b1]).reshape(2*D, 1)
    vbra = Matrix(R, [a0, -b0, a1, -b1]).reshape(1, 2*D)
    print(vbra * vket)

    I = Matrix.get_identity(R, 2)
    X = Matrix(R, [[0, 1], [1, 0]])
    Z = Matrix(R, [[1, 0], [0, -1]])
    Y = X*Z
    J = Y    

    assert J*J == -I

    def inner(lhs, rhs):
        a, b, c, d = [lhs[0,i] for i in range(4)]
        e, f, g, h = [rhs[i,0] for i in range(4)]
        u, v = prod(a, b, e, f)
        w, x = prod(c, d, g, h)
        a, b = (u+w, v+x)
        assert b==0 # should be real value only
        return a

    vals = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]

    # complexify the op to op@I
    for op in [X@I, Z@I, -Y@J]:
        print(op)
        print(vbra)
        rhs = op*vket
        print(rhs)
        #scale = rhs[3, 0]
        #print("scale:", scale)
        #if scale != 0:
        #    rhs = (1/scale)*rhs
        #    print(rhs)
        #top = (vbra * rhs)[0,0]
        top = inner(vbra, rhs)
        #print(vbra)
        #print(op*vket)
        print("top =", top)
        #bot = (vbra * vket)[0,0]
        bot = inner(vbra, vket)
    
        f = top / bot
        print("f =", f)
        items = []
        for v in affine:
            f_v = sage.diff(f, v)
            top_v = f_v.numerator()
            bot_v = f_v.denominator()
            print("f_%s = %s" % (v, f_v))
            for vs in vals:
                print("\tf_v(%s, %s) =" % vs, end=' ')
                print("(%s)/(%s)"%(
                    top_v(a0=vs[0], b0=vs[1]),
                    bot_v(a0=vs[0], b0=vs[1]),))
            #print(top_v)
            items.append(top_v)
            soln = A.subscheme([top_v])
            #print(soln.dimension(), end=' ')
        #print()
        
        #soln = P.subscheme(items)
        soln = A.subscheme(items)
        print("dimension:", soln.dimension()) # should be zero.. 
        print(soln)
    
        #print(' '.join(dir(soln)))
        print()

    return


def main_bell():
    # ----------------------------------------------------------------------

    def inner(lhs, rhs):
        a, b, c, d, e, f, g, h = [lhs[0,i] for i in range(8)]
        i, j, k, l, m, n, o, p = [rhs[i,0] for i in range(8)]
        #print("inner:", (a, b, c, d, e, f), (g, h, i, j, k, l))
        u, v = prod(a, b, i, j)
        w, x = prod(c, d, k, l)
        y, z = prod(e, f, m, n)
        s, t = prod(g, h, o, p)
        #print((u,w,y), (v,x,z))
        a, b = (u+w+y+s, v+x+z+t)
        assert b==0, (a,b) # should be real value only
        return a

    D = 4
    vs = []
    for i in range(D):
        vs.append("a%d"%i)
        vs.append("b%d"%i)

    K = sage.QQ
    #R = sage.PolynomialRing(K, vs)
    R = sage.PolynomialRing(K, vs[:6])
    FracR = sage.FractionField(R)
    affine = R.gens()
    #(a0, b0, a1, b1, a2, b2, a3, b3) = vs
    (a0, b0, a1, b1, a2, b2, ) = affine

    #P = sage.ProjectiveSpace(len(vs)-1, K, list(vs))

    A = sage.AffineSpace(K, len(affine), affine)

    def dag(ket):
        a, b, c, d, e, f, g, h = [ket[i,0] for i in range(8)]
        bra = Matrix(R, [a, -b, c, -d, e, -f, g, -h]).reshape(1, 2*D)
        return bra

    #vket = Matrix(R, [a0, b0, a1, b1, a2, b2, a3, b3]).reshape(2*D, 1)
    #vbra = Matrix(R, [a0, -b0, a1, -b1, a2, -b2, a3, -b3]).reshape(1, 2*D)
    #print(vbra * vket)

    # Affine chart
    a3, b3 = 1, 0
    vket = Matrix(R, [a0, b0, a1, b1, a2, b2, a3, b3]).reshape(2*D, 1)
    vbra = Matrix(R, [a0, -b0, a1, -b1, a2, -b2, a3, -b3]).reshape(1, 2*D)
    assert vbra == dag(vket)
    assert inner(vbra, vket) == a0**2 + b0**2 + a1**2 + b1**2 + a2**2 + b2**2 + 1

    I = Matrix.get_identity(R, 2)
    X = Matrix(R, [[0, 1], [1, 0]])
    Z = Matrix(R, [[1, 0], [0, -1]])
    #Y = X*Z
    II = I@I
    XX = X@X
    ZZ = Z@Z
    #YY = Y@Y

    #H = II + 2*XX + 3*ZZ + 4*YY
    #print(vbra)
    #print(vket)

    def divide(a, b, c, d):
        assert c!=0 or d!=0
        if (c,d) == (1,0):
            return (a, b)
        bot = c**2 + d**2
        u = (a*c+b*d) / bot
        v = (b*c - a*d) / bot
        return u, v

    def normalize(vec):
        a, b, c, d, e, f, g, h = [vec[i,0] for i in range(8)]
        a, b = divide(a, b, g, h)
        c, d = divide(c, d, g, h)
        e, f = divide(e, f, g, h)
        #g, h = divide(g, h, g, h)
        #assert g==1 and h==0
        return Matrix(FracR, [[a, b, c, d, e, f, 1, 0]]).t

    def getvalue(poly, vec):
        value = poly.subs(
            a0=vec[0], b0=vec[1], 
            a1=vec[2], b1=vec[3], 
            a2=vec[4], b2=vec[5])
        return value

    def getpoints(items):
        remain = set(cross([(-1,0,1)]*6))
        for poly in items:
            for vec in list(remain):
                if getvalue(poly, vec):
                    remain.remove(vec)
        return remain

    def expect(op, ket):
        assert ket.shape == (2*D, 1), ket.shape
        bra = dag(ket)
        top = inner(bra, op * ket)
        bot = inner(bra, ket)
        return top / bot

    XX, ZZ = XX@I, ZZ@I # complexify the op to op@I
    assert XX*ZZ == ZZ*XX

    ii = Matrix(R, [[1,0]]).t
    zero = Matrix(R, [[1,0]]).t
    one = Matrix(R, [[0,1]]).t
    plus = Matrix(R, [[1,1]]).t
    minus = Matrix(R, [[1,-1]]).t

    for u in [zero, one, plus, minus]:
      for v in [zero, one, plus, minus]:
        print((u@v@ii).t)

    for op in [XX, ZZ]:
     for u in [zero, one, plus, minus]:
      for v in [zero, one, plus, minus]:
        ket = u@v@ii
        assert expect(op, ket) == expect(op, -ket)
        assert expect(op, ket) == expect(op, 2*ket)
        print(expect(op, ket), end=' ')
      print()
     print()

    points = []
    for vec in [
        [1,0,0,1],
        [1,0,0,-1],
        [0,1,1,0],
        [0,1,-1,0]]:
        ket = Matrix(R, vec).t @ ii
        points.append(ket)
        for op in [XX,ZZ]:
            print(vec, expect(op, ket))

    items = []
    for op in [ZZ]:
        print(op)
        f = expect(op, vket)

        for v in affine:
            f_v = sage.diff(f, v)
            #print(f_v)
            top_v = f_v.numerator()
            bot_v = f_v.denominator()
            #print(top_v)
            items.append(top_v)
            print("\t", top_v)
            soln = A.subscheme([top_v])
            #print(soln.dimension(), end=' ')
            #print("%s/%s"%(
            #    getvalue(top_v, (-1, 0, 0, 0, 0, 0)),
            #    getvalue(bot_v, (-1, 0, 0, 0, 0, 0))), end=' ')
        #print()

        break

    return

#    for p in items:
#        #p = p.subs(a1=0,b1=0,a2=0,b2=0)
#        p = p.subs(b0=a0)
#        if p != 0:
#            print("\t", sage.factor(p))

    I = R.ideal(items)
    print(I.dimension())
    print(I)

    from sage.libs.singular.function import singular_function
    from sage.libs.singular.function import lib as singular_lib
    
    singular_lib('realrad.lib')
    realrad = singular_function('realrad')

    print("realrad:")
    RI = realrad(I)
    print("ideal:")
    J = R.ideal(RI)
    print("dimension:")
    print(J.dimension())

    return

    #return

    found = getpoints(items)
    print(found)
        
    #soln = P.subscheme(items)
    soln = A.subscheme(items)
    print(soln)
    print(soln.dimension()) # how to get this down to zero ?

    #print(' '.join(dir(soln)))

    #S = soln.coordinate_ring()
    #print(S)

    #vec = [1, 0, 0, 0, 


def main_example():
    K = sage.QQ
    
    H = Matrix(K, [[1, 1], [1,-1]])
    
    P = sage.ProjectiveSpace(K, 1)
    x, y = P.gens()
    R = sage.PolynomialRing(K, [x,y])
    
    v = Matrix(R, [[x,y]]).t
    Hv = H*v
    
    # 2x2 minor expressing rank([v Hv]) <= 1
    f = v[0]*Hv[1] - v[1]*Hv[0]
    
    E = P.subscheme([f])
    
    print("Hv =", Hv)
    print("Homogeneous equations:", E.defining_polynomials())
    print("Dimension:", E.dimension())
    print("Degree:", E.degree())
    

def main():

    K = sage.QQ

    n = 4
    P = sage.ProjectiveSpace(K, n-1)
    xs = P.gens()
    R = sage.PolynomialRing(K, xs)

    I = Matrix.get_identity(R, 2)
    X = Matrix(R, [[0, 1], [1, 0]])
    Z = Matrix(R, [[1, 0], [0, -1]])

    #print(v)

    def get_eqs(H, v):
        Hv = H*v
        eqns = [ v[i,0]*Hv[j,0] - v[j,0]*Hv[i,0]
            for i in range(n) for j in range(i+1, n) ]
        return eqns

    v = Matrix(R, [xs]).t
    E_XX = P.subscheme(get_eqs(X@X, v))
    assert E_XX.dimension() == 1

    E_ZZ = P.subscheme(get_eqs(Z@Z, v))
    assert E_ZZ.dimension() == 1

    E = E_XX.intersection(E_ZZ)
    assert E.dimension() == 0

    # simplify 
    E = E.reduce()
    #print(E)
    #print(E.irreducible_components())

    del E, n, v, xs
    # ------------------------------------------------------

    # are we teleporting yet?

    n = 2**3
    P = sage.ProjectiveSpace(K, n-1)
    xs = P.gens()
    (x0, x1, x2, x3, x4, x5, x6, x7) = xs

    R = sage.PolynomialRing(K, xs)

    I = Matrix.get_identity(R, 2)
    X = Matrix(R, [[0, 1], [1, 0]])
    Z = Matrix(R, [[1, 0], [0, -1]])

    #v = Matrix(R, [xs]).t
    #print(v.t)
    #cup = Matrix(R, [

    u = Matrix(R, [[x0, 0, 0, x0]]).t
    v = Matrix(R, [[x1, x2]]).t
    uv = u@v
    print(uv.t)

    E_0 = P.subscheme([uv[i] == xs[i] for i in range(n)])

    eqs = get_eqs(I@X@X, uv)
    #print(eqs)
    E_XX = P.subscheme(eqs)
    #print(P.dimension())
    #print(E_XX.dimension())

    eqs = get_eqs(I@Z@Z, uv)
    #print(eqs)
    E_ZZ = P.subscheme(eqs)
    #print(E_ZZ.dimension())

    E = E_XX.intersection(E_ZZ)
    E = E.intersection(E_0)
    #print(E.dimension())

    # simplify 
    E = E.reduce()
    print(E)
    print(E.dimension())
    for Ei in E.irreducible_components():
        print(Ei)


if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "main"
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




