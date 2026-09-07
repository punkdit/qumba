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


def main():
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
    R = sage.PolynomialRing(K, vs)
    FracR = sage.FractionField(R)
    vs = R.gens()
    P = sage.ProjectiveSpace(len(vs)-1, K, list(vs))

    affine = vs[:6]
    A = sage.AffineSpace(K, len(affine), affine)

    (a0, b0, a1, b1, a2, b2, a3, b3) = vs

    #vket = Matrix(R, [a0, b0, a1, b1, a2, b2, a3, b3]).reshape(2*D, 1)
    #vbra = Matrix(R, [a0, -b0, a1, -b1, a2, -b2, a3, -b3]).reshape(1, 2*D)
    #print(vbra * vket)

    # Affine chart
    a3, b3 = 1, 0
    vket = Matrix(R, [a0, b0, a1, b1, a2, b2, a3, b3]).reshape(2*D, 1)
    vbra = Matrix(R, [a0, -b0, a1, -b1, a2, -b2, a3, -b3]).reshape(1, 2*D)

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

    items = []
    for op in [XX@I, ZZ@I]: # complexify the op to op@I
        rhs = op*vket
        #print(rhs)
        #rhs = normalize(rhs) # nope...
        #print(rhs)
        top = inner(vbra, rhs)
        assert top == inner(vbra*op, vket)
        bot = inner(vbra, vket)

        #print(vbra*op)
        #print(op*vket)
        #continue

        f = top / bot
        #print(f)
        for v in affine:
            f_v = sage.diff(f, v)
            #print(f_v)
            top_v = f_v.numerator()
            #print(top_v)
            items.append(top_v)
            soln = A.subscheme([top_v])
            print(soln.dimension(), end=' ')
        print()
        
    #soln = P.subscheme(items)
    soln = A.subscheme(items)
    print(soln)
    print(soln.dimension()) # how to get this down to zero ?




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




