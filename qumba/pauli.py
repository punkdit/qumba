#!/usr/bin/env python

"""
Phase accurate, scalable, implementation of Pauli operators

"""


from random import shuffle, choice
from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul
from math import prod

import numpy

from qumba.lin import (shortstr, dot2, identity2, eq2, intersect, direct_sum, zeros2,
    kernel, span)
from qumba.lin import int_scalar as scalar
from qumba.action import mulclose, mulclose_find
from qumba.matrix import Matrix, DEFAULT_P, pullback
from qumba.symplectic import symplectic_form
from qumba.qcode import QCode, strop
from qumba.util import cross
from qumba import construct


# Crucial equation: 
# beta(u,v)-beta(v,u)=2*omega(u,v) in Z/4Z. 
# See Gurevich & Hadani sec 0.2


# modified from qumba/_weil_repr.py:
def beta(u, v):
    # beta is a 2-cocycle for constructing a "Heisenberg group"
    # as a central extension:
    # Z_4 >---> H(V) -->> V
    assert u.shape == v.shape
    u0 = u[::2]
    v0 = v[1::2]
    uv = u0.t*v0
    result = 2*int(uv)
    return result



class Pauli:
    def __init__(self, vec, phase=0):
        assert phase in [0,1,2,3]
        vec = Matrix.promote(vec)
        self.vec = vec
        nn = len(vec)
        assert nn%2==0
        self.n = nn//2
        self.phase = phase
        self.key = (vec, phase)

    def __str__(self):
        v = self.vec
        s = strop(v, "I")
        phase = (self.phase + s.count("Y")) % 4
        phase = ["", "i", "-", "-i"][phase]
        return phase + s

    def sign(self):
        v = self.vec
        s = strop(v, "I")
        phase = (self.phase + s.count("Y")) % 4
        assert phase in [0,2]
        return [1,-1][phase//2]

    def __repr__(self):
        v = self.vec
        s = strop(v, "I")
        return "Pauli(%s, %s)"%(s, self.phase)

    def __eq__(self, other):
        assert isinstance(other, Pauli)
        return self.key == other.key

    def __hash__(self):
        return hash(self.key)

    def __mul__(self, other):
        assert isinstance(other, Pauli)
        assert other.n == self.n
        vec = self.vec + other.vec
        phase = (self.phase + other.phase + beta(self.vec, other.vec)) % 4
        return Pauli(vec, phase)

    def __neg__(self):
        return Pauli(self.vec, (self.phase+2)%4)

    def __matmul__(self, other):
        vec = self.vec.concatenate(other.vec)
        return Pauli(vec, (self.phase + other.phase)%4)

    def get_wenum(self):
        vec = self.vec
        n = self.n
        ws = [0,0,0,0] # I,X,Y,Z
        for i in range(n):
            u = vec[2*i:2*i+2]
            #print(str(u), end=' ')
            #i = str(u).count("1")
            i = {'..':0,'1.':1,'11':2,'.1':3}[str(u)]
            ws[i] += 1
        return tuple(ws)

    def get_full_wenum(self):
        vec = self.vec
        n = self.n
        ws = []
        for i in range(n):
            u = vec[2*i:2*i+2]
            i = {'..':0,'1.':1,'11':2,'.1':3}[str(u)] # w,x,y,z
            ws.append(i)
        return tuple(ws)



#    @classmethod
#    def fromstr(cls, h):
        

I = Pauli([0,0])
wI = Pauli([0,0], 1)
nI = wI*wI
X = Pauli([1,0])
Y = Pauli([1,1], 3)
Z = Pauli([0,1])
w_phase = Pauli([], 1)

def fromstr(h):
    ops = [{"I":I, "X":X, "Y":Y, "Z":Z}[op] for op in h]
    op = reduce(matmul, ops)
    return op


#def get_poly



def get_full_wenum(code, verbose=False):
    #print(code.longstr())
    H = code.H
    #for h in strop(H):
    stabs = []
    for h in H:
        #print(strop(h))
        g = fromstr(strop(h, "I"))
        #print(g)
        stabs.append(g)

    for h in stabs:
      for g in stabs:
        assert h*g == g*h

    L = code.L
    LX = fromstr(strop(L[0], "I"))
    LZ = fromstr(strop(L[1], "I"))
    LY = w_phase@LX*LZ
    LI = LX*LX
    if verbose:
        print("get_wenum", code)
        print(code.longstr())
        print("LX =", LX)
        print("LY =", LY)
        print("LZ =", LZ)
        print("LI =", LI)
    assert LX*LZ==-LZ*LX

    from sage import all_cmdline as sage
    n = code.n
    gens = []
    for i in range(n):
        gens.append("w%d"%i)
        gens.append("x%d"%i)
        gens.append("y%d"%i)
        gens.append("z%d"%i)
    R = sage.PolynomialRing(sage.ZZ, gens)
    gens = R.gens()

    def get_poly(S):
        p = 0
        for s in S:
            r = s.sign()
            idxs = s.get_full_wenum()
            for i,idx in enumerate(idxs): 
                g = gens[4*i + idx]
                r = g*r
            #print(r, s, ws)
            p = p+r
        return p

    S = mulclose(stabs)
    p = get_poly(S)

    result = []
    for g in [LX,LY,LZ,LI]:
        S1 = [g*s for s in S]
        p = get_poly(S1)
        #print(S1, p)
        result.append(p)

    return result


def get_wenum(code, verbose=False):
    #print("get_wenum")
    #print(code.longstr())
    H = code.H
    #for h in strop(H):
    gens = []
    for h in H:
        #print(strop(h))
        g = fromstr(strop(h, "I"))
        #print(g)
        gens.append(g)

    for h in gens:
      for g in gens:
        assert h*g == g*h

    L = code.L
    LX = fromstr(strop(L[0], "I"))
    LZ = fromstr(strop(L[1], "I"))
    LY = w_phase@LX*LZ
    LI = LX*LX
    if verbose:
        print("get_wenum", code)
        print(code.longstr())
        print("LX =", LX)
        print("LY =", LY)
        print("LZ =", LZ)
        print("LI =", LI)
    assert LX*LZ==-LZ*LX

    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, list("xyzw"))
    x,y,z,w = R.gens()
    p = 0

    S = mulclose(gens)

    def get_poly(S):
        p = 0
        for s in S:
            wenum = s.get_wenum()
            #print(repr(s), s, wenum)
            r = s.sign()
            for e,v in zip(wenum, (w,x,y,z)):
                #print(r,v,e)
                r = r * (v**e)
            p = p+r
        return p

    result = []
    for g in [LX,LY,LZ,LI]:
        S1 = [g*s for s in S]
        p = get_poly(S1)
        #print(S1)
        #print(' '.join(str(s) for s in S1))
        result.append(p)

    return result





def test():

    assert Y == wI*X*Z
    assert X*Z == -Z*X
    assert X*Z*X*Z == nI
    assert X*X == I
    assert Y*Y == I
    assert Z*Z == I

    G = mulclose([wI,X,Y,Z])
    assert len(G) == 16

    XI = X@I
    IZ = I@Z
    assert XI*IZ == IZ*XI

    G2 = set(g@h for g in G for h in G)
    assert len(G2) == 4*len(G), len(G2)

    G2 = list(G2)
    G2.sort(key = str)
    for g in G2:
        s = str(g)
        phase = 0
        if s.startswith("i"):
            phase = 1
            s = s[1:]
        elif s.startswith("-i"):
            phase = 3
            s = s[2:]
        elif s.startswith("-"):
            phase = 2
            s = s[1:]
        phase = Pauli([], phase)
        assert g == phase @ fromstr(s)

    assert str(X@Y@Z@I) == "XYZI"
    assert str(X@Y@Y@I) == "XYYI"
    assert str(nI@X@Z@I) == "-IXZI"

    code = construct.get_15_1_3()
    op = code.space.S()
    assert (op*code).is_equiv(code)

    #code = construct.get_412()
    #code = QCode.fromstr("YYZI IXXZ ZIYY")
    #code = construct.get_10_2_3()
    #code = QCode.fromstr("ZZZII IIZZZ XIXXI IXXIX")
    #code = QCode.fromstr("XXXX ZZZZ YYII")

    #code = construct.get_surface(4,4)
    #print(code)
    #result = get_wenum(code)
    #print(result[0])


def test_wenum():

    from sage import all_cmdline as sage
    R = sage.PolynomialRing(sage.ZZ, list("xyzw"))
    x,y,z,w = R.gens()
    I = sage.I

    from qumba.distill import get_code

#    #code = construct.get_913()
#    code = construct.get_512()
#
#    _code = QCode.fromstr("""
#    ZZ.ZZ....
#    .XX.XX...
#    ...XX.XX.
#    ....ZZ.ZZ
#    XX.......
#    .......XX
#    ...Z..Z..
#    ..Z..Z...
#    """, None, "X..X..X.. ZZZ......")

    code = get_code()
    assert code is not None

    pstr = lambda wenum : str(wenum).replace("*", "")

    result = get_wenum(code)
#    for i,wenum in enumerate(result):
#        print("[%d]:"%i, pstr(wenum))
#        #print("\t", sage.factor(wenum))

    px, py, pz, pw = result
    #print( "px", pstr(px) )
    #print( "py", pstr(py) )
    #print( "pz", pstr(pz) )
    #print( "pw", pstr(pw) )
    print()

    #print( px(x=z, y=1, z=0, w=1) )
    #print( px(x=1, y=z, z=0, w=z) )

    tgt = 0

    # [[7,1,3]] idx=1
    #tgt = z**7+7*z**5+7*z**3-7*z # pw(1, I, z, z)
    #tgt = 7*z**6 - 7*z**4 - 7*z**2 - 1 # px(-1, I, z, z)

    # [[8,1,3]]
    #tgt = z**8 - 12*z**6 + 38*z**4 - 12*z**2 + 1  # pw(1, I, z, z)
    tgt = 8*(z**7 + z**5 - z**3 - z) # py(-1, I, z, z)

    #tgt = -z**13 - 65*z**9 + 117*z**5 + 13*z # pw(1, I, z, z)
    #tgt = 13*z**12 + 117*z**8 - 65*z**4 - 1 # pz(1, I, z, z)

    # [[9,1,3]]
    #tgt = z**9 + 3*z**3
    #print( pw(x=1, y=I, z=z, w=z) ) # 64z**9 + 192*z**3

    def factor(p):
        if p==0:
            return p
        return sage.factor(p)

    print("px =", px(x=1, y=I, z=z, w=z), end=" " )
    print("\t=", factor(px(x=1, y=I, z=z, w=z) ))
    print("py =", py(x=1, y=I, z=z, w=z), end=" " )
    print("\t=", factor(py(x=1, y=I, z=z, w=z) ))
    print("pz =", pz(x=1, y=I, z=z, w=z), end=" " )
    print("\t=", factor(pz(x=1, y=I, z=z, w=z) ))
    print("pw =", pw(x=1, y=I, z=z, w=z), end=" " )
    print("\t=", factor(pw(x=1, y=I, z=z, w=z) ))

    return

    if argv.tgt:
        tgt = eval(argv.tgt, locals())
    print("tgt =", tgt)

    if not tgt:
        return

    found = set()
    for i,poly in enumerate(result):
        name = "px py pz pw".split()[i]
        for items in cross([(z,1,-1, I, -I)]*4):
            p1 = poly(x=items[0], y=items[1], z=items[2], w=items[3])
            #if p1 in found:
            #    continue
            #found.add(p1)
            #s = str(p1)
            #if "z" in str(s) and p1.degree() == 7:
            #    if "14" in s or len(s)<10 or "21" in s:
            #        continue
            #    print(s, ":", items)
            if p1 == tgt or p1==-tgt:
                print(p1, ":", "%s%s"%(name, items))

    return

#    wenum = result[0]
#    top = (wenum(w=1, x=1, y=1, z=z))
#    bot = (wenum(w=0, x=z, y=0, z=1))
#    print(top, "/", bot)

    wenum = result[3]
    top = (wenum(w=z, x=1, y=0, z=0))
    bot = (wenum(w=1, x=z, y=0, z=0))
    print(top, "/", bot)
    top = sage.factor(top)
    bot = sage.factor(bot)
    print(top, "/", bot)



if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

    start_time = time()


    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)

    if profile:
        #import cProfile as profile
        #profile.run("%s()"%name)
        from pyinstrument import Profiler
        with Profiler(interval=0.01) as profiler:
            test()
        profiler.print()



    elif name is not None:
        fn = eval(name)
        fn()

    else:
        test()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)


