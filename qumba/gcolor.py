#!/usr/bin/env python

"""
Here we look at Guage Color Codes, Bombin, arXiv:1311.0879v3
"""

import sys, os
from operator import add
from math import sqrt
from random import randint, shuffle

import numpy

from qumba.lin import zeros2, shortstr, dot2, parse, array2, shortstrx
from qumba.lin import find_logops, span, eq2, linear_independent, rank, solve
from qumba.argv import argv
    
from qumba.csscode import CSSCode
from qumba.csscode import check_commute
from functools import reduce


EPSILON = 1e-8


def gcd(p, q):
    if p < 0:
        p = -p
    assert q > 0
    r = q
    while p > 0:
        r = p
        p = q % p
        q = r
    return r


class Rational(object):
    def __init__(self, p, q=1):
        if q==0:
            raise ZeroDivisionError
        if q < 0:
            p = -p
            q = -q
        r = gcd(p, q)
        if r > 1:
            p //= r
            q //= r
        self.p = p
        self.q = q

    @classmethod
    def promote(cls, x):
        if type(x) is Rational:
            return x
        x = int(x)
        return Rational(x)

    def __getitem__(self, idx):
        return self.xs[idx]

    def getreal(A):
        x = float(A.p) / A.q
        return x

    def __str__(A):
        if A.q==1:
            return str(A.p)
        return "%d/%d"%(A.p, A.q)

    def __repr__(A):
        return "Rational(%d, %d)"%(A.p, A.q)

    def __hash__(A):
        #assert gcd(A.p, A.q)==1
        return hash((A.p, A.q))

    def __eq__(A, B):
        B = Rational.promote(B)
        return A.p == B.p and A.q == B.q

    def __ne__(A, B):
        B = Rational.promote(B)
        return A.p != B.p or A.q == B.q

    def __ge__(A, B):
        B = Rational.promote(B)
        return A.getreal() >= B.getreal() - EPSILON # i'm so lazy
        #return A.p*B.q >= B.p*A.q
    def __gt__(A, B):
        B = Rational.promote(B)
        return A.getreal() > B.getreal() + EPSILON # i'm so lazy

    def __add__(A, B):
        #if type(B)!=int and type(B)!=Rational:
        #    return NotImplemented
        B = Rational.promote(B)
        q = A.q * B.q
        p = A.p * B.q + B.p * A.q
        return Rational(p, q)
    __radd__ = __add__

    def __neg__(A):
        return Rational(-A.p, A.q) 

    def __sub__(A, B):
        #if type(B)!=int and type(B)!=Rational:
        #    return NotImplemented
        B = Rational.promote(B)
        q = A.q * B.q
        p = A.p * B.q - B.p * A.q
        return Rational(p, q)

    def __rsub__(A, B):
        B = Rational.promote(B)
        return -A + B

    def __mul__(A, B):
        if type(B) not in (Rational, int, int):
            return NotImplemented
        B = Rational.promote(B)
        p = A.p * B.p
        q = A.q * B.q
        return Rational(p, q)

    __rmul__ = __mul__

    def __truediv__(A, B):
        #if type(B)!=int and type(B)!=Rational:
        #    return NotImplemented
        B = Rational.promote(B)
        p = A.p * B.q
        q = A.q * B.p
        return Rational(p, q)

    def __rdiv__(A, B):
        assert 0, (A, B)

    def __pow__(A, n):
        return Rational(A.p**n, A.q**n)

    def floor(A):
        k = A.p // A.q
        return k

    def __abs__(A):
        return Rational(abs(A.p), abs(A.q))

#    def __mod__(A, B):
#        assert A>=0
#        B = Rational.promote(B)
#        if abs(A) >= B:
#            C = A / B
#            k = C.p//C.q
#            C = C - k
#        else:
#            C = A
#        return C

    def __mod__(A, B):
        B = Rational.promote(B)
        assert B>=0
        p = (A.p*B.q) % (B.p*A.q)
        q = A.q*B.q
        return Rational(p, q)

    def __cmp__(A, B):
        #if type(B)!=int and type(B)!=Rational:
        #    return NotImplemented
        B = Rational.promote(B)
        #print "__cmp__", A, B
        return cmp(A.p*B.q, B.p*A.q)

class Vector(object):
    colormap = {
        Rational(0, 4) : "red",
        Rational(1, 4) : "yellow",
        Rational(2, 4) : "green",
        Rational(3, 4) : "blue",
    }
    cache = {}

    def __new__(cls, *xs):
        if len(xs)==1 and type(xs[0]) in (list, tuple):
            xs = xs[0]
        xs = [Rational.promote(x) for x in xs] # for consistent hash!
        xs = tuple(xs)
        ob = cls.cache.get(xs)
        if ob is not None:
            return ob
        ob = object.__new__(cls)
        ob.xs = xs
        ob.d = len(xs)
        ob.rng = list(range(ob.d))
        cls.cache[xs] = ob
        return ob

    def __init__(self, *xs):
        pass

    #def __lt__(self, other):
    #    return str(self) < str(other)

    @property
    def color(self):
        key = self.dot(l0) % 1
        color = Vector.colormap.get(key, '')
        return color

    def __getitem__(self, idx):
        return self.xs[idx]

    def getreal(self):
        return tuple(x.getreal() for x in self.xs)

    def __str__(self):
        return "%s<%s>"%(self.color[:1], ','.join(str(x) for x in self.xs))

    def __repr__(self):
        return "Vector%s"%(str(self.xs),)

    @classmethod
    def promote(cls, v):
        if type(v) is Vector:
            return v
        v = Vector(v)
        return v

    def __add__(A, B):
        B = Vector.promote(B)
        assert A.d == B.d
        xs = [A.xs[i] + B.xs[i] for i in A.rng]
        return Vector(xs)

    def __sub__(A, B):
        assert A.d == B.d
        B = Vector.promote(B)
        xs = [A.xs[i] - B.xs[i] for i in A.rng]
        return Vector(xs)

    def __neg__(A):
        xs = [-1*A.xs[i] for i in A.rng]
        return Vector(xs)

    def __rmul__(A, r):
        xs = [r*A.xs[i] for i in A.rng]
        return Vector(xs)

    def __div__(A, r):
        xs = [A.xs[i]//r for i in A.rng]
        return Vector(xs)

    def __hash__(A):
        return hash(A.xs)

#    def __eq__(A, B):
#        B = Vector.promote(B)
#        assert A.d == B.d
#        return A.xs==B.xs
#
#    def __ne__(A, B):
#        B = Vector.promote(B)
#        assert A.d == B.d
#        return A.xs!=B.xs
#
#    def __cmp__(A, B):
#        return cmp(A.xs, B.xs)

    def dot(A, B):
        B = Vector.promote(B)
        assert A.d == B.d
        B = Vector.promote(B)
        r = sum(A.xs[i]*B.xs[i] for i in A.rng)
        return r

    def norm(A):
        r = sqrt(sum(x.getreal()**2 for x in A.xs))
        return r

    def normalized(A):
        r = A.norm()
        xs = [x.getreal()/r for x in A.xs]
        return xs

    def cross(A, B):
        assert A.d == 3
        B = Vector.promote(B)
        a, b = A.xs, B.xs
        xs = [
            a[1]*b[2]-a[2]*b[1],
           -a[0]*b[2]+a[2]*b[0],
            a[0]*b[1]-a[1]*b[0]]
        return Vector(xs)


#class Color(object):
#    def __init__(self, name):
#        self.name = name
#    @property
#    def color(self):
#        return self.name
#    def __str__(self):
#        return self.name
#    def __repr__(self):
#        return "Color(%r)"%self.name
#
#colors = [Color(name) for name in 'red yellow green blue'.split()]
#del Color

colors = 'red yellow green blue'.split()

class Simplex(object):
    "just a set of points"
    cache = {}
    def __new__(cls, vs):
        vs = list(vs)
        vs.sort(key = str) # canonical form
        vs = tuple(vs)
        ob = cls.cache.get(vs)
        if ob is not None:
            assert ob.d==len(vs)-1
            return ob
        ob = object.__new__(cls)
        ob.vs = vs
        ob.d = len(vs)-1
#        ob.is_qubit = False
        cls.cache[vs] = ob
        #if ob.d==0:
        #    print "__new__", ob, hash(ob), hash(ob.vs[0])
        return ob

    def __init__(self, vs):
        pass

    def __str__(self):
        return "%d-{%s}"%(self.d, ', '.join(str(v) for v in self.vs))

    def get_center(self):
        v = sum(self.vs, Vector(0, 0, 0))
        v = v // len(self.vs)
        return v

#    def __eq__(A, B):
#        return A.vs == B.vs
#
#    def __ne__(A, B):
#        return A.vs != B.vs

    def __cmp__(A, B):
        return cmp((A.d, A.vs), (B.d, B.vs))

    def __hash__(A):
        return hash(A.vs)

    def subiter(A):
        for i in range(len(A.vs)):
            vs = A.vs[:i] + A.vs[i+1:]
            s = Simplex(vs)
            yield s

    def deepiter(A):
        for s in A.subiter():
            for s1 in s.deepiter():
                yield s1
            yield s

    def contains(A, s):
        for s1 in A.deepiter():
            if s is s1:
                return True
        return False

    def intersects(A, B):
        "do A & B intersect on a d-1 dimensional simplice ?"
        Asub = list(A.subiter())
        for s in B.subiter():
            if s in Asub:
                return True
        return False

    def complement(A):
        vs = list(A.vs)
        cs = [(v.color if isinstance(v, Vector) else v) for v in vs]
        cs = [c for c in colors if c not in cs]
        return tuple(cs)

    def is_real(A):
        for v in A.vs:
            if type(v) != str:
                return True
        return False

    def is_internal(A):
        for v in A.vs:
            if type(v) == str:
                return False
        return True



r12 = Rational(1, 2)

l0 = Vector(r12, r12, r12)
assert l0 is Vector(r12, r12, r12)
l1 = Vector(r12, -r12, -r12)
l2 = Vector(-r12, r12, -r12)
l3 = Vector(-r12, -r12, r12)

assert 4*l0.dot(l0) == 3

assert Rational(1) % r12 == 0
assert r12 % 3 == r12


def find_reduced_guage(Gx, Hx):

#    print "find_reduced_guage"
#    print shortstr(Hx)

    mg, n = Gx.shape
    mx, _ = Hx.shape
    
    m = mx
    A = Hx
    ra = rank(Hx)
    assert ra==mx
    for i in range(mg):
        assert A.shape == (m, n)
        B = zeros2(m+1, n)
        B[:m] = A
        B[m] = Gx[i]
        r = rank(B) # XX work with a row reduced A
        assert m+1>=r>=m, (m, r)
        if r==m+1:
            m += 1
            A = B

    return A[mx:]


class Lattice(object):
    def __init__(self, n):

        self.n = n
    
        #simplices = {0:[], 1:[], 2:[], 3:[]} # map d -> list of d-simplices
        simplices = {0:set(), 1:set(), 2:set(), 3:set()} # map d -> list of d-simplices
        children = {} # map d-simplex -> (d-1)-simplices
        parents = {} # map d-simplex -> (d+1)-simplices

        self.simplices = simplices
        self.children = children # boundary
        self.parents = parents # co-boundary

        # Vertices
        vs = []
        span = list(range(-n, n))
        for i in span:
         for j in span:
          for k in span:
            v = Vector(i, j, k)
            vs.append(v)
            vs.append(v+l0)
    
        for x in vs:
            if not self.accept(x):
                continue
            s = Simplex([x])
            simplices[0].add(s)
        for x in vs:
            if not self.accept(x):
                continue
            s = Simplex([x])
            assert s in simplices[0]
    
        xs = [x for x in vs if self.accept(x)]
        #print "vertices:", len(xs)

        ss = []
        for x in xs:
            ss += list(self.cells3(x))
        #print len(ss)
        ss = [s for s in ss if self.accept(s)]
        #print len(ss)
        ss = set(ss)
        #print "simplices:", len(ss)

        for s in ss:
            assert self.accept(s)
            simplices[3].add(s)
            #print s
            for face in s.subiter():
                #print "face:", face
                simplices[2].add(face)
                self.connect(s, face)
        for face in simplices[2]:
            for edge in face.subiter():
                simplices[1].add(edge)
                self.connect(face, edge)
        for edge in simplices[1]:
            assert self.accept(edge)
            for vertex in edge.subiter():
                assert vertex in simplices[0], vertex
                self.connect(edge, vertex)

        # Complete to form a 3-sphere
        s0 = Simplex(colors) # Do not add to simplices[3] !

        for face in list(simplices[2]):
            count = len(parents[face])
            assert count in (1, 2), count
            if count==2:
                continue
#            print "face:", face
            cs = face.complement()
            assert len(cs)==1, cs
            body = Simplex(face.vs + cs)
#            print body
            simplices[3].add(body)

            for edge in face.subiter():
                count = len(parents[edge])
                assert count >= 2
                if count > 2:
                    continue
#                print "edge:", edge
                cs = edge.complement()
                assert len(cs)==2, cs
                body = Simplex(edge.vs + cs)
#                print body

                simplices[3].add(body)
                for vertex in edge.subiter():
                    count = len(parents[vertex])
                    assert count >= 3
                    if count > 3:
                        continue
#                    print "vertex:", vertex
                    cs = vertex.complement()
                    assert len(cs)==3, cs
                    body = Simplex(vertex.vs + cs)
#                    print body
                    simplices[3].add(body)

        for body in simplices[3]:
            for face in body.subiter():
                if not face.is_real():
                    continue
                simplices[2].add(face)
                if not body in parents.get(face, []):
                    self.connect(body, face)
                for edge in face.subiter():
                    if not edge.is_real():
                        continue
                    simplices[1].add(edge)
                    if not face in parents.get(edge, []):
                        self.connect(face, edge)

        for face in list(simplices[2]):
            count = len(parents[face])
            assert count==2
        for edge in list(simplices[1]):
            vertices = list(edge.subiter())
            assert len(vertices) == 2
            for vertex in vertices:
                if edge not in parents.get(vertex, []):
                    self.connect(edge, vertex)

        qubits = list(set(simplices[3]))
        qubits.sort(key = str)
        #print "qubits:", len(qubits)
        assert len(qubits) == 1+4*n+6*n**2+4*n**3 # magic formula

        # Build a code
        for i, qubit in enumerate(qubits):
#            print i, qubit
            qubit.i = i

#        stabs = []
#        for p in simplices[0]:
#            op = self.get_qubits(p)
#            stabs.append(op)
#            assert len(op) in [8, 12, 18, 24]
#            #print len(op),
#        #print
#        #print len(qubits), len(stabs)

        edges = list(simplices[1])
        edges.sort(key = str)

        guage = []
        for edge in edges:
#            print edge
            op = self.get_qubits(edge)
#            for v in op:
#                print '\t', v
            assert op not in guage
            guage.append(op)
            assert len(op) in [4, 6]

        self.qubits = qubits
        self.edges = edges
        self.guage = guage

        n = len(qubits)
        mg = len(guage)

        Gx = zeros2(mg, n)
        Gz = zeros2(mg, n)

        for j, op in enumerate(guage):
            for qubit in op:
                Gx[j, qubit.i] = 1
                Gz[j, qubit.i] = 1
        self.Gz = Gz
        self.Gx = Gx
        self.mg = mg

        print("guage ops:", mg)
        print("qubits:", n)
        mx = mz = len(simplices[0])
        print("stabs:", mx)

        #print shortstr(Gx)
        #print
        #print shortstr(Gz)
        #print 

        #rGx = linear_independant(Gx)
        ##print shortstr(rGx)
        #print "rGx:", rGx.shape

        vertices = list(simplices[0])
        vertices.sort(key = str)

        Hx = zeros2(mx, n)
        Hz = zeros2(mz, n)

        stabs = []
        for i, vertex in enumerate(vertices):
            for qubit in simplices[3]:
                if qubit.contains(vertex):
                    Hx[i, qubit.i] = 1
                    Hz[i, qubit.i] = 1
        self.Hx = Hx
        self.Hz = Hz

#        Ex = zeros2(len(simplices[2]), n)
#        for i, face in enumerate(simplices[2]):
#            qubits = self.get_qubits(face)
#            for qubit in qubits:
#                Ex[i, qubit.i] = 1
#        self.Ex = Ex

        #rHx = linear_independant(Hx)
        #print shortstr(rHx)
        #print "rHx:", rHx.shape

    def get_syndrome(self, op):
        syndrome = []
        for i in range(len(self.Gz)):
            r = dot2(op, self.Gz[i].transpose())
            if r:
                syndrome.append(i)
        return syndrome

    def get_overlap(self, op):
        syndrome = []
        for i in range(len(self.Gz)):
            r = op * self.Gz[i]
            if r.max():
                syndrome.append(i)
        return syndrome

    def get_op(self, weight=1):
        n = len(self.qubits)
        op = zeros2(n)
        for i in range(weight):
            j = randint(0, n-1)
            op[j] = 1
        return op

    def build_code(self, build=False, check=False):

        qubits = self.qubits
        guage = self.guage

        n = len(qubits)
        m = len(guage)

        Gx, Gz = self.Gx, self.Gz
        #Hx, Hz, Lx, Lz = build_stab(Gx, Gz)
        Hx, Hz = self.Hx, self.Hz

        if build:
            Lx = find_logops(Gz, Hx)
            Lz = find_logops(Gx, Hz)
        else:
            Lx = Lz = None

        code = CSSCode(Lx, Lz, Hx, None, Hz, None, Gx, Gz, build=build, check=check)

        return code

    def connect(self, parent, child):
        #print "connect", parent, child
        assert parent.d == child.d + 1, (parent, child)
        children = self.children.setdefault(parent, set())
        assert child not in children
        children.add(child)
        parents = self.parents.setdefault(child, set())
        assert parent not in parents
        parents.add(parent)

    def get_qubits(self, simplex):
        qubits = set()
        items = [simplex]
        while items:
            for item in items:
                if item.d==3:
                    qubits.add(item)
            items = reduce(add, [list(self.parents.get(item, [])) for item in items])
            items = set(items)
        return qubits

    def __str__(self):
        return "Lattice(%s)" % (
            ', '.join("%d:%d" % (i, len(self.simplices[i])) for i in (3, 2, 1, 0)))

    @classmethod
    def cells3(cls, x):
        "yield all 3-simplices containing point x"
        i = Vector(1, 0, 0)
        j = Vector(0, 1, 0)
        k = Vector(0, 0, 1)
        for s in (-1, +1):
          for (a, b, c) in [
            (i, j, k), (i, k, j), (j, i, k),
            (j, k, i), (k, i, j), (k, j, i)]:
            vs = [x, x+a, x+r12*(a+s*b+c), x+r12*(a+s*b-c)]
            simplex = Simplex(vs)
            yield simplex

    # Implement equation (11)
    ks = [0, 1, 2, 3]
    ls = [l0, l1, l2, l3]
    def accept(self, x):
        n = self.n
        if type(x) is Simplex:
            for x in x.vs:
                if not self.accept(x): # recurse
                    return False
            return True
        for k in self.ks:
            rhs = Rational(k, 4)
            if k==0:
                rhs += n-1
            if x.dot(self.ls[k]) > rhs:
                return False
        return True
    
    def make_op(self, p):
        n = len(self.qubits)
        a = zeros2(n)
        for qubit in self.get_qubits(p):
            a[qubit.i] = 1
        return a


def commutes(gx, gz):
    return (gx*gz).sum()%2==0

def colorof(p):
    if isinstance(p, str):
        return p
    return p.color

"""
Internal faces generate $R$.

Homologically, this is a 3-sphere
with a hole.
The 2-sphere enclosing this
hole is left out of the stabilizer
group and so becomes a logical operator.

Bare logical operator $L_X$, sheet of qubits.

$L_X$ generated by all faces on
on one boundary-face 
& three corners of that boundary-face.

$L_X$ generated by all faces on
one boundary-face of a single bi-colour
& a string along one boundary-edge.

"""

def skip_rank(Hx, R, skip):
    A = Hx.copy()
    r = rank(A)

    for a in R:
        if (a*skip).max():
            continue
        A1 = numpy.concatenate((A, a))
        if rank(A1)==r:
            #write("\n")
            continue
        r = r+1
        A = A1

    return r


def search_skip(Hx, R, r, bitss, skip, stack):
    #print "search_skip", len(stack), stack
    i = len(stack)
    if i==len(bitss):
        return stack
    for j in bitss[i]:
        if j in stack:
            continue
        stack.append(j)
        assert skip[j]==0
        skip[j] = 1
        r1 = skip_rank(Hx, R, skip)
        if r1 == r and search_skip(Hx, R, r, bitss, skip, stack):
            return stack
        assert skip[j]==1
        skip[j] = 0
        stack.pop()


def find_skip(Hx, Lx, R):

    m, n = Hx.shape

    bitss = []
    for i in range(m):
      for _ in range(2): # we double like this, right???
        bits = [j for j in range(n) if Hx[i, j]]
        shuffle(bits)
        bitss.append(bits)
    bits = [j for j in range(n) if Lx[0, j]]
    shuffle(bits)
    bitss.append(bits)

    shuffle(bitss)
    print(bitss)
    print(len(bitss))

    skip = zeros2(n)
#    for i in bits:
#        skip[i] = 1
    #print skip

    r = skip_rank(Hx, R, skip)
    stack = search_skip(Hx, R, r, bitss, skip, [])

    print("FOUND")
    print(stack)


def dump_transverse(Hx, Lx, t=3):
    print("dump_transverse")
    import CSSLO
    SX,LX,SZ,LZ = CSSLO.CSSCode(Hx, Lx)
    #CSSLO.CZLO(SX, LX)
    N = 1<<t # i don't think this works with "N = 2<<t"
    zList, qList, V, K_M = CSSLO.comm_method(SX, LX, SZ, t, compact=True, debug=False)
    for z,q in zip(zList,qList):
        #print(z, q)
        print(CSSLO.CP2Str(2*q,V,N),"=>",CSSLO.z2Str(z,N))
    print()
    #return zList



def main():
    l = argv.get("l", 2)

    lattice = Lattice(l)
    n = len(lattice.qubits)
    print(lattice)

    code = lattice.build_code(build=True, check=True)
    #Ex = lattice.Ex
    code.build_from_gauge()
    Gx, Gz = code.Gx, code.Gz
    Hx, Hz = code.Hx, code.Hz
    Lx = find_logops(Gz, Hx)
    #print Lx
    #print dot2(Lx, Gz.transpose())
    print(code)
    #print(code.longstr())

    #print(shortstrx(Gx))
    Hz = numpy.concatenate((Hz, Gz))
    Hz = linear_independent(Hz)
    print(shortstrx(Hz), Hz.shape)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.bz_distance()
    print(code)

    dump_transverse(code.Hx, code.Lx)

    return

    print("Gx:")
    print(shortstrx(Gx))
    print("Hx:")
    print(shortstrx(Hx))
    print("Lx:")
    print(shortstrx(Lx))

    print("0-simplices (bodies):", len(lattice.simplices[0]))
    print("1-simplices (faces):", len(lattice.simplices[1]))
    print("2-simplices (edges):", len(lattice.simplices[2]))

    corners = []
    edges = []
    faces = []
    internal = []
    for i in range(n):
        gw = Gx[:, i].sum()
        hw = Hx[:, i].sum()
        assert gw>=3
        assert hw in [1, 2, 3, 4]
        assert gw in [3, 5, 6]
        if gw==3:
            corners.append(i)
            assert hw==1
        elif gw==5:
            edges.append(i)
            assert hw==2
        elif gw==6:
            if hw==3:
                faces.append(i)
            else:
                assert hw==4
                internal.append(i)

    assert len(corners) + len(edges) + len(faces) + len(internal) == n

    #return

#    op = zeros2(n)
#    for i in corners:
#        op[i] = 1
#    assert solve(Gx.transpose(), op)

    if 0:
        # ops spanned by gauge operators are all even weight
        for trial in range(100):
    
            m = len(Gx)
            op = zeros2(n)
            for i in range(m//2):
                op += Gx[randint(0, m-1)]
            op %= 2
            w = op.sum()%2
            assert w%2 == 0
            print(op)

    #return

    desc = "stabs gauge strings qubits".split()

    for d in range(4):
        counts = {}
        print("%d:"%d, desc[d])
        for p in lattice.simplices[d]:
            #print '\t', p
            #if not p.is_internal():
            #    continue
            cs = [colorof(v) for v in p.vs]
            cs.sort()
            cs = tuple(cs) + (len(lattice.get_qubits(p)),)
            counts[cs] = counts.get(cs, 0)+1
        print(counts)

    R = []
    for p in lattice.simplices[1]:
        a = lattice.make_op(p)
        a.shape = 1, n
        R.append(a)

    find_skip(Hx, Lx, R)

    return

    A = Hx[:, :]
    r = rank(A)

    source = []
    for p in lattice.simplices[1]:
        if not p.is_internal():
            continue
        vs = p.vs
        v0, v1 = vs
        key = (v1-v0), p
        source.append(key)
    source.sort(key = lambda k_p:(k_p[0].color, k_p[0]), reverse=True)

    source = [(None, p) for p in lattice.simplices[1]]
    gauges = {}
    for key, p in source:
        if p.is_internal():
            vs = p.vs
            v0, v1 = vs
            #write(str(v1-v0))
        a = lattice.make_op(p)
        a.shape = 1, n
        A1 = numpy.concatenate((A, a))
        if rank(A1)==r:
            #write("\n")
            continue
        #write(" OK\n")
        r = r+1
        A = A1
        key = [colorof(v) for v in p.vs]
        key.sort()
        key = tuple(key)# + (len(lattice.get_qubits(p)),)
        gauges.setdefault(key, []).append(p)

    #print

    for key, value in list(gauges.items()):
        print(key, len(value))

    print("rank:", r)
    print(rank(A))
    #print shortstrx(A)

    return

    key = list(gauges.keys())[0]
    A = []
    for op in gauges[key]:
        a = zeros2(n)
        for qubit in lattice.get_qubits(op):
            a[qubit.i] = 1
        A.append(a)
    A = array2(A)
    #print shortstrx(A)
    print(A.shape)
    print(rank(A))

    A = numpy.concatenate((Hx, A))
    print(A.shape)
    print(rank(A))

    #code.build_from_gauge()

    #A = dot2(code.Gx, code.Gz.transpose())
    #print shortstrx(code.Gxr, code.Gzr)
    #print shortstr(A)


    assert rank(Hx)==Hx.shape[0] # L.I.
    #Hx = linear_independant(Hx)
    #Hz = linear_independant(Hz)

    n = code.n
    m = Hx.shape[0] + Hz.shape[0]

    r = n - m - 1
    assert r%2==0

    return

    Rx = []
    Rz = []
    for gx, gz in pairs:
        Rx.append(gx)
        Rz.append(gz)
        Rx.append(gz)
        Rz.append(gx)

    Rx = array2(Rx)
    Rz = array2(Rz)

    print(shortstrx(Rx, Rz))
    assert Rx.shape[0] == r
    assert rank(Rx) == r
    assert rank(numpy.concatenate((Rx, Hx))) == r+m//2

    A = dot2(Rx, Rz.transpose())
    print(shortstrx(A))

    return

    Rx = slow_remove(Gx, Hx)
    Rz = slow_remove(Gz, Hz)

    r = rank(Rx)
    assert r + m + 1 == n

    print("r =", r)

    #print rank(numpy.concatenate((Hx, Gx))), n

    #print
    #print shortstrx(Gx, Gz)

    #return


if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next()
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
        main()


    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")






