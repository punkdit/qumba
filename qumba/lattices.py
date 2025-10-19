#!/usr/bin/env python

from sage.all_cmdline import QQ
one = QQ.gens()[0]

from huygens import config
config(text="pdflatex")
from huygens.namespace import (
    Canvas, path, red, black, orange, st_arrow, grey,
    st_southwest, Scale)

from qumba.matrix_sage import Matrix
from qumba.argv import argv



class Bag:
    def __init__(self, ops):
        ops = list(ops)
        ops.sort(key = str)
        self.ops = tuple(ops)

    def __len__(self):
        return len(self.ops)

    def __eq__(self, other):
        return self.ops == other.ops

    def __hash__(self):
        return hash(self.ops)

    def __str__(self):
        s = ','.join(str(op).replace("\n","") for op in self.ops)
        return "Bag(%s)"%s
    __repr__ = __str__

    def __getitem__(self, i):
        return self.ops[i]

    def __mul__(self, other):
        if isinstance(other, Matrix):
            ops =  [op*other for op in self.ops]
        else:
            assert isinstance(other, Bag)
            ops =  [op*pp for op in self.ops for pp in other.ops]
        return Bag(ops)

    def __rmul__(self, other):
        assert isinstance(other, Matrix)
        ops =  [other*op for op in self.ops]
        return Bag(ops)


def generate(gens, accept=lambda op:True, verbose=False, maxsize=None):
    gens = [op for op in gens if accept(op)]
    els = set(gens)
    bdy = list(els)
    while bdy:
        if verbose:
            print(len(els), end=" ", flush=True)
        _bdy = []
        for A in gens:
            for B in bdy:
                for C in [A*B, B*A]:
                  if C not in els and accept(C):
                    els.add(C)
                    _bdy.append(C)
                    if maxsize and len(els)>=maxsize:
                        return els
        bdy = _bdy
    if verbose:
        print()
    return Bag(els)


def find_orbit(gens, x):
    orbit = set([x])
    bdy = list(orbit)
    while bdy:
        _bdy = []
        for g in gens:
            x1 = g*x
            if x1 not in orbit:
                _bdy.append(x1)
                orbit.add(x1)
        bdy = _bdy
    return orbit


class Show:
    def __init__(self):

        st = st_arrow+[orange.alpha(0.5)]
        cvs = Canvas()
        #cvs.stroke(path.rect(0,0,1,1))
        cvs.stroke(path.line(-1, 0, 3, 0), st)
        cvs.stroke(path.line(0, -1, 0, 3), st)
        self.cvs = cvs

    def show(self, vs, radius=0.05, st=[]):
        cvs = self.cvs
        for v in vs:
            assert isinstance(v, Matrix)
            x, y, _ = [eval(str(xx)) for xx in v[:, 0]]
            cvs.stroke(path.circle(x, y, radius), st)
        return self

    def save(self, name="lattice"):
        self.cvs.writePDFfile(name)
        return self

    def render(self, op, label=''):
        assert isinstance(op, Matrix)
        assert op.shape == (3,3)
        assert op[0,2] == 0
        assert op[1,2] == 0
        op = op[:2, :2]
        ev = op.eigenvectors()
        for (val,vec,dim) in ev:
            if val == 1:
                break
        else:
            assert 0
        x, y = [eval(str(xx)) for xx in vec[:, 0]][:2]
        cvs = self.cvs
        cvs.stroke(path.line(-x, -y, x, y), [grey])
        if label:
            cvs.text(x, y, label, st_southwest+[Scale(0.5)])

    def draw_poly(self, vs):
        pts = []
        for v in vs:
            x, y = [eval(str(xx)) for xx in v[:, 0]][:2]
            pts.append([x,y])
        #print("draw_poly", pts)
        from huygens.the_turtle import Turtle
        from huygens.front import RGB
        from random import random
        st = [RGB(random(),random(),random())]
        t = Turtle(*pts[0], cvs=self.cvs)
        for p in pts[:]:
            t.moveto(*p)
        t.stroke(closepath=True, attrs=st)
        
        
    

from qumba.qcode import QCode, fromstr, lin
def get_sd(n, checks):
    ops = []
    for check in checks:
        for o in 'XZ':
            op = ['.']*n
            for i in check:
                op[i] = o
            op = ''.join(op)
            ops.append(op)
    H = fromstr(ops)
    H = lin.linear_independent(H)
    print(n, H.shape)

    code = QCode(H)
    return code


def test():

    I = Matrix(QQ, [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    tx = Matrix(QQ, [
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 1],
    ])

    ty = Matrix(QQ, [
        [1, 0, 0],
        [0, 1, 1],
        [0, 0, 1],
    ])

    A = Matrix(QQ, [
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ])

    B = Matrix(QQ, [
        [1, 0, 0],
        [0,-1, 0],
        [0, 0, 1],
    ])

    C = Matrix(QQ, [
        [-1, 0, 1],
        [ 0, 1, 0],
        [ 0, 0, 1],
    ])

    assert I*I == I
    assert A*A == I
    assert B*B == I
    assert (A*B)**2 != I
    assert (A*B)**3 != I
    assert (A*B)**4 == I

    gens = [A, B, C]
    v0 = Matrix(QQ, [[1,0,1]]).t
    v0 = Matrix(QQ, [[one/3, one/6, 1]]).t

    def accept(op):
        v = op*v0
        x = v[0,0]
        y = v[1,0]
        return -one<=x<=3 and -one<=y<=3

    G = generate(gens, accept)
    print(len(G))

    AB = generate([A, B]) # octa
    AC = generate([A, C]) # octa
    BC = generate([B, C]) # square

    def accept(v):
        x = v[0,0]
        y = v[1,0]
        return 0<=x and 0<=y and (x+y)<=5*one/2 and  (x-y)<=one/2 
    bits = set(op*v0 for op in G if accept(op*v0))
    bits = list(bits)
    bits.sort(key=str)
    mask = set(bits)
    lookup = {bit:i for (i,bit) in enumerate(bits)}

    s = Show()
    s.show(bits, st=[black.alpha(0.3)])
    #s.show([v0], st=[red])
    #s.show([C*v0], st=[red])

    # act on cosets on the left
    #s.show(set(op*v0 for op in A*B*BC), st=[red])


    checks = []

    for H in [AB, AC, BC]:
        faces = find_orbit(G, H)
        print("faces:", len(faces))
        for face in faces:
            face = set(g*v0 for g in face)
            face = face & mask
            if not len(face): continue
            idxs = [lookup[f] for f in face]
            checks.append(idxs)
            #s.show(face, st=[red.alpha(0.5)])


    #s.render(A, r"$A$")
    #s.render(B, r"$B$")

    #print(checks)
    checks = [c for c in checks if len(c) >= 4]

    for idxs in checks:
        vs = [bits[i] for i in idxs]
        s.draw_poly(vs)
    s.save()

    n = len(bits)
    code = get_sd(n, checks)
    print(code)



if __name__ == "__main__":

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





