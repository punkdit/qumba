#!/usr/bin/env python

import numpy

from sage.all_cmdline import QQ
one = QQ.gens()[0]

from huygens import config
config(text="pdflatex")
from huygens.namespace import (
    Canvas, path, red, black, orange, st_arrow, grey,
    st_southwest, Scale)

from qumba.qcode import QCode, fromstr, lin, shortstr
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
        self.fg = Canvas()

    def show(self, vs, radius=0.05, st=[]):
        from huygens.namespace import white
        cvs = self.fg
        for v in vs:
            assert isinstance(v, Matrix)
            x, y, _ = [eval(str(xx)) for xx in v[:, 0]]
            p = path.circle(x, y, radius)
            cvs.fill(p, [white])
            cvs.stroke(p, st)
        return self

    def save(self, name="lattice"):
        self.cvs.append(self.fg)
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

    def draw_poly(self, vs, st=[]):
        from huygens.the_turtle import Turtle
        from huygens.front import RGB
        from huygens.namespace import st_round
        from random import random

        pts = []
        x0,y0 = 0,0
        for v in vs:
            x, y = [eval(str(xx)) for xx in v[:, 0]][:2]
            x0 += x; y0 += y
            pts.append([x,y])
        x0 /= len(pts)
        y0 /= len(pts)
        pts.sort(key = lambda xy:(xy[0]-x0)**2+(xy[1]-y0)**2)
        ps = [pts.pop()]
        while pts:
            best = None
            rbest = 9999
            x0,y0 = ps[-1]
            for (x1,y1) in pts:
                r = (x1-x0)**2+(y1-y0)**2
                if r < rbest:
                    best = [x1,y1]
                    rbest = r
            assert best in pts, (best, pts)
            pts.remove(best)
            ps.append(best)

        t = Turtle(*ps[0], cvs=self.cvs)
        for p in ps[:]:
            t.moveto(*p)
        #st = [RGB(random(),random(),random())]+st_round
        t.fill(closepath=True, attrs=st)
        
        
    

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
    #print(shortstr(H))
    #print(n, H.shape)

    # remove dead qubits
    cols = []
    for (i,weight) in enumerate(H.sum(axis=0)):
        if weight:
            cols.append(i)
    H = H[:, cols]

    code = QCode(H)

    return code


def get_wenum(code):
    assert code.is_selfdual()
    css = code.to_css()
    H = css.Hx
    m, n = H.shape
    wenum = [0]*(n+1)
    assert m < 21, m
    for v in numpy.ndindex((2,)*m):
        d = lin.dot2(v, H).sum()
        wenum[d] += 1
    return wenum


def test():

    I = Matrix(QQ, [
        [1, 0, 0],
        [0, 1, 0],
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
        return -one<=x<=4*one/2 and -2<=y<=2

    G = generate(gens, accept)
    print(len(G))

    AB = generate([A, B]) # octa
    AC = generate([A, C]) # octa
    BC = generate([B, C]) # square

    def accept(v):
        x = v[0,0]
        y = v[1,0]
        return (x-y)>=0  and (x+y)>=0 
    verts = set(op*v0 for op in G if accept(op*v0))
    verts = list(verts)
    verts.sort(key=str)

    mask = set(verts)

    def r_accept(v):
        x,y = v[0,0], v[1,0]
        return (x-y)>one/3  and (x+y)>=0 
    def g_accept(v):
        x,y = v[0,0], v[1,0]
        return (x-y)>=0  and (x+y)>one/3
    def b_accept(v):
        x,y = v[0,0], v[1,0]
        return (x-y)>=0  and (x+y)>=0 

    r_mask = {v for v in verts if r_accept(v)}
    b_mask = {v for v in verts if b_accept(v)}
    g_mask = {v for v in verts if g_accept(v)}

    lookup = {vert:i for (i,vert) in enumerate(verts)}

    s = Show()
    s.show(verts, st=[black.alpha(0.3)])
    #s.show([v0], st=[red])
    #s.show([C*v0], st=[red])

    # act on cosets on the left
    #s.show(set(op*v0 for op in A*B*BC), st=[red])

    checks = []
    colours = []

    for idx,H in enumerate([AB, AC, BC]):
        faces = find_orbit(G, H)
        #print("faces:", len(faces))
        for face in faces:
            face = set(g*v0 for g in face)
            face = face & [r_mask, g_mask, b_mask][idx]
            if len(face) < 4:
                continue
            idxs = [lookup[f] for f in face]
            checks.append(idxs)
            colours.append(idx)
            #s.show(face, st=[red.alpha(0.5)])


    #s.render(A, r"$A$")
    #s.render(B, r"$B$")

    #print(checks)
    #checks = [c for c in checks if len(c) >= 4]

    from huygens.namespace import red, green, blue

    for idx,check in zip(colours, checks):
        vs = [verts[c] for c in check]
        cl = [red, green, blue][idx]
        s.draw_poly(vs, [cl])
    s.save()

    for d in range(1,12,2):
        n = d**2/2 + d - 1/2
        print("\t[[%d,1,%d]]"%(n,d))

    n = len(verts)
    code = get_sd(n, checks)
    print(code)

    wenum = get_wenum(code)
    print(wenum)

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
    """)
    wenum = get_wenum(code)
    print(wenum)


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





