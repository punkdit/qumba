#!/usr/bin/env python

"""

See:
https://arxiv.org/abs/1801.04255
Three-dimensional surface codes: Transversal gates and
fault-tolerant architectures
Michael Vasmer, Dan E. Browne

"""


import warnings
warnings.filterwarnings('ignore')


from math import gcd
from functools import reduce, cache
from operator import matmul, add, mul, lshift
from random import random, randint, choice, shuffle

import sage.all_cmdline as sage

import numpy

from huygens.namespace import *
from huygens.pov import View, Mat

from qumba.argv import argv
from qumba.qcode import strop, QCode, SymplecticSpace, fromstr
from qumba.csscode import CSSCode
from qumba import construct 
from qumba.matrix import Matrix
from qumba.lin import shortstr



def render(view, poly, fill):
    for v in poly.faces(0):
        v = v.vertices()[0]
        v = Mat(v)
        view.add_circle(v, 1.0, fill=grey)

    edges = poly.faces(1)
    for e in edges:
        v0, v1 = e.vertices()
        v0, v1 = Mat(v0), Mat(v1)
        view.add_line(v0, v1, lw=0.4, st_stroke=[black]+st_round)

    faces = poly.faces(2)
    for f in faces:
        vs = [Mat(v) for v in f.vertices()]
        if len(vs) == 3:
            view.add_poly(vs, fill=fill)
            continue
        if len(vs) < 3:
            assert 0 # ?!
            continue
        ws = [vs.pop(0)]
        while vs:
            w = ws[-1]
            ds = [(w-v).norm() for v in vs]
            idx = numpy.argmin(ds)
            v = vs.pop(idx)
            ws.append(v)
        view.add_poly(ws, fill=fill)


class Geometry:
    def __init__(self, items=[]):
        self.items = []
        self.lookup = {}
        self.verts = []
        for item in items:
            self.add(*item)

    def __getitem__(self, i):
        return self.items[i]

    def add(self, poly, deco):
        assert deco in "red blue green".split()
        self.items.append((poly, deco))
        lookup = self.lookup
        for v in poly.vertices():
            if v in lookup:
                continue
            lookup[v] = len(lookup)
            self.verts.append(v)

    def remove(self, poly, deco):
        self.items.remove((poly, deco))

    def render(self, view):
        for (poly, deco) in self.items:
            cl = eval(deco)
            render(view, poly, cl.alpha(0.5))

    def get(self, *decos):
        items = [item for item in self.items if item[1] in decos]
        return Geometry(items)

    def clip(self, ieq):
        # take intersection with half-space, coords are (a,x,y,z)
        halfspace = sage.Polyhedron(ieqs=[ieq], base_ring=sage.ZZ)
        items = []
        for poly,deco in self.items:
            p1 = poly.intersection(halfspace)
            if len(p1.vertices())>1:
                items.append((p1, deco))
        return Geometry(items)

    def vertices(self):
        vs = reduce(add, [item[0].vertices() for item in self.items])
        vs = list(set(vs))
        return vs

    def get_code(self, deco):
        print("get_code", deco)
        vs = self.vertices()
        #lookup = {v:i for (i,v) in enumerate(vs)}
        lookup = self.lookup
        n = len(lookup)
        nn = 2*n
        assert n == len(vs)
        others = "red green blue".split()
        others.remove(deco)
        xstab = self.get(deco)   
        ops = []
        for (poly,_) in xstab:
            verts = [tuple(v) for v in poly.vertices()]
            op = ['.']*n
            for v in poly.vertices():
                op[lookup[v]] = 'X'
            #print(op)
            ops.append(''.join(op))
        #print("xstab:", len(ops))
        Hx = fromstr('\n'.join(ops))
        green, red = others
        green = self.get(green)
        red = self.get(red)
        for (l,_) in green:
          for (r,_) in red:
            face = l.intersection(r)
            if len(face.vertices()) < 2:
                #print("skip", face)
                continue
            op = ['.']*n
            vec = [0]*nn
            for v in face.vertices():
                op[lookup[v]] = 'Z'
                vec[2*lookup[v]] = 1
            if len(face.vertices()) == 2:
                vec = Matrix(vec)
                #print(vec, Hx*vec)
                if (Hx*vec).sum():
                    #print("skipping")
                    continue
                #assert 0 # nope..
                #print("adding zstab on", face)
            ops.append(''.join(op))
        #for (poly,_) in faces:
        ops = '\n'.join(ops)
        #print(ops)
        H = fromstr(ops)
        #print("H:", H.shape)
        H = H.linear_independent()
        #print("H:", H.shape)
        code = QCode(H)
        #print(code.longstr())
        css = (code.to_css())
        css.bz_distance()
        print(css)
        return code


def make_view():
    stroke = orange
    st_axis = st_thick+[grey]

    v0 = Mat([0,0,0])
    cx = Mat([1,0,0])
    cy = Mat([0,1,0])
    cz = Mat([0,0,1])

    view = View(sort_gitems=True)
    view.perspective()

    x, y, z = 4, 1.5, 10
    view.lookat([x, y, z], [0., 0, 0], [0, 1, 0]) # eye, center, up

    L = 2
    view.add_line(v0-L*cx, L*cx, st_stroke=st_axis+st_arrow)
    view.add_line(v0-L*cy, L*cy, st_stroke=st_axis+st_arrow)
    view.add_line(v0-L*cz, L*cz, st_stroke=st_axis+st_arrow)

    return view


def test():
    cubo = sage.polytopes.cuboctahedron()
    for v in cubo.vertices():
        print(tuple(v), end=' ')
    print()

    octa = sage.polytopes.octahedron()
    for v in octa.vertices():
        print(tuple(v), end=' ')
    print()
    
    geometry = Geometry()
    for j in range(3):
        dy = 2*(j-1)
        for i in range(-1,3):
          for k in range(-1,3):
            dx = 2*i
            dz = 2*k
            if (i+k+j)%2==0:
                deco = "red"
            else:
                deco = "blue"
            p = cubo.translation((dx, dy, dz))
            geometry.add(p, deco)
    
    for j in range(3):
        dy = 2*(j-1)
        for i in range(3):
          for k in range(3):
            dx = 2*i
            dz = 2*k
            p = octa.translation((dx-1,dy-1,dz-1))
            geometry.add(p, "green")

    #geometry = geometry.get("blue")
    #geometry = geometry.get("green", "red")

    up = (0,1,0)
    dn = (0,-1,0)
    fwd = (0,0,1)
    back = (0,0,-1)
    left = (-1,0,0)
    right = (+1,0,0) # pointing right
    geometry = geometry.clip((+2,)+dn)
    geometry = geometry.clip((+2,)+up)
    geometry = geometry.clip((+1,)+fwd) # back at -1
    geometry = geometry.clip((+3,)+back) # front (closest) boundary at +3
    geometry = geometry.clip((+3,)+left)
    geometry = geometry.clip((+1,)+right) # left boundary

    plane = lambda *arg:sage.Polyhedron(eqns=[arg], base_ring=sage.ZZ)
    front = plane(3,0,0,-1)
    back = plane(1,0,0,+1)
    left = plane(3,-1,0,0)
    right = plane(1,1,0,0)
    for (poly,deco) in list(geometry):
        if deco=="blue" and (
            poly.intersection(front)==poly or poly.intersection(back)==poly):
            geometry.remove(poly, deco)
        if deco=="red" and (
            poly.intersection(left)==poly or poly.intersection(right)==poly):
            geometry.remove(poly, deco)

    cvs = Canvas()
    view = make_view()
    geometry.render(view)
    cvs = view.render(bg=None)
    cvs.writePDFfile("cubeocta3.pdf")

    cvs = Canvas()
    x = 0
    for deco in "red green blue".split():
        view = make_view()
        geometry.get(deco).render(view)
        fg = view.render(bg=None)
        cvs.insert(x, 0, fg)
        x = x + 1.2*fg.get_bound_box().width

    cvs.writePDFfile("cubeocta.pdf")

    print("vertices:", len(geometry.vertices()))

    code = geometry.get_code("red")
    code = geometry.get_code("green")
    code = geometry.get_code("blue")

    if 0:
        css = code.to_css()
        Hx, Hz = css.Hx, css.Hz
        wz = Hz.sum(0)
    
        op = "....X........X........X.................X...X......"
        radius = 7.0
        idxs = list(range(code.n))
        for (i,w) in enumerate(wz):
            if w:
                continue
            idxs.remove(i)
            v = geometry.verts[i]
            print(i, v)
            view.add_circle(Mat(v), 2, fill=red)
            radius -= 1
    
    #    for v in geometry.verts:
    #        if v[2]==3:
    #            print(v)


    #Hz = Hz[:, idxs]
    #Hx = Hx[:, idxs]
    #code = CSSCode(Hx=Hx, Hz=Hz)
    #print(code) # [[45,0,?]]



def test_14():
    code = construct.get_15_1_3()
    code = code.shorten(0)
    print(code)
    code = code.to_css()
    code.bz_distance()
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)

    

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


