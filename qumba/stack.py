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
from qumba.csscode import CSSCode, distance_z3_css
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

    def get_code(self, deco, lookup=None):
        print("get_code", deco)
        vs = self.vertices()
        #lookup = {v:i for (i,v) in enumerate(vs)}
        lookup = self.lookup if lookup is None else lookup
        values = list(set(lookup.values()))
        values.sort()
        n = len(values)
        assert values == list(range(n))
        nn = 2*n
        #assert n == len(vs)
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
                if (Hx*vec).sum(): # XXX this is a bit of a hack...
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
        if css.n < 20:
            css.bz_distance()
        print(css)
        return code


def make_view(x=4, y=1.5, z=10):
    stroke = orange
    st_axis = st_thick+[grey]

    v0 = Mat([0,0,0])
    cx = Mat([1,0,0])
    cy = Mat([0,1,0])
    cz = Mat([0,0,1])

    view = View(sort_gitems=True)
    view.perspective()

    view.lookat([x, y, z], [0., 0, 0], [0, 1, 0]) # eye, center, up

    L = 2
    view.add_line(v0-L*cx, L*cx, st_stroke=st_axis+st_arrow)
    view.add_line(v0-L*cy, L*cy, st_stroke=st_axis+st_arrow)
    view.add_line(v0-L*cz, L*cz, st_stroke=st_axis+st_arrow)

    return view


def build_cubocta(N):
    assert N%2 == 0

    cubo = sage.polytopes.cuboctahedron()
    octa = sage.polytopes.octahedron()
    
    geometry = Geometry()
    for j in range(N): # or N-1 ?
        dy = 2*(j-1)
        for i in range(-1,N):
          for k in range(-1,N):
            dx = 2*i
            dz = 2*k
            if (i+k+j)%2==0:
                deco = "red"
            else:
                deco = "blue"
            p = cubo.translation((dx, dy, dz))
            geometry.add(p, deco)
    
    for j in range(N):
        dy = 2*(j-1)
        for i in range(N):
          for k in range(N):
            dx = 2*i
            dz = 2*k
            p = octa.translation((dx-1,dy-1,dz-1))
            geometry.add(p, "green")
    return geometry


def render_geometry(geometry, name="cubeocta"):
    cvs = Canvas()
    view = make_view(4, 8, 20)
    geometry.render(view)
    cvs = view.render(bg=None)
    cvs.writePDFfile(name + "_all.pdf")

    cvs = Canvas()
    x = 0
    for deco in "red green blue".split():
        view = make_view(4, 8, 20)
        geometry.get(deco).render(view)
        fg = view.render(bg=None)
        cvs.insert(x, 0, fg)
        x = x + 1.2*fg.get_bound_box().width

    cvs.writePDFfile(name + ".pdf")


def test_periodic():
    N = argv.get("N", 4)

    cubo = sage.polytopes.cuboctahedron()
    octa = sage.polytopes.octahedron()
    print("cuboctahedron:", end=' ')
    for v in cubo.vertices():
        print(tuple(v), end=' ')
    print()

    geometry = build_cubocta(N)
    tlookup = {}
    vlookup = {}
    for v in geometry.lookup.keys():
        dw = [0,0,0]
        #dw[2] = 1*(v[0]//N) # attempt to add shear... FAIL
        w = [(vi+dwi)%N for vi,dwi in zip(v,dw)]
        #print(tuple(v), "-->", w, v[0]//N)
        w = tuple(w)
        if w not in tlookup:
            tlookup[w] = len(tlookup)
        i = tlookup[w]
        #print("\t", tuple(v), w, "-->", i)
        vlookup[v] = i
    print(len(geometry.lookup), "-->", len(set(vlookup.values())))

    codes = []
    for deco in "red green blue".split():
        code = geometry.get_code(deco, vlookup)
        codes.append(code)
    
        css = code.to_css()
        css.bz_distance()
        print(css)
    
    if argv.ccz:
        build_ccz(*codes)

    #render_geometry(geometry, "cubeocta_toric")



def test():
    N = argv.get("N", 4)

    geometry = build_cubocta(N)
    #geometry = geometry.get("blue")
    #geometry = geometry.get("green", "red")

    up = (0,1,0)
    dn = (0,-1,0)
    fwd = (0,0,1)
    back = (0,0,-1)
    left = (-1,0,0)
    right = (+1,0,0) # pointing right
    geometry = geometry.clip((+2,)+up) # lower boundary
    geometry = geometry.clip((+N-2,)+dn) # upper boundary
    geometry = geometry.clip((+1,)+fwd) # back at -1
    geometry = geometry.clip((+N-1,)+back) # front (closest) boundary at +3
    geometry = geometry.clip((+1,)+right) # left boundary
    geometry = geometry.clip((+N-1,)+left) # right boundary

    plane = lambda *arg:sage.Polyhedron(eqns=[arg], base_ring=sage.ZZ)
    back = plane(1,0,0,+1) # back at -1
    front = plane(N-1,0,0,-1)
    right = plane(1,1,0,0)
    left = plane(N-1,-1,0,0)
    for (poly,deco) in list(geometry):
        if deco=="blue" and (
            poly.intersection(front)==poly or poly.intersection(back)==poly):
            geometry.remove(poly, deco)
        if deco=="red" and (
            poly.intersection(left)==poly or poly.intersection(right)==poly):
            geometry.remove(poly, deco)

    print("vertices:", len(geometry.vertices()))

    C0 = geometry.get_code("red")
    #print(C0.longstr())
    #print()
    C1 = geometry.get_code("green")
    #print(C1.longstr())
    #print()
    C2 = geometry.get_code("blue")
    #print(C2.longstr())
    #print()

    if argv.ccz:
        build_ccz(C0, C1, C2)


def test_12():
    C0 = QCode.fromstr("""
    XXXXXXXX....
    ....X....XX.
    ......X.X..X
    Z.Z.........
    Z...Z....Z..
    ZZ..........
    Z.....Z.Z...
    ...Z.Z......
    ....ZZ....Z.
    .....Z.Z....
    .....ZZ....Z
    """)
    C1 = QCode.fromstr("""
    X.X.X....X..
    XX....X.X...
    ...XXX....X.
    .....XXX...X
    ZZZ.........
    ..ZZZ.......
    .Z....ZZ....
    ...Z.Z.Z....
    Z.......ZZ..
    ....Z....ZZ.
    ......Z.Z..Z
    """)
    C2 = QCode.fromstr("""
    XXX.........
    ...X.X.X....
    X...XXX.XXXX
    Z.Z.Z.......
    ZZ....Z.....
    ...ZZZ......
    .....ZZZ....
    ....Z....Z..
    ....Z.....Z.
    ......Z.Z...
    ......Z....Z
    """)
    build_ccz(C0, C1, C2)




def build_ccz(C0, C1, C2):

    n = C0.n
    assert C1.n == n
    assert C2.n == n

    right = C0+C1+C2
    print(right)

    cube = construct.get_832()
    #print(cube)

    Er = right.get_encoder()
    #code = QCode.from_encoder(Er, k=3)

    Er = SymplecticSpace(cube.m * n).get_identity() << Er
    #print(Er.shape)
    #print(Er)

    if 0:
        code = QCode.from_encoder(Er, k=3)
        print(code)
        code = code.to_css()
        code.bz_distance()
        print(code)

    #return

    E = cube.get_encoder()

    El = reduce(lshift, [E]*n)
    #print(El.shape)

    idxs = []
    for i in range(n):
      for j in range(cube.m):
        idxs.append(cube.n*i + j)

    N = cube.m*n
    for i in range(cube.k):
      for j in range(n):
        idxs.append(cube.n*j + cube.m + i)

    #print(idxs)

    assert len(set(idxs)) == len(idxs)
    assert set(idxs) == set(range(len(idxs)))

    assert len(idxs)*2 == len(El)
    assert len(idxs) == n * cube.n
    P = SymplecticSpace(n*cube.n).get_perm(idxs).t
    E = El * P * Er
    code = QCode.from_encoder(E, k=right.k)
    #d = code.distance("z3")
    #print(code, d)
    
    #print(code.longstr())
    print(code)

    #return

    if 0:
        code = code.to_css()
        code.bz_distance()
        print(code)

    from qumba.gcolor import dump_transverse
    code = code.to_css()
    #code.bz_distance()
    if argv.distance:
        distance_z3_css(code, verbose=True)
        print(code)
    dump_transverse(code.Hx, code.Lx)

    

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



