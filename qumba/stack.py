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


def get_distance(code):
    css = code.to_css()
    if css.n > 100:
        print(css)
        distance_z3_css(css, verbose=True)
    else:
        distance_z3_css(css)
    return css

def show_info(code):
    css = get_distance(code)
    print(css)

def dump_transverse(code):

    from qumba.gcolor import dump_transverse
    code = code.to_css()
    #code.bz_distance()
    if argv.distance:
        distance_z3_css(code, verbose=True)
        print(code)
    dump_transverse(code.Hx, code.Lx)

    


def test_periodic():
    N = argv.get("N", 4)
    assert N%4 == 0

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
        #css.bz_distance()
        #print(css)
    
    if argv.stack:
        code = stack3(*codes)
        print(code)

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
    if N==2:
        C1 = get_middle_12() # fix a glitch 
    else:
        C1 = geometry.get_code("green")
    C2 = geometry.get_code("blue")

    print("green:")
    print(C1)
    print(C1.longstr())
    #return

    if argv.verbose:
        print(C0.longstr())
        print()
        print(C1.longstr())
        print()
        print(C2.longstr())
        print()

    if not argv.stack:
        return

    code = stack3(C0, C1, C2)
    print(code)
    if argv.verbose:
        print(code.longstr())

    if argv.render:
        render_geometry(geometry, "cubeocta_stack")

    css = code.to_css()

    Hx = css.Hx
    print(Hx, Hx.shape)

    N, perms = Hx.get_autos()
    print(N)
    print(perms)

    #from qumba.autos import get_autos_css
    #result = get_autos_css(css)
    #print(result)

    return

    css = code.to_css()
    Hx = css.Hx
    from qumba.triorthogonal import is_morthogonal
    print("is_morthogonal(2):", is_morthogonal(Hx, 2))
    print("is_morthogonal(3):", is_morthogonal(Hx, 3))
    print("is_morthogonal(4):", is_morthogonal(Hx, 4))

    print(Hx, Hx.shape)
    print()
    #w = Hx.get_wenum()
    #print(w)

    m, n = Hx.shape
    rows = []
    for bits in numpy.ndindex((2,)*m):
        u = numpy.dot(bits, Hx.A)%2
        if u.sum() == 12:
            rows.append(u)
    U = Matrix(rows)
    print(U, U.shape, U.rank())


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
    code = stack3(C0, C1, C2)

    if argv.stitch:
        stitch(code)


def get_middle_12():
    # this is N=2 test() with a hack
    # see also test_12 

    # broken code:
    code = QCode.fromstr("""
    X.X.X....X..
    XX....X.X...
    ...XXX....X.
    .....XXX...X
    ZZZ.........
    ...Z.Z.Z....
    Z...ZZZ.....
    ....Z....ZZ.
    ......Z.Z..Z
    """)

    # See cubeocta_stack_num.pdf
    """
        E   HGF      
        012345678901
        X.X.X....X..
        XX....X.X...
        .....XXX...X
        ...XXX....X.
        ZZZ.........
        Z.......ZZ.. +
        ...Z.Z.Z....
        .....Z....ZZ +
        ....Z....ZZ.
        ..ZZZ....... +
        ......Z.Z..Z
        .Z....ZZ.... +
        Z...ZZZ.....
    """

    H = fromstr("""
        X.X.X....X..
        XX....X.X...
        .....XXX...X
        ...XXX....X.
        ZZZ.........
        Z.......ZZ..
        ...Z.Z.Z....
        .....Z....ZZ
        ....Z....ZZ.
        ..ZZZ.......
        ......Z.Z..Z
        .Z....ZZ....
        Z...ZZZ.....
    """)
    #print(H, H.shape, H.rank())
    H = H.linear_independent()
    code = QCode(H)

    css = code.to_css()
    css.bz_distance()
    #print(css)
    return code

    
def stitch(code):
    import stabgraph as stg
    import flag_stitcher

    code.distance("z3")

    #H = code.H.concatenate(
    stabs = strop(code.H, "I").split()
    L = strop(code.L, "I").split()
    stabs += [op for op in L if "Z" in op]
    #print(stabs)
    
    #stabs=['XXIXIXI','IXXXXII','IIIXXXX','ZZIZIZI','IZZZZII','IIIZZZZ','ZZZIIII']
    #distance=3
    distance = code.d
    if distance % 2==0:
        distance -= 1
    G,c,t,_,_=stg.convert(stabs,shuffle=True)
    circuit = flag_stitcher.build_ft_circuit(G,c,t,distance)

    print("circuit:", len(circuit))
    print(circuit[:100], "...")



def stack3(C0, C1, C2):

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
    css = code.to_css()
    print(css)
    wx = ([int(w) for w in css.Hx.sum(1)])
    print({w:wx.count(w) for w in set(wx)})
    wz = ([int(w) for w in css.Hz.sum(1)])
    print({w:wz.count(w) for w in set(wz)})
    if argv.distance:
        show_info(C0)
        show_info(C1)
        show_info(C2)
        show_info(code)
    if argv.dump:
        print(code.longstr())

    return code


def test_stack2():

    C0 = construct.get_512()
    C1 = C0.get_dual()
    print(C0.longstr())
    print(C1.longstr())

    D = stack2(C0, C1)
    D.distance("z3")
    print(D)
    print(D.longstr())
    assert D.is_selfdual()




def stack2(C0, C1):

    n = C0.n
    assert C1.n == n

    right = C0+C1
    print(right)

    cube = construct.get_422()
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
    css = code.to_css()
    print(css)
    wx = ([int(w) for w in css.Hx.sum(1)])
    print({w:wx.count(w) for w in set(wx)})
    wz = ([int(w) for w in css.Hz.sum(1)])
    print({w:wz.count(w) for w in set(wz)})
    if argv.distance:
        show_info(C0)
        show_info(C1)
        show_info(C2)
        show_info(code)
    if argv.dump:
        print(code.longstr())

    return code



def test_14():
    code = construct.get_15_1_3()
    code = code.shorten(0)
    print(code)
    code = code.to_css()
    code.bz_distance()
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)


def test_16():
    s = """
    0123456789012345
    11.11...........
    1.11..1.........
    .1..11.1........
    ..1...1.1.1.....
    ...11.1111.11...
    .....1.1.1...1..
    ........1.11..1.
    ...........11.11
    .........1..11.1
    """
    s = s.strip()
    s = s[16:].strip()
    H = Matrix.parse(s)
    print(H, H.shape, H.rank())
    print()
    assert (H * H.t).sum() == 0

    H = H.linear_independent()
    css = CSSCode(Hx=H, Hz=H)
    print(css)




def test_autos():
    N = 2

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
    if N==2:
        C1 = get_middle_12() # fix a glitch 
    else:
        C1 = geometry.get_code("green")
    C2 = geometry.get_code("blue")

    print("green:")
    print(C1)
    print(C1.longstr())
    #return

    code = stack3(C0, C1, C2)
    print(code)
    if argv.verbose:
        print(code.longstr())

    if argv.render:
        render_geometry(geometry, "cubeocta_stack")

    css = code.to_css()

    Hz = css.Hz
    Hx = css.Hx
    print(Hx, Hx.shape)

    #N, perms = Hx.get_autos()
    #print(N)
    #print(perms)

    perms =  [
[1, 0, 3, 2, 5, 4, 7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[2, 3, 0, 1, 6, 7, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 49, 48, 51, 50, 53, 52, 55, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 50, 51, 48, 49, 54, 55, 52, 53, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 52, 53, 54, 55, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 40, 43, 42, 45, 44, 47, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 40, 41, 46, 47, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 44, 45, 46, 47, 40, 41, 42, 43, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 32, 35, 34, 37, 36, 39, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 32, 33, 38, 39, 36, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 36, 37, 38, 39, 32, 33, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 14, 15, 12, 13, 16, 17, 18, 19, 22, 23, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 16, 19, 18, 21, 20, 23, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 6, 7, 4, 5, 8, 9, 10, 11, 14, 15, 12, 13, 18, 19, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 16, 17, 18, 19, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 64, 67, 66, 69, 68, 71, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 66, 67, 64, 65, 70, 71, 68, 69, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 68, 69, 70, 71, 64, 65, 66, 67, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 56, 59, 58, 61, 60, 63, 62, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 58, 59, 56, 57, 62, 63, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 60, 61, 62, 63, 56, 57, 58, 59, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 28, 29, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 24, 27, 26, 29, 28, 31, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 24, 25, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 46, 47, 44, 45, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 63, 60, 61, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 28, 29, 30, 31, 24, 25, 26, 27, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 54, 55, 52, 53, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 70, 71, 68, 69, 66, 67, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 94, 95, 92, 93, 90, 91],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 88, 91, 90, 93, 92, 95, 94],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 90, 91, 88, 89, 94, 95, 92, 93],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 72, 75, 74, 77, 76, 79, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 74, 75, 72, 73, 78, 79, 76, 77, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 76, 77, 78, 79, 72, 73, 74, 75, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 36, 37, 34, 35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 78, 79, 76, 77, 74, 75, 80, 81, 86, 87, 84, 85, 82, 83, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 80, 83, 82, 85, 84, 87, 86, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 82, 83, 80, 81, 86, 87, 84, 85, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 9, 8, 11, 10, 13, 12, 15, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 8, 9, 14, 15, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 8, 9, 10, 11, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95],
[0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53, 54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35, 36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 72, 73, 74, 75, 76, 77, 78, 79, 64, 65, 66, 67, 68, 69, 70, 71, 88, 89, 90, 91, 92, 93, 94, 95, 80, 81, 82, 83, 84, 85, 86, 87],
[40, 41, 42, 43, 44, 45, 46, 47, 24, 25, 26, 27, 28, 29, 30, 31, 56, 57, 58, 59, 60, 61, 62, 63, 8, 9, 10, 11, 12, 13, 14, 15, 48, 49, 50, 51, 52, 53, 54, 55, 0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39, 16, 17, 18, 19, 20, 21, 22, 23, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
[48, 49, 52, 53, 50, 51, 54, 55, 64, 65, 68, 69, 66, 67, 70, 71, 88, 89, 92, 93, 90, 91, 94, 95, 80, 81, 84, 85, 82, 83, 86, 87, 40, 41, 44, 45, 42, 43, 46, 47, 32, 33, 36, 37, 34, 35, 38, 39, 0, 1, 4, 5, 2, 3, 6, 7, 72, 73, 76, 77, 74, 75, 78, 79, 8, 9, 12, 13, 10, 11, 14, 15, 56, 57, 60, 61, 58, 59, 62, 63, 24, 25, 28, 29, 26, 27, 30, 31, 16, 17, 20, 21, 18, 19, 22, 23]]


    print(len(perms))

    N = 879

    for perm in perms:
        J = Hx[:, perm]
        u = J.t.solve(Hx.t)
        assert (u is not None)

        J = Hz[:, perm]
        u = J.t.solve(Hz.t)
        assert (u is not None)

        dode = code.apply_perm(perm)
        assert (dode.is_equiv(code))
        L = (dode.get_logical(code))

    from bruhat.gset import Group, Perm, mulclose
    perms = [Perm(idxs) for idxs in perms]
    #G = mulclose(perms, verbose=True)
    G = Group(gens=perms, build=False)

    from bruhat.gap import Gap
    gap = Gap()
    _G = gap.define(G)
    print(gap.Order(_G, get=True)) # 8796093022208 = 2**43

    #return

    print(G.structure_description())
    #print(len(G))

    print(len(G))

    #from qumba.autos import get_autos_css
    #result = get_autos_css(css)
    #print(result)


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



