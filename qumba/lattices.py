#!/usr/bin/env python
"""
build some k=1 colour codes

https://arxiv.org/abs/1108.5738

"""

import math
from random import random

import numpy

from sage import all_cmdline as sage
QQ = sage.QQ
one = QQ.gens()[0]

from huygens import config
config(text="pdflatex")
from huygens.namespace import (
    Canvas, path, red, black, orange, st_arrow, grey, st_thin, yellow, LineWidth,
    st_southwest, Scale, Rotate)

from huygens.the_turtle import Turtle
from huygens.front import RGB, Rotate, RGBA
from huygens.namespace import red, green, blue, orange, yellow, white, st_round

from qumba.qcode import QCode, fromstr, lin, shortstr
from qumba.matrix_sage import Matrix
from qumba.argv import argv
from qumba import transversal 



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

def pystr(x):
    s = str(x)
    s = s.replace("?", "")
    s = s.replace("r3", str(3**0.5))
    try: 
        eval(s)
    except:
        print(s)
        raise
    return s

def pyeval(x):
    s = pystr(x)
    return eval(s)


# red,green,blue colour scheme
scheme = [
    #red.alpha(0.8) + 0.2*white,
    RGB(1.00, 0.30, 0.2).alpha(0.8),
    RGB(0.10, 0.5, 0.30).alpha(0.8),
    RGBA(0.10, 0.2000, 0.7000, 0.5000),
]


class Show:
    def __init__(self, cvs=None, axis=False):
        st = st_arrow+[orange.alpha(0.5)]
        if cvs is None:
            cvs = Canvas()
        if axis:
            #cvs.stroke(path.rect(0,0,1,1))
            cvs.stroke(path.line(-3, 0, 3, 0), st)
            cvs.stroke(path.line(0, -1, 0, 3), st)
        self.cvs = cvs
        self.fg = Canvas()

    def show(self, vs, radius=0.02, st=[0.3*white]+[LineWidth(0.01)]):
        cvs = self.fg
        for v in vs:
            assert isinstance(v, Matrix)
            x, y, _ = [eval(pystr(xx)) for xx in v[:, 0]]
            p = path.circle(x, y, radius)
            cvs.fill(p, [white])
            cvs.stroke(p, st)
        return self

    def save(self, name="lattice"):
        print("save:", name)
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
        x, y = [eval(pystr(xx)) for xx in vec[:, 0]][:2]
        cvs = self.cvs
        cvs.stroke(path.line(-x, -y, x, y), [grey])
        if label:
            cvs.text(x, y, label, st_southwest+[Scale(0.5)])

    def draw_poly(self, vs, st=[]):
        pts = []
        x0,y0 = 0,0
        for v in vs:
            x, y = [eval(pystr(xx)) for xx in v[:, 0]][:2]
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
        
        
    

def get_sd(n, checks, d=None):
    ops = []
    for check in checks:
        for o in 'XZ':
            op = ['.']*n
            for i in check:
                op[i] = o
            op = ''.join(op)
            ops.append(op)
    H = fromstr(ops)
    #H = lin.linear_independent(H)
    #print(shortstr(H))
    #print(n, H.shape)

    # remove dead qubits
    cols = []
    for (i,weight) in enumerate(H.sum(axis=0)):
        if weight:
            cols.append(i)
    H = H[:, cols]

    code = QCode(H, d=d)

    return code


def get_wenum(code):
    assert code.is_selfdual()
    css = code.to_css()
    H = css.Hx
    m, n = H.shape
    wenum = [0]*(n+1)
    assert m <= 31, m
    for v in numpy.ndindex((2,)*m):
        d = lin.dot2(v, H).sum()
        wenum[d] += 1
    return wenum


def build_colour_488(d=3):
    assert d>0, d
    assert d%2, d

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
        lim = ((d+1)//2)*one/2 
        return 0<=x<=lim and -lim<=y<=lim

    G = generate(gens, accept)
    #print("G =", len(G))

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

    s = Show(Canvas([Scale(2), Rotate(math.pi/2)]))
    s.show(verts, 0.02, st=[0.3*white]+[LineWidth(0.01)])

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

#    scheme = [ 
#        grey+RGB(0,0,0.0), orange.alpha(0.8), yellow.alpha(0.6),
#        RGBA(0.10, 0.2000, 0.7000, 0.5000),
#        green+RGB(0,0,0.2)]

    for idx,check in zip(colours, checks):
        vs = [verts[c] for c in check]
        cl = scheme[idx]
        s.draw_poly(vs, [cl])
    if argv.save:
        s.save("lattice_488_%d"%d)

    n = len(verts)
    code = get_sd(n, checks, d)
    return code


def build_colour_666(d=3):
    global one
    assert d>0, d
    assert d%2, d

#    w = sage.polygen(sage.ZZ, 'w')
#    K = sage.NumberField(w**2+w+1, names=('w',))
#    w, = K.gens()
#    print(K)
#    assert w**3==1
#
#    print( (w+w**2) )


    if 0:
        K = sage.QQbar # VERY SLOW ....
    
        r3 = (3*one)**(one/2)
        r3 = K(r3)
    else:

        r3 = sage.polygen(sage.ZZ, 'r3')
        K = sage.NumberField(r3**2-3, names=('r3',))
        r3, = K.gens()
        assert r3**2==3

    one = K.one()

    I = Matrix(K, [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ])

    A = Matrix(K, [
        [one/2, r3/2, 0],
        [r3/2, -one/2, 0],
        [0, 0, 1],
    ])

    B = Matrix(K, [
        [1, 0, 0],
        [0,-1, 0],
        [0, 0, 1],
    ])

    C = Matrix(K, [
        [-1, 0, one/2],
        [ 0, 1, 0],
        [ 0, 0, 1],
    ])

    assert I*I == I
    assert A*A == I
    assert B*B == I
    assert (A*B)**2 != I
    assert (A*B)**3 != I
    assert (A*B)**6 == I

    #gens = [A, B, C]
    AB = A*B
    BC = B*C
    gens = [AB, BC] # rotation subgroup
    #v0 = Matrix(K, [[one/2, 0, 1]]).t
    v0 = Matrix(K, [[one/4, r3/12, 1]]).t

    def accept(op):
        if op.shape == (3,3):
            v = op*v0
        else:
            v = op
        x = v[0,0]
        y = v[1,0]
        lim = 3
        return -lim<=x<=lim and -lim<=y<=lim

    G = generate(gens, accept)
    #print("G =", len(G))

    verts = [g*v0 for g in G]
    #for v in verts:
    #    print(v[0,0], ",", v[1,0])

    s = Show(Canvas([Scale(2), Rotate(+math.pi/6)]))

    face = generate([A,B])
    assert len(face) == 12

    g1 = C*AB*C*AB*C*AB*AB*A
    g2 = AB*C*AB*C*AB*C*AB*AB*AB*A

    tx = Matrix(K, [
        [1, 0,  one/2],
        [ 0, 1, 0],
        [ 0, 0, 1],
    ])
    ty = AB*tx*AB*AB*AB*AB*AB
    T = [tx**i*ty**j for i in range(-3,3) for j in range(-3,3)]

    v1 = Matrix(K, [[one/8, r3/12, 1]]).t

    #s.show([v1], 0.05, st=[red.alpha(0.5)])
    #s.show([tx*v1], 0.05, st=[blue.alpha(0.5)])
    #s.show([ty*v1], 0.05, st=[blue.alpha(0.5)])

    # index 3 sublattice
    N = 4
    H = []
    #verts = []
    for i in range(-N,N+1):
      for j in range(-N,N+1):
        g = g1**i * g2**j
        v = g*v1
        if accept(v):
            #verts.append(v)
            H.append(g)

    for (i,op) in enumerate([I, tx, ty]):
      for g in H:
        ps = [op*g*h*v0 for h in face]
        #s.draw_poly(ps, st=[scheme[i]])

    #print(v0)
    dx, dy = -one*3/12, r3/4
    def accept(v):
        i = d//2
        assert i>0
        aa = [0,2,5,5,8,8,11,11]
        bb = [0,3,3,6,6,9,9,]
        x, y = v[0,0],v[1,0]
        if x < 0: return False # green bdy
        #if x*dx-y*dy <= -2*one/8: return False # blue d==3
        if x*dx-y*dy <= -aa[i]*one/8: return False # blue d==7
        #if x*dx-y*dy <= -6*one/8: return False # red d==7

        #if x*dx+y*dy <= -3*one/8: return False # red d==3
        if x*dx+y*dy <= -bb[i]*one/8: return False # red d==7
        #if x*dx+y*dy <= -5*one/8: return False # blue d==7
        return True

    verts = set([g*v0 for g in G])
    found = set()
    for v in verts:
        x, y = v[0,0],v[1,0]
        u = x*dx + y*dy
        v = x*dx - y*dy
        found.add(u)
        #print((u,v), end=' ')
    found = list(found)
    found.sort()
    #print(found)
    print()

    verts = [v for v in verts if accept(v)]
    verts.sort(key = str)
    lookup = {v:i for (i,v) in enumerate(verts)}

    #print(verts[0])
    #s.show(verts, 0.03, st=[black.alpha(0.3)])
    s.show(verts, 0.02, st=[0.3*white]+[LineWidth(0.01)])

    n = len(verts)
    
    checks = []
    for (ii,op) in enumerate([I, tx, ty]):
      for g in H:
        vs = [op*g*h*v0 for h in face]
        idxs = {lookup.get(v) for v in vs if v in lookup}
        if len(idxs)<4:
            continue
        #print(idxs)
        s.draw_poly([verts[i] for i in idxs], st=[scheme[ii]])
        h = [0]*n
        for i in idxs:
            h[i] = 1
        checks.append(h)

    H = lin.array2(checks)
    #print(H.shape)

    if argv.save:
        s.save("lattice_666_%d"%d)
        s.save()

    code = QCode.build_css(H,H, d=d)
    print(code)

    return code


def test_666():

    for d in range(3,10,2):
        code = build_colour_666(d)



def show_params():

    for d in range(1,15,2):
        n = (d**2-1)/2 + d
        assert int(n) == n
        print("$[[%d,1,%d]]$ "%(n,d), end=' ')
        n = (3*d**2+1)/4
        assert int(n) == n
        print("& $[[%d,1,%d]]$ "%(n,d), end=' ')
        n = (3*d**2+5)/2 - 3*d
        assert int(n) == n
        print("& $[[%d,1,%d]]$ "%(n,d))


def augment(code):
    H = code.to_css().Hx
    m, n = H.shape

    H1 = lin.zeros2(m+1, n+1)
    H1[:m, :n] = H
    H1[m, :] = 1

    return QCode.build_css(H1, H1)


def test():

    R = sage.PolynomialRing(sage.ZZ, "x")
    x = R.gens()[0]

    w2 = x**2 + 1
    w7 = 7*x**4  + 1
    w8 = 1 + 14*x**4 + x**8
    w24 = x**24 + 759 *x**16 + 2576*x**12 + 759*x**8 + 1

    #print(sage.factor(w8)) # two factors
    #print(sage.factor(w24)) # no factors
    #return


    def divide(p, q):
        print("divide by", q)
        div = p // q
        rem = p - div*q
        if rem == 0:
            print("div =", div)
            print("\t =", sage.latex(div))
        else:
            print("rem =", rem)
        assert p == div*q + rem
        return rem

    for d in [3,5,7,9]:
        code = build_colour_488(d)
        continue

        code = build_colour_666(d)
        #continue

        print(code)
        wenum = get_wenum(code)
        w = 0
        for (i,v) in enumerate(wenum):
            w += v*(x**i)
        #print(sage.latex(w))
        s = sage.latex(w)
        print("W(x) = %s"%s)

        code = augment(code)
        print(code)

        wenum = get_wenum(code)

        w = 0
        for (i,v) in enumerate(wenum):
            w += v*(x**i)
        #print(sage.latex(w))
        s = sage.latex(w)
        print("W'(x) = %s"%s)
        print()

        print(sage.factor(w))
        print()

        divide(w, w24)
        divide(w, w8)
        divide(w, w2)

        #w1 = w.subs({x:1})
        #assert w1 == 2**((code.n-1)//2)

        print("_"*79)
        print()


def choose_colour():
    d = 7
    build_colour_666(d)


def find_lw():
    d = argv.get("d", 3)

    w = argv.get("w", 4)

    ws = argv.get("ws", [w])

    if d%2:
        code = build_colour_666(d)
        print(code)
    else:
        code = build_colour_666(d-1)
        print(code)
        code = augment(code)
        print(code)

    css = code.to_css()
    Hx = css.Hx

    for w in ws:
        count = 0
        for h in transversal.find_lw(Hx, w):
            #print(h)
            if count%100==0:
                print('.',flush=True,end='')
            count += 1
        print()
        print("+ %d x^%d"%(count, w))

    return



    code = QCode.fromstr("""
XIIIIIIIIXIIXIIXI
IXIIIIIIIXIIXIIIX
IIXIIIIIIIIXIXXII
IIIXIIIIXXIIXIIII
IIIIXIIIXIXIXXXXX
IIIIIXIIIIXXIXIII
IIIIIIXIXXXIIXXXX
IIIIIIIXIIXXIIXII
ZIIIIIIIIZIIZIIZI
IZIIIIIIIZIIZIIIZ
IIZIIIIIIIIZIZZII
IIIZIIIIZZIIZIIII
IIIIZIIIZIZIZZZZZ
IIIIIZIIIIZZIZIII
IIIIIIZIZZZIIZZZZ
IIIIIIIZIIZZIIZII
    """)
    wenum = get_wenum(code)
    print(wenum)


    return

    code = build_colour_488(5)

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





