#!/usr/bin/env python
"""

"""

from functools import reduce
from operator import matmul, add


import numpy
from numpy import linalg
from numpy import random
from numpy import exp, pi, cos, arccos, sin, arctan

from qumba.argv import argv
from qumba.dense import EPSILON, scalar, w4, w8, Matrix, Space

from qumba.qcode import strop, QCode
from qumba import construct 


# ------------------------------------------------------------
# plot functions: uses https://github.com/punkdit/huygens

def save(cvs, name):
    print("save", name)
    cvs.writePDFfile(name)


def wiremesh(view, polytope, st=[], back=False, front=False):
    from huygens.namespace import st_thick, orange, st_round
    _back, _front = [], []
    for face in polytope:
        if reduce(add, face).sum() < 0:
            _back.append(face)
        else:
            _front.append(face)
    if back:
        polytope = _back
    if front:
        polytope = _front
    count = 0
    for face in polytope:
        n = len(face)
        for i in range(n):
            v0, v1 = face[i], face[(i+1)%n]
            if str(v0) > str(v1):
                continue
            view.add_line(v0, v1, st_stroke=st_round+st)
            count += 1
        #print(face)
    #print("lines:", count)


def render(pts=[], colors=None, connect=False, 
        eye=[0.5,0.4,1.4], up=[0,1,0],
    ):
    from huygens.namespace import (st_thick, orange, st_round, st_arrow, st_center,
        Canvas, st_west, st_south, st_north, color, black, grey, green, blue)
    from huygens.pov import View, Mat
    from huygens import config
    config(text="pdflatex")

    from bruhat.platonic import make_octahedron

    polytope = make_octahedron()
    polytope = [[Mat(list(v)) for v in face] for face in polytope]


    view = View(sort_gitems=True)
    view.perspective()

    eye = Mat(eye)
    eye = eye / eye.norm()
    eye = 8*eye
    up = Mat(up)
    center = Mat([0., 0, 0])
    view.lookat(eye, center, up)
    #view.add_light(point, (1., 1., 1., 1.))

    #lookat = Mat.lookat(eye, center, up)

    fill = (0.9, 0.8, 0., 0.4)
    stroke = (0, 0, 0, 1)

    #view.translate(-4., 0, 2)
    st_axis = st_thick+[orange]
    v0 = Mat([0,0,0])
    vx = Mat([1,0,0])
    vy = Mat([0,1,0])
    vz = Mat([0,0,1])
    view.add_line(v0, vx, st_stroke=st_axis)
    view.add_line(v0, vy, st_stroke=st_axis)
    view.add_line(v0, vz, st_stroke=st_axis)

    #wiremesh(view, cube, [grey], back=True)

    for verts in polytope:
        #print([v for v in verts])
        view.add_poly(verts, fill, None)

    #wiremesh(view, cube, [grey], front=True)
    wiremesh(view, polytope)

    R = 1.3
    view.add_line(vx, 1.3*vx, st_stroke=st_axis+st_arrow)
    view.add_line(vy, 1.3*vy, st_stroke=st_axis+st_arrow)
    view.add_line(vz, 1.5*vz, st_stroke=st_axis+st_arrow)

    R *= 1.1
    view.add_cvs(R*vx, Canvas().text(0, 0, r"$|x\rangle$", st_center))
    view.add_cvs(R*vy, Canvas().text(0, 0, r"$|y\rangle$", st_center))
    view.add_cvs(1.2*R*vz, Canvas().text(0, 0, r"$|z\rangle$", st_center))
    #view.add_circle(v0, 1, stroke=stroke, fill=None)

    if colors is None:
        colors = [green]*len(pts)

    pts = [Mat(v) for v in pts]
    for i,v in enumerate(pts):
        if connect:
            view.add_line(pts[i], pts[(i+1)%len(pts)], stroke=colors[i])
        else:
            view.add_circle(v, 0.3, fill=colors[i])

    bg = color.rgb(0.2, 0.2, 0.2, 1.0)

    cvs = view.render(bg=None)

    #u0 = Mat(lookat[1, :3]) # norm = 1
    #r0 = view.trafo_view_distance(u0)

    #v0 = view.trafo_view(u0)
    #x, y = view.trafo_canvas(v0)
    #cvs.stroke(path.circle(x, y, .1), [blue])

    #print(cvs.get_bound_box())
    #v0 = face[0]
    v0 = Mat([0,0,0])
    u0 = view.trafo_view(v0)
    x, y = view.trafo_canvas(u0)
    #cvs.stroke(path.circle(x, y, 1.))


    return cvs

# end of plot functions
# ------------------------------------------------------------



ket0 = Matrix([[1,0]]).d
ket1 = Matrix([[0,1]]).d

c = Space(1)
H = c.H()
S = c.S()
T = c.T()
X = c.X()
Y = c.Y()
Z = c.Z()
I = c.I



def eqphase(u, v): # equality up to phase
    r = u.d*v
    return abs(r.conjugate()*r - 1) < EPSILON


def mix(u, r=0.0):
    I = Space(1).I
    uu = (1-r)*(u@u.d) + 0.5*r*I # um ...
    r = uu.trace()
    uu = (1/r)*uu
    return uu

def fidelity(rho, sho):
    return (sho*rho).trace().real

def to_rho(x, y, z):
    rho = 0.5*(I + x*X + y*Y + z*Z)
    return rho

def conv(a,b,r=0.5):
    return (1-r)*a + r*b

def metric(p, q):
    return sum( (pi-qi)**2 for (pi,qi) in zip(p,q))**0.5

def normalize(x,y,z):
    r = (x*x+y*y+z*z)**(1/2)
    x, y, z = (x/r,y/r,z/r)
    return x,y,z

def rnd():
    x,y,z = random.normal(0, 1, 3)
    x, y, z = normalize(x,y,z)
    return (x,y,z)

def norm(x,y,z):
    r = (x**2+y**2+z**2)**(1/2)
    return r


class Distill:
    def __init__(self, n):
        self.n = n

    def distill(self, rho):
        return rho

    def diff(self, x0,y0,z0,d=0.1): # FIX THIS TRASH
        vs = []
        x,y,z = self(x0,y0,z0)
        for (x1,y1,z1) in [
            (x0+d,y0,z0),
            (x0,y0+d,z0),
            (x0,y0,z0+d),
        ]:
            x2,y2,z2 = self(x1,y1,z1)
            vs.append([x2-x, y2-y, z2-z])
        vs = numpy.array(vs)
        #print(vs)
        return (vs*vs).sum()

    def plot_fiber(self, tgt, epsilon=0.1, N=100, accept=lambda x,y,z:True):
        print("plot_fiber")
        distill = self.distill
        target = normalize(*tgt)
        pts = []
        while len(pts) < N:
            x,y,z = random.normal(0, 1, 3)
            x, y, z = normalize(x,y,z)
            rho = to_rho(x,y,z)
            sho = distill(rho)
            x1,y1,z1 = self.coords(sho)
            #if metric( (x,y,z), (x1,y1,z1) ) < 0.05:
            #if metric( tgt, (x1,y1,z1) ) < epsilon and (x<0 or y<0 or z<0):
            if metric( tgt, (x1,y1,z1) ) < epsilon and accept(x,y,z):
                pts.append( (x,y,z) )
                print(".",end='', flush=True)
        print()
        return pts


class GateDistill(Distill):
    def __init__(self, n, op):
        self.n = n
        self.op = op

    def distill(self, rho):
        rho_n = reduce(matmul, [rho]*self.n)
    
        op = self.op
        sho = op*rho_n*op.d
        r = sho.trace()
        sho = (1/r)*sho
        return sho

    def coords(self, rho):
        x = (rho*X).trace().real
        y = (rho*Y).trace().real
        z = (rho*Z).trace().real
        #from huygens.pov import Mat
        #v = Mat([x, y, z])
        v = [x,y,z]
        return v

    def __call__(self, x,y,z):
        x,y,z = normalize(x,y,z)
        rho = to_rho(x,y,z)
        sho = self.distill(rho)
        x,y,z = [float(u) for u in self.coords(sho)]
        return x,y,z


class Distill_513(GateDistill):

    def __init__(self):
        n = 5
        Distill.__init__(self, n)

        # encoder for the 513
        # see: https://quantumcomputing.stackexchange.com/a/40280
    
        # reverse order:
        HSSHSSHSH = H*S*S*H*S*S*H*S*H
        SSSH = S*S*S*H
        SSS = S*S*S
        SH = S*H
        HSSS = H*S*S*S
    
        s = Space(5)
        CX = s.CX
    
        E = HSSHSSHSH @ H @ SSSH @ SSS @ SH
        E = CX(1,2) * E
        E = CX(0,3) * E
        E = (I @ SSSH @ I @ H @ I ) * E
        E = CX(3,1) * E
        E = (I @ S @ I @ (S*H) @ I) * E
        E = CX(4,1) * CX(4,0) * E
        E = ((S*H)@I@S@I@I) * CX(0,2)*(HSSS @I@I@I@I) *E
        assert (E * E.d == s.I)
    
        u = Matrix([[1,0]]).d
        assert str(u) == "|0>"
    
        u = I@u@u@u@u # Logical input is the *first* qubit !
    
        v = E*u
        
        for op in [
            X@Z@Z@X@I,
            I@X@Z@Z@X,
            X@I@X@Z@Z,
            Z@X@I@X@Z,
            Z@Z@X@I@X,
        ]:
            assert op*v == v
    
        LX = X@X@X@X@X 
        LZ = Z@Z@Z@Z@Z 
        assert LX*v == v*X
        assert LZ*v == v*Z
    
        SH = S*H
        L = reduce(matmul, [SH]*5)
        assert ( L*v == -v*SH )
    
        A = SH.A
        result =  linalg.eig( A )
    
        vals = (result[0])
        vecs = (result[1])
    
        r0, r1 = vals
        ev = Matrix(vecs[:,0:1])
        eu = Matrix(vecs[:,1:2])
    
        assert (SH * ev == r0*ev)
        assert (SH * eu == r1*eu)
    
        beta = arccos(3**(-1/2))/2
        t = cos(beta)*ket0 + w8*sin(beta)*ket1
        assert (t==ev)
    
        bra0 = Matrix([[1,0]])
        op = reduce(matmul, [I] + [bra0]*4) * E.d
    
        u = reduce(matmul, [ev]*5)
        u = op * u
    
        #print("ev:", ev)
        #print("eu:", eu)
    
        r = u.d * u
        r = r**0.5
        #print("r =", 1/r**2) # 1/6
        u = (1/r)*u
        u = -w8*u
        assert u == eu
        #print(" u:", u)
        
        r = (eu.d*u)
        assert( eqphase(S.d*H*S*eu, ev) )

        op = S.d*H*S*op # cliff correction
        self.op = op
    
#    def distill(self, rho):
#        rho_n = reduce(matmul, [rho]*5)
#    
#        op = self.op
#        sho = op*rho_n*op.d
#        r = sho.trace()
#        sho = (1/r)*sho
#    
#        cliff = S.d*H*S
#        sho = cliff*sho*cliff.d
#        return sho



class CodeDistill(Distill):
    def __init__(self, code):
        H = code.H
        H = strop(H, "I")
    
        n = code.n
    
        space = Space(n)
        P = space.I
        for h in H.split():
            s = space.get_pauli(h)
            P = 0.5*(space.I+s)*P
        assert P*P == P
    
        L = strop(code.L, "I").split()
        LX = space.get_pauli(L[0])
        LZ = space.get_pauli(L[1])
        LY = w4*LX*LZ
        self.LX = LX
        self.LZ = LZ
        self.LY = LY
        self.P = P
        self.Pd = P.d
        self.n = n

    def coords(self, rho):
        x = (rho*self.LX).trace().real
        y = (rho*self.LY).trace().real
        z = (rho*self.LZ).trace().real
        #from huygens.pov import Mat
        #v = Mat([x, y, z])
        v = [float(u) for u in [x,y,z]]
        return v

    def distill(self, rho):
        n = self.n
        P = self.P
        Pd = self.Pd
        rho = reduce(matmul, [rho]*n)
        rho = P*rho*Pd
        rho = (1/rho.trace())*rho
        return rho

    def call(self, x, y, z):
        rho = to_rho(x,y,z)
        sho = self.distill(rho)
        x,y,z = [float(u) for u in self.coords(sho)]
        return x,y,z

    def __call__(self, x,y,z):
        x,y,z = normalize(x,y,z)
        rho = to_rho(x,y,z)
        sho = self.distill(rho)
        x,y,z = [float(u) for u in self.coords(sho)]
        return x,y,z


def get_CH():

    n = 2
    CZ = Space(2).CZ()
    CH = I<<H

    plus = (2**(-1/2))*Matrix([[1,1]])
    zero = Matrix([[1,0]])

    assert abs(plus*plus.d-1) < EPSILON
    op = (plus@I)*CH
    #op = (I@zero)*CH
    #op = (plus@I)*CZ

    proto = GateDistill(n, op)

    return proto


# -------------------------------------------------------------
# testing

def make_plots(proto, name):

    fiber = argv.get("fiber", 100)

    if fiber:
        tgt = normalize(+1,+2,3)
        #accept = lambda x,y,z:x<0
        pts = proto.plot_fiber(tgt, 0.1, fiber)
        cvs = render(pts)
        save(cvs, "test_plot_degree_%s.pdf"%name)
        #return

    print("diff")
    count = argv.get("count", 100)
    tol = argv.get("tol", 0.001)
    diff = proto.diff
    pts = []
    while len(pts) < count:
        x,y,z = rnd()
        r = diff(x,y,z)
        if r<tol:
            pts.append((x,y,z))
            print(".",end='',flush=True)
    print()
    
    cvs = render(pts)
    save(cvs, "test_plot_singular_%s.pdf"%name)
    


def test_code():
    code = None
    idx = argv.get("idx", 0)
    if argv.code:
        n,k,d = argv.code
        if (n,k,d) == (4,1,2):
            code = [
                QCode.fromstr("YYZI IXXZ ZIYY"),
                QCode.fromstr("XXXX ZZZZ YYII")][idx]
        if (n,k,d) == (5,1,2):
            code = construct.get_512()
        if (n,k,d) == (5,1,3):
            code = construct.get_513()
        if (n,k,d) == (7,1,3):
            code = [
                construct.get_713(),
                QCode.fromstr("""
            XXIZIZI
            IXXIZIZ
            ZIXXIZI
            IZIXXIZ
            ZIZIXXI
            IZIZIXX""")][idx]
        if (n,k,d) == (8,1,3):
            code = QCode.fromstr("""
            YYZZIIZZ
            ZYYZZIIZ
            ZZYYZZII
            IZZYYZZI
            IIZZYYZZ
            ZIIZZYYZ
            ZZIIZZYY""")

    elif argv.CZ:

        n = argv.get("n", 2)
        CZ = Space(2).CZ()
        #CH = I<<H
    
        plus = (2**(-1/2))*Matrix([[1,1]])
        zero = Matrix([[1,0]])
    
        assert abs(plus*plus.d-1) < EPSILON
        #op = (plus@I)*CH
        #op = (I@zero)*CH

        if n==2:
            op = (plus@I)*CZ
        else:
            op = (plus@plus@I)*(CZ@I)*(I@CZ)*Space(3).CZ(0,2)
    
        proto = GateDistill(n, op)

    else:
        code = QCode.fromstr("XXXX ZZZZ YYII")

    if code is not None:
        print(code)
        print(code.longstr())
        proto = CodeDistill(code)

    name = argv.get("name")
    assert name is not None

    make_plots(proto, name)




def test_fixed():

    code = construct.get_512()
    proto = CodeDistill(code)

    src = normalize(1,0,1)
    x,y,z = proto(*src)
    print(x,y,z)

    print(proto.diff(*src))

    return

    from huygens.namespace import green, blue, red, Canvas
    diff = proto.diff

    pts = []
    trial = 0
    colors = []
    while len(pts) < 300:
        trial += 1
        x,y,z = src = rnd()
        #if x>0 or z>0:
        #    continue
        tgt = proto(*src)
        r = metric(tgt, src)
        if r>0.05:
            continue
        pts.append(src)
        colors.append(green.alpha(0.6))
        pts.append(tgt)
        colors.append(blue.alpha(0.6))
        tgt = proto(*tgt)
        pts.append(tgt)
        colors.append(red.alpha(0.6))
        
        print(".",end='',flush=True)
    print()
    
    cs = [
        render(pts, colors),
        render(pts, colors, eye=[-1,0.2,-1]),
        #render(pts, colors, eye=[0.1,1,0.1], up=[0.1,0.1,1]),
        render(pts, colors, eye=[1,0,1]),
    ]

    cvs = Canvas()

    x = 0.
    for c in cs:
        cvs.insert(x,0,c); 
        x += 7
    save(cvs, "test_plot_fix.pdf")
    



def test_plot():
    pts = [(0.5,0,0),]
    cvs = render(pts)

    save(cvs, "test_plot.pdf")


def test_rho():
    rho = to_rho(*normalize(1,1,1))
    rho = to_rho(0.9,0,0)
    print(rho)
    print(rho.trace())
    vals, vecs = linalg.eig(rho.A)
    a, b = vals[0], vals[1]
    print("%.4f, %.4f"%(a.real, b.real))



if __name__ == "__main__":

    numpy.set_printoptions(
        precision=4, threshold=1024, suppress=True, 
        formatter={'float': '{:0.4f}'.format}, linewidth=200)

    from random import seed
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



