#!/usr/bin/env python

from functools import reduce
from operator import matmul, add


import numpy
from numpy import linalg
from numpy import random
from numpy import exp, pi, cos, arccos, sin

from qumba.argv import argv
from qumba.dense import EPSILON, scalar, w4, w8, Matrix, Space


ket0 = Matrix([[1,0]]).d
ket1 = Matrix([[0,1]]).d



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



def test_513():

    # see: https://quantumcomputing.stackexchange.com/a/40280
    c = Space(1)
    H = c.H()
    S = c.S()
    X = c.X()
    Y = c.Y()
    Z = c.Z()
    I = c.I

    s = Space(3)
    CX = s.CX
    

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

    def distill(rho):
        rho_n = reduce(matmul, [rho]*5)
    
        sho = op*rho_n*op.d
        r = sho.trace()
        sho = (1/r)*sho
    
        cliff = S.d*H*S
        sho = cliff*sho*cliff.d
        return sho

    #rho = mix(eu, 0.34)
    rho = mix(eu, 0.1)
    #print("fidelity:", fidelity(rho, mix(eu,0)))
    def err(rho):
        return (1 - eu.d*rho*eu).real

    sho = distill(rho)
    #print(err(rho))
    #print(err(sho))
    #print( fidelity(sho, mix(eu,0)) )

    def to_rho(x, y, z):
        rho = 0.5*(I + x*X + y*Y + z*Z)
        return rho

    from huygens.pov import Mat
    def coords(rho):
        x = (rho*X).trace().real
        y = (rho*Y).trace().real
        z = (rho*Z).trace().real
        v = Mat([x, y, z])
        return v

    def conv(a,b,r=0.5):
        return (1-r)*a + r*b

    def metric(p, q):
        return sum( (pi-qi)**2 for (pi,qi) in zip(p,q))**0.5

    def normalize(x,y,z):
        r = (x*x+y*y+z*z)**(1/2)
        x, y, z = (x/r,y/r,z/r)
        return x,y,z

    #tgt = normalize(0.3,-0.5,-0.1)
    tgt = normalize(*random.normal(0,1,3))
    #pts = [tgt]
    #cvs = render(pts)
    #cvs.writePDFfile("test_plot.pdf")
    #return

    def rnd():
        x,y,z = random.normal(0, 1, 3)
        x, y, z = normalize(x,y,z)
        return (x,y,z)

    def send(x,y,z):
        x,y,z = normalize(x,y,z)
        rho = to_rho(x,y,z)
        sho = distill(rho)
        x,y,z = coords(sho)
        return x,y,z

    print(send(+1,-1,+1))
    return

    def diff(x0,y0,z0,d=0.1):
        vs = []
        x,y,z = send(x0,y0,z0)
        for (x1,y1,z1) in [
            (x0+d,y0,z0),
            (x0,y0+d,z0),
            (x0,y0,z0+d),
        ]:
            x2,y2,z2 = send(x1,y1,z1)
            vs.append([x2-x, y2-y, z2-z])
        vs = numpy.array(vs)
        return (vs*vs).sum()

    #x,y,z = normalize(1,1,1)
    #print(diff(x,y,z))

    pts = []
    while len(pts) < 100:
        x,y,z = rnd()
        r = diff(x,y,z)
        if r<0.001:
            pts.append((x,y,z))
            print(".",end='',flush=True)
    print()
    
    cvs = render(pts)
    cvs.writePDFfile("test_plot.pdf")

    return

    def plot_fiber(tgt, epsilon=0.1, N=100):
        target = normalize(*tgt)
        pts = []
        #sign = lambda x : [-1,+1][int(x>EPSILON)]
        def sign(x):
            if x< -0.3:
                return -1
            if x> 0.3:
                return +1
            return 0
        found = set()
        while len(pts) < N:
            x,y,z = random.normal(0, 1, 3)
            x, y, z = normalize(x,y,z)
            rho = to_rho(x,y,z)
            sho = distill(rho)
            x1,y1,z1 = coords(sho)
            #if metric( (x,y,z), (x1,y1,z1) ) < 0.05:
            if metric( tgt, (x1,y1,z1) ) < epsilon and (x<0 or y<0 or z<0):
                pts.append( (x,y,z) )
                found.add( (sign(x), sign(y), sign(z)) )
                print(".",end='', flush=True)
                #print((x,y,z), (sign(x), sign(y), sign(z)) )
        print()
        # ugh, too stupid 
        #from scipy.cluster.hierarchy import fclusterdata
        #X = numpy.array(pts)
        #print(X.shape)
        #T = fclusterdata(X, t=2.0, depth=3, method="average")
        #print(T, T.min(), T.max())
        return pts

    #tgt = (0,0,1)
    tgt = normalize(+1,+1,-1)
    tgt = normalize(1,1,1)
    pts = plot_fiber(tgt, 0.1, 10)
    cvs = render(pts)
    cvs.writePDFfile("test_plot.pdf")
    return

    if 0:
        rhos = []
        N = 17
        #for i in range(N+1):
        #  for j in range(N-i+1):
        for i in range(1,N):
          for j in range(1,N-i):
            if i>1 and j!=1 and j!=N-i-1:
                continue
            x = (i/N)
            y = j/N
            z = 1-x-y
            #print((x,y,z))
            x = +conv(x, 1, 0.2)
            y = +conv(y, 1, 0.2)
            z = +conv(z, 1, 0.2)
            r = (x*x+y*y+z*z)**(1/2)
            rhos.append(to_rho(x/r,y/r,z/r))
        rhos = [distill(rho) for rho in rhos]
        pts = [coords(rho) for rho in rhos]
        cvs = render(pts)
        cvs.writePDFfile("test_plot.pdf")

    rhos = []
    N = 11
    #for i in range(N+1):
    #  for j in range(N-i+1):
    for i in range(1,N):
      for j in range(1,N-i):
        x = (i-N/2)/N
        y = j/N
        z = 1-y
        #print((x,y,z))
        x = +conv(x, 0, 0.7)
        y = +conv(y, 0.7, 0.7)
        z = +conv(z, 0.5, 0.7)
        r = (x*x+y*y+z*z)**(1/2)
        rhos.append(to_rho(x/r,y/r,z/r))

    N = 90
    rhos = []
    
    for i in range(N):
        y = 0.413
        theta = 2*pi*i/N
        x,z = sin(theta), cos(theta)
        x,y,z = normalize(x,y,z)
        rhos.append(to_rho(x,y,z))

    #rhos = [distill(rho) for rho in rhos]
    pts = [coords(rho) for rho in rhos]
    cvs = render(pts, connect=True)
    cvs.writePDFfile("test_plot.pdf")

    return


    x = y = z = 0.4
    rho = to_rho(x,y,z)

    #r = (to_rho(1,0,0))
    #print(r, r.trace(), (r*r).trace())
    #print(coords(r))

    #x,y,z = coords(rho)
    #print(x*x+y*y+z*z)
    rhos = [rho]
    for i in range(5):
        sho = distill(rhos[-1])
        rhos.append(sho)
    pts = [coords(rho) for rho in rhos]
    cvs = render(pts, connect=True)
    cvs.writePDFfile("test_plot.pdf")


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


def render(pts=[], colors=None, connect=False):
    from huygens.namespace import (st_thick, orange, st_round, st_arrow, 
        Canvas, st_west, st_south, st_north, color, black, grey, green, blue)
    from huygens.pov import View, Mat
    from huygens import config
    config(text="pdflatex")

    from bruhat.platonic import make_octahedron

    polytope = make_octahedron()
    polytope = [[Mat(list(v)) for v in face] for face in polytope]


    view = View(sort_gitems=True)
    view.perspective()

    #eye = Mat([1.1, 0.4, 1.6])
    eye = Mat([0.5, 0.4, 1.4])
    eye = eye / eye.norm()
    eye = 8*eye
    center = Mat([0., 0, 0])
    up = Mat([0, 1, 0])
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

    R *= 1.03
    view.add_cvs(R*vx, Canvas().text(0, 0, r"$x$", st_west))
    view.add_cvs(R*vy, Canvas().text(0, 0, r"$y$", st_south))
    view.add_cvs(R*vz, Canvas().text(0, 0, r"$z$", st_north))
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



def test_plot():
    pts = [(0.5,0,0),]
    cvs = render(pts)

    cvs.writePDFfile("test_plot.pdf")




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



