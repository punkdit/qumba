#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle, randint, choice, seed
from operator import add, matmul, mul
from functools import reduce

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)
from qumba.qcode import QCode, SymplecticSpace, strop, Matrix
from qumba.csscode import CSSCode
from qumba import csscode, construct
from qumba.construct import get_422, get_513, golay, get_10_2_3, reed_muller
from qumba.action import mulclose, mulclose_hom
from qumba.symplectic import Building
from qumba.smap import SMap
from qumba.argv import argv

from huygens.zx import Spider, Circuit, Canvas, path, st_southwest, Relation, Black, White, black, white


def latex(A):
    rows = ['&'.join(str(i) for i in row) for row in A]
    rows = r'\\'.join(rows)
    rows = r"\begin{bmatrix}%s\end{bmatrix}"%rows
    rows = rows.replace("0",".")
    #return "$%s$"%rows
    return rows


def test():
    s1 = SymplecticSpace(1)
    s2 = SymplecticSpace(2)
    gen = [s2.get_S(0), s2.get_S(1), s2.get_H(0), s2.get_H(1), s2.get_CNOT(0,1)]
    G = mulclose(gen)
    print("|G| =", len(G))
    G = list(G)
    G.sort( key = lambda g : (str(g).count('1'), str(g)))
    #G = G[:8+16]
    G = G[100:110]
    #for g in G:
    #    print(g)
    #    print(g.latex())
    #return

    s3 = SymplecticSpace(3)
    s4 = SymplecticSpace(4)
    assert s2.get_CNOT()*s2.get_CZ() == s2.get_CZ()*s2.get_CNOT()

    G = [
        s2.get_CZ(),
        #s2.get_CZ()*s2.get_S(0),
        s2.get_CNOT(), 
        #s2.get_CNOT(1,0), 
        s2.get_CNOT()*s2.get_H(0),
        s2.get_CNOT()*s2.get_H(1),
        s2.get_CNOT()*s2.get_S(0),
        s2.get_CNOT()*s2.get_S(1),
        s2.get_CNOT(1,0)*s2.get_CNOT(0,1),
        s2.get_CZ()*s2.get_P(1,0),
        s2.get_H(0)*s2.get_CZ(),
        s2.get_CZ()*s2.get_CNOT(0,1),
        s2.get_CNOT()*s2.get_H(0)*s2.get_CZ(),

        s3.get_CNOT(2,1) * s3.get_CNOT(0,1),
        s3.get_CNOT(1,2) * s3.get_CNOT(1,0),
    ]
    G = [
        s2.get_CNOT(),
        s2.get_CNOT(1,0),
        s2.get_CNOT(0,1)*s2.get_CNOT(1,0),
        #s2.get_CNOT(0,1)*s2.get_H(1)*s2.get_CNOT(0,1),
        #s2.get_CNOT(0,1)*s2.get_H(0)*s2.get_CNOT(0,1),
        #s2.get_H(0)*s2.get_CNOT(0,1)*s2.get_H(0)*s2.get_CNOT(0,1),
        #s2.get_H(1)*s2.get_CNOT(0,1)*s2.get_H(1)*s2.get_CNOT(0,1),
        s3.get_CNOT(0,1)*s3.get_CNOT(0,2),
        s3.get_CNOT(0,1)*s3.get_CNOT(1,2),
        s3.get_CNOT(1,2)*s3.get_CNOT(0,1),
        s3.get_CNOT(0,1)*s3.get_CNOT(1,2)*s3.get_CNOT(2,0),
        s3.get_CNOT(0,1)*s3.get_CNOT(2,1),
        s3.get_CNOT(0,1)*s3.get_CNOT(0,2),
        s3.get_CNOT(1,2)*s3.get_CNOT(0,2),
        s4.get_CNOT(1,3)*s4.get_CNOT(3,0)*s4.get_CNOT(2,0)*s4.get_CNOT(2,1) ,
        s4.get_CNOT(2,0)*s4.get_CNOT(2,1)*s4.get_CNOT(2,3),
    ]
    S, H = s1.get_S(), s1.get_H()
    _G = [
        S, H, S*H, H*S, H*S*H,
        s2.get_CNOT(),
        s2.get_S(0) * s2.get_CNOT(),
        s2.get_S(1) * s2.get_CNOT(),
        s2.get_H(0) * s2.get_CNOT(),
        s2.get_H(1) * s2.get_CNOT(),
    ]

    gen = [
#        s4.get_CNOT(i,j) for i in range(4) for j in range(4) if i!=j
        s4.get_CNOT(i,j) for i in range(4) for j in range(i+1,4)
    ]
    print(len(gen))
    #seed(1)

    for i in range(0):
        g = gen[0]
        for _ in range(43):
            h = choice(gen)
            g = h*g
        g.name = s4.get_name(g)
        G.append(g)


    radius = Spider.pip_radius
    p = path.circle(+radius, 0, radius)
    bcvs = Canvas().fill(p, [black]).stroke(p)
    p = path.circle(-radius, 0, radius)
    wcvs = Canvas().fill(p, [white]).stroke(p)

    cvs = Canvas()
    y = 0
    for g in G:

        x = 0.
        n = len(g)//2
        s = SymplecticSpace(n)
        I = Circuit(2*n).get_identity()
        I.min_width = 0.5

        H = s.get_H()
        if g.name is not None:
            fg = s.render_expr(g.name, width=n+2, height=2.)
            bb = fg.get_bound_box()
            cvs.insert(x-bb.width, y, fg)

        cvs.text(x+0.4, y+0.4, "$%s$"%g.latex(), st_southwest)
        min_width = 1.5
        right = Relation(g, bcvs, wcvs, min_width=2*min_width)
        left = Relation(H*g.transpose()*H, bcvs, wcvs, min_width=min_width)

        box = I * left * I * right * I
        fg = box.render(width=(I.min_width*3 + left.min_width + right.min_width), height=4)
        #fg = box.render()
        cvs.insert(x + n + 1, y, fg)
        x += n + 1 + fg.get_bound_box().width

        A = numpy.dot(left.A, right.A)
        #print(str(A).replace("0", "."))
        cvs.text(x, y+0.4, "$%s$"%latex(A), st_southwest)


        y -= 4.5
    cvs.writePDFfile("Sp4.pdf")



def hexagons_2():

    n = 2
    s = SymplecticSpace(n)
    F = s.F
    
    def get_paths(g):
        h = F*g.transpose()*F
        A = numpy.dot(h.A, g.A)
        return A

    gen = [s.get_CNOT(i,j) for i in range(n) for j in range(n) if i!=j]
    #gen = [s.get_SWAP(i,i+1) for i in range(n-1)]
    gen += [s.get_S(i) for i in range(n)]
    gen += [s.get_H(i) for i in range(n)]

    G = mulclose(gen)
    print(len(G))

    for g in G:
        if g[0,1]:
            print(g, g.name)
            break


def hexagons_4():

    n = 4
    s = SymplecticSpace(n)
    F = s.F
    
    def get_paths(g):
        h = F*g.transpose()*F
        A = numpy.dot(h.A, g.A)
        return A

    gs = {(i,j):s.get_CNOT(i,j) for i in range(n) for j in range(n) if i!=j}
    gen = [s.get_CNOT(i,j) for i in range(n) for j in range(n) if i!=j]

    I = s.get_identity()

    while 1:
        g = s.get_identity()
        for i in range(4):
            g = g*choice(gen)
        print(g, ''.join(g.name))
        A = get_paths(g) #- I.A
        print(str(A).replace('0', '.'))
        assert A.max() <= 3, A.max()
        print()

    if 0:
        #g = gs[0,1]*gs[1,2]*gs[2,3]*gs[3,0]
        g = gs[0,1]*gs[1,2]*gs[2,3]
        #g = gs[1,2]*gs[2,0]*gs[0,1]
    
        A = get_paths(g) #- I.A
        print(str(A).replace('0', '.'))
        assert A.max() <= 2, A.max()
        print()
    



if __name__ == "__main__":

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





