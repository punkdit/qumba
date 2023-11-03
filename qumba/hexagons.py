#!/usr/bin/env python

from time import time
start_time = time()
from random import shuffle, randint
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


def test():
    s2 = SymplecticSpace(2)
    gen = [s2.get_S(0), s2.get_S(1), s2.get_H(0), s2.get_H(1), s2.get_CNOT(0,1)]
    G = mulclose(gen)
    print("|G| =", len(G))
    G = list(G)
    G.sort( key = lambda g : (str(g).count('1'), str(g)))
    #G = G[:8+16]
    G = G[100:110]
    for g in G:
        print(g)
        print(g.latex())
    #return

    s3 = SymplecticSpace(3)

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
    assert s2.get_CNOT()*s2.get_CZ() == s2.get_CZ()*s2.get_CNOT()

    lhs = reduce(matmul, [Black(1,1)]*4)
    rhs = reduce(matmul, [White(1,1)]*4)

    radius = Spider.pip_radius
    p = path.circle(+radius, 0, radius)
    bcvs = Canvas().fill(p, [black]).stroke(p)
    p = path.circle(-radius, 0, radius)
    wcvs = Canvas().fill(p, [white]).stroke(p)

    cvs = Canvas()
    x, y = 0, 0
    for g in G:
        if len(g) == 4:
            IIII = Circuit(4).get_identity()
            HH = s2.get_H()
            fg = s2.render_expr(g.name, width=4., height=2.)
            bb = fg.get_bound_box()
            cvs.insert(x-bb.width, y, fg)
            cvs.text(x+0.4, y+0.4, "$%s$"%g.latex(), st_southwest)
            right = Relation(g, bcvs, wcvs)
            left = Relation(HH*g.transpose()*HH, bcvs, wcvs)
            box = IIII * left * IIII * right * IIII
            #box = lhs * box * rhs
            fg = box.render(width=5, height=4)
            #fg = box.render()
            cvs.insert(x + 3, y, fg)
            y -= 4.5
        else:
            IIII = Circuit(6).get_identity()
            HH = s3.get_H()
            fg = s3.render_expr(g.name, width=4., height=2.)
            bb = fg.get_bound_box()
            cvs.insert(x-bb.width, y, fg)
            cvs.text(x+0.4, y+0.4, "$%s$"%g.latex(), st_southwest)
            right = Relation(g, bcvs, wcvs)
            left = Relation(HH*g.transpose()*HH, bcvs, wcvs)
            box = IIII * left * IIII * right * IIII
            #box = lhs * box * rhs
            fg = box.render(width=5, height=4)
            #fg = box.render()
            cvs.insert(x + 3, y, fg)
            y -= 4.5
    cvs.writePDFfile("Sp4.pdf")




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





