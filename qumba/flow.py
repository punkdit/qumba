#!/usr/bin/env python

"""
render Pauli flows (aka Pauli webs)

https://arxiv.org/abs/2105.06244v2

"""

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul, lshift
from random import choice, shuffle

import numpy

from huygens.zx import (Canvas, Box, Diagram, 
    st_THick, st_THIck, st_THICk, st_thin, MultiDeco,
    grey, black, green, red, yellow, purple, white, path, Scale, LineWidth)

a = 0.4
st_black = [black]+st_thin
#st = st_THIck
st = [LineWidth(lw=0.2)]
st_red = [MultiDeco([red.alpha(a)]+st, st_black)]
st_green = [MultiDeco([green.alpha(a)]+st, st_black)]
st_purple = [MultiDeco([purple.alpha(a)]+st, st_black)]
st_yellow = [MultiDeco([yellow.alpha(a)]+st, st_black)]

from qumba.qcode import strop, QCode, SymplecticSpace
from qumba.syntax import Syntax
from qumba.argv import argv
from qumba.lagrel import Module, Lagrangian, w_ww


class Flow:
    def __init__(self, nleft, nright, term):
        self.nleft = nleft
        self.nright = nright
        self.term = term

    def __mul__(self, other):
        nleft = self.nleft
        nright = other.nright
        term = self.term * other.term
        return Flow(nleft, nright, term)


class Builder:
    def __init__(self, n=0):
        self.n = n

    def get_identity(self):
        pass

    def MX(self, idx):
        n = self.n
        #flow = Flow(n-1, n, 


def build_flow(n, term):
    print("build_flow", n, term)

    nn = 2*n
    mod = Module(n)

    ops = []
    A = mod.get_identity()
    for atom in reversed(term):
        name, arg, kw = atom
        meth = getattr(mod, name)
        op = meth(*arg, **kw)
        A = op*A
        mod = op.target
        ops.append((atom, op))

    print(A)

    box = term * Diagram(n)
    cvs = box.render()
    cvs.writePDFfile("test_flow.pdf")

    #return

    del term
    #syntax = Syntax()

    def get_st(vec, idx):
        a, b = vec[0, 2*idx:2*idx+2]
        st = {(0,0):st_black, (1,0):st_red, 
            (0,1):st_green, (1,1):st_yellow}[a,b]
        return st

    def get_cx(left, right, idx, jdx):
        print("get_cx", left, right, idx, jdx)
        l = left[:, 2*idx:2*idx+2]
        r = right[:, 2*idx:2*idx+2]
        #print(l, r)
        lr = l.concatenate(r, axis=1)
        #print(lr, lr.shape)
        #v = Lagrangian(l) @ Lagrangian(r)
        v = Lagrangian(lr)
        #print(v)
        v = w_ww * v
        #print(v)
        return get_st(v.left, 0)

    cvs = Canvas()
    x = y = 0
    count = 0
#    for bits in numpy.ndindex((2,)*nn):
#
#        v = numpy.array(bits).reshape(1, nn)
    for v0 in A.right.span():
        #print(v0, v0.shape)
        v = v0.reshape(1,nn)
        v = Lagrangian(v)
        #if str(v).strip() != "ZZ.|":
        #    continue

        #spec = syntax.get_identity()
        #st_wires = [st_red if bits[i] else st_black for i in range(n)]
        st_wires = [get_st(v.left, i) for i in range(n)]
        diagram = Diagram(n, st_wires)
        box = diagram.get_identity()

        print(v, v.shape)
        for atom, op in ops:
            right = v.left
            w = op*v
            left = w.left
            print(w, w.shape)
            #print(w.left, w.right)
            print(atom)
            name, arg, kw = atom
            kw = dict(kw)
            if name == "CX":
                i, j = arg
                print("\t", left, left.shape)
                kw["st_idx"] = get_st(left, i)
                kw["st_jdx"] = get_st(left, j)
                kw["st_gate"] = get_cx(left, right, i, j)
                
            abox = getattr(diagram, name)(*arg, **kw)
            box = abox * box
            diagram = box.target

            v = w

        print()

        fg = box.render(width=2, height=2)
        r = 1.9
        fg = Canvas([Scale(r), fg])
        fg = Canvas([fg])
        bb = fg.get_bound_box()
        #fg.text(0, 0, str(v0.A))
        bb = fg.get_bound_box()
        fg.stroke(path.rect(bb.llx, bb.lly, bb.width, bb.height), 
            [black.alpha(0.7)]+st_THick)
        cvs.insert(x, y, fg)
        count += 1
        if count % 4==0:
            y -= 1.4*bb.height
            x = 0
        else:
            x += 1.4*bb.width

        #if count>1:
        #    break
            

    bb = cvs.get_bound_box()
    cvs.stroke(path.rect(bb.llx-1, bb.lly-1, bb.width+2, bb.height+2), [white])

    cvs.writePDFfile("test_flow.pdf")
        



def test_render():
    Box.st_stroke = []

    syntax = Syntax()
    CX, H = syntax.CX, syntax.H
    PX, PZ = syntax.PX, syntax.PZ
    MX, MZ = syntax.MX, syntax.MZ

    n = 3
    term = MX(0) * CX(0,1) * CX(0,2)
    build_flow(n, term)
    return
    
    
    n = 4
    term = MX(n)*CX(n,3)*CX(n,2)*CX(n,1)*CX(n,0)*PX(n)
    term = (
         MX(n)*MZ(n+1)
        *CX(n,n+1)
        *CX(n+1,3)*CX(n,2)*CX(n+1,1)*CX(n,0)
        *CX(n,n+1)
        *PZ(n+1)*PX(n))

    build_flow(n, term)
    return

    flow = term * builder

    c = Diagram(n)
    box = term * c

    print(box)

    cvs = box.render()
    cvs.writePDFfile("test_render.pdf")





if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "test"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("\nfinished in %.3f seconds.\n"%(time() - start_time))



