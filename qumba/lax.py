#!/usr/bin/env python

"""
see also: graph_states.py

"""

import os
from random import shuffle
from math import sin, cos, pi
from functools import cache, reduce
from operator import add

import numpy

from qumba.action import mulclose
from qumba.symplectic import SymplecticSpace
from qumba.matrix import Matrix
from qumba.qcode import QCode, fromstr, strop
from qumba.smap import SMap
from qumba import lin
from qumba.argv import argv
from qumba.util import binomial, factorial, choose
from qumba import construct


from huygens import config
config(text="pdflatex")
from huygens.namespace import Canvas, path, grey, black, red, white, st_thick





@cache
def get_bits(n, arity=2):
    bits = list(numpy.ndindex((arity,)*n))
    assert len(bits) == arity**n
    bits.sort(key = sum)
    bits = tuple(bits)
    return bits


@cache
def get_idxs(n):
    idxss = []
    for bits in get_bits(n):
        idxs = tuple(i for (i,ii) in enumerate(bits) if ii==1)
        idxss.append(idxs)
    return tuple(idxss)

@cache
def get_splits(n, arity=2):
    splits = []
    for bits in get_bits(n, arity):
        split = tuple(
            tuple(i for (i,ii) in enumerate(bits) if ii==bit)
            for bit in range(arity))
        splits.append(split)
    return tuple(splits)


class Signature:
    def __init__(self, n, counts):
        self.n = n
        self.counts = tuple(counts)
    def __eq__(self, other):
        assert self.n == other.n
        return self.counts==other.counts
    def __hash__(self):
        return hash(self.counts)
    def __str__(self):
        #return "Signature(%s)"%(self.counts,)
        return ''.join(str(i) for i in self.counts)
    __repr__ = __str__
    def __xor__(self, other):
        assert self.n == other.n
        counts = [min(a,b) for (a,b) in zip(self.counts, other.counts)]
        return Signature(self.n, counts)
    def __le__(self, other):
        assert self.n == other.n
        for (a,b) in zip(self.counts, other.counts):
            if a>b:
                return False
        return True
    def __lt__(self, other):
        return self <= other and self != other

    def render(self, size=4, r=0.08):
        #def render_func(n, f, r=0.08):
        n = self.n
        idxs = get_bits(n)
        dx = size/n
        dy = 1.4*dx
        cvs = Canvas()
        cols = [0]*(n+1)
        coords = {}
        for idx in idxs:
            row = sum(idx)
            col = cols[row] - 0.5*binomial(n,row)
            x = dx*col
            y = dy*row
            coords[idx] = (x,y)
            cols[row] += 1
        for idx in idxs:
          for jdx in idxs:
            kdx = tuple(i-j for (i,j) in zip(idx,jdx))
            if min(kdx)>=0 and sum(kdx)==1:
                x0, y0 = coords[idx]
                x1, y1 = coords[jdx]
                cvs.stroke(path.line(x0,y0,x1,y1), [grey])
        for i,idx in enumerate(idxs):
            x, y = coords[idx]
            for j in range(self.counts[i]):
                cvs.stroke(path.circle(x,y,(j+1)*r), [black])
        return cvs
    




class Functor:
    def __init__(self, H):
        assert isinstance(H, Matrix)
        H = H.normal_form()
        m, nn = H.shape
        assert nn%2 == 0
        n = nn//2
        self.H = H
        #print("__init__")
        #print(H, H.shape, m, n)
        self.H2 = H.reshape(m, n, 2)
        self.m = m
        self.n = n
        self.nn = nn
        self.shape = m, nn

    def __eq__(self, other):
        return self.H == other.H

    def __hash__(self):
        return hash(self.H)

    def __str__(self):
        return (str(self.H) or "[]") + " " + str(self.shape)

    def __add__(self, other):
        H = self.H.concatenate(other.H)
        return Functor(H)

    def __le__(self, other):
        H = self.H
        J = other.H
        U = J.t.solve(H.t)
        #print("__le__")
        #print(H)
        #print("-"*self.nn)
        #print(J)
        #print("?")
        return U is not None

    def signature(self, strict=False):
        counts = []
        lookup = {}
        for idxs in get_idxs(self.n):
            F = self.get(idxs)
            counts.append(F.m)
            lookup[idxs] = F.m
        if not strict:
            return Signature(self.n, counts)

        counts = []
        for idx in get_idxs(self.n):
            c = lookup[idx]
            prev = 0
            for jdx in get_idxs(self.n):
                if jdx==idx or jdx==() or not set(jdx).issubset(idx):
                    continue
                kdx = tuple(i for i in idx if i not in jdx)
                assert len(kdx) + len(jdx) == len(idx)
                result = lookup[jdx]+lookup[kdx]
                prev = max(prev, result)
                #if result == c:
                #    print(idx, "=", jdx, "+", kdx, ":", c)
            assert c>=prev
            counts.append(c-prev)

        return Signature(self.n, counts)



class Upper(Functor):
    def get(self, idxs):
        #H2 = self.H2[:, idxs, :]
        #H = H2.reshape(self.m, 2*self.n)
        mask = lin.zeros2(1,self.nn)
        mask[0,[2*i for i in idxs]] = 1
        mask[0,[2*i+1 for i in idxs]] = 1
        #print(mask)
        A = self.H.A * mask
        H = Matrix(A)
        return self.__class__(H)

class Lower(Functor):
    def get(self, idxs):
        A = lin.zeros2(2*len(idxs), self.nn)
        I = lin.array2([[1,0],[0,1]])
        for i,ii in enumerate(idxs):
            A[2*i:2*i+2, 2*ii:2*ii+2] = I
        A = lin.intersect(self.H.A, A)
        H = Matrix(A)
        return self.__class__(H)


def lc_orbits(n, items, perms=False):
    space = SymplecticSpace(n)
    gens = []
    if perms:
        gens = [space.SWAP(i,i+1).t for i in range(n-1)]
    for i in range(n):
        gens.append(space.S(i))
        gens.append(space.H(i))
    remain = set(items)
    found = set()
    orbits = []
    while remain:
        H = remain.pop()
        orbit = [H]
        found.add(H)
        bdy = list(orbit)
        while bdy:
            _bdy = []
            for H in bdy:
              for g in gens:
                J = (H*g).normal_form()
                if J in found:
                    continue
                _bdy.append(J)
                found.add(J)
                remain.remove(J)
                orbit.append(J)
            bdy = _bdy
        orbit.sort(key = str)
        orbits.append(orbit)
    return orbits



def upjump(code):
    if isinstance(code, Matrix):
        H = code
        code = QCode(H)
    if code.L is None:
        code.build()
    H = code.H
    L = code.L
    #K = H.kernel()
    #print(K, K.shape)
    for v in L.rowspan():
        if v.sum() == 0:
            continue
        H1 = H.concatenate(v)
        dode = QCode(H1)
        yield dode


class Graph:
    def __init__(self):
        self.rows = []
        self.items = set()
        self.links = set()

    def add(self, i, item):
        assert item not in self.items
        self.items.add(item)
        rows = self.rows
        while i >= len(rows):
            rows.append([])
        rows[i].append(item)

    def link(self, item, jtem):
        assert item in self.items
        assert jtem in self.items
        self.links.add((item, jtem))

    def get_crossings(self):
        links = self.links
        lookup = {}
        for row in self.rows:
          for (i,H) in enumerate(row):
            lookup[H] = i
        links = list(links)
        count = 0
        for link,mink in choose(links, 2):
            i0, j0 = lookup[link[0]], lookup[link[1]]
            i1, j1 = lookup[mink[0]], lookup[mink[1]]
            if i0==i1 or j0==j1:
                continue
            if (i0<i1) != (j0<j1):
                count += 1
        return count

    def swap(self, i, j0, j1):
        row = self.rows[i]
        assert j0!=j1
        row[j0], row[j1] = row[j1], row[j0]

    def optimize(self, idx=0):
        rows = self.rows
        row = rows[idx]
        n = len(row)
        best = self.get_crossings()
        print("optimize(%d)"%idx, best, end=" ", flush=True)
        done = False
        found = False
        while not done:
          done = True
          for i in range(n-1):
            self.swap(idx, i, i+1)
            c = self.get_crossings() 
            if c >= best:
                self.swap(idx, i, i+1)
            else:
                best = c
                print(best, end=" ", flush=True)
                done = False
                found = True
        print()
        return found

    def render(self, name, width=25, height=4):

        rows = self.rows
        links = self.links
    
        radius = None
        front = Canvas()
        y = 0
        xy = {}
        for row in rows:
          x = 0
          dx = width / len(row)
          if len(row)%2:
            x += 0.5*dx
          for i,H in enumerate(row):
            sig = Lower(H).signature()
            code = QCode(H)
            d = code.d
            print(sig, code, d)
            fg = sig.render(1.4)
            bb = fg.get_bound_box()
            if radius is None:
                radius = max(bb.height, bb.width) * 0.6
            cx, cy = bb.center
            fg = Canvas().translate(-cx, -cy).append(fg)
            if d and d>1:
                fg.text(0, -0, str(d))
            p = path.circle(x, y, radius)
            front.fill(p, [white])
            front.stroke(p, [black]+st_thick)
            front.insert(x, y, fg)
            xy[H] = (x,y)
            x += dx
    
          y -= height
    
        cvs = Canvas()
        for (H,J) in links:
            x0, y0 = xy[H]
            x1, y1 = xy[J]
            x2 = (x0+x1)/2
            y2 = (y0+y1)/2
            p = path.curve(x0, y0, x0, y2, x1, y2, x1, y1)
            cvs.stroke(p, [grey]+st_thick)
        cvs.append(front)
    
        print(name)
        cvs.writePDFfile(name)


            


def main():

    n = argv.get("n", 3)
    perm = True

    levels = [
        {code.H.normal_form() for code in construct.all_codes(n,k)}
        for k in range(n)
    ]

    lookup = {}
    graph = Graph()

    lcs = []
    for i,level in enumerate(levels):
        lc = lc_orbits(n, level, perm)
        lc.sort(key = len)
        print([len(o) for o in lc])
        for orbit in lc:
            for v in orbit:
                lookup[v] = orbit
            H = orbit[0]
            graph.add(i, H)
        lcs.append(lc)


    for lc in lcs[1:]:
      for idx1,orbit in enumerate(lc):
        H = orbit[0]
        code = QCode(H)
        for dode in upjump(code):
            orbit = lookup[dode.H.normal_form()]
            J = orbit[0]
            graph.link(H, J)

    found = True
    while found:
        found = False
        for k in range(n):
            found = found or graph.optimize(k)

    if perm:
        if n<=3:
            width = 15
            height = 4
        elif n==4:
            width = 60
            height = 6
        elif n==5:
            width = 150
            height = 12
        #graph.render("jump_perm_strict_%d.pdf"%n, width, height)
        graph.render("jump_perm_%d.pdf"%n, width, height)

    else:
        if n==3:
            width = 25
            height = 4
        elif n==4:
            width = 140
            height = 10
    
        graph.render("jump_%d.pdf"%n, width, height)



def render_hecke_dot():

    n = 4

    v0 = {code.H.normal_form() for code in construct.all_codes(n,0)}
    v1 = {code.H.normal_form() for code in construct.all_codes(n,1)}

    lookup = {}

    orbit0 = lc_orbits(n, v0)
    orbit0.sort(key = len)
    print([len(o) for o in orbit0])
    for orbit in orbit0:
        for v in orbit:
            lookup[v] = orbit

    orbit1 = lc_orbits(n, v1)
    orbit1.sort(key = len)
    print([len(o) for o in orbit1])
    for orbit in orbit1:
        for v in orbit:
            lookup[v] = orbit
    assert len(lookup) == len(v0)+len(v1)

    links = set()
    for idx1,orbit in enumerate(orbit1):
        H = orbit[0]
        code = QCode(H)
        for dode in upjump(code):
            o = lookup[dode.H.normal_form()]
            idx0 = orbit0.index(o)
            links.add((idx1,idx0))
    print(links)

    # --------------- render ------------------

    dot = open("hecke.dot", "w")
    print("""
    graph {
       node [
           shape = circle
           label = ""
       ]
       edge [
           penwidth = 2.0
       ]
    """, file=dot)

    cvs = Canvas()
    x = 0
    y = 0
    xy1 = {}
    for i,orbit in enumerate(orbit1):
        H = orbit[0]
        sig = Lower(H).signature()
        print(sig)
        fg = sig.render(1.)
        name = "hecke/%s.svg"%sig
        fg.writeSVGfile(name)
        print('    c%d [image="%s"];'%(i,name), file=dot)
        cvs.insert(x, y, fg)
        xy1[i] = (x,y)
        x += 2

    x = 0
    y += 4
    xy0 = {}
    for i,orbit in enumerate(orbit0):
        H = orbit[0]
        sig = Lower(H).signature()
        print(sig)
        fg = sig.render(1.)
        name = "hecke/%s.svg"%sig
        fg.writeSVGfile(name)
        print('    s%d [image="%s"];'%(i,name), file=dot)
        cvs.insert(x, y, fg)
        xy0[i] = (x,y)
        x += 2
    #fg.writeSVGfile("tmp.svg")
    #print(open("tmp.svg").read())

    for (i1,i0) in links:
        x0, y0 = xy0[i0]
        x1, y1 = xy1[i1]
        y1 += 2
        cvs.stroke(path.line(x0,y0,x1,y1), [red])

    #print("hecke_%d.pdf"%n)
    #cvs.writePDFfile("hecke_%d.pdf"%n)

    for (i1,i0) in links:
        print("  c%d -- s%d;"%(i1,i0), file=dot)
    print("}", file=dot)
    dot.close()

    rval = os.system("dot hecke.dot -Tpdf > hecke_%d.pdf"%n)
    assert rval == 0




def test_normal():

    n = 4

    v0 = {code.H.normal_form() for code in construct.all_codes(n,0)}
    v1 = {code.H.normal_form() for code in construct.all_codes(n,1)}

    lookup = {}

    orbit0 = lc_orbits(n, v0)
    print([len(o) for o in orbit0])
    for orbit in orbit0:
        for v in orbit:
            lookup[v] = orbit

    orbit1 = lc_orbits(n, v1)
    print([len(o) for o in orbit1])
    for orbit in orbit1:
        for v in orbit:
            lookup[v] = orbit
    assert len(lookup) == len(v0)+len(v1)

    cls = Lower
    cls = Upper

    sig0 = set()
    for vs in orbit0:
        v = vs[0]
        sig = cls(v).signature()
        sig0.add(sig)
        #print(v, sig)
    assert (len(sig0) == len(orbit0))
    sig0 = list(sig0)
    sig0.sort(key=str)
    
    sig1 = set()
    for vs in orbit1:
        v = vs[0]
        sig = cls(v).signature()
        sig1.add(sig)
    assert (len(sig1) == len(orbit1))
    sig1 = list(sig1)
    sig1.sort(key=str)

    for a in sig0:
        assert a^a == a
        #for b in sig0:
        #    print( int(a^b in sig1), end=' ')
        #print()
        c = (len([b for b in sig1 if b<=a]))
        bits = [int(b<=a) for b in sig1]
        print(a, ''.join('.1'[i] for i in bits), c)

    print()
    print()
    for a in sig1:
        c = (len([b for b in sig0 if a<=b]))
        bits = [int(a<=b) for b in sig0]
        print(a, ''.join('.1'[i] for i in bits), c)



    return

    edges = {v:[] for v in v0}

    for v in v0:
        for w in v0:
            u = v.intersect(w)
            if len(u) == n-1:
                edges[v].append(w)
                edges[w].append(v)
        print(".", flush=True, end="")
    print()
    print([len(v) for (k,v) in edges.items()][:20])
    


def test_lax():

    # FIX ME

    code = construct.get_713()
    L = Lower(code.H)
    U = Upper(code.H)

    print(L)

    print()
    print(L.get([1,2,3,6]))

    print()
    print(U.get([1,2,3,6]))

    l = U.get([1,2,3,6])
    r = U.get([0,4,5])
    #print( (r+l) <= F )
    #print( F <= (r+l) ) # FIX ME
    print( r+l )

    # ------------------------------------

    code = construct.get_512()
    L = Lower(code.H)
    U = Upper(code.H)

    for (idxs, jdxs) in get_splits(F.n):
        # upper is colax monoidal
        l = U.get(idxs)
        r = U.get(jdxs)
        rl = r+l
        #print(int(rl <= F), end='')
        #print(rl.m-F.m, end=':')
        print(rl.m, end='+')
        #assert F <= rl # FIX ME

        # lower is lax monoidal
        l = L.get(idxs)
        r = L.get(jdxs)
        rl = r+l
        print(rl.m, end=' ')
        #assert rl <= F
    print()

    inclusions = []
    for idxs in get_idxs(F.n):
      for jdxs in get_idxs(F.n):
        for i in idxs:
            if i not in jdxs:
                break
        else:
            inclusions.append((idxs, jdxs))

    for idxs, jdxs in inclusions:
        assert F.lower(idxs) <= F.lower(jdxs) # OK

    #for idxs, jdxs in inclusions:
    #    if not F.upper(idxs) <= F.upper(jdxs): # wrong category

    for (idxs, jdxs, kdxs) in get_splits(F.n, 3):
        assert F.upper(idxs+jdxs).upper(idxs) == F.upper(idxs)
        assert F.upper(idxs+jdxs).upper(jdxs) == F.upper(jdxs+kdxs).upper(jdxs)
        assert F.upper(kdxs) == F.upper(jdxs+kdxs).upper(kdxs)

        assert F.lower(idxs+jdxs).lower(idxs) == F.lower(idxs)
        assert F.lower(idxs+jdxs).lower(jdxs) == F.lower(jdxs+kdxs).lower(jdxs)
        assert F.lower(kdxs) == F.lower(jdxs+kdxs).lower(kdxs)



if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    print("%s()"%fn)

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))


