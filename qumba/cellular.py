#!/usr/bin/env python

"""
Build some 
2-dimensional cellular complexes, mostly by mutation.
"""


from time import time
from random import choice, randint, seed, shuffle

import numpy

from qumba.argv import argv
from qumba.distance import distance_z3, search_distance_z3
from qumba.qcode import QCode, fromstr, linear_independent, strop
from qumba.transversal import is_local_clifford_equiv
from qumba.unwrap import unwrap


class Cell(object):
    def __init__(self, dim=0, children=[]):
        self.dim = dim
        if type(children) in [list, tuple, set]:
            children = {child:1 for child in children}
        self.children = dict(children)

    def __str__(self):
        name = "Vert Edge Face".split()[self.dim]
        return "%s(%d)"%(name, len(self.children))
    __repr__ = __str__

    def __delitem__(self, child):
        del self.children[child]

    def __setitem__(self, child, r):
        assert child not in self.children
        self.children[child] = r

    def __iter__(self):
        return iter(self.children)

    def get(self, child):
        return self.children.get(child)

    def __getitem__(self, child):
        return self.children[child]

    def items(self):
        return self.children.items()

    def __contains__(self, cell):
        return cell in self.children

    def __len__(self):
        return len(self.children)

    def intersect(self, other):
        return [child for child in self if child in other]

    @property
    def src(self):
        assert self.dim == 1, "not an edge"
        assert len(self.children) == 2
        for (cell, r) in self.children.items():
            if r == -1:
                return cell
        assert 0, self.children

    @property
    def tgt(self):
        assert self.dim == 1, "not an edge"
        assert len(self.children) == 2
        for (cell, r) in self.children.items():
            if r == +1:
                return cell
        assert 0, self.children


class Complex(object):
    def __init__(self, grades=None):
        if grades is None:
            grades = [[], [], []]
        self.grades = [list(g) for g in grades]
        self.lookup = {}

    def check(self):
        grades, lookup = self.grades, self.lookup
        for (dim, grade) in enumerate(grades):
            for i,cell in enumerate(grade):
                assert lookup[cell] == i
        for (cell, i) in lookup.items():
            assert grades[cell.dim][i] is cell
        for e in self.edges:
            assert len(e) == 2
            vals = []
            for v,r in e.items():
                vals.append(r)
            vals.sort()
            assert vals == [-1, 1], e.children

    def is_built(self):
        H0 = self.bdy(0)
        H1 = self.bdy(1)
        #print(H0)
        #print(H1)
        for e in self.edges:
            vals = []
            for f in self.faces:
                if f.get(e):
                    vals.append(f[e])
            vals.sort()
            assert vals == [-1, 1], e
        return True

    @property
    def sig(self):
        return [len(g) for g in self.grades]

    def __str__(self):
        return "Complex%s"%(tuple(self.sig),)

    @property
    def verts(self):
        return self.grades[0]

    @property
    def edges(self):
        return self.grades[1]

    @property
    def faces(self):
        return self.grades[2]

    @property
    def euler(self):
        r = 0
        sgn = 1
        for grade in self.grades:
            r += sgn * len(grade)
            sgn = -sgn
        return r

    def bdy(self, dim=None):
        if dim is None:
            return [self.bdy(d) for d in range(len(self.grades)-1)]
        tgt, src = self.grades[dim:dim+2]
        lookup = self.lookup
        m, n = len(tgt), len(src)
        H = numpy.zeros((m, n), dtype=int)
        #print("bdy", dim, src)
        for j,cell in enumerate(src):
            #print(j, cell)
            for dell,r in cell.children.items():
                i = lookup[dell]
                #i = tgt.index(dell)
                #print('\t', (i, j), "<--", r)
                H[i, j] = r
        return H

    @classmethod
    def frombdy(cls, H0, H1, check=True):
        #print("frombdy")
        #print(H0)
        #print(H1)
        if check:
            HH = numpy.dot(H0, H1)
            assert numpy.alltrue(HH==0)
        cx = cls()
        m, n = H0.shape
        verts = [cx.vertex() for i in range(m)]
        edges = []
        for j in range(n):
            src, tgt = None, None
            for i in range(m):
                r = H0[i,j]
                if r == -1:
                    assert src is None
                    src = verts[i]
                if r == +1:
                    assert tgt is None
                    tgt = verts[i]
            assert src is not None
            assert tgt is not None
            edges.append(cx.edge(src, tgt))
        m, n = H1.shape
        faces = []
        for j in range(n):
            rs = {}
            for i in range(m):
                r = H1[i,j]
                if r != 0:
                    rs[edges[i]] = r
            faces.append(cx.cell(2, rs))
        return cx

    def dual(self):
        H0 = self.bdy(0)
        H1 = self.bdy(1)
        #print("dual")
        #print(H0)
        #print(H1)
        cx = Complex.frombdy(H1.transpose(), H0.transpose())
        return cx

    def is_homology(self):
        assert self.is_built()
        H0, H1 = self.bdy(0), self.bdy(1)
        H = numpy.dot(H0, H1)
        for idx in numpy.ndindex(H.shape):
            if H[idx] != 0:
                return False
        return True

    def cell(self, dim=0, children=[]):
        cell = Cell(dim, children)
        grade = self.grades[dim]
        self.lookup[cell] = len(grade)
        grade.append(cell)
        self.check()
        return cell

    def remove(self, cell):
        grades, lookup = self.grades, self.lookup
        idx = lookup[cell]
        grade = grades[cell.dim]
        assert grade.pop(idx) == cell
        del lookup[cell]
        for jdx in range(idx, len(grade)):
            cell = grade[jdx]
            lookup[cell] = jdx
        self.check()

    def face(self, children):
        cell = self.cell(2, children)
        return cell

    def edge(self, v0, v1):
        cell = self.cell(1, {v0:-1, v1:+1})
        return cell

    def vertex(self):
        cell = self.cell(0)
        return cell

    def cut_edge(self, e):
        # add new vertex bisecting edge
        assert e.dim == 1
        self.remove(e)
        v0 = e.src
        v2 = e.tgt
        v1 = self.vertex()
        e0 = self.edge(v0, v1)
        e1 = self.edge(v1, v2)
        for face in self.grades[2]:
            r = face.get(e)
            if r is None:
                continue
            del face[e]
            face[e0] = r
            face[e1] = r
        self.check()
        return e0, e1

    def cut_face(self, face):
        # add new vertex in center of face, with edge to each vertex of face
        assert face.dim == 2
        self.remove(face)
        cv = self.vertex() # center
        es = list(face) # edges
        vs = [e.tgt for e in es] # _vertices
        vs += [e.src for e in es] # _vertices
        vs = set(vs)
        assert len(es) == len(vs)
        edges = {v:self.edge(v, cv) for v in vs}
        for e,r in face.items():
            e0 = edges[e.src]
            e1 = edges[e.tgt]
            self.face({e:r, e1:r, e0:-r})
        self.check()

    def split_edge(self, edge):
        # insert bigon face at this edge
        assert edge.dim == 1
        v0, v1 = edge.src, edge.tgt
        faces = [face for face in self.faces if edge in face]
        assert len(faces) == 2
        #print("split_edge", [face[edge] for face in faces])
        f0, f1 = faces
        if f0[edge] == -1:
            f0, f1 = f1, f0
        assert f0[edge] == 1
        assert f1[edge] == -1
        del f1[edge]
        e0, e1 = edge, self.edge(v0, v1)
        f2 = self.face({e0:-1, e1:1})
        f1[e1] = -1
        self.check()

    def barycenter(self, face):
        assert face.dim == 2
        for e in list(face):
            e0, e1 = self.cut_edge(e)
        self.cut_face(face)

    def bevel(self, v0):
        assert v0.dim == 0
        faces = []
        for face in self.faces:
          for e in face:
            if v0 in e:
                faces.append(face)
                break
        #print(len(faces))
        edges = []
        for e in list(self.edges):
            if v0 not in e:
                continue
            e0, e1 = self.cut_edge(e)
            if v0 in e0:
                edges.append(e0)
            elif v0 in e1:
                edges.append(e1)
            else:
                assert 0
        assert len(edges) == len(faces)
        face = {} # new face
        for f in faces:
            #print(len(f), end=" ")
            src = tgt = None
            for e in edges:
                if e not in f:
                    continue
                for v in e:
                    if v != v0:
                        break
                r = e[v]
                if f[e] == r:
                    assert tgt is None
                    tgt = v
                elif f[e] == -r:
                    assert src is None
                    src = v
                del f[e]
            assert tgt is not None
            assert src is not None
            #print(len(f), end=" ")
            e = self.edge(src, tgt)
            f[e] = 1
            #print(len(f))
            face[e] = -1
        for e in edges:
            self.remove(e)
        self.remove(v0)
        f = self.face(face)
        self.check()
        return f

    def find_bone(cx):
        "find edge with 2 degree 3 verts"
        # ARRRGGGGHHHH cache this stuff ???!?!?!?!!!
        H0 = cx.bdy(0)
        m, n = H0.shape
        A0 = numpy.abs(H0)
        v_degree = A0.sum(1)
        def degree_vertex(d):
            for i in range(m):
                if v_degree[i] == d:
                    yield i
        def incident_edge(vidx):
            for eidx in range(n):
                # for each incident edge
                if A0[vidx,eidx]:
                    yield eidx
        def incident_vert(eidx):
            for vidx in range(m):
                # for each incident vert
                if A0[vidx,eidx]:
                    yield vidx
        #print(shortstr(A0), A0.shape)
        found = set()
        for vidx in degree_vertex(3):
            for eidx in incident_edge(vidx):
                for v1 in incident_vert(eidx):
                    if v1!=vidx and v_degree[v1]==3:
                        #print("edge", eidx, vidx, v1)
                        return cx.edges[eidx]

    def get_degree(self):
        H0 = self.bdy(0)
        m, n = H0.shape
        A0 = numpy.abs(H0)
        v_degree = A0.sum(1)
        return v_degree

    def get_genons(self):
        v_degree = self.get_degree()
        return [self.verts[i] for i in range(len(self.verts)) if v_degree[i] == 3]
    
    def remove_bones(self):
        #print("remove_bones", self)
        while 1:
            edge = self.find_bone()
            if edge is None:
                break
            self.split_edge(edge) # mutates indexes



def make_ball():
    cx = Complex()
    vertex, edge, face = cx.vertex, cx.edge, cx.face
    v0 = vertex()
    v1 = vertex()
    e0 = edge(v0, v1)
    e1 = edge(v1, v0)
    f0 = face({e0, e1})
    f1 = face({e0:-1, e1:-1})
    return cx


def make_torus(rows=2, cols=2):
    cx = Complex()
    vertex, edge, face = cx.vertex, cx.edge, cx.face
    verts = {}
    for i in range(rows):
      for j in range(cols):
        verts[i, j] = vertex()
    hedge = {}
    vedge = {}
    for i in range(rows):
      for j in range(cols):
        hedge[i, j] = edge(verts[i,j], verts[i,(j+1)%cols])
        vedge[i, j] = edge(verts[i,j], verts[(i+1)%rows,j])
    for i in range(rows):
      for j in range(cols):
        face({hedge[i,j]:1, vedge[i,(j+1)%cols]:1, 
            hedge[(i+1)%rows,j]:-1, vedge[i,j]:-1})
    return cx


def make_tetrahedron():
    cx = make_ball()
    cx.cut_edge(cx.edges[0])
    cx.cut_face(cx.faces[0])
    return cx


def make_octahedron():
    cx = make_ball()
    e0, e1 = cx.edges
    cx.cut_edge(e0)
    cx.cut_edge(e1)
    f0, f1 = cx.faces
    cx.cut_face(f0)
    cx.cut_face(f1)
    return cx


def make_cube():
    cx = make_octahedron()
    cx = cx.dual()
    return cx


def test():

    cx = make_ball()
    H0 = cx.bdy(0)
    H1 = cx.bdy(1)
    HH = numpy.dot(H0, H1)
    assert numpy.alltrue(HH==0)
    assert cx.is_homology()
    assert cx.euler == 2

    e0 = cx.edges[0]
    cx.cut_edge(e0)
    assert cx.is_homology()
    assert cx.euler == 2

    f = cx.faces[0]
    cx.cut_face(f)
    assert cx.is_homology()
    assert cx.euler == 2

    f = cx.faces[0]
    cx.barycenter(f)
    assert cx.is_homology()
    assert cx.euler == 2

    cx = make_tetrahedron()
    assert cx.is_homology()
    assert cx.euler == 2
    cx.cut_face(cx.faces[0])
    assert cx.is_homology()
    assert cx.euler == 2

    cx = make_octahedron()
    assert cx.is_homology()
    assert cx.euler == 2

    cx = make_cube()
    assert cx.is_homology()
    assert cx.euler == 2
    cx.cut_face(cx.faces[0])
    assert cx.is_homology()
    assert cx.euler == 2
    
    cx = make_torus()
    assert cx.is_homology()
    assert cx.euler == 0

    d = cx.dual()
    assert d.euler == 0

    cx.cut_face(cx.faces[0])
    assert cx.is_homology()
    assert cx.euler == 0
    
    d = cx.dual()
    assert d.euler == 0

    cx = make_tetrahedron()
    for v in list(cx.verts):
        f = cx.bevel(v)
    assert cx.is_homology()
    assert cx.euler == 2, cx.euler
    assert cx.sig == [12, 18, 8]

    cx = make_cube()
    for v in list(cx.verts):
        f = cx.bevel(v)
    assert cx.is_homology()
    assert cx.euler == 2, cx.euler
    assert cx.sig == [24, 36, 14]


def get_code(cx, verbose=False):
    H0, H1 = cx.bdy()
    A = numpy.dot(H0%2, H1%2) # vert -- face
    #print(A)
    vdeg = (H0%2).sum(1)
    assert vdeg.min() >= 3
    assert vdeg.max() <= 4

    n = len(cx.verts)
    m = len(cx.faces)
    H = numpy.empty((m, n), dtype=object)
    H[:] = "."
    def swap(j0,j1):
        assert H[j0,i] != H[j1,i]
        H[j0,i], H[j1,i] = H[j1,i], H[j0,i]
    for i in range(n):
        #for j in range(m):
        #    if A[i, j]:
        #        H[j, i] = 'x'
        if vdeg[i] == 3:
            labels = list('XYZ')
            shuffle(labels)
            for j in range(m):
                if A[i,j]:
                    H[j,i] = labels.pop()
        elif vdeg[i] == 4:
            labels = choice("XXZZ XXYY ZZYY".split()) # no...
            #labels = "XXZZ"
            labels = list(labels)
            shuffle(labels)
            jdxs = [j for j in range(m) if A[i,j]]
            for j in jdxs:
                H[j,i] = labels.pop()
            j0, j1, j2, j3 = jdxs
            f0 = cx.faces[j0]
            f1 = cx.faces[j1]
            f2 = cx.faces[j2]
            f3 = cx.faces[j3]
            #print(j0, j1, j2, j3, ''.join(H[:,i].transpose()))

            # 0 is next to 1,3 or 2,3 or 1,2:
            if f0.intersect(f1) and f0.intersect(f3):
                if H[j0,i] == H[j1,i]: swap(j0,j3)
                elif H[j0,i] == H[j3,i]: swap(j0,j1)
            elif f0.intersect(f2) and f0.intersect(f3):
                if H[j0,i] == H[j2,i]: swap(j0,j3)
                elif H[j0,i] == H[j3,i]: swap(j0,j2)
            elif f0.intersect(f1) and f0.intersect(f2):
                if H[j0,i] == H[j1,i]: swap(j0,j2)
                elif H[j0,i] == H[j2,i]: swap(j0,j1)
            else:
                assert 0
            #print('       ', ''.join(H[:,i].transpose()))
        else:
            assert 0

    #print(cx)
    lookup = cx.lookup
    walls = {} # map edge -> 0,1
    for i,edge in enumerate(cx.edges):
        jdxs = [j for j,f in enumerate(cx.faces) if edge in f]
        assert len(jdxs) == 2
        idxs = lookup[edge.src], lookup[edge.tgt]
        config = ''.join([H[jdx,idx] for jdx in jdxs for idx in idxs])
        top, bot = config[:2], config[2:]
        if top in ['XX','ZZ'] or bot in ['XX','ZZ']:
            value = 0
        elif top in ['XZ','ZX'] or bot in ['XZ','ZX']:
            value = 1
        elif config in 'XYYX YXXY ZYYZ YZZY'.split():
            value = 1
        elif config in 'XYYZ YXZY ZYYX YZXY'.split():
            value = 0
        else:
            assert 0, config
        walls[edge] = value
    for jdx,face in enumerate(cx.faces):
        value = 0
        for edge in face:
            value += walls[edge]
        for idx in range(H.shape[1]):
            # go through each vertex with valence 3
            w = (H0[idx,:]%2).sum()
            if w==3 and H[jdx, idx] == 'Y': # H[face, vert]
                # there's a twist here
                #print("*", end="")
                value += 1
        #print(value, end=' ')
        assert value%2 == 0
    #print()

    if verbose:
        print('_'*len(cx.verts))
        shortstr = lambda H : ('\n'.join(''.join(row) for row in H))
        print(shortstr(H))
        print('_'*len(cx.verts))
        print()

    H = fromstr(H)
    #print("get_code", H.shape)
    H = linear_independent(H)
    #print("get_code", H.shape)
    code = QCode(H)
    #print("get_code", code)
    return code


def mutate(cx, count=3):
    ch = cx.euler

    for _ in range(count):
        i = randint(0, 2)
        if i==0:
            v = choice(cx.verts)
            cx.bevel(v)
        #elif i==1:
        #    e = choice(cx.edges)
        #    #cx.cut_edge(e)
        elif i==1:
            f = choice(cx.faces)
            cx.barycenter(f)
        elif i==2:
            f = choice(cx.faces)
            cx.cut_face(f)
        assert cx.euler == ch
        #print(cx)
    H0, H1 = cx.bdy()
    H0 %= 2
    H1 %= 2
    vdeg = H0.sum(1)
    fdeg = H1.sum(0)
    #print(vdeg)
    #print(fdeg)
    verts = list(cx.verts)
    for i,v in enumerate(verts):
        if vdeg[i] > 4:
            cx.bevel(v)
    vdeg = (cx.bdy(0)%2).sum(1)
    assert vdeg.min() >= 3
    assert vdeg.max() <= 4
    v3 = (vdeg==3).sum()
    v4 = (vdeg==4).sum()
    assert len(cx.faces) - v3//2 - v4 == ch
    return cx


def remove_d2(code):
    # remove distance 2 logops by adding these as stabilizers
    print("remove_d2")
    while 1:
        print('\t', code.get_params())
        l = search_distance_z3(code, 2)
        if l is None or code.k==0:
            break
        print("adding logop")
        l = strop(l)
        H = strop(code.H)
        H = l + '\n' + H
        code = QCode.fromstr(H)

    return code
    

def shortstr(H):
    rows = [''.join(['%2s'%(i or '.') for i in row]) for row in H]
    return '\n'.join(rows)



def test_mutate():
    cx = make_octahedron()
    code = get_code(cx)
    assert code.k == 0

    cx = make_tetrahedron()
    code = get_code(cx)
    assert code.get_params() == (4, 1, 2)

    cx = make_cube()
    code = get_code(cx)
    assert code.get_params() == (8, 3, 2)



def main():
    #test()
    #test_mutate()
    #test_torus()

    key = argv.get("key", (5,4))
#    for idx in range(7, 13):
    for idx in range(90):
        build_geometry(key, idx)


def build_geometry(key=(5,4), idx=8):
    from bruhat.qcode import Geometry, get_adj, Group
    geometry = Geometry(key, idx)
    G = geometry.G
    print("|G| = %d, idx = %d" % (len(G), idx))
    if len(G) < 80:
        return

    G = geometry.G
    gens = G.gens
    v, e, f = gens # vert, edge, face
    H = Group.generate([f*e, f*v, e*v])
    assert len(G) % len(H) == 0
    index = len(G) // len(H)
    assert index in [1, 2]
    orientable = index==2

    H = Group.generate([v,e]) # face stabilizer
    assert len(H) == 2*key[0], len(H)
    H = Group.generate([f,e]) # vert stabilizer
    assert len(H) == 2*4
    H = Group.generate([v,e,v*e*v*e*f*e*f*e])
    assert len(G)%len(H)==0
    index = len(G)//len(H)
    assert index in [1, 2]
    bicolour = index==2

    # um, poincare dual:
    faces = geometry.get_cosets([0,1,1])
    edges = geometry.get_cosets([1,0,1])
    verts = geometry.get_cosets([1,1,0])
    #print("faces=%d, edges=%d, verts=%d"%(len(faces), len(edges), len(verts)))

    chi = len(verts) - len(edges) + len(faces)
    #assert chi%2 == 0
    print("chi=%d"%chi, "g=%d"%(1-chi//2), "orientable=%s"%orientable, 
        "bicolour=%s"%bicolour, )
    A = get_adj(faces, edges)
    B = get_adj(edges, verts)

    m, n = A.shape
    for col in range(n):
        for row in range(m):
            if A[row, col]:
                A[row, col] = -1
                break

    if 0:
        print("A =")
        print(shortstr(A), A.shape)
        print("B =")
        print(shortstr(B), B.shape)
    cx = Complex.frombdy(A, B, check=False)
    #print(cx)
    vd = cx.get_degree()
    assert numpy.alltrue(vd==4), vd

    try:
        code = get_code(cx)
    except AssertionError:
        print("AssertionError", e)
        return
    code.d = distance_z3(code)
    print("code:", code)
    print()


def make_klein():
    "Klein's quartic"
    # um... fail..
    # 2 verts, 7 edges, 1 face
    # genus 3
    cx = Complex()
    v0 = cx.vertex()
    v1 = cx.vertex()
    # edge identification
    pairs = [(1,6), (2,11), (3,8), (4,13), (5,10), (7,12), (9,14)]
    pairs = [(i-1,j-1) for (i,j) in pairs]
    # see page 139 Girondo et al


def test_torus():

    for _ in range(40):
        cx = make_torus(4,4)
        cx = mutate(cx, 2)
        cx.remove_bones()
        m = len(cx.get_genons())
        assert m%2 ==0
        code = get_code(cx)
        #assert code.k==1+m//2, (code.k, m)
        print(code, "m =",m, code.k==1+m//2, )

    return

    for _ in range(10):
        cx = make_torus(3,3)
        cx = mutate(cx, 2)
        cx.remove_bones()
        m = len(cx.get_genons())
        assert m%2 ==0
        code = get_code(cx)
        assert code.k == 1 + m//2
        print(code)

    return

    for _ in range(1):
    #while 1:
        #cx = make_cube()
        print()
        cx = make_torus(3, 4)
        print("torus:", cx)
        c0 = get_code(cx)
        print("c0:", c0.get_params())

        cx = mutate(cx, 2)
        print("mutate", cx)
        cx.remove_bones()
        print("remove_bones", cx)
        c0 = get_code(cx)
        print("c0:", c0.get_params())
        print("genons:", len(cx.get_genons()))
        g = len(cx.get_genons())
        assert g%2 == 0
        assert c0.k == g//2 + 1
        if c0.k == 0:
            continue
        #code = remove_d2(c0)
        c0.d = distance_z3(c0)
        assert c0.d > 2
        print("c0:", c0.get_params())
        c2 = unwrap(c0)
        c2.d = distance_z3(c2)
        print("c2:", c2.get_params())

        continue

        if c2.k and c2.d > c0.d:
            #print("unwrap:", c2.d)
            #print(cx)
            print(c0.longstr())
            print("------>")
            print(c2.longstr())
            break

        #for trial in range(3):
        #    dode = get_code(cx)
        #    assert is_local_clifford_equiv(code, dode)
        #    print(code.is_equiv(dode))
        #print()

    return

    #while 1:
    for trial in range(100):
        cx = choice([make_torus, make_tetrahedron])()
        cx = mutate(cx)
        code = get_code(cx)
        print(code.get_params())
    
        #d = distance_z3(code)
        #print("d =", d)



if __name__ == "__main__":

    start_time = time()


    profile = argv.profile
    name = argv.next()
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
        main()

    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")

