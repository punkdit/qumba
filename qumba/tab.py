#!/usr/bin/env python

"""
Here we implement 
_phased lagrangian relations (see lagrel.py etc.)
A kind of _stabilizer tableaux.

"""

from random import choice, randint
from operator import mul, matmul, add
from functools import reduce
from functools import cache

import numpy

from qumba.argv import argv
from qumba.action import mulclose
from qumba.matrix import Matrix, pullback
from qumba.symplectic import symplectic_form
from qumba.smap import SMap
from qumba.qcode import strop
from qumba.pauli import Pauli
from qumba import pauli 

zeros = lambda a,b : Matrix.zeros((a,b))
pad = lambda n:Matrix.zeros((n,))


#def normalize(left, right, phase):
#    #print("normalize")
#    assert left.shape[0] == right.shape[0]
#    m, n = left.shape[1], right.shape[1]
#    A = left.concatenate(right, axis=1)
#    A = A.normal_form()
#    left = A[:, :m]
#    right = A[:, m:]
#    return left, right, phase
#
#
#
#
#class Tab:
#    def __init__(self, left, right=None, phase=None):
#        left = Matrix.promote(left)
#        if right is None:
#            right = Matrix.identity(left.shape[0])
#        else:
#            right = Matrix.promote(right)
#        assert left.shape[0] == right.shape[0], \
#            "%s %s"%(left.shape, right.shape)
#        if phase is None:
#            phase = Matrix.zeros((len(left),)) # yes..?
#        left, right, phase = normalize(left, right, phase)
#        rank = left.shape[0]
#        self.left = left
#        self.right = right
#        self.tgt = left.shape[1]
#        self.src = right.shape[1]
#        self.rank = rank
#        self.A = left.concatenate(right, axis=1)
#        #B = self._left.concatenate(self._right, axis=1)
#        #AB = self.A.intersect(B)
#        #assert len(AB) == rank # yes
#        self.shape = (rank, self.tgt, self.src)
#        self.phase = phase
#        self.check()
#
#    def __repr__(self):
#        left, right, phase = self.left, self.right, self.phase
#        left = str(left.A).replace("\n", "")
#        right = str(right.A).replace("\n", "")
#        phase = str(phase.A).replace("\n", "")
#        return "Tab(%s, %s, %s)"%(left, right, phase)
#
#    def __str__(self):
#        left, right, phase = self.left, self.right, self.phase
#        A = self.A
#        smap = SMap()
#        c = 1
#        smap[0,c] = strop(left)
#        w = left.shape[1] // 2
#        for i in range(left.shape[0]):
#            smap[i,c+w] = "|"
#            s = strop(A[i])
#            pi = (phase[i] + s.count("Y")) % 4
#            assert pi in [0,2]
#            smap[i,0] = "+-"[pi//2]
#        smap[0,c+w+1] = strop(right)
#        return str(smap)
#
#    @classmethod
#    def get_identity(cls, n):
#        I = Matrix.identity(2*n)
#        return cls(I, I)
#
#    def _check(self):
#        assert self.is_lagrangian()
#        A = self.A
#        phase = self.phase
#        for i in range(A.shape[0]):
#            s = strop(A[i])
#            pi = (phase[i] + s.count("Y")) % 4
#            assert pi in [0,2], pi
#
#    def check(self):
#        try:
#            self._check()
#        except:
#            print("!"*79)
#            print("Tab.check: FAIL")
#            print(repr(self))
#            raise
#
#    def is_lagrangian(self):
#        A = self.A
#        m, nn = A.shape
#        assert nn%2 == 0
#        assert nn//2 == m, A.shape
#        F = symplectic_form(m)
#        # assert isotropic
#        At = A.transpose()
#        AFA = A * F * At
#        return AFA.sum() == 0


def row_reduce(ops, jj0=None, jj1=None):
    ops = list(ops)
    if not ops:
        return ops
    m = len(ops)
    nn = 2*ops[0].n
    if jj0 is None:
        jj0 = 0
    if jj1 is None:
        jj1 = nn
    #print("="*79)
    #print("normal_form", m, jj0, jj1)
    #for op in ops:
    #    print(op)
    #print("="*79)

    i = 0
    j = jj0
    while i < m and j < jj1:
        assert i <= j-jj0

        # find nonzero in col j
        for i1 in range(i, m):
            if ops[i1].vec[j]:
                break
        else:
            # go to next col
            j += 1
            continue # <-------------- continue

        if i != i1:
            ops[i], ops[i1] = ops[i1], ops[i] # swap rows

        assert ops[i].vec[j] != 0

        for i1 in range(i+1, m):
            if ops[i1].vec[j]:
                ops[i1] = ops[i1] * ops[i]

        i += 1
        j += 1
    return ops


def normal_form(ops, jj0=None, jj1=None):
    ops = row_reduce(ops, jj0, jj1)
    if not ops:
        return ops
    m = len(ops)
    nn = 2*ops[0].n
    if jj0 is None:
        jj0 = 0
    if jj1 is None:
        jj1 = nn
    pivots = []
    j = jj0
    for i in range(m):
        while j < jj1 and ops[i].vec[j] == 0:
            j += 1
        if j==jj1:
            break
        pivots.append((i,j))
        i0 = i-1
        while i0>=0:
            r = ops[i0].vec[j]
            if r!=0:
                ops[i0] = ops[i0] * ops[i]
            i0 -= 1
        j += 1
    return ops, pivots


def unify(lops, rops, ljdx, rjdx):
    ln = lops[0].n
    assert ln-ljdx == rjdx
    lnn = 2*lops[0].n
    rnn = 2*rops[0].n
    lops, lpivots = normal_form(lops, 2*ljdx, 2*ln)
    rops, rpivots = normal_form(rops, 0, 2*rjdx)
    #print("unify")
    #print(lops, lpivots)
    #print(rops, rpivots)

#    ops = []
#    lj = 2*ljdx
#    rj = 0
#    li = 0
#    ri = 0
#    lget = lambda i,j:lops[i].vec[j]
#    rget = lambda i,j:rops[i].vec[j]
#
#    #while lj < 2*ln and rj < 2*rjdx and li < len(lops) and ri < len(rops):
#    #    l, r = lget(li, lj), rget(ri, rj)
#    #    #if l==0 and r==0:

    ops = []
    while lpivots and rpivots:
        li, lj = lpivots[0]
        ri, rj = rpivots[0]
        #print(li, lj, ri, rj)
        if lj-2*ljdx < rj:
            lpivots.pop(0) # argh, use index 
            continue
        elif rj < lj-2*ljdx:
            rpivots.pop(0) # argh, use index 
            continue
        assert lj-2*ljdx==rj
        l, r = lops[li], rops[ri]
        #print("\trows:", l, r)
        if l.vec[2*ljdx:] == r.vec[:2*rjdx]:
            vec = l.vec[:2*ljdx].concatenate(r.vec[2*rjdx:])
            phase = (l.phase + r.phase) % 4 # ?!?!?
            op = Pauli(vec, phase)
            #print("\tappend:", op)
            ops.append(op)

        lpivots.pop(0)        
        rpivots.pop(0)        

    return ops




class Tab:
    def __init__(self, ops, nleft):
        n = None
        for op in ops:
            assert isinstance(op, Pauli)
            assert n is None or n == op.n
            n = op.n
        assert len(ops) == n
        ops, pivots = normal_form(ops)
        self.ops = tuple(ops)
        self.n = n
        assert 0<=nleft<=n
        self.nleft = nleft
        self.nright = n-nleft

    def __str__(self):
        smap = SMap()
        n = self.n
        nleft = self.nleft
        c = 2
        for i,op in enumerate(self.ops):
            s = str(op).rjust(n+c)
            assert len(s) == c+n
            for j in range(c+nleft):
                smap[i,j] = s[j]
            smap[i,c+nleft] = "|"
            for j in range(c+nleft, c+n):
                smap[i,j+1] = s[j]
        return str(smap)

    def __hash__(self):
        return hash((self.ops, self.nleft))

    def __eq__(self, other):
        assert self.n == other.n
        if self.nleft != other.nleft:
            return False
        for i in range(self.n):
            if self.ops[i] != other.ops[i]:
                return False
        return True

    @classmethod
    def get_identity(cls, n):
        ops = []
        nn = n*2
        for i in range(n):
            v = [0]*nn
            v[2*i] = 1
            op = Pauli(v)
            ops.append(op@op)
            v = [0]*nn
            v[2*i+1] = 1
            op = Pauli(v)
            ops.append(op@op)
        tab = Tab(ops, n)
        return tab

    def __mul__(self, other):
        assert self.nright == other.nleft
        ops = unify(self.ops, other.ops, self.nleft, other.nleft)
        if ops is None:
            return None
        tab = Tab(ops, self.nleft)
        return tab

    def __matmul__(self, other):
        ops = []
        #for op in self.ops:
        #    ops.append(op @ Pauli.get_identity(other.n))
        #for op in other.ops:
        #    ops.append(Pauli.get_identity(self.n) @ op)
        nn0 = 2*self.n
        nn1 = 2*other.n
        l0 = 2*self.nleft
        r0 = nn0-l0
        l1 = 2*other.nleft
        r1 = nn1-l1
        for op in self.ops:
            #print(op)
            vec = op.vec
            #print(vec)
            vec = Matrix.concatenate(vec[:l0], pad(l1), vec[l0:], pad(r1))
            #print(vec)
            op = Pauli(vec, op.phase)
            #print(op)
            ops.append(op)
            
        for op in other.ops:
            #print(op)
            vec = op.vec
            #print(vec)
            vec = Matrix.concatenate(pad(l0), vec[:l1], pad(r0), vec[l1:])
            #print(vec)
            op = Pauli(vec, op.phase)
            #print(op)
            ops.append(op)
            
        return Tab(ops, (l0+l1)//2)

    def act(self, tgt):
        assert isinstance(tgt, Pauli)
        #print("\nact", tgt, tgt.vec)
        #print(self)
        assert tgt.n == self.nright
        jj0 = 2*self.nleft
        jj1 = 2*self.n
        ops, pivots = normal_form(self.ops, jj0, jj1)
        #for op in ops:
        #    print("\t", op)
        lhs = Pauli(pad(2*self.n), tgt.phase)
        vec = tgt.vec
        for (i,jj) in pivots:
            #print("lhs =", lhs)
            r = vec[jj-jj0]
            if r==0:
                continue
            lhs = lhs*ops[i]
        #print("lhs =", lhs)
        lhs = Pauli(lhs.vec[:2*self.nleft], lhs.phase)
        #print("lhs =", lhs)
        return lhs



def test():

    I, X, Y, Z = pauli.I, pauli.X, pauli.Y, pauli.Z

    assert normal_form([Z, X])[0] == [X, Z]
    assert (Y@Y)*(X@Z) == Z@X
    assert normal_form([Y@Y, X@Z])[0] == [X@Z, Z@X]
    #print( normal_form([Y@Y, Y@I])[0] )
    assert normal_form([Y@Y, Y@I])[0] == [Y@I, I@Y]

    i = Tab.get_identity(1)
    assert i == Tab([X@X, Z@Z], 1)

    h = Tab([X@Z, Z@X], 1)
    assert h != i

    x = Tab([X@X, -Z@Z], 1)
    z = Tab([-X@X, Z@Z], 1)
    assert x!=i
    assert x!=z
    assert x*x == i
    assert x*z == z*x
    xz = (x*z)
    assert xz*xz == i

    h = Tab([X@Z, Z@X], 1)
    assert h*h == i

    s = Tab([-Y@X, Z@Z], 1)
    assert s*s == z

    G = mulclose([s, h])
    assert len(G) == 24

    #from bruhat.gset import cayley
    #G = cayley(G)
    #print(G.structure_description(True)) # S4

    # ------------
    # n=2

    ii = Tab.get_identity(2)
    cx = Tab([X@X@X@I, I@X@I@X, Z@I@Z@I, Z@Z@I@Z], 2)
    assert cx*cx == ii

    assert i@i == ii

    gen = [h@i, i@h, s@i, i@s, cx]
    G = mulclose([h@i, i@h, s@i, i@s, cx])
    G = list(G)
    assert len(G) == 11520
    print(len(G))

    basis = [X@I, I@X, Z@I, I@Z, Y@I, I@Y]
    Pauli = mulclose(basis)
    Pauli = list(Pauli)
    assert len(Pauli) == 64
    N = len(Pauli)
    lookup = {g:i for (i,g) in enumerate(Pauli)}
    assert len(lookup) == N

    for trial in range(1000):
        g = choice(G)
        h = choice(G)
        for p in basis:
            lhs = g.act(h.act(p))
            rhs = (g*h).act(p)
            assert lhs == rhs

    from bruhat.gset import Group, Perm
    perms = []
    for g in gen:
        idxs = []
        for (i,p) in enumerate(Pauli):
            q = g.act(p)
            assert q in lookup
            j = lookup[q]
            idxs.append(j)
        perm = Perm(idxs)
        perms.append(perm)
    G = Group.generate(perms)
    assert len(G) == 11520
    assert G.structure_description() == "((C2 x C2 x C2 x C2) : A6) : C2"
        



def test_dense():

    from qumba.clifford import Clifford

    c2 = Clifford(2)
    II = c2.I
    XI = c2.X(0)
    IX = c2.X(1)
    ZI = c2.Z(0)
    IZ = c2.Z(1)
    wI = c2.wI()

    Pauli = mulclose([wI*wI, XI, IX, ZI, IZ])
    assert len(Pauli) == 64, len(Pauli)

    assert c2 is Clifford(2)

    SI = c2.S(0)
    IS = c2.S(1)
    HI = c2.H(0)
    IH = c2.H(1)
    CZ = c2.CZ(0, 1)

    C2 = mulclose([SI, IS, HI, IH, CZ], maxsize=None, verbose=True) # slow
    #assert len(C2) == 92160, len(C2)
    print()

    # The action of the clifford group by conjugation
    # on the Pauli group.
    # Find the Clifford _stabilizer of a (Pauli) _stabilizer state.

    state = [ZI, IZ, ZI*IZ] # |00>
    found = []
    for g in C2:
        for p in state[:2]:
            if (~g)*p*g not in state:
                break
        else:
            found.append(g)
    print("_stab:", len(found))
    assert len(found) == 1536 # 1536 == 92160 // 60








if __name__ == "__main__":

    from time import time
    start_time = time()

    _seed = argv.get("seed")
    if _seed is not None:
        from random import seed
        print("seed(%s)"%_seed)
        seed(_seed)

    profile = argv.profile
    fn = argv.next() or "test"

    print("%s()"%fn)

    if profile:
        import cProfile as profile
        profile.run("%s()"%fn)

    else:
        fn = eval(fn)
        fn()

    print("\nOK: finished in %.3f seconds"%(time() - start_time))
    print()

