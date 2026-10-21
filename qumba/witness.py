#!/usr/bin/env python

"""
Use linear relations for CSS _circuits (aka phase-free ZX-calculus).
Keep track of how we compose operators using Composite objects.
and more...

See previous: css.py 

test_encoder: works
test_simple_detectors: fail

"""

import time
from functools import reduce, cache
from operator import matmul, lshift, mul
from random import choice, randint

import numpy

#from qumba.syntax import Syntax # ugh... the API doesn't fit _very well...
from qumba.matrix import Matrix, pullback
from qumba.rel import Relation # linear relations
from qumba.csscode import CSSCode
from qumba.qcode import QCode, SymplecticSpace
from qumba import construct

from qumba.argv import argv
from qumba.util import choose
from qumba.smap import SMap


# ----------------------------------------------------------------------------
# Another attempt at a Syntax class 

class Atom:
    def __init__(self, name, *args):
        assert type(name) is str, "what's this: %r"%(name,)
        self.name = name
        self.args = args
        self.key = (name, args)
        self.atoms = [self]
    def __str__(self):
        s = "%s%s"%(self.name, self.args)
        s = s.replace(",)", ")")
        s = s.replace(" ", "")
        return s
    __repr__ = __str__
    def __mul__(self, other):
        assert isinstance(other, Atom)
        return Molecule(self.atoms + other.atoms) # algebraic order of atoms
    def __len__(self):
        return len(self.atoms)
    def __getitem__(self, idx):
        return self.atoms[idx]
# use object identity ... 
#    def __eq__(self, other):
#        assert isinstance(other, Atom)
#        return self.key == other.key
#    def __hash__(self):
#        return hash(self.key)


class Molecule(Atom):
    def __init__(self, atoms):
        self.atoms = list(atoms)
    def __str__(self):
        return "*".join(str(atom) for atom in self.atoms)
    def __eq__(self, other):
        assert 0, "um?"
        return self.atoms == other.atoms


class Method:
    def __init__(self, name):
        assert type(name) is str
        self.name = name

    def __call__(self, *arg, **kw):
        assert not kw, "um...?"
        return Atom(self.name, *arg)


class Syntax:
    def __getattr__(self, name):
        return Method(name)


# ----------------------------------------------------------------------------




class Circuit:
    def __init__(self, op, tgt, src):
        assert isinstance(op, Operator)
        self.op = op
        self.tgt = tgt
        self.src = src
        self.shape = (tgt, src)

#    @property
#    def tgt(self)

    def __mul__(self, other):
        assert self.src == other.tgt, (self.shape, other.shape)
        return Composite(self, other)

    @classmethod
    def promote(cls, item):
        if isinstance(item, Circuit):
            op = item
        elif isinstance(item, (Matrix, Relation)):
            op = Operator(item)
        else:
            assert 0, "whats this: %s"%(item.__class__,)
        return op

    def deepstr(self, depth=0):
        return " "*depth + str(self)

    @classmethod
    def from_code(cls, code):
        assert code.is_css()
        css = code.to_css()
    
        left = code.Hx.stack(code.Lx)
        right = Matrix.zeros((code.mx, code.k))
        right = right.stack(Matrix.identity(code.k))
        Ez = Relation(left, right)
    
        left = code.Hz.stack(code.Lz)
        right = Matrix.zeros((code.mz, code.k))
        right = right.stack(Matrix.identity(code.k))
        Ex = Relation(left, right)
    
        return Operator(Ex, Ez)
    
    def to_code(self):
        op = self.op
        assert isinstance(op, Operator)
        Ez, Ex = op.pair
    
        #print("from_encoder")
        #print(Ex, Ex.shape)
        assert isinstance(Ex, Relation)
        k = Ex.src
        assert Ez.src == k
    
        #if k==0:
        #    print(Ex)
    
        w = Relation.white(1,0)
        if k:
            op = reduce(matmul, [w]*k)
            #print(op, op.shape)
            Hx = (Ex*op).left
            Hz = (Ez*op).left
        else:
            Hx = Ex.left
            Hz = Ez.left
    
        mx, n = Hx.shape
        #print("Hx =")
        #print(Hx, Hx.shape)
    
        mz = len(Hz)
        assert mz+mx+k == n
        #print("Hz =")
        #print(Hz, Hz.shape)
    
        op = Relation(Matrix.identity(k), Matrix.zeros((k,0)))
        HLx = (Ex*op).left
        HHLx = Hx.stack(HLx).linear_independent()
        assert HHLx[:mx, :] == Hx
        Lx = HHLx[mx:, :]
    
        #print("Lx =")
        #print(Lx, Lx.shape)
    
        op = Relation(Matrix.identity(k), Matrix.zeros((k,0)))
        HLz = (Ez*op).left
        HHLz = Hz.stack(HLz).linear_independent()
        assert HHLz[:mz, :] == Hz
        Lz = HHLz[mz:, :]
    
        code = CSSCode(Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz)
        return code


get_encoder = Circuit.from_code
from_encoder = Circuit.to_code


class Operator(Circuit):
    "The _atomic Operator's in a Composite Circuit"
    def __init__(self, Ax, Az=None, check=True):
        if isinstance(Ax, Circuit):
            Ax = Ax.op.Ax
        if isinstance(Az, Circuit):
            Az = Az.op.Ax
        Ax = Relation.promote(Ax)
        if Az is None:
            Az = Ax.dual
        Az = Relation.promote(Az)
        self.Ax = Ax
        self.Az = Az
        self.pair = (Ax, Az)
        if check:
            assert Ax.dual == Az
        self.atom = None # my Syntax
        Circuit.__init__(self, self, Ax.tgt, Ax.src) # i am my own Operator

    @property
    #@cache need my own atom
    def reverse(self):
        Ax, Az = self.Ax, self.Az
        return Operator(Ax.reverse, Az.reverse)

    @property
    @cache
    def dual(self):
        Ax, Az = self.Ax, self.Az
        return Operator(Az, Ax)

    def copy(self):
        return Operator(self.Ax.copy(), self.Az.copy())

    def set(self, atom):
        assert type(atom) is Atom
        assert self.atom is None, "%s already got atom %r"%(self, self.atom,)
        self.atom = atom

    def walk(self):
        yield self

    def mul(self, other):
        assert isinstance(other, Operator), type(other)
        Ax = self.Ax * other.Ax
        Az = self.Az * other.Az
        return Operator(Ax, Az)

    def __matmul__(self, other):
        assert isinstance(other, Operator), type(other)
        Ax = self.Ax @ other.Ax
        Az = self.Az @ other.Az
        return Operator(Ax, Az)

    def __eq__(self, other):
        assert isinstance(other, Operator), (type(self), type(other))
        return self.pair == other.pair

    def __hash__(self):
        return hash(self.pair)

    def __str__(self):
        return "(%s | %s)"%(self.Ax.spanstr(), self.Az.spanstr())

    @classmethod
    def black(cls, l, r):
        Rx = Relation.black(l, r)
        Rz = Relation.white(l, r)
        return Operator(Rx, Rz)

    @classmethod
    def white(cls, l, r):
        Rx = Relation.white(l, r)
        Rz = Relation.black(l, r)
        return Operator(Rx, Rz)

    @classmethod
    #@cache # don't cache because i need my own atom
    def identity(cls, n):
        Rx = Relation.identity(n)
        return Operator(Rx, Rx)

    def is_identity(self):
        tgt, src = self.shape
        if tgt != src:
            return False
        return self == Operator.identity(tgt)

    @classmethod
    def zero(cls, l, r):
        Rx = Relation.zero(l, r)
        return Operator(Rx)


def witness(lhs, rhs):
    """
        compose Relation's and return the detector and witness
    """
    assert isinstance(lhs, Relation)
    assert isinstance(rhs, Relation)

    assert lhs.src == rhs.tgt, (lhs.src, rhs.tgt)
    l, r = pullback(lhs.right.t, rhs.left.t)

    left = lhs.left.t * l
    right = rhs.right.t * r

    f = left.stack(right)
    k = f.kernel().t # find the detect'ing regions

    ll = lhs.right.t * l
    rr = rhs.left.t * r
    assert ll==rr
    d = (ll*k).t
    op = Relation(left.t, right.t)
    w = None # TODO the detector is the kernel of the witness
    return d, w, op
    


class Composite(Circuit):
    def __init__(self, lhs, rhs):
        lhs = Circuit.promote(lhs)
        rhs = Circuit.promote(rhs)
        self.lhs = lhs
        self.rhs = rhs
        lop = lhs.op
        rop = rhs.op
        #op = lop.mul(rop.op)

        dz, wx, Ax = witness(lop.Ax, rop.Ax) # XXX get rid of witness call TODO
        dx, wz, Az = witness(lop.Az, rop.Az) # XXX get rid of witness call TODO
        op = Operator(Ax, Az)

        Circuit.__init__(self, op, op.tgt, op.src)

        self.dx = dx
        self.wx = wx
        self.dz = dz
        self.wz = wz

    def __str__(self):
        return "(%s*%s=%s, (%d|%d))"%(self.lhs, self.rhs, self.op, 
            len(self.dx), len(self.dz))

    def deepstr(self, depth=0):
        lhs = self.lhs.deepstr(depth+1)
        rhs = self.rhs.deepstr(depth+1)
        d = "(%d|%d)"%(len(self.dx), len(self.dz))
        lines = [self.op.deepstr(depth) + d + "=", lhs + "*", rhs]
        return '\n'.join(lines)

    def __matmul__(self, other):
        assert isinstance(other, Operator), type(other)
        lhs = self.lhs
        rhs = self.rhs
        comp = (lhs@other) * (rhs@other) # Argghh, recursive
        return comp

    @property
    @cache
    def reverse(self):
        assert 0, "todo"

    def walk(self):
        for item in self.lhs.walk():
            yield item
        for item in self.rhs.walk():
            yield item

    

class Hom:
    " Hom( tgt <--- src ) "

    _w = Operator.white(0, 1)
    w_ = Operator.white(1, 0)
    _b = Operator.black(0, 1)
    b_ = Operator.black(1, 0)
    I = Operator.identity(1)

    @cache # singleton
    def __new__(cls, m, n=None):
        ob = object.__new__(cls)
        return ob

    def __init__(self, m, n=None):
        if n is None:
            n = m
        self.m = m
        self.n = n

    @property
    def reverse(self):
        return Hom(self.n, self.m)

    def __str__(self):
        return "(%s<--%s)"%(self.m, self.n)
    __repr__ = __str__

    def __getitem__(self, i):
        # i am a shape tuple
        return (self.m, self.n)[i]

    def identity(self):
        assert self.m == self.n
        return Operator.identity(self.m)

    def RX(self, *idxs):
        "prepare X state: |+> on idxs"
        m, n = self
        assert m>n, "wrong Hom %s"%(self,)
        if not idxs:
            idxs = list(range(m-n))
        else:
            idxs = list(idxs)
        assert m == n+len(idxs), "wrong Hom %s"%(self,)
        idxs.sort(reverse=True)
        rels = [self.I]*n
        for i in idxs:
            assert 0<=i<m, (i, m)
            rels.insert(i, self.w_.copy())
        assert len(rels) == m
        rel = reduce(matmul, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n), (str(rel), str(self))
        return rel

    def RZ(self, *idxs):
        "prepare Z state: |0> on idxs"
        m, n = self
        assert m>n, "wrong Hom %s"%(self,)
        if not idxs:
            idxs = list(range(m-n))
        else:
            idxs = list(idxs)
        assert m == n+len(idxs), "wrong Hom %s"%(self,)
        idxs.sort(reverse=True)
        rels = [self.I]*n
        for i in idxs:
            assert 0<=i<m, (i, m)
            rels.insert(i, self.b_.copy())
        assert len(rels) == m
        rel = reduce(matmul, rels)
        assert (rel.tgt,rel.src) == (self.m, self.n)
        return rel

    def MX(self, *idxs):
        "postselect <+| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.reverse.RX(*idxs).reverse
        return rel

    def MZ(self, *idxs):
        "postselect <0| on idxs"
        m, n = self
        assert not idxs or m+len(idxs)==n, "wrong Hom %s"%(self,)
        rel = self.reverse.RZ(*idxs).reverse
        return rel

    def CX(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        lhs = Matrix.identity(m)
        rhs = numpy.identity(m, dtype=int)
        rhs[jdx, idx] = 1
        rhs = Matrix(rhs)
        rel = Relation(lhs, rhs)
        return Operator(rel)

    def PERM(self, idxs):
        """ send idxs[j]<---j """
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m == len(idxs), "wrong Hom %s"%(self,)
        assert len(set(idxs)) == m
        lhs = Matrix.identity(m)
        rhs = Matrix.get_perm(idxs)
        rel = Relation(lhs, rhs)
        return Operator(rel)
    get_perm = PERM

    def SWAP(self, idx=0, jdx=1):
        m, n = self
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        f = list(range(self.n))
        f[idx], f[jdx] = f[jdx], f[idx]
        rel = self.get_perm(f)
        return Operator.promote(rel)

    def random(self):
        m, n = self
        if m==n==0:
            op = self.identity()
        elif m==0 and n==1:
            op = choice([self.MX(0), self.MZ(0)])
        elif n==0 and m==1:
            op = choice([self.RX(0), self.RZ(0)])
        elif m==n and n==1:
            op = choice([
                self.identity(), 
                self.w_*self._w,
                self.w_*self._b,
                self.b_*self._b,
                self.b_*self._w])
            assert op.tgt == m, (op.tgt, op.src, self)
            assert op.src == n, (op.tgt, op.src, self)
        elif m==n>1:
            i = randint(0, n-1)
            j = randint(0, n-2)
            j = j+1 if j>=i else j
            op = self.CX(i,j)
        elif m == n-1:
            i = randint(0, n-1)
            op = choice([self.MX(i), self.MZ(i)])
        elif n == m-1:
            i = randint(0, n-1)
            op = choice([self.RX(i), self.RZ(i)])
        else:
            assert 0, (m, n)
        assert op.tgt == m, (op.tgt, op.src, self)
        assert op.src == n, (op.tgt, op.src, self)
        return op
        

class Module:
    """ A mutable context for keeping track of live qubits
        and evaluating Syntax expressions.
    """
    # is this an Actegory ?
    # another idea: We could take "hom" as a factory arg (for Hom)

    # XXX dynamically allocating qubit indexes is a pain in the butt. XXX
    # XXX just pre-allocate the "train track" indexes !               XXX

    def __init__(self, idxs=[]):
        self.alloc = {} # qubit idxs
        for i in idxs:
            self.add(i)
        #self.lookup = None

    def get_idxs(self):
        self.check()
        alloc = self.alloc
        #n = len(alloc)
        n = max(list(alloc.keys()) or [-1])+1
        idxs = [None]*n
        for i in range(n):
            if i in alloc:
                idxs[alloc[i]] = i
        idxs = [i for i in idxs if i is not None]
        return idxs

    @property
    def n(self):
        return len(self.alloc)

    def check(self):
        alloc = self.alloc
        idxs = list(alloc.values())
        idxs.sort()
        n = len(idxs)
        assert idxs == list(range(n)), idxs

    def add(self, i):
        alloc = self.alloc
        #print("add:", self.get_idxs(), alloc)
        assert i not in alloc, alloc
        n = len(alloc)
        alloc[i] = n
        #print("\tadd:", alloc)
        self.check()
        #print("\t", self.get_idxs())
        return n

    def remove(self, i):
        alloc = self.alloc
        assert i in alloc, alloc
        idx = alloc[i]
        del alloc[i]
        for key in list(alloc.keys()):
            jdx = alloc[key]
            assert jdx != idx
            if jdx > idx:
                alloc[key] = jdx-1
        self.check()

    def get_reorder(self):
        "fix up the order of alloc'ated qubit indexes"
        prev = dict(self.alloc)
        idxs = self.get_idxs()
        jdxs = list(idxs)
        jdxs.sort()
        #if idxs == jdxs:
        #    return None
        lookup = {j:i for (i,j) in enumerate(jdxs)}
        kdxs = [lookup[i] for i in idxs]
        n = len(kdxs)
        P = Hom(n, n).get_perm(kdxs)
        alloc = {i:j for (j,i) in enumerate(jdxs)}
        return P, alloc

#    def _order(self):
#        "fix up the order of alloc'ated qubit indexes"
#        prev = dict(self.alloc)
#        idxs = self.get_idxs()
#        jdxs = list(idxs)
#        jdxs.sort()
#        #if idxs == jdxs:
#        #    return None
#        lookup = {j:i for (i,j) in enumerate(jdxs)}
#        kdxs = [lookup[i] for i in idxs]
#        n = len(kdxs)
#        P = Hom(n, n).get_perm(kdxs)
#        self.alloc = {i:j for (j,i) in enumerate(jdxs)}
#        #if self.alloc != prev:
#        #    print("_order: CHANGED", prev, self.alloc)
#        #if kdxs == list(range(n)):
#        #    return None
#        return P

    def interpret(self, syntax):
        op = None
        I = Operator.identity(1)
        for atom in reversed(syntax.atoms):
            meth = getattr(self, atom.name)
            idxs = atom.args
#            for i in idxs:
#                if i not in self.alloc:
#                    j = self.add(i)
#                if op is not None:
#                    op = op @ I

            prev = dict(self.alloc)
            lhs = meth(*idxs) # may add qubits 
            assert type(lhs) is Operator, "must be atom Operator, not %r"%type(lhs)
            P, alloc = self.get_reorder()
            #print(P.is_identity())

#            print("\t", atom)
            if op is None:
                lhs = (P*lhs).op
                lhs.set(atom)
                self.alloc = alloc
                op = lhs
                continue

#            assert lhs.src >= op.tgt
#            if lhs.src > op.tgt:
#                alloc = self.alloc
#                change = [i for i in alloc if i not in prev]
#                print("tensor", prev, "-->", alloc, change)
#                
#                # XXX not sure this is always right ... ?!?
#                I = Operator.identity(lhs.src - op.tgt)
#                op = op@I
#            #else:
#            #    assert len(self.alloc) == len(prev)

            lhs = (P*lhs).op
            lhs.set(atom)
            op = lhs*op
            #op = (P*op).op
            self.alloc = alloc
        return op

    def get_identity(self):
        assert 0, "don't use __mul__, use interpret"

    def identity(self):
        return Operator.identity(self.n)

    def RX(self, i):
        alloc = self.alloc
        n = len(alloc)
        idx = self.add(i)
        op = Hom(n+1, n).RX(idx)
#        P = self._order()
#        if P is not None:
#            op = (P*op).op
        return op

    def RZ(self, i):
        alloc = self.alloc
        n = len(alloc)
        idx = self.add(i)
        op = Hom(n+1, n).RZ(idx)
        _op = Hom(n+1, n).RZ(idx)
        assert op is not _op
#        P = self._order()
#        if P is not None:
#            op = (P*op).op
        return op

    def CX(self, i, j):
        alloc = self.alloc
        if i not in alloc:
            self.add(i)
        if j not in alloc:
            self.add(j)
        n = len(alloc)
        idx = alloc[i]
        jdx = alloc[j]
        op = Hom(n, n).CX(idx, jdx)
#        P = self._order()
#        if P is not None:
#            op = (P*op).op
        return op

    def SWAP(self, i, j):
        alloc = self.alloc
        n = len(alloc)
        idx = alloc[i]
        jdx = alloc[j]
        op = Hom(n, n).SWAP(idx, jdx)
        return op

    def MX(self, i):
        alloc = self.alloc
        n = len(alloc)
        idx = alloc[i]
        self.remove(i)
        op = Hom(n-1, n).MX(idx)
        return op

    def MZ(self, i):
        alloc = self.alloc
        n = len(alloc)
        idx = alloc[i]
        self.remove(i)
        op = Hom(n-1, n).MZ(idx)
        return op

    def TICK(self):
        n = len(self.alloc)
        return Hom(n, n).identity()


def from_stim(data):

    lines = data.split("\n")
    gates = []
    rlook = {}
    mlook = {}
    for idx,line in enumerate(lines):
        flds = line.strip().split()
        gate = tuple([flds[0]] + [int(fld) for fld in flds[1:]])
        gates.append(gate)

    for idx,gate in enumerate(gates):
        if gate[0] == "R":
            j = gate[1]
            assert j not in rlook
            rlook[j] = idx
        if gate[0] == "M":
            j = gate[1]
            assert j not in mlook
            mlook[j] = idx

    print(mlook)
    for idx,gate in enumerate(gates):
        if gate[0] == "CX":
            _, i, j = gate
            if i in rlook:
                del rlook[i]
            if j in rlook:
                del rlook[j]
        if gate[0] == "H":
            _, i = gate
            if i in rlook:
                j = rlook[i]
                assert gates[j] == ("R", i)
                gates[j] = ("RX", i)
                del rlook[i]
            elif i in mlook:
                j = mlook[i]
                assert gates[j] == ("M", i)
                gates[j] = ("MX", i)
                del mlook[i]

    #print(gates)
    idx = 0
    while idx < len(gates):
        gate = gates[idx]
        op = gate[0]
        if op == "H":
            gates.pop(idx)
            continue
        if op == "R":
            gates[idx] = ("RZ", gate[1])
        if op == "M":
            gates[idx] = ("MZ", gate[1])
        idx += 1
    #print(gates)

    tick = False
    idx = 0
    while idx < len(gates):
        gate = gates[idx]
        op = gate[0]
        if op == "CX":
            tick = True
        if op[0] == "M":
            if tick:
                gates.insert(idx, ("TICK",))
            tick = False
        if tick:
            gates.insert(idx, ("TICK",))
            idx += 1
        idx += 1

    c = Syntax()
    spec = []
    for gate in gates:
        op = gate[0]
        args = gate[1:]
        s = getattr(c, op)(*args)
        spec.append(s)
    spec = list(reversed(spec))
    spec = reduce(mul, spec)
    return spec




def stack(code):
    assert code.is_css()
    assert "Y" not in code.longstr()
    #print(code.longstr())
    css = code.to_css()
    #print("stack", css)
    HLx = (css.Hx.stack(css.Lx))
    return HLx.t


def encoder(code, verbose=False):

    # TODO: clean up and optimize this mess.
    # (now we are going via QCode, to keep track of the reduced H matrix.)

    syntax = Syntax()
    space = SymplecticSpace(code.n)
    H = stack(code)
    n, m = H.shape
    assert n>=m

    # we do echelon row/col reduction form with CX gates
    spec = []
    pivots = [] # choose one row for every colum
    col = 0
    count = 0
    while col < m:
        count += 1
        assert count < m*n, "woah"
        if verbose:
            print()
            line = ['_']*m
            line[col] = 'V'
            print(''.join(line))
        H = stack(code)
        if verbose:
            print(H)
        assert H.shape == (n, m)
        for pivot in range(n):
            if pivot in pivots:
                continue
            if H[pivot, col]:
                break
        else:
            assert 0, "wup"
        assert H[pivot, col]
        idxs = list(range(n))
        for j in range(n):
            if j==pivot:
                continue
            if H[j, col]:
                item = syntax.CX(pivot, j)
                op = space.CX(pivot, j)
                if verbose:
                    print("CX(%d,%d)"%(pivot, j))
                break
        else:
            pivots.append(pivot)
            col += 1
            continue 
        spec.append(item)
        E = code.get_encoder()
        E = op * E
        code = QCode.from_encoder(E, k=code.k)

    H = stack(code)
    css = code.to_css()
    assert len(pivots) == css.mx + css.k
    for i in range(css.mx):
        idx = pivots[i]
        spec.append(syntax.RX(idx))
    src = pivots[css.mx:]
    pivots = set(pivots)
    remain = [i for i in range(n) if i not in pivots]
    assert len(remain) == css.mz
    for i in remain:
        spec.append(syntax.RZ(i))
    spec = reduce(mul, spec)

    module = Module(src)
    E = module.interpret(spec)

    return spec, E


def render(spec, name):
    from qhap.circuit import RX, RZ, CX, TICK, MX, MZ
    from qhap.layout import to_svg
    circuit = eval(str(spec))*TICK()
    flow = None
    #flow = circuit.z_flow(0,6)
    svg = to_svg(circuit, flow)
    f = open("%s.svg"%name, "w").write(svg)
    import os
    cmd = 'rsvg-convert -f pdf -o %s.pdf %s.svg'%(name, name)
    print(cmd)
    rval = os.system(cmd)
    assert rval == 0




def test_encoder():

    s = Syntax()
    MX, MZ, CX, RX, RZ = s.MX, s.MZ, s.CX, s.RX, s.RZ

    spec = RX(0)*RZ(2)
    op = Module([1]).interpret(spec)
    rhs = op.op

    # check that Module doesn't scramble the indexes
    lhs = Hom(3,2).RZ(2)*(Hom(1,0).RX(0) @ Hom(1,1).identity())
    assert eqop(lhs, rhs)

    spec = CX(0,1)*CX(1,2)*CX(2,0)
    circ = Module([0,1,2]).interpret(spec)
    lhs = circ.op

    hom = Hom(3,3)
    cx = hom.CX
    rhs = cx(0,1)*cx(1,2)*cx(2,0)
    assert eqop(lhs, rhs)

    # -------------------------------------------
    n = 5
    hom = Hom(n, n)
    syntax = []
    op = Operator.identity(n)
    for i in range(10):
        j = randint(0, n-1)
        k = randint(0, n-1)
        if j!=k:
            syntax.append(CX(j,k))
            op = op*hom.CX(j,k)
    syntax = reduce(mul, syntax)
    print(syntax)
    comp = Module(list(range(n))).interpret(syntax)
    assert comp.op == op.op

    module = Module(list(range(n)))
    comp = module.interpret(syntax) # TODO: check we touch every qubit
#    P = module._order()
#    comp = (P*comp).op
#    assert comp.op == op.op
#    return
    print(comp.op == op.op)

    # -------------------------------------------

    spec = [CX(0,3), CX(0,5), CX(0,6), CX(1,3), CX(1,4),
    CX(1,5), CX(2,4), CX(2,5), CX(2,6), CX(3,0), CX(3,1),
    CX(3,4), CX(3,6), RX(0), RX(1), RX(2), RZ(4), RZ(5), RZ(6)]
    spec = CX(0,3)*CX(0,5)*CX(0,6)*CX(1,3)*CX(1,4)*CX(1,5)*CX(2,4)\
        *CX(2,5)*CX(2,6)*CX(3,0)*CX(3,1)*CX(3,4)*CX(3,6)\
        *RX(0)*RX(1)*RX(2)*RZ(4)*RZ(5)*RZ(6)

    op = Module([3]).interpret(spec)
    E = op.op
    code = (from_encoder(E))
    assert code.is_equiv(construct.get_713()) 

    # -------------------------------------------

    op = Hom(1,1).identity()
    n = 1
    for i in [0,1,2]:
        op = Hom(n+1,n).RX(i)*op
        n += 1
    for i in [4,5,6]:
        op = Hom(n+1,n).RZ(i)*op
        n += 1
    hom = Hom(n,n)
    for (i,j) in reversed([(0,3), (0,5), (0,6), (1,3), (1,4),
        (1,5), (2,4), (2,5), (2,6), (3,0), (3,1), (3,4), (3,6)]):
        op = hom.CX(i,j)*op

    E = op.op
    code = (from_encoder(E))
    code.bz_distance()
    #print(code)
    assert code.is_equiv(construct.get_713())

    # -------------------------------------------

    for code in [
        construct.get_422(),
        construct.get_713(),
        construct.get_10_2_3(),
        #construct.get_colour_488(5),
        #construct.get_bring(),
        CSSCode.random(7, 2, 4, distance=2),
        CSSCode.random(11, 4, 4, distance=2),
    ]:

        print()
        #code = code.to_css()
        #code.bz_distance()
        print(code)
        spec, E = encoder(code, verbose=False)
        print(spec)
        dode = from_encoder(E)
        dode.bz_distance()
        print(dode)
        assert (dode.is_equiv(code))
        #assert dode.d == code.d

    #print(code.longstr())
    #render(spec, "prep_%d_%d_%d"%(code.n, code.k, code.d))

    # -------------------------------------------



def find_detectors(syntax, src=[], dual=False, verbose=False):
    if verbose:
        print("find_detectors(%s)"%syntax, "z_flow" if dual else "x_flow")
    module = Module(src)
    #comp = syntax*Module(src)
    comp = module.interpret(syntax)
    assert isinstance(comp, Circuit)
    ops = list(comp.walk())
    #print(len(ops))
    for op in ops:
        #print(op, op.atom)
        assert op.atom is not None
        assert op.atom in syntax # XXX remove (slow)
        #print(module.lookup.get(id(op)))
    ops = [op for op in ops if not op.is_identity()] # get rid of the TICK's
    #print(len(ops))
    ops = list(reversed(ops)) # <------ backwards <----
    # edges: (idx,jdx) --> [tgt:Relation, src:Relation]
    # idx is the time index
    # jdx is the qubit index
    edges = {}
    prev = None
    if dual:
        rels = [op.Az for op in ops]
    else:
        rels = [op.Ax for op in ops]
    for idx,A in enumerate(rels):
        #print("idx =", idx)
        #print("(%s<--%s)"%(A.tgt, A.src))
        #print("edges:", list(edges.keys()))
        if idx == 0:
            for jdx in range(A.src):
                edges[idx,jdx] = [A, None] # tgt,src
        else:
            assert prev.tgt == A.src
            for jdx in range(A.src):
                key = (idx,jdx) # previous timeslot
                assert key in edges, (key, edges)
                edge = edges[idx,jdx]
                assert edge[0] is None
                edge[0] = A # edge tgt is A
        for jdx in range(A.tgt):
            key = idx+1,jdx # the next time index
            assert key not in edges, "%s in %s"%(key,edges)
            edges[key] = [None, A] # tgt,src
        #print("edges:", list(edges.keys()))
        prev = A
    #print()
    keys = list(edges.keys())
    keys.sort(key = lambda key:key[1]) # sort by qubit index? do we care?
    #keys.sort() # do we care
    #print(keys)
    #lookup = {key:i for (i,key) in enumerate(keys)}
    m = sum(A.rank for A in rels)
    n = len(edges)
    H = numpy.zeros((m,n), dtype=int)
    #print(H)
    src = {}
    tgt = {}
    for A in rels:
        src[A] = [None]*A.src
        tgt[A] = [None]*A.tgt
    for col,key in enumerate(keys):
        edge = edges[key]
        A, B = edge # tgt, src
        time, idx = key
        #print("key =", key)
        if A is not None:
            #print(A)
            src[A][idx] = col
        if B is not None:
            #print(B)
            tgt[B][idx] = col
    row = 0
    for A in rels:
        idxs = tgt[A] + src[A]
        #print(idxs)
        H[row:row+A.rank, idxs] = A.A.A # triple A !
        row += A.rank
    H = Matrix(H)
    K = H.kernel()
    if verbose:
        print("H =", H.shape)
        print("K =", K.shape)
        print(K)
    flows = []
    for i in range(len(K)):
        flow = []
        for j in range(n):
            if K[i,j] == 0:
                continue
            idx, jdx = keys[j]
            assert idx <= len(ops)
            if idx < len(ops):
                op = ops[idx]
                atom = op.atom
                if jdx in atom.args:
                    #print("%s:%s"%(op.atom,jdx), end=' ')
                    flow.append((op.atom, jdx))
            else:
                flow.append((None, jdx)) # output
                #print("%s<--" % jdx, end=' ')
        flows.append(flow)
        #print()
    #print()
    if verbose:
        print(flows)
    return flows
    

def test_simple_detectors():
    print("="*79)
    print()

    verbose = True

    s = Syntax()
    MX, MZ, CX, RX, RZ, TICK = s.MX, s.MZ, s.CX, s.RX, s.RZ, s.TICK

    syntax = MX(0)*TICK()*RX(0)
    flows = find_detectors(syntax, [], verbose=verbose) # x_flow 
    flows = find_detectors(syntax, [], dual=True, verbose=verbose) # z_flow

    syntax = CX(0,1)
    flows = find_detectors(syntax, [0,1], verbose=verbose)
    assert len(flows) == 2

    flows = find_detectors(syntax, [0,1], dual=True, verbose=verbose)
    assert len(flows) == 2

    syntax = MX(0)*TICK()*CX(0,1)*TICK()*RX(0)*RZ(1)
    flows = find_detectors(syntax, [], verbose=verbose)
    assert len(flows) == 1

    flows = find_detectors(syntax, [], dual=True, verbose=verbose)
    assert len(flows) == 0

    syntax = MX(0)*CX(0,1)*CX(0,1)*RZ(0)
    flows = find_detectors(syntax, [1], verbose=verbose)
    assert len(flows) == 1

    flows = find_detectors(syntax, [1], dual=True, verbose=verbose)
    assert len(flows) == 1
    

    #test_goto()


def test_goto():

    s = Syntax()
    RZ, RX, CX = s.RZ, s.RX, s.CX
    MZ, MX = s.MZ, s.MX
    TICK = s.TICK
    spec = []
    for i in [0,4,5,6,7]:
        spec.append(RZ(i))
    for i in [1,2,3]:
        spec.append(RX(i))
    spec.append(TICK())
    for (i,j) in "10 35 26 14 20 36 15 64 07 57 67".split():
        spec.append(CX(int(i), int(j)))
        spec.append(TICK())
    spec.append(MZ(7))
    spec.append(TICK())
    #print(spec)
    syntax = reduce(mul, reversed(spec))
    #print(syntax)

    find_detectors(syntax)
    find_detectors(syntax, dual=True)
    #render(spec, "goto_713")


def test_iceberg():
    # See https://arxiv.org/abs/2504.21172 Fig 3.
    n = 20
    s = Syntax()
    MX, MZ, CX, RX, RZ = s.MX, s.MZ, s.CX, s.RX, s.RZ
    TICK = s.TICK

    syntax = TICK()*RZ(n)*RX(n+1) # ancilla
    for ij in "01 10 23 45 32 54 67 89 76 98 10,11 12,13 11,10 13,12 14,15 16,17 15,14 17,16 18,19 19,18".split():
        if "," in ij:
            ij = ij.split(",")
        i,j = ij
        syntax = (CX(int(i), int(n))) * syntax
        syntax = (CX(int(n+1), int(j))) * syntax
        syntax = (TICK()) * syntax

    syntax = MZ(n)*MX(n+1)*syntax
    #print(syntax)

    #render(syntax, "iceberg")

    src = list(range(n))
    find_detectors(syntax, src)
    find_detectors(syntax, src, dual=True)


def load_stim():
    name = argv.next() or "zero_prep_20_2_6.stim"
    data = open(name).read()
    spec = from_stim(data)
    find_detectors(spec)

def test_detectors():
    test_simple_detectors()
    test_goto()
    test_iceberg()
    

def test_syntax():
    s = Syntax()
    MX, MZ, CX, RX, RZ = s.MX, s.MZ, s.CX, s.RX, s.RZ
    spec = MX(0)*CX(0,1)*RX(0)*RZ(1)

    assert str(spec) == "MX(0)*CX(0,1)*RX(0)*RZ(1)"

    mod = Module()
    op = spec * mod
    assert mod.alloc == {1:0}

    lhs, rhs = (RX(0)*RZ(1))*Module(), (RZ(1)*RX(0))*Module()
    assert eqop(lhs, rhs)


def test_composite():

    white = Operator.white
    black = Operator.black
    _w = white(0, 1)
    w_ = white(1, 0)
    _b = black(0, 1)
    b_ = black(1, 0)
    I = Operator.identity(1)
    zero = Operator.zero(0,0)

    cx01 = (I @ black(1,2))*(white(2,1) @ I)
    cx10 = (I @ white(1,2))*(black(2,1) @ I)

    circ = _b * b_
    circ = _w * w_

    circ = (I@_b) * cx01*cx10 * (b_@I)
    #print(circ)

    circ = white(1,2)*white(2,1)
    s = str(circ)



def test_rnd():

    def getrand(n, m=None):
        if m is None:
            m = n
        hom = Hom(n,n)
        ops = [hom.CX(i,j) for i in range(n) for j in range(n) if i!=j]
        op = hom.identity()
        count = 0
        while count < n**2:
            g = choice(ops)
            op = g*op
            count += 1
        return op

    n = 5
    op = getrand(n)

    i = randint(0,n-1)
    mx = Hom(n-1, n).MZ(i) # 1 <-- 2 --> 2
    op = mx*op

    n -= 1
    op = getrand(n)*op

    i = randint(0,n-1)
    mx = Hom(n-1, n).MZ(i) # 1 <-- 2 --> 2
    op = mx*op


        

def get_random(size):
    n = 0
    hom = Hom(n,n)
    circ = hom.identity()
    for count in range(size):
        #print(circ.op)
        i = randint(0,1)
        if i==0:
            n = circ.src
            m = randint(max(0, n-1), n+1)
            hom = Hom(n, m)
            rhs = hom.random()
            #print(hom, rhs)
            circ = circ * rhs
        elif i==1:
            m = circ.tgt
            n = randint(max(0, m-1), m+1)
            hom = Hom(n, m)
            lhs = hom.random()
            #print(hom, lhs)
            circ = lhs * circ
    return circ

def test_random():
    for trial in range(10):
        circ = get_random(10)
        #print(circ.deepstr())


def eqop(A, B):
    assert isinstance(A, Circuit)
    assert isinstance(B, Circuit)
    return A.op == B.op


def test_hom():

    hom = Hom(3, 2)

    for i in range(3):
        c = hom.RX(i)
        c = hom.RZ(i)

    hom = Hom(2, 3)
    for i in range(3):
        c = hom.MX(i)
        c = hom.MZ(i)

    # test get_perm ----------------------------------------

    rel = Hom(3).get_perm([1,2,0])
    rhs = Hom(3,2).RX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).RX(1) * Hom(2,2).SWAP()
    assert lhs.op == rhs.op
    
    rel = Hom(3).get_perm([2, 0, 1])
    rhs = Hom(3,2).RX(0)
    rhs = rel * rhs
    lhs = Hom(3,2).RX(2)
    assert lhs.op == rhs.op
    
    # test CX ----------------------------------------

    w_ww = Operator.white(1, 2)
    bb_b = Operator.black(2, 1)
    identity = Operator.identity(1)
    CX = (w_ww @ identity) * (identity @ bb_b)

    space = Hom(2)
    assert CX.op == space.CX().op

    cx01 = space.CX(0, 1)
    cx10 = space.CX(1, 0)

    rel = cx01*cx10*cx01
    assert eqop(rel, space.SWAP())

    n = 3
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        cx = Hom(n).CX(i,j)
        assert cx.src == n
        assert cx.tgt == n
        mz = Hom(n-1, n).MZ(i)
        assert mz.src == n
        assert mz.tgt == n-1
        lhs = mz * cx
        rhs = mz 
        assert eqop(lhs, rhs)
    
        lhs = Hom(n-1, n).MX(j) * cx
        rhs = Hom(n-1, n).MX(j)
        assert eqop(lhs, rhs)

    # ------------------------------------------------

    b3 = Operator.black(3,1)
    w3 = Operator.white(3,1)

    iso = Hom(9,9).PERM([0,3,6,1,4,7,2,5,8])
    Ex = (b3@b3@b3)*w3
    #print("Ex:")
    #print(Ex)

    Hx = (Ex*Operator.white(1,0)).op.Ax.left
    #print(Hx)

    Ez = (w3@w3@w3)*b3
    #print("Ez:")
    #print(Ez)

    assert Ez.op == Ex.op.dual
    assert Ex.op == Ez.op.dual

    Hz = (Ez*Operator.white(1,0)).op.Ax.left
    #print(Hz)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.distance()
    #print(code)
    #print(code.longstr())

    assert eqop(get_encoder(code), Operator(Ez, Ex))

    # ------------------------------------------------
    # test from_encoder

    for trial in range(10):

        code = CSSCode.random(20, 5, 5)
        Ex, Ez = get_encoder(code).pair
        assert Ex.dual == Ez
        assert Ez.dual == Ex
        dode = from_encoder(Operator(Ex, Ez))
        assert code.is_equiv(dode)
    

    # ------------------------------------------------

    code = construct.get_422().to_css()
    Ex = get_encoder(code).pair[0]
    left = Ex.left
    #print(left)
    #print(left @ left)

    EE = Ex.tensor(Ex)
    #print(EE)
    dode = from_encoder(Operator(EE))
    #dode.bz_distance()
    #print(dode)
    assert dode.n == 16
    assert dode.k == 4
    

def test():

    from qumba.rel import test_linear
    test_linear()
    test_hom()
    test_random()
    test_composite()
    test_syntax()
    test_encoder()



if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        print("seed(%s)"%(_seed))
        seed(_seed)
        ra.seed(_seed)

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

