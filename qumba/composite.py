#!/usr/bin/env python

"""
Use linear relations for CSS _circuits (aka phase-free ZX-calculus).
Keep track of how we compose operators using Composite objects.
and more...


"""

import time
from functools import reduce, cache
from operator import matmul, lshift, mul
from random import choice, randint

import numpy

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
    # NO: use object identity ... 
    #    def __eq__(self, other):
    #        assert isinstance(other, Atom)
    #        return self.key == other.key
    #    def __hash__(self):
    #        return hash(self.key)


class Molecule(Atom):
    def __init__(self, atoms):
        atoms = list(atoms)
        assert len(set(atoms)) == len(atoms), "non unique objects"
        self.atoms = list(atoms)
    def __str__(self):
        return "*".join(str(atom) for atom in self.atoms)
    #def __eq__(self, other):
    #    assert 0, "um?"
    #    return self.atoms == other.atoms


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
    def __init__(self, op, hom):
        assert isinstance(op, Operator)
        assert isinstance(hom, Hom)
        self.op = op
        self.hom = hom
        self.src = hom.n
        self.tgt = hom.m
        self.shape = (self.tgt, self.src)

#    @property
#    def tgt(self)

    def __mul__(self, other):
        assert self.src == other.tgt, (self.hom, other.hom)
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
        Rz = Relation(left, right)
    
        left = code.Hz.stack(code.Lz)
        right = Matrix.zeros((code.mz, code.k))
        right = right.stack(Matrix.identity(code.k))
        Rx = Relation(left, right)
    
        hom = Hom(code.n, code.k)
        return Operator(Rx, Rz, hom)
    
    def to_code(self):
        op = self.op
        assert isinstance(op, Operator)
        Rz, Rx = op.pair
    
        #print("from_encoder")
        #print(Rx, Rx.shape)
        assert isinstance(Rx, Relation)
        k = Rx.src
        assert Rz.src == k
    
        #if k==0:
        #    print(Rx)
    
        w = Relation.white(1,0)
        if k:
            op = reduce(matmul, [w]*k)
            #print(op, op.shape)
            Hx = (Rx*op).left
            Hz = (Rz*op).left
        else:
            Hx = Rx.left
            Hz = Rz.left
    
        mx, n = Hx.shape
        #print("Hx =")
        #print(Hx, Hx.shape)
    
        mz = len(Hz)
        assert mz+mx+k == n
        #print("Hz =")
        #print(Hz, Hz.shape)
    
        op = Relation(Matrix.identity(k), Matrix.zeros((k,0)))
        HLx = (Rx*op).left
        HHLx = Hx.stack(HLx).linear_independent()
        assert HHLx[:mx, :] == Hx
        Lx = HHLx[mx:, :]
    
        #print("Lx =")
        #print(Lx, Lx.shape)
    
        op = Relation(Matrix.identity(k), Matrix.zeros((k,0)))
        HLz = (Rz*op).left
        HHLz = Hz.stack(HLz).linear_independent()
        assert HHLz[:mz, :] == Hz
        Lz = HHLz[mz:, :]
    
        code = CSSCode(Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz)
        return code


get_encoder = Circuit.from_code
from_encoder = Circuit.to_code


class Operator(Circuit):
    "The _atomic Operator's in a Composite Circuit"
    def __init__(self, Rx, Rz, hom, check=True):
        if isinstance(Rx, Circuit):
            Rx = Rx.op.Rx
        if isinstance(Rz, Circuit):
            Rz = Rz.op.Rx
        assert isinstance(hom, Hom)
        Rx = Relation.promote(Rx)
        if Rz is None:
            Rz = Rx.dual
        Rz = Relation.promote(Rz)
        self.Rx = Rx
        self.Rz = Rz
        self.pair = (Rx, Rz)
        if check:
            assert Rx.dual == Rz
        self.info = None
        assert Rx.tgt == Rz.tgt
        assert Rx.src == Rz.src
        assert Rx.tgt == hom.m
        assert Rx.src == hom.n
        # i am my own Operator
        Circuit.__init__(self, self, hom)

    @property
    def reverse(self):
        Rx, Rz = self.Rx, self.Rz
        return Operator(Rx.reverse, Rz.reverse, self.hom.reverse)

    @property
    def dual(self):
        Rx, Rz = self.Rx, self.Rz
        return Operator(Rz, Rx, self.hom)

    def copy(self):
        return Operator(self.Rx.copy(), self.Rz.copy(), self.hom)

    #def set(self, atom):
        #assert type(atom) is Atom
        #assert self.atom is None, "%s already got atom %r"%(self, self.atom,)
        #self.atom = atom

    def set(self, *info):
        assert self.info is None, "%s already got info %r"%(self, self.info,)
        self.info = info

    def walk(self):
        yield self

    def mul(self, other):
        assert isinstance(other, Operator), type(other)
        assert self.hom.src == other.hom.tgt # reconcile TODO
        Rx = self.Rx * other.Rx
        Rz = self.Rz * other.Rz
        return Operator(Rx, Rz, self.hom*other.hom)

    def __matmul__(self, other):
        assert isinstance(other, Operator), type(other)
        Rx = self.Rx @ other.Rx
        Rz = self.Rz @ other.Rz
        return Operator(Rx, Rz, self.hom@other.hom)

    def __eq__(self, other):
        assert isinstance(other, Operator), (type(self), type(other))
        assert self.hom == other.hom, "%s != %s"%(self.hom, other.hom)
        return self.pair == other.pair

    def __hash__(self):
        assert 0
        return hash(self.pair)

    def __str__(self):
        return "(%s | %s)"%(self.Rx.spanstr(), self.Rz.spanstr())

    def is_identity(self):
        tgt, src = self.shape
        if tgt != src:
            return False
        return self == self.hom.identity()


class Composite(Circuit): # XXX binary tree, just make this a list ??
    def __init__(self, lhs, rhs):
        lhs = Circuit.promote(lhs)
        rhs = Circuit.promote(rhs)
        self.lhs = lhs
        self.rhs = rhs
        lop = lhs.op
        rop = rhs.op
        #op = lop.mul(rop.op)

        Rx = (lop.Rx * rop.Rx)
        Rz = (lop.Rz * rop.Rz)
        hom = lhs.hom*rhs.hom
        op = Operator(Rx, Rz, hom)

        Circuit.__init__(self, op, hom)

    def __str__(self):
        return "(%s*%s=%s)"%(self.lhs, self.rhs, self.op)

    def deepstr(self, depth=0):
        lhs = self.lhs.deepstr(depth+1)
        rhs = self.rhs.deepstr(depth+1)
        lines = [self.op.deepstr(depth) + "=", lhs + "*", rhs]
        return '\n'.join(lines)

    def __matmul__(self, other):
        assert isinstance(other, Operator), type(other)
        lhs = self.lhs
        rhs = self.rhs
        composite = (lhs@other) * (rhs@other) # Argghh, recursive
        return composite

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
    # keep a list of tgt&src qubit indexes for Operator

    def __init__(self, tgt, src=None):
        if src is None:
            src = tgt
        if type(tgt) is int:
            tgt = list(range(tgt))
        if type(src) is int:
            src = list(range(src))
        assert len(set(tgt)) == len(tgt), "not unique"
        assert len(set(src)) == len(src), "not unique"
        self.tgt = list(tgt)
        self.src = list(src)
        self.m = len(tgt)
        self.n = len(src)
        self.shape = (self.m, self.n)

    @property
    def reverse(self):
        return Hom(self.src, self.tgt)

    def __matmul__(self, other):
        assert isinstance(other, Hom)
        return Hom(self.src+other.src, self.tgt+other.tgt)

    def __mul__(lhs, rhs):
        assert isinstance(rhs, Hom)
        assert lhs.src == rhs.tgt, "%s != %s"%(lhs.src, rhs.tgt)
        return Hom(lhs.tgt, rhs.src)

    def __str__(self):
        return "(%s<--%s)"%(self.tgt, self.src)
    __repr__ = __str__

    def __eq__(self, other):
        return self.tgt==other.tgt and self.src==other.src

    def black(self):
        l, r = self.shape
        Rx = Relation.black(l, r)
        Rz = Relation.white(l, r)
        return Operator(Rx, Rz, self)

    def white(self):
        l, r = self.shape
        Rx = Relation.white(l, r)
        Rz = Relation.black(l, r)
        return Operator(Rx, Rz, self)

    def zero(self):
        l, r = self.shape
        Rx = Relation.zero(l, r)
        return Operator(Rx, None, self)

    def identity(self):
        assert self.tgt == self.src
        n = self.n
        Rx = Relation.identity(n)
        return Operator(Rx, Rx, self)

    def RX(self):
        "prepare X state"
        assert self.m > self.n, "wut"
        src = [i for i in self.tgt if i in self.src]
        assert src == self.src, "bad hom %s"%(self,)
        rels = []
        w_ = Relation.white(1,0)
        I = Relation.identity(1)
        for idx in self.tgt:
            if idx in self.src:
                rels.append(I)
            else:
                rels.append(w_)
        rel = reduce(matmul, rels)
        op = Operator(rel, None, self)
        return op

    def RZ(self):
        "prepare Z state"
        op = self.RX().dual
        return op

    def MX(self):
        "postselect <+|"
        #m, n = self.shape
        op = self.reverse.RX().reverse
        return op

    def MZ(self):
        "postselect <0|"
        #m, n = self.shape
        op = self.reverse.RZ().reverse
        return op

    def CX(self, idx=0, jdx=1):
        assert self.src == self.tgt, "bad Hom"
        assert idx != jdx
        assert idx in self.src
        assert jdx in self.src
        i = self.src.index(idx)
        j = self.src.index(jdx)
        m = self.m
        lhs = Matrix.identity(m)
        rhs = numpy.identity(m, dtype=int)
        rhs[j, i] = 1
        rhs = Matrix(rhs)
        rel = Relation(lhs, rhs)
        return Operator(rel, None, self)

    def PERM(self, perm):
        """ send perm[j]<---j """
        assert self.tgt == self.src
        m, n = self.shape
        assert m == n, "wrong Hom %s"%(self,)
        assert m == len(perm), "wrong Hom %s"%(self,)
        assert len(set(perm)) == m
        assert set(perm) == set(self.tgt)
        #assert self.tgt == list(range(m)), "%s"%(self.tgt,)
        tgt = self.tgt
        idxs = [tgt.index(i) for i in perm]
        lhs = Matrix.identity(m)
        rhs = Matrix.get_perm(idxs)
        rel = Relation(lhs, rhs)
        return Operator(rel, None, self)

    def SWAP(self, idx, jdx):
        m, n = self.shape
        assert m == n, "wrong Hom %s"%(self,)
        assert m >= 2, "wrong Hom %s"%(self,)
        assert idx != jdx
        assert self.tgt == self.src
        assert idx in self.src
        assert jdx in self.src
        i = self.src.index(idx)
        j = self.src.index(jdx)
        f = list(self.src)
        f[i], f[j] = f[j], f[i]
        op = self.PERM(f)
        return op

#    def random(self):
#        m, n = self.shape
#        if m==n==0:
#            op = self.identity()
#        elif m==0 and n==1:
#            op = choice([self.MX(0), self.MZ(0)])
#        elif n==0 and m==1:
#            op = choice([self.RX(0), self.RZ(0)])
#        elif m==n and n==1:
#            op = choice([
#                self.identity(), 
#                self.w_*self._w,
#                self.w_*self._b,
#                self.b_*self._b,
#                self.b_*self._w])
#            assert op.tgt == m, (op.tgt, op.src, self)
#            assert op.src == n, (op.tgt, op.src, self)
#        elif m==n>1:
#            i = randint(0, n-1)
#            j = randint(0, n-2)
#            j = j+1 if j>=i else j
#            op = self.CX(i,j)
#        elif m == n-1:
#            i = randint(0, n-1)
#            op = choice([self.MX(i), self.MZ(i)])
#        elif n == m-1:
#            i = randint(0, n-1)
#            op = choice([self.RX(i), self.RZ(i)])
#        else:
#            assert 0, (m, n)
#        assert op.tgt == m, (op.tgt, op.src, self)
#        assert op.src == n, (op.tgt, op.src, self)
#        return op
        

class Module:
    """ A mutable context for keeping track of live qubits
        and evaluating Syntax expressions.
    """
    def __init__(self, idxs=[]): # TODO: why do we need idxs ?? get rid of idxs
        idxs = list(idxs)
        idxs.sort()
        #print("Module(%s)"%(idxs,))
        self.hom = Hom(idxs, idxs) # <-- mutates ?

    def interpret(self, syntax, verbose=False):
        if verbose: print("interpret", syntax)
        hom = self.hom
        op = None
        for atom in reversed(syntax.atoms):
            if verbose: print("\tget:", atom, hom)
            method = getattr(self, atom.name)
            idxs = atom.args
            lhs = method(hom.tgt, *idxs)
            if verbose: print("\t  ==", lhs.hom)
            assert type(lhs) is Operator,\
                "must be atom Operator, not %r"%type(lhs)
            assert lhs.hom.src == hom.tgt, "wup"
            lhs.set(atom, lhs.hom)
            if op is None:
                op = lhs
            else:
                op = lhs*op
            hom = op.hom
        return op

    def RX(self, src, i):
        assert i not in src
        tgt = src + [i]
        tgt.sort()
        op = Hom(tgt, src).RX()
        return op

    def RZ(self, src, i):
        assert i not in src
        tgt = src + [i]
        tgt.sort()
        op = Hom(tgt, src).RZ()
        return op

    def CX(self, src, i, j):
        op = Hom(src, src).CX(i, j)
        return op

    def SWAP(self, src, i, j):
        op = Hom(src, src).SWAP(i, j)
        return op

    def MX(self, src, i):
        tgt = list(src)
        tgt.remove(i)
        op = Hom(tgt, src).MX()
        return op

    def MZ(self, src, i):
        tgt = list(src)
        tgt.remove(i)
        op = Hom(tgt, src).MZ()
        return op

    def TICK(self, src):
        return Hom(src, src).identity()


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


OUTPUT = "OUTPUT"
INPUT = "INPUT"


class Item:
    def __init__(self, atom, idx, tgt=None, src=None):
        self.atom = atom
        self.idx = idx
        self.tgt = tgt
        self.src = src
    def __str__(self):
        return str((self.atom, self.idx, self.tgt, self.src))
    __repr__ = __str__


class Flow:
    def __init__(self, syntax, src=[], dual=False, detectors=False, verbose=False):
        if verbose:
            print("\nget_flow(%s)"%syntax, src, "z_flow" if dual else "x_flow")
        module = Module(src)
        composite = module.interpret(syntax, verbose=verbose)
        assert isinstance(composite, Circuit)
        ops = list(composite.walk())
        for op in ops:
            assert op.info is not None
            assert op.info[0] in syntax # XXX remove (slow)
            #print(op, op.hom)
        ops = [op for op in ops if not op.is_identity()] # get rid of the TICK's
        #print(len(ops))
        ops = list(reversed(ops)) # <------ backwards <----
        del syntax, composite # don't touch these (reversed)
    
        if dual:
            rels = [op.Rz for op in ops]
        else:
            rels = [op.Rx for op in ops]
    
        lookup = [] # irel index --> col idxs = tgts+srcs
        R = rels[0]
        assert len(src) == R.src
        prev = list(range(R.src))
        init = list(prev)
        n = len(prev) # next col idx
        m = 0
        cols = {} # map col idx --> (irel , qubit index)
        for irel,R in enumerate(rels):
            if verbose:
                print(R.tgt, "<--", R.src, "prev=%s"%prev)
            for ii,i in enumerate(prev):
                cols[i] = (irel, ii)
            new = []
            for i in range(R.tgt):
                new.append(n + i)
                cols[n+i] = (irel+1, i) # the next irel 
            assert len(prev) == R.src
            # these are tgt indexes + src indexes
            # so we can paste Relation's directly into H below
            idxs = new+prev 
            #jdxs = list(range(m, m+R.rank))
            lookup.append((m, idxs))
            n += R.tgt
            m += R.rank
            prev = new
    
        #print(lookup)
    
        # now ready to build the check matrix
        H = numpy.zeros((m,n), dtype=int)
        for irel,R in enumerate(rels):
            row,idxs = lookup[irel]
            H[row:row+R.rank, idxs] = R.A.A # <-- paste the Relation into H

        H = Matrix(H)
        K = H.kernel()
        if verbose:
            print("H =")
            print(H, H.shape)
            print("K =")
            print(K, K.shape)

        flows = []
        for i in range(len(K)):
            flow = []
            for j in range(n):
                if K[i,j] == 0:
                    continue
                idx, jdx = cols[j]
                #assert idx is None or idx < len(ops)
                assert idx <= len(ops)
                flow.append((idx, jdx)) # output
            flows.append(flow)
            #print()
        #print()
        if verbose:
            print(flows)
        #return flows
        self.flows = flows
        self.ops = ops
        self.init = init

    def get_iflow(self, i):
        ops = self.ops
        #print("ops:", len(ops))
        flow = self.flows[i]
        iflow = []
        init = list(self.init)
        for (idx,jdx) in flow:
            assert idx <= len(ops)
            #iflow.append((idx, jdx)) # output
            #if jdx in init:
            #if idx == 0:
                #iflow.append((INPUT, idx, jdx))
            item = None
            if idx < len(ops):
                op = ops[idx]
                atom, hom = op.info
                #if (idx,jdx) == (1,2):
                #if idx <= 1:
                #    print(atom, hom, idx, jdx)
                assert jdx < len(hom.src)
                j = hom.src[jdx] # look at atom src (input)
                #if jdx in atom.args:
                #if jdx in hom.src:
                if j in atom.args: # this flow is *input* to this atom
                    #print("%s:%s"%(op.atom,jdx), end=' ')
                    #iflow.append((atom, idx, jdx))
                    item = Item(atom, idx, None, jdx)
            else:
                assert idx == len(ops)
                #iflow.append((OUTPUT, idx, jdx)) # output
                item = Item(OUTPUT, idx, None, jdx)
                #print("%s<--" % jdx, end=' ')
            if item is not None:
                iflow.append(item)
        return iflow

    def get_oflow(self, i):
        ops = self.ops
        #print("ops:", len(ops))
        flow = self.flows[i]
        oflow = []
        init = list(self.init)
        for (idx,jdx) in flow:
            assert idx <= len(ops)
            #oflow.append((idx, jdx)) # output
            #if jdx in init:
            item = None
            if idx == 0:
                #oflow.append((INPUT, idx, jdx))
                item = Item(INPUT, idx, jdx, None)
            elif 0 < idx < len(ops):
                op = ops[idx-1]
                atom, hom = op.info
                assert jdx < len(hom.tgt)
                j = hom.tgt[jdx] # look at atom tgt (output)
                if j in atom.args: # this flow is *output* of this atom
                    #oflow.append((atom, idx, jdx))
                    item = Item(atom, idx, jdx, None)
            #else:
            #    assert idx == len(ops)
            #    oflow.append((OUTPUT, idx, jdx)) # output
            #    #print("%s<--" % jdx, end=' ')
            if item is not None:
                oflow.append(item)
        return oflow

    def get_flow(self, i):
        flow = self.get_iflow(i) + self.get_oflow(i)
        flow.sort(key = lambda item: -item.idx)
        return flow

    def __getitem__(self, i):
        return self.flows[i]

    def __len__(self):
        return len(self.flows)
    


# ----------------------------------------------------------------------------
# test functions




def test_encoder():

    s = Syntax()
    MX, MZ, CX, RX, RZ = s.MX, s.MZ, s.CX, s.RX, s.RZ

    spec = RX(0)*RZ(2)
#    op = Module([1]).interpret(spec)
    op = Module().interpret(spec)
    rhs = op.op

    print(rhs.hom)

    # check that Module doesn't scramble the indexes ?
    #lhs = Hom([0,1,2],[0,1]).RZ()*(Hom([0],[]).RX() @ Hom([1],[1]).identity())
    lhs = Hom([0,2],[0]).RZ()*Hom([0],[]).RX()
    assert lhs.op.Rx == rhs.op.Rx
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
    op = hom.identity()
    for i in range(10):
        j = randint(0, n-1)
        k = randint(0, n-1)
        if j!=k:
            syntax.append(CX(j,k))
            op = op*hom.CX(j,k)
    syntax = reduce(mul, syntax)
    print(syntax)
    composite = Module(list(range(n))).interpret(syntax)
    assert composite.op == op.op

    module = Module(list(range(n)))
    composite = module.interpret(syntax) # TODO: check we touch every qubit
    #print(composite.op == op.op)
    assert composite.op == op.op

    # -------------------------------------------

    spec = CX(0,3)*CX(0,5)*CX(0,6)*CX(1,3)*CX(1,4)*CX(1,5)*CX(2,4)\
        *CX(2,5)*CX(2,6)*CX(3,0)*CX(3,1)*CX(3,4)*CX(3,6)\
        *RX(0)*RX(1)*RX(2)*RZ(4)*RZ(5)*RZ(6)

    op = Module([3]).interpret(spec)
    E = op.op
    code = (from_encoder(E))
    assert code.is_equiv(construct.get_713()) 

    # -------------------------------------------

    src = [3]
    op = Hom(src, src).identity()
    for i in [0,1,2]:
        src = op.hom.tgt
        tgt = src + [i]
        tgt.sort()
        op = Hom(tgt, src).RX()*op
    for i in [4,5,6]:
        src = op.hom.tgt
        tgt = src + [i]
        tgt.sort()
        op = Hom(tgt, src).RZ()*op
    tgt = op.hom.tgt
    hom = Hom(tgt, tgt)
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


def test_detectors():
    print("="*79)
    print()

    verbose = True

    s = Syntax()
    MX, MZ, CX, RX, RZ, TICK = s.MX, s.MZ, s.CX, s.RX, s.RZ, s.TICK

    #syntax = CX(0,1)
    #flow = Flow(syntax, [0,1], verbose=True)
    #assert len(flow) == 2


    syntax = CX(1,2)*CX(0,1)
    flow = Flow(syntax, [0,1,2], verbose=False)
    assert len(flow) == 3

    flow = Flow(MZ(0)*RZ(0))
    assert len(flow)==0

    src = [0,1]
    syntax = MZ(2)*MX(3)*CX(3,2)*CX(1,2)*CX(0,2)*CX(3,2)*RZ(2)*RX(3)
    flow = Flow(syntax, src, verbose=True)
    print(len(flow))
    #for i in range(len(flow)):
    for i in [0,1]:
        print(flow[i])
        print("\t", flow.get_flow(i))
        #print("\t", flow.get_iflow(i))
        #print("\t", flow.get_oflow(i))
    print()

    flow = Flow(syntax, src, dual=True, verbose=False)
    print(len(flow))

    syntax = MX(0)*TICK()*RX(0)
    flow = Flow(syntax, [], verbose=verbose) # x_flow 
    assert len(flow) == 1
    flow = Flow(syntax, [], dual=True, verbose=verbose) # z_flow
    assert len(flow) == 0

    syntax = MX(0)*RX(0)*MX(0)*TICK()*RX(0)
    flow = Flow(syntax, [], verbose=verbose) # x_flow 
    assert len(flow) == 2

    syntax = MX(0)*TICK()*RZ(0)
    flow = Flow(syntax, [], verbose=verbose)
    assert len(flow) == 0
    flow = Flow(syntax, [], dual=True, verbose=verbose)
    assert len(flow) == 0

    syntax = CX(0,1)
    flow = Flow(syntax, [0,1], dual=True, verbose=verbose)
    assert len(flow) == 2

    syntax = MX(0)*TICK()*CX(0,1)*TICK()*RX(0)*RZ(1)
    flow = Flow(syntax, [], verbose=verbose)
    assert len(flow) == 1

    flow = Flow(syntax, [], dual=True, verbose=verbose)
    assert len(flow) == 0

    syntax = CX(0,1)*CX(0,1)
    flow = Flow(syntax, [0,1], verbose=verbose)
    assert len(flow) == 2
    assert list(flow) == [[(0, 0), (1, 0), (1, 1), (2, 0)], [(0, 1), (1, 1), (2, 1)]]

    syntax = MX(0)*CX(0,1)*CX(0,1)*RZ(0)
    flow = Flow(syntax, [1], verbose=verbose)
    assert len(flow) == 1

    flow = Flow(syntax, [1], dual=True, verbose=verbose)
    assert len(flow) == 1
    

    test_goto()


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

    flow = Flow(syntax)
    assert len(flow) == 3
    #for i in range(len(flow)):
    #    print(i, flow.get(i))

    flow = Flow(syntax, dual=True)
    assert len(flow) == 5
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
    Flow(syntax, src)
    Flow(syntax, src, dual=True)


def test_prep():
    name = argv.next() or "zero_prep_20_2_6.stim"
    data = open(name).read()
    spec = from_stim(data)
    Flow(spec)

def test_all_detectors():
    test_simple_detectors()
    test_iceberg()
    test_prep()
    

def test_syntax():
    s = Syntax()
    MX, MZ, CX, RX, RZ = s.MX, s.MZ, s.CX, s.RX, s.RZ
    spec = MX(0)*CX(0,1)*RX(0)*RZ(1)

    assert str(spec) == "MX(0)*CX(0,1)*RX(0)*RZ(1)"

    mod = Module()
    op = mod.interpret(spec)
    assert op.hom == Hom([1], [])

    lhs, rhs = (RX(0)*RZ(1)), (RZ(1)*RX(0))
    lhs = Module().interpret(lhs)
    rhs = Module().interpret(rhs)
    assert eqop(lhs, rhs)


def test_composite():

    white = lambda m,n: Hom(m,n).white()
    black = lambda m,n: Hom(m,n).black()

    _w = white(0, 1)
    w_ = white(1, 0)
    _b = black(0, 1)
    b_ = black(1, 0)
    I = Hom(1,1).identity()
    zero = Hom(0,0).zero()

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

    import pytest

    hom = Hom(3, 2)

    for i in range(3):
        c = hom.RX()
        c = hom.RZ()

    hom = Hom(2, 3)
    for i in range(3):
        c = hom.MX()
        c = hom.MZ()

    # test get_perm ----------------------------------------

    lhs = Hom(3).PERM([1,2,0])
    rhs = Hom([0,1,2],[1,2]).RX()
    assert rhs.hom == Hom([0,1,2],[1,2])
    rhs = lhs * rhs

    rhs = Hom(3).PERM([1,2,0]) * Hom([0,1,2],[1,2]).RX()
    lhs = Hom([0,1,2],[0,2]).RX() * Hom([0,2],[0,2]).SWAP(0,2)
    assert lhs.op.Rx == rhs.op.Rx
    #assert lhs.op == rhs.op
    
    rel = Hom([0,1,2]).PERM([2, 0, 1])
    rhs = Hom([0,1,2],[1,2]).RX()
    rhs = rel * rhs
    lhs = Hom([0,1,2],[0,1]).RX()
    assert lhs.op.Rx == rhs.op.Rx
    #assert lhs.op == rhs.op

    with pytest.raises(AssertionError):
        Hom([2,1,0], [0,1]).RX()

    #return # TODO:

    # test CX ----------------------------------------

    #w_ww = Operator.white(1, 2)
    w_ww = Hom(1,2).white()
    #bb_b = Operator.black(2, 1)
    bb_b = Hom(2,1).black()
    #identity = Operator.identity(1)
    identity = Hom(1,1).identity()
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

#    b3 = Operator.black(3,1)
#    w3 = Operator.white(3,1)
    b3 = Hom(3,1).black()
    w3 = Hom(3,1).white()

    iso = Hom(9,9).PERM([0,3,6,1,4,7,2,5,8])
    Rx = (b3@b3@b3)*w3
    #print("Rx:")
    #print(Rx)

#    Hx = (Rx*Operator.white(1,0)).op.Rx.left
    Hx = (Rx*Hom(1,0).white()).op.Rx.left
    #print(Hx)

    Rz = (w3@w3@w3)*b3
    #print("Rz:")
    #print(Rz)

    assert Rz.op == Rx.op.dual
    assert Rx.op == Rz.op.dual

    #Hz = (Rz*Operator.white(1,0)).op.Rx.left
    Hz = (Rz*Hom(1,0).white()).op.Rx.left
    #print(Hz)

    code = CSSCode(Hx=Hx, Hz=Hz)
    code.distance()
    #print(code)
    #print(code.longstr())

    op = Operator(Rz, Rx, Hom(code.n, code.k))
    assert eqop(get_encoder(code), op)

    # ------------------------------------------------
    # test from_encoder

    for trial in range(10):

        code = CSSCode.random(20, 5, 5)
        Rx, Rz = get_encoder(code).pair
        assert Rx.dual == Rz
        assert Rz.dual == Rx
        hom = Hom(code.n, code.k)
        dode = from_encoder(Operator(Rx, Rz, hom))
        assert code.is_equiv(dode)
    

    # ------------------------------------------------

    code = construct.get_422().to_css()
    Rx = get_encoder(code).pair[0]
    left = Rx.left
    #print(left)
    #print(left @ left)

    EE = Rx.tensor(Rx)
    #print(EE)
    hom = Hom(16, 4)
    dode = from_encoder(Operator(EE, None, hom))
    #dode.bz_distance()
    #print(dode)
    assert dode.n == 16
    assert dode.k == 4
    

def test():

    from qumba.rel import test_linear
    test_linear()
    test_hom()
    #test_random()
#    test_composite()
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

