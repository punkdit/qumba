#!/usr/bin/env python

"""
experimenting with _simulating qec circuits .
previous version: wstim.py 

"""

if __name__ == "__main__":
    # much slower with threading so we disable this
    import os
    os.environ["OMP_NUM_THREADS"] = "1"


from functools import reduce
from operator import add, mul, matmul
from random import choice, seed

import numpy

from qumba.smap import SMap
from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba import construct
from qumba import lin
from qumba.qcode import strop, SymplecticSpace
from qumba import dense 
from qumba.pauli import Pauli

from qumba.qubitos.decode import SimpleDecoder
from qumba.qubitos.circuit import Circuit, to_density, basic_syndrome



class Tableau:
    def __init__(self, ops):
        m = len(ops)
        assert m>0
        n = ops[0].n
        self.ops = list(ops)
        space = SymplecticSpace(n)
        self.space = space
        self.shape = (m, 2*n)

    def __getitem__(self, idx):
        return self.ops[idx]

    def __len__(self):
        return len(self.ops)

    @classmethod
    def zero(cls, n):
        ops = [Pauli.Z(n, i) for i in range(n)]
        return Tableau(ops)

    @classmethod
    def plus(cls, n):
        ops = [Pauli.X(n, i) for i in range(n)]
        return Tableau(ops)

    @classmethod
    def I(cls, n):
        return Tableau([Pauli.I(n)])

    @classmethod
    def X(cls, n, i):
        return Tableau([Pauli.X(n, i)])

    @classmethod
    def Y(cls, n, i):
        return Tableau([Pauli.Y(n, i)])

    @classmethod
    def Z(cls, n, i):
        return Tableau([Pauli.Z(n, i)])

    def __str__(self):
        m, nn = self.shape
        n = nn//2
        smap = SMap()
        smap[0,1] = "="*n
        #smap[1,1] = strop(self.M)
        for i,op in enumerate(self.ops):
            smap[i+1,1] = str(op)
        smap[m+1,1] = "="*n
        return str(smap)

    def shortstr(self):
        return ",".join(str(op) for op in self.ops)

    def __repr__(self):
        return "Tableau(%r)"%(self.ops,)

    def act(self, *arg):
        #print("act", arg)
        #print(self)
        space = self.space
        ops = self.ops
        gate, idxs = arg[:2]
        if type(idxs) is int:
            idxs = [idxs]
        if gate == "H":
            for idx in idxs:
                #cliffop = space.H(idx)
                ops = [op.H(idx) for op in ops]
        elif gate == "CX":
            i, j = idxs
            #cliffop = space.CX(i, j)
            #M = M*op.t
            ops = [op.CX(i, j) for op in ops]
        elif gate == "R":
            pass
        elif gate == "TICK":
            pass
        elif gate == "M":
            pass
        else:
            assert 0, arg
        tab = Tableau(ops)
        #print("-->")
        #print(tab)
        return tab

    def run(self, circuit):
        tab = self
        for item in circuit:
            tab = tab.act(*item)
        return tab


class PauliDist:
    "a distribution over Pauli errors"
    def __init__(self, items):
        self.items = items

    def __str__(self):
        return "PauliDist(%s)"%(
            ', '.join("%s:%.4f"%(tab.shortstr(), p) for (tab,p) in self.items))

    def act(self, *arg):
        #print("PauliDist.act", arg, self)
        items = [(tab.act(*arg), p) for (tab,p) in self.items]
        tensor = PauliDist(items)
        #print("\t", tensor)
        return tensor

    def channel(self, space, rho):
        #print("channel", self)
        result = space.get_zero()
        #for t, p in self.items:
            #desc = strop(t)
            #print("\t", desc, p)
            #if t.M.sum() == 0:
            #    result += p * rho
            #else:
            #    op = space.get_pauli(desc)
            #    result += p*op*rho*op
        for tab, p in self.items:
            assert len(tab) == 1
            pauli = tab[0]
            op = pauli.get_dense()
            result += p*op*rho*op
        return result


class Model:
    def __init__(self, n):
        self.n = n
        self.nn = 2*n
        self.state = Tableau.zero(n)
        self.noise = []

    def DEPOLARIZE1(self, idxs, kw):
        #print("Model.DEPOLARIZE1", idxs, kw)
        p = kw['p'] # um...
        noise = self.noise
        n = self.n
        assert 0. <= p <= 1., repr(p)
        for idx in idxs:
            tensor = PauliDist([
                (Tableau.I(n), 1-p),
                (Tableau.X(n, idx), p/3),
                (Tableau.Y(n, idx), p/3),
                (Tableau.Z(n, idx), p/3)])
            noise.append(tensor)

    def X_ERROR(self, idxs, kw):
        #print("Model.X_ERROR", idxs, kw)
        p = kw['p'] # um...
        noise = self.noise
        n = self.n
        assert 0. <= p <= 1., repr(p)
        for idx in idxs:
            tensor = PauliDist([(Tableau.I(n), 1-p), (Tableau.X(n, idx), p)])
            noise.append(tensor)

    def Z_ERROR(self, idxs, kw):
        #print("Model.Z_ERROR", idxs, kw)
        p = kw['p'] # um...
        noise = self.noise
        n = self.n
        assert 0. <= p <= 1., repr(p)
        for idx in idxs:
            tensor = PauliDist([(Tableau.I(n), 1-p), (Tableau.Z(n, idx), p)])
            noise.append(tensor)

    def apply_gate(self, *arg):
        #print("Model.apply_gate", arg, len(self.noise))
        self.state = self.state.act(*arg)
        self.noise = [tensor.act(*arg) for tensor in self.noise]
        #print(self)

    def apply(self, circuit):
        for item in circuit:
            name = item[0]
            meth = getattr(self, name, None) 
            if meth is not None:
                meth(*item[1:])
            else:
                self.apply_gate(*item)

    def __str__(self):
        #return "Model\n%s\n%s"%(
            #self.state, self.noise)
        lines = ["Model", str(self.state)]
        #for (tab,p) in self.noise:
        #    lines.append("%s p=%s"%(tab, p))
        for t in self.noise:
            lines.append(str(t))
        return "\n".join(lines)

    def density(self):
        n = self.n
        assert n<12, "too big?"

        space = dense.Space(1)
        I = space.I
        X = space.X()
        Z = space.Z()
        zero = 0.5*(I+Z)
        plus = 0.5*(I+X)
    
        space = dense.Space(n)
        rho = reduce(matmul, [zero]*n)
        #print(rho)

        In = space.I
        rho = In
        for pauli in self.state:
            #desc = strop(row)
            #print(desc)
            #op = space.get_pauli(desc)
            op = pauli.get_dense()
            op = 0.5*(In + op)
            rho = op*rho

        for dist in self.noise:
            rho = dist.channel(space, rho)

        return rho

    def get_dist(self):
        print("get_dist")
        assert 0
        tab = self.state
        M = tab.M
        print(strop(M))
        m, nn = M.shape
        n = nn//2
        M = M.reshape((m,n,2))
        #print(M)
        op = Matrix([1,0]).reshape((1,1,2))
        M = M.hadamard_product(op)
        M = M.reshape((m,nn))
        print()
        M = M.row_reduce()
        print(strop(M))
        M = M[:, ::2]
        print(M)
        dist = {}
        denom = 2**len(M)
        for v in M.span():
            key = tuple(int(i) for i in v)
            dist[key] = 1/denom
        return dist





if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

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

