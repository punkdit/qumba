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


class Circuit:
    def __init__(self):
        self.lookup = {} # name -> idx
        self.names = [] # list of names of qubit's
        self.items = []
        self.measure = {} # name -> idx for measure's

    @property
    def n(self):
        return len(self.names)

    def __getitem__(self, idx):
        return self.items[idx]

    def __len__(self):
        return len(self.items)

    ##########################################################################
    # circuit constructors:

    def ALLOC(self, name=None):
        lookup = self.lookup
        names = self.names
        if name is None:
            name = "q%d"%len(names)
        assert type(name) is str
        assert name not in lookup
        idx = len(names)
        #print("ALLOC", name, "-->", idx)
        lookup[name] = idx
        names.append(name)
        self.append("R", [name]) # <---- reset
        return name

    def M(self, names):
        if isinstance(names, str):
            names = [names]
        self.append("M", names)
        # mark these qubits as dead 
        for name in names:
            idx = self.lookup[name]
            assert name not in self.measure
            self.measure[name] = len(self.measure)

    def FINI(self):
        for name in self.names:
            if name not in self.measure:
                self.M(name)

    def TICK(self):
        self.append("TICK", [])

    def CX(self, nami, namj):
        self.append("CX", [nami, namj])

    def H(self, names):
        self.append("H", names)

    def DEPOLARIZE1(self, names, p):
        self.append("DEPOLARIZE1", names, p=p)

    def X_ERROR(self, names, p):
        self.append("X_ERROR", names, p=p)

    def Z_ERROR(self, names, p):
        self.append("Z_ERROR", names, p=p)

    ##########################################################################

    @classmethod
    def random(cls, n, depth):
        circuit = cls()
        p = 0.01
        data = [circuit.ALLOC("q%d"%i) for i in range(n)]
        pairs = [(data[i], data[j]) for i in range(n) for j in range(n) if i!=j]
        for i in range(depth):
            op = choice("CX CX H X_ERROR Z_ERROR".split())
            if op == "CX":
                ij = choice(pairs)
                circuit.CX(*ij)
            elif op == "H":
                circuit.H(choice(data))
            elif op == "X_ERROR":
                circuit.X_ERROR(choice(data), p)
            elif op == "Z_ERROR":
                circuit.Z_ERROR(choice(data), p)
            else: 
                assert 0
        return circuit
            

    def __str__(self):
        return str(self.items)

    def dump(self, result):
        print("\nresult:")
        N, M = result.shape
        smap = SMap()
        smap[1,0] = str(result.t)
        for i in range(N):
            smap[0,i] = str(i%10)
        for i in range(M):
            name = self.names[i]
            smap[1+i,N+2] = name
        print(smap)
        print()
        #print(bits, bits.shape)

    def append(self, gate, names, **kw):
        if isinstance(names, str):
            names = [names]
        lookup = self.lookup
        idxs = []
        for name in names:
            assert isinstance(name, str), "name expected, got %r instead"%name
            assert name in lookup, "qubit %r not found"%name
            idx = lookup[name]
            idxs.append(idx)
        arg = (gate, idxs, kw)
        self.items.append(arg)

    def get_stim(self):
        import stim
        circuit = stim.Circuit()
        for arg in self.items:
            gate, idxs, kw = arg
            if "p" in kw:
                circuit.append(gate, idxs, kw["p"]) # whack whack
            else:
                assert not kw, str(kw)
                circuit.append(gate, idxs)
            #print(gate, idxs)
        return circuit

    def simulate(self, N):
        circuit = self.get_stim()
        sampler = circuit.compile_sampler()
        result = sampler.sample(shots=N)
        result = result.astype(int)
        result = Matrix(result)
        #print(result.shape)
        #assert 0
        cols = [self.measure[name] for name in self.names]
        #print(cols)
        assert result.shape == (N, len(cols)), "measure all the qubits?"
        result = result[:, cols]
        return result

    def prep(circuit, code, data): 
        "prepare a logical (plus?) state for <code> on <data> qubits"
        css = code.to_css()
        n = css.n
        assert n==len(data)
        Hx = css.Hx
        Hz = css.Hz
        mx = len(Hx)
        mz = len(Hz)
    
        #circuit.R(idxs) # |0>^n
        HLx = Hx.concatenate(css.Lx)
        Jx = HLx.normal_form()
        px = Jx.get_pivots()
        #print("Jx:")
        #print(Jx, px)
    
        circuit.H([data[col] for (row,col) in px])
        #for i in range(mx):
        for (row, col) in px:
            for j in range(col+1, n):
                if Jx[row,j]:
                    circuit.CX(data[col], data[j])
        
        #circuit.H(idxs) # |+>^n
        #circuit.TICK()



def basic_syndrome(circuit, code, p, R=3):

    css = code.to_css()
    n = css.n # 1 ancilla
    k = css.k

    #idxs = list(range(n)) # code 
    #adx = idxs[-1]+1 # ancilla

    #ERROR = circuit.DEPOLARIZE1
    ERROR = circuit.Z_ERROR

    data = [circuit.ALLOC("q%d"%i) for i in range(n)]
    #print(data)

    circuit.prep(css, data)

    Hx = css.Hx
    Hz = css.Hz
    mx = len(Hx)
    mz = len(Hz)

    #for i in range(n):
    #    circuit.X_ERROR(data[i], 0.1)

    for count in range(R):
        ERROR(data, p)
        for idx,h in enumerate(css.Hz):
            #print("Hz", h)
            #circuit.R(adx)
            ancilla = ("Hz[%d,%d]"%(idx, count))
            circuit.ALLOC(ancilla)
            ERROR([ancilla], p)
            for i in range(n):
                if h[i]:
                    #ERROR([data[i], ancilla], p) # <--- the really bad errors
                    circuit.CX(data[i], ancilla)
                    circuit.TICK()
            circuit.M(ancilla)
    
        for idx,h in enumerate(css.Hx):
            #print("Hx", h)
            #circuit.R(adx)
            ancilla = ("Hx[%d,%d]"%(idx, count))
            circuit.ALLOC(ancilla)
            circuit.H(ancilla)
            ERROR([ancilla], p)
            for i in range(n):
                if h[i]:
                    #ERROR([data[i], ancilla], p) # <--- the really bad errors
                    circuit.CX(ancilla, data[i])
                    circuit.TICK()
            circuit.H(ancilla)
            circuit.M(ancilla)

    circuit.H(data)
    circuit.M(data)

    circuit.R = R
    circuit.p = p

    return circuit


def to_density(circuit):
    #print("to_density")

    space = dense.Space(1)
    I = space.I
    X = space.X()
    Z = space.Z()
    zero = 0.5*(I+Z)
    plus = 0.5*(I+X)

    n = circuit.n
    assert n<12, "too big?"

    space = dense.Space(n)
    rho = reduce(matmul, [zero]*n)
    #print(rho)

    In = space.I
    for item in circuit:
        #print(item)
        gate = item[0]
        idxs = item[1]
        op = None
        if gate == "R":
            pass
        elif gate == "H":
            op = In
            for i in idxs:
                op = space.H(i) * op
            rho = op*rho*op.d
        elif gate == "CX":
            i, j = idxs
            op = space.CX(i, j)
            rho = op*rho*op.d
        elif gate == "X_ERROR":
            kw = item[2]
            p = kw["p"]
            assert 0.<=p<=1.
            for i in idxs:
                Xi = space.X(i)
                rho = (1-p) * rho + p*Xi*rho*Xi # apply Xi with probability p
        elif gate == "Z_ERROR":
            kw = item[2]
            p = kw["p"]
            assert 0.<=p<=1.
            for i in idxs:
                Zi = space.Z(i)
                rho = (1-p) * rho + p*Zi*rho*Zi # apply Zi with probability p
        elif gate == "DEPOLARIZE1":
            kw = item[2]
            p = kw["p"]
            assert 0.<=p<=1.
            for i in idxs:
                Xi = space.X(i)
                Yi = space.Y(i)
                Zi = space.Z(i)
                rho = (1-p) * rho + (p/3)*(Xi*rho*Xi + Yi*rho*Yi + Zi*rho*Zi) # apply Xi,Yi or Zi
        elif gate == "TICK":
            pass
        elif gate == "M":
            pass
        else:
            assert 0, item
    #print(rho)
    return rho




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

