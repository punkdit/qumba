#!/usr/bin/env python

"""
experimenting with _simulating qec circuits .
previous version: wstim.py 

"""

import numpy

from qumba.smap import SMap
from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba import construct
#from qumba import decode
from qumba import lin
from qumba.qcode import strop, SymplecticSpace


class Decoder:
    def __init__(self, code):
        code = code.to_css()
        self.code = code

    def get_T(self, syndrome):
        # bitflip X-type errors, frustrate Hz checks, and produce Tx
        code = self.code
        n = code.n
        Hz = code.Hz
        Tx = code.Tx
        T = syndrome * Tx
        return T

    def decode(self, p, err_op, verbose=False, **kw):
        return None


class SimpleDecoder(Decoder):
    """
    Simple (&slow) optimal decoder that sums probabilities over cosets.
    """
    def __init__(self, code):
        Decoder.__init__(self, code)
        self.all_Lx = list(code.Lx.span())
        self.all_Hx = list(code.Hx.span())
        self.code = code

    def get_dist(self, p, T):
        "distribution over logical operators"
        code = self.code
        dist = []
        sr = 0.
        n = code.n
        for l_op in self.all_Lx:
            r = 0.
            T1 = l_op + T
            for s_op in self.all_Hx:
                T2 = s_op + T1
                d = T2.sum()
                r += (1-p)**(n-d)*p**d
            sr += r
            dist.append(r)
        dist = [r/sr for r in dist]
        return dist

    def decode(self, p, syndrome, verbose=False, **kw):
        T = self.get_T(syndrome)
        dist = self.get_dist(p, T)
        p1 = max(dist)
        idx = dist.index(p1)
        l_op = self.all_Lx[idx]
        op = l_op+T
        return op



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

    def TICK(self):
        self.append("TICK", [])

    def CX(self, i, j):
        self.append("CX", [i, j])

    def H(self, idxs):
        self.append("H", idxs)

    def DEPOLARIZE1(self, idxs, p):
        self.append("DEPOLARIZE1", idxs, p=p)

    def X_ERROR(self, idxs, p):
        self.append("X_ERROR", idxs, p=p)

    def Z_ERROR(self, idxs, p):
        self.append("Z_ERROR", idxs, p=p)

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



def basic_syndrome(circuit, code, p=0.01, R=3):

    css = code.to_css()
    n = css.n # 1 ancilla
    k = css.k

    #idxs = list(range(n)) # code 
    #adx = idxs[-1]+1 # ancilla

    ERROR = circuit.DEPOLARIZE1
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
                    #ERROR([i, adx], p) # <--- the really bad errors
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
                    #ERROR([i, adx], p) # <--- the really bad errors
                    circuit.CX(ancilla, data[i])
                    circuit.TICK()
            circuit.H(ancilla)
            circuit.M(ancilla)

    circuit.H(data)
    circuit.M(data)

    circuit.R = R
    circuit.p = p

    return circuit


class Tableau:
    def __init__(self, M):
        if len(M.shape) == 1:
            M = M.reshape(1, M.shape[0])
        m, nn = M.shape
        assert nn%2 == 0
        n = nn//2
        space = SymplecticSpace(n)
        assert space.is_isotropic(M)
        self.M = M
        self.space = space
        self.shape = M.shape

    @classmethod
    def zero(cls, n):
        nn = 2*n
        M = []
        for i in range(n):
            v = [0]*nn
            v[2*i+1] = 1
            M.append(v)
        M = Matrix(M)
        return Tableau(M)

    @classmethod
    def plus(cls, n):
        nn = 2*n
        M = []
        for i in range(n):
            v = [0]*nn
            v[2*i] = 1
            M.append(v)
        M = Matrix(M)
        return Tableau(M)

    @classmethod
    def I(cls, n):
        nn = 2*n
        v = [0]*nn
        return Tableau(Matrix(v))

    @classmethod
    def X(cls, n, i):
        nn = 2*n
        v = [0]*nn
        v[2*i] = 1
        return Tableau(Matrix(v))

    @classmethod
    def Y(cls, n, i):
        nn = 2*n
        v = [0]*nn
        v[2*i:2*i+2] = [1,1]
        return Tableau(Matrix(v))

    @classmethod
    def Z(cls, n, i):
        nn = 2*n
        v = [0]*nn
        v[2*i+1] = 1
        return Tableau(Matrix(v))

    def __str__(self):
        m, nn = self.shape
        n = nn//2
        smap = SMap()
        smap[0,1] = "="*n
        smap[1,1] = strop(self.M)
        smap[m+1,1] = "="*n
        return str(smap)

    def shortstr(self):
        return strop(self.M)

    def __repr__(self):
        return "Tableau(%r)"%(self.M,)

    def act(self, *arg):
        #print("act", arg)
        #print(self)
        space = self.space
        M = self.M
        gate, idxs = arg[:2]
        if type(idxs) is int:
            idxs = [idxs]
        if gate == "H":
            for idx in idxs:
                op = space.H(idx)
                M = M*op.t
        elif gate == "CX":
            i, j = idxs
            op = space.CX(i, j)
            M = M*op.t
        elif gate == "R":
            pass
        elif gate == "TICK":
            pass
        #else:
        #    assert 0, arg
        tab = Tableau(M)
        #print("-->")
        #print(tab)
        return tab

    def run(self, circuit):
        tab = self
        for item in circuit:
            tab = tab.act(*item)
        return tab


class Tensor:
    "a distribution over Pauli errors"
    def __init__(self, items):
        self.items = items

    def __str__(self):
        return "Tensor(%s)"%(
            ', '.join("%s:%.4f"%(tab.shortstr(), p) for (tab,p) in self.items))

    def act(self, *arg):
        #print("Tensor.act", arg, self)
        items = [(tab.act(*arg), p) for (tab,p) in self.items]
        tensor = Tensor(items)
        #print("\t", tensor)
        return tensor


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
            tensor = Tensor([
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
            tensor = Tensor([(Tableau.I(n), 1-p), (Tableau.X(n, idx), p)])
            noise.append(tensor)

    def Z_ERROR(self, idxs, kw):
        #print("Model.Z_ERROR", idxs, kw)
        p = kw['p'] # um...
        noise = self.noise
        n = self.n
        assert 0. <= p <= 1., repr(p)
        for idx in idxs:
            tensor = Tensor([(Tableau.I(n), 1-p), (Tableau.Z(n, idx), p)])
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


def test():

    print("\ntest():")

    #test_decode()
    #return

    # ------------------------------------------------------------------------

    n = 2
    tab = Tableau.zero(n)
    #print(tab)
    tab = tab.act("H", 0)
    #print(tab)
    tab = tab.act("CX", [0, 1])
    #print(repr(str(tab)))
    assert str(tab).replace(" ", "") == "==\nXX\nZZ\n=="

    # ------------------------------------------------------------------------

    n = 2
    circuit = Circuit()
    data = [circuit.ALLOC("q%d"%i) for i in range(n)]

    p = 0.01
    circuit.H(data[0])
    circuit.X_ERROR(data, p)
    circuit.CX(data[0], data[1])

    #print(circuit)

    model = Model(n)
    model.apply(circuit)
    #print(model)

    #return


    # ------------------------------------------------------------------------

    code = construct.get_713()
    #print(code.longstr())
    n = code.n

    circuit = Circuit()
    #basic_syndrome(circuit, code)
    
    data = [circuit.ALLOC("q%d"%i) for i in range(n)]

    p = 0.01
    circuit.DEPOLARIZE1(data, p)

    circuit.prep(code, data)
    #print(circuit)

    #tab = Tableau.zero(n)
    #tab = tab.apply(circuit)
    #print(tab)

    model = Model(n)
    model.apply(circuit)

    #print(model)

    # ------------------------------------------------------------------------

    code = construct.get_713()
    circuit = Circuit()
    basic_syndrome(circuit, code)

    n = circuit.n
    model = Model(n)
    model.apply(circuit)

    print(model)


def test_decode():

    code = construct.get_713()

    code = construct.get_15_1_3()
    #code = code.get_dual()

    #code = construct.get_512()

    #code = CSSCode.random(17, 7, 7, distance=3)
    #code = CSSCode.random(27, 11, 11, distance=4)
    #code = CSSCode.random(27, 13, 13, distance=5)
    #code = construct.get_golay(23)

    print(code)
    print(code.to_qcode().longstr())

    n = code.n
    css = code.to_css()
    mx = css.mx
    mz = css.mz
    k = css.k

    circuit = Circuit()
    basic_syndrome(circuit, css)

    #print(circuit)
    items = circuit.items

#    for arg in items:
#        if arg[0] == "M":
#            print(arg)
#
#    return

    #return

    N = 60
    result = circuit.simulate(N)
    circuit.dump(result)

    decoder = SimpleDecoder(css.get_dual())
    print(decoder)

    x_syndrome = []
    for i in range(mx):
        rows = [circuit.lookup["Hx[%d,%d]"%(i,j)] for j in range(circuit.R)]
        row = result[:, rows]
        #print("row:")
        #print(row.t)
        A = row.A.sum(1)
        A = A.astype(float)
        A /= circuit.R
        A = numpy.round(A)
        assert numpy.max(A) <= 1.
        assert numpy.min(A) >= 0.
        A = A.astype(int)
        A = Matrix(A)
        #print("average:")
        #print(A, A.shape)
        x_syndrome.append(A)
        #x_syndrome.append(row)

    x_syndrome = Matrix(x_syndrome).t

    print("x_syndrome.t:")
    print(x_syndrome.t)

    ops = []
    for i in range(N):
        op = decoder.decode(circuit.p, x_syndrome[i])
        ops.append(op)

    #return
    bits = result[:, [circuit.lookup["q%d"%i] for i in range(n)]]

    correct = Matrix(ops)
    #print("\ncorrect:")
    #print(correct, correct.shape)
    bits = bits + correct

    assert bits.shape == (N, n)
    print("bits:")
    print(bits.t)
    print()

    Lx = css.Lx
    Lz = css.Lz
    Tx = css.Tx
    Tz = css.Tz
    Hx = css.Hx
    Hz = css.Hz
    HLx = Hx.concatenate(Lx)
    HLz = Hz.concatenate(Lz)

    # return to groundstate, brute force it
    assert len(HLz) < 20, "too big?"
    Uz = list(HLz.span())
    Uz = Matrix(Uz)
    print("Uz:", Uz.shape)

    send = []
    for u in bits:
        Jz = Uz + u
        jz = Jz.sum(1)
        weight = jz.min()
        idxs, = numpy.where(jz == weight)
        send.append(Uz[idxs[0], :].reshape(n))
    ubits = Matrix(send)
    #print(bits.shape)
    assert ubits.shape == (N, n)
    #print((bits + ubits).t)
    bits = ubits

    # read out logical errors:
    HTLz = Hz.concatenate(Tz).concatenate(Lz)

    #print(HLx.t.solve(bits.t))
    A = HTLz.t.solve(bits.t)
    assert A is not None

    print("\nfails:")
    fails = A[-k:, :]
    print(fails)
    fails = fails.sum(0)
    #print("fails:")
    #print(fails)
    idxs, = numpy.where(fails)
    print("fails:", idxs)
    err = len(idxs) / N
    print("err: %.2f%%"%(100*err))

    smap = SMap()
    for i in range(N):
        smap[0,i] = str(i%10)
    row = 1
    smap[row,0] = str(A)

    col = A.shape[1] + 2
    for i in range(mz):
        smap[row, col] = "Hz"
        row += 1
    for i in range(mx):
        smap[row, col] = "Tz"
        row += 1
    for i in range(css.k):
        smap[row, col] = "Lz"
        row += 1
    print()
    print("A = HTLz.solve(bits + correct)")
    print(smap)




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


