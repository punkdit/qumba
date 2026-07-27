#!/usr/bin/env python

"""
experimenting with stim .

"""

import numpy

from qumba.smap import SMap
from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba import construct
#from qumba import decode

from qumba import lin


class Decoder:
    def __init__(self, code):
        code = code.to_css()
        self.code = code

#    def get_T(self, err_op):
    def get_T(self, syndrome):
        # bitflip X-type errors, frustrate Hz checks, and produce Tx
        code = self.code
        n = code.n
        #T = zeros2(n)
        Hz = code.Hz
        Tx = code.Tx
        T = syndrome * Tx
        return T
#        m = Hz.shape[0]
#        for i in range(m):
#            #if dot2(err_op, Hz[i]):
#            if syndrome[i]:
#                T += Tx[i]
#        T %= 2
#        return T

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
        import stim
        self.circuit = stim.Circuit()

    def CX(self, i, j):
        self.circuit.append("CX", [i, j])

    def CZ(self, i, j):
        self.circuit.append("CZ", [i, j])

    def __getattr__(self, name):
        f = lambda *args, **kw : self.circuit.append(name, *args, **kw)
        return f

    def __str__(self):
        circuit = self.circuit
        return str(circuit)



def test_decode():

    code = construct.get_713()

    code = construct.get_15_1_3()
    #code = code.get_dual()

    code = construct.get_512()

    #code = CSSCode.random(17, 7, 7, distance=3)
    #code = CSSCode.random(27, 11, 11, distance=4)
    #code = CSSCode.random(27, 13, 13, distance=5)
    #code = construct.get_golay(23)

    print(code)
    print(code.to_qcode().longstr())

    css = code.to_css()
    #print(css)

    #print(css.longstr())

    #E = code.get_encoder()
    #print(E)


    n = css.n # 1 ancilla
    k = css.k

    idxs = list(range(n)) # code 
    adx = idxs[-1]+1 # ancilla

    c = Circuit()
    c.R(idxs) # |0>^n

    Hx = css.Hx
    Hz = css.Hz
    mx = len(Hx)
    mz = len(Hz)

    if 1:
        # state prep for logical |0>^k 
        HLx = Hx.concatenate(css.Lx)
        Jx = HLx.normal_form()
        px = Jx.get_pivots()
        #print("Jx:")
        #print(Jx, px)
    
        c.H([col for (row,col) in px])
        #for i in range(mx):
        for (row, col) in px:
            for j in range(col+1, n):
                if Jx[row,j]:
                    c.CX(col, j)
    
    #c.H(idxs) # |+>^n
    c.TICK()

    p = 0.01
    names = []
    R = 3 # repeat

    for count in range(R):
        c.DEPOLARIZE1(idxs, p)
        for idx,h in enumerate(css.Hz):
            #print("Hz", h)
            c.R(adx)
            c.DEPOLARIZE1([adx], p)
            for i in range(n):
                if h[i]:
                    #c.DEPOLARIZE1([i, adx], p) # <--- the really bad errors
                    c.CX(i, adx)
                    c.TICK()
            c.MR(adx)
            names.append("Hz[%d,%d]"%(idx, count))
    
        for idx,h in enumerate(css.Hx):
            #print("Hx", h)
            c.R(adx)
            c.H(adx)
            c.DEPOLARIZE1([adx], p)
            for i in range(n):
                if h[i]:
                    #c.DEPOLARIZE1([i, adx], p) # <--- the really bad errors
                    c.CX(adx, i)
                    c.TICK()
            c.H(adx)
            c.MR(adx)
            names.append("Hx[%d,%d]"%(idx, count))

    lookup = {name:i for (i,name) in enumerate(names)}
    print(lookup)

    c.H(idxs)
    c.M(idxs)

    #import stim
    #print(stim.target_rec(-1))
    #c.DETECTOR([stim.target_rec(-1)])
    #return

    #print()
    #print(c)

    N = 100
    circuit = c.circuit
    sampler = circuit.compile_sampler()
    result = sampler.sample(shots=N)
    result = result.astype(int)
    syndrome = result[:, :-n]
    bits = result[:, -n:]
    #print(result, result.shape)

    syndrome = Matrix(syndrome)
    bits = Matrix(bits)

    print("\nsyndrome:")
    smap = SMap()
    smap[1,0] = str(syndrome.t)
    for i in range(N):
        smap[0,i] = str(i%10)
    for (i, name) in enumerate(names):
        smap[1+i,N+2] = name
    print(smap)
    print()
    #print(bits, bits.shape)

    decoder = SimpleDecoder(css.get_dual())
    print(decoder)

    rows = [lookup["Hx[%d,%d]"%(i,0)] for i in range(mx)]
    #x_syndrome = syndrome[:, mz:]
    x_syndrome = syndrome[:, rows]

    x_syndrome = []
    for i in range(mx):
        rows = [lookup["Hx[%d,%d]"%(i,j)] for j in range(R)]
        row = syndrome[:, rows]
        #print("row:")
        #print(row.t)
        A = row.A.sum(1)
        A = A.astype(float)
        A /= R
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
        op = decoder.decode(p, x_syndrome[i])
        ops.append(op)

    #return

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


def test_sample():
    import stim

    circuit = stim.Circuit()

    circuit.append("R", [0, 1])

    
    # First, the circuit will initialize a Bell pair.
    circuit.append("H", [0])
    circuit.append("X_ERROR", [0, 1], 0.05)
    circuit.append("CNOT", [0, 1])
    
    # Then, the circuit will measure both qubits of the Bell pair in the Z basis.
    circuit.append("M", [0, 1])

    print(circuit)

    print(len(circuit))
    for op in circuit:
        print("\t", op)

    N = 1000
    sampler = circuit.compile_sampler()
    result = (sampler.sample(shots=N))
    result = (result.astype(int))
    stats = ( result.sum(axis=1) % 2 )

    print( stats.sum() / N )



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


