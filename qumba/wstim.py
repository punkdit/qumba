#!/usr/bin/env python

"""
Learning how to use stim .

"""


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

    #code = construct.get_713()
    code = construct.get_15_1_3()
    #code = CSSCode.random(17, 8, 8, distance=4)
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

    p = 0.05
    c.DEPOLARIZE1(idxs, p)
    names = []

    for h in css.Hz:
        #print("Hz", h)
        c.R(adx)
        for i in range(n):
            if h[i]:
                c.CX(i, adx)
                c.TICK()
        c.MR(adx)
        names.append("Hz")

    for h in css.Hx:
        #print("Hx", h)
        c.R(adx)
        c.H(adx)
        for i in range(n):
            if h[i]:
                c.CX(adx, i)
                c.TICK()
        c.H(adx)
        c.MR(adx)
        names.append("Hx")

    c.H(idxs)
    c.M(idxs)

    #import stim
    #print(stim.target_rec(-1))
    #c.DETECTOR([stim.target_rec(-1)])
    #return

    #print()
    #print(c)

    N = 120
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
    #print(bits, bits.shape)

    decoder = SimpleDecoder(css.get_dual())
    #print(decoder)

    x_syndrome = syndrome[:, mz:]
    ops = []
    for i in range(N):
        op = decoder.decode(p, x_syndrome[i])
        ops.append(op)

    correct = Matrix(ops)
    #print("\ncorrect:")
    #print(correct, correct.shape)
    A = bits.A + correct.A
    bits = Matrix(A)

    Lx = css.Lx
    Lz = css.Lz
    Tx = css.Tx
    Tz = css.Tz
    HLx = Hx.concatenate(Lx)
    HLz = Hz.concatenate(Lz)
    Jz = Hz.concatenate(Tz).concatenate(Lz)
    #print(HLx.t.solve(bits.t))
    A = Jz.t.solve(bits.t)
    assert A is not None

    fails = A[-k:, :]
    print(fails)
    err = fails.sum() / N
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


