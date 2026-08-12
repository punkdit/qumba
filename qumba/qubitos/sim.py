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
from qumba.qubitos.model import Model


def test_density(circuit):

    n = circuit.n
    print("\ntest_density:", 2**n)

    circuit.FINI()

    lhs = to_density(circuit)

    #print("lhs:")
    #print(lhs)

    model = Model(n)
    model.apply(circuit)
    #print(model)
    rhs = model.density()

    #print("rhs:")
    #print(rhs)

    if lhs!=rhs:
        print("lhs != rhs")
        print(lhs)
        print(rhs)
        print()
        print(circuit)
        print(model)
        assert 0


    bits = list(numpy.ndindex((2,)*n))
    stats = {bit:0 for bit in bits}

    N = 10000
    samples = circuit.simulate(N)
    #print(samples)
    for row in samples:
        row = tuple(int(i) for i in row)
        stats[row] += 1

    for i,idx in enumerate(bits):
        r = lhs[i,i].real
        if r > 0.01:
            print(idx, "%.4f"%r, "%.4f"%(stats[idx]/N))

    #print(model)
    #dist = model.get_dist()
    #for k,v in dist.items():
    #    print(k,v)
#
#    print()


def test():

    print("\ntest():")

    Y = dense.Y
    YY = Y@Y

    #print(YY)

    from qumba import pauli
    yy = (pauli.nI@pauli.I) *( pauli.Y @ pauli.Y )
    assert str(yy) == "-YY"

    op = yy.get_dense()
    assert op == -YY

    ZX = dense.Z @ dense.X
    CX = dense.Space(2).CX(1, 0)
    assert( CX.d * ZX * CX == -YY )

    ops = [dense.Z @ dense.I]
    s = dense.Space(2)
    seq = [s.H(1), s.CX(1,0), s.H(1), s.CX(1,0)] # reversed !
    for g in reversed(seq):
        ops.append( g.d * ops[-1] * g )

    assert ops[0] == dense.Z @ dense.I
    assert ops[1] == dense.Z @ dense.Z
    assert ops[2] == dense.Z @ dense.X
    assert ops[3] == -dense.Y @ dense.Y
    assert ops[4] == dense.Y @ dense.Y

    # ------------------------------------------------------------------------

    n = 2
    circuit = Circuit()
    data = [circuit.ALLOC("q%d"%i) for i in range(n)]
    H = lambda i:circuit.H(data[i])
    CX = lambda i,j:circuit.CX(data[i], data[j])

    CX(1,0)
    H(1)
    CX(1,0)
    H(1)

    if 1:

        s = dense.Space(2)
        ops = [s.H(1), s.CX(1,0), s.H(1), s.CX(1,0)] # reversed !
        op = reduce(mul, ops)
        zero = dense.ket0
        op = op*(zero@zero)
        #print()
        lhs = op*op.d
        #print("lhs =", lhs)
    
        # dense Tableau
        tbl = [s.Z(0), s.Z(1)]
        for op in reversed(ops):
            tbl = [op * pa * op.d for pa in tbl] # conjugate action

        II = s.I
        XZ = s.X(0)*s.Z(1)
        XI = s.X(0)
        rho = (1/4) * (II+YY) * (II+XZ)
        #print("rho =", rho)
        assert(rho == lhs)
        assert(tbl[0] == YY)
        assert(tbl[1] == XZ)
    
        #model = Model(2)
        #for arg in circuit:
        #    print(model.state.shortstr(), "-->", arg)
        #    model.apply([arg])
        #print(model)

    test_density(circuit)

    # ------------------------------------------------------------------------

    n = 3
    circuit = Circuit()
    data = [circuit.ALLOC("q%d"%i) for i in range(n)]


    p = 0.01
    circuit.DEPOLARIZE1(data, p)

    circuit.H(data[0])
    circuit.CX(data[0], data[1])
    circuit.CX(data[0], data[2])

    test_density(circuit)

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

    test_density(circuit)

    # ------------------------------------------------------------------------

    n = 2
    circuit = Circuit()
    data = [circuit.ALLOC("q%d"%i) for i in range(n)]
    circuit.H(data[1])
    circuit.CX(data[1], data[0])
    circuit.H(data[1])
    circuit.CX(data[1], data[0])
    test_density(circuit)
    #return

    # ------------------------------------------------------------------------

    #while 1:
    for trial in range(10):
        circuit = Circuit.random(4, 20)
        test_density(circuit)

    #return

    # ------------------------------------------------------------------------

    #code = construct.get_713()
    code = construct.get_422()
    circuit = Circuit()
    basic_syndrome(circuit, code, 0.01)

    n = circuit.n

    model = Model(n)
    model.apply(circuit)
    #print(model)
    #print(n)

    test_density(circuit)


def test_decode():

    #code = construct.get_713()

    code = construct.get_surface(5, 5)

    #code = construct.get_15_1_3()
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

    p = 0.05

    circuit = Circuit()
    basic_syndrome(circuit, css, p)

    #print(circuit)
    items = circuit.items

#    for arg in items:
#        if arg[0] == "M":
#            print(arg)
#
#    return

    #return

    N = 120
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

    p1 = 10*p
    ops = []
    for i in range(N):
        op = decoder.decode(p1, x_syndrome[i])
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

