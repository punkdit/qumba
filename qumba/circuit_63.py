#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
import pickle

import numpy

from qumba import lin
lin.int_scalar = numpy.int32 # qupy.lin
from qumba.lin import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba.autos import get_autos
from qumba import csscode, construct
from qumba.construct import get_422, get_513, get_golay, get_10_2_3, reed_muller
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.util import cross
from qumba.symplectic import Building
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
#from qumba import clifford, matrix
#from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
#from qumba.syntax import Syntax
from qumba.circuit import parsevec, Circuit, send, get_inverse, measure, barrier, variance, vdump, load

from qumba.circuit_css import process






prep = [
    'CX(11,31)', 'CX(51,50)', 'CX(58,9)', 'CX(53,51)', 'CX(45,43)',
    'CX(25,40)', 'CX(54,44)', 'CX(33,15)', 'CX(46,14)', 'CX(32,20)',
    'CX(48,35)', 'CX(29,39)', 'CX(33,30)', 'CX(43,22)', 'CX(29,40)',
    'CX(14,29)', 'CX(62,33)', 'CX(18,54)', 'CX(62,61)', 'CX(2,18)',
    'CX(23,49)', 'CX(31,37)', 'CX(25,42)', 'CX(47,34)', 'CX(0,22)',
    'CX(20,41)', 'CX(24,30)', 'CX(18,47)', 'CX(19,24)', 'CX(60,39)',
    'CX(38,54)', 'CX(53,16)', 'CX(15,52)', 'CX(62,30)', 'CX(4,36)',
    'CX(46,40)', 'CX(41,23)', 'CX(23,13)', 'CX(13,49)', 'CX(60,59)',
    'CX(50,11)', 'CX(20,27)', 'CX(57,7)', 'CX(47,40)', 'CX(49,27)',
    'CX(55,21)', 'CX(0,32)', 'CX(34,17)', 'CX(52,26)', 'CX(33,22)',
    'CX(26,21)', 'CX(27,25)', 'CX(51,15)', 'CX(22,4)', 'CX(22,49)',
    'CX(53,20)', 'CX(20,34)', 'CX(25,19)', 'CX(50,19)', 'CX(57,24)',
    'CX(30,5)', 'CX(31,61)', 'CX(28,52)', 'CX(0,28)', 'CX(42,29)',
    'CX(30,35)', 'CX(60,14)', 'CX(19,48)', 'CX(52,45)', 'CX(25,60)',
    'CX(62,59)', 'CX(54,33)', 'CX(29,56)', 'CX(52,60)', 'CX(42,26)',
    'CX(39,14)', 'CX(35,19)', 'CX(43,23)', 'CX(16,28)', 'CX(15,23)',
    'CX(9,29)', 'CX(57,54)', 'CX(21,23)', 'CX(53,17)', 'CX(16,1)',
    'CX(2,41)', 'CX(28,50)', 'CX(48,11)', 'CX(55,16)', 'CX(53,45)',
    'CX(10,53)', 'CX(48,40)', 'CX(21,61)', 'CX(38,47)', 'CX(30,38)',
    'CX(18,47)', 'CX(23,11)', 'CX(28,57)', 'CX(55,62)', 'CX(8,36)',
    'CX(42,21)', 'CX(27,47)', 'CX(17,0)', 'CX(53,21)', 'CX(23,0)',
    'CX(6,52)', 'CX(23,34)', 'CX(14,2)', 'CX(1,46)', 'CX(40,13)',
    'CX(28,30)', 'CX(26,17)', 'CX(47,12)', 'CX(37,56)', 'CX(19,39)',
    'CX(50,15)', 'CX(43,1)', 'CX(9,53)', 'CX(39,2)', 'CX(27,43)',
    'CX(25,39)', 'CX(58,30)', 'CX(14,15)', 'CX(45,14)', 'CX(36,62)',
    'CX(33,41)', 'CX(23,33)', 'CX(56,22)', 'CX(58,33)', 'CX(48,53)',
    'CX(41,18)', 'CX(9,59)', 'CX(19,3)', 'CX(47,62)', 'CX(32,3)',
    'CX(48,44)', 'CX(21,1)', 'CX(32,19)', 'CX(36,23)', 'CX(21,13)',
    'CX(9,21)', 'CX(60,38)', 'CX(47,51)', 'CX(21,61)', 'CX(5,21)',
    'CX(45,40)', 'CX(42,9)', 'CX(59,16)', 'CX(60,20)', 'CX(47,50)',
    'CX(12,35)', 'CX(58,51)', 'CX(22,26)', 'CX(7,41)', 'CX(27,9)',
    'CX(9,2)', 'CX(25,32)', 'CX(24,62)', 'CX(44,52)', 'CX(46,15)',
    'CX(32,31)', 'CX(43,26)', 'CX(17,59)', 'CX(26,39)', 'CX(0,28)', 
    'CX(59,60)', 'CX(62,44)', 'CX(15,19)', 'CX(51,1)', 'CX(34,22)', 
    'CX(27,62)', 'CX(41,48)', 'CX(22,47)', 'CX(26,17)', 'CX(6,59)', 
    'CX(52,55)', 'CX(12,5)', 'CX(37,23)', 'CX(49,14)', 'CX(11,62)',
    'CX(0,62)', 'CX(12,31)', 'CX(8,12)', 'CX(1,55)', 'CX(2,25)',
    'CX(59,6)', 'CX(19,42)', 'CX(61,47)', 'CX(0,49)', 'CX(12,13)', 
    'CX(6,43)', 'CX(40,22)', 'CX(9,58)', 'CX(26,22)', 'CX(31,26)', 
    'CX(22,16)', 'CX(23,50)', 'CX(10,16)', 'CX(1,53)', 'CX(16,5)',
    'CX(6,40)', 'CX(23,40)', 'CX(21,16)', 'CX(19,10)', 'CX(21,50)',
    'CX(9,10)', 'CX(11,19)', 'CX(10,0)', 'CX(19,17)', 'CX(52,21)',
    'CX(10,52)', 'CX(27,61)', 'CX(61,27)', 'CX(61,31)', 'CX(61,44)',
    'CX(9,48)', 'CX(61,59)', 'CX(9,27)', 'CX(17,1)', 'CX(50,11)',
    'CX(11,31)', 'CX(14,61)', 'CX(13,44)', 'CX(7,23)', 'CX(44,7)',
    'CX(11,61)', 'CX(48,0)', 'CX(14,59)', 'CX(7,2)', 'CX(6,14)', 
    'CX(59,3)', 'CX(8,17)', 'CX(31,4)', 'CX(52,1)', 'CX(11,13)',
    'H(0)', 'H(1)', 'H(2)', 'H(3)', 'H(4)', 'H(5)']


def test_prep():

    code = QCode.fromstr("""
    XXXIIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIII
    ZZZIIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIII
    XXIIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIX
    ZZIIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZ
    XIIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXX
    ZIIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZ
    IIIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXXX
    IIIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZZ
    IIIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXXXI
    IIIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZZI
    IIXIIXIIIXXIXXIIXIXXIXIXXXIXXXXIIXXIIIXIXIXIIXXXXXXIXIIIIIXXXII
    IIZIIZIIIZZIZZIIZIZZIZIZZZIZZZZIIZZIIIZIZIZIIZZZZZZIZIIIIIZZZII
    """)

    print(code)

    n = code.n

    css = code.to_css()
    space = SymplecticSpace(n)
    H = space.H

    expr = tuple(prep)

    mx, mz = css.mx, css.mz
    E = space.get_expr(expr)

    dode = QCode.from_encoder(E, k=code.k)
    dode.distance()
    assert dode.is_equiv(code)

    css = dode.to_css()
    Hz = css.Hz
    Lz = css.Lz
    #print(Hz)
    #print(Lz)
    HLz = numpy.concatenate((Hz, Lz))


    circuit = Circuit(n)

    c = measure + barrier + expr

    print(c)

    qasms = []
    qasms.append(circuit.run_qasm(c))

    if argv.showqasm:
        print(qasms[-1])
        return

    if argv.load_samps:
        samps = load()

    else:
        shots = argv.get("shots", 1000)
        samps = send(qasms, shots=shots, error_model=True, opts={'use-dfl': False})
        #print(samps)

    H = HLz

    if argv.reversed:
        # For the 8,2,2 code H is symmetric in qubit reversal
        # Ie., this does not make any difference, even though
        # the qasm measurements are little-endian 
        H = H[:, list(reversed(range(n)))]

    if argv.shuffle:
        idxs = list(range(n))
        shuffle(idxs)
        #H = H[:, idxs]
        #code = QCode(H)
        code = code.apply_perm(idxs)

    #print("H =")
    #print(H)

    idxs = circuit.labels # final qubit permutation
    H = H[:, idxs] 

    print("samps:", len(samps))

    process(code, samps, circuit)



if __name__ == "__main__":

    from time import time
    start_time = time()

    print(argv)

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        from random import seed
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

