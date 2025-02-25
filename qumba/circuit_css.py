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
from qumba import csscode, construct
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
from qumba import transversal 
#from qumba import clifford, matrix
#from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
#from qumba.syntax import Syntax
from qumba.circuit import (Circuit, measure, barrier, send, vdump, variance,
    parsevec, strvec, find_state, get_inverse, load_batch, send_idxs)

def css_encoder(Hx, dual=False):
    _, n = Hx.shape
    s = Syntax()
    CX, CZ, H, X, Z, I = s.CX, s.CZ, s.H, s.X, s.Z, s.get_identity()
    g = I
    if dual:
        #print("dual encoder")
        CX = lambda i,j : s.CX(j,i) # swap this
    #else:
        #print("primal encoder")

    idxs = []
    for row in range(len(Hx)):
        j0 = row
        while Hx[row, j0] == 0:
            j0 += 1
        assert Hx[row, j0]
        idxs.append(j0)
        for j in range(j0+1, n):
            if Hx[row, j]:
                g = g*CX(j0,j)
    if dual:
        for i in range(n):
            if i not in idxs:
                g = g*H(i)
    else:
        for i in idxs:
            g = g*H(i)
    return g.name


def test():
    if argv.code == (10,2,3):
        code = unwrap(construct.get_513())
    elif argv.code == (7,1,3):
        code = construct.get_713()
    elif argv.code == (9,1,3):
        code = construct.get_surface(3, 3)
    elif argv.code == (18,2,3):
        code = construct.get_toric(3, 3)
    elif argv.code == (20,2,4):
        code = construct.get_toric(2, 4)
    elif argv.code == (16,6,4):
        code = construct.reed_muller() # [[16,6,4]]
    elif argv.code == (7,1,3):
        code = construct.get_toric(4,0) # [[16,2,4]]
    else:
        return

    dual = argv.dual
    #code.distance()
    qasm_code(code, dual)


def qasm_code(code, dual=False):
    print(code, "dual =", dual)
    #print(code)
    #print(code.longstr())
    n = code.n
    k = code.k

    css = code.to_css()
    Hx = css.Hx
    Hx = row_reduce(Hx)
    Hz = css.Hz
    Hz = row_reduce(Hz)

    print("Hx =")
    print(shortstr(Hx))
    print("Hz =")
    print(shortstr(Hz))

    prep_0 = css_encoder(Hx)
    prep_p = css_encoder(Hz, True)

    if argv.code == (10,2,3):
        print("using HH")
        HH = ('P(0,3,1,4,2,5,8,6,9,7)', 'H(0)', 'H(1)', 'H(2)',
            'H(3)', 'H(4)', 'H(5)', 'H(6)', 'H(7)', 'H(8)', 'H(9)')
        prep_0 = HH + prep_p


    Hn = tuple("H(%d)"%i for i in range(n))
    fini_0 = measure
    fini_p = measure + Hn

    if dual:
        c = fini_p + barrier + prep_p
        code = code.get_dual()
    else:
        c = fini_0 + barrier + prep_0

    circuit = Circuit(n)
    qasm = circuit.run_qasm(c)
    print(qasm)
    print("// %d CX's" % qasm.count("cx "))

    shots = argv.get("shots", 10000)
    samps = send([qasm], shots=shots, error_model=True)
    process(code, samps, circuit)


def process(code, samps, circuit):
    n,k = code.n, code.k
    shots = len(samps)
    if not shots:
        return

    #print(code.longstr())
    css = code.to_css()
    #print(css.longstr())
    #Hz = numpy.concatenate((css.Hz, css.Lz))
    Hz, Lz, Tx = css.Hz, css.Lz, css.Tx
    #print("Hz:")
    #print(Hz)

    idxs = circuit.labels # final qubit permutation
    idxs = list(reversed(idxs))

    Hz = Hz[:, idxs]
    Lz = Lz[:, idxs]
    Tx = Tx[:, idxs]
    #print(Hz)

    #print(samps)
    #fail = 0
    count = 0
    lookup = {}
    for v in samps:
        v = parse(v)
        #check = dot2(v, Hz.transpose())
        #syndrome = check[:,:-k]
        #err = check[:, -k:]
        syndrome = dot2(v, Hz.transpose())
        #print(v, syndrome)
        v = (v + dot2(syndrome, Tx)) % 2 # base error correction
        err = dot2(v, Lz.transpose()) # logical operator
        #print(v, syndrome, err)
        key = "%s %s"%(syndrome, err)
        key = str(syndrome), str(err)
        lookup[key] = lookup.get(key, 0) + 1
        if syndrome.sum():
            count += 1
        #if err.sum() and not syndrome.sum():
        #    fail += 1
    keys = list(lookup.keys())
    keys.sort()

    if argv.train:
        errs = list(set(e for (s,e) in keys))
        errs.sort()
        print(errs)
        syns = list(set(s for (s,e) in keys))
        syns.sort()
        print(syns)
        table = {s:[0]*len(errs) for s in syns}
        for (s,e) in keys:
            count = lookup[s,e]
            table[s][errs.index(e)] += count
            print('\t', s,e, count)
        print(table)
        decode = {}
        for s in syns:
            row = table[s]
            idx = row.index(max(row))
            decode[s] = errs[idx]
        print(decode)
        f = open(argv.train, 'w')
        print(decode, file=f)
        f.close()
        print("decode saved as %s"%argv.train) 
        return # <------------------------------------- return

    elif argv.load:
        print("loading decode at %s"%(argv.load,))
        decode = open(argv.load).read()
        decode = eval(decode)
        #print(decode)
        #print(lookup)
        errors = 0
        fails = 0
        for (s,e) in keys:
            count = lookup[s, e]
            if decode.get(s) is None:
                fails += 1
                errors += count
            elif decode.get(s) != e:
                errors += count
        
    else:
        print("WARNING: training decoder on test data")
        fails = 0
        bags = {s:[] for (s,e) in keys}
        for (s,e) in keys:
            count = lookup[s,e]
            bags[s].append(count)
            #print('\t', s,e, count)
        errors = 0
        for counts in bags.values():
            counts.remove(max(counts))
            for c in counts:
                errors += c
        print(str(bags)[:100]+" ...")

    print("shots:", shots)
    print("errors:", errors)
    if fails:
        print("fails:", fails) 
    p = errors/shots
    print("success:  %.6f" % (1-p))
    print("variance: %.6f" %variance(p, shots))




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

