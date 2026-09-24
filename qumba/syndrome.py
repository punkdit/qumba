#!/usr/bin/env python
r"""
Search and build bare syndrome extraction sequences.

For an X-type stabilizer:

 - The ancilla is reset to |0> then an H gate prepares it in |+>
 - The ancilla is the control of every CX.
 - Each data qubit in the stabilizer support is the target.
 - Finally, another H rotates the ancilla back before
    a Z-basis measurement—equivalent to measuring the ancilla in the X basis.

Conceptually:

 ancilla: |0⟩ ─ H ──●──●──●──●── H ─ M
                    │  │  │  │
 data:              X  X  X  X

For comparison, a Z-type stabilizer prepares the ancilla
in (|0>), uses each data qubit as the CX control
and the ancilla as the target, then measures the ancilla directly in Z.

"""



from random import shuffle, randint
from operator import add, matmul, mul
import operator 
from functools import reduce

import numpy

from qumba.qcode import QCode, SymplecticSpace, strop
from qumba.csscode import CSSCode, distance_z3_css
from qumba import construct 
from qumba.syntax import Syntax
from qumba.util import choose, all_perms
from qumba.argv import argv

from qumba.matrix import Matrix
#from qumba.umatrix import UMatrix, Solver, If, Not, And, Or, PbLe

def apply_CX(css, ctrl, tgt):
    code = css.to_qcode()
    dode = code.CX(ctrl, tgt)
    css = dode.to_css()
    distance_z3_css(css)
    return css


def find_sequence(code):
    code = code.to_css()
    dx, dz = distance_z3_css(code)

    print("find_sequence:", code)
    print(code.longstr())
    print()

    Hx = code.Hx
    Hz = code.Hz
    Lx = code.Lx
    Lz = code.Lz
    mx, n = Hx.shape
    mz, _ = Hz.shape
    k = code.k

    logicals = []
    for i in range(k):
        jdxs = tuple(j for j in range(n) if Lx[i,j])
        logicals.append(("X", jdxs))
        jdxs = tuple(j for j in range(n) if Lz[i,j])
        logicals.append(("Z", jdxs))

    plus = CSSCode(Hx=Matrix([[1]]), Hz=Matrix.zeros((0,1)))
    zero = CSSCode(Hz=Matrix([[1]]), Hx=Matrix.zeros((0,1)))

    #print(target, distance_z3_css(target))

    checks = []


    for (basis, H, ancilla, move, accept) in [
        ("X", Hx, plus, lambda code, j: apply_CX(code,n,j), lambda code : code.dx == dx, ),
        ("Z", Hz, zero, lambda code, j: apply_CX(code,j,n), lambda code : code.dz == dz, ),
    ]:
        m = len(H)
        for i in range(m):
            idxs = [j for j in range(n) if H[i,j]]
            #shuffle(idxs)
            assert len(idxs) <= 6, "um.."
            perms = list(all_perms(idxs))
            shuffle(perms)
            for jdxs in perms:
                dode = code+ancilla
                for j in jdxs:
                    dode = move(dode, j)
                    if not accept(dode):
                        print(jdxs, "skip")
                        break
                else:
                    assert accept(dode)
                    print("found:", jdxs)
                    checks.append((basis, tuple(jdxs)))
                    break
    return checks, logicals


    basis = "X"
    ctrl = n

    for i in range(mx):
        idxs = [j for j in range(n) if Hx[i,j]]
        #shuffle(idxs)
        assert len(idxs) <= 6, "um.."
        perms = list(all_perms(idxs))
        shuffle(perms)
        for jdxs in perms:
            dode = code+plus
            for j in jdxs:
                dode = apply_CX(dode, ctrl, j)
                if dode.dx < dx:
                    print(jdxs, "skip")
                    break
            else:
                print("found:", jdxs)
                checks.append((basis, tuple(jdxs)))
                break

    return

    basis = "Z"
    tgt = n
    for i in range(mz):
        dode = code+zero
        jdxs = [j for j in range(n) if Hz[i,j]]
        #shuffle(jdxs)
        for j in jdxs:
            #print(ctrl, j, "-->")
            dode = apply_CX(dode, j, tgt)
            assert dode.dz == dz
            print("\t", dode)
        print()
        checks.append((basis, tuple(jdxs)))

    return checks, logicals


def main():
    param = argv.get("param", (15,5,3))
    print("param:", param)
    #code = construct.get_10_2_3()
    #code = construct.get_bring()
    #code = construct.get_css((15,5,3))
    #code = construct.get_css((48,18,4))
    #code = construct.get_css((80,18,5))
    code = construct.get_css(param)
    checks, logicals = find_sequence(code)
    print("checks =", tuple(checks))
    print("logicals =", tuple(logicals))




if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "main"
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




