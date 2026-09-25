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


def apply_CX(css, ctrl, tgt):
    code = css.to_qcode()
    dode = code.CX(ctrl, tgt)
    css = dode.to_css()
    distance_z3_css(css)
    return css


def search(code, h, ancilla, move, accept):
    n = code.n
    idxs = [j for j in range(n) if h[j]]
    #shuffle(idxs)
    assert len(idxs) <= 8, "um.."
    perms = list(all_perms(idxs))
    shuffle(perms)
    for jdxs in perms:
        print(jdxs, end=' ', flush=True)
        dode = code+ancilla
        for i,j in enumerate(jdxs):
            dode = move(dode, j)
            if not accept(dode):
                print("skip", i)
                break
        else:
            assert accept(dode)
            print("found")
            return jdxs
    print("fail\n")


plus = CSSCode(Hx=Matrix([[1]]), Hz=Matrix.zeros((0,1)))
zero = CSSCode(Hz=Matrix([[1]]), Hx=Matrix.zeros((0,1)))

def search_X(code, h):
    dx, dz = distance_z3_css(code)
    (basis, ancilla, move, accept) = ("X", plus, 
            lambda code, j: apply_CX(code,code.n-1,j), 
            lambda code : code.dx == dx)
    jdxs = search(code, h, ancilla, move, accept)
    return jdxs

def search_Z(code, h):
    dx, dz = distance_z3_css(code)
    (basis, ancilla, move, accept) = ("Z", zero, 
            lambda code, j: apply_CX(code,j,code.n-1), 
            lambda code : code.dz == dz)
    jdxs = search(code, h, ancilla, move, accept)
    return jdxs


def find_sequence(code):
    code = code.to_css()
    dx, dz = distance_z3_css(code)

    print("find_sequence:", code)
    print()

    Hx = code.Hx
    Hz = code.Hz
    Lx = code.Lx
    Lz = code.Lz
    mx, n = Hx.shape
    mz, _ = Hz.shape
    k = code.k

    weight = argv.weight

    #print(target, distance_z3_css(target))

    checks = []
    metachecks = []
    for (basis, H, ancilla, move, accept) in [
        ("X", Hx, plus, 
            lambda code, j: apply_CX(code,n,j), 
            lambda code : code.dx == dx, ),
        ("Z", Hz, zero, 
            lambda code, j: apply_CX(code,j,n), 
            lambda code : code.dz == dz, ),
    ]:
        print(H.get_wenum())

        hs = []
        if weight is not None:
            for v in H.span():
                if v.sum() == weight:
                    hs.append(v)
        else:
            m = len(H)
            for i in range(m):
                h = H[i, :]
                hs.append(h)
        H = Matrix(hs)
        print(H.shape, H.rank())
        K = H.t.kernel()
        print("K =")
        print(K, K.shape)
        idxs = []
        for h in hs:
            jdxs = search(code, h, ancilla, move, accept)
            if jdxs is None:
                assert 0
            idxs.append(len(checks))
            checks.append((basis, tuple(jdxs)))
        N = len(idxs)
        for row in K:
            meta = [idxs[j] for j in range(N) if row[j]]
            metachecks.append(meta)

    logicals = []
    for i in range(k):
        jdxs = tuple(j for j in range(n) if Lx[i,j])
        logicals.append(("X", jdxs))
        jdxs = tuple(j for j in range(n) if Lz[i,j])
        logicals.append(("Z", jdxs))

    return checks, logicals, metachecks


def test_913():
    H = Matrix.parse("""
    11...1111
    1.11.111.
    11.1111..
    1..11.111
    """)
    m, n = H.shape

    code = CSSCode(Hx=H, Hz=H)
    distance_z3_css(code)
    print(code)

    if 0:
        checks, logicals, metachecks = find_sequence(code)
        print("checks =", tuple(checks))
        print("logicals =", tuple(logicals))
        print("metachecks =", tuple(metachecks))

    checks = []
    verify = [
        (5,0,7,1,6,8),
        (6,7,2,3,0,5),
        (3,0,4,1,6,5),
        (6,3,4,8,0,7)
    ]

    dx, dz = distance_z3_css(code)
    for (basis, ancilla, move, accept) in [
        ("Z", zero, 
            lambda code, j: apply_CX(code,j,code.n-1), 
            lambda code : code.dz == dz),
        ("X", plus, 
            lambda code, j: apply_CX(code,code.n-1,j), 
            lambda code : code.dx == dx)]:
        for i,jdxs in enumerate(verify):
            print(basis, jdxs)
            dode = code+ancilla
            for i,j in enumerate(jdxs):
                dode = move(dode, j)
                assert accept(dode)
            print("OK")
            checks.append((basis, jdxs))

    logicals = []
    for i in range(code.k):
        jdxs = tuple(j for j in range(code.n) if code.Lx[i,j])
        logicals.append(("X", jdxs))
        jdxs = tuple(j for j in range(code.n) if code.Lz[i,j])
        logicals.append(("Z", jdxs))

    print("checks =", tuple(checks))
    print("logicals =", tuple(logicals))


    

def main():
    param = argv.get("param", (15,5,3))
    print("param:", param)

    if argv.colour:
        d = argv.get("d", 3)
        code = construct.get_colour_666(d)
        code = code.to_css()
        H = code.Hx
        print(H.get_wenum())
        found = []
        for v in H.span():
            w = v.sum()
            if w==6:
                found.append(v)

        shuffle(found)
        for h in found:
            jdxs = search_X(code, h)
            #jdxs = search_Z(code, h)
            if jdxs is not None:
                return
        return

    elif param == (10,2,3):
        code = construct.get_10_2_3()
    else:
        code = construct.get_css(param)

    code = code.to_css()

    print(code)

    if argv.selfdual:
        H = code.Hx
        m, n = H.shape
        #J = Matrix.zeros((0, n))
        #print(J.shape)
        rows = []
        for v in H.span():
            w = v.sum()
            print(v, w)
            if 0 < w <= 4:
                #J = J.concatenate(v)
                rows.append(v)
    
        J = Matrix(rows)
        J = J.linear_independent()
    
        print(J)
        code = CSSCode(Hx=J, Hz=J)

    checks, logicals, metachecks = find_sequence(code)
    print("checks =", tuple(checks))
    print("logicals =", tuple(logicals))
    print("metachecks =", tuple(metachecks))




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




