#!/usr/bin/env python

"""

"""

from pathlib import Path


import numpy 

import stim

from qsweeper.circuits import build_css_bare_syndrome
from qsweeper.sweep import run_sweep


from qumba.matrix import Matrix
from qumba.csscode import CSSCode
from qumba.argv import argv

#def get_25_1_5():


def surface47(
    basis: str = "Z",
    rounds: int = 5,
    p: float = 0.001,
    inject: tuple[int, int, str] | None = None,
):
    checks = (
        # X checks: top boundary, bulk plaquettes, then bottom boundary.
        ("X", (0, 1)),
        ("X", (2, 3)),
        ("X", (1, 2, 6, 7)),
        ("X", (3, 4, 8, 9)),
        ("X", (5, 6, 10, 11)),
        ("X", (7, 8, 12, 13)),
        ("X", (11, 12, 16, 17)),
        ("X", (13, 14, 18, 19)),
        ("X", (15, 16, 20, 21)),
        ("X", (17, 18, 22, 23)),
        ("X", (21, 22)),
        ("X", (23, 24)),
        # Z checks: left boundary, bulk plaquettes, then right boundary.
        ("Z", (5, 10)),
        ("Z", (15, 20)),
        ("Z", (0, 5, 1, 6)),
        ("Z", (2, 7, 3, 8)),
        ("Z", (6, 11, 7, 12)),
        ("Z", (8, 13, 9, 14)),
        ("Z", (10, 15, 11, 16)),
        ("Z", (12, 17, 13, 18)),
        ("Z", (16, 21, 17, 22)),
        ("Z", (18, 23, 19, 24)),
        ("Z", (4, 9)),
        ("Z", (14, 19)),
    )
    logical = (
        ("X", (0, 5, 10, 15, 20)),
        ("Z", (0, 1, 2, 3, 4)),
    )

    n = 25
    xstab = [idxs for (c,idxs) in checks if c=="X"]
    mx = len(xstab)
    Hx = numpy.zeros((mx, n), dtype=int)
    for i,idxs in enumerate(xstab):
        Hx[i,idxs] = 1
    Hx = Matrix(Hx)
    #print(Hx)

    zstab = [idxs for (c,idxs) in checks if c=="Z"]
    mz = len(zstab)
    Hz = numpy.zeros((mz, n), dtype=int)
    for i,idxs in enumerate(zstab):
        Hz[i,idxs] = 1
    Hz = Matrix(Hz)
    #print(Hz)

    Lx = [0]*n
    Lz = [0]*n
    for c,idx in logical:
      for i in idx:
        if c=="X":
            Lx[i] = 1
        if c=="Z":
            Lz[i] = 1
    Lx = Matrix([Lx])
    Lz = Matrix([Lz])

    code = CSSCode(Hx=Hx, Hz=Hz, Lx=Lx, Lz=Lz)
    #code.bz_distance()
    print(code)

    circ = build_css_bare_syndrome(
        checks, logical, n, basis=basis, rounds=rounds,
        p=p, inject=inject,)
    return circ


def get_10_2_3(
    basis: str = "Z",
    rounds: int = 3,
    p: float = 0.001,
    inject: tuple[int, int, str] | None = None,
):

    n = 10
    checks = [('X', (0, 3, 6, 7)), ('X', (1, 4, 7, 8)), ('X',
    (0, 2, 8, 9)), ('X', (1, 3, 5, 9)), ('Z', (2, 3, 6, 9)),
    ('Z', (3, 4, 5, 7)), ('Z', (0, 4, 6, 8)), ('Z', (0, 1, 7, 9))]
    logicals = (('X', (0, 1, 2, 3, 4)), ('Z', (0, 1, 2, 3, 4)), ('X', (2, 3, 5)), ('Z', (1, 4, 5)))

    circ = build_css_bare_syndrome(
        checks, logicals, n, basis=basis, rounds=rounds,
        p=p, inject=inject,)
    return circ

def get_30_8_3(
    basis: str = "Z",
    rounds: int = 3,
    p: float = 0.001,
    inject: tuple[int, int, str] | None = None,
):

    n = 30
    checks = (('X', (6, 13, 21, 23, 27)), ('X', (1, 11, 12,
    15, 27)), ('X', (1, 6, 9, 14, 16)), ('X', (4, 7, 12,
    21, 29)), ('X', (5, 8, 10, 18, 19)), ('X', (3, 8, 14,
    15, 22)), ('X', (9, 13, 17, 24, 25)), ('X', (0, 2, 10,
    26, 29)), ('X', (2, 4, 11, 18, 22)), ('X', (7, 23, 24,
    26, 28)), ('X', (3, 5, 16, 20, 25)), ('Z', (6, 16, 20,
    23, 28)), ('Z', (0, 13, 17, 21, 29)), ('Z', (2, 11, 23,
    26, 27)), ('Z', (5, 10, 24, 25, 26)), ('Z', (4, 6, 14,
    21, 22)), ('Z', (8, 10, 12, 15, 29)), ('Z', (1, 7, 9,
    12, 24)), ('Z', (4, 7, 18, 19, 28)), ('Z', (8, 9, 14,
    17, 19)), ('Z', (1, 5, 11, 16, 18)), ('Z', (3, 13, 15, 25, 27)))
    logicals = (('X', (17, 19, 20, 21, 22, 28)), ('Z', (12,
    13, 14, 16, 18, 19, 20, 22, 24, 26, 27, 29)), ('X', (14,
    16, 18, 19, 20, 22, 25, 26, 27)), ('Z', (10, 13, 14,
    16, 17, 18, 20, 21, 22, 29)), ('X', (15, 20, 21, 22,
    23, 25, 26, 29)), ('Z', (13, 23, 24)), ('X', (8, 16,
    18, 19, 20, 21, 22, 25, 26, 27, 29)), ('Z', (13, 14,
    15, 16, 25, 27)), ('X', (12, 20, 21, 22, 23, 24, 26,
    29)), ('Z', (12, 14, 15, 16, 17, 25, 26, 28, 29)), ('X',
    (8, 19, 20, 21, 22, 25, 26, 27, 28, 29)), ('Z', (12,
    14, 15, 16, 24, 25, 26, 29)), ('X', (13, 20, 21, 22,
    23, 27)), ('Z', (10, 12, 13, 14, 15, 16, 17, 19, 20,
    23, 28, 29)), ('X', (25, 26, 27)), ('Z', (10, 13, 17,
    19, 23, 26)))
    
    circ = build_css_bare_syndrome(
        checks, logicals, n, basis=basis, rounds=rounds,
        p=p, inject=inject,)
    return circ




def test():
    circ = get()


def main():

    get = get_10_2_3
    get = get_30_8_3

    def circuit_factory(label: str, p: float) -> stim.Circuit:
        basis = {"memory_Z": "Z", "memory_X": "X"}[label]
        return get(basis=basis, p=p)

    shots = argv.get("shots", 100)

    run_sweep(
        circuit_factory,
        labels=("memory_Z", "memory_X"),
        display_names={"memory_Z": "Z memory", "memory_X": "X memory"},
        #p_values=numpy.geomspace(args.p_max, args.p_min, args.points),
        p_values=numpy.geomspace(0.01, 0.002, 5),
        #p_values = [2e-3], 
        shots=shots, K=512, Delta=12,
        decoder="frontier",
        engine="auto",
        #bp_max_iter=args.bp_max_iter,
        seed=1234,
        output=Path("output/"),
        title=" hi ",
        #metadata={"circuit": args.circuit, 
        # "distance": spec.distance, "rounds": rounds},
    )






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




