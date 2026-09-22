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


def get(
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


def test():
    circ = get()


def main():

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




