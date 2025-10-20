#!/usr/bin/env python

from qumba.action import mulclose
from qumba.symplectic import SymplecticSpace
from qumba.qcode import QCode
from qumba.argv import argv

def main():

    n = argv.get("n", 3)

    space = SymplecticSpace(n)
    S = space.S
    H = space.H
    I = space.get_identity()
    CX = space.CX
    CZ = space.CZ

    P = I
    for i in range(n):
        P = P*H(i)
    print(P)

    gen  = [S(i) for i in range(n)]
    gen += [H(i) for i in range(n)]

    LC = mulclose(gen)
    print("|LC| =", len(LC))

    orbits = []
    found = []
    for E in [
        I, 
        CZ(0,1),
        CZ(0,2), CZ(1,2), 
        CZ(0,1) * CZ(1,2),
        CZ(0,2) * CZ(1,2),
    ]:
        orbit = []
        overlap = 0
        src = QCode.from_encoder(E)
        for g in LC:
            code = g*src
            for dode in orbit:
                if code.is_equiv(dode):
                    break
            else:
                orbit.append(code)
                for dode in found:
                    if dode.is_equiv(code):
                        print("*", end="")
                        overlap += 1
                        break
                else:
                    found.append(code)
        if overlap:
            print((overlap))
        print("orbit =", len(orbit))
        orbits.append(orbit)

    print(sum(len(o) for o in orbits))
    #print(54 + 3*18 + 27)
    print(len(found))



if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))


