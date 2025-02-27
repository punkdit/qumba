#!/usr/bin/env python

from qumba.syntax import Syntax
from qumba.symplectic import SymplecticSpace
from qumba.qcode import QCode


def test():

    s = Syntax()
    h, cx = s.H, s.CX

    n = 2
    space = SymplecticSpace(n)
    prog = cx(0,1)
    U = prog*space
    print(QCode.from_encoder(U).longstr())
    #return


    prog = (cx(6,4)*cx(1,5)*cx(3,6)*cx(2,0)
        *cx(1,4)*cx(2,6)*cx(3,5)*cx(1,0)
        *h(1)*h(2)*h(3))
    #prog.atoms = list(reversed(prog.atoms))

    n = 7
    space = SymplecticSpace(n)

    print(prog)

    E = prog*space
    print(E)
    
    E = (~E).t
    code = QCode.from_encoder(E, k=0)

    print(code.longstr())

    code = QCode.fromstr("""
    ZZ..ZZ.
    Z.Z.Z.Z
    ...ZZZZ
    XX..XX.
    X.X.X.X
    ...XXXX
    """)

    print(code)
    print(code.longstr())





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


