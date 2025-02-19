#!/usr/bin/env python

from random import shuffle

from qumba.qcode import QCode, strop
from qumba.clifford import Clifford
from qumba.argv import argv


def gate_search(space, gates, target):
    gates = list(gates)
    A = space.get_identity()
    found = []
    while A != target:
        if len(found) > 100:
            return
        count = (A+target).sum()
        print(count, end=' ')
        shuffle(gates)
        for g in gates:
            B = g*A # right to left
            if (B+target).sum() <= count:
                break
        else:
            print("X ", end='', flush=True)
            return None
        found.insert(0, g)
        A = B
    print("^", end='')
    return found



def test():
    code = QCode.fromstr("""
    XXXXII
    IIXXXX
    ZZZZII
    IIZZZZ
    IYIYIY
    ZIZIZI
    """)

    _code = QCode.fromstr("""
    XXXXII
    IIXXXX
    ZZZZII
    IIZZZZ
    IXIXIX
    ZIZIZI
    """)

    print(code)

    n = code.n
    E = code.get_encoder()
    space = code.space
    print(space.get_name(E))

    return

    CX, CZ, H, S = space.CX, space.CZ, space.H, space.S

    gates = [H(i) for i in range(n)]
    gates += [CX(i,j) for i in range(n) for j in range(n) if i!=j]

    found = gate_search(space, gates, E)


def test_clifford():

    n = code.n
    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z

    ops = strop(code.H)
    print(ops)
    hs = []
    for op in ops.split():
        h = c.get_pauli(op)
        hs.append(h)

    print("hj==jh")
    for h in hs:
      for j in hs:
        assert h*j == j*h

    


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

