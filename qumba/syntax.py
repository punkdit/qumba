#!/usr/bin/env python

from qumba.symplectic import SymplecticSpace
from qumba.clifford import Clifford


class Term(object):
    def __init__(self, atoms):
        self.atoms = list(atoms)

    def __str__(self):
        atoms = self.atoms
        return "*".join("%s%s"%((name,)+(arg,)) for (name,arg) in atoms)

    @property
    def name(self):
        atoms = self.atoms
        return tuple("%s%s"%((name,)+(arg,)) for (name,arg) in atoms)

    def __mul__(self, other):
        atoms = self.atoms
        if isinstance(other, Term):
            atoms = atoms + other.atoms
            return Term(atoms)

        op = other.get_identity()
        for (name, arg) in reversed(atoms):
            meth = getattr(other, name)
            op = meth(*arg) * op
        return op


class Atom(object):
    def __init__(self, name):
        assert type(name) is str
        self.name = name

    def __call__(self, *arg):
        return Term([(self.name, arg)])


class Syntax(object):
    def __getattr__(self, name):
        return Atom(name)

    def get_identity(self):
        return Term([])


def test():
    s = Syntax()
    X, Z, Y = s.X, s.Z, s.Y
    S, H, CX = s.S, s.H, s.CX
    prog = X(0)*Z(2)
    assert str(prog) == "X(0)*Z(2)"

    n = 3
    space = Clifford(n)
    M = prog*space

    prog = CX(0, 1)
    print(prog*space)
    print(prog*SymplecticSpace(n))




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


