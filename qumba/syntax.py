#!/usr/bin/env python


class Term(object):
    def __init__(self, atoms):
        self.atoms = list(atoms)

    def __str__(self):
        atoms = self.atoms
        return "*".join("%s(%s)"%((name,)+arg) for (name,arg) in atoms)

    def __mul__(self, other):
        atoms = self.atoms
        if isinstance(other, Term):
            atoms = atoms + other.atoms
            return Term(atoms)

        for (name, arg) in reversed(atoms):
            meth = getattr(other, name)


class Atom(object):
    def __init__(self, name):
        assert type(name) is str
        self.name = name

    def __call__(self, *arg):
        return Term([(self.name, arg)])


class Syntax(object):
    def __getattr__(self, name):
        return Atom(name)


def test():
    s = Syntax()
    X, Z, Y = s.X, s.Z, s.Y
    print( X(0)*Z(2) )



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


