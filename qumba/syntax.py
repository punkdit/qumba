#!/usr/bin/env python



class Term(object):
    def __init__(self, atoms):
        self.atoms = list(atoms)

    def __str__(self):
        atoms = self.atoms
        s = "*".join("%s%s"%((name,)+(arg,)) for (name,arg) in atoms)
        s = s.replace(',)', ')')
        s = s.replace(' ', '')
        return s

    @property
    def name(self):
        atoms = self.atoms
        return tuple("%s%s"%((name,)+(arg,)) for (name,arg) in atoms)

    def __mul__(self, item):
        from qumba.qcode import QCode
        atoms = self.atoms
        if isinstance(item, Term):
            atoms = atoms + item.atoms
            return Term(atoms)

        if isinstance(item, QCode):
            return (self * item.space) * item

        if hasattr(item, "target") and item.target is not None:
            # this is so we can hit operators... good idea?
            op = item # we already are an op
            target = op.target # and this is it's target
        else:
            # start here
            target = item
            op = target.get_identity()

        #print("Term.__mul__")
        #print("\top =", str(op).replace("\n", " "), type(op))
        for (name, arg) in reversed(atoms):
            #print("\ttarget =", target)
            #print("\t%s(%s)"%(name, arg))
            meth = getattr(target, name, None)
            if meth is None:
                meth = getattr(target, "get_"+name)
            opa = meth(*arg)
            op = opa * op
            if hasattr(op, "target") and op.target is not None:
                # just hack this XXX
                target = op.target
                #print("\tnew target =", op.target)
        #print("\ttarget =", target)
        #print("\tfini")
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
    from qumba.symplectic import SymplecticSpace
    from qumba.clifford import Clifford
    from qumba.lagrel import Module

    s = Syntax()
    X, Z, Y = s.X, s.Z, s.Y
    S, H, CX, MX, PX = s.S, s.H, s.CX, s.MX, s.PX
    prog = X(0)*Z(2)
    assert str(prog) == "X(0)*Z(2)", str(prog)

    n = 3
    space = Clifford(n)
    M = prog*space

    prog = CX(0, 1)
    (prog*space)
    (prog*SymplecticSpace(n))
    (prog*Module(n))

    prog = (CX(6,7)*CX(5,7)*CX(0,7)*CX(6,4)
        *CX(1,5)*CX(3,6)*CX(2,0)
        *CX(1,4)*CX(2,6)*CX(3,5)*CX(1,0))

    op = prog * Module(8)

    mod = Module(5)
    op = MX(0) * mod
    assert op.source == Module(5)
    assert op.target == Module(4)

    op = PX(7) * mod
    assert op.target == Module(6)
    




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


