#!/usr/bin/env python

from functools import reduce
from operator import lshift

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.lin import identity2, zeros2
from qumba.util import cache
from qumba import csscode, qcode, construct
from qumba.smap import SMap


class Space:
    def __init__(self, n):
        self.n = n

    @cache
    def cnot(self, i, j):
        assert i!=j
        n = self.n
        A = identity2(n)
        A[i,j] = 1
        A = Matrix(A)
        return A


class CSS:
    def __init__(self, Hx, Hz):
        Hx = Matrix.promote(Hx)
        Hz = Matrix.promote(Hz)
        mx, n = Hx.shape
        assert Hz.shape[1] == n
        mz, _ = Hz.shape
        k = n - mx - mz
        self.Hx = Hx
        self.Hz = Hz
        self.n = n
        self.k = k
        self.mx = mx
        self.mz = mz
        code = csscode.CSSCode(Hx=Hx.A, Hz=Hz.A)
        dx, dz = code.bz_distance()
        self.d = min(dx, dz)

    def __str__(self):
        return "[[%s, %s, %s]]"%(self.n, self.k, self.d)

    def longstr(self):
        smap = SMap()
        n = self.n
        smap[0,0] = str(self)
        smap[1,0] = str(self.Hx)
        smap[1,n+1] = str(self.Hz)
        return str(smap)

    def __eq__(self, other):
        return self.Hx==other.Hx and self.Hz==other.Hz

    def __hash__(self):
        return hash((self.Hx, self.Hz))

    @classmethod
    def fromstr(cls, Hs):
        code = qcode.QCode.fromstr(Hs)
        code = code.to_css()
        return CSS(code.Hx, code.Hz)

    #def cnot(self, i, j):

    def __mul__(self, M):
        n = self.n
        assert M.shape == (n,n)
        Hx = self.Hx*M
        Hz = self.Hz*M.t
        return CSS(Hx, Hz)

    def normal_form(self):
        Hx, Hz = self.Hx, self.Hz
        Hx = Hx.normal_form()
        Hz = Hz.normal_form()
        return CSS(Hx, Hz)


def main():
    code = CSS.fromstr("""
    X..X..XX..
    .X..X..XX.
    X.X.....XX
    .X.X.X...X
    ..ZZ..Z..Z
    ...ZZZ.Z..
    Z...Z.Z.Z.
    ZZ.....Z.Z
    """)
    code = code.normal_form()
    print(code)
    n = code.n
    cnot = Space(n).cnot


    gen = []
    for i in range(n):
      for j in range(n):
        if i==j:
            continue
        gen.append(cnot(i,j))

    found = set([code])
    bdy = list(found)
    #send = set([code.longstr()])
    while bdy:
        print(len(bdy), end=",", flush=True)
        _bdy = []
        for g in gen:
            for A in bdy:
                B = A*g
                B = B.normal_form()
                if B not in found and B.d>=3:
                    _bdy.append(B)
                    found.add(B)
                    #s = B.longstr()
                    #assert s not in send
                    #send.add(s)
        bdy = _bdy
    print()
    print(len(found))
    found = list(found)
    found.sort(key=str)
    #for g in found:
    #    print(g, g.shape)



    


def main_cnot():
    n, m = argv.get("n", 3), argv.get("m", 2)
    
    A = zeros2(m, n)
    for i in range(m):
        A[i, n-m+i] = 1
    A = Matrix(A)

    gen = []
    for i in range(n):
      for j in range(n):
        if i<=j:
            continue
        g = identity2(n)
        g[i,j] = 1
        g = Matrix(g)
        gen.append(g)

    found = set([A])
    bdy = list(found)

    while bdy:
        #print(bin(len(bdy)), end=",", flush=True)
        s = bin(len(bdy))
        s = s.replace("0b","")
        s = s.replace("0",".")
        print(s.rjust(30))
        _bdy = []
        for g in gen:
            for A in bdy:
                B = A*g
                B = B.normal_form()
                if B not in found:
                    _bdy.append(B)
                    found.add(B)
        bdy = _bdy
    print()
    print(len(found))
    found = list(found)
    found.sort(key=str)
    #for g in found:
    #    print(g, g.shape)


def main_wenum():
    code = qcode.QCode.fromstr("""
    XIIIIIIIIIIIIIXIIXXIIIIIIXXIIIIIXIIX
    IXIIIIIIIIXIXIIIIIIXIIIIIXXIXIXIIIII
    IIXIXIIXIIIIIIIIIIIIXXXIIXIIIIIIIIXI
    IIIXIXIIIIIIIXIIIIIIIXIXIIIIXIIXIIIX
    IIXIXIIIXIIIIIIIIIIIXIXIIIXIIIIXIXII
    IIIXIXIIIIIIIIIXIIXXIXIXIIIIIIIIIXII
    IIIIIIXIIIIIIXIIXIXXIIIIXIIIIIIXIIXI
    IIXIIIIXIXIIIIIIIIIIXIIXXXIXIIIIIIII
    IIIIXIIIXIIXIIIIIIIIIIXXXIXIIXIIIIII
    IXIIIIIIIIXIIIXIIIIXXIIIIIIIXXIIXIII
    IXIIIIIIIIIIXIIIIXIXIIXIIIIXIIXIIIIX
    IIIXIIXIIIIIIXIIIIIIIXIIXIIIIIXXXIII
    ZIIIIZZIIIIIIIZIIZZIIIIIIIIIIIIIZIIZ
    IZIIIZZIIIZIZIIIIIIZIIIIIIIIZIZIIIII
    IIZIZIIZIIZIIIIIIZIIZIZIIZIIIIIIIIII
    IIZZIZIIIIIZIZIIIIIIIZIZIIIIIIIZIIII
    IIIZIZIZZIIIIIIZIIIIIZIZIIIIIIIIIZII
    IIIIIIZZZIIIIZIIZIIIIIIIZIIIIIIZIIZI
    ZZZIIIIZIZIIIIIIIIIIZIIIIZIZIIIIIIII
    ZZIIZIIIZIIZIIIIIIIIIIZIIIZIIZIIIIII
    IIIIIIIZIZIZZIZIIIIIIIIIIZIZIZIIIIII
    IZIZIIIIIIZIIIZIZIIZIIIIIIIIZIIIZIII
    IZIIIIIIIIIIZZIZIZIZIIIIIIIIIIZIIIIZ
    IIIZZIZIIZIIIZIIIIIIIZIIZIIIIIIZIIII
    """) # [[36, 12, 4]]

    code = qcode.QCode.fromstr("""
XIIIIXIIXIXIIIIIIIXIIIIIIIXXIIIIIIXI
IXIXIIIIIIIIIIXIIXIXXIIIIXIIIIIIIIIX
IIXIIIXIIIIXIIIIIXIIXIIIXXIIIIIIXIII
IIIXIIIIIXIIIXIIXIIIIXIIXIIIIIIIIIXX
IIIIXIIIIIXXIIIIXIXIIIXIIIIXXIIIIIII
XIIIXXIIIIIIIIIIIXXIIXIXIIIIIIXIIIII
IIXIIXXIIIIIXIIIIIIXIIIIXIIIIIXXIIII
IIIIIIXXIIXIIXIIIIIIIIIIIXIIIXIXXIII
XIIIIIIXXIIIIIXIIIIIIIXIIIXIIIIIXXII
IIIIIIIIXXIIXIIXIIIIIIIXIIIXIIIIIXXI
IIXIXIIXIIXIIIIIIIIIIIIIIXXIXIIIIXII
XIIIIIXIIXIIXIIIIIIIIXIIIIIXIIXIIIIX
IIIXIIIXIIIIXXIIIIIXXXIIIIIIIIIXIIII
IXIIIIIIXIIIIXXIIIIXIIIIIIIIXXIIXIII
IIIIIIIIIXIXIIXXIIXIIIXIIIIIIXIIIXII
IIIXXIIIIIIIIIIXXIIIIIXXIIIIIIXIIIXI
ZIIIZZIIIIIIIIIZIIZIIIIZIIZIIIZIIIII
IZIIIIZIIIIIIZZIIIIZIIIIIIIIIZIIZIIZ
IZZIIIIIIIIZIZIIIIIIZIIIZIIIZZIIIIII
IIIZIZIIIIIIZZIIIIIZIZIIIIIIIIIZIIZI
IIIIZIIIZIIIIIIZZIIIIIZZIIIIZIIIIIZI
IIIIIZIIIZIIIIIIZZZIIIIZZIIIIIIIIIIZ
IZZIIIIZIIZIIIIIIIIIIIIIIZZIZIIZIIII
ZIIIIIIIZIZZIIIIIIZIIIIIIIZZIIIIZIII
ZIIIZIIIIZIIZIIIIIIIIZIIIIIZIIZIIZII
IIIIZIIIIIZZIIZIIIZIIIZIIZIIZIIIIIII
IIIIIIIZIIIZIIZZIIIIZIZIIIIIIZIIIZII
IIIIIZZIIIIIZIIIZIIIIIIIZIIZIIZZIIII
IIIIIIZZIIIIIZIIIZIIIZIIIZIIIIIZZIII
IIZIIIIZZIIIIIZIIIIZIIIIIIZIIIIIZZII
IIIIIIIIZZZIIIIZIIIIIIIIIIIZIZIIIZZI
ZIIZIIIIIZIIIIIIZIIIIZZIIIIIIIIIIIZZ
    """)

    code = qcode.QCode.fromstr("""
XIIIXXIIXIIIIIIIIXIIIXIIXIXIIIII
XXIXXIIIIIIIIIIIXIIIXIIXIXIIIIII
IIXXIIIIIIIXXIIIIIIXIXIIIIIIXIIX
IIIXXIIIIIIIXXIIIIXIXIXIIIIIIXII
IIIIXXIIIIIIIXXIIIIXIXIXIIIIIIXI
IIIIIXXIIIIIIIXXIXIIXIXIIIIIIIIX
IIXIIIXXIIIIIIIXXIXIIXIXIIIIIIII
IXXXIIIXIIIIIIIIIXIXIIXIXIIIIIII
IIIIIXXIXXIIIIIIXIIIIIXIIXIXIIII
IIIIIIXXIXXIIIIIIIIIIIIXXIXIXIII
IXIIIIIXIIXXIIIIIXIIIIIIIXIXIXII
XXIIIIIIIIIXXIIIXIIIIIIIIIXIXIXI
XIIIIIIIXIIIXXIIIIIIIIIIXIIXIXIX
IIIIIIIIXXIIIXXIIIXIIIIIIXIIXIXI
IIIIIIIIIXXIIIXXIIIXIIIIIIXIIXIX
IZIIIIZIZIIZIIIIZZIIIIIIIIIZZIII
ZIIIIZIZIIZIIIIIIZIIIIIZIIZZIIII
IIIZIIZIIIIIIZIZIIZIIIZZIIIIIIIZ
IIZIZIIZIIIIIIZIIZZZIIIZIIIIIIII
IZIZIZIIIIIIIIIZZZIZZIIIIIIIIIII
ZIZIZIZIIIIIIIIIZIIIZZIIZIIIIIII
IIIZIZIZZIIIIIIIIIIIIZZIZZIIIIII
IZIIZIZIIZIIIIIIIIIIIIZZIZZIIIII
ZIIIIIIZIZIIZIIIZIIIIIIIZIIIZZII
IZIIIIIIZIZIIZIIIIIIIIIIZZIIIZZI
ZIIIIIIIIZIZIIZIIIIIIIIIIZZIIIZZ
IIIIIIIIZIZIZIIZIIZIIIIIIIZZIIIZ
IIZIIIIIIZIZIZIIIIZZIIIIIIIZZIII
IIIZIIIIIIZIZIZIIIIZZIIIIIIIZZII
IIIIZIIIIIIZIZIZIIIIZZIIIIIIIZZI
    """)

    Hx = code.to_css().Hx
    Hx = Matrix(Hx)
    #print(Hx, Hx.shape)

    w = Hx.get_wenum()
    print(w)

    d = 8
    vs = []
    for u in Hx.rowspan():
        if u.sum() == 8:
            vs.append(u[0])
    H = Matrix(vs)
    print(H.shape)
    print(H.sum(0))
    print(H.sum(1))



def main_tensor():
    H = Matrix([[1,1]])
    Ht = H.t

    m, n = H.shape
    concatenate = Matrix.concatenate
    identity = Matrix.identity
    zeros = Matrix.zeros

    I1 = identity(1)
    I2 = identity(2)

    H0 = concatenate(H@I2@I1, I2@H@I1, I2@I2@Ht)

    H1 = concatenate(
        concatenate(I1@H@I1, H@I1@I1, zeros((1, 8)), axis=1),
        concatenate(I1@I2@Ht, zeros((4,2)), H@I2@I2, axis=1),
        concatenate(zeros((4,2)), I2@I1@Ht, I2@H@I2, axis=1))

    assert (H1*H0).is_zero()

    H2 = concatenate(I1@I1@Ht, I1@H@I2, H@I1@I2, axis=1)
    assert (H2*H1).is_zero()

    Hx = H0.t
    Hz = H1
    css = csscode.CSSCode(Hx=Hx.A, Hz=Hz.A)
    css.bz_distance()
    print(css)

    dual = css.get_dual()
    code = css+css+css
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)

    return

    Hx = H1.t
    Hz = H2
    css = csscode.CSSCode(Hx=Hx.A, Hz=Hz.A)
    css.bz_distance()
    print(css)
    print(css.longstr())


def parse(n, stabs):
    import numpy
    #print(stabs)
    stabs = stabs.replace(" ", "")
    stabs = stabs.replace("\n", "")
    stabs = stabs.split(",")
    x_ops = []
    z_ops = []
    for stab in stabs:
        if "X" in stab:
            s = stab.replace("X", " ")
            ops = x_ops
        elif "Z" in stab:
            s = stab.replace("Z", " ")
            ops = z_ops
        else:
            assert 0
        idxs = s.split(" ")
        idxs = [int(i)-1 for i in idxs if len(i)]
        #print(idxs)
        op = [0]*n
        for i in idxs:
            op[i] = 1
        ops.append(op)
    Hx = numpy.array(x_ops)
    Hz = numpy.array(z_ops)
    #print(Hx)
    css = csscode.CSSCode(Hx=Hx, Hz=Hz)
    css.bz_distance()
    return css
        
    


def test_vasmer():
    # https://arxiv.org/abs/1801.04255
    # Appendix A
    n = 12
    stabs = """
    X5X6X7X8X9X10X11X12,
    X1X3X5, X2X4X7,
    Z6Z9, Z6Z10,
    Z8Z11, Z8Z12,
    Z1Z5Z6, Z2Z6Z7,
    Z4Z7Z8, Z3Z5Z8"""

    C0 = parse(n, stabs)
    print(C0)
    #print(C0.longstr())

    stabs = """
    X1X5X6X9, X2X6X7X10,
    X3X5X8X11, X4X7X8X12,
    Z1Z3Z5, Z1Z2Z6,
    Z2Z4Z7, Z3Z4Z8,
    Z5Z9Z11, Z6Z9Z10,
    Z7Z10Z12"""

    C1 = parse(n, stabs)
    print(C1)
    #print(C1.longstr())

    stabs = """
    X1X2X3X4X5X6X7X8,
    X6X9X10, X8X11X12,
    Z1Z5, Z3Z5,
    Z2Z7, Z4Z7,
    Z5Z8Z11, Z5Z6Z9,
    Z6Z7Z10, Z7Z8Z12"""

    C2 = parse(n, stabs)
    print(C2)
    #print(C2.longstr())

    SymplecticSpace = qcode.SymplecticSpace

    cube = construct.get_832()
    print(cube)

    right = C0+C1+C2
    right = right.to_qcode()
    print(right)

    Er = right.get_encoder()
    #code = qcode.QCode.from_encoder(Er, k=3)


    Er = SymplecticSpace(cube.m * n).get_identity() << Er
    print(Er.shape)
    #print(Er)

    if 0:
        code = qcode.QCode.from_encoder(Er, k=3)
        print(code)
        code = code.to_css()
        code.bz_distance()
        print(code)

    #return

    E = cube.get_encoder()

    El = reduce(lshift, [E]*n)
    print(El.shape)

    idxs = []
    for i in range(n):
      for j in range(cube.m):
        idxs.append(cube.n*i + j)

    N = cube.m*n
    for i in range(cube.k):
      for j in range(n):
        idxs.append(cube.n*j + cube.m + i)

    #print(idxs)

    assert len(set(idxs)) == len(idxs)
    assert set(idxs) == set(range(len(idxs)))

    assert len(idxs)*2 == len(El)
    assert len(idxs) == n * cube.n
    P = SymplecticSpace(n*cube.n).get_perm(idxs).t
    E = El * P * Er
    code = qcode.QCode.from_encoder(E, k=3)
    d = code.distance("z3")
    print(code, d)
    
    #print(code.longstr())

    #return

    code = code.to_css()
    code.bz_distance()
    print(code)

    from qumba.gcolor import dump_transverse
    dump_transverse(code.Hx, code.Lx)

    


if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next()
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
        main()


    t = time() - start_time
    print("finished in %.3f seconds"%t)
    print("OK!\n")




