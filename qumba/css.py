#!/usr/bin/env python

from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.lin import identity2, zeros2
from qumba.util import cache
from qumba import csscode, qcode
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




