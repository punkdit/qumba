#!/usr/bin/env python

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul

import numpy

from qumba.lin import (
    kernel, dot2, normal_form, enum2, shortstr, zeros2, solve,
    linear_independent, row_reduce)
from qumba.qcode import QCode, strop, Matrix, SymplecticSpace
from qumba.construct import all_codes
from qumba.unwrap import unwrap
from qumba.argv import argv
from qumba import construct

from bruhat.algebraic import qchoose_2


def find_css(n, mx, mz):
    if mx < mz:
        assert 0, "um... reverse this?"
        return find_css(n, mz, mx) # recurse
    found = set()
    for Hx in qchoose_2(n, mx):
        Jz = kernel(Hx)
        assert mx + len(Jz) == n
        assert dot2(Hx, Jz.transpose()).sum() == 0
        for Kz in qchoose_2(len(Jz), mz):
            Hz = dot2(Kz, Jz)
            Hz = normal_form(Hz)
            assert dot2(Hx, Hz.transpose()).sum() == 0
            key = str((Hx, Hz))
            assert key not in found
            found.add(key)
            yield (Hx, Hz)


@cache
def get_transversal_CZ(n):
    from qumba.clifford import Clifford
    assert n%2 == 0
    c = Clifford(n)
    op = reduce(mul, [c.CZ(2*i, 2*i+1) for i in range(n//2)])
    return op


@cache
def get_transversal_HHSwap(n):
    from qumba.clifford import Clifford
    assert n%2 == 0
    c = Clifford(n)
    op = reduce(mul, [c.SWAP(2*i, 2*i+1) for i in range(n//2)])
    op *= reduce(mul, [c.H(i) for i in range(n)])
    return op

def has_ZX_duality(code):
    n = code.n
    assert n%2 == 0
    dode = code
    for i in range(n):
        dode = dode.apply_H(i)
    perm = []
    for i in range(n//2):
        perm.append(2*i+1)
        perm.append(2*i)
    dode = dode.apply_perm(perm)
    return dode.is_equiv(code)


def accept_cz(Hx, Hz):
    code = QCode.build_css(Hx, Hz)
    P = code.get_projector()
    n = code.n
    L2 = get_transversal_HHSwap(n)
    is_hhswap = L2*P == P*L2
    is_zx_dual = has_ZX_duality(code)
    assert is_hhswap==is_zx_dual, (is_hhswap,is_zx_dual)

    even_fibers = None
    for v in enum2(len(Hx)):
        h = dot2(v, Hx)
        h.shape = (n//2, 2)
        i = str(h).count('[1 1]') % 2
        if i%2:
            even_fibers = False
            break
    else:
        even_fibers = True

    L1 = get_transversal_CZ(n)
    is_cz = L1*P == P*L1
    #assert is_cz==is_zx_dual
    assert (is_zx_dual and even_fibers) == is_cz, (even_fibers, is_cz, is_zx_dual)
    if is_cz:
        assert is_zx_dual
    return is_cz


def accept_code_cz(code):
    P = code.get_projector()
    n = code.n
    L1 = get_transversal_CZ(n)
    is_cz = L1*P == P*L1
    return is_cz


@cache
def get_cz(space):
    n = space.n
    g = space.get_identity()
    for i in range(n//2):
        g = g * space.CZ(2*i,2*i+1)
    return g

def fast_accept_cz(code):
    g = get_cz(code.space)
    dode = code.apply(g)
    return dode.is_equiv(code)
    

def accept_hhswap(Hx, Hz):
    code = QCode.build_css(Hx, Hz)
    #P = code.get_projector()
    #n = code.n
    #L2 = get_transversal_HHSwap(n)
    #is_hhswap = L2*P == P*L2
    is_zx_dual = has_ZX_duality(code)
    #assert is_hhswap==is_zx_dual, (is_hhswap,is_zx_dual)
    return is_zx_dual

def accept_hhswap_not_cz(Hx, Hz):
    code = QCode.build_css(Hx, Hz)
    P = code.get_projector()
    n = code.n
    L2 = get_transversal_HHSwap(n)
    is_hhswap = L2*P == P*L2
    is_zx_dual = has_ZX_duality(code)
    assert is_hhswap==is_zx_dual, (is_hhswap,is_zx_dual)

    L1 = get_transversal_CZ(n)
    is_cz = L1*P == P*L1
    return is_zx_dual and not is_cz
    


def main_0():
    for n in [2,4,6]:
        for m in range(1, n+1):
          count = 0
          for mx in range(m+1):
            mz = m-mx
            for Hx,Hz in find_css(n, mx, mz):
                if accept(Hx,Hz):
                    count += 1
          print(count, end=" ", flush=True)
        print()
        
"""
accept_hhswap
    3
    15 15
    63 315 135
== C series pascal

accept_cz:
n=2      2
n=4      9   6
n=6     35 105 30
n=8    135           
           

accept_hhswap_not_cz
    1
    6 9

"""

def main_1():
    accept = argv.get("accept", "accept_cz")
    print("accept=%s"%accept)
    accept = eval(accept)
    ns = argv.get("ns", [2,4])
    for n in ns:
      for m in range(1, n//2+1):
        count = 0
        for Hx,Hz in find_css(n, m, m):
            if accept(Hx,Hz):
                count += 1
        print(count, end=" ", flush=True)
      print()


def no_Y(H):
    #m, nn = H.shape
    #for bits in numpy
    #rows = list(H.rowspan())
    #H1 = [row for row in H.rowspan() if 'Y' not in strop(row)]
    H1 = [row for row in H.rowspan() if strop(row).count("Y")%2==0]
    H1 = reduce(Matrix.concatenate, H1)
    H1 = H1.linear_independent()
    #print(H1)
    return H1
        

def test_cz():
    for n in [2,4,6]:
      print("n=%d"%n, end=" ", flush=True)
      for m in range(n+1):
        k = n-m
        count = 0
        found = 0
        for code in all_codes(n, k, 0):
            count += 1
            #assert code.T is not None
            code.check()
            #print(code.longstr())
            if fast_accept_cz(code):
                found += 1
        print("%s:%s"%(count,found), end=" ", flush=True)
      print()



def main_cz():
    for n in [3,4]:
      print("n=%d"%n, end=" ", flush=True)
      for m in range(1,n+1):
      #for m in [1]:
        k = n-m
        count = 0
        found = 0
        for code in all_codes(n, k, 0):
            count += 1
            dode = unwrap(code, True)
            #print(strop(dode.H))
            #print()
            #assert has_ZX_duality(dode)
            assert fast_accept_cz(dode)
            a = accept_code_cz(dode)
            H1 = no_Y(code.H)
            s,t = len(code.H),len(H1)
            #b = 'Y' in strop(code.H)
            if a and s!=t:
                print()
                print(strop(code.H))
                print("-"*code.n)
                print(strop(H1))
                print("%d!=%d"%(s,t),end=":")
            print("%d%d"%(int(a), int(s==t)), end=" ", flush=True)
            if a:
                found += 1
        print("%s:%s"%(count,found), end=" ", flush=True)
      print()
            


def main_unwrap():
    for n in range(5,6):
      print("n=%d"%n, end=" ", flush=True)
      for m in range(1,n+1):
        k = n-m
        count = 0
        for code in all_codes(n, k, 0):
            dode = unwrap(code, True)
            #print(strop(dode.H))
            #print()
            assert has_ZX_duality(dode)
            if accept_code_cz(dode):
                count += 1
        print(count, end=" ", flush=True)
      print()
            

def distance(H):
    if H.sum(0).min() == 0:
        return 1
    K = kernel(H)
    k, n = K.shape
    w = n
    for bits in enum2(k):
        U = dot2(bits, K)
        w1 = U.sum()
        if w1 and w1 < w:
            w = w1
    return int(w)


def test_hypergraph_product():
    from qumba.csscode import CSSCode
    from qumba import construct
    d = argv.get("d", 4)
    #m, n = 4, 5 # --> [[41,1,4]] ...
    #m, n = 3,5 # --> [[34,4,3]] with weight 4,5 stabilizers
    #m, n = 3,4 # --> [[25,1,4]] with weight 3,5 stabilizers
    #m, n = 2,4
    #m, n = 2,3 # --> [[13,1,3]] with weight 3,4 stabilizers
    #m0, n0, m1, n1 = 3,4, 2,3 # --> [[18,1,3]] wt 2,3,4
    #m0, n0, m1, n1 = 2,4, 2,3 # --> None
    #m0, n0, m1, n1 = 3,4, 2,4 # --> None
    #m0, n0, m1, n1 = 3,4, 3,5 # --> [[29,2,3]] wt 3,4
    #m0, n0, m1, n1 = 2,3, 3,5 # --> [[21,2,3]] wt 3,4,5
    #m0, n0, m1, n1 = 3,5, 3,5 # --> [[34, 4, 3]]
    m0, n0, m1, n1 = 4,5, 4,6 # --> [[46, 2, 4]]

    m0, n0, m1, n1 = 4,5, 4,5 # --> [[41, 1, 5]]

    m0, n0, m1, n1 = 4,6, 4,6 # --> [[52, 4, 4]]
    m0, n0, m1, n1 = 4,6, 4,7 # --> [[58, 6, 4]]
    m0, n0, m1, n1 = 4,7, 4,7 # --> [[65, 9, 4]]
    m0, n0, m1, n1 = 6,7, 6,7 # --> [[65, 9, 4]]

    codes0 = [H for H in qchoose_2(n0, m0) if distance(H) >= d]
    codes1 = [H for H in qchoose_2(n1, m1) if distance(H) >= d]
    print(len(codes0))
    print(len(codes1))
    #return
    print("dz:",max([distance(H) for H in codes0]))
    print("dx:",max([distance(H) for H in codes1]))

    desc = set()
    for H0 in codes0:
      #print(H0)
      #print(H0.sum(0))
      for H1 in codes1:
        Hx, Hz = construct.hypergraph_product(H0, H1.transpose())
        code = CSSCode(Hx=Hx, Hz=Hz)
        print('.', end='', flush=True)
        dx, dz = code.bz_distance()
        if min(dx, dz) < d:
            continue
        key = str(code)
        if key not in desc:
            print()
            print(code, distance(H0), distance(H1))
            #print(code.longstr())
            desc.add(key)


def all_hypergraph_product():
    from qumba.csscode import CSSCode
    from qumba import construct

    codes = []
    for n in range(3, 7):
      for m in range(1, n):
        found = []
        d = 1
        for H in qchoose_2(n, m):
            d1 = distance(H)
            if d1==d:
                found.append(H)
            elif d1 > d:
                found = [H]
                d = d1
        if d < 3:
            continue
        found.sort(key = lambda H:H.sum())
        codes.append(found[0])
        print(n, m, len(found), d)
        print([int(H.sum()) for H in found])
        print()

    count = 0
    for H0 in codes:
      #print(H0)
      #print(H0.sum(0))
      for H1 in codes:
        Hx, Hz = construct.hypergraph_product(H0, H1.transpose())
        dx = distance(H1)
        dz = distance(H0)
        code = CSSCode(Hx=Hx, Hz=Hz, dx=dx, dz=dz)
        if code.k == 0:
            continue
        if code.n < 30:
            dx, dz = code.bz_distance()
            assert dx == distance(H1)
            assert dz == distance(H0)
        print(code)
        count += 1

        if argv.store_db:
            from qumba import db
            code = code.to_qcode(desc="hypergraph_product")
            db.add(code)
    print("found:", count)





@cache
def get_SH(n):
    space = SymplecticSpace(n)
    SH = space.get_SH()
    return SH.t

@cache
def get_H(n):
    space = SymplecticSpace(n)
    H = space.get_H()
    return H.t

def test_doctrine_gf4():
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      print("n=%s"%n, end=" ", flush=True)
      #for k in range(n+1):
      for m in range(n+1):
        k = n-m
        if (n+k)%2:
            print(".", end=" ", flush=True)
            continue
        count = 0
        for code in construct.all_codes(n, k, d):
            #if code.is_gf4():
            H1 = code.H * get_SH(n)
            dode = QCode(H1, check=False)
            if code.is_equiv(dode):
                count += 1
        print(count, end=" ", flush=True)
      print()




def test_doctrine_css():
    # https://oeis.org/A302595
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      print("n=%s"%n, end=" ", flush=True)
      for k in range(n+1):
        count = 0
        found = []
        for code in construct.all_codes(n, k, d):
            if not code.is_css():
                continue
            #assert code.is_css_slow()
            count += 1
            #for dode in found:
            #    assert not dode.is_equiv(code) # yes these are unique
            found.append(code)
        print(count, end=" ", flush=True)
      print()


def test_doctrine_css_ssd():
    A = Matrix([[1,1],[0,0]])
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      at = SymplecticSpace(n).get(A).t
      print("n=%s"%n, end=" ", flush=True)
      for k in range(n+1):
        count = 0
        for code in construct.all_codes(n, k, d):
            if not code.is_css():
                continue
            H = code.H
            J = H*at
            if H.t.solve(J.t) is None:
                continue
            count += 1
        print(count, end=" ", flush=True)
      print()


def test_doctrine_sd():
    A = Matrix([[0,1],[1,0]])
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      at = SymplecticSpace(n).get(A).t
      print("n=%s"%n, end=" ", flush=True)
      for k in range(n+1):
        count = 0
        for code in construct.all_codes(n, k, d):
            H = code.H
            J = H*at
            if H.t.solve(J.t) is None:
                continue
            count += 1
            #print()
            #print(code.longstr())
        print(count, end=" ", flush=True)
      print()

def test_doctrine_sd_scss():
    A = Matrix([[0,1],[1,0]])
    B = Matrix([[1,0],[1,0]])
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      at = SymplecticSpace(n).get(A).t
      bt = SymplecticSpace(n).get(B).t
      print("n=%s"%n, end=" ", flush=True)
      for k in range(n+1):
        count = 0
        for code in construct.all_codes(n, k, d):
            H = code.H
            J = H*at
            if H.t.solve(J.t) is None:
                continue
            J = H*bt
            if H.t.solve(J.t) is None:
                continue
            count += 1
        print(count, end=" ", flush=True)
      print()


def main_sd_scss():
    I = Matrix([[1,0],[0,1]])
    A = Matrix([[0,1],[1,0]])
    B = Matrix([[1,0],[1,0]])
    n = argv.get("n", 5)
    k = argv.get("k", 1)
    d = argv.get("d", 2)

    at = SymplecticSpace(n).get(A).t
    bt = SymplecticSpace(n).get(B).t
    print("search: [[%d, %d, %d]]"%(n, k, d))
    for code in construct.all_codes(n, k, d):
        H = code.H
        J = H*at
        if H.t.solve(J.t) is None:
            continue
        J = H*bt
        if H.t.solve(J.t) is None:
            continue

        dode = code.apply_H()
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        if L==I:
            continue

        print(code)
        print(code.longstr())


def parse_sd_scss():
    f = open("codes_612.out").readlines()
    I = Matrix([[1,0],[0,1]])
    
    rows = None
    for line in f:
        line = line.strip()
        if line.startswith("H ="):
            assert rows is None
            rows = []
        elif "=" in line:
            H = ' '.join(rows)
            rows = None
            code = QCode.fromstr(H)
            dode = code.apply_H()
            assert dode.is_equiv(code)
            L = dode.get_logical(code)
            if L==I:
                continue
            print(H)
            #print()
        elif rows is not None:
            rows.append(line)

def proc_612():
    # these are all the [[6,1,2]]'s with non-trivial & transversal H
    codes = """
    Y.ZXZX ZXZXY. ..XXXX ZZ..ZZ ZZZZ..
    XZZXY. .YZXZX ..XXXX ZZ..ZZ ZZZZ..
    YZ.XZX .X.XXX ZZXXY. Z.Z.ZZ ZZZZ..
    X..XXX ZY.XZX ZZXXY. .ZZ.ZZ ZZZZ..
    X..XXX ZXZXY. Z.YXZX .ZZ.ZZ ZZZZ..
    XZZXY. .X.XXX .ZYXZX Z.Z.ZZ ZZZZ..
    Y.ZZXX ZXZY.X ..XXXX ZZZ..Z ZZ.ZZ.
    XZZY.X .YZZXX ..XXXX ZZZ..Z ZZ.ZZ.
    YZ.ZXX .X.XXX ZZXY.X ZZZ..Z Z.ZZZ.
    X..XXX ZY.ZXX ZZXY.X ZZZ..Z .ZZZZ.
    X..XXX ZXZY.X Z.YZXX ZZZ..Z .ZZZZ.
    XZZY.X .X.XXX .ZYZXX ZZZ..Z Z.ZZZ.
    Y.ZZXX ZXZYX. ..XXXX ZZ.Z.Z ZZZ.Z.
    XZZYX. .YZZXX ..XXXX ZZ.Z.Z ZZZ.Z.
    YZ.ZXX .X.XXX ZZXYX. Z.ZZ.Z ZZZ.Z.
    X..XXX ZY.ZXX ZZXYX. .ZZZ.Z ZZZ.Z.
    X..XXX ZXZYX. Z.YZXX .ZZZ.Z ZZZ.Z.
    XZZYX. .X.XXX .ZYZXX Z.ZZ.Z ZZZ.Z.
    YZZ.XX ZYZXX. ZZYX.X ZZ.Z.Z Z.ZZZ.
    YZZXX. ZYZ.XX ZZYX.X ZZ.Z.Z .ZZZZ.
    YZZ.XX ZYZX.X ZZYXX. Z.ZZ.Z ZZ.ZZ.
    YZZX.X ZYZ.XX ZZYXX. .ZZZ.Z ZZ.ZZ.
    YZZXX. ZYZX.X ZZY.XX Z.ZZ.Z .ZZZZ.
    YZZX.X ZYZXX. ZZY.XX .ZZZ.Z Z.ZZZ.
    YZXZX. ZYXZ.X ZZ.YXX Z.ZZ.Z .ZZZZ.
    YZXZ.X ZYXZX. ZZ.YXX .ZZZ.Z Z.ZZZ.
    X.X.XX ZXYZ.X Z.ZYXX ZZ.Z.Z .ZZZZ.
    XZYZ.X .XX.XX .ZZYXX ZZ.Z.Z Z.ZZZ.
    X.X.XX ZXYZX. Z.ZYXX .ZZZ.Z ZZ.ZZ.
    XZYZX. .XX.XX .ZZYXX Z.ZZ.Z ZZ.ZZ.
    """.strip().split("\n")
    
    tgt = QCode.fromstr("XXXXII IIXXXX ZZZZII IIZZZZ IYIYIY")
    N, perms = tgt.get_autos()
    for H in codes:
        code = QCode.fromstr(H)
        print(code, "*" if code.is_equiv(tgt) else "") # the second one
        #print(strop(code.H))
        dode = code.apply_H()
        assert dode.is_equiv(code)
        L = dode.get_logical(code)
        print(L)
    assert len(codes) == 30
    assert N==24
    # 30 == 720 / 24 so these are all the same code!



def test_qchoose_2():
    from bruhat.algebraic import qchoose_2
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      for k in range(n+1):
        count = len(list(qchoose_2(n,n-k)))
        print(count, end=" ", flush=True)
      print()


def all_css_mxz(n, mx, mz):
    from bruhat.algebraic import qchoose_2
    for Hx in qchoose_2(n, mx):
        assert len(Hx) == mx
        K = kernel(Hx)
        for Mz in qchoose_2(K.shape[0], mz):
            #print(Mz)
            Hz = dot2(Mz, K)
            assert len(Hz) == mz
            #print(Hz)
            #assert dot2(Hx, Hz.transpose()).sum() == 0
            code = QCode.build_css(Hx, Hz)
            assert code.k == n-mx-mz
            yield code

@cache
def all_qchoose_2(n, m):
    from bruhat.algebraic import qchoose_2
    return list(qchoose_2(n,m))

def choose_css(n, mx, mz):
    Hxs = all_qchoose_2(n, mx)
    Hzs = all_qchoose_2(n, mz)
    #print("all_css")
    for Hx in Hxs:
      for Hz in Hzs:
        if dot2(Hx, Hz.transpose()).sum() == 0:
            code = QCode.build_css(Hx, Hz)
            yield code

def all_css(n, m):
    codes = []
    for mx in range(m+1):
        mz = m-mx
        assert mx+mz == m, (mx,mz,n)
        #codes += list(choose_css(n,mx,mz))
        codes += list(all_css_mxz(n,mx,mz))
    return codes


def test_all_css():
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      print("n=%s"%n, end=" ")
      for k in range(n+1):
        m = n-k
        codes = all_css(n,m)
        count = len(codes)
        print(count, end=" ", flush=True)
      print()
            


def test_doctrine_css_sd(): # Note: test_sd is much much faster !
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      print("n=%s"%n, end=" ")
      for k in range(n+1):
        if (n+k)%2:
            print(".", end=" ", flush=True)
            continue
        count = 0
        for code in construct.all_codes(n, k, d):
            if not code.is_css():
                continue
            H1 = code.H * get_H(n)
            dode = QCode(H1, check=False)
            if code.is_equiv(dode):
                count += 1
        print(count, end=" ", flush=True)
      print()


def test_doctrine_csssd(): # Note: test_sd is much faster !
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      print("n=%s"%n, end=" ")
      for k in range(n+1):
        if (n+k)%2:
            print(".", end=" ", flush=True)
            continue
        count = 0
        for code in all_css(n,n-k):
            H1 = code.H * get_H(n)
            dode = QCode(H1, check=False)
            if code.is_equiv(dode):
                count += 1
                #print()
                #print(strop(code.H))
        print(count, end=" ", flush=True)
      print()


def test_sd():
    from bruhat.algebraic import qchoose_2
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 5)):
      print("n=%s"%n, end=" ")
      for k in range(n+1):
        if (n+k)%2:
            print(".", end=" ", flush=True)
            continue
        m = (n-k)//2
        count = 0
        for H in qchoose_2(n,m):
            assert H.shape == (m,n)
            if dot2(H, H.transpose()).sum() == 0:
                code = QCode.build_css(H, H)
                assert code.n == n
                assert code.k == k, (str(code), n, k)
                code.get_distance()
                #if code.k and code.d>1:
                #    print(code)
                count += 1
        print(count, end=" ", flush=True)
      print()


def test_classical_sd():
    from bruhat.algebraic import qchoose_2
    #for m in range(1, 6):
    for m in [6]:
        n = 2*m
        count = 0
        u = zeros2(1, n)
        u[:] = 1
        L = u[:, :-1]
        found = set()
        for H in qchoose_2(n,m):
            assert H.shape == (m,n)
            if dot2(H, H.transpose()).sum() != 0:
                continue
            count += 1
            #U = solve(H.transpose(), u.transpose())
            #assert U is not None
            H = numpy.concatenate((u, H))
            H = row_reduce(H)
            assert H.shape == (m, n)
            H = H[1:, 1:]
            assert dot2(H, H.transpose()).sum() == 0
            code = QCode.build_css(H, H, None, None, L, L)
            w = Matrix(H).get_wenum()
            if w in found:
                continue
            found.add(w)
            print(shortstr(H), code, w)
            print()

        print((m,n), count)


def test_bijection():
    from qumba.lin import zeros2

    n = argv.get("n", 3)
    k = argv.get("k", 0)
    nn = 2*n

    count = 0
    found = set()
    for code in construct.all_codes(n, k, 1):
        count += 1
        H = code.H
        #print()
        #print(H)
        J = zeros2(n, nn+1)
        J[:, :nn] = H
        M = Matrix(J)
        #print("-"*(nn+1))
        #print(M)
        #print("-"*(nn+1))
        MMt = M*M.t
        #print(MMt)
        found.add(MMt)

        for i in range(n):
            if MMt[i,i]:
                J[i,nn] = 1
        M = Matrix(J)
        #print("-"*(nn+1))
        #print(M)
        #print("-"*(nn+1))
        MMt = M*M.t
        #print(MMt)

    print(count)
    print("MMt's:", len(found))
    for op in found:
        #print(op)
        assert op==op.t
        #print()





def test_selfdual():
    from bruhat.algebraic import qchoose_2
    d = 0
    for n in range(argv.get("n0", 1), argv.get("n1", 10)):
      print("n=%s"%n, end=" ")
      for k in [1]:
        if (n+k)%2:
            continue
        m = (n-k)//2
        count = 0
        for H in qchoose_2(n,m):
            assert H.shape == (m,n)
            if dot2(H, H.transpose()).sum() == 0:
                code = QCode.build_css(H, H)
                assert code.n == n
                assert code.k == k, (str(code), n, k)
                code.get_distance()
                #if code.k and code.d>1:
                #    print(code)
                count += 1
        print(count, end=" ", flush=True)
      print()


def all_sd(n,m):
    from bruhat.algebraic import qchoose_2
    count = 0
    for H in qchoose_2(n,m):
        if dot2(H, H.transpose()).sum() == 0:
            yield H


def is_triorthogonal(H):
    # check triorthogonal
    m, n = H.shape
    for i in range(m):
     for j in range(i+1, m):
        #if (H[i]*H[j]).sum() % 2:
        #    return False
        Hij = H[i]*H[j]
        #if Hij.sum()==0:
        #    return True
        for k in range(j+1, m):
          if (Hij*H[k]).sum() % 2:
            return False
    return True


def test_triorthogonal():
    for n in range(argv.get("n0", 1), argv.get("n1", 7)):
      print("n=%s"%n, end=" ")
      for m in range(0,n,1):
        Hs = list(all_sd(n,m))
        count = 0
        for H in all_sd(n,m):
            if is_triorthogonal(H):
                count += 1
        print("%4s"%(count or "."), end=" ", flush=True)
      print()
        

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



