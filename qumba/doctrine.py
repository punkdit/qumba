#!/usr/bin/env python

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul

from qumba.lin import kernel, dot2, normal_form, enum2, shortstr
from qumba.clifford import Clifford, green, red, I, r2, half
from qumba.qcode import QCode, strop, Matrix
from qumba.construct import all_codes
from qumba.unwrap import unwrap
from qumba.argv import argv

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
    assert n%2 == 0
    c = Clifford(n)
    op = reduce(mul, [c.CZ(2*i, 2*i+1) for i in range(n//2)])
    return op


@cache
def get_transversal_HHSwap(n):
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





