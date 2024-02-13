#!/usr/bin/env python

from functools import reduce, lru_cache
cache = lru_cache(maxsize=None)
from operator import add, mul, matmul

from qumba.solve import kernel, dot2, normal_form, enum2
from qumba.clifford_sage import Clifford, green, red, I, r2, half
from qumba.qcode import QCode, strop
from qumba.construct import all_codes
from qumba.unwrap import unwrap
from qumba.argv import argv

from bruhat.algebraic import qchoose_2


def find_css(n, mx, mz):
    if mx < mz:
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
        

def main_cz():
    for n in [3,4]:
      print("n=%d"%n, end=" ", flush=True)
      for m in range(1,n+1):
      #for m in [1]:
        k = n-m
        count = 0
        found = 0
        for code in all_codes(n, k, 0):
            dode = unwrap(code, True)
            #print(strop(dode.H))
            #print()
            #assert has_ZX_duality(dode)
            #if accept_code_cz(dode):
            count += 1
            if fast_accept_cz(dode):
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





