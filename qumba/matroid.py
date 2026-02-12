#!/usr/bin/env python

from functools import cache

import numpy

from bruhat.matroid import Matroid, SpMatroid, mask_le
from bruhat.action import mulclose, get_orbits

from qumba import construct 
from qumba.symplectic import SymplecticSpace
from qumba.qcode import QCode
from qumba import db
from qumba.argv import argv
from qumba.matrix import Matrix
from qumba.lax import lc_orbits, Lower, Upper
from qumba.util import all_subsets



@cache
def all_sp_masks(nn): # zip order
    n = nn//2
    masks = []
    for mask in numpy.ndindex((2,)*nn):
        for i in range(n):
            if mask[2*i] and mask[2*i+1]:
                break
        else:
            masks.append(mask)
    return masks

def get_matroid(code):
    H = code.H
    m, nn = H.shape
    masks = []
    skip = []
    count = 0
    for mask in all_sp_masks(nn):
        count += 1
        #print("\t", mask)
        idxs = [i for i in range(nn) if mask[i]]
        #H1 = H[:, idxs].transpose()
        H1 = H[:, idxs].t
        #H1 = row_reduce_p(H1, q, True)
        H1 = H1.row_reduce()
        if len(H1) == len(idxs):
        #if Matrix(H1,q).t.rank() == len(idxs):
            masks.append(mask)
        else:
            skip.append((idxs,mask))
    print("all_sp_masks:", count, "masks:", len(masks), "diff:", count-len(masks))
    #print(masks)
    M = SpMatroid(nn, masks)
    #M.check()
    for idxs,mask in skip:
        print("skip:", idxs, M.restrict(mask).rank)
    return M



def test_signature():

    n = 5
    k = 1
    idxs = tuple(range(n))
    idxss = [tuple(A) for A in all_subsets(idxs)]
    assert len(idxss) == 2**n
    count = 0
    for code in construct.all_codes(n, k):
        code.build()
        count += 1
        H = code.H
        F = Upper(H)
        sig = F.signature()
        print(code, sig)
        lookup = {A:F(A) for A in idxss}
        for A in idxss:
            assert 0<=lookup[A]
            #assert lookup[A]<=len(A), "%s > len(%s)"%(lookup[A], A)
            for B in idxss:
                join = tuple(i for i in idxs if i in A or i in B)
                meet = tuple(i for i in idxs if i in A and i in B)
                lhs = lookup[join] + lookup[meet] 
                rhs = lookup[A] + lookup[B]
                #print(A, B, join, meet)
                #print("\t", lhs, "<=", rhs)
                assert lhs<=rhs
                if meet == A:
                    assert lookup[A] <= lookup[B]
                
    print("count:", count)


def main():

    code = construct.get_412()

    print(code)
    print(code.longstr())

    n = code.n
    nn = 2*n
    print(code.H)

    M = get_matroid(code)
    print(M, M.rank, len(M.get_basis()))


def get_avoid(code):
    H = code.H
    L = code.L
    n = code.n
    d = code.d
    k = len(L)
    m = len(H)
    avoid = []
    for idx in numpy.ndindex((2,)*k):
        if sum(idx)==0:
            continue
        l = numpy.dot(idx, L.A) % 2
        for jdx in numpy.ndindex((2,)*m):
            h = numpy.dot(jdx, H.A) % 2
            lh = (l+h)%2
            lh.shape = n,2
            #s = str(lh.reshape(n,2)).replace("\n", "")
            s = str(lh).replace("\n", "")
            w = s.count("0 1") + s.count("1 0") + s.count("1 1")
            if w > d:
                continue
            idxs = [i for i in range(n) if str(lh[i]).count("1")]
            #print(s, w, idxs)
            avoid.append(idxs)
    return idxs





def test():
    #code = construct.get_513()
    from qumba.unwrap import get_avoid
    code = construct.get_10_2_3()
    #code = construct.get_surface(3,3)
    #code = construct.get_913()
    code = construct.toric(3,3)

    assert code.is_css()
    code = code.to_css()
    code.bz_distance()
    print(code)
    dual = code.get_dual()
    dual.bz_distance()

    avoid = get_avoid(code) 
    idxs = [set(numpy.where(a)[0]) for a in avoid]
    avoid += get_avoid(dual)
    idxs = [set(numpy.where(a)[0]) for a in avoid]
    print(idxs)

    n = code.n
    accept = []
    for i in range(n):
      for j in range(i+1,n):
        for ii in idxs:
            if i in ii and j in ii:
                break
        else:
            accept.append((i,j))
    print(accept)


#def get_orbits(gen, found, verbose=False):
#    remain = set(found)
#    orbits = []
#    while remain:
#        c = remain.pop()
#        orbit = [c]
#        bdy = list(orbit)
#        while bdy:
#            _bdy = []
#            for g in gen:
#                for c in bdy:
#                    d = g*c
#                    if d in remain:
#                        orbit.append(d)
#                        remain.remove(d)
#                        _bdy.append(d)
#            bdy = _bdy
#        orbits.append(orbit)
#        if verbose:
#            print("[%d]"%len(orbit), end='', flush=True)
#    if verbose:
#        print()
#    counts = [len(o) for o in orbits]
#    #print(counts, sum(counts))
#    assert sum(counts) == len(found)
#    return orbits


def test_orbit():
    n = argv.get("n",4)
    k = argv.get("k",1)
    d = argv.get("d",2)

    found = []
    for code in construct.all_codes(n,k,d):
        #print(code, end=' ', flush=True)
        print('.', end='', flush=True)
        found.append(code)
    print()
    print("found:", len(found))

    space = SymplecticSpace(n)
    S, H = space.S, space.H

    gen = []
    for i in range(n):
        gen.append(S(i))
        gen.append(H(i))

    G = mulclose(gen)
    print("|G| =", len(G))

    orbits = get_orbits(gen, found, True)


def detect_classical(H):
    m, n = H.shape

    masks = list(numpy.ndindex((2,)*n))
    found = set()
    found.add( (0,)*n )
    for mask in masks:
        v = Matrix(mask)
        Hv = H*v
        if Hv.sum(): # non-zero syndrome
            found.add(mask)

    for a in masks:
        if a in found:
            continue
        for b in list(found):
            if mask_le(n, a, b):
                #print(a, "<=", b)
                found.remove(b)
    
    #print(found)
    M = Matroid(n, found)
    M.check()

    #print()
    return M


def s_correct_classical(H):
    #print("s_correct_classical")
    #print(H)
    m, n = H.shape

    K = H.kernel()
    #print("K =")
    #print(K)
    C = list(K.span())
    #print("C =", C)

    masks = list(numpy.ndindex((2,)*n))
    found = []
    for mask in masks:
        # this error is correctable if it is
        # the unique error of minimum weight in its coset.
        error = Matrix(mask)
        vC = [error+u for u in C] # <-- the coset
        vC.sort(key = sum)
        #print(error, vC)
        w = vC[0].sum()
        items = [v for v in vC if v.sum() == w]
        if len(items) != 1:
            continue
        u = items[0]
        if u==error:
            #print("\tcorrectable", mask)
            found.append(mask)

    for a in masks:
        if a in found:
            continue
        for b in list(found):
            if mask_le(n, a, b):
                #print(a, "<=", b)
                found.remove(b)
    
    #print(found)
    M = Matroid(n, found)
    #M.check() # FAIL !!

    #print()
    return M



def correct_classical(H):
    m, n = H.shape
    masks = list(numpy.ndindex((2,)*n))
    syns = [Matrix(u) for u in numpy.ndindex((2,)*m)]

    lookup = {u:[] for u in syns}
    for mask in masks:
        v = Matrix(mask)
        Hv = H*v
        lookup[Hv].append(mask)

    found = []
    #found.append( (0,)*n )
    for Hv,items in lookup.items():
        #assert len(items)
        if not len(items):
            continue
        items.sort(key = sum)
        #print(Hv, "-->", items)
        w = sum(items[0])
        items = [v for v in items if sum(v) == w]
        #print(Hv, items)
        if len(items)!=1:
            continue
        v = items[0]
        found.append(v)
        #print("\tcorrectable:", v)
    #print(found)

    for a in masks:
        if a in found:
            continue
        for b in list(found):
            if mask_le(n, a, b):
                #print(a, "<=", b)
                found.remove(b)
    
    #print(found)
    M = Matroid(n, found)
    #M.check() # FAIL

    #print()
    return M

def tpl_le(lhs, rhs):
    for (i,j) in zip(lhs, rhs):
        if i>j:
            return False
    return True

def erase_classical(H): # XXX what is this ?!?
    print("erase_classical")
    print(H)
    m, n = H.shape

    masks = list(numpy.ndindex((2,)*n))
    syns = [Matrix(u) for u in numpy.ndindex((2,)*m)]
    print(syns)

#    lookup = {(erase,u):[] for u in syns for erase in masks}
#    for erase in masks:
#        for mask in masks:
#            if not tpl_le(mask, erase):
#                continue
#            v = Matrix(mask)
#            Hv = H*v
#            key = erase,Hv
#            lookup[key].append(mask)

    lookup = {erase:{u:[] for u in syns} for erase in masks}
    for erase in masks:
        for mask in masks:
            if not tpl_le(mask, erase):
                continue
            v = Matrix(mask)
            Hv = H*v
            key = erase,Hv
            lookup[erase][Hv].append(mask)

    #return

    found = []
    #found.append( (0,)*n )
    #for key,items in lookup.items():
    for erase in masks:
        send = lookup[erase]
        for Hv,items in send.items():
            print(erase, Hv, "->", items)
            #assert len(items)
            if not len(items):
                continue
            items.sort(key = sum)
            #print(Hv, items)
            w = sum(items[0])
            items = [v for v in items if sum(v) == w]
            #print(Hv, items)
            if len(items)!=1:
                break
        else:
            print("\terase:", erase)
            found.append(erase)

    for a in masks:
        if a in found:
            continue
        for b in list(found):
            if mask_le(n, a, b):
                #print(a, "<=", b)
                found.remove(b)
    
    #print(found)
    M = Matroid(n, found)
    M.check()

    #print()
    return M


def test_classical():
    from bruhat.algebraic import qchoose_2

    H = Matrix([[1,1,1,1]])
    M = detect_classical(H)
    assert M == Matroid.uniform(4,1)

    M = correct_classical(H)
    assert M == Matroid.uniform(4,0)

    M = s_correct_classical(H)
    assert M == Matroid.uniform(4,0)

    #M = erase_classical(H)
    #assert M == Matroid.uniform(4,1)

    #return

    found = set()
    for H in qchoose_2(4, 2):
        H = Matrix(H)
        M = detect_classical(H)
        if M in found:
            continue
        found.add(M)
    assert len(found) == 35 # missing one non-F2-representable matroid

    found = set()
    for H in qchoose_2(4, 2):
        H = Matrix(H)
        M = correct_classical(H)
        s_M = s_correct_classical(H)
        assert M==s_M, M
        if M in found:
            continue
        #print(M)
        found.add(M)
    assert len(found) == 21

    n = 6
    m = 4
    #for m in range(1, n):
      #print("qchoose_2(%d, %d)"%(n, m))
    for H in qchoose_2(n, m):
        H = Matrix(H)
        M = correct_classical(H)
        s_M = s_correct_classical(H)
        assert M==s_M, M
        try:
            M.check()
        except AssertionError:
            print(H)
            print(H.latex())
            K = H.kernel()
            print("G =")
            print(K)
            print(K.latex())
            masks = list(M.masks)
            masks.sort()
            for mask in masks:
                print(mask)
            print(detect_classical(H))
            raise

#    found = set()
#    for H in qchoose_2(4, 2):
#        H = Matrix(H)
#        M = erase_classical(H)
#        if M in found:
#            continue
#        #print(M)
#        found.add(M)
#    assert len(found) == 14, len(found)
#
#    return

    m = 4
    n = 8
    masks = list(numpy.ndindex((2,)*n))
    for trial in range(10):
        H = Matrix.rand(m, n)
        if H.rank() != m:
            continue
        M0 = detect_classical(H)
        M1 = correct_classical(H)
        M2 = s_correct_classical(H)
        assert M1==M2
        M1.check()
        assert M1.less_equal(M0)
        f0 = M0.rankfunc()
        f1 = M1.rankfunc()
        f2 = M1.get_dual().rankfunc()

        #for mask in masks:
        #    nask = tuple(1-i for i in mask)
        #    print(f0[mask]-f2[nask], end=' ')
        #print()

        r0, r1 = (M0.rank, M1.rank)

    return

    for n in range(2,8):
      for m in range(1,n+1):
        found = set()
        for H in qchoose_2(n, m):
            H = Matrix(H)
            #M = detect_classical(H)
            M = correct_classical(H)
            if M in found:
                continue
            #print(M)
            found.add(M)
        print(len(found), end=' ', flush=True)
      print()





class Mask(tuple):
    def __new__(cls, *items):
        ob = tuple.__new__(cls, items)
        return ob

    def __str__(self):
        return "Mask%s"%(tuple.__str__(self),)

    def __add__(self, other):
        assert len(self) == len(other)
        n = len(self)
        items = [self[i]+other[i] for i in range(n)]
        return Mask(*items)


def test_find():

    m = Mask(1, 0, 0)
    m = m +  Mask(0, 1, 0)

    lookup = {(1,1,0):5}
    assert lookup[m] == 5
    
    n = 10
    accept = lambda m:sum(m)<3
    items = list(find_masks(n, accept))
    assert len(items) == 1 + 10 + 5*9, len(items)


def find_masks(n, accept = lambda m:True):
    root = Mask(*[0]*n)

    gen = []
    for i in range(n):
        items = [0]*n
        items[i] = 1
        gen.append(Mask(*items))

    children = {}

    bdy = []
    item = accept(root)
    if item is not None:
        yield item
        bdy.append(root)
    found = set(bdy)
    weight = 0
    while bdy:
        _bdy = []
        for m in bdy:
          for i in range(n):
            if m[i]:
                continue
            m1 = m + gen[i]
            if m1 in found:
                continue
            found.add(m1)
            item = accept(m1)
            if item is not None:
                yield item
                _bdy.append(m1)
                #children.setdefault(m1, []). # ?!?
        bdy = _bdy
        


def build_detectable(code):
    H = code.H
    m, nn = H.shape
    n = nn//2
    #print(H)

    masks = list(numpy.ndindex((2,)*n))
    lookup = {bits:set() for bits in masks}
    for bits in numpy.ndindex((2,)*nn):
        v = Matrix(bits)
        v2 = v.reshape(n,2)
        w = v2.sum(1)
        mask = tuple(min(1,int(wi)) for wi in w)
        Hv = H*v
        lookup[mask].add(str(Hv))

    detectable = [(0,)*n]
    for mask in masks:
        vals = lookup[mask]
        if '.'*m in vals:
            pass
            #print("   ", end='')
        else:
            #print(" + ", end='')
            detectable.append(mask)
        #print(mask, ' '.join(lookup[mask]))

    found = set(detectable)
    for a in masks:
        if a in found:
            continue
        for b in list(found):
            if mask_le(n, a, b):
                #print(a, "<=", b)
                found.remove(b)
    
    M = Matroid(n, found)
    M.check()

    #print()
    return M


def find_detectable(code):
    H = code.H
    m, nn = H.shape
    n = nn//2
    #print(H)

    def accept(mask):
        if sum(mask) == 0:
            return (0,)*n
        v = Matrix(mask)
        v2 = v.reshape(n,2)
        w = v2.sum(1)
        mask = tuple(min(1,int(wi)) for wi in w)
        Hv = H*v
        if Hv.sum()==0:
            print("reject:", mask)
            return 
        return mask

    masks = []
    for mask in find_masks(nn, accept):
        print(mask)
        masks.append(mask)
    
    M = Matroid(n, masks)
    M.check()

    return M


def test_detect():
    #code = construct.get_713()

    n = argv.get("n", 3)
    k = argv.get("k", 1)
    d = argv.get("d", 1)
    found = set()
    for code in construct.all_codes(n,k,d):
        M = build_detectable(code)
        M1 = find_detectable(code)
        assert M==M1
        if M not in found:
            code.build()
            found.add(M)
            #print(code, end=' ')
            print("%d:%d"%(M.rank,code.d), end=' ', flush=True)
    print()
    print(len(found))


def test_distance():
    code = QCode.fromstr("""
    XZIZIXIZZI
    IYIZZYIIZZ
    IIYZZYIXII
    IZZYIYZIIZ
    IZIIXYZYII
    IZZIZZXXZZ
    IIZZZZZIXZ
    IZZZZIIZZX
    ZZZZZZIIII
    """) # rank 4

    code = construct.get_713() # rank 3
    code = construct.get_513() # rank 2

    for code in db.get_codes():
    
        print(code)
        M = build_detectable(code)
        #print(M)
        print("rank:", M.rank)


if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    print("%s()"%fn)

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))



