#!/usr/bin/env python
"""
linear algebra over Z/2
"""


import sys
from random import random, randint, shuffle, seed

import numpy
import numpy.random as ra
from numpy import dot, concatenate

from qumba.argv import argv

njit = lambda f:f # TODO

int_scalar = numpy.int64

def array(items):
    return numpy.array(items, dtype=int_scalar)

def zeros(*shape):
    return numpy.zeros(shape, dtype=int_scalar)

def identity(n):
    return numpy.identity(n, dtype=int_scalar)

def xdot(*items, **kw):
    p = kw.get("p", 2)
    idx = 0
    A = items[idx]
    while idx+1 < len(items):
        B = items[idx+1]
        A = numpy.dot(A, B)
        idx += 1
    A = A%p
    return A


def compose(*items, **kw):
    p = kw.get("p", 2)
    items = list(reversed(items))
    A = xdot(*items, p=p)
    return A


def eq(A, B, p):
    AB = (A-B)%p
    return numpy.abs(AB).sum() == 0


def rand(m, n, p):
    A = [randint(0, p-1) for i in range(m*n)]
    A = array(A)
    A.shape = (m,n)
    return A





@njit
def swap_row(A, j, k):
    row = A[j, :].copy()
    A[j, :] = A[k, :]
    A[k, :] = row


@njit
def swap_col(A, j, k):
    col = A[:, j].copy()
    A[:, j] = A[:, k]
    A[:, k] = col




@njit
def row_reduce(H, p, truncate=True, inplace=False, check=False, debug=False):
    """Remove zero rows if truncate=True
    """

    assert type(p) is int
    assert 1<p
    #assert len(H.shape)==2, H.shape
    m, n = H.shape
    orig = H
    if not inplace:
        H = H.copy()

    if m*n==0:
        return H[:0, :] if truncate else H

#    if debug:
#        print("solve:")
#        print("%d rows, %d cols" % (m, n))

    i = 0
    j = 0
    while i < m and j < n:
#        if debug:
#            print("i, j = %d, %d" % (i, j))

        assert i<=j
        if i and check:
            assert H[i:,:j].max() == 0 # XX rm

        # first find a nonzero entry in this col
        for i1 in range(i, m):
            if H[i1, j]:
                break
        else:
            j += 1 # move to the next col
            continue # <----------- continue ------------

        if i != i1:
#            if debug:
#                print("swap", i, i1)
            swap_row(H, i, i1)

        assert H[i, j]
        for i1 in range(i+1, m):
            if H[i1, j]:
#                if debug: 
#                    print("add %s to %s" % (i, i1))
                Hij = H[i,j]
                ri = int(Hij)**(p-2)
                assert (ri*Hij)%p == 1
                r = -H[i1,j] * ri
                H[i1, :] += r*H[i, :]
                H[i1, :] %= p

        assert 0<=H.max()<=(p-1), orig

        i += 1
        j += 1

    if truncate:
        m = H.shape[0]-1
        #print "sum:", m, H[m, :], H[m, :].sum()
        while m>=0 and H[m, :].sum()==0:
            m -= 1
        H = H[:m+1, :]

    return H


def normal_form(A, p, truncate=True):
    "reduced row-echelon form"
    A = row_reduce(A, p, truncate)
    #print(A)
    m, n = A.shape
    j = 0
    for i in range(m):
        while j < n and A[i, j] == 0:
            j += 1
        if j==n:
            break
        r = A[i,j]
        inv = int(r)**(p-2) 
        A[i,:] *= inv
        A[i,:] %= p
        assert A[i,j] == 1
        i0 = i-1
        while i0>=0:
            r = A[i0, j]
            if r!=0:
                A[i0, :] -= r*A[i, :]
                A[i0, :] %= p
            assert A[i0,j] == 0
            i0 -= 1
        j += 1
    #print(A)
    return A


def kernel(A, p, inplace=False, check=False, verbose=False):
    """return a list of vectors that span the nullspace of A
    """

    if check:
        A0 = A.copy() # save

#    U = row_reduce(A, p, inplace=inplace)
    U = normal_form(A, p)

    # We are looking for a basis for the nullspace of A

    m, n = U.shape

    if verbose:
        print("kernel: shape", m, n)

    items = []
    for row in range(m):
        cols = numpy.where(U[row, :])[0]
        if not len(cols):
            break
        col = cols[0]
        items.append((row, col))

    leading = [int(col) for (row, col) in items]
    degeneracy = m - len(leading)

    if verbose:
        print("leading:", leading)
        print("degeneracy:", degeneracy)

    # Look for the free variables
    idxs = []
    row = 0
    col = 0
    while row < m and col < n:
        #print row, col
        if U[row:, col].max() == 0: # XXX optimize this XXX
            #print "*"
            assert U[row:, col].max() == 0, U[row:, col]
            idxs.append(col)
        else:
            #print U[row:, col]
            while row<m and U[row:, col].max():
                row += 1
                #print "row", row
                #if row<m:
                #    print U[row:, col]
        col += 1
    for k in range(col, n):
        idxs.append(k)

    if verbose:
        print("found %d free vars:" % len(idxs), idxs)

    basis = []
    for var in idxs:

        #print "var:", var
        v = numpy.zeros((n,), dtype=int_scalar)
        v[var] = 1
        row = min(var-1, m-1)
        while row>=0:
            u = dot(U[row], v)
            r = u.sum()%p
            if r:
                col = leading[row]
                #print "\trow", row, "leading:", col
                v[col] = p-r
                #print '\t', shortstr(v)
            assert dot(U[row], v).sum()%p==0, row
            row -= 1
        #print '\t', shortstr(v)
        if check:
            assert dot(A0, v).sum()%p == 0, shortstr(v)
        basis.append(v)

    K = numpy.array(basis, dtype=int_scalar)
    if not basis:
        K.shape = (0, A.shape[1])
    else:
        assert K.shape[1] == A.shape[1]

    return K


@njit
def u_inverse(U, p, check=False, verbose=False):
    """invert a row reduced U
    """

    m, n = U.shape

    #items = []
    leading = []
    for row in range(m):
        cols = numpy.where(U[row, :])[0]
        if not len(cols):
            break
        col = cols[0]
        leading.append(col)

    U1 = zeros(n, m)

    #print shortstrx(U, U1)

    # Work backwards
    i = len(leading)-1 # <= m
    while i>=0:

        j = leading[i]
        #print("i=", i, "j=", j)
        r = U[i, j]
        assert r != 0
        rinv = (int(r)**(p-2))%p
        U1[j, i] = rinv

        #print("U1 =")
        #print(U1)

        k = i-1
        while k>=0:
            #print("k =", k)
            #print (U[k,:])
            #print (U1[:,i])
            r = xdot(U[k, :], U1[:, i], p=p)
            if r!=0:
                j = leading[k]
                s = U[k, j]
                sinv = (int(s)**(p-2))%p
                #print "set", j, i
                U1[j, i] = (-r*sinv)%p
            #print shortstr(U1[:,i])
            assert xdot(U[k, :], U1[:, i], p=p) == 0
            k -= 1
        i -= 1

    return U1


@njit
def l_inverse(L, p, check=False, verbose=False):
    """invert L (lower triangular, 1 on diagonal)
    """

    m, n = L.shape
    assert m==n
    L1 = identity(m)

    # Work forwards
    for i in range(m):
        #u = L1[:, i]
        for j in range(i+1, m):
            r = xdot(L[j, :], L1[:, i], p=p)
            if r:
                L1[j, i] = p-r
            assert xdot(L[j, :], L1[:, i], p=p) == 0

    assert eq(xdot(L, L1, p=p), identity(m), p=p)
    return L1


@njit
def plu_reduce(A, p, truncate=False, check=False, verbose=False):
    """
    Solve PLU = A, st. P is permutation, L is lower tri, U is upper tri.
    Remove zero rows from U if truncate=True.
    """

    m, n = A.shape
    P = identity(m)
    L = identity(m)
    U = A.copy()

    assert m>0 and n>0

#    if verbose:
#        print("plu_reduce:")
#        print("%d rows, %d cols" % (m, n))

    i = 0
    j = 0
    while i < m and j < n:
#        if verbose:
#            print("i, j = %d, %d" % (i, j))
#            print("P, L, U:")
#            print(shortstrx(P, L, U))

        assert i<=j
        if i and check:
            assert U[i:,:j].max() == 0

        # first find a nonzero entry in this col
        for i1 in range(i, m):
            if U[i1, j]:
                break
        else:
            j += 1 # move to the next col
            continue # <----------- continue ------------

        if i != i1:
#            if verbose:
#                print("swap", i, i1)
            swap_row(U, i, i1)
            swap_col(P, i, i1)
            swap_col(L, i, i1)
            swap_row(L, i, i1)

        #if check:
        #    A1 = _dot2(P, _dot2(L, U))
        #    assert eq2(A1, A)

        #print()
        #print("i,j =", i, j)
        #print("L =")
        #print(L)
        #print("U =")
        #print(U)

        r = U[i, j]
        assert r != 0
        rinv = int(r)**(p-2)
        #print("rinv =", rinv)
        for i1 in range(i+1, m):
            s = U[i1, j]
            if s == 0:
                continue
            #print("i1 =", i1)
            #print("add %s to %s" % (i, i1))
            t = (-s*rinv)%p
            L[i1, i] = p-t
            U[i1, :] += t*U[i, :]
            U[i1, :] %= p
            #print("U =")
            #print(U)
            assert U[i1, j] == 0

        if check:
            A1 = xdot(P, L, U, p=p)
            assert eq(A1, A, p=p)

        i += 1
        j += 1

    if truncate:
        m = U.shape[0]-1
        #print "sum:", m, U[m, :], U[m, :].sum()
        while m>=0 and U[m, :].sum()==0:
            m -= 1
        U = U[:m+1, :]

    return P, L, U



def pseudo_inverse(A, p, check=False):
    m, n = A.shape
    if m*n == 0:
        A1 = zeros((n, m), p)
        return A1
    P, L, U = plu_reduce(A, p, verbose=False, check=check)
    L1 = l_inverse(L, p, check=check)
    U1 = u_inverse(U, p, check=check)
    A1 = xdot(U1, L1, P.transpose(), p=p)
    return A1



def projector(A, p, check=False):

    """
        Find universal projector that kills the columns of A,
        ie. PP=P and PA = 0, st. given any other Q with
        QQ=Q and QA=0, then there exists R st. Q=RP.
    """

    """
        Alternatively
        Find universal projector that preserves the columns of A,
        ie. PP=P and PA=A, st. given any other Q with
        QQ=Q and QA=A, then there exists R st. P=RQ.
    """

    m, n = A.shape

    P = identity(m) - xdot(A, pseudo_inverse(A, p), p=p)
    P %= p

    return P




def pushout(J, K, p, J1=None, K1=None, check=False):
    """
    Return JJ,KK given J and K in the following diagram:

       J
    A ---> B
    |      |
    | K    | JJ
    v      v
    C ---> B+C/~
       KK

    if also given J1:B->T and K1:C->T (st. J1*J=K1*K)
    return unique arrow F : B+C/~ --> T (st. F*JJ=J1 and F*KK=K1).
    """
    assert J.shape[1] == K.shape[1]
    assert type(p) is int
    assert 1<p

    b, c = J.shape[0], K.shape[0]
    JJ = zeros(b+c, b)
    JJ[:b] = identity(b)

    KK = zeros(b+c, c)
    KK[b:] = identity(c)

    kern = (compose(J, JJ, p=p) - compose(K, KK, p=p))%p
    # We need to kill the columns of kern
    R = projector(kern, p, check=check)
    R = row_reduce(R, p, truncate=True, check=check)

    assert eq(compose(J, JJ, R, p=p), compose(K, KK, R, p=p), p=p)

    JJ = compose(JJ, R, p=p)
    KK = compose(KK, R, p=p)

    if J1 is not None:
        assert K1 is not None
        assert J1.shape[0] == K1.shape[0]
        assert eq(compose(J, J1, p=p), compose(K, K1, p=p), p=p)
        m = J1.shape[0]
        n = R.shape[0]
        F = zeros(m, n)

        Rinv = pseudo_inverse(R, p=p, check=check)

        for i in range(n):
            r = Rinv[:, i]
            u, v = r[:b], r[b:]
            u = xdot(J1, u, p=p)
            v = xdot(K1, v, p=p)
            #assert eq(u, v)
            uv = (u+v)%p
            F[:, i] = uv

        assert eq(compose(JJ, F, p=p), J1, p=p)
        assert eq(compose(KK, F, p=p), K1, p=p)

        return JJ, KK, F # <--------------- return

    return JJ, KK




def rowspan(A, p):
    m, n = A.shape
    for vec in numpy.ndindex((p,)*m):
        u = numpy.dot(vec, A) % p
        yield u


def test_pseudoinverse():
    p = 3

    m, n = 3, 3

    for i in range(1000):
        A = rand(m, n, p)
        B = row_reduce(A, p)
        if len(B) < m:
            continue

        #print(A)

        P, L, U = plu_reduce(A, p, check=True)
        #print(P)
        #print(L)
        #print(U)

        PLU = xdot(P, L, U, p=p)
        assert eq(PLU, A, p)
        #print()

    for (m,n) in [(2,3)]:
      for bits in numpy.ndindex((p,)*m*n):
        A = array(bits)
        A.shape = m,n
        B = pseudo_inverse(A, p)
        AB = xdot(A, B, p=p)
        BA = xdot(B, A, p=p)
        ABA = xdot(A, B, A, p=p)
        BAB = xdot(B, A, B, p=p)
        assert eq(A, ABA, p=p)
        assert eq(B, BAB, p=p)

    print("OK")


def test_pushout():

    p = 3
    J = zeros(2, 1)
    J[0, 0] = 1
    K = zeros(2, 1)
    K[1, 0] = 1

    JJ, KK = pushout(J, K, p)

    #print shortstrx(JJ, KK)

    assert eq(compose(J, JJ, p=p), compose(K, KK, p=p), p=p)

    J1, K1 = JJ, KK

    JJ, KK, F = pushout(J, K, p, J1, K1)

    #print
    #print shortstr(F)
    assert eq(F, identity(3), p=p)





def test():
    m, n = 3, 5
    p = 3

    for trial in range(100):
        A = rand(m, n, p)
    
        #print(A)
        B = row_reduce(A, p)
        #print(B)
    
        found = {str(u) for u in rowspan(A, p)}
        assert found == {str(u) for u in rowspan(B, p)}
    
        C = normal_form(A, p)
        #print(C)
        assert found == {str(u) for u in rowspan(C, p)}
        #print(len(found))

        #break

        K = kernel(A, p)
        #print(K)

        AK = numpy.dot(A, K.transpose()) % p
        assert AK.sum() == 0
        assert len(K) + len(B) == n

        #print()

    p = 5
    I = identity(n)
    for trial in range(100):
        L = rand(n, n, p)
        for i in range(n):
          for j in range(n):
            if i==j:
                L[i,j] = 1
            elif j>i:
                L[i,j] = 0
        L1 = l_inverse(L, p=p)
        assert eq(xdot(L1, L, p=p) , I, p=p)

    n = 4
    I = identity(n)
    for trial in range(100):
        U = rand(n, n, p)
        for i in range(n):
          for j in range(n):
            if i==j:
                U[i,j] = randint(1,p-1)
            elif j<i:
                U[i,j] = 0
        #print(U)
        U1 = u_inverse(U, p=p)
        assert eq(xdot(U1, U, p=p) , I, p=p)

    for trial in range(100):
        A = rand(m, n, p)
        B = rand(m, n, p)
        #print(A)
        #print(B)

        C, D = pushout(A, B, p)
        #print(C)
        #print(D)
    


if __name__ == "__main__":

    from time import time
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


