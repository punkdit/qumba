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
                ri = Hij**(p-2)
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
        inv = r**(p-2) 
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

#    L, U = lu_decompose(A)
#    assert eq2(dot(L, U), A)

    U = row_reduce(A, p, inplace=inplace)

    # We are looking for a basis for the nullspace of A

    m, n = U.shape

    if verbose:
        print("kernel: shape", m, n)
        #print shortstr(U, deco=True)
        print()

    items = []
    for row in range(m):
        cols = numpy.where(U[row, :])[0]
        if not len(cols):
            break
        col = cols[0]
        items.append((row, col))

    #items.sort(key = lambda item : item[1])
    #print items
    #rows = [row for (row, col) in items]
    #U = U[rows]
    leading = [col for (row, col) in items]
    degeneracy = m - len(leading)

    if verbose:
        print("leading:", leading)
        print("degeneracy:", degeneracy)

    # Look for the free variables
    vars = []
    row = 0
    col = 0
    while row < m and col < n:
        #print row, col
        if U[row:, col].max() == 0: # XXX optimize this XXX
            #print "*"
            assert U[row:, col].max() == 0, U[row:, col]
            vars.append(col)
        else:
            #print U[row:, col]
            while row<m and U[row:, col].max():
                row += 1
                #print "row", row
                #if row<m:
                #    print U[row:, col]
        col += 1
    for k in range(col, n):
        vars.append(k)

    if verbose:
        print("found %d free vars:" % len(vars), vars)

    basis = []
    for var in vars:

        #print "var:", var
        v = numpy.zeros((n,), dtype=int_scalar)
        v[var] = 1
        row = min(var-1, m-1)
        while row>=0:
            u = dot(U[row], v)
            if u.sum()%p:
                col = leading[row]
                #print "\trow", row, "leading:", col
                v[col] = 1
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





def rowspan(A, p):
    m, n = A.shape
    for vec in numpy.ndindex((p,)*m):
        u = numpy.dot(vec, A) % p
        yield u


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
        #print()

        K = kernel(A, p)



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


