#!/usr/bin/env python3

"""
Dynamic programming search
"""

import sys, os
from math import *
from random import *
import time
import heapq
import itertools
import gc


gc.enable()
#gc.set_debug(gc.DEBUG_LEAK)


from qumba import solve
from qumba.solve import shortstr, shortstrx, eq2, dot2, array2, zeros2
#from qumba.tree import Tree
from qumba.tool import write


class Op(object):
    def __init__(self, T):
        self.T = T
        self.w = T.sum()
        self.hash = hash(T.tostring())
        #self.hash = hash(tuple(T))

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        if self.w!=other.w:
            return False # that was easy
        return eq2(self.T, other.T)

    def __le__(self, other):
        return self.w<=other.w

#    def __del__(self):
#        del self.T


class Tanner(object):
    """ Tanner graph for a parity check matrix.
    """
    def __init__(self, H):
        m, n = H.shape
        checks = {}
        for j in range(n):
            checks[j] = []
        for i in range(m):
          for j in range(n):
            if H[i, j]:
              checks[j].append(i)
        #print checks
        self.checks = checks
        self.m, self.n = m, n
        self.H = H

    def __str__(self):
        return "Tanner(%d, %d)"%(self.m, self.n)

    __repr__ = __str__

    def add_dependant(self):
        H = self.H
        m, n = H.shape
        H1 = zeros2(m+1, n)
        H1[:m] = H
        H1[m] = H.sum(0)
        m += 1
        H = H1
        self.__init__(H1)

    def nbd(self, op): # 50% of search time spent here
        checks = self.checks
        n = self.n
        H = self.H
        T = op.T
        for j in range(n): # XXX use numpy where
            if T[j]==0:
                continue
            for i in checks[j]:
                T1 = (T+H[i])%2
                yield Op(T1)

    def split(self):
        m, n = self.m, self.n
        checks = self.checks # map j in n -> list of i in m
        nbd = {} # map i in m -> list of j in m
        for idxs in checks.values():
          for i0 in idxs:
            for i1 in idxs:
              if i0==i1:
                continue
              nbd.setdefault(i0, []).append(i1)
        #print nbd

        root = 0
        tree = Tree(root)
        last = None
        while tree.leaves:
            last = list(tree.leaves)[0]
            tree.grow(nbd)
            #print tree.leaves
        #print "last:",last
        left, right = set([root]), set([last])

        ltree = Tree(root)
        rtree = Tree(last)

        while ltree.leaves or rtree.leaves:
            ltree.grow(nbd)
            lleaves = list(ltree.leaves)
            rtree.grow(nbd)
            rleaves = list(rtree.leaves)
            #print lleaves, rleaves
            i = 0
            while 1:
                #print i, left, right
                if i<len(lleaves) and i<len(rleaves):
                    li = lleaves[i]
                    ri = rleaves[i]
                    if len(left)<=len(right):
                        if li not in right:
                            left.add(li)
                        if ri not in left:
                            right.add(ri)
                    else:
                        if ri not in left:
                            right.add(ri)
                        if li not in right:
                            left.add(li)
                elif i<len(lleaves):
                    li = lleaves[i]
                    if li not in right:
                        left.add(li)
                elif i<len(rleaves):
                    ri = rleaves[i]
                    if ri not in left:
                        right.add(ri)
                else:
                    break
                i += 1
        #print len(left), len(right)
        #print left, right
        left = list(left)
        left.sort()
        left = Tanner(self.H[left])
        right = list(right)
        right.sort()
        right = Tanner(self.H[right])
        return left, right

    def search(self, T):
        root = Op(T)
        active = set([root])
        seen = set([root])

        done = False
        while not done:
            done = True
            _active = set()
            for op in active:
              for op1 in self.nbd(op):
                if op1 not in seen:
                  seen.add(op1)
                  _active.add(op1)
                  done = False
            active = _active

    def minimize(self, T, target=None, maxsize=None, maxiter=None, verbose=False):
        """
        """
        root = Op(T)
        active = set([root])
        priority = [root]
        seen = set([root])
        best = root
        w = best.w

        if w<=target:
            return T

        count = 0
        while priority:

            op = heapq.heappop(priority)
            assert op in seen
            if op.w < best.w:
                if verbose:
                    write("%s:%s:%d "%(len(seen), len(priority), op.w))
                best = op

                # Start again... release memory...
                priority = [best]
                seen = set(priority)
                if target is not None and best.w<=target:
                    break

                count += 1
                if maxiter and count>=maxiter:
                    break

            for op1 in self.nbd(op):
                if op1 not in seen:
                    seen.add(op1)
                    heapq.heappush(priority, op1)

            # Hmm... this is kind of useless...
            if maxsize and len(priority)>maxsize:
                priority = priority[:maxsize/2]
                seen = set(priority)
                if verbose:
                    write("prune %s:%s:%d "%(len(seen), len(priority), op.w))


        #if verbose:
        #    print

        return best.T
    
    def localmin(self, T, MAXSEEN=100000, stopat=None, verbose=False):
        """ Search, considering operators of non-increasing weight. 
        """
        root = Op(T)
        active = set([root])
        seen = set([root])
        best = root

        #gc.set_debug(gc.DEBUG_UNCOLLECTABLE)

        done = False
        while not done:
            done = True
            _active = set()
            for op in active:
              for op1 in self.nbd(op):
                if op1.w>op.w:
                  continue
                if op1 not in seen:
                  seen.add(op1)
                  if op1.w<=best.w:
                    _active.add(op1)
                    done = False
                  if op1.w<best.w:
                    best = op1
                    _active = set([best])
                    seen = set([best])
                    break
              else:
                continue
              break
            active = _active
            gc.collect()
            if verbose:
                write("%s:%d:%d "%(len(seen), len(active), best.w))
            if len(seen)>MAXSEEN:
                write("localmin:MAXSEEN ")
                break
            if stopat is not None and best.w<=stopat:
                break
        #if verbose:
        #    print

        #print "\ngc:", gc.garbage

        return best.T
    



