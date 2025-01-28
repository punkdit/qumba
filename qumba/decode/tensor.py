#!/usr/bin/env python3

import sys
import math
from random import choice, randint, seed, shuffle
from time import time
from operator import mul
from functools import reduce

import numpy
from numpy import dot
import numpy.linalg

from qumba.lin import dot2, zeros2, shortstr, shortstrx, span, array2
from qumba.decode.simple import Decoder
from qumba.decode.network import TensorNetwork
from qumba.argv import argv


def genidx(shape):
    return numpy.ndindex(*shape)

scalar = numpy.float64



class ExactDecoder(Decoder):
    "Tensor network decoder. Computes exact probabilities."
    "See OEDecoder for a faster version (smarter contractions.)"
    def __init__(self, code):
        self.code = code
        assert code.k <= 24, "too big...?"
        self.logops = list(span(code.Lx))

    def get_p(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        net = TensorNetwork()

        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            links = numpy.where(h)[0]

            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

            net.append(A, links)

        net.contract_all(verbose)
        #net.contract_all_slow(verbose)

        return net.value

    def get_p_slow(self, p, op, verbose=False):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        r = 0
        for idx in genidx((2,)*mx):
            h = op.copy()
            for i in range(mx):
                if idx[i]:
                    h += Hx[i]
            h %= 2
            w = h.sum()
            r += (p**w)*((1.-p)**(n-w))

        return r

    def decode(self, p, err_op, argv=None, verbose=False, **kw):
        code = self.code
        Hz = code.Hz
        Tx = code.Tx
        Hx = code.Hx
        Lx = code.Lx
        n = code.n
        mx = code.mx

        T = self.get_T(err_op)

        #if T.sum()==0:
        #    return T

        dist = []

        #print
        best = None
        best_r = 0.
        for logop in self.logops:
            op = (T+logop)%2
            r = self.get_p(p, op, verbose=verbose)
            #r1 = self.get_p_slow(p, op, verbose=verbose)
            #print "%.6f"%r, "%.6f"%r1
            if r>best_r:
                best_r = r
                best = op
            dist.append(r)

        #print(dist)
        return best


class OEDecoder(ExactDecoder):
    "Faster version of ExactDecoder "
    def __init__(self, code):
        self.code = code
        assert code.k <= 24, "too big...?"
        self.logops = list(span(code.Lx))
        self.n = code.n
        self.build()

    def build(self):
        import opt_einsum as oe # pip3 install opt_einsum

        #from opt_einsum.backends.dispatch import _has_einsum
        #_has_einsum['numpy'] = False

        code = self.code
        #Hz = code.Hz
        #Tx = code.Tx
        Hx = code.Hx
        #Lx = code.Lx
        n = code.n
        mx = code.mx

        net = []
        As = []
        linkss = []

        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch: w=%d"%w
            shape = (2,)*w
            A = numpy.zeros(shape, dtype=scalar)
            As.append(A)
            links = list(numpy.where(h)[0])
            linkss.append(links)
            net.append((A, links))
            #print(A.shape, links)
        #print(linkss)

        self.net = net
        self.linkss = linkss
        self.As = As
        self.check()

        kw = {"optimize" : "random-greedy"}

        str_args = []
        shapes = []
        for A, links in net:
            links = ''.join(oe.get_symbol(i) for i in links)
            #print(A.shape, links)
            str_args.append(links)
            shapes.append(A.shape)
        #print(shapes)
        #print(linkss)
        str_args = ','.join(str_args)
        #print(str_args)
        path, path_info = oe.contract_path(str_args, *As, **kw)
        #print(path_info)
        sz = path_info.largest_intermediate
        print("OEDecoder: size=%d" % sz)

#        if sz > 4194304:
        if sz > 134217728:
            assert 0, "too big... maybe"

        self.do_contract = oe.contract_expression(str_args, *shapes, **kw)

    def get_links(self):
        links = []
        for _links in self.linkss:
            links += _links
        links = list(set(links))
        links.sort()
        return links

    def has_link(self, link): # TOO SLOW
        linkss = self.linkss
        idxs = [i for i in range(self.n) if link in linkss[i]]
        return idxs

    def check(self):
        As = self.As
        linkss = self.linkss
        assert len(As)==len(linkss)
        for A, links in self.net:
            assert len(A.shape)==len(links), ((A.shape), (links))
        for link in self.get_links():
            idxs = self.has_link(link)
            shape = [As[idx].shape[linkss[idx].index(link)] for idx in idxs]
            assert len(set(shape))==1, shape
        return True

    do_contract = None
    def contract_oe(self):
        if self.do_contract is None:
            import opt_einsum as oe
            args = []
            for A, links in self.net:
                args.append(A)
                args.append(links)
                links = ''.join(oe.get_symbol(i) for i in links)
    
            v = oe.contract(*args)
            #print("contract_oe", v.shape)

        else:
            v = self.do_contract(*self.As)

        assert v.shape == ()
        return v[()]

    t0 = 0.
    t1 = 0.
    def get_p(self, p, op, verbose=False):
        code = self.code
        Hx = code.Hx
        n = code.n
        mx = code.mx

        t0 = time()
        
        # one tensor for each qubit
        for i in range(n):
            h = Hx[:, i]
            w = h.sum()
            assert w<20, "ouch"
            shape = (2,)*w
            A, links = self.net[i]
            opi = op[i]

            for idx in genidx(shape):
                if sum(idx)%2 == opi:
                    A[idx] = 1.-p # qubit is off
                else:
                    A[idx] = p # qubit is on

        t1 = time()

        value = self.contract_oe()

        t2 = time()

        self.t0 += t1-t0
        self.t1 += t2-t1

        #write(".")
        return value

    def fini(self):
        print("\nOEDecoder.t0 =", self.t0)
        print("OEDecoder.t1 =", self.t1)




