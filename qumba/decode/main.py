#!/usr/bin/env python3

from random import randint, seed, choice, shuffle
from functools import reduce
from operator import add

import numpy
import numpy.random as ra

from qumba.solve import shortstr, zeros2, array2, dot2, parse, linear_independent, solve, rank
from qumba.argv import argv

from qumba import construct
from qumba import decode
from qumba.tool import write
from qumba.csscode import CSSCode, distance_z3_css
from qumba.construct import get_css


def main():

    code = None
    name = argv.get("code")
    if name == "toric":
        a = argv.get("a", 4)
        b = argv.get("a", 0)
        code = construct.get_toric(a, b)
        code = code.to_css()
        print("d =", distance_z3_css(code))
    if name == "surface":
        a = argv.get("a", 3)
        b = argv.get("a", 3)
        code = construct.get_surface(a, b)
        code = code.to_css()
        print("d =", distance_z3_css(code))

    if argv.copy:
        codes = [code] * argv.copy
        code = reduce(add, codes)
        print(code)

    param = argv.param
    if param is not None:
        code = get_css(param)

    if code is None:
        print("no code found")
        return

    if argv.dual:
        code = code.get_dual()

    print(code)

    name = argv.get("decode", "cluster")

#    decode = decode.SimpleDecoder(code)
#    decode = decode.ExactDecoder(code)
#    decode = decode.OEDecoder(code)
#    decode = decode.ClusterCSSDecoder(code)
#    decode = decode.ClusterCSSDecoder(code, minimize=True)
#    decode = decode.RadfordNealBPDecoder(code)

    if name == "chain":
        decoder = decode.ChainDecoder(
            code, 
            [decode.LookupDecoder(code), decode.OEDecoder(code)])

    else:
        Decoder = {
            "simple"  : decode.SimpleDecoder,
            "lookup"  : decode.LookupDecoder,
            "oe"      : decode.OEDecoder,
            "cluster" : decode.ClusterCSSDecoder,
            "bp"      : decode.RadfordNealBPDecoder,
            "retrybp" : decode.RetryBPDecoder,
            "match"   : decode.MatchingDecoder, # XX only works on surface codes
            }.get(name)
        if Decoder is None:
            Decoder = getattr(decode, name, None)
        decoder = Decoder(code)
    #print(decoder.__class__.__name__)

    N = argv.get('N', 10)
    p = argv.get('p', 0.04)

    #if argv.noerr:
    #    print("redirecting stderr to stderr.out")
    #    fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
    #    os.dup2(fd, 2)

    e = monte_carlo(code, N, p, decoder)
    print(e)


def make_plot():

    copy = lambda code,n : reduce(add, [code]*n)

    surf_9 = construct.get_surface(3, 3).to_css() # [[9,1,3]]
    surf_16 = construct.get_surface(4, 4).to_css() # [[16,1,4]]
    surf_25 = construct.get_surface(5, 5).to_css() # [[25,1,5]]
    surf_36 = construct.get_surface(6, 6).to_css() # [[36,1,6]]
    surf_49 = construct.get_surface(7, 7).to_css() # [[
    surf_64 = construct.get_surface(8, 8).to_css() # [[
    codes = [
        #(construct.get_713().to_css(), "[[7,1,3]]"), # slightly worse than [[9,1,3]]
        (surf_9, "[[9,1,3]]"),
        (surf_16, "[[16,1,4]]"),
        #(surf_16.get_dual(), "[[16,1,4]]^T"), # slightly better than [[16,1,4]]
        (surf_25, "[[25,1,5]]"),
        (surf_36, "[[36,1,6]]"),
        #(surf_36.get_dual(), "[[36,1,6]]^T"), # very close
        (surf_49, "[[49,1,7]]"),
        (surf_64, "[[64,1,8]]"),
        #(surf_64.get_dual(), "[[64,1,8]]^T"), # very close
    ]
    #ps = [0.096, 0.072, 0.036]
    #Ns = [10000,   10000, 10000]
    #render(codes[:6], ps, Ns)
    #return

    ps = [0.120, 0.096, 0.072, 0.048, 0.024, 0.012]#,  0.008]#, 0.004]
    Ns = [10000,   10000,   100000,  100000,  100000, 100000]#, 100000]#, 1000000]
    #render(codes, ps, Ns, "surface_codes.pdf")

    #codes = [
        #(surf_913, "[[9,1,3]]"),
        #(get_css((15,5,3)), "[[15,5,3]]"), # ZX self-dual, weight 5
        #(copy(surf_9,5), "[[45,5,3]]surf"),
    #codes = [
        #(copy(surf_9,8), "[[72,8,3]]surf"),
        #(copy(surf_16,8), "[[128,8,4]]surf"),
#        (get_css((30,7,3)), "[[30,7,3]]"), # weight 6,4
#        (get_css((36,8,4)), "[[36,8,4]]"), # weight 6,4
#        (get_css((32,12,4)), "[[32,12,4]]"), # self-dual weight 8
#        #(get_css((40,10,4)),"[[40,10,4]]"), # ZX self-dual, weight 5
#        #(get_css((40,6,4)), "[[40,6,4]]"), # weight 5,4
#        #(get_css((56,14,6)), "[[56,14,6]]"), # self-dual weight 8
#        (get_css((30,5,3)),  "[[30,5,3]]"), # weight 5,4

    ps = [0.120, 0.096, 0.072, 0.048, 0.024, 0.012]#,  0.008]#, 0.004]
    Ns = [10000,   10000,   100000,  100000,  100000, 100000]#, 100000]#, 1000000]
    ps = [0.036,  0.024, 0.012, 0.004]
    Ns = [10000, 10000, 100000, 1000000]
    codes = [
        (copy(surf_9, 5), "[[45,5,3]]=5x[[9,1,3]]"),
        (get_css((15,5,3)), "[[15,5,3]] weight 5"), # ZX self-dual, weight 5
        #(surf_25, "[[25,1,5]]"),
    ]
    #render(codes, ps, Ns, "code_compare_1.pdf")

    ps = [0.024, 0.012, 0.004]
    Ns = [1000, 10000, 100000]
    codes = [
        (surf_9, "[[72,8,3]]=8x[[9,1,3]]", 8),
        #(surf_25, "[[200,8,5]]=8x[[25,1,5]]", 8),
        (get_css((30,8,3)), "[[30,8,3]] weight 5"), # ZX self-dual, weight 5
        (get_css((30,7,3)), "[[30,7,3]] weight 4"),
        (get_css((30,7,3)).get_dual(), "[[30,7,3]]^T weight 6"),
    ]
    #render(codes, ps, Ns, "code_compare_2.pdf")

    ps = [0.024, 0.012, 0.004]
    Ns = [1000, 1000, 100000]
    codes = [
        (get_css((32,12,4)), "[[32,12,4]] sd weight 8"), # self-dual weight 8
        (get_css((40,10,4)),"[[40,10,4]] weight 5"), # ZX self-dual, weight 5
        (surf_25, "[[250,10,5]]=10x[[25,1,5]]", 10),
        (get_css((36,8,4)), "[[36,8,4]] weight 4"),
        (get_css((36,8,4)).get_dual(), "[[36,8,4]]^T weight 6"), # weight 6,4
        (get_css((40,6,4)), "[[40,6,4]] weight 4"), # weight 5,4
        (get_css((40,6,4)).get_dual(), "[[40,6,4]]^T weight 5"), # weight 5,4
        #(surf_25, "[[200,8,5]]=8x[[25,1,5]]", 8),
        #(get_css((56,14,6)), "[[56,14,6]] sd weight 8"), # self-dual weight 8
    ]
    #render(codes, ps, Ns, "code_compare.pdf")


    ps = [0.016, 0.012, 0.008]
    Ns = [1000, 10000, 10000]
    codes = [
        (surf_25, "[[350,14,5]]=14x[[25,1,5]]", 14),
        (get_css((56,14,6)), "[[56,14,6]] sd weight 8"), # self-dual weight 8
    ]
    render(codes, ps, Ns, "code_compare.pdf")


def render(codes, ps, Ns, name="code_performance.pdf"):
    import matplotlib.pyplot as plt
    plt.figure(constrained_layout=True)

    print("render", name)

    # binomial var'iance (we hope)
    var = lambda p,n: ((p*(1-p))/n)**0.5
    fig = plt.semilogy()
    fig = plt.loglog()

    plt.xlabel('qubit error')
    plt.ylabel('decode error')

    for item in codes:
        if len(item) == 2:
            code, label = item
            copy = 1
        else:
            code, label, copy = item

        assert code is not None
        decoder = decode.ChainDecoder(
            code, [decode.LookupDecoder(code), decode.OEDecoder(code)])
        xs, ys = [], []
        for p,N in zip(ps, Ns):
            e = monte_carlo(code, N, p, decoder, copy)
            print("%dx%s %.6f"%(copy, code, e))
            if e > 0:
                ys.append(e)
                xs.append(p)
            else:
                break

        #label = str(code)
        errs = [var(p,N) for p,N in zip(ys, Ns)]
        plt.errorbar(xs, ys, yerr=errs, label=label)

    plt.legend(loc='lower right')
    plt.savefig(name)
    print("savefig", name)

    
def monte_carlo(code, N, p, decoder, copy=1, verbose=False):
    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    scramble = lambda err_op, H:(err_op + dot2(ra.binomial(1, 0.5, (len(H),)), H)) % 2
    print("weights:", code.Hz.sum(1))

    for i in range(N):

        for trial in range(copy):
            # We use Hz to look at X type errors (bitflip errors)
            err_op = ra.binomial(1, p, (code.n,))
    
            if verbose:
                write(str(err_op.sum()))
    
            s = dot2(code.Hz, err_op)
            if verbose:
                write(":s%d:"%s.sum())
    
            _err_op = scramble(err_op, code.Hx)
            _err_op = scramble(_err_op, code.Lx)
            op = decoder.decode(p, _err_op, verbose=False, argv=argv)
    
            c = 'F'
            success = False
            if op is not None:
                op = (op+err_op)%2
                # Should be a codeword of Hz (kernel of Hz)
                if dot2(code.Hz, op).sum() != 0:
                    print(dot2(code.Hz, op))
                    print("\n!!!!!  BUGBUG  !!!!!", sparsestr(err_op))
                    continue
                if verbose:
                    write("%d:"%op.sum())
    
                # Are we in the image of Hx ? If so, then success.
                success = dot2(code.Lz, op).sum()==0
    
                if success and op.sum():
                    nonuniq += 1
                c = '.' if success else 'x'
                if op.sum() and not success:
                    distance = min(distance, op.sum())
    
            else:
                failcount += 1
    
            if verbose:
                write(c+' ')
            if not success:
                break # all _copies must _succeed

        count += success

    errors = 1 - count / (i+1)

    if N and verbose:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate  = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    return errors


if __name__ == "__main__":

    import os
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()

    from time import time
    start_time = time()


    profile = argv.profile
    name = argv.next() or "main"
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

    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)



