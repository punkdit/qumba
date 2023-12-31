#!/usr/bin/env python3

from random import randint, seed, choice, shuffle

import numpy
import numpy.random as ra

from qumba.solve import shortstr, zeros2, array2, dot2, parse, linear_independent, solve, rank
from qumba.argv import argv

from qumba import construct
#from qumba.decode.bpdecode import RadfordBPDecoder
#from qumba.decode.cluster import ClusterCSSDecoder
from qumba import decode
from qumba.tool import write



def main():
    from qumba.csscode import CSSCode

    name = argv.get("code", "toric")

    if name == "toric":
        l = argv.get('l', 4)
        code = construct.toric(l, l)
    for name in argv:
        if name.endswith(".txt"):
            s = open(name).read()
            H = parse(s)
            code = CSSCode(Hx=H, Hz=H)

    print(code)

    name = argv.get("decode", "")

#    decode = decode.SimpleDecoder(code)
#    decode = decode.ExactDecoder(code)
#    decode = decode.OEDecoder(code)
#    decode = decode.ClusterCSSDecoder(code)
#    decode = decode.ClusterCSSDecoder(code, minimize=True)
#    decode = decode.RadfordNealBPDecoder(code)

    Decoder = {
        "oe"      : decode.OEDecoder,
        "cluster" : decode.ClusterCSSDecoder,
        "bp"      : decode.RadfordNealBPDecoder,
        "retrybp" : decode.RetryBPDecoder,
        "match"   : decode.MatchingDecoder, # XX only works on surface codes
        }.get(name)
    if Decoder is None:
        Decoder = getattr(decode, name, None)
    decoder = Decoder(code)
    print(decoder.__class__.__name__)

    N = argv.get('N', 10)
    p = argv.get('p', 0.04)

    #if argv.noerr:
    #    print("redirecting stderr to stderr.out")
    #    fd = os.open("stderr.out", os.O_CREAT|os.O_WRONLY)
    #    os.dup2(fd, 2)

    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    scramble = lambda err_op, H:(err_op + dot2(ra.binomial(1, 0.5, (len(H),)), H)) % 2

    for i in range(N):

        # We use Hz to look at X type errors (bitflip errors)
        err_op = ra.binomial(1, p, (code.n,))

        write(str(err_op.sum()))

        s = dot2(code.Hz, err_op)
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
            write("%d:"%op.sum())

            # Are we in the image of Hx ? If so, then success.
            success = dot2(code.Lz, op).sum()==0

            if success and op.sum():
                nonuniq += 1
            #    print "\n", shortstr(err_op)
            #    return

            c = '.' if success else 'x'

            if op.sum() and not success:
                distance = min(distance, op.sum())

        else:
            failcount += 1

        write(c+' ')
        count += success

    if N:
        print()
        print(datestr)
        print(argv)
        print("error rate = %.8f" % (1. - 1.*count / (i+1)))
        print("fail rate  = %.8f" % (1.*failcount / (i+1)))
        print("nonuniq = %d" % nonuniq)
        print("distance <= %d" % distance)

    
if __name__ == "__main__":

    import os
    datestr = os.popen('date "+%F %H:%M:%S"').read().strip()

    from time import time
    start_time = time()

    _seed = argv.get('seed')
    if _seed is not None:
        seed(_seed)
        ra.seed(_seed)

    if argv.profile:
        import cProfile as profile
        profile.run("main()")

    else:
        main()

    print("%.3f seconds"%(time() - start_time))


