#!/usr/bin/env python3

from random import randint, seed, choice, shuffle
from functools import reduce
from operator import add

import numpy
import numpy.random as ra

from qumba.lin import shortstr, zeros2, array2, dot2, parse, linear_independent, solve, rank
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


def get_dist(code):
    Hx, Lx = code.Hx, code.Lx
    m, n = Hx.shape
    k, _ = Lx.shape
    dist = [0 for i in range(n+1)]
    lx = Lx[0]
    for v in numpy.ndindex((2,)*m):
        u = (dot2(v, Hx) + lx)%2
        r = u.sum()
        dist[r] += 1
    return dist

def weight_enum():
    codes = [construct.get_surface(d,d).to_css() for d in [2,3,4,5,6]]
    for code in codes:
        print(get_dist(code))
    

def plot_mwpm_25():
    surf_25 = construct.get_surface(5, 5).to_css() # [[25,1,5]]

    ps = [
        #0.005, 0.00736842105263158, 0.009736842105263158,
        0.012105263157894737, 0.014473684210526316, 0.016842105263157894,
        0.019210526315789477, 0.021578947368421055, 0.023947368421052634,
        0.026315789473684213, 0.02868421052631579, 0.03105263157894737,
        0.03342105263157895, 0.03578947368421053, 0.038157894736842106,
        0.04052631578947368, 0.04289473684210526, 0.045263157894736845,
        0.04763157894736842, 0.05]
    logical_error = [
        #3.5e-05, 0.00011, 0.000295,
        0.000462, 0.000843, 0.001277, 0.001726, 0.002451, 0.003443, 0.004413,
        0.0056, 0.006806, 0.008431, 0.010428, 0.012084, 0.014187,
        0.016526, 0.018882, 0.021644, 0.024665]

    ps = [0.05, 0.052631578947368425, 0.05526315789473685,
    0.05789473684210526, 0.060526315789473685, 0.06315789473684211,
    0.06578947368421054, 0.06842105263157895, 0.07105263157894737,
    0.0736842105263158, 0.07631578947368421, 0.07894736842105263,
    0.08157894736842106, 0.08421052631578949, 0.08684210526315789,
    0.08947368421052632, 0.09210526315789475, 0.09473684210526316,
    0.09736842105263158, 0.1]
    logical_error = [0.024315, 0.027729, 0.031484, 0.035366,
    0.039654, 0.043963, 0.048394, 0.053421, 0.058192, 0.063555,
    0.068975, 0.074383, 0.080669, 0.086115, 0.092523, 0.098576,
    0.104138, 0.110912, 0.11766, 0.124469]

    ps =  [0.1, 0.10526315789473685, 0.1105263157894737,
    0.11578947368421053, 0.12105263157894737, 0.12631578947368421,
    0.13157894736842107, 0.1368421052631579, 0.14210526315789473,
    0.1473684210526316, 0.15263157894736842, 0.15789473684210525,
    0.1631578947368421, 0.16842105263157897, 0.17368421052631577,
    0.17894736842105263, 0.1842105263157895, 0.18947368421052632,
    0.19473684210526315, 0.2]
    logical_error =  [0.123794, 0.138026, 0.15175, 0.165656,
    0.180528, 0.194511, 0.208327, 0.222136, 0.237275, 0.251092,
    0.264853, 0.278095, 0.290848, 0.303073, 0.316649, 0.327816,
    0.340201, 0.349842, 0.360729, 0.371713]

    Ns = [10000]*len(ps)
    render([(surf_25, "[[25,1,5]] Exact")], ps, Ns, "compare_mwpm.pdf", 
        lambda plt:plt.plot(ps, logical_error, label="[[25,1,5]] MWPM"))


    return


def plot_mwpm_49():
    surf_49 = construct.get_surface(7, 7).to_css()

    ps =  [
    #0.01, 0.013877551020408163, 0.017755102040816328,
    0.021632653061224492, 0.025510204081632654, 0.029387755102040815,
    0.03326530612244898, 0.037142857142857144, 0.041020408163265305,
    0.04489795918367347, 0.048775510204081635, 0.052653061224489796,
    0.056530612244897964, 0.060408163265306125, 0.06428571428571428,
    0.06816326530612245, 0.07204081632653062, 0.07591836734693877,
    0.07979591836734694, 0.08367346938775509, 0.08755102040816326,
    0.09142857142857143, 0.09530612244897958, 0.09918367346938775,
    0.10306122448979592, 0.10693877551020407, 0.11081632653061224,
    0.11469387755102041, 0.11857142857142856, 0.12244897959183673,
    0.1263265306122449, 0.13020408163265307, 0.13408163265306122,
    0.1379591836734694, 0.14183673469387756, 0.1457142857142857,
    0.1495918367346939, 0.15346938775510205, 0.1573469387755102,
    0.16122448979591839, 0.16510204081632654, 0.1689795918367347,
    0.17285714285714288, 0.17673469387755103, 0.18061224489795918,
    0.18448979591836737, 0.18836734693877552, 0.19224489795918367,
    0.19612244897959186, 0.2]
    logical_error =  [
    #4.3e-05, 0.000183, 0.000373, 
    0.000779,
    0.001455, 0.002503, 0.003995, 0.005852, 0.008412, 0.011254,
    0.015035, 0.019526, 0.024463, 0.030332, 0.036676, 0.043919,
    0.051972, 0.060117, 0.070034, 0.080192, 0.090065, 0.101424,
    0.112134, 0.124161, 0.136565, 0.149708, 0.162201, 0.174889,
    0.188217, 0.201358, 0.214769, 0.227193, 0.240409, 0.252791,
    0.265897, 0.277847, 0.290047, 0.301329, 0.313648, 0.324358,
    0.334997, 0.345657, 0.355504, 0.365745, 0.374934, 0.38212,
    0.390764, 0.399181, 0.406991, 0.413205]
    Ns = [100000]*len(ps)

    n = 20
    ps = ps[n:]
    logical_error = logical_error[n:]
    Ns = Ns[n:]

    render([(surf_49, "Exact")], ps, Ns, "compare_mwpm_49.pdf", 
        lambda plt:plt.plot(ps, logical_error, label="MWPM"))


def plot_mwpm_k():
    ps =  [0.01, 0.013877551020408163, 0.017755102040816328,
    0.021632653061224492, 0.025510204081632654, 0.029387755102040815,
    0.03326530612244898, 0.037142857142857144, 0.041020408163265305,
    0.04489795918367347, 0.048775510204081635, 0.052653061224489796,
    0.056530612244897964, 0.060408163265306125, 0.06428571428571428,
    0.06816326530612245, 0.07204081632653062, 0.07591836734693877,
    0.07979591836734694, 0.08367346938775509, 0.08755102040816326,
    0.09142857142857143, 0.09530612244897958, 0.09918367346938775,
    0.10306122448979592, 0.10693877551020407, 0.11081632653061224,
    0.11469387755102041, 0.11857142857142856, 0.12244897959183673,
    0.1263265306122449, 0.13020408163265307, 0.13408163265306122,
    0.1379591836734694, 0.14183673469387756, 0.1457142857142857,
    0.1495918367346939, 0.15346938775510205, 0.1573469387755102,
    0.16122448979591839, 0.16510204081632654, 0.1689795918367347,
    0.17285714285714288, 0.17673469387755103, 0.18061224489795918,
    0.18448979591836737, 0.18836734693877552, 0.19224489795918367,
    0.19612244897959186, 0.2]
    logical_error =  [0.000751, 0.001879, 0.003708, 0.006576,
    0.010156, 0.015205, 0.021649, 0.028542, 0.037159, 0.046705,
    0.057236, 0.068582, 0.081698, 0.095043, 0.109758, 0.125742,
    0.141656, 0.158261, 0.1762, 0.193368, 0.211988, 0.230845,
    0.248317, 0.268417, 0.286144, 0.305274, 0.323686, 0.343326,
    0.361115, 0.379792, 0.397122, 0.414618, 0.431529, 0.447932,
    0.463404, 0.48019, 0.494753, 0.51004, 0.52371, 0.537625,
    0.550637, 0.561951, 0.574362, 0.586014, 0.595567, 0.606469,
    0.616711, 0.626678, 0.63546, 0.642769]
    ps, logical_error = ps[::2], logical_error[::2]
    Ns = [10000]*len(ps)
    #render(
    #    [ (construct.get_toric(6,0).to_css(), "[[36,2,6]] exact") ],
    #    ps, Ns, "compare_mwpm_36_2_6.pdf",
    #    lambda plt:plt.plot(ps, logical_error, label="[[36,2,6]] MWPM"))

    ps =  [0.01, 0.013877551020408163, 0.017755102040816328,
    0.021632653061224492, 0.025510204081632654, 0.029387755102040815,
    0.03326530612244898, 0.037142857142857144, 0.041020408163265305,
    0.04489795918367347, 0.048775510204081635, 0.052653061224489796,
    0.056530612244897964, 0.060408163265306125, 0.06428571428571428,
    0.06816326530612245, 0.07204081632653062, 0.07591836734693877,
    0.07979591836734694, 0.08367346938775509, 0.08755102040816326,
    0.09142857142857143, 0.09530612244897958, 0.09918367346938775,
    0.10306122448979592, 0.10693877551020407, 0.11081632653061224,
    0.11469387755102041, 0.11857142857142856, 0.12244897959183673,
    0.1263265306122449, 0.13020408163265307, 0.13408163265306122,
    0.1379591836734694, 0.14183673469387756, 0.1457142857142857,
    0.1495918367346939, 0.15346938775510205, 0.1573469387755102,
    0.16122448979591839, 0.16510204081632654, 0.1689795918367347,
    0.17285714285714288, 0.17673469387755103, 0.18061224489795918,
    0.18448979591836737, 0.18836734693877552, 0.19224489795918367,
    0.19612244897959186, 0.2]
    logical_error =  [0.01138, 0.021571, 0.034678, 0.050179,
    0.068527, 0.088229, 0.110237, 0.133356, 0.159355, 0.185046,
    0.212415, 0.238812, 0.266676, 0.297028, 0.323638, 0.352937,
    0.380653, 0.40964, 0.436634, 0.464718, 0.491447, 0.517415,
    0.543492, 0.567512, 0.590769, 0.614292, 0.63708, 0.658073,
    0.678372, 0.697079, 0.716409, 0.734486, 0.751428, 0.767109,
    0.782353, 0.797477, 0.810413, 0.824043, 0.835752, 0.846548,
    0.85787, 0.867755, 0.877589, 0.886378, 0.894876, 0.902236,
    0.909205, 0.916407, 0.922577, 0.928542]
    ps, logical_error = ps[::4], logical_error[::4]
    ps, logical_error = ps[2:4], logical_error[2:4]
    Ns = [100000]*len(ps)
    render(
        [ (construct.get_css((30,8,3)), "[[30,8,3]] exact") ],
        ps, Ns, "compare_mwpm_30_8_3.pdf",
        lambda plt:plt.plot(ps, logical_error, label="[[30,8,3]] MWPM"))


def make_plot():

    copy = lambda code,n : reduce(add, [code]*n)

    surf_9 = construct.get_surface(3, 3).to_css() # [[9,1,3]]
    surf_16 = construct.get_surface(4, 4).to_css() # [[16,1,4]]
    surf_25 = construct.get_surface(5, 5).to_css() # [[25,1,5]]
    surf_36 = construct.get_surface(6, 6).to_css() # [[36,1,6]]
    surf_49 = construct.get_surface(7, 7).to_css()
    surf_64 = construct.get_surface(8, 8).to_css()

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
    #return

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
    ]
    #render(codes, ps, Ns, "code_compare.pdf")

    ps = [0.016, 0.012, 0.008]
    Ns = [10000, 100000, 100000]
    codes = [
        (surf_25, "[[350,14,5]]=14x[[25,1,5]]", 14),
        (get_css((56,14,6)), "[[56,14,6]] sd weight 8"), # self-dual weight 8
    ]
    #render(codes, ps, Ns, "code_compare_3.pdf")

    ps = [0.016, 0.012, 0.008]
    Ns = [10000, 100000, 100000]
    codes = [
        (surf_9, "[[72,8,3]]=8x[[9,1,3]]", 8),
        (get_css((30,8,3)), "[[30,8,3]] weight 5"), # ZX self-dual, weight 5
        (get_css((27,11,3)), "[[27,11,3]] weight 6"),
    ]
    #render(codes, ps, Ns, "code_compare_4.pdf")

    ps = [0.016, 0.012, 0.008]
    Ns = [1000, 10000, 10000]
    codes = [
        (get_css((48,18,4)), "[[48,18,4]] weight 6"),
    ]
    #render(codes, ps, Ns, "code_compare_5.pdf") # 2 hours to run


def render(codes, ps, Ns, name="code_performance.pdf", callback=None):
    import matplotlib.pyplot as plt
    plt.figure(constrained_layout=True)

    print("render", name)

    # binomial var'iance (we hope)
    var = lambda p,n: ((p*(1-p))/n)**0.5
    fig = plt.semilogy()
    fig = plt.loglog()

    plt.xlabel('physical error')
    plt.ylabel('logical error')

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
            print("%dx%s p=%.4f"%(copy, code, p), end=" ", flush=True)
            e = monte_carlo(code, N, p, decoder, copy)
            print("e=%.6f"%(e,), end=" ")
            print("weights=%s" % code.Hz.sum(1))
            if e > 0:
                ys.append(e)
                xs.append(p)
            else:
                break

        #label = str(code)
        errs = [var(p,N) for p,N in zip(ys, Ns)]
        plt.errorbar(xs, ys, yerr=errs, label=label)

    if callback:
        callback(plt)
    plt.legend(loc='lower right')
    plt.savefig(name)
    print("savefig", name)

    
def monte_carlo(code, N, p, decoder, copy=1, verbose=False):
    distance = code.n
    count = 0
    failcount = 0
    nonuniq = 0

    scramble = lambda err_op, H:(err_op + dot2(ra.binomial(1, 0.5, (len(H),)), H)) % 2

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



