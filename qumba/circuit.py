#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
import pickle

import numpy

from qumba import solve
solve.int_scalar = numpy.int32 # qupy.solve
from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce)
from qumba.qcode import QCode, SymplecticSpace, strop, fromstr
from qumba.csscode import CSSCode, find_logicals
from qumba.autos import get_autos
from qumba import csscode, construct
from qumba.construct import get_422, get_513, golay, get_10_2_3, reed_muller
from qumba.action import mulclose, mulclose_hom, mulclose_find, Perm
from qumba.util import cross
from qumba.symplectic import Building
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
from qumba import clifford, matrix
from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
from qumba.syntax import Syntax


def fix_qubit_order(job, samps):
    code = job.code
    start = "// final qubit order: "
    assert start in code
    lines = code.split("\n")
    idxs = None
    for line in lines:
        if line.startswith(start):
            assert idxs is None
            line = line[len(start):]
            idxs = eval(line)
    assert idxs is not None, "%r not found"%start
    print("fix_order", idxs)
    #print(samps)
    samps = [''.join(reversed(samp)) for samp in samps]
    #print(samps)
    samps = [''.join(samp[idx] for idx in idxs) for samp in samps]
    #print(samps)
    return samps



def send(qasms=None, shots=1, 
        error_model=True, 
        simulator="stabilizer",  # "state-vector" is slower
        name = argv.get("name", "job"),
        flatten=True, # return flat list of samps
        p1_errors=True,  # bool
        p2_errors=True, # bool
        init_errors=True, # bool
        meas_errors=True, # bool
        memory_errors=True, # bool
        leak2depolar=False,
        reorder=False,
        **kw): # kw goes into params
    from qjobs import QPU, Batch
    local = not argv.live
    machine = argv.get("machine", "H1-1E")
    print("machine:", machine, ("local" if local else ""))
    qpu = QPU(machine, domain="prod", local=local)
    
    batch = Batch(qpu, save=not local)
    
    if qasms is None:
        # create & measure Bell state
        qasms = """
        OPENQASM 2.0;
        include "hqslib1.inc";
        
        qreg q[2];
        creg m[2];
        
        h q[0];
        
        CX q[0], q[1];
        
        measure q -> m;
        """

    if type(qasms) is str:
        qasms = [qasms]

    options = {
        'simulator': simulator,
        'error-model': error_model, # bool
        'error-params': 
        {
            'leak2depolar': leak2depolar, # bool
            # these all give "Unexpected error_params key!" error:
            #'p1_errors': p1_errors,  # bool
            #'p2_errors': p2_errors, # bool
            #'init_errors': init_errors, # bool
            #'meas_errors': meas_errors, # bool
            #'memory_errors': memory_errors, # bool
        }
    }
    print("options:", options)
    
    # We can append jobs to the Batch object to run
    for i,qasm in enumerate(qasms):
        if len(qasms)>1:
            the_name = "%s_%d"%(name,i)
        else:
            the_name = name
        batch.append(qasm, shots=shots, options=options, name=the_name, params=kw)
    
    batch.submit()
    batch.retrieve(wait=argv.wait)
    
    sampss = []
    for job in batch.jobs:
        results = job.results
        if results['status'] == 'completed':
            samps = results["results"]["m"]
            if reorder:
                samps = fix_qubit_order(job, samps)
            sampss.append(samps)
        else:
            samps = []
            print(results['status'])
            print(results)
            if argv.showcode:
                print(job.code)
    if flatten:
        return reduce(add, sampss, [])
    return sampss


def load(flatten=True, reorder=False, match_jobs=False):
    sampss = []
    code = None
    while 1:
        name = argv.next()
        if name is None:
            break
        assert name.endswith(".p"), "not a batch file: %r"%name
        print("batch:", name)
        f = open(name, "rb")
        batch = pickle.load(f)
        f.close()
        for job in batch.jobs:
            print("job:", job.id)
            results = job.retrieve()
            #results = job.results
            status = results["status"]
            print("\tstatus:", status)
            #print("params:", results["params"])
            #print(' '.join(results.keys()))
            print("job:", job.id)
            if job.params:
                print("\t%s"%(job.params,))
            if code is not None:
                # check we are using the same circuit in each batch
                lhs, rhs = code, job.code
                lhs, rhs = lhs.split("\n"), rhs.split("\n")
                #print(lhs)
                #print(rhs)
                lhs = [line for line in lhs if not line.startswith("//")]
                rhs = [line for line in rhs if not line.startswith("//")]
                if not match_jobs:
                    pass
                elif lhs == rhs:
                    print("\tcode == job.code")
                else:
                    print("="*79)
                    print("job.code missmatch!")
                    print("="*79)
                    print('\n'.join(lhs))
                    print("="*79)
                    print('\n'.join(rhs))
                    print("="*79)
                    assert 0
            else:
                code = job.code
            if argv.showcode:
                print(code)
            samps = []
            if status == "completed":
                samps = results["results"]["m"]
            if reorder:
                samps = fix_qubit_order(job, samps)
            sampss.append(samps)
    if flatten:
        return reduce(add, sampss, [])
    return sampss

load_batch = load

    

class Circuit(object):
    def __init__(self, n):
        self.n = n
        self.labels = list(range(n)) # mutates !

    def header(self):
        n = self.n
        qasm = """
OPENQASM 2.0;
include "hqslib1.inc";
        
qreg q[%d];
creg m[%d];

reset q;

"""%(n,n)
        return qasm

#    def footer(self):
#        n = self.n
#        return "measure q -> m;\n"

    def measure(self):
        return "measure q -> m;\n"

    def barrier(self):
        return "barrier q;\n"

    def op1(self, op, i=0):
        assert type(op) is str, (op, i)
        assert type(i) is int, (op, i)
        assert 0<=i<self.n
        labels = self.labels
        return "%s q[%d];"%(op, labels[i])

    def op2(self, op, i=0, j=1):
        assert type(op) is str, (op, i, j)
        assert type(i) is int, (op, i, j)
        assert type(j) is int, (op, i, j)
        assert 0<=i<self.n
        assert 0<=j<self.n
        labels = self.labels
        return "%s q[%d], q[%d];"%(op, labels[i], labels[j])

    def op(self, op, *args):
        if len(args) == 1:
            return self.op1(op, *args)
        elif len(args) == 2:
            return self.op2(op, *args)
        assert 0, args

    def CNOT(self, *args):
        return self.CX(*args)

    def P(self, *idxs):
        #assert len(idxs) == self.n
        assert set(idxs) == set(range(len(idxs)))
        idxs = idxs + tuple(range(len(idxs), self.n))
        assert len(idxs) == self.n
        labels = self.labels
        self.labels = [labels[i] for i in idxs]
        return "// P%s\n// labels = %s"%(str(idxs), self.labels)

    def COMMENT(self, comment):
        return "// %s"%comment

    def SWAP(self, i, j):
        idxs = list(range(self.n))
        idxs[i], idxs[j] = idxs[j], idxs[i]
        return self.P(*idxs)

    def __getattr__(self, name):
        name = name.lower()
        meth = lambda *args : self.op(name, *args)
        return meth

    def run_expr(self, expr):
        lines = []
        for e in reversed(expr): # execute expr right-to-left 
            #print(e)
            dag = e.endswith(".d")
            if dag:
                e = e[:-2]
            s = "self.%s"%(e,)
            op = eval(s, {"self":self})
            if dag:
                assert op.startswith("s "), op
                op = op.replace("s ", "sdg ")
            lines.append(op+"\n")
        return ''.join(lines)

    def run_qasm(self, expr):
        assert expr[0] == measure[0]
        self.labels = list(range(self.n))
        qasm = self.header() + self.run_expr(expr)
        qasm += "// final qubit order: %s\n\n"%(self.labels,)
        return qasm


def parsevec(s):
    s = s.strip()
    s = s.replace('\n', '')
    s = s.replace(' ', '')
    if '+' in s:
        vals = [parsevec(item) for item in s.split("+")]
        return reduce(add, vals)
    if '*' in s:
        vals = [parsevec(item) for item in s.split("*")]
        return reduce(mul, vals)
    if s=="-1":
        return -1
    if s=="i":
        return w4
    if s=="-i":
        return -w4
    if s=="0":
        return 0
    s = s.replace('.', '0')
    assert len(s) == s.count('0')+s.count('1') # etc
    lookup = {
        '0' : red(1,0,0),
        '1' : red(1,0,2),
        '+' : green(1,0,0),
        '-' : green(1,0,2),
    }
    items = [lookup[c] for c in s]
    return reduce(matmul, items)
        

def strvec(u, sep=" "):
    basis = """
    0000 0001 0010 0011 0100 0101 0110 0111
    1000 1001 1010 1011 1100 1101 1110 1111
    """.strip().split()
    lookup = {
        1 : "",
        -1 : "-",
        w4 : "i",
        -w4 : "-i",
    }
    items = []
    for i in range(16):
        x = u[i][0]
        if x == 0:
            continue
        
        v = r"|%s\rangle"%basis[i]
        items.append("%s%s"%(lookup[x], v)+sep)
    s = "+".join(items)
    s = s.replace("+-", "-")
    return s


def test_parsevec():
    v0 = parsevec("""
        0000
     +i*0011
    +-1*0101
     +i*0110
     +i*1001
    +-1*1010
     +i*1100
     +  1111
    """)
    #v0 = (r2/4)*v0
    v1 = parsevec("""
        0001
    +-i*0010
    +-1*0100
    +-i*0111
     +i*1000
       +1011
     +i*1101
    +-1*1110
    """)
    x = (v0.d * v1)[0][0]
    assert x==0



def sbin(i, n):
    c = str(bin(i))[2:]
    c = '0'*(n-len(c))+c
    return c

def vdump(v):
    print("[", end=" ")
    N = len(v)
    n = 0
    while 2**n < N:
        n += 1
    assert 2**n == N
    for i in range(N):
        #if abs(v[i]) > 1e-4:
        r = v[i]
        if abs(r.imag) < 1e-4:
            r = r.real
        if r != 0:
            print("%s:%s"%(sbin(i, n),r), end=" ")
    print("]")

def cdump(v):
    print("[", end=" ")
    N = len(v)
    n = 0
    while 2**n < N:
        n += 1
    assert 2**n == N
    for i in range(N):
        #if abs(v[i]) > 1e-4:
        r = v[i]
        if r != 0:
            print("%s:%s"%(sbin(i, n),r), end=" ")
    print("]")



def find_state(tgt, src, gen, verbose=False, maxsize=None):
    "find sequence of gen's that produces tgt state from src state (up to phase)"
    assert isinstance(tgt, clifford.Matrix)
    assert isinstance(src, clifford.Matrix)

    # choose a canonical vector up to phase
    def canonical(u):
        m,n = u.shape
        for i in range(m):
            if u[i][0]:
                break
        x = u[i][0]
        if x == 1:
            return u
        u = (1/x)*u
        return u
        
    src = canonical(src)
    tgt = canonical(tgt)

    bdy = [src] # new states to explore
    paths = {src : ()} # states and how we got there
    if tgt in paths:
        return paths[tgt]
    while bdy:
        if verbose:
            print(len(paths), end=" ", flush=True)
        _bdy = []
        for A in gen:
            #assert isinstance(A, Matrix)
            for v in bdy:
                u = A*v
                u = canonical(u)
                #assert isinstance(u, Matrix)
                if u in paths:
                    continue
                paths[u] = A.name + paths[v]
                _bdy.append(u)
                if u == tgt:
                    if verbose:
                        print(len(paths), "found!")
                    return paths[u]
                if maxsize and len(paths)>=maxsize:
                    if verbose:
                        print(len(paths), "maxsize!")
                    return
        bdy = _bdy
    if verbose:
        print("exhausted search")
    return


def inverse_P(item):
    perm = eval(item[1:])
    assert type(perm) is tuple
    n = len(perm)
    perm = Perm(list(perm), list(range(n)))
    #print("inverse_P", perm)
    perm = ~perm
    perm = tuple(perm[i] for i in range(n))
    #print("inverse_P", perm)
    name = "P"+str(perm)
    name = name.replace(" ", "")
    return name

def get_inverse(name):
    items = []
    for item in reversed(name):
        stem = item[:item.index("(")]
        if item.endswith(".d"):
            item = item.replace(".d", "")
        elif stem == "S":
            item = item + ".d"
        elif stem == "P":
            #item = item + ".t"
            item = inverse_P(item)
        else:
            assert stem in "H X Z Y CX CZ CY".split(), stem
        items.append(item)
    return tuple(items)


def send_idxs(name, send, n): # UGH
    #print("send_idxs", name, send)
    items = []
    for item in name:
        if item == "?":
            return ("?",)
        i = item.index("(")
        stem, idxs = item[:i], item[i:]
        #print(item, stem, idxs)
        idxs = eval(idxs)
        idxs = (idxs,) if type(idxs) is int else idxs
        if stem == "P":
            _idxs = list(range(n))
            for i,ii in enumerate(idxs):
                _idxs[send[i]] = send[ii]
            idxs = _idxs
            assert len(idxs) == len(list(set(idxs))), idxs
        else:
            idxs = [send[idx] for idx in idxs]
        idxs = "(%s)"%idxs[0] if len(idxs)==1 else tuple(idxs)
        item = "%s%s"%(stem, idxs)
        items.append(item)
    items = tuple(items)
    #print("\t", items)
    return items


def test_circuit():
    circuit = Circuit(4)
    assert circuit.H(0) == "h q[0];"
    assert circuit.CX(0,2) == "cx q[0], q[2];"
    assert circuit.CY(1,2) == "cy q[1], q[2];"

barrier = ("barrier()",)
measure = ("measure()",)


def variance(p, n):
    return ((p*(1-p))/n)**0.5



if __name__ == "__main__":

    from time import time
    start_time = time()

    print(argv)

    profile = argv.profile
    name = argv.next() or "test"
    _seed = argv.get("seed")
    if _seed is not None:
        from random import seed
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

