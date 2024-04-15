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
from qumba.action import mulclose, mulclose_hom, mulclose_find
from qumba.util import cross
from qumba.symplectic import Building
from qumba.unwrap import unwrap, unwrap_encoder
from qumba.smap import SMap
from qumba.argv import argv
from qumba.unwrap import Cover
from qumba import clifford, matrix
from qumba.clifford import Clifford, red, green, K, r2, ir2, w4, w8, half, latex
from qumba.syntax import Syntax



def send(qasms=None, shots=1, 
        error_model=True, 
        simulator="stabilizer",  # "state-vector" is slower
        name = argv.get("name", "job"),
        p1_errors=True,  # bool
        p2_errors=True, # bool
        init_errors=True, # bool
        meas_errors=True, # bool
        memory_errors=True, # bool
        leak2depolar=False,
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
    
    samps = []
    for job in batch.jobs:
        results = job.results
        if results['status'] == 'completed':
            samps += results["results"]["m"]
        else:
            print(results['status'])
            print(results)
            #print(job.code)
    return samps


def load():
    name = argv.next()
    assert name.endswith(".p"), name
    f = open(name, "rb")
    batch = pickle.load(f)
    f.close()
    samps = []
    for job in batch.jobs:
        results = job.retrieve()
        #results = job.results
        status = results["status"]
        print("status:", status)
        #print("params:", results["params"])
        #print(' '.join(results.keys()))
        print(job.params)
        if status == "completed":
            samps += results["results"]["m"]
    return samps

    

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
        return "// P%s"%str(idxs)

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



def get_inverse(name):
    items = []
    for item in reversed(name):
        stem = item[:item.index("(")]
        if item.endswith(".d"):
            item = item.replace(".d", "")
        elif stem == "S":
            item = item + ".d"
        else:
            assert stem in "H X Z Y CX CZ CY".split(), stem
        items.append(item)
    return tuple(items)


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

