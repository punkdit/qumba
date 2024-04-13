#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint, choice
from operator import add, matmul, mul
from functools import reduce
import pickle

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum, row_reduce)
from qumba.qcode import QCode, SymplecticSpace, strop, Matrix, fromstr
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
from qumba.clifford import Clifford, red, green, K, Matrix, r2, ir2, w4, w8, half, latex



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


def find_state(tgt, src, gen, verbose=False, maxsize=None):
    "find sequence of gen's that produces tgt state from src state (up to phase)"
    assert isinstance(tgt, Matrix)
    assert isinstance(src, Matrix)

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


def test_412():
    code = construct.get_412()
    n = code.n
    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z
    get_perm = c.get_P
    #E = get_encoder(code)
    EE = code.get_clifford_encoder()
    P = code.get_projector()

    E2 = CZ(2,0)*CY(2,3)*H(2)
    E1 = CY(1,2)*CZ(1,3)*H(1)
    E0 = CY(0,1)*CZ(0,2)*H(0)
    E = E0*E1*E2 # WORKS

    assert  EE == E

    # pick the correct logical basis
    E = E*S(3)*H(3)*S(3).d

    print(E.name)

    r0 = red(1,0,0)
    r1 = red(1,0,2)
    v0 = r0@r0@r0@r0
    v1 = r0@r0@r0@r1
    u0 = E*v0
    u1 = E*v1
    assert P*u0 == u0
    assert P*u1 == u1
    u = (r2*u1)
    
    # look for logical zero and logical one

    X, Y, Z = c.X, c.Y, c.Z
    Lx = Z(0) * X(1)
    Lz = Z(1) * X(2)
    Ly = w4 * Lx * Lz
    assert Lx * Lz == -Lz * Lx
    assert Lx * Ly == -Ly * Lx
    assert Lz * Ly == -Ly * Lz
    assert Lx*P == P*Lx
    assert Lz*P == P*Lz
    assert Ly*P == P*Ly

    dot = lambda l,r : (l.d*r)[0][0]

    # Use codespace projector to generate vectors in the codespace:

    states = []

    # logical |+>,|->
    v0 = 2*P*parsevec("0000 + -1*0001") # +1 eigenvector of Lx
    v1 = Lz*v0 # -1 eigenvector of Lx
    assert Lx*v0 == v0
    assert Lx*v1 == -v1
    assert dot(v0,v1) == 0
    states.append([v0,v1])

    # logical |+i>,|-i>
    v0 = 2*P*parsevec("0000") # +1 eigenvector of Ly 
    v1 = Lx*v0 # -1 eigenvector of Ly
    assert Ly*v0 == v0
    assert Ly*v1 == -v1
    assert dot(v0,v1) == 0
    states.append([v0,v1])

    # logical |0>,|1>
    v0 = 2*P*parsevec("0000 + i*0001") # +1 eigenvector of Lz
    v1 = Lx*v0 # -1 eigenvector of Lz
    assert Lz*v0 == v0
    assert Lz*v1 == -v1
    assert dot(v0,v1)==0
    assert v0 == u0
    states.append([v0,v1])

    if argv.find_state:
        v0 = states[argv.i][argv.j]
        # search for logical |0> state prep 
        gen = [op(i) for i in range(n) for op in [X,Y,Z,S,H]]
        gen += [CZ(i,j) for i in range(n) for j in range(i)]
        gen += [op(i,j) for i in range(n) for j in range(n) for op in [CX,CY] if i!=j]
        name = find_state(v0, parsevec("0000"), gen, maxsize = 100000, verbose = True)
        print(name)

    # logical |0> state prep 
    v0, v1 = states[2] # |0>,|1>
    prep = ('Z(0)', 'X(0)', 'H(0)', 'CX(0,3)', 'CY(1,2)', 'H(2)', 'CY(0,1)', 'H(0)', 'H(1)')
    U = c.get_expr(prep)
    u = U*parsevec("0000")
    assert u == v0

    #M = (half**4)*Matrix(K,[[dot(u0,v0),dot(u0,v1)],[dot(u1,v0),dot(u1,v1)]])
    #print(M)

    if 0:
        # logical H
        L = get_perm(1,2,3,0)
        assert L*Lx*L.d == Lz # H action
    elif 1:
        L = get_perm(0,3,2,1)
        L = H(0)*H(1)*H(2)*H(3)*X(0)*X(2)*L
        lz = L*Lx*L.d
        lx = L*Lz*L.d
        assert lz*Lx == -Lx*lz
        assert lz*Lz == Lz*lz
        assert lx*Lx == Lx*lx
        assert lx*Lz == -Lz*lx
        #assert L*Lx*L.d == Lz # H action
    elif 0:
        # logical X*Z
        L = get_perm(1,0,3,2)*H(0)*H(1)*H(2)*H(3)*X(0)*X(2)
    else:
        # logical S
        L = SHS(0)*SH(1)*HS(2)*S(3)
        L = L*get_perm(0,2,1,3)
        L = L*X(0)*X(2)
        print(L.name)

    c1 = Clifford(1)
    I,X,Y,Z,S,H = c1.get_identity(), c1.X(), c1.Y(), c1.Z(), c1.S(), c1.H()
    phases = []
    for i in range(8):
        phase = (w8**i)*I
        phase.name = ("(1^{%d/8})"%i,)
        phases.append(phase)
    gen = phases + [X, Z, Y, S, S.d, H]
    #g = mulclose_find(gen, M)
    #print(g.name)
    #return

    G = mulclose([S, H])
    assert len(G) == 192

    def getlogop(L):
        basis = [v0, v1]
        M = []
        for l in basis:
          row = []
          for r in basis:
            u = l.d * L * r
            row.append(u[0][0])
          M.append(row)
        return Matrix(K, M)
    
    #print("Lx =")
    lx = (half**4)*getlogop(Lx)
    assert lx == X
    #print(lx)
    #print("Lz =")
    #print(getlogop(Lz))

    assert P*L==L*P

    # this is the encoded logical
    l = (half**4)*getlogop(L)
    print("l =")
    print(l)
    #print("l^2 =")
    #gen = [(w8**i)*I for i in range(8)] + [X, Z, Y, S, S.d, H]
    # now we find a name for the encoded logical
    g = mulclose_find(gen, l)
    print("g =")
    print(g, "name =", g.name)

    return

    assert g==l
    assert g**2 == Y

    assert g*X == Z*g
    assert g*Y == Y*g

    assert (S*H)**3 == w8*I
    assert S*H*S.d*H*S.d*H == (w8**7)*Z
    assert g == (w8**7)*Z*H
    print(H)
    print(Z*H)
    print((w8**7)*Z*H)
    return

    G = mulclose([X, Z, Y, S, S.d, H])
    print("|Cliff(1)|=", len(G))
    for g in G:
        if g*X==X*g and g*Z==Z*g and g*Y==Y*g:
            print(g.name)
    
    return

    c1 = Clifford(1)
    H, S = c1.H(), c1.S()
    print(H)
    print(H*H)


def get_inverse(name):
    items = []
    for item in reversed(name):
        stem = item[:item.index("(")]
        if item.endswith(".d"):
            item = item.replace(".d", "")
        elif stem == "S":
            item = item + ".d"
        else:
            assert stem in "H X Z Y CX CZ CY".split()
        items.append(item)
    return tuple(items)


def test_circuit():
    circuit = Circuit(4)
    assert circuit.H(0) == "h q[0];"
    assert circuit.CX(0,2) == "cx q[0], q[2];"
    assert circuit.CY(1,2) == "cy q[1], q[2];"

barrier = ("barrier()",)
measure = ("measure()",)


def gen_412():
    c = Clifford(1)
    gen = [c.X(), c.Z(), c.S(), c.H()]
    G = mulclose(gen)
    assert len(G) == 192
    #print(len(G))
    names = [g.name for g in G]
    #for g in G:
    #    print(g.name)
    return names


def run_412_qasm():
    circuit = Circuit(4)
    encode = ('CY(0,1)', 'CZ(0,2)', 'H(0)', 'CY(1,2)', 'CZ(1,3)',
        'H(1)', 'CZ(2,0)', 'CY(2,3)', 'H(2)', 'S(3)', 'H(3)',
        'S(3).d')
    decode = get_inverse(encode)

    # state prep for logical |0>
    prep = ('Z(0)', 'X(0)', 'H(0)', 'CX(0,3)', 'CY(1,2)', 'H(2)', 'CY(0,1)', 'H(0)', 'H(1)')

    protocol = {}
    #physical = ("P(1,2,3,0)",)
    #logical = ("Z(3)", "H(3)") # inverse logical
    #protocol.append((physical, logical))

    # logical S gate
    physical = (
        'S(0)', 'H(0)', 'S(0)', 
        'S(1)', 'H(1)', 
        'H(2)', 'S(2)', 
        'S(3)', 
        'P(0, 2, 1, 3)', 'X(0)', 'X(2)')
    logical = ("S(3).d",) # inverse logical 
    #protocol.append((physical, logical))
    protocol["S(0)"] = (physical, logical)

    # logical X
    physical = ('Z(0)', 'X(1)',)
    logical = ("X(3)",) # inverse logical 
    protocol["X(0)"] = (physical, logical)
    #protocol.append((physical, logical))

    # logical Z
    physical = ('Z(1)', 'X(2)',)
    logical = ("Z(3)",) # inverse logical 
    protocol["Z(0)"] = (physical, logical)
    #protocol.append((physical, logical))

    # logical H gate
    #physical = ("Z(1)", "X(2)", "P(3,0,1,2)") # XXX the paper uses P(0321)
    #logical = ("H(3)" ) # inverse logical 
    #physical = ("H(0)", "H(1)", "H(2)", "H(3)", "X(0)", "X(2)", "P(0,3,2,1)")
    #logical = ("H(3)", "Z(3)", ) # inverse logical 
    physical = ("Z(1)", "X(2)", "H(0)", "H(1)", "H(2)", "H(3)", "X(0)", "X(2)", "P(0,3,2,1)")
    logical = ("H(3)",) # inverse logical 
    protocol["H(0)"] = (physical, logical)
    #protocol.append((physical, logical))
    del physical, logical

    names = gen_412()
    #names = [nam for nam in names if len(nam)==1]

    N = argv.get("N", 4) # circuit depth
    trials = argv.get("trials", 4)
    print("circuit depth:", N)
    print("trials:", trials)

    if argv.nobarrier:
        global barrier
        barrier = ()

    qasms = []
    for trial in range(trials):
        physical = ()
        logical = ()
        #print("protocol:")
        for i in range(N):
            name = choice(names)
            #print("name:", name)
            p, l = (), ()
            for nami in name:
                p = p + protocol[nami][0]
                l = protocol[nami][1] + l
            #print(p)
            
            physical = barrier + p + physical
            logical = logical + l
    
        # left <---<---< right 
        if argv.encode:
            #print("encode")
            c = measure + logical + decode + physical + barrier + encode
    
        else:
            #print("prep")
            c = measure + logical + decode + physical + barrier + prep
    
        qasm = circuit.run_qasm(c)
        #print(qasm)
        qasms.append(qasm)

    if argv.dump:
        for qasm in qasms:
            print("\n// qasm job")
            print(qasm)
            print("// end qasm\n")

    else:

        kw = {}
        if not argv.get("leakage", False):
            kw['p1_emission_ratio'] = 0
            kw['p2_emission_ratio'] = 0
    
        shots = argv.get("shots", 100)
        samps = send(qasms, shots=shots, N=N, 
            simulator="state-vector", 
            memory_errors=argv.get("memory_errors", False),
            leak2depolar = argv.get("leak2depolar", False),
            **kw,
        )
        process_412(samps)

def variance(p, n):
    return ((p*(1-p))/n)**0.5

def process_412(samps):
    print("samps:", len(samps))
    if not samps:
        return
    succ=samps.count('0000')
    fail=samps.count('0001')
    print("succ: ", succ)
    print("err:  ", len(samps)-succ-fail)
    print("fail: ", fail)
    n = fail+succ
    if n:
        p = (1 - fail / n)
        print("p   = %.6f" % p) 
        print("var = %.6f" % variance(p, n))
    else:
        print("p = %.6f" % 1.)


def load_412():
    samps = load()
    process_412(samps)


def test_822_clifford_unwrap_encoder():
    base = QCode.fromstr("XYZI IXYZ ZIXY")
    print(base)

    tgt = unwrap(base)

    # fix the logicals:
    code = QCode.fromstr("""
    XX...XX.
    .XX...XX
    ..XXX..X
    .ZZ.ZZ..
    ..ZZ.ZZ.
    Z..Z..ZZ
    """, Ls="""
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """)
    assert code.is_equiv(tgt)
    print(code)
    n = code.n

    fibers = [(i, i+base.n) for i in range(base.n)]
    print("fibers:", fibers)

    # 412 state prep for logical |0>
    prep = ('Z(0)', 'X(0)', 'H(0)', 'CX(0,3)', 'CY(1,2)', 'H(2)', 'CY(0,1)', 'H(0)', 'H(1)')

    cover = Cover(base, code, fibers)

    # unwrap 412 state prep... ?
    E = cover.get_expr(prep)
    prep = E.name
    for (i,j) in fibers:
        Hj = ("H(%d)"%(j,),)
        prep = prep + Hj
    print("prep:", prep)
    #return

    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z
    get_perm = c.get_P
    
    v0 = parsevec("0"*n)
    #E = c.get_expr(prep)
    for g in reversed(prep):
        g = c.get_expr(g)
        v0 = g*v0
    #print(v0)

    for op in """
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """.strip().split():
        print(op)
        g = c.get_pauli(op)
        v1 = g*v0
        #u0 = v0+v1
        #u1 = v0-v1
        print(g*v0 == v0)

    # we have prepared the |+0> state doh !

    P = code.get_projector()
    assert P*v0 == v0


def test_822_clifford():
    base = QCode.fromstr("XYZI IXYZ ZIXY")
    print(base)

    tgt = unwrap(base)

    # fix the logicals:
    code = QCode.fromstr("""
    XX...XX.
    .XX...XX
    ..XXX..X
    .ZZ.ZZ..
    ..ZZ.ZZ.
    Z..Z..ZZ
    """, Ls="""
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """)
    assert code.is_equiv(tgt)
    print(code)
    n = code.n

    # row reduced X stabilizer/logops
    S0 = parse("""
    X......X
    .X..X...
    XX...XX.
    .XX...XX
    ..XXX..X
    """)
    SX = parse("""
    X......X
    .X..X...
    ..X.X.XX
    ...X..X.
    ....XXXX
    """)

    assert shortstr(SX) == shortstr(row_reduce(S0))
    

    c = Clifford(n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
    I = c.get_identity()
    SHS = lambda i:S(i)*H(i)*S(i)
    SH = lambda i:S(i)*H(i)
    HS = lambda i:H(i)*S(i)
    X, Y, Z = c.X, c.Y, c.Z
    get_perm = c.get_P
    cx, cz = CX, CZ

    if 0:
        fibers = [(i, i+base.n) for i in range(base.n)]
        print("fibers:", fibers)
        cover = Cover(base, code, fibers)
        expr = tuple("S(0) H(0) S(0) S(1) H(1) H(2) S(2) S(3) P(0,2,1,3) X(0) X(2)".split())
        gate = cover.get_expr(expr).name
        print("gate:", gate)
        E = c.get_expr(gate)
        print(E1 == E)

    if argv.prep_00:
        # row reduced Z stabilizer/logops
        SZ = """
        01234567
        Z....Z..
        .ZZ.ZZ..
        ..ZZ.ZZ.
        ...ZZ...
        ....ZZZZ
        """

        # prepare |00> state
        g =   cx(5,0) # src,tgt
        g = g*cx(2,1)*cx(4,1)*cx(5,1)
        g = g*cx(3,2)*cx(5,2)*cx(6,2)
        g = g*cx(4,3)
        g = g*cx(5,4)*cx(6,4)*cx(7,4)
        g = g*H(5)*H(6)*H(7)

    elif argv.prep_pp:
        ("""
        01234567
        X......X
        .X..X...
        ..X.X.XX
        ...X..X.
        ....XXXX
        """)
        g =   cx(0,7)
        g = g*cx(1,4)
        g = g*cx(2,4)*cx(2,6)*cx(2,7)
        g = g*cx(3,6)
        g = g*cx(4,5)*cx(4,6)*cx(4,7)
        g = g*H(0)*H(1)*H(2)*H(3)*H(4)

    else:
        return

    print("prep:", g.name)
    #return

    # logical's
    LX0 = X(0)*X(7)
    LX1 = X(1)*X(4)
    LZ0 = Z(0)*Z(5)
    LZ1 = Z(3)*Z(4)

    gate = """
    P(0,4,2,6,1,5,3,7)
    CX(4,0)
    SWAP(1,5)
    CX(1,5)
    CX(2,6)
    SWAP(2,6)
    CX(3,7)
    """.strip().split()
    gate = tuple(reversed(gate))

    # See test.test_822
    if 0:
        # lifted 1,0,2,3
        gate = ('CX(7,3)', 'CX(2,6)', 'P(0,5,2,3,4,1,6,7)', 'CX(1,5)',
            'CX(0,4)', 'P(4,1,2,3,0,5,6,7)', 'P(1,0,2,3,5,4,6,7)')
    else:
        # lifted 0,2,1,3 is Dehn twist
        gate = ('CX(3,7)', 'P(0, 1, 6, 3, 4, 5, 2, 7)', 'CX(2,6)', 'CX(1,5)', 'P(0, 5, 2, 3, 4, 1, 6, 7)', 'CX(4,0)', 'P(0, 2, 1, 3, 4, 6, 5, 7)')
    print("gate:", gate)
    E = c.get_expr(gate)

    v0 = parsevec("00000000")
    v0 = g*v0 # |00>

    if argv.tom:
        #v0 = LX0 * v0 # |10>
        v0 = LX1 * v0 # |01>
        u0 = E*v0
        print( u0==v0 )
    
        for l in [I, LX0, LZ0]:
          for r in [I, LX1, LZ1]:
            op = l*r
            v1 = op*v0
            print(int(v1==v0), int(v1==-v0), op.name)

    # still in groundspace
    P = code.get_projector()
    assert P*E == E*P
    assert P*v0 == v0


def opt_822_prep():
    # from test_822_clifford:
    prep = (
        'CX(5,0)', 'CX(2,1)', 'CX(4,1)', 'CX(5,1)', 'CX(3,2)',
        'CX(5,2)', 'CX(6,2)', 'CX(4,3)', 'CX(5,4)', 'CX(6,4)', 'CX(7,4)', )
        #'H(5)', 'H(6)', 'H(7)')

    n = 8
    s = SymplecticSpace(n)
    H, CX = s.H, s.CX
    ops = [eval(name, {"CX":CX, "H":H}) for name in prep]
    tgt = reduce(mul, ops)
    print(tgt)

    gen = [CX(i,j) for i in range(n) for j in range(n) if i!=j]
    print(len(gen))

    metric = lambda g : str(g+tgt).count('1')

    best_count = 20
    while 1:
        g = s.get_identity()
        d = metric(g)
        while 1:
            done = True
            best = None
            best_d = d
            hgs = [h*g for h in gen]
            ds = [metric(hg) for hg in hgs]
            d0 = min(ds)
            if d0 >= d:
                break
            hgs = [hg for hg in hgs if metric(hg)==d0]
            g = choice(hgs)
            d = d0
    
        if d!=0:
            continue

        count = len(g.name)
        if count < best_count:
            print(g.name, count)
            best_count = count



def run_822_qasm():
    base = QCode.fromstr("XYZI IXYZ ZIXY")
    #print(base)

    tgt = unwrap(base)

    # fix the logicals:
    code = QCode.fromstr("""
    XX...XX.
    .XX...XX
    ..XXX..X
    .ZZ.ZZ..
    ..ZZ.ZZ.
    Z..Z..ZZ
    """, Ls="""
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """)
    assert code.is_equiv(tgt)
    #print(code)

    fibers = [(i, i+base.n) for i in range(base.n)]
    #print("fibers:", fibers)

    # from test_822_clifford:
    # prep |00> state
    prep_00 = ('CX(5,0)', 'CX(2,1)', 'CX(4,1)', 'CX(5,1)', 'CX(3,2)',
        'CX(5,2)', 'CX(6,2)', 'CX(4,3)', 'CX(5,4)', 'CX(6,4)',
        'CX(7,4)', 'H(5)', 'H(6)', 'H(7)')

    if argv.opt:
        # from opt_822_prep, not as good 
        prep_00 = ('CNOT(5,3)', 'CNOT(5,4)', 'CNOT(4,1)', 'CNOT(2,1)', 'CNOT(6,4)', 'CNOT(6,3)', 'CNOT(3,2)', 'CNOT(4,3)', 'CNOT(7,4)', 'CNOT(5,0)')
        prep_00 += ('H(5)', 'H(6)', 'H(7)')

    # from test_822_clifford
    # prepare |++> state
    prep_pp = ('CX(0,7)', 'CX(1,4)', 'CX(2,4)', 'CX(2,6)', 'CX(2,7)', 'CX(3,6)', 'CX(4,5)', 'CX(4,6)', 'CX(4,7)', 
        'H(0)', 'H(1)', 'H(2)', 'H(3)', 'H(4)')

    # lifted 0,2,1,3 from test_822_clifford is Dehn twist
    gate = ('CX(3,7)', 'P(0,1,6,3,4,5,2,7)', 'CX(2,6)', 'CX(1,5)',
        'P(0,5,2,3,4,1,6,7)', 'CX(4,0)','P(0,2,1,3,4,6,5,7)')

    """
    01234567
    X......X
    Z....Z..
    .X..X...
    ...ZZ...
    """
    I = ()
    X0 = ("X(0)", "X(7)")
    X1 = ("X(1)", "X(4)")
    Z0 = ("Z(0)", "Z(5)")
    Z1 = ("Z(3)", "Z(4)")

    n = code.n
    circuit = Circuit(n)

    #c = measure + logical + decode + physical + barrier + prep_00
    #c = measure + X1 + barrier + (gate + barrier) + barrier + X1 + prep_00
    #c = measure + Z0 + barrier + (gate + barrier) + barrier + Z0 + prep_00

    qasms = []
    def tomography():
        for op in [
            I,   
            X0    ,
            Z0    ,
                X1,
                Z1,
            X0+Z0,
            X0+X1,
            X0+Z1,
            Z0+X1,
            Z0+Z1,
            X1+Z1,
        ]:
            #c = measure + op + barrier + (gate + barrier) + barrier + Z0 + prep_00
            c = measure + op + prep_00
            qasm = circuit.run_qasm(c)
            #print(qasm)
            qasms.append(qasm)
        #return

    if 0:
        # these work:
        c = measure + X0 + barrier + (gate + barrier) + barrier + X0 + prep_00
        c = measure + X0+X1 + barrier + (gate + barrier) + barrier + X1 + prep_00
        c = measure + Z0+Z1 + barrier + (gate + barrier) + barrier + Z0 + prep_00
        c = measure + Z1 + barrier + (gate + barrier) + barrier + Z1 + prep_00
        c = measure + X0+X1+Z1 + barrier + (gate + barrier) + barrier + X1+Z1 + prep_00

    h8 = tuple("H(%d)"%i for i in range(n))

    if argv.spam:
        if argv.prep_00:
            c = measure + barrier + prep_00 # SPAM
        elif argv.prep_pp:
            c = measure + h8 + barrier + prep_pp # SPAM
        else:
            return
    elif argv.state == (0,0):
        c = measure + barrier + gate + barrier + prep_00
    elif argv.state == (1,0):
        c = measure + X0 + barrier + gate + barrier + X0 + prep_00
    elif argv.state == (0,1):
        c = measure + X0+X1 + barrier + gate + barrier + X1 + prep_00
    elif argv.state == (1,1):
        c = measure + X1 + barrier + gate + barrier + X0+X1 + prep_00
    elif argv.state == "pp":
        c = measure + h8 + barrier + gate + barrier + prep_pp
    elif argv.state == "mp":
        c = measure + h8 + Z0+Z1 + barrier + gate + barrier + Z0 + prep_pp
    elif argv.state == "pm":
        c = measure + h8 + Z1 + barrier + gate + barrier + Z1 + prep_pp
    elif argv.state == "mm":
        c = measure + h8 + Z0 + barrier + gate + barrier + Z0 + Z1 + prep_pp
    else:
        return

    print(c)

    qasms.append(circuit.run_qasm(c))

    if argv.load:
        samps = load()

    else:
        shots = argv.get("shots", 10)
        samps = send(qasms, shots=shots, error_model=True)
        #print(samps)

    idxs = circuit.labels # final qubit permutation

    if type(argv.state) is str or argv.prep_pp:
        #print("measure X syndromes")
        H = parse("""
        XX...XX.
        .XX...XX
        ..XXX..X
        X......X
        .X..X...
        """)
    else:
        #print("measure Z syndromes")
        assert type(argv.state) is tuple or argv.prep_00
        H = parse("""
        .ZZ.ZZ..
        ..ZZ.ZZ.
        Z..Z..ZZ
        Z....Z..
        ...ZZ...
        """)
    H = H[:, idxs] # shuffle

    #print(H)
    succ = 0
    fail = 0
    for i,v in enumerate(samps):
        v = parse(v)
        u = dot2(H, v.transpose())
        if u.sum() == 0:
            succ += 1
        elif u[:3].sum() == 0:
            fail += 1
        if argv.show:
            print(shortstr(u.transpose()), end=" ")
            if (i+1)%20==0:
                print()
    #print()
    #get_syndrome(samps)

    print("samps:", len(samps))
    if not samps:
        return

    print("succ: ", succ)
    print("err:  ", len(samps)-succ-fail)
    print("fail: ", fail)
    shots = fail+succ
    if shots:
        p = (1 - fail / (fail+succ))
        print("p   = %.6f" % p)
        print("var = %.6f" % variance(p, shots))


def get_syndrome(S):
    from pecos import BinArray #This is just a cute function is PECOS that reverses results so you look at them in a more human readable direction
    
    #Simple processing code
    #Calculate logical operator, calculate syndromes
    #Post-select on non-trivial syndromes
    #Calculate logical fidelity
    suc1 = 0
    suc2 = 0
    suc3 = 0
    suc_count = 0
    for j in range(len(S)):
        raw_out = numpy.array([int(i) for i in BinArray(S[j])])
        S1 = raw_out[1]^raw_out[2]^raw_out[4]^raw_out[5]
        S2 = raw_out[2]^raw_out[3]^raw_out[5]^raw_out[6]
        S3 = raw_out[0]^raw_out[3]^raw_out[7]^raw_out[6]
        S4 = raw_out[0]^raw_out[1]^raw_out[7]^raw_out[4]
    
    
        log_1 = raw_out[3]^raw_out[4]
        log_2 = raw_out[0]^raw_out[5]
        log_3 = raw_out[2]^raw_out[7]
        if S1 == 0 & S2 == 0 & S3 == 0 & S4==0:
            #print('succes!')
            suc_count +=1
            if log_1 == 0:
                suc1 += 1
    
            if log_2 == 0:
                suc2 += 1
            if log_3 == 0:
                suc3 += 1
    
    print(suc1/suc_count)
    print(suc2/suc_count)
    print(suc3/suc_count)
    

def test_513():
    base = construct.get_513()
    n = base.n
    print(base)
    print(base.longstr())

    s = base.space
    c = Clifford(n)
    E = reduce(mul, [c.get_CZ(i, (i+1)%n) for i in range(n)])
    v0 = E*red(n, 0, 0)
    v1 = E*red(n, 0, 2)
    assert (v0.d*v1)[0][0] == 0 # orthogonal

    P = base.get_projector()
    assert P*P == P
    assert P*v0 == v0
    assert P*v1 == v1

    lx = -c.get_pauli("XXXXX") # arf, what is this minus sign ??
    lz = c.get_pauli("ZZZZZ")
    #ly = c.get_pauli("YYYYY")
    ly = lz*lx
    assert lz*v0 == v0
    assert lz*v1 == -v1
    assert lx*v0 == v1
    assert lx*v1 == v0

    S, H = c.S, c.H
    g = reduce(mul, [S(i)*H(i) for i in range(n)])
    assert g*P == P*g
    u = g*v0
    print(lz*u==u, lz*u==-u)
    print(lx*u==u, lx*u==-u)
    print(ly*u==u, ly*u==-u)


def symplectic_10_2_3():
    base = construct.get_513()
    code = unwrap(base)
    print(code.longstr())
    #css = code.to_css()
    perms = get_autos(code)
    print(perms)
    swap = SymplecticSpace(2).get_SWAP()
    for perm in perms:
        dode = code.apply_perm(perm)
        assert dode.is_equiv(code)
        if dode.get_logical(code) == swap:
            print(perm)


def clifford_10_2_3():
    base = construct.get_513()
    code = unwrap(base)
    n = base.n
    nn = code.n
    #print(code.longstr())
    print(base)
    print(code)

    if 0:
        lhs = red(n, 0)
        rhs = reduce(matmul, [red(1,0)]*n)
        print(lhs == rhs)
        print(lhs)
        print(rhs)
    
        return

    P = code.get_projector()
    #assert P*P == P # SLOOOW

    cc = Clifford(nn)
    CX = cc.CX
    H = code.H
    H = strop(H).split()
    #for h in H:
    #    print(h)
    #    op = cc.get_pauli(h) # SLOOOW

    def unwrap_CZ(src, tgt, v):
        #return CX(2*src, 2*tgt+1) * (CX(2*src+1, 2*tgt) * v)
        return CX(src, tgt+n) * (CX(tgt, src+n) * v)

    #v = red(nn, 0)
    v = red(n, 0) @ green(n, 0)
    print("v", v.shape)
    for i in range(n):
        v = unwrap_CZ(i, (i+1)%n, v)
        print("i =", i)
    print(P*v == v)

    Z0 = cc.get_pauli("ZZZZZ.....")
    assert Z0*v == v
    
    X1 = cc.get_pauli("..XX.X....")
    assert X1*v == v



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

