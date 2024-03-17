#!/usr/bin/env python

import warnings
warnings.filterwarnings('ignore')


from random import shuffle, randint
from operator import add, matmul, mul
from functools import reduce

import numpy

from qumba.solve import (parse, shortstr, linear_independent, eq2, dot2, identity2,
    rank, rand2, pseudo_inverse, kernel, direct_sum)
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

from qumba.clifford import Clifford, red, green, K, Matrix, r2, ir2, w4, w8, half, latex




def send(qasm=None, shots=10):
    from qjobs import QPU, Batch
    qpu = QPU("H1-1E", domain="prod", local=True)
    # Setting local=True means that qjobs will use PECOS to run simulations on your device
    # Otherwise, it will attempt to connect to a device in the cloud
    
    batch = Batch(qpu)
    
    
    if qasm is None:
        # create & measure Bell state
        qasm = """
        OPENQASM 2.0;
        include "hqslib1.inc";
        
        qreg q[2];
        creg m[2];
        
        h q[0];
        
        CX q[0], q[1];
        
        measure q -> m;
        """
    
    # We can append jobs to the Batch object to run
    batch.append(qasm, shots=shots, options={"simulator": "stabilizer"})
    
    # Submit all previously unsubmitted jobs to the QPU
    batch.submit()
    # Note: Each time you submit or retriece jobs, 
    # the Batch object will save itself as a pickle
    
    # Retrieve
    batch.retrieve()
    
    print(batch.jobs)

    for job in batch.jobs:
        r = job.results
        print(r["results"])
    
    if 0:
        #To get an individual job object you can use indexes from
        #this list of jobs. Or use a job's job id like this:
        j = batch["local60bb85f56c0b4d8ca6bebe49525c9373"]
        print(j.code)
        j.results
        batch["localeb2dbe1147fe49db80ede0a289c6008f"].params


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

    def P(self, *idxs):
        assert len(idxs) == self.n
        self.labels = list(idxs)
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
        return qasm


def parsevec(s):
    from qumba.clifford import red, w4
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
    s = s.replace('.', '0')
    assert len(s) == s.count('0')+s.count('1') # etc
    lookup = {
        '0' : red(1,0,0),
        '1' : red(1,0,2),
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


def test_412_clifford():
    code = QCode.fromstr("XYZI IXYZ ZIXY") # 412 code
    c = Clifford(code.n)
    CX, CY, CZ, H, S = c.CX, c.CY, c.CZ, c.H, c.S
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

    if 0: # nope..
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
        assert P*P == P
        #print(v0)
        #print(P*v0)
        assert P*v0 == v0
        assert P*v1 == v1
        x = (v0.d * v1)[0][0]
        assert x==0

    X, Y, Z = c.X, c.Y, c.Z
    Lx = Z(0) * X(1)
    Lz = Z(1) * X(2)
    assert Lx * Lz == -Lz * Lx
    assert Lx*P == P*Lx
    assert Lz*P == P*Lz

    #v0 = 2*P*parsevec("0000") # eigenvector of Ly 
    #v0 = 2*P*parsevec("0000 + 0001") # eigenvector of Lx
    #v0 = 2*P*parsevec("0000 + -1*0001") # eigenvector of Lx
    _v0 = 2*P*parsevec("0000 + i*0001") # +1 eigenvector of Lz, yay!
    _v1 = Lx*_v0

    x = (_v0.d * _v1)[0][0]
    assert x==0

    print(strvec(_v0))
    print(strvec(_v1))

    dot = lambda l,r : (l.d*r)[0][0]

    v0, v1 = _v0, _v1
    #M = (half**4)*Matrix(K,[[dot(u0,v0),dot(u0,v1)],[dot(u1,v0),dot(u1,v1)]])
    #print(M)
    assert v0 == u0


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

    L = get_perm(1,2,3,0)
    assert P*L==L*P
    assert L*Lx*L.d == Lz
    l = (half**4)*getlogop(L)
    print("l =")
    print(l)
    print("l^2 =")
    assert l**2 == Y
    #print()
    gen = [(w8**i)*I for i in range(8)] + [X, Z, Y, S, S.d, H]
    g = mulclose_find(gen, l)
    print("g =")
    print(g, g.name)
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


def test_qasm():
    circuit = Circuit(4)
    assert circuit.H(0) == "h q[0];"
    assert circuit.CX(0,2) == "cx q[0], q[2];"
    assert circuit.CY(1,2) == "cy q[1], q[2];"

    encode = ('CY(0,1)', 'CZ(0,2)', 'H(0)', 'CY(1,2)', 'CZ(1,3)',
        'H(1)', 'CZ(2,0)', 'CY(2,3)', 'H(2)', 'S(3)', 'H(3)',
        'S(3).d')
    decode = get_inverse(encode)
    barrier = ("barrier()",)
    measure = ("measure()",)
    perm = ("P(1,2,3,0)",)
    #perm = ("P(1,2,0,3)",)

    c = measure + decode + perm + barrier + encode

    qasm = circuit.run_qasm(c)
    print(qasm)

    send(qasm, shots=100)



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

