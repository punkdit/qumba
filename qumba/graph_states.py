#!/usr/bin/env python
"""
see also: lax.py
"""



from random import shuffle
from math import sin, cos, pi
from functools import cache, reduce
from operator import add

import numpy

from qumba.action import mulclose
from qumba.symplectic import SymplecticSpace
from qumba.matrix import Matrix
from qumba.qcode import QCode, fromstr, strop
from qumba.smap import SMap
from qumba import lin
from qumba.argv import argv
from qumba.util import binomial, factorial
from qumba import construct



def test():

    n = argv.get("n", 3)

    space = SymplecticSpace(n)
    S = space.S
    H = space.H
    I = space.get_identity()
    CX = space.CX
    CZ = space.CZ

    P = I
    for i in range(n):
        P = P*H(i)
    print(P)

    gen  = [S(i) for i in range(n)]
    gen += [H(i) for i in range(n)]

    LC = mulclose(gen)
    print("|LC| =", len(LC))

    orbits = []
    found = []
    for E in [
        I, 
        CZ(0,1),
        CZ(0,2), CZ(1,2), 
        CZ(0,1) * CZ(1,2),
        CZ(0,2) * CZ(1,2),
    ]:
        orbit = []
        overlap = 0
        src = QCode.from_encoder(E)
        for g in LC:
            code = g*src
            for dode in orbit:
                if code.is_equiv(dode):
                    break
            else:
                orbit.append(code)
                for dode in found:
                    if dode.is_equiv(code):
                        print("*", end="")
                        overlap += 1
                        break
                else:
                    found.append(code)
        if overlap:
            print((overlap))
        print("orbit =", len(orbit))
        orbits.append(orbit)

    print(sum(len(o) for o in orbits))
    #print(54 + 3*18 + 27)
    print(len(found))




def get_graph(code): # XXX SLOW
    if isinstance(code, QCode):
        assert code.k == 0, "not a state"
        #print(code.longstr(False))
        #print()
        n = code.n
        H = code.H
    else:
        assert isinstance(code, Matrix)
        H = code
        n, nn = H.shape
        assert 2*n==nn

    H = list(H.rowspan())
    shuffle(H)
    H = Matrix(H)
    H = H.reshape(2**n, 2*n)
    s = strop(H)
    stabs = s.split("\n")
    idxs = set()
    A = lin.zeros2(n,n)
    for stab in stabs:
        if stab.count("X") != 1:
            continue
        if "Y" in stab:
            continue
        #print(stab)
        idx = stab.index("X")
        if idx in idxs:
            return
        idxs.add(idx)
        for i,s in enumerate(stab):
            if s=='Z':
                A[idx,i] = 1
    if len(idxs) != n:
        return
    A = Matrix(A)
    #print(A)
    if A==A.t:
        return A



def find_lagrangian(): # SLOW
    n = 4
    code = QCode.fromstr(""" XZZZ ZZXZ ZXZZ ZZZX """)
    code = QCode.fromstr(""" XIZI ZIXZ IXIZ IZZX """) # U shape
    pode = QCode.fromstr(""" XZZI ZXIZ ZIXZ IZZX """)
    #pode = QCode.fromstr(""" XZZZ ZXII ZIXI ZIIX """)

    assert get_graph(pode) is not None

    H = pode.H.normal_form()
    assert QCode(H).is_equiv(pode)

    found = set()
    for code in construct.all_codes(4,0,0):
        H = code.H.normal_form()
        dode = QCode(H)
        found.add(dode)
        assert dode.is_equiv(code)
    print(H.shape)
    print(len(found))

    space = code.space
    gen = [space.S(i) for i in range(n)]
    gen += [space.H(i) for i in range(n)]
    G = mulclose(gen)
    print("|G| =", len(G))

    orbits = []
    remain = set(found)
    graphs = set()
    while remain:
        code = remain.pop()
        orbit = [code]
        for g in G:
            dode = g*code
            H = dode.H.normal_form()
            eode = QCode(H)
            assert eode.k == 0
            assert eode.is_equiv(dode)
            assert eode in found
            if eode in remain:
                remain.remove(eode)
                orbit.append(eode)
        orbits.append( orbit )
    orbits.sort(key = len)

    for orbit in orbits:
        col = 0
        smap = SMap()
        for code in orbit:
            A = get_graph(code)
            if A is not None:
                graphs.add(A)
                smap[0, col] = str(A)
                col += n+1
        print(len(orbit))
        print(smap)
        print()

    print("graphs:", len(graphs))


def all_graphs(n):
    idxs = [(i,j) for i in range(n) for j in range(i+1,n)]
    assert len(idxs) == n*(n-1)//2
    N = len(idxs)

    graphs = set()
    S = numpy.empty(shape=(n,n), dtype=int)
    for bits in numpy.ndindex((2,)*N):
        S[:] = 0
        for (i,bit) in enumerate(bits):
            if bit==0:
                continue
            j,k = idxs[i]
            S[j,k] = 1
            S[k,j] = 1
        graph = Matrix(S)
        assert graph not in graphs
        graphs.add(graph)
        yield graph


@cache
def get_bits(n):
    bits = tuple(numpy.ndindex((2,)*n))
    assert len(bits) == 2**n
    return bits


@cache
def get_idxs(n):
    idxss = []
    for bits in get_bits(n):
        idxs = tuple(i for (i,ii) in enumerate(bits) if ii==1)
        idxss.append(idxs)
    return tuple(idxss)


def upper_signature(H):
#    print("upper_signature")
#    print(H)
    m, nn = H.shape
    assert nn%2==0
    n = nn//2
    H = H.reshape(m, n, 2)
    counts = []
    for cols in get_idxs(n):
        H1 = H[:, cols, :]
        r = H1.reshape(m, len(cols)*2).rank()
#        print("\t", cols, r)
        counts.append(r)
    counts = tuple(counts)
#    print("\t =", counts)
    return counts


def lower_signature(H):
#    print("lower_signature")
#    print(H)
    m, nn = H.shape
    assert nn%2==0
    n = nn//2
    rows = []
    for i in range(n):
        row = lin.zeros2(2, nn)
        row[0,2*i] = 1
        row[1,2*i+1] = 1
        rows.append(row)
        #print(row)

    counts = []
    for cols in get_idxs(n):
#        print(cols)
        A = lin.zeros2(2*len(cols), nn)
        for i,ii in enumerate(cols):
            A[2*i:2*i+2, :] = rows[ii]
#        print(A)
        B = lin.intersect(H.A, A)
#        print(B)
        r = len(B)
        counts.append(r)
    counts = tuple(counts)
#    print("\t =", counts)
    return counts


def test_ul_signature():
    for n in [1,2,3,4]:
      for m in range(n+1):
        #print(n, m)
        k = n-m
        print("n=%d"%n, "m=%d"%m)
        found = set()
        lsigs = set()
        usigs = set()
        count = 0
        for code in construct.all_codes(n, k, 0):
            count += 1
            H = code.H
            H = H.normal_form()
            found.add(H)
            lsig = (lower_signature(H))
            usig = (upper_signature(H))
            for (i,j) in zip(lsig, usig):
                assert i<=j
            lsigs.add(lsig)
            usigs.add(usig)
        lsigs = list(lsigs)
        lsigs.sort()
        #for sig in lsigs:
        #    print("\t", sig)
        print(len(found), len(lsigs), len(usigs))
        assert len(lsigs) == len(usigs)



def test_signature():

    code = construct.get_713()
    #code = construct.get_422()
    #print(code.longstr())
    H = code.H

    sigstr = lambda sig:(''.join(str(i) for i in sig))

    print(sigstr(lower_signature(H)))
    print(sigstr(upper_signature(H)))
    print()

    #return

    L = code.L
    #print(L)
    usigs = set()
    lsigs = set()
    for op in L.rowspan():
        if op.sum() == 0:
            continue
        J = H.concatenate(op)
        lsigs.add(lower_signature(J))
        usigs.add(upper_signature(J))
    print(len(lsigs), len(usigs))
    l = lsigs.pop()
    print(sigstr(l))
    u = usigs.pop()
    print(sigstr(u))


def orbit_isotropic(n, found):
    space = SymplecticSpace(n)
    gens = []
    if argv.perms:
        gens = [space.SWAP(i,i+1).t for i in range(n-1)]
    for i in range(n):
        gens.append(space.S(i))
        gens.append(space.H(i))
    remain = set(found)
    found = set()
    orbits = []
    while remain:
        H = remain.pop()
        orbit = [H]
        found.add(H)
        bdy = list(orbit)
        while bdy:
            _bdy = []
            for H in bdy:
              for g in gens:
                J = (H*g).normal_form()
                if J in found:
                    continue
                _bdy.append(J)
                found.add(J)
                remain.remove(J)
                orbit.append(J)
            bdy = _bdy
        orbits.append(orbit)
    return orbits


def test_codes():

#    #for n in [2,3]:
#      for k in range(n+1):
    #for (n,m) in [(2,1), (3,2), (4,3)]:
    for n in [1,2,3,4, 5]:
      print("n=%d"%n, end=' ', flush=True)
      for m in range(n+1):
        #print(n, m)
        k = n-m
        found = set()
        #sigs = set()
        count = 0
        for code in construct.all_codes(n, k, 0):
            count += 1
            H = code.H
            H = H.normal_form()
            found.add(H)
            #sigs.add(upper_signature(H))
        orbits = orbit_isotropic(n, found)
        #print([len(o) for o in orbits], len(orbits))
        orbits.sort(key = len)
        s = [len(o) for o in orbits]
        #print("%d:%d:%s"%(count, len(orbits), s), end=' ', flush=True)
        print(len(orbits), end=' ', flush=True)
        #assert len(sigs) == len(orbits)
        #sigs = list(sigs)
        #sigs.sort()
        #print("sigs:", len(sigs))
        #for sig in sigs:
        #    print("\t", sig)
      print()



def all_graphs_perm(n):
    #from bruhat.gset import Group, Perm
    #G = Group.symmetric(n)
    #print(len(G))
    #assert 0, "untested ..."

    gens = []
    for i in range(n-1):
        g = list(range(n))
        g[i], g[i+1] = i+1,i
        gens.append(g)

    found = set()
    for A in all_graphs(n):
        if A not in found:
            yield A
        orbit = {A}
        bdy = [orbit]
        while bdy:
            _bdy = []
            for g in gens:
                B = A[g,:]
                B = B[:,g]
                if B in found:
                    continue
                orbit.add(B)
                _bdy.append(B)
                found.add(B)
            bdy = _bdy


def all_graph_states(n):
    S = numpy.empty(shape=(n,n), dtype=object)
    S[:] = '.'
    I,X,Z = ".XZ"

    def shortstr(S):
        return ("\n".join(''.join(row) for row in S))

    idxs = [(i,j) for i in range(n) for j in range(i+1,n)]
    assert len(idxs) == n*(n-1)//2
    N = len(idxs)

    graphs = set()
    for bits in numpy.ndindex((2,)*N):
        S[:] = I
        for i in range(n):
            S[i,i] = X

        for (i,bit) in enumerate(bits):
            if bit==0:
                continue
            j,k = idxs[i]
            S[j,k] = Z
            S[k,j] = Z

        s = shortstr(S)
        #print(s)
        #code = QCode.fromstr(s)
        #H = code.H.normal_form()
        H = fromstr(s)
        H = Matrix(H).normal_form()
        #print(H)
        #print()
        graphs.add(H)

    assert len(graphs) == 2**N
    return graphs


def lc_graph_states(n, verbose=True):

    # first we build all the graph states
    graphs = all_graph_states(n)

    if verbose:
        print("graphs:", len(graphs))

    space = SymplecticSpace(n)
    gens = [space.S(i) for i in range(n)]
    gens += [space.H(i) for i in range(n)]

    #graphs = list(graphs)
    orbits = []
    while graphs:
        H = graphs.pop()
        orbit = {H}
        bdy = [H]
        #A = get_graph(H)
        #print(A)
        while bdy:
            _bdy = []
            for H in bdy:
              #code = QCode(H)
              for g in gens:
                J = H*g.t
                #dode = g*code
                #J = dode.H
                #assert J == H*g.t
                J = J.normal_form()
                if J in orbit:
                    continue
                #B = get_graph(J) 
                if J in graphs:
                    graphs.remove(J)
                _bdy.append(J)
                orbit.add(J)
            bdy = _bdy
        orbits.append(orbit)
        if verbose:
            print(len(orbit), end=' ', flush=True)
    if verbose:
        print()

    orbits.sort(key=len)
    return orbits



def get_func(A):
    n = len(A)
    assert A.shape == (n,n)
    ranks = []
    P = get_bits(n)
    for idxs in P:
        rows = [i for (i,ii) in enumerate(idxs) if ii==1]
        cols = [i for (i,ii) in enumerate(idxs) if ii==0]
        B = A[rows, :]
        B = B[:, cols]
        ranks.append(B.rank())
    return tuple(ranks)


def test_perms():
    for n in [2,3,4,5,6]:
        graphs = list(all_graphs_perm(n))
        print(n, len(graphs))


def test_funcs():
    n = argv.get("n", 3)
    
    graphs = list(all_graphs_perm(n))
    print(n, len(graphs))

    funcs = {get_func(A) for A in graphs}
    print(len(funcs))

    orbits, action = get_orbits(n, funcs)
    print(len(orbits))
    print()

    funcs = [orbit[0] for orbit in orbits]
    funcs.sort()
    for func in funcs:
        print(func)


def test_graphs():
    # here we try summing over all the LC equivalent graphs

    n = argv.get("n", 4)
    graphs = all_graph_states(n)
    orbits = lc_graph_states(n)

    gens = []
    for i in range(n-1):
        g = list(range(n))
        g[i], g[i+1] = i+1,i
        gens.append(g)

    from bruhat.gset import Group, Perm
    G = Group.symmetric(n)

    found = set()
    count = 0
    for orbit in orbits:

        # get the graphs only
        orbit = [M for M in orbit if M in graphs] # count multiple copies (are there any?)

        orbit = [get_graph(M).A.astype(int) for M in orbit]

        A = reduce(add, orbit)
        key = str(A)
        if key in found:
            continue

        H = []
        for g in G:
            B = A[g, :]
            B = B[:, g]
            ley = str(B)
            if ley == key:
                H.append(g)
            found.add(ley)
        H = Group(H)

        print(key, len(H), H.structure_description())
        count += 1

        #print(len(orbit), end=' ', flush=True)

    print()
    print("orbits:", len(orbits))
    print("keys:", count)



def main():
    n = argv.get("n", 3)

    if argv.dump:
        #graphs = all_graph_states(n)
        graphs = list(all_graphs(n))
        #graphs = list(all_graphs_lc(n)) # TODO
        print(len(graphs))

        #H = graphs.pop()
        #print(H)
        #A = get_graph(H)
        #print(A)

        funcs = set()
        count = 0
        for A in graphs:
            #A = get_graph(H)
            #assert A is not None
            func = get_func(A)
            if func not in funcs:
                funcs.add(func)
                count += 1
                if count%100==0:
                    print('.', flush=True,end='')
        print()
        print(len(funcs))

        f = open("funcs_%d.py"%n, "w")
        funcs = list(funcs)
        funcs.sort()
        for func in funcs:
            print(func, file=f)
        f.close()

        return

    if n>4:
        lines = open("funcs_%d.py"%n).readlines()
        funcs = [eval(line) for line in lines]
        print("read %d funcs"%len(funcs))

    else:

        graphs = all_graph_states(n)
        orbits = lc_graph_states(n)
    
        funcs = set()
        count = 0
        for orbit in orbits:
            orbit = [M for M in orbit if M in graphs]
    
            func = None
            for M in orbit:
                A = get_graph(M)
                gunc = get_func(A)
                assert func is None or func==gunc
                func = gunc
            assert func
            funcs.add(func)
            count += len(orbit)
    
            print(len(orbit), end=' ', flush=True)
        print()
        print("total:", count)
        print("orbits:", len(orbits))


    print("funcs:", len(funcs))

    if argv.show_structure:
        show_structure(n, funcs)
        return

    if argv.structures:
        structures(n, funcs)

    if argv.show_poset:
        show_poset(n, funcs)
        return

    if argv.show_stats:
        show_stats(n, funcs)
        return

    if argv.plot_funcs:
        plot_funcs(n, funcs)

    if argv.render_funcs:
        render_funcs(n, funcs)


def get_orbits(n, funcs):
    from bruhat.gset import Group, Perm

    P = list(numpy.ndindex((2,)*n))
    N = factorial(n)

    G = Group.symmetric(n)
    assert len(G) == N
    print("%d!=%d"%(n, N))
    perms = []
    action = {}
    idxs = P
    lookup = {idx:i for (i,idx) in enumerate(idxs)}
    for g in G:
        perm = []
        for idx in idxs:
            jdx = tuple(idx[g[i]] for i in range(n))
            perm.append(lookup[jdx])
        perm = Perm(perm)
        action[g] = perm
        perms.append(perm)

    #print()
    remain = set(funcs)
    orbits = []
    while remain:
        func = remain.pop()
        orbit = [func]
        for perm in perms:
            gunc = tuple(func[i] for i in perm)
            if gunc in remain:
                orbit.append(gunc)
                remain.remove(gunc)
        #print(orbit)
        orbit.sort()
        orbit = tuple(orbit)
        orbits.append(orbit)

    print("%d! orbits:"%n, len(orbits))
    return orbits, action


def get_action(G, action, func):
    from bruhat.gset import Group, Perm
    H = []
    for g in G:
        perm = action[g]
        gunc = tuple(func[i] for i in perm)
        if gunc == func:
            H.append(g)
    H = Group(H)
    desc = H.structure_description()
    #print(G.rank, H.rank)
    if desc == "D12":
        desc = "C2xS3"
    if desc == "D8":
        desc = "(C2xC2):C2"
    desc = desc.replace("D8", "((C2xC2):C2)")
    return desc, H


def structures(n, funcs):
    from bruhat.gset import Group, Perm

    G = Group.symmetric(n)
    N = factorial(n)
    orbits, action = get_orbits(n, funcs)

    #found = []
    orbits.sort(key=len)
    for idx,orbit in enumerate(orbits):
        K = len(orbit)
        assert N%K == 0
        func = orbit[0]
        desc, H = get_action(G, action, func)
        key = tuple(func.count(i) for i in range(n-1))
        print("\tidx=%d"%idx, "\t%d funcs, |H|=%d  %s\t%s"%(K, N//K, desc, key))
        #if N//K == 48:
        #    found.append(H)
    #for H in found:
    #    print(H.gapstr())

            

def render_func(n, f, r=0.08):
    from huygens.namespace import Canvas, path, grey, black, white
    P = list(numpy.ndindex((2,)*n))
    dx = 4/n
    dy = 1.4*dx
    cvs = Canvas()
    cols = [0]*(n+1)
    coords = {}
    x0 = y0 = x1 = y1 = 0
    for idx in P:
        row = sum(idx)
        col = cols[row] - 0.5*binomial(n,row)
        x = dx*col
        y = dy*row
        coords[idx] = (x,y)
        x0 = min(x, x0)
        x1 = max(x, x1)
        y0 = min(y, y0)
        y1 = max(y, y1)
        cols[row] += 1
    # fix bounding box
    margin = 6*r
    cvs.stroke(path.rect(x0-margin, y0-margin, x1-x0+2*margin, y1-y0+2*margin), [white])
    for idx in P:
      for jdx in P:
        kdx = tuple(i-j for (i,j) in zip(idx,jdx))
        if min(kdx)>=0 and sum(kdx)==1:
            x0, y0 = coords[idx]
            x1, y1 = coords[jdx]
            cvs.stroke(path.line(x0,y0,x1,y1), [grey])
    for i,idx in enumerate(P):
        x, y = coords[idx]
        for j in range(f[i]):
            cvs.stroke(path.circle(x,y,(j+1)*r), [black])
    cvs.coords = coords
    return cvs



def plot_funcs(n, funcs):
    "render entanglement functions"
    from huygens.namespace import Canvas, path, grey, black

    mod = 99999
    if n==4:
        mod = 6 # 18
    elif n==5:
        mod = 3
    elif n==6:
        mod = 6

    cvs = Canvas()

    funcs = list(funcs)
    funcs.sort(key = sum)
    x = 0
    y = 0
    for i,f in enumerate(funcs):
        print(f)
        fg = render_func(n, f)
        cvs.insert(x, y, fg)
        bb = fg.get_bound_box()
        x += 1.1*bb.width
        if (i+1)%mod==0:
            x = 0
            y -= 1.1*bb.height

    print("entanglement_%d.pdf"%n)
    cvs.writePDFfile("entanglement_%d.pdf"%n)


def render_funcs(n, funcs):
    "render entanglement functions "
    orbits, action = get_orbits(n, funcs)
    orbits.sort(key = lambda orbit:sum(orbit[0]))

    from bruhat.gset import Group
    G = Group.symmetric(n)

    from huygens.namespace import Canvas, path, grey, black
    from huygens import config
    config(text="pdflatex")

    mod = argv.get("mod", 4)

    cvs = Canvas()

    x = 0
    y = 0
    for i,orbit in enumerate(orbits):
        func = orbit[0]
        desc, H = get_action(G, action, func)
        print(func, desc)
        fg = render_func(n, func)
        cvs.insert(x, y, fg)
        cvs.text(x, y, desc)
        bb = fg.get_bound_box()
        x += 1.1*bb.width
        if (i+1)%mod==0:
            x = 0
            y -= 1.1*bb.height

    print("render_%d.pdf"%n)
    cvs.writePDFfile("render_%d.pdf"%n)


def show_orbits():
    n = argv.get("n", 4)
    orbits = lc_graph_states(n)

    print()
    smap = SMap()
    i = 0
    for orbit in orbits:
        for M in orbit:
            A = get_graph(M) # SLOW
            if A is not None:
                break
        col = i*(n+3)
        smap[0,col] = str(A)
        smap[n,col] = str(len(orbit))

        i += 1
        if i%8==0:
            print(smap)
            print()
            smap = SMap()
            i = 0
    if i:
        print(smap)
        print()

    counts = ([len(orbit) for orbit in orbits])
    print("found %d orbits" % (len(orbits)))
    print(counts, "==", sum(counts))


    from huygens import config
    config(text="pdflatex")
    from huygens.namespace import Canvas, path, grey, black, white, st_west, st_east

    R = 0.5
    dx = 3*R
    dy = 3.2*R

    coords = {}
    for i in range(n):
        theta = 2*pi*(i+0.5)/n
        coords[i] = (R*sin(theta), R*cos(theta))
    def make_cvs(A):
        cvs = Canvas()
        cvs.fill(path.circle(0,0,1.2*R),[grey])
        for i in range(n):
            cvs.fill(path.circle(*coords[i],0.1*R))
        for i in range(n):
          for j in range(i+1,n):
            if A[i,j]:
                cvs.stroke(path.line(*coords[i],*coords[j]),[white])
        return cvs

    cvs = Canvas()
    y = 0
    for orbit in orbits:
        x = 0
        cvs.text(x-2*R,y,"%s"%(len(orbit),), st_east)
        for M in orbit:
            A = get_graph(M)
            if A is None:
                continue
            fg = make_cvs(A)
            cvs.insert(x, y, fg)
            x += 2.5*R
        y -= 3.0*R
    cvs.writePDFfile("graph_states_%d"%n)
        


def show_stats(n, fs):
    print("show_stats", n)

    keys = []
    for f in fs:
        key = tuple(f.count(i) for i in range(n-1))
        keys.append(key)
    keys.sort()
    uniq = list(set(keys))
    uniq.sort(key = lambda key:keys.count(key))
    for key in uniq:
        print(keys.count(key), key)


def show_poset(n, fs):
    print("show_poset", n)

    fs = list(fs)
    fs.sort()
    print(fs[0])

    pairs = set()
    up = {f:set() for f in fs}
    dn = {f:set() for f in fs}
    for f in fs:
      for g in fs:
        if f is g:
            continue
        for (i,j) in zip(f,g):
            if i>j:
                break
        else:
            up[f].add(g)
            dn[g].add(f)
            pairs.add((f,g))

#    pairs = list(pairs)
#    shuffle(pairs)
#    f,g = pairs[0]
#    print(f)
#    print("<")
#    print(g)
#    return
    print("pairs:", len(pairs))

    for (f,g) in list(pairs):
        for h in up[g]:
            if (f,h) in pairs:
                pairs.remove((f,h))
    print("pairs:", len(pairs))

    names = {}
    for i,f in enumerate(fs):
        name = ''.join(str(k) for k in f[:len(f)//2])
        names[f] = "v"+name

    dot = open("poset_%d.dot"%n, "w")
    print("digraph {", file=dot)
    for (f,g) in pairs:
        print("  %s -> %s;"%(names[f], names[g]), file=dot)
    print("}", file=dot)
    dot.close()


def show_structure(n, funcs):

    from bruhat.gset import Group, Perm

    G = Group.symmetric(n)
    N = factorial(n)
    orbits, action = get_orbits(n, funcs)

    print("%d! orbits:"%n, len(orbits))

    def less_than(orb, prb):
        if orb is prb:
            return False
        # some f in orb is less_than some g in prb?
        for f in orb:
          for g in prb:
            for (i,j) in zip(f,g):
                if i>j:
                    break
            else:
                return True # yes!
        return False # no

    pairs = set()
    up = {f:set() for f in orbits}
    dn = {f:set() for f in orbits}
    for f in orbits:
      for g in orbits:
        if f is g:
            continue
        if less_than(f, g):
            up[f].add(g)
            dn[g].add(f)
            pairs.add((f,g))

    print("pairs:", len(pairs))

    for (f,g) in list(pairs):
        for h in up[g]:
            if (f,h) in pairs:
                pairs.remove((f,h))
    print("pairs:", len(pairs))

    names = {}
    found = set()
    for i,orbit in enumerate(orbits):
        f = orbit[0]
        counts = [0]*n
        for i in range(n):
            counts[i] = f.count(i)
        while counts[-1]==0:
            counts.pop()
        name = ','.join(str(k) for k in counts)
        print(name)
        if name in found:
            name = name+"*"
        assert name not in found
        found.add(name)
        names[f] = name

    print("structure_%d.dot"%n)
    dot = open("structure_%d.dot"%n, "w")
    print("digraph {", file=dot)
    for (o,p) in pairs:
        print('  "(%s)" -> "(%s)";'%(names[o[0]], names[p[0]]), file=dot)
    print("}", file=dot)
    dot.close()


def test_selfdual():
    n = argv.get("n", 5)

    space = SymplecticSpace(n)
    gens = [space.S(), space.H()]
    G = mulclose(gens)
    assert len(G) == 6

    #graphs = all_graph_states(n)
    #codes = list(construct.all_codes(n,0,0))
    #print(len(codes))

    count = 0
    found = []
    #for k in range(0,n+1):
    k = 0
    for code in construct.all_codes(n,k,0):
        H = code.H
        H = H.normal_form()
        found.append(H)
    print(len(found))

    #G = Group.symmetric(n)

    gens = [space.SWAP(i,i+1).t for i in range(n-1)]
    for i in range(n):
        gens.append(space.S(i))
        gens.append(space.H(i))
    remain = set(found)
    found = set()
    orbits = []
    while remain:
        H = remain.pop()
        orbit = [H]
        found.add(H)
        bdy = list(orbit)
        while bdy:
            _bdy = []
            for H in bdy:
              for g in gens:
                J = (H*g).normal_form()
                if J in found:
                    continue
                _bdy.append(J)
                found.add(J)
                remain.remove(J)
                orbit.append(J)
            bdy = _bdy
        orbits.append(orbit)
    print([len(o) for o in orbits], len(orbits))


if __name__ == "__main__":
    from time import time
    start_time = time()
    fn = argv.next() or "main"

    if argv.profile:
        import cProfile as profile
        profile.run("%s()"%fn)
    else:
        fn = eval(fn)
        fn()

    print("finished in %.3f seconds.\n"%(time() - start_time))


