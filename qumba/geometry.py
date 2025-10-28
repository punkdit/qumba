#!/usr/bin/env python

from string import ascii_lowercase


from bruhat.action import Perm, Group, Coset, mulclose, close_hom, is_hom
from bruhat.todd_coxeter import Schreier
from bruhat import lins_db

from qumba.lin import zeros2, linear_independent, dot2
from qumba.csscode import CSSCode
from qumba.argv import argv
from qumba.smap import SMap



class Geometry(object):
    "A geometry specified by a Coxeter reflection group"
    def __init__(self, orders, lins_idx=0, build=True):
        ngens = len(orders)+1
        self.gens = {c:(i,) for (i,c) in enumerate(ascii_lowercase[:ngens])}
        orders = tuple(orders)
        assert orders in lins_db.db, (list(lins_db.db.keys()))
        #print("oeqc.Geometry.__init__:", len(lins_db.db[orders]))
        rels = lins_db.db[orders][lins_idx]
        rels = lins_db.parse(rels, **self.gens)
        self.orders = orders
        self.ngens = ngens
        self.rels = rels
        self.dim = len(orders)
        if build:
            self.G = self.build_group()

    def build_graph(self, figure=None, hgens=None):
        gens = [(i,) for i in range(5)]
        ngens = self.ngens
        rels = [gens[i]*2 for i in range(ngens)]
        orders = self.orders
        for i in range(ngens-1):
            order = orders[i]
            if order is not None:
                rels.append( (gens[i]+gens[i+1])*order )
            for j in range(i+2, ngens):
                rels.append( (gens[i]+gens[j])*2 )
        rels = rels + self.rels
        #print(rels)
        graph = Schreier(ngens, rels)
        if figure is not None:
            assert len(figure) == len(gens)
            gens = [gens[i] for i, fig in enumerate(figure) if fig] or [G.identity]
            graph.build(gens)
        elif hgens is not None:
            graph.build(hgens)
        else:
            graph.build()
        return graph


    def build_group(self):
        graph = self.build_graph()
        self.G = graph.get_group()
        return self.G

    def get_cosets(self, figure):
        G = self.G
        gens = G.gens
        assert len(figure) == len(gens)
        gens = [gens[i] for i, fig in enumerate(figure) if fig] or [G.identity]
        #print("gens:", gens)
        H = Group.generate(gens)
        #pairs = G.left_cosets(H)
        cosets = G.left_cosets(H)
        return cosets



def get_adj(left, right):
    A = zeros2((len(left), len(right)))
    for i, l in enumerate(left):
      for j, r in enumerate(right):
        lr = l.intersection(r)
        A[i, j] = len(lr)>0
    return A


def build(shape, index):
    N = len(shape)+1
    geometry = Geometry(shape, index, False)
    total = geometry.build_graph()
    geometry.build_group()
    #total = total.compress()
    words = total.get_words()
    n = len(total)
    #print("idx = %d, |G| = %d"%(index, n))

    dim = geometry.dim
    #print("dim =", dim)

    if dim == 2:
        faces = geometry.get_cosets([0,1,1])
        edges = geometry.get_cosets([1,0,1])
        verts = geometry.get_cosets([1,1,0])
        #print("faces=%d, edges=%d, verts=%d"%(len(faces), len(edges), len(verts)))

    else:
        bodis = geometry.get_cosets([0,1,1,1])
        faces = geometry.get_cosets([1,0,1,1])
        edges = geometry.get_cosets([1,1,0,1])
        verts = geometry.get_cosets([1,1,1,0])
        partial_flags = [
            bodis, faces, edges, verts,
            geometry.get_cosets([0,0,1,1]),
            geometry.get_cosets([0,1,0,1]),
            geometry.get_cosets([0,1,1,0]),
            geometry.get_cosets([1,0,0,1]),
            geometry.get_cosets([1,0,1,0]),
            geometry.get_cosets([1,1,0,0]),
            geometry.get_cosets([1,0,0,0]),
            geometry.get_cosets([0,1,0,0]),
            geometry.get_cosets([0,0,1,0]),
            geometry.get_cosets([0,0,0,1]),
        ]
        print("bodis=%d, faces=%d, edges=%d, verts=%d"%(
            len(bodis), len(faces), len(edges), len(verts)))

    if argv.selfdual:

        if argv.flip:
            H = get_adj(faces, verts) # ?
        else:
            H = get_adj(verts, faces)

        H = linear_independent(H)
        Hx = Hz = H

    else:
    
        #if argv.homology == 1:
        Hz = get_adj(faces, edges)
        Hx = get_adj(verts, edges)
        #print("Hx:")
        #print(shortstr(Hx), Hx.shape)
    
        Hx = linear_independent(Hx)
        Hz = linear_independent(Hz)

    A = dot2(Hx, Hz.transpose())

    #print("chain condition:", A.sum() == 0)

    if A.sum() != 0:
        #print("not commutative\n")
        return

    code = CSSCode(Hx=Hx, Hz=Hz)
    return code






def main():

    start_idx = argv.get("start_idx", 1)
    stop_idx = argv.get("stop_idx", None)
    shape = argv.get("shape", (5,5))
    index = argv.get("index", 1000)
    if shape not in lins_db.db:
        print("lins_db.build_db...", end='', flush=True)
        lins_db.build_db(shape, index)
        print(" done")
    if argv.idx:
        build(shape, argv.idx)
        return
    print("shape =", shape)
    n = len(lins_db.db[shape])
    idx = start_idx

    best = {}
    while idx < n and (stop_idx is None or idx < stop_idx):
        code = build(shape, idx)
        idx += 1

        if code is None or code.k == 0:
            #print(code)
            continue
    
        if code.n > 100:
            break

        print(code, end=' ', flush=True)
        code.bz_distance()
        #print(code)
        if code.dx < 3 or code.dz < 3:
            continue
        if code.k <= 2:
            continue

        #found.append(code)
        if argv.all:
            key = code.n, code.k, code.dx, code.dz
        else:
            key = code.n, code.k
        other = best.get(key)
        if other is None or other.d < code.d:
            best[key] = code

        Hx, Hz = code.Hx, code.Hz
        Hxs = Hx.sum(1)
        Hzs = Hz.sum(1)
        rw = Hxs.max(), Hzs.max()
        #print("Hx weights:", Hxs.min(), "to", Hxs.max())
        #print("Hz weights:", Hzs.min(), "to", Hzs.max())

        print("\t%s"%code)
        if argv.show:
            print(code.longstr())

    print("best:")
    for code in best.values():
        print(code)
    
    #if argv.store_db:
    print()
    print("write to db (n)?", end=" ", flush=True)
    val = input()
    if val=="y":
        from qumba.db import add
        for code in best.values():
            code = code.to_qcode(homogeneous=True, desc="hyperbolic_2d", shape=str(shape))
            add(code, dummy=False)






if __name__ == "__main__":

    from time import time
    from qumba.argv import argv

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

    else:
        main()


    t = time() - start_time
    print("OK! finished in %.3f seconds\n"%t)






