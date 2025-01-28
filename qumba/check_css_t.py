#!/usr/bin/env python

"""
codes from: 
https://github.com/RodrigoSanJose/Cyclic-CSS-T?tab=readme-ov-file
git clone git@github.com:RodrigoSanJose/Cyclic-CSS-T.git
"""

import os
import numpy
from qumba.csscode import CSSCode, distance_z3
from qumba.lin import rank, dot2, shortstr, kernel


def dump_transverse(Hx, Lx, t=3):
    import CSSLO
    SX,LX,SZ,LZ = CSSLO.CSSCode(Hx, Lx)
    #CSSLO.CZLO(SX, LX)
    N = 1<<t
    zList, qList, V, K_M = CSSLO.comm_method(SX, LX, SZ, t, compact=True, debug=False)
    for z,q in zip(zList,qList):
        #print(z, q)
        print(CSSLO.CP2Str(2*q,V,N),"=>",CSSLO.z2Str(z,N))
    print()
    #return zList


def main_cyclic():
    path = "Cyclic-CSS-T/Matrices/"
    names = os.listdir(path)
    
    names = [n for n in names if n.endswith(".npy")]
    names.sort()
    #print(names)
    
    stems = []
    for name in names:
        if "C1" in name:
            stem = name.split("_C1")[0]
            stems.append(stem)
    
    #print(stems)
    stems.sort( key = lambda stem : int(stem.split("_")[0]) )
    
    for stem in stems:
        print(stem)
        n,k,d = stem.split("_")
        d = int(d)
    
        C1 = numpy.load(path+stem+'_C1.npy')
        C2 = numpy.load(path+stem+'_C2.npy')
    
        H = numpy.concatenate((C1,C2))
    
        if H.shape[1] > 500:
            break
    
        Hz = C1
        Hx = kernel(C2)
        assert dot2(Hz, Hx.transpose()).sum() == 0
    
        code = CSSCode(Hx=Hx, Hz=Hz)
        #code.d = d
        #code.bz_distance()
        print(code)
    
        #print(distance_z3(code))
        dump_transverse(code.Hx, code.Lx)
        #print("Hx =")
        #print(shortstr(code.Hx))
        #print("Hz =")
        #print(shortstr(code.Hz))
        #print()
    
        if 0:
            code = code.to_qcode(desc="CSS-T")
            code.d = d
            from qumba import db
            db.add(code)


def main_asym():
    name = "AsymptoticallygoodCSST-Data.txt"
    f = open(name)
    Hx = Hz = rows = None
    for line in f:
        line = line.strip()
        #print(line)
        if "H_x" in line:
            rows = Hx = []
            continue
        elif "H_z" in line:
            rows = Hz = []
            continue
        elif "[" in line and rows is not None:
            for c in " []":
                line = line.replace(c, "")
            line = [int(i) for i in line]
            rows.append(line)
            continue
        elif Hx is None or Hz is None:
            continue
        Hx = numpy.array(Hx)
        Hz = numpy.array(Hz)
        code = CSSCode(Hx=Hx, Hz=Hz)
        print(code)
        print(code.Hx)
        print(code.Lx)
        dump_transverse(code.Hx, code.Lx)
        rows = Hx = Hz = None
        return

def main_code_lins():
    from qumba.qcode import QCode
    # list(lins_db.db.keys()) [(4, 8, 8), (3, 3, 6), (4, 3, 4),]
    code = QCode.fromstr("""
.XXX.XXX...X.XXXXX.X...XXX......
X...X...XXX.X.....X.XXX...XXXXXX
X......X..X..XX....XX........X..
.XX.X.X.X.............XX.......X
...X.X..........X.X......X.XX.X.
.X.X.X..XX....XX...XXX.XX...XXXX
..XX....XXXX.X..X..X..XXX.XXXX..
.ZZZ.ZZZ...Z.ZZZZZ.Z...ZZZ......
Z...Z...ZZZ.Z.....Z.ZZZ...ZZZZZZ
.ZZ.Z.Z.Z.............ZZ.......Z
...Z.Z..........Z.Z......Z.ZZ.Z.
.Z.Z.Z..ZZ....ZZ...ZZZ.ZZ...ZZZZ
..ZZ....ZZZZ.Z..Z..Z..ZZZ.ZZZZ..
.......Z.....ZZ....Z............
.........Z..Z........Z....Z.....
....Z...Z.............Z........Z
...Z.Z..........Z........Z......
...Z.Z........Z....Z............
.......Z...Z.Z...Z..............
..Z...Z.........Z........Z......
Z.........Z.......Z........Z....
........Z...................Z.ZZ
..ZZ.......Z.Z..Z..Z...ZZ.......
.Z......Z..............Z.......Z
Z......Z......Z.....Z...........
............Z..Z.Z...Z..........
..Z.....Z.............ZZ........
Z...Z.ZZ....Z....ZZ......Z......
    """)
    Hs = open("Hs.out").read()
    code = QCode.fromstr(Hs)
    print(code)
    css = code.to_css()
    print(css.bz_distance())
    dump_transverse(css.Hx, css.Lx)

def main_code():
    from qumba.qcode import QCode
    code = QCode.fromstr("""
    """)
    css = code.to_css()
    print(css.bz_distance())
    dump_transverse(css.Hx, css.Lx)

#main_cyclic()
#main_asym()
main_code()


print("done.\n")



