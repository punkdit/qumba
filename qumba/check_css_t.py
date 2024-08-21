#!/usr/bin/env python

"""
codes from: 
https://github.com/RodrigoSanJose/Cyclic-CSS-T?tab=readme-ov-file
git clone git@github.com:RodrigoSanJose/Cyclic-CSS-T.git
"""

import os
import numpy
import CSSLO
from qumba.csscode import CSSCode, distance_z3
from qumba.solve import rank, dot2, shortstr, kernel


def dump_transverse(Hx, Lx, t=3):
    SX,LX,SZ,LZ = CSSLO.CSSCode(Hx, Lx)
    CSSLO.CZLO(SX, LX)
    #N = 1<<t
    #zList, qList, V, K_M = CSSLO.comm_method(SX, LX, SZ, t, compact=True, debug=False)
    #for z,q in zip(zList,qList):
    #    #print(z, q)
    #    print(CSSLO.CP2Str(2*q,V,N),"=>",CSSLO.z2Str(z,N))
    #print()
    #return zList


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

    C1 = numpy.load(path+stem+'_C1.npy')
    C2 = numpy.load(path+stem+'_C2.npy')

    H = numpy.concatenate((C1,C2))

    Hz = C1
    Hx = kernel(C2)
    assert dot2(Hz, Hx.transpose()).sum() == 0

    #if C1.shape[1]>32:
    #    break

    code = CSSCode(Hx=Hx, Hz=Hz)
    print(code)
    print(distance_z3(code))
    #dump_transverse(code.Hx, code.Lx)
    print("Hx =")
    print(shortstr(code.Hx))
    print("Hz =")
    print(shortstr(code.Hz))
    print()



print("done.\n")



