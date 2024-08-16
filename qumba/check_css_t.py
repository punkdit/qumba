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
from qumba import solve


def dump_transverse(Hx, Lx, t=3):
    SX,LX,SZ,LZ = CSSLO.CSSCode(Hx, Lx)
    N = 1<<t
    zList, qList, V, K_M = CSSLO.comm_method(SX, LX, SZ, t, compact=True, debug=False)
    for z,q in zip(zList,qList):
        #print(z, q)
        print(CSSLO.CP2Str(2*q,V,N),"=>",CSSLO.z2Str(z,N))
    print()
    return zList


names = os.listdir("Matrices")

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

    H1 = numpy.load("Matrices/"+stem+'_C1.npy')
    H2 = numpy.load("Matrices/"+stem+'_C2.npy')

    K1 = solve.kernel(H1)
    K2 = solve.kernel(H2)

    code = CSSCode(Hx=K2, Hz=K1)
    print(code)
    print(distance_z3(code))
    #dump_transverse(code.Hx, code.Lx)
    print()

    #break






