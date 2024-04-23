#!/usr/bin/env python

from qumba.argv import argv

if argv.sage:
    from qumba.clifford_sage import *
else:
    from qumba.clifford_object import *


