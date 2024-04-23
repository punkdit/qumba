#!/usr/bin/env python

"""
Used in clifford.py , modify as needed before importing clifford module.
"""

from qumba.argv import argv

degree = argv.get("degree", 8)
assert degree % 8 == 0



