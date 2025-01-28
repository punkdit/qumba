#!/usr/bin/env python

from qumba.lin import dot2


class MatchingDecoder(object):
    def __init__(self, code):
        self.code = code
        from pymatching import Matching
        H = code.Hz
        self.matching = Matching(H)

    def decode(self, p, err_op, verbose=False, **kw):
        code = self.code
        matching = self.matching
        syndrome = dot2(code.Hz, err_op)
        prediction = matching.decode(syndrome)
        return prediction



