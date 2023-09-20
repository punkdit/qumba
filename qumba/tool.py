#!/usr/bin/env python

from qumba.argv import argv


if argv.silent:
    def write(s):
        pass

else:

    def write(s):
        print(s, end="", flush=True)


