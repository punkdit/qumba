#!/usr/bin/env python

from time import time

import pymongo

from qumba.qcode import strop
from qumba.argv import argv

#client = pymongo.MongoClient("mongodb://localhost:27017/")
user,passwd = argv.user, argv.passwd
if user is None:
    user,passwd = open("qumba_creds.txt").read().split()
client = pymongo.MongoClient("mongodb://%s:%s@arrowtheory.com:27017"%(user,passwd))
db = client["qumba"]
codes = db["codes"]

def add(code, force=False):
    data = {
        "name" : str(code).replace(" ", ""),
        "H" : strop(code.H),
        "T" : strop(code.T),
        "L" : strop(code.L),
        #"J" : strop(code.J),
        #"A" : strop(code.A),
        "n" : code.n,
        "k" : code.k,
        "d" : code.d,
        "d_lower_bound" : code.d_lower_bound,
        "d_upper_bound" : code.d_upper_bound,
        "desc" : code.desc,
        "timestamp" : int(time()), # is this UTC? I don't know. Do we care?
    }
    res = codes.find_one(data)
    if res is None or force:
        codes.insert_one(data)
    else:
        print("%s already in db" % str(code))

def get(name, **attrs):
    from qumba.qcode import QCode
    data = {"name" : name}
    data.update(attrs)
    cursor = codes.find(data)
    for data in cursor:
#        code = QCode.fromstr(
#            data["H"],
#            data.get("T"),
#            data.get("L"),
#            name = data.get("name"),
#            n = data.get("n"),
#            k = data.get("k"),
#            d = data.get("d"),
#            d_lower_bound = data.get("d_lower_bound"),
#            d_upper_bound = data.get("d_upper_bound"),
#        )
        attrs = {}
        for k,v in data.items():
            if k not in list("HTL"):
                attrs[k] = v
        code = QCode.fromstr(
            data["H"],
            data.get("T"),
            data.get("L"),
            **attrs)
        yield code
        

def query():
    name = argv.next()
    for code in get(name):
        print(code, code._id)


def test():
    from qumba import construct 

#    code = construct.get_513()
    code = construct.get_412()

    add(code)


def dump():
    cursor = codes.find()
    for data in cursor:
        print(data)


def drop():
    assert 0, "no"
    codes.drop()
    


if __name__ == "__main__":

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


