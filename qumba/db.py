#!/usr/bin/env python

import pymongo

from qumba.qcode import strop
from qumba.argv import argv

client = pymongo.MongoClient("mongodb://localhost:27017/")
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
    }
    #codes.insert_one({"H":"ZXXZI IZXXZ ZIZXX XZIZX"})
    #codes.delete_one({"_id":res.inserted_id})
    #codes.insert_one({"name":"[[5,1,3]]", "H":"XZZXI IXZZX XIXZZ ZXIXZ"})
    #codes.find({"name":"[[5,1,3]]"})
    #codes.find_({"name":"[[5,1,3]]"})
    #codes.find_({"name":"[[5,1,3]]"})
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

    code = construct.get_513()

    add(code)


def dump():
    cursor = codes.find()
    for data in cursor:
        print(data)


def drop():
    assert 0, "no"
    codes.drop()
    


if __name__ == "__main__":

    from time import time
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


