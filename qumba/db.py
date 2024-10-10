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
    ichar = "I"
    sep = " "
    data = {
        #"name" : str(code).replace(" ", ""), # nah..
        "name" : str(code),
        "H" : strop(code.H, ichar, sep),
        "T" : strop(code.T, ichar, sep),
        "L" : strop(code.L, ichar, sep),
        #"J" : strop(code.J, ichar, sep),
        #"A" : strop(code.A, ichar, sep),
        "n" : code.n,
        "k" : code.k,
    }
    res = codes.find_one(data)
    if not force and res:
        print("qumba.db.add: %s already in db" % str(code))
        return
    data.update({
        "d" : code.d,
        "d_lower_bound" : code.d_lower_bound,
        "d_upper_bound" : code.d_upper_bound,
        "desc" : code.desc,
        "created" : int(time()), # is this UTC? I don't know. Do we care?
    })
    for (k,v) in code.attrs.items():
        assert k not in data
        data[k] = v
    codes.insert_one(data)
    print("qumba.db.add: %s added." % str(code))


def delete(code):
    ichar = "I"
    sep = " "
    data = {
        #"name" : str(code).replace(" ", ""), # nah..
        "name" : str(code),
        "H" : strop(code.H, ichar, sep),
        "T" : strop(code.T, ichar, sep),
        "L" : strop(code.L, ichar, sep),
        "n" : code.n,
        "k" : code.k,
    }
    print("qumba.db.delete:", code)
    res = codes.delete_one(data)
    print(res)


def get(**attrs):
    from qumba.qcode import QCode
    data = {}
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
        print("qumba.db.get:", attrs)
        code = QCode.fromstr(
            data["H"],
            data.get("T"),
            data.get("L"),
            **attrs)
        yield code


def get_codes():
    name = argv.name
    n = argv.n
    k = argv.k
    d = argv.d
    ns = {}
    for attr in "name n k d css gf4 cyclic sd desc".split():
        value = getattr(argv, attr)
        if type(value) is list:
            value = str(value)
        if value is not None:
            ns[attr] = value
    if not ns:
        return
    for code in get(**ns):
        yield code


def query():
    for code in get_codes():
        print(code, "_id=%s"%code._id)

        if argv.delete:
            delete(code)


def test():
    from qumba import construct 

    code = construct.get_513()
    add(code)

    code = construct.get_412()
    add(code)


def load_codetables():
    from qumba.qcode import QCode
    for code in QCode.load_codetables():
        if code.k == 0:
            continue
        if code.d_lower_bound < 3:
            continue
        code.is_css()
        code.is_gf4()
        add(code)
        #delete(code)


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


