#!/usr/bin/env python

from time import time, sleep

import pymongo
from bson.objectid import ObjectId

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

    if code.n < 100:
        code.get_tp()
    for (k,v) in code.attrs.items():
        assert k not in data, "%r found in %s"%(k,data)
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
    for attr in "name n k d css dx dz gf4 cyclic sd desc homogeneous d_lower_bound d_upper_bound".split():
        value = getattr(argv, attr)
        if type(value) is list:
            value = str(value)
        if value is not None:
            ns[attr] = value
    for k,v in argv.argmap.items():
        if k == "_id":
            v = ObjectId(v)
        if k not in ns:
            ns[k] = v
        #print(arg)
    if not ns:
        return []
    print("query:", ns)
    codes = list(get(**ns))
    codes.sort(key = lambda code : (code.k, code.d or 0))
    return codes


def check(code):
    print("check:", code)
    print("\tgf4:", code.is_gf4())
    print("\tcss:", code.is_css())
    print("\tselfdual:", code.is_selfdual())




def query():
    for code in get_codes():
        print(code, "_id=%s"%code._id)

        if argv.show:
            print(code.longstr())

        if argv.delete:
            delete(code)

        if argv.check:
            check(code)

        if argv.distance:
            if code.css:
                code = code.to_css()
                dx, dz = code.bz_distance()
                print(dx, dz)

        #if argv.update:
        #    update(code)


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


def normalize():
    from qumba.qcode import QCode
    n = 4
    while n < 100:
        query = {"n" : n}
        cursor = codes.find(query)
        count = 0
        for data in cursor:
            count += 1
            attrs = {}
            for k,v in data.items():
                if k not in list("HTL"):
                    attrs[k] = v
            #print("qumba.db.get:", attrs)
            #print(data)
            code = QCode.fromstr(
                data["H"],
                data.get("T"),
                data.get("L"),
                **attrs)
            print(code, end=" ", flush=True)
            if count % 8 == 0:
                print()
            #print('\t', code.attrs)
            code.is_gf4()
            code.is_css()
            code.is_selfdual()
            data.update(code.attrs)
            #print('\t', code.attrs)
            d_set = {}
            #d_unset = {}
            #if 'sd' in data:
            #    #del data['sd'] # rename as selfdual
            #    d_unset = 
            data["tp"] = code.get_tp()
            #print(data)
            key = {"_id":data["_id"]}
            del data["_id"]
            res = codes.update_one(key, {"$set":data})
            #print(res)
            #return
        print()
        sleep(1)
        n += 1


            


def drop():
    assert 0, "no"
    codes.drop()
    


if __name__ == "__main__":

    start_time = time()

    profile = argv.profile
    name = argv.next() or "query"
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


