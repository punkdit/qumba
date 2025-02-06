#!/usr/bin/env python

import os
from time import time, sleep

import pymongo
from bson.objectid import ObjectId

from qumba.argv import argv

#client = pymongo.MongoClient("mongodb://localhost:27017/")
user,passwd = argv.user, argv.passwd
if user is None:
    user,passwd = open("qumba_creds.txt").read().split()
if os.environ.get("LOCALDB"):
    client = pymongo.MongoClient()
else:
    client = pymongo.MongoClient("mongodb://%s:%s@qecdb.org:27017"%(user,passwd))
db = client["qumba"]
codes = db["codes"]

def add(code, force=False, dummy=False):
    from qumba.qcode import strop
    code = code.to_qcode()
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

    if dummy:
        print(data)
    else:
        codes.insert_one(data)
        print("qumba.db.add: %s added." % str(code))


def delete(code):
    from qumba.qcode import strop
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


def remove():
    while 1:
        _id = argv.next()
        if _id is None:
            break
        v = ObjectId(_id)
        data = {"_id":v}
        print(data)
        res = codes.find_one(data)
        print(res)
        print("remove? ", end="", flush=True)
        arg = input()
        if arg.strip() == "y":
            res = codes.delete_one(data)
            print(res)



def get(**attrs):
    from qumba.qcode import QCode
    data = {}
    data.update(attrs)
    for (k,v) in list(data.items()):
        if k == "_id" and type(v) is str:
            v = ObjectId(v)
            data[k] = v
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
        #print("qumba.db.get:", attrs)
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
    n = argv.n
    if n is None:
        ns = list(range(4, 100))
    else:
        ns = [n]
    for n in ns:
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
            sleep(0.1)
        print()
        sleep(1)
        n += 1


def prune_slow():
    n = argv.get("n", 15)
    k = argv.get("k")
    d = argv.get("d")

    attrs = {"n":n}
    if k is not None:
        attrs["k"] = k
    if d is not None:
        attrs["d"] = d
    print(attrs)
    codes = list(get(**attrs))
    print("codes:", len(codes))
    #return
    remove = prune_z3(codes)
    prune_remove(remove)


def prune_slow(codes):
    from qumba.autos import get_iso
    maxw = argv.maxw

    remove = []
    i = 0
    while i < len(codes):
        code = codes[i]
        j = i+1
        while j < len(codes):
            dode = codes[j]
            if dode.tp != code.tp:
                j += 1
                continue
            f = get_iso(code, dode, maxw)
            if f:
                print("dup[%d,%d]"%( i, j))
                codes.pop(j)
                remove.append( dode )
                break
            else:
                j += 1
                print(".", end="", flush=True)
        i += 1
        print(len(codes), end=" ", flush=True)

    if remove:
        print("remove:", len(remove))
    else:
        print()
    return remove



def prune_z3(codes):
    from qumba.transversal import find_isomorphisms

    remove = []
    i = 0
    while i < len(codes):
        code = codes[i]
        j = i+1
        while j < len(codes):
            dode = codes[j]
            if dode.tp != code.tp:
                j += 1
                continue
            for f in find_isomorphisms(code, dode):
                print("dup[%d,%d]"%( i, j))
                codes.pop(j)
                remove.append( dode )
                break
            else:
                j += 1
                print(".", end="", flush=True)
        i += 1
        print(len(codes), end=" ", flush=True)

    if remove:
        print("remove:", len(remove))
    else:
        print()
    return remove


def prune_remove(remove):
    if not remove:
        return
    print()
    print("to remove:")
    for code in remove:
        print(code, code._id)

    while 1:
        print("delete %d codes (y/n)? "%len(remove), end="", flush=True)
        yesno = input()
        if yesno == "y":
            break
        if yesno == "n":
            return
    #for code in remove:
    #    delete(code)
    query = {"$or":[{"_id":ObjectId(code._id)} for code in remove]}
    res = codes.delete_many(query)
    print(res)
    #found = codes.find(query)
    #for res in found:
    #    print(res)

            

def prune():
    from qumba.transversal import find_isomorphisms
    n = argv.get("n", 15)
    k = argv.get("k")
    d = argv.get("d")

    attrs = {"n":n}
    if k is not None:
        attrs["k"] = k
    if d is not None:
        attrs["d"] = d
    print(attrs)
    codes = list(get(**attrs))
    print("codes:", len(codes))
    #return

    lookup = {}
    for code in codes:
        H = code.H
        w = H.get_wenum()
        print(code, code.tp, w)
        #print(H)
        lookup.setdefault(w, []).append(code)
    keys = list(lookup.keys())
    keys.sort()

    remove = []
    for key in keys:
        codes = lookup[key]
        print(key, len(codes))
        remove += prune_z3(codes)

    prune_remove(remove)


def prune_sparse():
    from qumba.transversal import find_isomorphisms, find_lw
    n = argv.get("n", 23)
    assert n is not None
    k = argv.get("k")
    d = argv.get("d")

    attrs = {"n":n}
    if k is not None:
        attrs["k"] = k
    if d is not None:
        attrs["d"] = d
    print(attrs)
    codes = list(get(**attrs))
    print("codes:", len(codes))
    #return

    codes = [code for code in codes if code.d]
    ds = set(code.d for code in codes)
    ds = list(ds)
    ds.sort()

    remove = []
    for d in ds:
        print("distance:", d)
        lookup = {}
        for code in codes:
            if code.d != d:
                continue
            w = 1
            while w < n:
                #print("w =", w)
                vecs = list(find_lw(code.H, w))
                if vecs:
                    break
                w += 1
            assert vecs
            key = (w, len(vecs))
            print(code, key)
            #print([str(v) for v in vecs])
    
            lookup.setdefault(key, []).append(code)
        keys = list(lookup.keys())
        keys.sort()
    
        for key in keys:
            codes = lookup[key]
            print(key, len(codes))
            #remove += prune_z3(codes)
            remove += prune_slow(codes)
    
    prune_remove(remove)


def prune_cyclic():
    from qumba.cyclic import get_cyclic_perms
    from qumba.action import mulclose
    n = argv.get("n", 15)
    assert n is not None
    k = argv.get("k")
    d = argv.get("d")

    attrs = {"n":n}
    if k is not None:
        attrs["k"] = k
    if d is not None:
        attrs["d"] = d
    print(attrs)
    codes = list(get(**attrs))
    print("codes:", len(codes))
    #return

    codes = [code for code in codes if code.is_cyclic()]
    print("cyclic:", len(codes))
    if not codes:
        return

    space = codes[0].space

    perms = get_cyclic_perms(n)
    for code in codes:
        for p in perms:
            assert (p*code).is_cyclic()

    S, H = space.get_S(), space.get_H()
    G = mulclose([S,H])
    assert len(G) == 6
    gates = [p*g for g in G for p in perms]
    print("G:", len(gates))

    remove = []

    mul = {}

    i = 0
    while i < len(codes):
        code = codes[i]
        src = [p*code for p in gates]
        j = i+1
        while j < len(codes):
            dode = codes[j]
            #if dode.tp != code.tp or dode.name != code.name:
            if (dode.n,dode.k) != (code.n,code.k):
                j += 1
                continue
#            for p in gates:
#                key = (p, code)
#                eode = mul.get(key)
#                if eode is None:
#                    eode = p*code
#                    mul[key] = eode
            if dode.is_css() and not code.is_css():
                print("[css]", end="", flush=True)
                j += 1
                continue
            if dode.is_gf4() and not code.is_gf4():
                print("[gf4]", end="", flush=True)
                j += 1
                continue
            for eode in src:
                if eode.is_equiv(dode):
                    print("dup[%d,%d]"%( i, j), end="", flush=True)
                    codes.pop(j) 
                    remove.append( dode ) # <-------- remove the dode
                    break
            else:
                j += 1
                print(".", end="", flush=True)
        i += 1
        print("(%s)"%len(codes), end=" ", flush=True)

    print()
    if remove:
        print("remove:", len(remove))



    prune_remove(remove)



def drop():
    assert 0, "no"
    codes.drop()
    


if __name__ == "__main__":

    start_time = time()

    profile = argv.profile
    show = argv.show
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


