#!/usr/bin/env python3

"""
Group actions.

"""

import sys
import string
from random import randint, shuffle
from functools import reduce, cache

#from qumba.util import factorial, all_subsets, write, uniqtuples, cross
from qumba.equ import Equ, quotient_rep
from qumba.util import factorial
from qumba.argv import argv
from qumba import isomorph
from qumba.smap import SMap, tabulate

long = int


def mulclose_fast(gen, verbose=False, maxsize=None):
    els = set(gen)
    bdy = list(els)
    changed = True 
    while bdy:
        if verbose:
            print(len(els), end=" ", flush=True)
        _bdy = []
        for A in gen:
            for B in bdy:
                C = A*B  
                if C not in els: 
                    els.add(C)
                    _bdy.append(C)
                    if maxsize and len(els)>=maxsize:
                        return els
        bdy = _bdy
    if verbose:
        print()
    return els 


mulclose = mulclose_fast

def mulclose_names(gen, names, verbose=False, maxsize=None):
    bdy = list(set(gen))
    assert len(names) == len(gen)
    names = dict((gen[i], (names[i],)) for i in range(len(gen)))
    changed = True 
    while bdy:
        _bdy = []
        for A in gen:
            for B in bdy:
                C = A*B  
                if C not in names: 
                    #els.add(C)
                    names[C] = names[A] + names[B]
                    _bdy.append(C)
                    if maxsize and len(names)>=maxsize:
                        return list(names)
        bdy = _bdy
    return names 



def mulclose_hom(gen1, gen2, verbose=False, maxsize=None):
    "build a group hom from generators: gen1 -> gen2"
    hom = {}
    assert len(gen1) == len(gen2)
    for i in range(len(gen1)):
        hom[gen1[i]] = gen2[i]
    bdy = list(gen1)
    changed = True 
    while bdy:
        if verbose:
            print(len(hom), end=" ", flush=True)
        _bdy = []
        for A in gen1:
            for B in bdy:
                C1 = A*B  
                if C1 not in hom: 
                    hom[C1] = hom[A] * hom[B]
                    _bdy.append(C1)
                    if maxsize and len(hom)>=maxsize:
                        if verbose:
                            print()
                        return hom
                else:
                    assert hom[A] * hom[B] == hom[C1], "not a hom!"
        bdy = _bdy
    if verbose:
        print()
    return hom 


def mulclose_find(gen, tgt, verbose=False, maxsize=None):
    bdy = list(set(gen))
    names = set(gen)
    if tgt in gen:
        return tgt
    while bdy:
        if verbose:
            print(len(names), end=" ", flush=True)
        _bdy = []
        for A in gen:
            for B in bdy:
                C = A*B
                if C not in names:
                    #els.add(C)
                    names.add(C)
                    _bdy.append(C)
                    if C == tgt:
                        if verbose:
                            print()
                        return C
                    if maxsize and len(names)>=maxsize:
                        if verbose:
                            print()
                        return
        bdy = _bdy
    if verbose:
        print()
    return


def identity(items):
    return dict((i, i) for i in items)



class Perm(object):

    """
    A permutation of a list of items.
    """
    def __init__(self, perm, items, word=''):
        #print "Perm.__init__", perm, items
        #if isinstance(perm, list):
        #    perm = tuple(perm)
        if perm and isinstance(perm, (list, tuple)) and isinstance(perm[0], (int, long)):
            perm = list(items[i] for i in perm)
        #print "\t", perm
        if not isinstance(perm, dict):
            perm = dict((perm[i], items[i]) for i in range(len(perm)))
        #print "\t", perm
        self.perm = perm # map item -> item
        #print "\t", perm
        set_items = set(items)
        self.set_items = set_items
        self.items = list(items)
        assert len(perm) == len(items), (perm, items)
        self.n = len(perm)
        self._str = None
        for key, value in perm.items():
            assert key in set_items, repr(key)
            assert value in set_items, repr(value)
        self.word = word
        self._str_cache = None
        self._hash_cache = None

    @classmethod
    def promote(cls, perm, items=None):
        if isinstance(perm, Perm):
            return perm
        assert type(perm) is list # ?
        if items is None:
            n = len(perm)
            items = list(range(n))
            assert set(items) == set(perm)
        return Perm(perm, items)

    @classmethod
    def identity(cls, items, *args, **kw):
        n = len(items)
        perm = dict((item, item) for item in items)
        return Perm(perm, items, *args, **kw)

    def is_identity(self):
        for k, v in self.perm.items():
            if k != v:
                return False
        return True

    def rename(self, send_items, items):
        perm = {}
        assert len(items) == len(self.items)
        for item in self.items:
            jtem = send_items[item]
            assert jtem in items
            perm[jtem] = send_items[self.perm[item]]
        return Perm(perm, items)

    @classmethod
    def fromcycles(cls, cycles, items, *args, **kw):
        perm = {}
        for cycle in cycles:
            m = len(cycle)
            for i in range(m):
                perm[cycle[i]] = cycle[(i+1)%m]
        return Perm(perm, items, *args, **kw)

    def order(self):
        i = 1
        g = self
        while not g.is_identity():
            g = self*g
            i += 1
            #assert i <= len(self.items)+1 # how big can this get ??
        return i

    def restrict(self, items, *args, **kw):
        perm = dict((i, self.perm[i]) for i in items)
        return Perm(perm, items, *args, **kw)

    def fixes(self, items):
        items = set(items)
        for item in items:
            item = self(item)
            if item not in items:
                return False
        return True

    def intersection(self, other):
        assert self.items == other.items
        items = []
        for i in self.items:
            if self.perm[i] == other.perm[i]:
                items.append(i)
        return items

    def _X_str__(self):
        #return str(dict((i, self.perm[i]) for i in range(self.n)))
        #return str(dict((i, self.perm[i]) for i in range(self.n)))
        if self._str:
            return self._str
        perm = self.perm
        keys = perm.keys()
        keys.sort()
        items = ["%s:%s"%(key, perm[key]) for key in keys]
        s = "{%s}"%(', '.join(items))
        self._str = s
        return s

    def cycles(self):
        remain = set(self.set_items)
        cycles = []
        while remain:
            item = iter(remain).__next__()
            orbit = [item]
            #print(type(self), type(item), "__mul__")
            item1 = self(item)
            while item1 != item:
                orbit.append(item1)
                item1 = self(item1)
                assert len(orbit) <= len(self.items)
            assert orbit
            cycles.append(orbit)
            n = len(remain)
            for item in orbit:
                remain.remove(item)
            assert len(remain) < n
        return cycles

    def sign(self):
        s = 1
        for orbit in self.cycles():
            if len(orbit)%2 == 0:
                s *= -1
        return s

    def cycle_str(self):
        remain = set(self.set_items)
        s = []
#        print "__str__", self.perm, self.items
        while remain:
            item = iter(remain).__next__()
#            print "item:", item
            orbit = [item]
            item1 = self*item
#            print "item1:", item1
            while item1 != item:
                orbit.append(item1)
                item1 = self*item1
                assert len(orbit) <= len(self.items)
            s.append("(%s)"%(' '.join(str(item) for item in orbit)))
            assert orbit
            n = len(remain)
            for item in orbit:
                remain.remove(item)
            assert len(remain) < n
        #return "Perm(%s)"%(''.join(s))
#        print "__str__: end"
        return ''.join(s)
#    __repr__ = __str__

    def get_idxs(self):
        lookup = dict((v, k) for (k, v) in enumerate(self.items))
        idxs = [lookup[self.perm[i]] for i in self.items]
        return idxs

    def slowstr(self): # HOTSPOT
        if self._str_cache:
            return self._str_cache
        perm = self.perm
        items = self.items
        s = []
        for i, item in enumerate(items):
            j = items.index(perm[item]) # <--- oof
            s.append("%d:%d"%(i, j))
        s = "{%s}"%(', '.join(s))
        self._str_cache = s
        return s

    def str(self): # HOTSPOT
        if self._str_cache:
            return self._str_cache
        perm = self.perm
        items = self.items
        lookup = dict((v,k) for (k,v) in enumerate(items))
        s = []
        for i, item in enumerate(items):
            j = lookup[perm[item]]
            s.append("%d:%d"%(i, j))
        s = "{%s}"%(', '.join(s))
        self._str_cache = s
        return s

    def __str__(self):
        # this is the *index* action, not the real permutation action
        return "Perm(%s)"%self.str()
    __repr__ = __str__

    def __hash__(self):
        #return hash(str(self))
        if self._hash_cache is None:
            self._hash_cache = hash(str(self))
        return self._hash_cache

    """
    _hash = None
    def __hash__(self):
        if self._hash is None:
            self._hash = hash(str(self))
        return self._hash
    """

    def leftmul(self, action):
        perms = [self*perm for perm in action]
        action = Group(perms, action.items)
        return action

    def rightmul(self, action):
        perms = [perm*self for perm in action]
        action = Group(perms, action.items)
        return action

    def __mul__(self, other):
        # break this into three cases: 
        if isinstance(other, Group):
            perms = [self*perm for perm in other]
            action = Group(perms, self.items)
            return action # <----- return

        if not isinstance(other, Perm):
            item = self.perm[other]
            return item # <---- return

        assert self.items == other.items
        perm = {}
        #for item in self.items:
            #perm[item] = other.perm[self.perm[item]]
        for item in other.items:
            perm[item] = self.perm[other.perm[item]]
        return Perm(perm, self.items, self.word+other.word)

    # More strict than __mul__:
    def __call__(self, item):
        return self.perm[item]
    __getitem__ = __call__

    def __pow__(self, n):
        assert int(n)==n
        if n==0:
            return Perm.identity(self.items)
        if n<0:
            self = self.__invert__()
            n = -n
        g = self
        for i in range(n-1):
            g = self*g
        return g

    def __invert__(self):
        perm = {}
        for item in self.items:
            perm[self.perm[item]] = item
        return Perm(perm, self.items)
    inverse = __invert__

    def __eq__(self, other):
        assert self.items == other.items
        return self.perm == other.perm

    def __ne__(self, other):
        assert self.items == other.items
        return self.perm != other.perm

    def fixed(self):
        items = []
        for item in self.items:
            if self.perm[item] == item:
                items.append(item)
        return items

    def orbits(self):
        remain = set(self.items)
        orbits = []
        while remain:
            item = iter(remain).__next__()
            orbit = [item]
            item0 = item
            while 1:
                item = self.perm[item]
                if item == item0:
                    break
                orbit.append(item)
                assert len(orbit) <= len(remain)
            remain = remain.difference(orbit)
            orbits.append(orbit)
        return orbits

    def conjugacy_cls(self):
        "this is a partition of len(items)"
        orbits = self.orbits()
        sizes = [len(orbit) for orbit in orbits]
        sizes.sort() # uniq
        return tuple(sizes)

    def preserves_partition(self, part):
        for items in part:
            for item in items:
                jtem = self.perm[item]
                if jtem not in items:
                    return False
        return True


#class Species(object):
#    def __call__(self, group, items):
#        pass

#class PointedSpecies(Species):

class Item(object):
    def __init__(self, item, name=None):
        self.item = item
        if name is None:
            name = str(item)
        self.name = name # a canonical name

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name
    def __repr__(self):
        return "I(%s)"%(self.name)
    def __eq__(self, other):
        return self.name == other.name
    def __ne__(self, other):
        return self.name != other.name


class TupleItem(Item):
    def __init__(self, items):
        Item.__init__(self, items)


class SetItem(Item):
    "a hashable unordered set"
    def __init__(self, items):
        items = list(items)
        #for i in range(len(items)):
        #    item = items[i]
        #    if isinstance(item, Item):
        #        pass
        #    elif isinstance(items[i], (int, long, str)):
        #        items[i] = Item(item)
        #    else:
        #        assert 0, repr(item)
        items.sort(key = lambda item : str(item))
        #items = tuple(items)
        Item.__init__(self, items)

    def __iter__(self):
        return iter(self.item)

    def __len__(self):
        return len(self.item)


def disjoint_union(items, _items):
    items = [(0, item) for item in items]
    _items = [(1, item) for item in _items]
    return items + _items
    

def all_functions(source, target):
    m, n = len(source), len(target)
    source = list(source)
    target = list(target)
    assert n**m < 1e8, "%d too big"%(n**m,)
    if m==0:
        yield {}
    elif n==0:
        return # no functions here
    elif m==1:
        for i in range(n):
            yield {source[0]: target[i]}
    else:
        for func in all_functions(source[1:], target):
            for i in range(n):
                _func = dict(func)
                _func[source[0]] = target[i]
                yield _func

assert len(list(all_functions('ab', 'abc'))) == 3**2
assert len(list(all_functions('abc', 'a'))) == 1
assert len(list(all_functions('a', 'abc'))) == 3


def __choose(items, k):
    "choose k elements"
    n = len(items)
    assert 0<=k<=n
    if k==0:
        yield [], items, [] # left, right, chosen
    elif k==n:
        yield items, [], [] # left, right, chosen
    elif k==1:
        for i in range(n):
            yield items[:i], items[i+1:], [items[i]]
    else:
        for left, right, chosen in __choose(items, k-1):
            n = len(right)
            for i in range(n):
                yield left + right[:i], right[i+1:], chosen+[right[i]]

def _choose(items, *ks):
    "yield a tuple "
    if len(ks)==0:
        yield items,
    elif len(ks)==1:
        k = ks[0]
        for left, right, chosen in __choose(items, k):
            yield items, chosen
    else:
        k = ks[0]
        for flag in _choose(items, *ks[:-1]):
            chosen = flag[-1]
            for chosen, _chosen in _choose(chosen, ks[-1]):
                yield flag + (_chosen,)


def choose(items, *ks):
    "choose k elements"
    items = list(items)
    _items = []
    #for left, right, chosen in _choose(items, k):
    #    _items.append((SetItem(left+right), SetItem(chosen)))
    for flag in _choose(items, *ks):
        flag = tuple(SetItem(item) for item in flag)
        _items.append(flag)
    return _items

items4 = list('abcd')
assert len(choose(items4, 0)) == 1
assert len(choose(items4, 1)) == 4
assert len(choose(items4, 2)) == 4*3//2
assert len(choose(items4, 3)) == 4
assert len(choose(items4, 4)) == 1

class Group(object):
    """
    A collection of Perm's.
    """

    def __init__(self, perms, items, check=False):
        perms = list(perms)
        self.perms = perms # ordered  ( see Group .str and .__hash__ )
        self.items = list(items)
        self.set_items = set(items) # unordered
        self.set_perms = set(perms) # unordered
        for perm in perms:
            assert isinstance(perm, Perm), type(perm)
            assert perm.items == self.items, (perm.items, items)
        self._str = None # cache
        self.conjugates = []

    def str(self):
        if not self._str:
            ss = [perm.str() for perm in self.perms]
            ss.sort() # <-- ordered 
            self._str = ''.join(ss)
        return self._str

    def __eq__(self, other): # HOTSPOT 
        if len(self.perms) != len(other.perms):
            return False
        return (self.set_items == other.set_items and self.set_perms == other.set_perms)

    def __ne__(self, other):
        if len(self.perms) != len(other.perms):
            return True
        return (self.set_items != other.set_items or self.set_perms != other.set_perms)

    def __hash__(self):
        return hash(self.str())

    def __contains__(self, g):
        assert g.items == self.items
        return g in self.set_perms

#    def __lt__(self, other):
#        return id(self) < id(other)

    @classmethod
    def generate(cls, gen, *args, **kw):
        items = kw.get("items")
        if items is not None:
            del kw["items"]
        elif gen:
            items = gen[0].items
        else:
            items = []
        if not gen:
            gen = [Perm.identity(items)]
        perms = list(mulclose(gen, *args))
        G = cls(perms, items, **kw)
        G.gen = gen
        G.gens = gen # deprecate this?
        return G

    def is_cyclic(self):
        n = len(self)
        for g in self:
            if g.order() == n:
                return True
        return False

    def cgy_cls(self):
        "conjugacy classes of elements of G"
        found = set() # map Perm to it's conjugacy class
        itemss = []
        for g in self:
            if g in found:
                continue
            items = set([g])
            itemss.append(items)
            found.add(g)
            for h in self:
                k = h*g*(~h)
                items.add(k)
                found.add(k)
        itemss.sort(key = lambda items : iter(items).__next__().order())
        return itemss

    @property
    def identity(self):
        p = Perm.identity(self.items)
        return p

    @classmethod
    def trivial(cls, items_or_n=1, check=False):
        "the trivial action on items fixes every item"
        if type(items_or_n) in (int, long):
            items = range(items_or_n)
        else:
            items = list(items_or_n)
        perm = Perm.identity(items)
        G = Group([perm], items, check=check)
        return G

    @classmethod
    def symmetric(cls, items_or_n, check=False):
        if type(items_or_n) in (int, long):
            items = range(items_or_n)
        else:
            items = list(items_or_n)
        perms = []
        n = len(items)
        for i in range(n-1):
            perm = dict((item, item) for item in items)
            perm[items[i]] = items[i+1]
            perm[items[i+1]] = items[i]
            perms.append(perm)
        if not perms:
            perms.append({items[0]:items[0]}) # trivial perm
        perms = [Perm(perm, items) for perm in perms]
        G = Group.generate(perms, check=check)
        assert len(G) == factorial(n)
        return G

    @classmethod
    def alternating(cls, items_or_n, check=False):
        if type(items_or_n) in (int, long):
            items = range(items_or_n)
        else:
            items = list(items_or_n)
        n = len(items)
        gen = []
        for i in range(n-2):
            perm = dict((j, j) for j in items)
            perm[items[i]] = items[i+1]
            perm[items[i+1]] = items[i+2]
            perm[items[i+2]] = items[i]
            perm = Perm(perm, items)
            gen.append(perm)
        perms = mulclose(gen)
        G = Group(perms, items)
        assert len(G)==factorial(n)//2
        return G

    @classmethod
    def cyclic(cls, items_or_n, check=False):
        if type(items_or_n) in (int, long):
            items = range(items_or_n)
        else:
            items = list(items_or_n)
        perms = []
        n = len(items)
        perms = [dict((items[i], items[(i+k)%n]) for i in range(n))
            for k in range(n)]
        assert len(perms) == n
        perms = [Perm(perm, items) for perm in perms]
        G = Group(perms, items, check=check)
        return G

    @classmethod
    def dihedral(cls, items_or_n, check=False):
        if type(items_or_n) in (int, long):
            items = range(items_or_n)
        else:
            items = list(items_or_n)
        perms = []
        n = len(items)
        perms = [
            dict((items[i], items[(i+1)%n]) for i in range(n)),
            dict((items[i], items[(-i)%n]) for i in range(n))]
        perms = [Perm(perm, items) for perm in perms]
        G = Group.generate(perms, check=check)
        assert len(G) == 2*n
        return G

    @classmethod
    def coxeter_bc(cls, n, check=False):
        items = [(i,0) for i in range(n)]
        items += [(i,1) for i in range(n)]
        perms = []
        for i in range(n-1):
            jtems = list(items)
            jtems[i:i+2] = jtems[i+1], jtems[i]
            jtems[n+i:n+i+2] = jtems[n+i+1], jtems[n+i]
            perms.append(Perm(dict(zip(items, jtems)), items))
        jtems = list(items)
        jtems[n-1], jtems[2*n-1] = jtems[2*n-1], jtems[n-1]
        perms.append(Perm(dict(zip(items, jtems)), items))
        G = Group.generate(perms, check=check)
        return G

    def __repr__(self):
        return "%s(%s, %s)"%(self.__class__.__name__, self.perms, self.items)

    def __str__(self):
        return "%s(%d, %d)"%(self.__class__.__name__, len(self.perms), len(self.items))

    def __len__(self):
        return len(self.perms)

    def __getitem__(self, idx):
        return self.perms[idx]
    
    def __mul__(self, other):
        if isinstance(other, Group):
            assert self.items == other.items
            perms = set(g*h for g in self for h in other)
            return Group(perms, self.items)
        elif isinstance(other, Perm):
            assert self.items == other.items
            perms = set(g*other for g in self)
            return Group(perms, self.items)
        raise TypeError

    def __add__(self, other):
        items = disjoint_union(self.items, other.items)
        perms = []
        for perm in self:
            _perm = {}
            for item in self.items:
                _perm[0, item] = 0, perm[item]
            for item in other.items:
                _perm[1, item] = 1, item # identity
            _perm = Perm(_perm, items)
            perms.append(_perm)
        for perm in other:
            _perm = {}
            for item in self.items:
                _perm[0, item] = 0, item # identity
            for item in other.items:
                _perm[1, item] = 1, perm[item]
            _perm = Perm(_perm, items)
            perms.append(_perm)
        perms = list(mulclose(perms))
        return Group(perms, items)

    def direct_product(self, other):
        items = [(i, j) for i in self.items for j in other.items]
        perms = []
        for g in self:
          for h in other:
            perm = {}
            for i, j in items:
                perm[i, j] = g[i], h[j]
            perm = Perm(perm, items)
            perms.append(perm)
        group = Group(perms, items)
        return group

    def fixed(self):
        fixed = {}
        for el in self.perms:
            fixed[el] = el.fixed()
        return fixed

    def stabilizer(self, *items):
        "subgroup that fixes every point in items"
        perms = []
        for g in self.perms:
            if len([item for item in items if g[item]==item])==len(items):
                perms.append(g)
        return Group(perms, self.items)

    def invariant(self, *items):
        "subgroup that maps set of items to itself"
        perms = []
        items = set(items)
        for g in self.perms:
            items1 = set(g[item] for item in items)
            if items1.issubset(items):
                perms.append(g)
        return Group(perms, self.items)

    def preserve_partition(self, part):
        H = [g for g in self if g.preserves_partition(part)]
        return Group(H, self.items)

    def orbit(self, item):
        items = set(g(item) for g in self.perms)
        return items

    def orbits(self):
        #print "orbits"
        #print self.perms
        #print self.items
        remain = set(self.set_items)
        orbits = []
        while remain:
            #print "remain:", remain
            item = iter(remain).__next__()
            orbit = set(g(item) for g in self.perms)
            #print "orbit:", orbit
            for item in orbit:
                remain.remove(item)
            orbits.append(orbit)
        return orbits

    def restrict(self, items):
        G = Group([perm.restrict(items) for perm in self.perms], items)
        return G

    def components(self):
        orbits = self.orbits()
        groups = [self.restrict(orbit) for orbit in orbits]
        return groups

    def shape_item(self): # HOTSPOT
        shape_item = []
        for item in self.items:
            shape = []
            for g in self:
                gitem = item
                n = 1
                while 1:
                    gitem = g(gitem)
                    if gitem == item:
                        break
                    n += 1
                shape.append(n)
            shape_item.append(tuple(shape))
        return shape_item

    def check(group):
        #print "check"
        orbits = group.orbits()
        assert sum(len(orbit) for orbit in orbits) == len(group.items)

        # orbit stabilizer theorem
        n = sum(len(group.stabilizer(item)) for item in group.items)
        assert n == len(group) * len(orbits)

        # Cauchy-Frobenius lemma
        assert n == sum(len(g.fixed()) for g in group)

    def subgroups_slow(self): # XXX SLOW
        "All subgroups, _acting on the same items."
        subs = set()
        I = self.identity
        items = self.items
        group = Group([I], self.items)
        subs = set([group, self])
        bdy = set()
        if len(self)>1:
            bdy.add(group)
        set_perms = self.set_perms
        while bdy:
            _bdy = set()
            for group in bdy:
                assert len(group)<len(self)
                # XXX do something clever with cosets...
                for perm in set_perms.difference(group.set_perms):
                    perms = mulclose(group.perms + [perm])
                    _group = Group(perms, items)
                    if _group not in subs:
                        _bdy.add(_group)
                        subs.add(_group)
            bdy = _bdy

        return subs

    def cyclic_subgroups(self, verbose=False):
        # find all cyclic subgroups
        I = self.identity
        trivial = Group([I], self.items)
        cyclic = set()
        for g0 in self:
            if g0==I:
                continue
            perms = [g0]
            g1 = g0
            while 1:
                g1 = g0*g1
                if g1==g0:
                    break
                perms.append(g1)
            group = Group(perms, self.items)
            assert len(group)>1
            cyclic.add(group)
        return cyclic

    @cache
    def subgroups(self, verbose=False):
        I = self.identity
        items = self.items
        trivial = Group([I], items)
        cyclic = self.cyclic_subgroups()
        #if verbose:
        #    print "Group.subgroups: cyclic:", len(cyclic)
        n = len(self) # order
        subs = set(cyclic)
        subs.add(trivial)
        subs.add(self)
        bdy = set(G for G in cyclic if len(G)<n)
        while bdy:
            _bdy = set()
            # consider each group in bdy
            for G in bdy:
                # enlarge by a cyclic subgroup
                for H in cyclic:
                    perms = set(G.perms+H.perms)
                    k = len(perms)
                    if k==n or k==len(G) or k==len(H):
                        continue
                    #perms = mulclose(perms)
                    perms = mulclose_fast(perms)
                    K = Group(perms, items)
                    if K not in subs:
                        _bdy.add(K)
                        subs.add(K)
                        if verbose:
                            write('.')
                    #else:
                    #    write('/')
            bdy = _bdy
            #if verbose:
            #    print "subs:", len(subs)
            #    print "bdy:", len(bdy)
        return subs

    def left_cosets(self, H=None):
        cosets = set()
        if H is not None:
            Hs = [H]
        else:
            Hs = self.subgroups()
        lookup = dict((g, g) for g in self) # remember canonical word 
        for action in Hs:
            for g in self:
                coset = Coset([lookup[g*h] for h in action], self.items)
                cosets.add(coset)
        return list(cosets)

    def right_cosets(self, H=None):
        cosets = set()
        if H is not None:
            Hs = [H]
        else:
            Hs = self.subgroups()
        lookup = dict((g, g) for g in self) # remember canonical word 
        for action in Hs:
            for g in self:
                coset = Coset([lookup[h*g] for h in action], self.items)
                cosets.add(coset)
        return list(cosets)

    def left_action(self, items, basepoint=None):
        send_perms = {}
        perms = []
        lookup = dict((item, item) for item in items)
        for g in self:
            perm = {}
            for item in items:
                perm[item] = lookup[g*item]
            perm = Perm(perm, items)
            perms.append(perm)
            send_perms[g] = perm
        f = quotient_rep(perms)
        send_perms = dict((k, f[v]) for (k, v) in send_perms.items())
        perms = list(f.values())
        hom = Action(self, send_perms, items, basepoint)
        return hom

    def action_subgroup(self, H):
        assert self.items == H.items
        assert self.is_subgroup(H)
        cosets = self.left_cosets(H)
        hom = self.left_action(cosets, H)
        return hom

    def tautological_action(self):
        send_perms = {g:g for g in self}
        action = Action(self, send_perms, self.items)
        return action

    def cayley_action(self, H=None):
        "the left Cayley action of a subgroup on self"
        if H is None:
            H = self
        items = self.perms
        send_perms = {}
        for g in H:
            perm = Perm({h : g*h for h in items}, items)
            send_perms[g] = perm
        action = Action(H, send_perms, items)
        return action

    def is_subgroup(self, H):
        assert H.items == self.items
        for g in H.perms:
            if not g in self.perms:
                return False
        return True

    def is_abelian(self):
        pairs = [(g, h) for g in self for h in self]
        shuffle(pairs)
        for g, h in pairs:
            if g*h != h*g:
                return False
        return True

    def regular_rep(self):
        items = range(len(self))
        lookup = dict((v,k) for (k,v) in enumerate(self.perms))
        perms = []
        for perm in self:
            _perm = {}
            for i in items:
                j = lookup[perm*self[i]]
                _perm[i] = j
            perms.append(Perm(_perm, items))
        return Group(perms, items)

    @classmethod
    def product(cls, H, J):
        perms = list(set(H.perms+J.perms))
        return cls.generate(perms)
    
    def conjugacy_subgroups(G, Hs=None):
    
        # Find all conjugacy classes of subgroups
    
        if Hs is None:
            Hs = G.subgroups()
        #print "subgroups:", len(Hs)
        #for H in Hs:
        #  for K in Hs:
        #    print (int(H.is_subgroup(K)) or "."),
        #  print
    
        equs = dict((H1, Equ(H1)) for H1 in Hs)
        for H1 in Hs:
            for g in G:
                if g in H1:
                    continue
                H2 = g * H1 * (~g) # conjugate
                if H2 == H1:
                    continue
                else:
                    #print len(H1), "~", len(H2)
                    if H2 not in equs:
                        equs[H2] = Equ(H2)
                    equs[H1].merge(equs[H2])
    
        # get equivalance classes
        equs = list(set(equ.top for equ in equs.values()))
        equs.sort(key = lambda equ : (-len(equ.items[0]), equ.items[0].str()))
        for equ in equs:
            #print "equ:", [len(H) for H in equ.items]
            for H in equ.items:
                H.conjugates = list(equ.items)
        #print "total:", len(equs)
        Hs = [equ.items[0] for equ in equs] # pick unique (up to conjugation)
        #Hs.sort(key = lambda H : (-len(H), H.str()))
        return Hs



class Coset(Group):
    def intersect(self, other):
        assert self.items == other.items
        perms = self.set_perms.intersection(other.set_perms)
        return Coset(perms, self.items)
    intersection = intersect




class Action(object):
    """
        A Group _acting on a set, possibly with a basepoint.
        For each perm in the source Group G we map to a perm of items.
    """
    def __init__(self, G, send_perms, items, basepoint=None, check=False):
        assert isinstance(G, Group)
        self.G = G
        assert isinstance(send_perms, dict)
        self.send_perms = dict(send_perms)
        self.items = list(items)
        self.basepoint = basepoint
        self.repr = {}
        if basepoint is not None:
            assert basepoint in items
            for g in G:
                item = self.send_perms[g](basepoint)
                self.repr[item] = g
        if check:
            self.check()

    @property
    def src(self):
        assert 0, "use .G"
        return self.G

    # Equality on-the-nose:
    def __eq__(self, other):
        assert isinstance(other, Action)
        return (self.G==other.G and self.send_perms==other.send_perms)

    def __str__(self):
        return "Action(%s, %s)"%(self.G, len(self.items))

    def __ne__(self, other):
        assert isinstance(other, Action)
        return (self.G!=other.G or self.send_perms!=other.send_perms)

    def __hash__(self):
        send_perms = self.send_perms
        send_perms = tuple((perm, send_perms[perm]) for perm in self.G)
        return hash((self.G, send_perms))

    @classmethod
    def identity(cls, G, check=False):
        send_perms = dict((g, g) for g in G)
        return cls(G, send_perms, G.items, check=check)

    def check(self):
        G, items, send_perms = self.G, self.items, self.send_perms

        assert len(send_perms)==len(G.perms)
        for perm in G.perms:
            assert perm in send_perms
            perm = send_perms[perm]
            assert perm.items == items

        # Here we check that we have a homomorphism of groups.
        for g1 in G.perms:
          h1 = send_perms[g1]
          for g2 in G.perms:
            h2 = send_perms[g2]
            assert send_perms[g1*g2] == h1*h2

    def __call__(self, g):
        perm = self.send_perms[g]
        return perm

#    def __getitem__(self, g): # use __call__ ???
#        assert 0, "use __call__"
#        perm = self.send_perms[g]
#        return perm
#    # should __getitem__ index into .items ?? seems more useful/meaningful...

    # i am a set (list) of items
    def __getitem__(self, idx):
        return self.items[idx]

    # with a len'gth
    def __len__(self):
        return len(self.items)

    def __contains__(self, x):
        return x in self.items # ouch it's a list

    def get_repr(self, x):
        return self.repr[x]

    def rename(self, send_items, items):
        #G, items, send_perms = self.G, self.items, self.send_perms
        for item in self.items:
            assert item in send_items
        new_send = {}
        for g in self.G.perms:
            assert g in self.send_perms
            perm = self.send_perms[g]
            perm = perm.rename(send_items, items)
            new_send[g] = perm
        return Action(self.G, new_send, items)

    def coproduct(*Xs):
        self = Xs[0]
        for X in Xs:
            assert X.G is self.G
        items = [(i, x) for (i,X) in enumerate(Xs) for x in X]
        #print(items)
        send_perms = {}
        for g in self.G:
            perm = {}
            for (i,X) in enumerate(Xs):
                for x in X:
                    perm[i,x] = (i, X(g)[x])
            #print(perm)
            perm = Perm(perm, items)
            send_perms[g] = perm
        return Action(self.G, send_perms, items)
    __add__ = coproduct

    def product(self, other): # HOTSPOT
        assert self.G == other.G
        items = []
        for a1 in self.items:
          for a2 in other.items:
            items.append((a1, a2))
        send_perms = {}
        for g in self.G:
            perm = {}
            h1 = self.send_perms[g]
            h2 = other.send_perms[g]
            for a1, a2 in items:
                perm[(a1, a2)] = h1(a1), h2(a2)
            perm = Perm(perm, items)
            send_perms[g] = perm
        return Action(self.G, send_perms, items)
    __mul__ = product

    def __pow__(self, n):
        assert n>0
        X = reduce(mul, [self]*n)
        return X

    def hecke(self, other):
        import numpy
        assert self.G == other.G
        m, n = (len(self.items), len(other.items))
        marked = set((i, j) for i in range(m) for j in range(n))
        assert marked
        while len(marked):
            H = numpy.zeros((m, n), dtype=numpy.float64)
            i, j = iter(marked).__next__()
            marked.remove((i, j))
            H[i, j] = 1
            ai = self.items[i]
            aj = other.items[j]
            for g in self.G:
                bi = self.send_perms[g][ai]
                bj = other.send_perms[g][aj]
                ii = self.items.index(bi)
                jj = other.items.index(bj)
                if H[ii, jj] == 0:
                    H[ii, jj] = 1
                    marked.remove((ii, jj))
            yield H

    def orbits(self, G=None):
        #print "orbits"
        #print self.perms
        #print self.items
        if G is None:
            G = self.G
        send_perms = self.send_perms
        remain = set(self.items)
        orbits = []
        while remain:
            #print "remain:", remain
            item = iter(remain).__next__()
            orbit = set(send_perms[g](item) for g in G.perms)
            #print "orbit:", orbit
            for item in orbit:
                remain.remove(item)
            orbits.append(orbit)
        return orbits

    def get_components(self):
        G = self.G
        orbits = self.orbits()
        actions = []
        for orbit in orbits:
            send_perms = {}
            perms = []
            for perm in G:
                perm1 = self.send_perms[perm].restrict(orbit)
                send_perms[perm] = perm1
                perms.append(perm1)
            actions.append(Action(G, send_perms, orbit))
        return actions
    components = get_components # backward compat

    def get_atoms(self):
        G = self.G
        orbits = self.orbits()
        homs = []
        for orbit in orbits:
            send_perms = {}
            send_items = {item:item for item in orbit} # inclusion
            perms = []
            for perm in G:
                perm1 = self.send_perms[perm].restrict(orbit)
                send_perms[perm] = perm1
                perms.append(perm1)
            src = Action(G, send_perms, orbit)
            hom = Hom(src, self, send_items)
            homs.append(hom)
        return homs

    def _get_homs_atomic(X, Y):
        assert X.G is Y.G
        if not len(X.items):
            yield Hom(X, Y, {})
            return
        G = X.G
        #x = iter(X.items).__next__()
        x = X[0]
        for y in Y:
            send_items = {x:y}
            for g in G:
                gx = X(g)[x]
                gy = Y(g)[y]
                _gy = send_items.get(gx)
                if _gy is None:
                    send_items[gx] = gy
                elif _gy != gy:
                    break # not a Hom
            else:
                #print("Hom", send_items)
                yield Hom(X, Y, send_items)

    def get_homs(X, Y):
        assert X.G is Y.G
        Xs = X.get_components()
        Ys = Y.get_components()
        #print("get_homs")
        #print(Xs, Ys)
        for section in cross([Ys]*len(Xs)):
            homss = []
            for (Xi,Yi) in zip(Xs, section):
                homs = list(Xi._get_homs_atomic(Yi))
                homss.append(homs)
            for homs in cross(homss):
              send_items = {}
              for hom in homs:
                send_items.update(hom.send_items)
              #print(send_items)
              yield Hom(X, Y, send_items)

    def get_graph(self):
        graph = isomorph.Graph()
        G = self.G
        send_perms = self.send_perms
        items = self.items
        n = len(items)
        # fibres are all the same:
        fibres = [graph.add("fibre") for item in items]
        for i, perm in enumerate(G):
            lookup = dict((item, graph.add()) for item in items)
            for j in range(n):
                graph.join(fibres[j], lookup[items[j]])
            # each layer is unique and contains the "action of g"
            layer = graph.add("layer_%d"%i)
            for item in items:
                graph.join(lookup[item], layer)
                gitem = send_perms[perm](item)
                if gitem == item:
                    # don't need an edge here
                    continue
                graph.add_directed(lookup[item], lookup[gitem])
        return graph

    def get_shape(self):
        G = self.G
        send_perms = self.send_perms
        shape = []
        for perm in G:
            perm = send_perms[perm]
            shape.append(perm.conjugacy_cls())
        return shape

    def isomorphisms(self, other):
        """ yield all isomorphisms in the category of G-sets.
            Each isomorphism is a dict:item->item 

            self  maps G -> H1
            other maps G -> H2
            an isomorphism maps H1.items -> H2.items
        """
        assert isinstance(other, Action)
        assert self.G == other.G
        n = len(self.items)
        if n != len(other.items):
            return
        if self.get_shape() != other.get_shape():
            return
        graph0 = self.get_graph()
        graph1 = other.get_graph()
        for fn in isomorph.search(graph0, graph1):
        #for fn in isomorph.search_recursive(graph0, graph1):
            send_items = {}
            for i in range(n):
                send_items[self.items[i]] = other.items[fn[i]]
            yield send_items

    def check_isomorphism(self, other, send_items):
        assert isinstance(other, Action)
        for perm in self.G:
            perm1 = self.send_perms[perm]
            perm2 = other.send_perms[perm]
            for item1 in self.items:
                item2 = send_items[perm1[item1]]
                assert item2 == perm2[send_items[item1]]

    def slow_isomorphic(self, other, check=False):
        "is isomorphic in the category of G-sets"
        assert isinstance(other, Action)
        for send_items in self.isomorphisms(other):
            self.check_isomorphism(other, send_items)
            return True
        return False

    def fixed_points(self, H):
        send_perms = self.send_perms
        fixed = set(self.items)
        for g in H:
            g = send_perms[g]
            fixed = fixed.intersection(g.fixed())
            if not fixed:
                break
        return fixed

    def signature(self, Hs=None):
        if Hs is None:
            Hs = self.G.subgroups()
        return [len(self.fixed_points(H)) for H in Hs]

    def isomorphic(self, other):
        return self.signature() == other.signature()

    def _refute_isomorphism(self, other, Hs):
        "return True if it is impossible to find an isomorphism"
        assert isinstance(other, Action)
        assert self.G == other.G
        n = len(self.items)
        if n != len(other.items):
            return True

        n = len(self.G)
        for perm in self.G:
            perm1 = self.send_perms[perm]
            perm2 = other.send_perms[perm]
            if perm1.conjugacy_cls() != perm2.conjugacy_cls():
                return True

        # Not able to refute
        return False

    def refute_isomorphism(self, other, Hs):
        # see: http://math.stackexchange.com/a/1891096/360303
        ref = self._refute_isomorphism(other, Hs)
        if ref==True:
            assert self.signature(Hs) != other.signature(Hs)
        elif self.signature(Hs) != other.signature(Hs):
            ref = True
        return ref


class Hom(object):
    "Hom'omorphism of Action's"
    def __init__(self, src, tgt, send_items, check=True):
        assert isinstance(src, Action)
        assert isinstance(tgt, Action)
        G = src.G
        assert G == tgt.G
        self.src = src
        self.tgt = tgt
        self.G = G
        self.send_items = dict(send_items)
        if check:
            self.do_check()

    def do_check(self):
        src = self.src
        tgt = self.tgt
        send_items = self.send_items
        for x,y in send_items.items():
            assert x in src
            assert y in tgt
        for x in src:
            assert x in send_items
        for g in self.G:
            for item in src.items:
                left = send_items[src(g)[item]]
                right = tgt(g)[send_items[item]]
                if left != right:
                    print("item =", item)
                    print("g =", g, "src(g) =", src(g), "tgt(g) =", tgt(g))
                    print("send_items =", send_items)
                    print("%s != %s"%(left, right))
                    assert 0, "not a Hom of Action's"

    def __str__(self):
        return "Hom(%s, %s, %s)"%(self.src, self.tgt, self.send_items)
    __repr__ = __str__

    def __eq__(self, other):
        assert self.G is other.G # too strict ?
        assert self.src == other.src
        assert self.tgt == other.tgt
        return self.send_items == other.send_items

    def __ne__(self, other):
        assert self.G is other.G # too strict ?
        assert self.src == other.src
        assert self.tgt == other.tgt
        return self.send_items != other.send_items

    _hash = None
    def __hash__(self):
        if self._hash is not None:
            return self._hash
        pairs = list(self.send_items.items())
        pairs.sort(key = str) # canonical form
        pairs = tuple(pairs)
        self._hash = hash(pairs)
        return self._hash

    def compose(self, other):
        # other o self
        assert isinstance(other, Hom)
        assert self.tgt == other.src
        a = self.send_items
        b = other.send_items
        send_items = [b[i] for i in a] # wut ???
        return Hom(self.src, other.tgt, send_items)

    def __mul__(self, other):
        assert isinstance(other, Hom)
        return other.compose(self)

#    def mul(f, g):
#        assert isinstance(g, Hom)
#        cone = Action.mul(f.src, g.src)
#        src = cone.apex
#        cone = Cone(src, [cone[0].compose(f), cone[1].compose(g)])
#        cone, univ = Action.mul(f.tgt, g.tgt, cone)
#        return univ
#    __mul__ = mul






