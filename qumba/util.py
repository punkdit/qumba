#!/usr/bin/env python3


def all_subsets(items):

    items = list(items)
    n = len(items)
    if n==0:
        yield []
        return

    if n==1:
        yield []
        yield [items[0]]
        return

    for subset in all_subsets(items[:n-1]):
        yield subset
        yield subset + [items[n-1]] # sorted !!

assert len(list(all_subsets(list(range(5))))) == 2**5





# tried caching this, not any faster
def factorial(n):
    r = 1
    for i in range(1, n+1):
        r *= i
    return r

assert factorial(0) == 1
assert factorial(1) == 1
assert factorial(2) == 2
assert factorial(3) == 2*3
assert factorial(4) == 2*3*4


def choose(items, n):
    if n > len(items):
        return
    if n == 0:
        yield ()
        return
    if n == 1:
        for item in items:
            yield (item,)
        return
    for i, item in enumerate(items):
        for rest in choose(items[i+1:], n-1):
            yield (item,)+rest

assert len(list(choose(range(4), 1))) == 4
assert len(list(choose(range(4), 2))) == 6
assert len(list(choose(range(4), 3))) == 4



def allperms(items):
    items = tuple(items)
    if len(items)<=1:
        yield items
        return
    n = len(items)
    for i in range(n):
        for rest in allperms(items[:i] + items[i+1:]):
            yield (items[i],) + rest

assert list(allperms("abc")) == [
    ('a', 'b', 'c'),
    ('a', 'c', 'b'),
    ('b', 'a', 'c'),
    ('b', 'c', 'a'),
    ('c', 'a', 'b'),
    ('c', 'b', 'a')]

all_perms = allperms



def cross(itemss):
    if len(itemss)==0:
        yield ()
    else:
        for head in itemss[0]:
            for tail in cross(itemss[1:]):
                yield (head,)+tail


