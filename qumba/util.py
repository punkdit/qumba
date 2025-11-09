#!/usr/bin/env python3

from functools import lru_cache
cache = lru_cache(maxsize=10)



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
    if type(items) is int:
        items = list(range(items))
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


def binomial(m,n):
    top = factorial(m)
    bot = factorial(n) * factorial(m-n)
    assert top%bot == 0
    return top // bot




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


def all_primes(n, ps=None):
    "list of primes < n"

    items = [0]*n
    p = 2

    while p**2 < n:
        i = 2
        while p*i < n:
            items[p*i] = 1
            i += 1

        p += 1
        while p < n and items[p]:
            p += 1

    ps = [i for i in range(2, n) if items[i]==0]
    return ps


def is_prime(n):
    for p in all_primes(n+1):
        if p==n:
            return True
        elif n%p == 0:
            return False
    assert n==1
    return True


def factorize(n):
    factors = []
    top = int(ceil(n**0.5))
    if n==1:
        return [1]
    for p in all_primes(top+1):
        while n%p == 0:
            factors.append(p)
            n //= p
    if n>1:
        factors.append(n)
    return factors

def divisors(n):
    divs = [1]
    for i in range(2, n):
        if n%i == 0:
            divs.append(i)
    if n>1:
        divs.append(n)
    return divs



