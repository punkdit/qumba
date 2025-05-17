#!/usr/bin/env python

"""
build network of ZX spiders & find wavefunction
"""

from qumba.umatrix import UMatrix, Solver, Var, And, Or

from qumba.argv import argv


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, idx):
        return self.nodes[idx]

    def red(self):
        return Red(self)

    def green(self):
        return Green(self)

    def solve(self):
        solver = Solver()
        links, nodes = self.links, self.nodes
        N = len(links)
        vs = {link:Var() for link in links}
        for node in nodes:
            nbd = [vs[link] for link in node.nbd]
            term = node.constrain(nbd)
            solver.add(term)

        while 1:
            result = solver.check()
            if str(result) == "unsat":
                break
            model = solver.model()
            found = [vs[link].get_interp(model) for link in links]
            yield found

            for value,link in zip(found,links):
                solver.add( vs[link] != value )

            

class Link:
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def __str__(self):
        return "Link(%s, %s)"%(self.left, self.right)
    __repr__ = __str__
    def __getitem__(self, i):
        return [self.left, self.right][i]


class Node:
    def __init__(self, graph):
        self.nbd = []
        graph.nodes.append(self)
        self.graph = graph

    def __str__(self):
        return "%s(%d)"%(self.__class__.__name__, len(self.nbd),)
    __repr__ = __str__

    def link(self, other=None):
        link = Link(self, other)
        self.nbd.append(link)
        if other is not None:
            #op = Link(other, self)
            other.nbd.append(link)
        self.graph.links.append(link)

#    def get(self, idxs):
#        pass

    def constrain(self, vs):
        pass


class Red(Node):
#    def get(self, idxs):
#        assert idxs
#        assert len(idxs) == len(self.nbd)
#        s = sum(idxs) % 2
#        if s==0:
#            return 1
#        return 0

    def constrain(self, vs):
        return sum(vs)==0
        

class Green(Node):
#    def get(self, idxs):
#        assert idxs
#        assert len(idxs) == len(self.nbd)
#        idx = idxs[0]
#        for jdx in idxs:
#            if jdx != idx:
#                return 0
#        return 1

    def constrain(self, vs):
        left = And(*[(v==0) for v in vs])
        right = And(*[(v==1) for v in vs])
        term = Or(left, right)
        return term
        

def test():

    N = 4
    graph = Graph()
    for i in range(N):
        if i%2:
            graph.red()
        else:
            graph.green()

    for i in range(N):
        graph[i].link(graph[(i+1)%N])

    print(graph.nodes)

    for u in graph.solve():
        print(u)


def make_rect(N):
    graph = Graph()
    lookup = {}
    for i in range(N):
      for j in range(N):
        if (i+j)%2:
            node = graph.red()
        else:
            node = graph.green()
        lookup[i,j] = node

    for i in range(N):
      for j in range(N):
        lookup[i,j].link(lookup[(i+1)%N,j])
        lookup[i,j].link(lookup[i, (j+1)%N])
    return graph


def make_hex(N):
    graph = Graph()
    lookup = {}
    for i in range(N):
      for j in range(N):
        if (i+j)%2:
            node = graph.red()
        else:
            node = graph.green()
        lookup[i,j] = node

    for i in range(N):
      for j in range(N):
        lookup[i,j].link(lookup[i, (j+1)%N])
        if (i+j)%2==0:
            lookup[i,j].link(lookup[(i+1)%N,j])
        #else:
        #    lookup[i,j].link()
    return graph


def main():

    N = 4
    graph = make_hex(N)

    print(graph.nodes)
    for link in graph.links:
        print(link)
    print("links:", len(graph.links))

    for u in graph.solve():
        s = str([int(ui) for ui in u])
        s = s.replace(",", "")
        s = s.replace(" ", "")
        s = s.replace("0", ".")
        print(s)




if __name__ == "__main__":

    from time import time
    start_time = time()

    profile = argv.profile
    name = argv.next() or "main"

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





