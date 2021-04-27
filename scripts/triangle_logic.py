#-------------------------------------------------------------------------------
# @author Arseny Tolmachev <t.arseny@fujitsu.com>
#
# COPYRIGHT Fujitsu Limited 2020 and FUJITSU LABORATORIES LTD. 2020
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
# THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#-------------------------------------------------------------------------------

import random

from typing import Set, Dict, Optional, Iterable, List
from collections import Counter
import heapq

import numpy as np
import scipy.spatial.distance

class Edge(object):
    __slots__ = ["start", "end"]

    def __init__(self, a: int, b: int, directed=False):
        a = int(a)
        b = int(b)

        if not directed and a > b:
            self.start = b
            self.end = a
        else:
            self.start = a
            self.end = b

    def __len__(self) -> int:
        return 2

    def __iter__(self) -> Iterable[int]:
        yield self.start
        yield self.end

    def __getitem__(self, item) -> int:
        if item == 0:
            return self.start
        elif item == 1:
            return self.end
        else:
            raise IndexError("Index can be 0 or 1, was", item)

    def __hash__(self):
        return (self.start * (31 << 5)) ^ (self.end * 31)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.start == other.start and self.end == other.end

    def __lt__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        if self.start < other.start:
            return True
        elif self.start == other.start:
            return self.end < other.end
        return False

    def __repr__(self):
        return f'{self.start}-{self.end}'


edge = Edge


class SynthExampleCandidate(object):
    def __init__(self, num: int, conn, dmatrix=None):
        if dmatrix == 'random':
            self.graph = self._generate_rand_graph(num)
        else:
            self.points = np.random.uniform(size=(num, 2))
            m = scipy.spatial.distance.pdist(self.points, 'euclidean')
            self.dist_matrix = scipy.spatial.distance.squareform(m)
            if isinstance(conn, int):
                self.graph = self._compute_knn(conn)
            elif isinstance(conn, float):
                self.graph = self._compute_nn_cutoff(conn)
            else:
                raise Exception('conn should be int or float, was ' + conn, conn)

    def _generate_rand_graph(self, num):
        graph = []
        for i in range(num):
            graph.append(set())
        for i in range(num):
            nedges = random.randint(1, 3)
            for n in range(nedges):
                j = random.randrange(num)
                if i == j:
                    continue
                graph[i].add(j)
                graph[j].add(i)
        return graph

    def _compute_nn_cutoff(self, conn: float) -> List[Set[int]]:
        good = np.where(self.dist_matrix < conn)
        connections = []
        for _ in range(self.dist_matrix.shape[0]):
            connections.append(set())
        for x, y in zip(*good):
            if x == y:
                continue
            connections[x].add(y)
        return connections

    def _compute_knn(self, k: int) -> List[Set[int]]:
        indices: np = np.argsort(self.dist_matrix, axis=0)
        top_k = indices.T[:, :k]
        connections = []
        for i in range(indices.shape[0]):
            myconn = set(top_k[i])
            if i in myconn:
                myconn.remove(i)
            connections.append(myconn)
        for i in range(indices.shape[0]):
            myconn = top_k[i]
            for j in myconn:
                if j != i:
                    connections[j].add(i)
        return connections

    def enumerate_edges(self) -> Iterable[Edge]:
        for i, other in enumerate(self.graph):
            for j in other:
                if i < j:
                    yield Edge(i, j)

    def gen_edge_labels(self, nlabels: int):
        res = []
        for e in self.enumerate_edges():
            label: int = random.randrange(nlabels)
            res.append((e, label))
        return sorted(res)

class ToyExampleBase(object):
    has_labels = False

    def __init__(self, cand: SynthExampleCandidate):
        super().__init__()
        self.cand = cand
        self.vertices = cand.graph

class Triangle(object):
    __slots__ = ["a", "b", "c"]

    def __init__(self, x: int, y: int, z: int):
        x = int(x)
        y = int(y)
        z = int(z)

        if x > y:
            y, x = x, y
        if y > z:
            z, y = y, z
        if x > y:
            y, x = x, y

        self.a = x
        self.b = y
        self.c = z

    def __len__(self) -> int:
        return 3

    def __iter__(self):
        yield self.a
        yield self.b
        yield self.c

    def __getitem__(self, item) -> int:
        if item == 0:
            return self.a
        elif item == 1:
            return self.b
        elif item == 2:
            return self.c
        else:
            raise IndexError("Index can be 0 or 1, was", item)

    def __contains__(self, item) -> bool:
        if isinstance(item, int):
            return self.a == item or self.b == item or self.c == item
        if isinstance(item, Edge):
            return item.start in self and item.end in self
        return False

    def __hash__(self):
        return (self.a * (31 << 5)) ^ (self.b * (31 << 3)) ^ (self.c * 31)

    def __eq__(self, other):
        if not isinstance(other, Triangle):
            return False
        return self.a == other.a and self.b == other.b and self.c == other.c

    def edge(self, i: int):
        if i == 0:
            return Edge(self.a, self.b)
        elif i == 1:
            return Edge(self.b, self.c)
        elif i == 2:
            return Edge(self.a, self.c)

    def __repr__(self):
        return f'<{self.a}, {self.b}, {self.c}>'


class TriangleTaskExample(object):
    has_labels = False

    def __init__(self, cand: SynthExampleCandidate, _):
        self.cand = cand
        self.vertices = cand.graph
        self.triangles = self._find_triangles()

    def _find_triangles(self):
        vx = self.vertices
        result = set()
        for a in range(len(vx)):
            for b in vx[a]:
                for c in vx[b]:
                    if a in vx[c]:
                        result.add(Triangle(a, b, c))
        return result

    def edges(self, directed=False):
        for a in range(len(self.vertices)):
            for b in self.vertices[a]:
                if directed:
                    yield Edge(a, b, directed)
                else:
                    if a < b:
                        yield Edge(a, b)

    def fix_clazz(self, target: int) -> bool:
        trs = self.triangles

        for _ in range(len(trs) * 10):
            ranks = Counter()
            for t in trs:
                ranks[t.edge(0)] += 1
                ranks[t.edge(1)] += 1
                ranks[t.edge(2)] += 1
            n_common = min(5, len(ranks))
            n_tgt = random.randrange(n_common)
            tgt_e: Edge = ranks.most_common(n_common)[n_tgt][0]

            to_delete = set()
            for t in trs:
                if tgt_e in t:
                    to_delete.add(t)
            if len(trs) - len(to_delete) >= target:
                for t in to_delete:
                    trs.remove(t)
                self.vertices[tgt_e.start].remove(tgt_e.end)
                self.vertices[tgt_e.end].remove(tgt_e.start)
            else:
                continue
            if len(trs) <= target:
                break

        return len(trs) == target

    def is_valid(self):
        return sum(len(x) for x in self.vertices) > 4

    def cur_clazz(self) -> Optional[int]:
        return len(self.triangles) if self.triangles is not None else None

    def check_clazz(self, target) -> bool:
        return len(self._find_triangles()) == target

    def write_dt(self, wr, name):
        for e in self.edges():
            es = e.start
            ee = e.end
            columns = [
                name,
                str(es),
                str(ee),
                '1.0'
            ]
            wr.sp.write("\t".join(columns))
            wr.sp.write("\n")
            columns = [
                name,
                str(ee),
                str(es),
                '1.0'
            ]
            wr.sp.write("\t".join(columns))
            wr.sp.write("\n")
        wr.cl.write(f"{name}\t{self.cur_clazz()}\n")

    def write_graphviz(self, fl, title=""):
        reasons = set()
        for tr in self.triangles:
            reasons.add(tr.edge(0))
            reasons.add(tr.edge(1))
            reasons.add(tr.edge(2))
        fl.write('graph {\n')
        fl.write(f'  graph [label="{title}"];\n')
        for edge in self.edges():
            es, ee = edge
            color = 'black'
            if edge in reasons:
                color = 'red'
            fl.write(f'  {es} -- {ee} [color="{color}"];\n')
        fl.write('}\n')
