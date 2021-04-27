import csv
from typing import List, Optional
from dataclasses import dataclass
import itertools
from pathlib import Path


class CachedTsvReader(object):
    __slots__ = ['fobj', 'csvrdr', 'current']

    def __init__(self, filename):
        self.fobj = open(filename, 'rt', encoding='utf-8-sig', newline='')
        sniffer = csv.Sniffer()
        start = self.fobj.read(4096)
        dialect = sniffer.sniff(start)
        header = sniffer.has_header(start)
        self.fobj.seek(0)
        self.csvrdr = csv.reader(self.fobj, dialect)
        if header:
            self.next()
        self.next()

    def next(self):
        v = next(self.csvrdr, None)
        self.current = v
        return v

    def __bool__(self):
        return self.current is not None

    def close(self):
        self.fobj.close()


def reader(fn):
    if fn is None:
        return None
    return CachedTsvReader(fn)


@dataclass
class DTLabel:
    __slots__ = ['label_mode', 'node', 'label', 'weight']

    label_mode: int
    node: int
    label: int
    weight: float


def mode_map_identity(mode: int) -> int:
    return mode


@dataclass
class DTDatum:
    id: str
    nodes: List[List[int]]
    node_weights: List[float]
    value: Optional[float]
    labels: Optional[List[DTLabel]]

    def dense_labels(self, mode_map=mode_map_identity, offset=1):
        mode_node = {}
        for l in self.labels:
            mode = mode_map(l.label_mode)
            mode_node.setdefault(mode, {})[l.node] = (l.label, l.weight)

        modes = {}
        for nid, n in enumerate(self.nodes):
            for mode in mode_node:
                nodemap = mode_node.get(mode, None)
                if nodemap is None:
                    continue
                node_info = nodemap.get(n[mode])
                if node_info is None:
                    continue

                label, weight = node_info

                vals = modes.get(mode, None)
                if vals is None:
                    indices = [0] * len(self.node_weights)
                    weights = [0.0] * len(self.node_weights)
                    modes[mode] = (indices, weights)
                else:
                    indices, weights = vals

                indices[nid] = label + offset
                weights[nid] = weight
        return modes

    def write_to(self, wr):
        for n, w in zip(self.nodes, self.node_weights):
            line = map(str, itertools.chain([self.id], n, [w]))
            wr.sp.write("\t".join(line))
            wr.sp.write("\n")
        wr.cl.write(f"{self.id}\t{self.value}\n")
        if self.labels is not None:
            for l in self.labels:
                data = map(str, [self.id, l.label_mode, l.node, l.label, l.weight])
                wr.lb.write("\t".join(data))
                wr.lb.write("\n")


class DTDataReader(object):
    def __init__(self, sp, cl=None, lb=None, weighted_topology=True, header=False):
        self.sp_filename = sp
        self.sp = reader(sp)
        self.cl = reader(cl)
        self.lb = reader(lb)
        self.weighted_topology = weighted_topology
        self.header = header

    def __iter__(self):
        return self

    def __next__(self):
        nv = self.read_one()
        if nv is None:
            self.close()
            raise StopIteration
        return nv

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def read_all(self) -> List[DTDatum]:
        return list(self)

    def close(self):
        if self.sp is None:
            return
        self.sp.close()
        self.sp = None
        if self.cl is not None:
            self.cl.close()
            self.cl = None
        if self.lb is not None:
            self.lb.close()
            self.lb = None

    def read_one(self) -> Optional[DTDatum]:
        if self.sp.current is None:
            return None
        datum_id, nodes, weights = self._read_topology()
        labels = self._read_labels(datum_id)
        value = self._read_value(datum_id)
        return DTDatum(datum_id, nodes, weights, value, labels)

    def _read_topology(self):
        if self.weighted_topology:
            return self._read_topology_weighted()
        else:
            return self._read_topology_non_weighted()

    def _read_topology_weighted(self):
        sp = self.sp
        cur = sp.current
        datum_id = cur[0]
        nodes = []
        weights = []
        while cur is not None and cur[0] == datum_id:
            coords = list(map(int, cur[1:-1]))
            nodes.append(coords)
            weights.append(float(cur[-1]))
            cur = sp.next()
        return datum_id, nodes, weights

    def _read_topology_non_weighted(self):
        sp = self.sp
        cur = sp.current
        datum_id = cur[0]
        nodes = []
        weights = []
        while cur is not None and cur[0] == datum_id:
            coords = list(map(int, cur[1:]))
            nodes.append(coords)
            weights.append(1.0)
            cur = sp.next()
        return datum_id, nodes, weights

    def _read_labels(self, datum_id):
        lb = self.lb
        if lb is None:
            return None
        cur = lb.current
        labels = []
        while cur is not None and cur[0] == datum_id:
            labels.append(DTLabel(
                int(cur[1]),
                int(cur[2]),
                int(cur[3]),
                float(cur[4]),
            ))
            cur = lb.next()
        return labels

    def _read_value(self, datum_id):
        cl = self.cl
        if cl is None:
            return None
        if cl.current is None:
            raise Exception("no cl datum id=" + datum_id + ". possibly cl file is truncated")
        if cl.current[0] == datum_id:
            value = float(cl.current[1])
            cl.next()
            return value
        raise Exception(f"CL file error, current ID = {datum_id}, but CL file had = {cl.current[0]}")


class DTFiles(object):
    def __init__(self, root: Path, prefix: str = '', suffix: str = '', check: bool = True):
        self.sp = root / f"{prefix}sp{suffix}.txt"
        self.cl = root / f"{prefix}cl{suffix}.txt"
        self.lb = root / f"{prefix}lb{suffix}.txt"
        if check:
            if not self.cl.exists():
                self.cl = None
            if not self.lb.exists():
                self.lb = None

    def __repr__(self):
        return f"DTFiles(sp={self.sp}, cl={self.cl}, lb={self.lb})"
