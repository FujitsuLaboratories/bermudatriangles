#!/usr/bin/env python3
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
import pathlib
import subprocess
import argparse
import numpy as np

from dtdata.writer import DTDataWriter
from triangle_logic import TriangleTaskExample, SynthExampleCandidate


class GraphWriter(object):
    def __init__(self, root, do_merge=True):
        self.root = pathlib.Path(root)
        self.do_merge = do_merge
        if False:
            self.pdf_root = self.root / 'ps'
            self.format_flag = '-Teps'
            self.gw_ext = 'eps'
        else:
            self.pdf_root = self.root / 'pdf'
            self.format_flag = '-Tpdf'
            self.gw_ext = 'pdf'
        self.dot_root = self.root / 'dot'

        self.dot_root.mkdir(parents=True, exist_ok=True)
        self.pdf_root.mkdir(parents=True, exist_ok=True)

        self.files = []
        self.ready_pdfs = []

    def check_ready(self):
        files = []
        for pdf, proc in self.files:
            if proc.poll():
                self.ready_pdfs.append(pdf)
            else:
                files.append((pdf, proc))
        self.files = files

    def add_graph(self, name, g):
        dotf = self.dot_root / f'{name}.dot'
        with dotf.open('wt', encoding='utf-8') as fl:
            g.write_graphviz(fl, title=name)
        outf = self.pdf_root / f"{name}.{self.gw_ext}"
        p = subprocess.Popen([
            'dot',
            self.format_flag,
            f'-o{outf}',
            str(dotf)
        ])

        self.files.append((outf, p))

    def wait(self):
        pdfs = self.ready_pdfs
        for pdf, proc in self.files:
            proc.wait(10.0)
            pdfs.append(pdf)
        self.files.clear()

    def finish(self):
        self.wait()
        if not self.do_merge:
            return

        all_pdf = self.root / "all.pdf"
        ready_pdfs = sorted(self.ready_pdfs)
        subprocess.call(
            [
                "gs",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                f"-sOUTPUTFILE={all_pdf}",
                "-dBATCH",
            ] + ready_pdfs)


generators = {
    'triangle': TriangleTaskExample
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=generators.keys(), default='triangle')
    p.add_argument('--num-data', type=int, default=30)
    p.add_argument('--graphviz')
    p.add_argument('--graphviz-merge', action='store_true')
    p.add_argument('--min-nodes', type=int, default=6)
    p.add_argument('--max-nodes', type=int, default=20)
    p.add_argument('--max-labels', type=int, default=12)
    p.add_argument('--out')
    p.add_argument('--out-prefix', default='train_')
    p.add_argument('--prefix', default='TR')
    p.add_argument('--dist-matrix')
    p.add_argument('--seed', type=int, default=0xdeadbeef)
    return p.parse_args()


class DataGenerator(object):
    def __init__(self, args):
        self.args = args
        self.task_maker = generators[args.task]
        self.gw = None
        if args.graphviz:
            self.gw = GraphWriter(args.graphviz, args.graphviz_merge)
        self.dtw = DTDataWriter(args.out, args.out_prefix, self.task_maker.has_labels)
        self.limit = self.args.num_data

    def run(self):
        limits = [self.limit / 2, self.limit / 2]
        prefix = self.args.prefix
        count = 0
        min_nodes = self.args.min_nodes
        max_nodes = self.args.max_nodes
        dmat_alg = self.args.dist_matrix

        for i in range(1, self.limit * 10):
            if limits[0] <= 0 and limits[1] <= 0:
                break
            if i % 1000 == 0:
                print(f"Generated {i}/{count} of {self.limit} {prefix} data items, {limits} remaining")
                if self.gw:
                    self.gw.check_ready()
            num_nodes = random.randint(min_nodes, max_nodes)
            nn_mode = random.randint(0, 1)
            if nn_mode == 0:  # k-nn
                nn_param = random.randint(3, 6)
                nn_gen = nn_param
                nn_prefix = f"K{nn_param}"
            else:
                nn_gen = random.randint(3, 6)
                nn_param = nn_gen * 0.05
                nn_prefix = f"T{nn_gen * 5}"
            exc = SynthExampleCandidate(num_nodes, nn_param, dmatrix=dmat_alg)
            num_lbl = min(num_nodes + nn_gen - 3, 12)
            ex = self.task_maker(exc, num_lbl)

            clz = ex.cur_clazz()
            name = f"{prefix}{i:06}-{num_nodes}-{nn_prefix}-{num_lbl}"

            if not ex.is_valid():
                print(f"rejected {name} as invalid")
                continue

            if clz > 1 or limits[clz] <= 0:
                prob = limits[0] / sum(limits)
                rval = random.random()
                if rval < prob:
                    clz = 0
                else:
                    clz = 1
                if limits[clz] <= 0:
                    clz = 1 - clz
                if not ex.fix_clazz(clz):
                    # print(f"failed to fix class of {name} ({clz})")
                    continue
                if not ex.check_clazz(clz):
                    print(f"fix_class != check_class -> BUG! {name} -> {clz}")
                    continue
            gname = f"{name}-{clz}"
            if self.gw:
                self.gw.add_graph(gname, ex)
            ex.write_dt(self.dtw, gname)
            limits[clz] -= 1
            count += 1
        if self.gw:
            self.gw.finish()


if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    gen = DataGenerator(args)
    gen.run()
