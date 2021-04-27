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

import argparse
import pathlib
import random
from collections import Counter
from collections import defaultdict

import numpy as np
import sklearn
import sklearn.feature_extraction
import sklearn.linear_model

from dtdata import DTDataReader, DTDatum, DTDataWriter


class CVFold(object):
    def __init__(self, args, train: 'DataTuple'):
        self.args = args
        self.lr = sklearn.linear_model.LogisticRegression(
            solver='saga',
            max_iter=args.niters,
            verbose=args.verbose,
            n_jobs=args.njobs,
            penalty='elasticnet',
            l1_ratio=0.3
        )
        self.lr.fit(train.vector, train.y)


class DataTuple(object):
    def __init__(self, dts, features, vector, ys):
        self.dts = dts
        self.features = features
        self.vector = vector
        self.y = ys


class CVFoldData(object):
    def __init__(self, idata: 'InputData', fold, train_data, test_data):
        self.fold = fold
        self.idata = idata
        self.indexer = sklearn.feature_extraction.DictVectorizer()
        self.train = self.process_data(train_data, fit_ids=True)
        self.test = self.process_data(test_data)

    def process_data(self, data, fit_ids=False) -> DataTuple:
        indexer = self.indexer
        dts = []
        features = []
        ys = []
        for dx in data:
            for d, rd, y in dx:
                dts.append(d)
                features.append(rd)
                ys.append(y)
        if fit_ids:
            vector = indexer.fit_transform(features)
        else:
            vector = indexer.transform(features)
        return DataTuple(dts, features, vector, np.asarray(ys))

class DTDatumBatch(object):
    def __init__(self, items):
        self.items = items

class DegreeFeatures(object):
    labels = False

    def __init__(self, ctx) -> None:
        self.ctx = ctx


    def extract(self, x):
        if isinstance(x, DTDatum):
            return self.__call__(x)
        elif isinstance(x, DTDatumBatch):
            result = []
            for item in x.items:
                result.append(self.__call__(item))
            return result
        else:
            raise Exception("unknown thing:" + x)

    def __call__(self, datum: DTDatum):
        node_degrees = defaultdict(int)
        nodes = set()

        for edge, weight in zip(datum.nodes, datum.node_weights):
            n0 = edge[0]
            node_degrees[n0] += 1
            nodes.add(n0)

        features = defaultdict(float)

        for edge in datum.nodes: # slightly bad name, but it is an edge
            x = edge[0]
            y = edge[1]
            xp = node_degrees[x]
            yp = node_degrees[y]
            features[f'npe_{xp}_{yp}'] += 1.0 # edge + node degrees
        for x, y in node_degrees.items():
            features[f'np_{y}'] += 1.0 #
        features[f'nnodes_{len(nodes)}'] += 1.0
        features[f'nedges_{len(datum.nodes)}'] += 1.0

        for f, cnt in list(features.items()):
            features[f'{f}_c{int(cnt)}'] += 1
            features[f] = 1.0

        return datum, features, int(datum.value)

processors = {
    'degree': DegreeFeatures
}


class InputData(object):
    def __init__(self, args):
        self.args = args
        self.by_id = {}
        proc_ctor = processors[args.task]
        self.need_labels = proc_ctor.labels
        self.process = proc_ctor(self)
        self.folds = self._load_data()

    def _load_data(self):
        root = pathlib.Path(self.args.root)
        sp = root / f"{self.args.prefix}sp.txt"
        cl = root / f"{self.args.prefix}cl.txt"
        lb = None
        if self.need_labels:
            lb = root / f"{self.args.prefix}lb.txt"
        folds = []
        nfolds = self.args.n_folds
        for _ in range(nfolds):
            folds.append([])

        by_id = self.by_id
        with DTDataReader(sp=sp, cl=cl, lb=lb) as rdr:
            for i, d in enumerate(rdr):
                fold = i % nfolds
                folds[fold].append(self.process(d))
                by_id[d.id] = d
                if (i + 1) % 10000 == 0:
                    print(f"Loaded {i + 1} training examples...")
            print(f"Loaded {i + 1} training examples...")

        return folds


class CVAggregator(object):
    def __init__(self, data: InputData):
        self.data = data
        self.nfolds = data.args.n_folds
        self.hits = defaultdict(int)

    def _dump_features(self, root, fold, fold_data, run_info):
        lr_weights = run_info.lr.coef_
        feature_dict = fold_data.indexer.vocabulary_

        data = []
        for name, idx in feature_dict.items():
            data.append((lr_weights[0, idx], name))
        data.sort()

        full_root = pathlib.Path(root) / '{:02}'.format(fold)
        full_root.mkdir(parents=True, exist_ok=True)
        features_file = full_root / "features.tsv"

        with features_file.open('wt') as fl:
            for weight, name in data:
                fl.write(f"{name}\t{weight}\n")



    def run_cv(self, fold: int):
        train_data = []
        test_data = []
        for i in range(self.nfolds):
            idx = (i + fold) % self.nfolds
            if i < self.data.args.train_folds:
                train_data.append(self.data.folds[idx])
            else:
                test_data.append(self.data.folds[idx])
        fold_data = CVFoldData(self.data, fold, train_data, test_data)
        runner = CVFold(self.data.args, fold_data.train)

        if self.data.args.dump_features is not None:
            self._dump_features(self.data.args.dump_features, fold, fold_data, runner)

        return runner, fold_data

    def count_hits(self, fold: int):
        runner, data = self.run_cv(fold)
        probs = runner.lr.predict_proba(data.test.vector).T
        diffs = probs[0] - probs[1]
        threshold = self.data.args.threshold
        pred_0 = diffs > threshold
        pred_1 = diffs < -threshold

        ys = data.test.y

        is_0 = ys == 0
        is_1 = ys == 1

        hit_0 = pred_0 & is_0
        miss_0 = pred_0 & is_1
        hit_1 = pred_1 & is_1
        miss_1 = pred_1 & is_0

        ctp = sum(hit_1)
        cfp = sum(miss_1)
        cfn = sum(miss_0)

        prec = ctp / max(ctp + cfn, 1)
        recall = ctp / max(ctp + cfp, 1)
        hits = hit_0 | hit_1

        hitdict = self.hits
        for d, cnt in zip(data.test.dts, hits):
            hitdict[d.id] += cnt

        pred_at5 = probs[1] > 0.5
        mat_at5 = pred_at5 == ys
        accuracy = sum(mat_at5) / len(mat_at5)
        return accuracy, prec, recall

    def hits_stats(self):
        args = self.data.args
        max_hit_cnt = args.n_folds - args.train_folds + 1
        counts = [0] * max_hit_cnt
        for cnt in self.hits.values():
            counts[cnt] += 1
        return counts

    def _compute_targets_misses_only(self):
        by_id = self.data.by_id
        targets = [set(), set()]
        selected_cnt = 0
        sel_stats = Counter()
        target_cnt = self.data.args.target_cnt
        half = target_cnt // 2
        rem_classes = [half, target_cnt - half]

        for id, nhits in self.hits.items():
            clz = int(by_id[id].value)
            if rem_classes[clz] <= 0:
                continue
            if nhits > 0:
                continue

            targets[clz].add(id)
            rem_classes[clz] -= 1
            sel_stats[(clz, nhits)] += 1
            selected_cnt += 1
            if selected_cnt >= target_cnt:
                break

        print(sel_stats)
        return targets

    def _compute_targets_weighted(self):
        stats = self.hits_stats()
        target_cnt = self.data.args.target_cnt

        hits_prob = self.data.args.sample_hits
        miss_prob = 1 - hits_prob
        num_hits_bkt = len(stats) - 1

        probs = [target_cnt * miss_prob / stats[0]]

        for i in range(1, len(stats)):
            probs.append(target_cnt * hits_prob / num_hits_bkt / stats[i])

        print(f"sample probs: {probs}")

        half = target_cnt / 2
        rem_classes = [half, target_cnt - half]
        by_id = self.data.by_id
        targets = [set(), set()]
        selected_cnt = 0
        sel_stats = Counter()

        # selection loop
        for id, nhits in self.hits.items():
            clz = int(by_id[id].value)
            if rem_classes[clz] <= 0:
                continue
            prob = random.random()
            if prob <= probs[nhits]:
                targets[clz].add(id)
                rem_classes[clz] -= 1
                sel_stats[(clz, nhits)] += 1
                selected_cnt += 1
            if selected_cnt >= target_cnt:
                print("Selected:", sorted(sel_stats.items()))
                return targets

        # if we did not sample enough elements, sample something additionally
        print("couldn't satisfy sampling condition sampling from other buckets, remaining classes/buckets:\n",
            sorted(sel_stats.items()), rem_classes)
        for id in self.hits:
            clz = int(by_id[id].value)
            if rem_classes[clz] <= 0 or id in targets[clz]:
                continue
            targets[clz].add(id)
            rem_classes[clz] -= 1
            selected_cnt += 1
            if selected_cnt >= target_cnt:
                break

        return targets

    def _compute_write_targets(self):
        hits_prob = self.data.args.sample_hits
        if hits_prob <= 0 or hits_prob >= 1:
            return self._compute_targets_misses_only()
        else:
            return self._compute_targets_weighted()

    def write_filtered_all(self, out):
        targets = self._compute_write_targets()
        tgt_0 = sorted(targets[0])
        tgt_1 = sorted(targets[1])
        nelems = len(tgt_0)
        part_len = nelems // 12
        targets = list(tgt_0[:part_len * 10])
        targets.extend(tgt_1[:part_len * 10])
        random.shuffle(targets)
        self.write_filtered(targets, out, prefix='train_')
        targets = list(tgt_0[part_len * 10:part_len * 11])
        targets.extend(tgt_1[part_len * 10:part_len * 11])
        random.shuffle(targets)
        self.write_filtered(targets, out, prefix='test_')
        targets = list(tgt_0[part_len * 11:])
        targets.extend(tgt_1[part_len * 11:])
        random.shuffle(targets)
        self.write_filtered(targets, out, prefix='valid_')

    def write_filtered(self, targets, out, prefix):
        wr = DTDataWriter(out, prefix)
        byid = self.data.by_id
        for t in targets:
            d: DTDatum = byid[t]
            nhits = self.hits[d.id]
            d.id = d.id + f"-FH{nhits}"
            d.write_to(wr)
        wr.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--root')
    p.add_argument('--prefix', default='train_')
    p.add_argument('--n_folds', type=int, default=10)
    p.add_argument('--train_folds', type=int, default=9)
    p.add_argument('--out')
    p.add_argument('--target_cnt', type=int, default=12000)
    p.add_argument('--verbose', type=int, default=0)
    p.add_argument('--njobs', type=int, default=None)
    p.add_argument('--niters', type=int, default=200)
    p.add_argument('--sample_hits', type=float, default=0.1)
    p.add_argument('--task', type=str, choices=processors.keys(), default='degree')
    p.add_argument('--threshold', type=float, default=0.4)
    p.add_argument('--dump_features')
    return p.parse_args()


def main(args):
    data = InputData(args)
    agg = CVAggregator(data)
    for fold in range(args.n_folds):
        acc, prec, rec = agg.count_hits(fold)
        print(f"fold {fold} apr={acc * 100:.2F}/{prec * 100:.2F}/{rec * 100:.2F}")
    print("hits stats=", agg.hits_stats())
    if args.out:
        agg.write_filtered_all(args.out)


if __name__ == '__main__':
    random.seed(0xdeadbeef)
    main(parse_args())
