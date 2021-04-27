import argparse
from pathlib import Path
from nndt.dtdata import DTDataReader, DTFiles, DTDatum
from zipfile import ZipFile, ZIP_DEFLATED


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('input', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--name')
    p.add_argument('--dt-prefix', default='all_')
    p.add_argument('--out-zip', type=Path)
    return p.parse_args()


class TUDatasetWriter(object):
    def __init__(self, dir: Path, prefix: str):
        self.dir = dir
        if prefix is None:
            prefix = dir.name

        self.prefix = prefix
        self.A = open(dir / f"{prefix}_A.txt", "wt", encoding='utf-8')
        self.edge_labels = open(dir / f"{prefix}_edge_labels.txt", "wt", encoding='utf-8')
        self.node_labels = open(dir / f"{prefix}_node_labels.txt", "wt", encoding='utf-8')
        self.graph_labels = open(dir / f"{prefix}_graph_labels.txt", "wt", encoding='utf-8')
        self.graph_indicator = open(dir / f"{prefix}_graph_indicator.txt", "wt", encoding='utf-8')
        self.start_node = 1
        self.num_grahs = 1

    def write(self, d: DTDatum):
        offset = self.start_node
        num_nodes = 0
        for x in d.nodes:
            n0 = x[0]
            n1 = x[1]
            num_nodes = max(n0 + 1, n1 + 1, num_nodes)
            self.A.write(f"{n0 + offset}, {n1 + offset}\n")
            if len(x) == 3:
                el = x[2]
                self.edge_labels.write(f'{el}\n')

        labels = {}
        if d.labels is not None:
            for l in d.labels:
                if l.label_mode == 0:
                    labels[l.node] = l.label

        graph_id = self.num_grahs

        # must not skip nodes
        for n in range(num_nodes):
            self.graph_indicator.write(f"{graph_id}\n")
            l = labels.get(n, 0)
            self.node_labels.write(f"{l}\n")

        self.graph_labels.write(f"{int(d.value)}\n")
        self.start_node += num_nodes
        self.num_grahs += 1

    def close(self):
        for x in [self.A, self.edge_labels, self.node_labels, self.graph_labels, self.graph_indicator]:
            x.close()

        if Path(self.node_labels.name).stat().st_size == 0:
            Path(self.node_labels.name).unlink()

        if Path(self.edge_labels.name).stat().st_size == 0:
            Path(self.edge_labels.name).unlink()

    def save_zip(self, loc: Path):
        with ZipFile(loc / f"{self.dir.name}.zip", 'w', ZIP_DEFLATED) as f:
            for x in self.dir.iterdir():
                f.write(x, f"{self.dir.name}/{x.name}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def main(args):
    files = DTFiles(args.input, prefix=args.dt_prefix)
    output: Path = args.output
    output.mkdir(exist_ok=True, parents=True)

    with DTDataReader(files.sp, files.cl, files.lb) as rdr:
        with TUDatasetWriter(output, args.name) as wr:
            for d in rdr:
                wr.write(d)
        if args.out_zip is not None:
            wr.save_zip(args.out_zip)


if __name__ == '__main__':
    main(parse_args())
