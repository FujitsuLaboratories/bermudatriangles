import pathlib
from typing import Union

from .reader import DTDatum


class DTDataWriter(object):
    def __init__(self, root: Union[str, pathlib.Path], prefix: str = '', lb: bool = False):
        if isinstance(root, pathlib.Path):
            self.root = root
        else:
            self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.sp = (self.root / f'{prefix}sp.txt').open("wt", encoding='utf-8')
        self.cl = (self.root / f'{prefix}cl.txt').open("wt", encoding='utf-8')
        self.lb = None
        if lb:
            self.lb = (self.root / f'{prefix}lb.txt').open("wt", encoding='utf-8')

    def write(self, d: DTDatum):
        d.write_to(self)

    def close(self):
        self.sp.close()
        self.cl.close()
        if self.lb:
            self.lb.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
