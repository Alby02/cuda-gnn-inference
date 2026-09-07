"""Version-1 workload interchange; standard library only, no model mathematics."""
from array import array
from dataclasses import dataclass
from pathlib import Path
import shlex
import struct
import sys


@dataclass
class Matrix:
    rows: int
    cols: int
    values: array


def read_matrix(path):
    raw = Path(path).read_bytes()
    rows, cols = struct.unpack_from('<QQ', raw)
    values = array('f')
    values.frombytes(raw[16:])
    if sys.byteorder != 'little':
        values.byteswap()
    return Matrix(rows, cols, values)


def write_matrix(path, rows, cols, values):
    values = array('f', values)
    if sys.byteorder != 'little':
        values.byteswap()
    Path(path).write_bytes(struct.pack('<QQ', rows, cols) + values.tobytes())


def read_graph(path):
    raw = Path(path).read_bytes()
    n, m, directed, weighted = struct.unpack_from('<QQBB', raw)
    pointers = struct.unpack_from(f'<{n + 1}Q', raw, 18)
    sources = struct.unpack_from(f'<{m}Q', raw, 18 + 8 * (n + 1))
    weights = struct.unpack_from(f'<{m}f', raw, 18 + 8 * (n + 1 + m)) if weighted else None
    return dict(nodes=n, stored_edges=m, directed=bool(directed), col_ptr=pointers,
                row_ind=sources, weights=weights)


class Tokens:
    def __init__(self, path):
        self.path = Path(path)
        self.tokens = iter(shlex.split(self.path.read_text(encoding='utf-8'), comments=False))

    def value(self):
        return next(self.tokens)

    def skip(self, count=1):
        for _ in range(count):
            next(self.tokens)

    def field(self):
        self.skip()
        return self.value()

    def resolve(self, value):
        return self.path.parent / value


def read_model(path):
    t = Tokens(path)
    t.skip(2)  # GNN_MODEL 1
    count = int(t.field())
    result = []
    for _ in range(count):
        t.skip()  # layer
        kind, activation = t.value(), t.value()
        neigh, self_path, bias_path = t.value(), t.value(), t.value()
        w = read_matrix(t.resolve(neigh))
        ws = None if self_path == '-' else read_matrix(t.resolve(self_path))
        bias = None if bias_path == '-' else read_matrix(t.resolve(bias_path))
        result.append(dict(kind=kind, activation=activation, w_neigh=w, w_self=ws, bias=bias))
    return result


def read_workload(graph_path, feature_path, model_path):
    return dict(graph=read_graph(graph_path), features=read_matrix(feature_path),
                layers=read_model(model_path), provenance='explicit input files', seed='unspecified')
