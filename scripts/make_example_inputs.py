"""Produce minimal format fixtures for T-IO-01; these are not trained models."""
import argparse
from pathlib import Path
import struct
import random
from workload_io import write_matrix


def generate(directory):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    # Two directed entries: 0 -> 1 (weight 2), 1 -> 1 (weight 3).
    (root / 'graph.bin_graph').write_bytes(struct.pack('<QQBB3Q2Q2f', 2, 2, 1, 1, 0, 0, 2, 0, 1, 2, 3))
    write_matrix(root / 'features.bin_matrix', 2, 2, [1, -2, 3, 4])
    write_matrix(root / 'weights.bin_matrix', 2, 1, [2, -1])
    write_matrix(root / 'self.bin_matrix', 2, 1, [1, 3])
    write_matrix(root / 'bias.bin_matrix', 1, 1, [0.5])
    for kind in ('GCN', 'GRAPHSAGE'):
        name = kind.lower()
        self_path = 'self.bin_matrix' if kind == 'GRAPHSAGE' else '-'
        (root / f'{name}.model').write_text(
            f'GNN_MODEL 1\nlayers 1\nlayer {kind} RELU "weights.bin_matrix" "{self_path}" "bias.bin_matrix"\n', encoding='utf-8')
    generate_multilayer(root / "multilayer")
    return root


def generate_multilayer(root):
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    nodes = 8
    # Node 0 is isolated; include unequal weights, explicit self-loops and positive weights.
    edges = [(1, 1, 2.0), (1, 2, 0.5), (3, 2, 3.0), (2, 3, 1.25),
             (3, 3, 0.75), (4, 5, 0.25), (5, 6, 2.5), (6, 5, 1.0)]
    pointers, sources, weights = [0], [], []
    for v in range(nodes):
        for u, dst, weight in edges:
            if dst == v:
                sources.append(u)
                weights.append(weight)
        pointers.append(len(sources))
    for weighted in (True, False):
        raw = struct.pack('<QQBB', nodes, len(edges), 1, int(weighted))
        raw += struct.pack(f'<{len(pointers)}Q', *pointers)
        raw += struct.pack(f'<{len(sources)}Q', *sources)
        if weighted:
            raw += struct.pack(f'<{len(weights)}f', *weights)
        (root / ('graph.bin_graph' if weighted else 'unweighted.bin_graph')).write_bytes(raw)
    dimensions = [5, 7, 3, 2]
    write_matrix(root / 'features.bin_matrix', nodes, dimensions[0],
                 [rng.uniform(-2, 2) for _ in range(nodes * dimensions[0])])
    for i, (rows, cols) in enumerate(zip(dimensions, dimensions[1:])):
        for branch in ('neighbor', 'self'):
            write_matrix(root / f'{branch}{i}.bin_matrix', rows, cols,
                         [rng.uniform(-0.6, 0.6) for _ in range(rows * cols)])
        write_matrix(root / f'bias{i}.bin_matrix', 1, cols,
                     [rng.uniform(-0.3, 0.3) for _ in range(cols)])
    for name, kinds in (('gcn', ['GCN'] * 3), ('graphsage', ['GRAPHSAGE'] * 3),
                        ('mixed', ['GCN', 'GRAPHSAGE', 'GCN'])):
        lines = ['GNN_MODEL 1', 'layers 3']
        for i, kind in enumerate(kinds):
            activation = 'NONE' if i == 2 else 'RELU'
            self_path = f'self{i}.bin_matrix' if kind == 'GRAPHSAGE' else '-'
            bias_path = '-' if i == 1 else f'bias{i}.bin_matrix'
            lines.append(f'layer {kind} {activation} "neighbor{i}.bin_matrix" "{self_path}" "{bias_path}"')
        (root / f'{name}.model').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    print(generate(parser.parse_args().directory))
