"""Validate native inference against the independent dense Python reference."""
import argparse
import math
from pathlib import Path
import struct

from framework_runner import main as run_framework
from make_example_inputs import generate
from reference_inference import reference_output
from workload_io import read_matrix


def run_checks(native_path, work_dir):
    root = Path(work_dir).resolve()
    inputs = root / 'inputs'
    results = root / 'results'
    generate(inputs)
    results.mkdir(parents=True, exist_ok=True)

    # Eight isolated nodes exercise the zero-neighbour paths.
    empty = inputs / 'empty.bin_graph'
    empty.write_bytes(struct.pack('<QQBB9Q', 8, 0, 1, 0, *([0] * 9)))

    # Two reciprocal weighted edges and six isolated nodes.
    undirected = inputs / 'undirected.bin_graph'
    undirected.write_bytes(
        struct.pack(
            '<QQBB9Q2Q2f',
            8, 2, 0, 1,
            0, 1, 2, 2, 2, 2, 2, 2, 2,
            1, 0,
            2.0, 2.0,
        )
    )

    multilayer = inputs / 'multilayer'
    no_bias = inputs / 'graphsage-no-bias.model'
    neighbor_path = (multilayer / 'neighbor0.bin_matrix').as_posix()
    self_path = (multilayer / 'self0.bin_matrix').as_posix()
    no_bias.write_text(
        'GNN_MODEL 1\n'
        'layers 1\n'
        f'layer GRAPHSAGE NONE "{neighbor_path}" "{self_path}" "-"\n',
        encoding='utf-8',
    )

    cases = [
        ('empty-gcn', empty, multilayer / 'gcn.model'),
        ('empty-sage', empty, no_bias),
        ('undirected-mixed', undirected, multilayer / 'mixed.model'),
        ('no-bias-sage', multilayer / 'graph.bin_graph', no_bias),
    ]

    native = Path(native_path).resolve()
    for name, graph, model in cases:
        output_csv = results / f'{name}.csv'
        output_matrix = results / f'{name}.bin_matrix'
        result = run_framework([
            '--native', str(native),
            '--graph', str(graph),
            '--features', str(multilayer / 'features.bin_matrix'),
            '--model', str(model),
            '--warmups', '0',
            '--repetitions', '2',
            '--output', str(output_csv),
            '--embeddings', str(output_matrix),
        ])
        if result:
            raise RuntimeError(f'{name}: framework runner failed')

        actual = read_matrix(output_matrix)
        expected = [
            value
            for row in reference_output(graph, multilayer / 'features.bin_matrix', model)
            for value in row
        ]
        if len(actual.values) != len(expected):
            raise RuntimeError(f'{name}: output dimension mismatch')
        if not all(
            math.isfinite(value)
            and abs(value - reference) <= 1e-4 + 1e-4 * abs(reference)
            for value, reference in zip(actual.values, expected)
        ):
            raise RuntimeError(f'{name}: numerical mismatch against dense reference')
        print(f'{name}: passed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_native = 'builddir/gnn.exe' if Path('builddir/gnn.exe').exists() else 'builddir/gnn'
    parser.add_argument('--native', default=default_native, help='Path to the native executable')
    parser.add_argument('--work-dir', default='experiment_results/edge-cases',
                        help='Directory for generated fixtures and results')
    args = parser.parse_args(argv)
    run_checks(args.native, args.work_dir)


if __name__ == '__main__':
    main()
