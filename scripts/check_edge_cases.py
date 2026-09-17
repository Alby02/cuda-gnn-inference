"""Validate native inference against the independent dense Python reference."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess

from framework_runner import main as run_framework
from make_example_inputs import generate
from reference_inference import reference_output
from workload_io import read_matrix


def assert_matches(path, expected, label):
    actual = read_matrix(path)
    if len(actual.values) != len(expected):
        raise RuntimeError(f'{label}: output dimension mismatch')
    maximum_error = max(
        (abs(value - reference) for value, reference in zip(actual.values, expected)),
        default=0.0,
    )
    if not all(
        math.isfinite(value)
        and abs(value - reference) <= 1e-4 + 1e-4 * abs(reference)
        for value, reference in zip(actual.values, expected)
    ):
        raise RuntimeError(f'{label}: numerical mismatch (max abs error={maximum_error:.6g})')
    return maximum_error


def supported_backends(native):
    result = subprocess.run([str(native), '--help'], check=True, capture_output=True, text=True)
    modes = next((line for line in result.stdout.splitlines() if line.startswith('Modes:')), '')
    return [name for name in ('sequential', 'parallel', 'cuda') if name in modes]


def run_checks(native_path, work_dir, requested_backends=None):
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
    available = supported_backends(native)
    backends = requested_backends or available
    unavailable = sorted(set(backends) - set(available))
    if unavailable:
        raise RuntimeError(f'Native executable does not provide requested backends: {unavailable}')

    report = []
    for name, graph, model in cases:
        expected = [
            value
            for row in reference_output(graph, multilayer / 'features.bin_matrix', model)
            for value in row
        ]

        output_csv = results / f'{name}-pyg.csv'
        output_matrix = results / f'{name}-pyg.bin_matrix'
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
        error = assert_matches(output_matrix, expected, f'{name}/PyG')
        report.append(dict(case=name, implementation='PyG/CPU', max_abs_error=error,
                           passed=True))

        for backend in backends:
            native_matrix = results / f'{name}-{backend}.bin_matrix'
            command = [
                str(native), '--backend', backend,
                '--graph', str(graph),
                '--features', str(multilayer / 'features.bin_matrix'),
                '--model', str(model),
                '--warmups', '0', '--repetitions', '2',
                '--embeddings', str(native_matrix),
            ]
            if backend == 'parallel':
                command += ['--threads', '2']
            elif backend == 'cuda':
                command += ['--block-size', '256']
            native_result = subprocess.run(command, capture_output=True, text=True)
            (results / f'{name}-{backend}.stdout.txt').write_text(
                native_result.stdout, encoding='utf-8')
            (results / f'{name}-{backend}.stderr.txt').write_text(
                native_result.stderr, encoding='utf-8')
            if native_result.returncode:
                raise RuntimeError(
                    f'{name}/{backend}: native execution failed: {native_result.stderr.strip()}')
            error = assert_matches(native_matrix, expected, f'{name}/{backend}')
            report.append(dict(case=name, implementation=f'native/{backend}',
                               max_abs_error=error, passed=True))

        print(f'{name}: PyG + {", ".join(backends)} passed')

    (root / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Completed {len(report)} independent implementation checks; report: '
          f'{root / "verification.json"}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_native = 'builddir/gnn.exe' if Path('builddir/gnn.exe').exists() else 'builddir/gnn'
    parser.add_argument('--native', default=default_native, help='Path to the native executable')
    parser.add_argument('--work-dir', default='experiment_results/edge-cases',
                        help='Directory for generated fixtures and results')
    parser.add_argument('--backend', nargs='+', choices=['sequential', 'parallel', 'cuda'],
                        help='Backends to test (default: every backend in the native executable)')
    args = parser.parse_args(argv)
    run_checks(args.native, args.work_dir, args.backend)


if __name__ == '__main__':
    main()
