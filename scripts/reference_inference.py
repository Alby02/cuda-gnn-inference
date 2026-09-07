"""Small-workload dense Python reference for GCN and weighted-mean GraphSAGE.

Uses double-precision Python arithmetic and dense adjacency; intended for correctness
checks on small generated workloads, not performance benchmarks.
"""
import math
from workload_io import read_workload


def matrix_rows(matrix):
    return [list(matrix.values[i * matrix.cols:(i + 1) * matrix.cols]) for i in range(matrix.rows)]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def reference_output(graph_path, features_path, model_path):
    workload = read_workload(graph_path, features_path, model_path)
    graph = workload['graph']
    n = graph['nodes']
    adjacency = [[0.0] * n for _ in range(n)]
    explicit_self = [False] * n
    for v in range(n):
        for e in range(graph['col_ptr'][v], graph['col_ptr'][v + 1]):
            u = graph['row_ind'][e]
            adjacency[v][u] += graph['weights'][e] if graph['weights'] is not None else 1.0
            explicit_self[v] |= u == v
    gcn = [row[:] for row in adjacency]
    for v in range(n):
        if not explicit_self[v]:
            gcn[v][v] += 1.0
    degree = list(map(sum, gcn))
    gcn = [[value / math.sqrt(degree[v] * degree[u]) if degree[v] * degree[u] > 0 else 0.0
            for u, value in enumerate(row)] for v, row in enumerate(gcn)]
    mean = [[0.0] * n for _ in range(n)]
    for v in range(n):
        for e in range(graph['col_ptr'][v], graph['col_ptr'][v + 1]):
            u = graph['row_ind'][e]
            if u != v:
                mean[v][u] += 1.0
        total = sum(mean[v])
        if total > 0:
            mean[v] = [value / total for value in mean[v]]
    output = matrix_rows(workload['features'])
    for layer in workload['layers']:
        # Aggregate first, then multiply: independent order from the native GCN path.
        aggregated = multiply(gcn if layer['kind'] == 'GCN' else mean, output)
        result = multiply(aggregated, matrix_rows(layer['w_neigh']))
        if layer['kind'] == 'GRAPHSAGE' and layer['w_self'] is not None:
            own = multiply(output, matrix_rows(layer['w_self']))
            result = [[a + b for a, b in zip(row, self_row)] for row, self_row in zip(result, own)]
        if layer['bias'] is not None:
            result = [[x + bias for x, bias in zip(row, layer['bias'].values)] for row in result]
        if layer['activation'] == 'RELU':
            result = [[max(0.0, x) for x in row] for row in result]
        output = result
    return output


def check_runs(directories, report_path):
    import csv
    import json
    from pathlib import Path
    from workload_io import read_matrix
    references = {}
    report = []
    for directory in map(Path, directories):
        for line in (directory / 'runs.jsonl').read_text(encoding='utf-8').splitlines():
            run = json.loads(line)
            command = run['argv']
            def option(name):
                return command[command.index(name) + 1]
            key = tuple(option(name) for name in ('--graph', '--features', '--model'))
            if key not in references:
                references[key] = reference_output(*key)
            expected = references[key]
            actual = read_matrix(option('--embeddings'))
            flat = [x for row in expected for x in row]
            assert run['returncode'] == 0
            assert (actual.rows, actual.cols) == (len(expected), len(expected[0]))
            assert len(actual.values) == len(flat)
            errors = [abs(a - b) for a, b in zip(actual.values, flat)]
            assert all(math.isfinite(a) and abs(a - b) <= 1e-4 + 1e-4 * abs(b)
                       for a, b in zip(actual.values, flat)), run['name']
            with Path(option('--output')).open(newline='', encoding='utf-8') as stream:
                samples = list(csv.DictReader(stream))
            assert len(samples) == int(option('--repetitions'))
            assert all(math.isfinite(float(row['compute_ms'])) and float(row['compute_ms']) >= 0
                       for row in samples)
            report.append(dict(name=run['name'], directory=str(directory), passed=True,
                               samples=len(samples), max_abs_error=max(errors, default=0)))
    Path(report_path).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"{len(report)} runs passed independent reference checks; "
          f"{sum(row['samples'] for row in report)} measured samples; "
          f"max absolute error {max(row['max_abs_error'] for row in report):.8g}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', nargs='+', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    check_runs(args.results_dir, args.report)
