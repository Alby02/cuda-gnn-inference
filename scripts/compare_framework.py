"""Compare native and PyG inference on identical GCN, GraphSAGE or mixed workloads."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from result_schema import FIELDS
from workload_io import read_matrix


def read_samples(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def embedding_error(actual_path, reference_path, atol, rtol):
    actual, reference = read_matrix(actual_path), read_matrix(reference_path)
    if (actual.rows, actual.cols) != (reference.rows, reference.cols):
        return False, math.inf
    passed = len(actual.values) == len(reference.values) and all(
        math.isfinite(a) and math.isfinite(b) and abs(a - b) <= atol + rtol * abs(b)
        for a, b in zip(actual.values, reference.values))
    return passed, max((abs(a - b) for a, b in zip(actual.values, reference.values)), default=0.0)


def default_cpu_threads():
    count = os.cpu_count() or 1
    threads, t = [], 1
    while t < count:
        threads.append(t)
        t *= 2
    if count not in threads:
        threads.append(count)
    return threads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--model', nargs='+', required=True)
    parser.add_argument('--backend', nargs='+', choices=['sequential', 'parallel', 'cuda'],
                        default=['sequential', 'parallel'])
    parser.add_argument('--threads', nargs='+', type=int, default=default_cpu_threads(),
                        help='Thread counts for parallel CPU backend (default: 2^x scaling up to cpu_count)')
    parser.add_argument('--block-size', nargs='+', type=int, default=[256])
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--check-each-run', action='store_true')
    args = parser.parse_args(argv)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    common = ['--graph', str(Path(args.graph).resolve()),
              '--features', str(Path(args.features).resolve()),
              '--warmups', str(args.warmups), '--repetitions', str(args.repetitions),
              '--atol', str(args.atol), '--rtol', str(args.rtol)]

    def run(name, command, log):
        result = subprocess.run(command, capture_output=True, text=True)
        (root / f'{name}.stdout.txt').write_text(result.stdout, encoding='utf-8')
        (root / f'{name}.stderr.txt').write_text(result.stderr, encoding='utf-8')
        log.write(json.dumps(dict(name=name, argv=command, returncode=result.returncode)) + '\n')
        log.flush()
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {root / (name + ".stderr.txt")} and stdout log')

    summary, samples, frameworks = [], [], {}
    with (root / 'commands.jsonl').open('w', encoding='utf-8') as log:
        native_dir = root / 'native'
        run('native', [sys.executable, str(scripts / 'benchmark_runner.py'),
                       '--native', str(Path(args.native).resolve()), *common,
                       '--model', *map(lambda p: str(Path(p).resolve()), args.model),
                       '--backend', *args.backend, '--threads', *map(str, args.threads),
                       '--block-size', *map(str, args.block_size),
                       '--output-dir', str(native_dir)], log)
        native_runs = [json.loads(line) for line in
                       (native_dir / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
        references = {}
        for native in native_runs:
            rows = read_samples(native_dir / f'{native["name"]}.csv')
            samples.extend(rows)
            record = rows[0]
            model, backend = record['model_path'], record['backend']
            embeddings = native_dir / f'{native["name"]}.bin_matrix'
            if backend == 'sequential':
                references[model] = embeddings
            if backend not in args.backend:
                continue
            device = 'cuda' if backend == 'cuda' else 'cpu'
            threads = int(record['threads']) if backend == 'parallel' else 1
            key = (model, device, threads)
            if key not in frameworks:
                name = f'framework-{len(frameworks):03d}-{Path(model).stem}-{device}-t{threads}'
                output, framework_embeddings = root / f'{name}.csv', root / f'{name}.bin_matrix'
                run(name, [sys.executable, str(scripts / 'framework_runner.py'), *common,
                           '--model', model, '--reference', str(references[model]),
                           '--device', device, '--threads', str(threads),
                           '--output', str(output), '--embeddings', str(framework_embeddings),
                           *(['--check-each-run'] if args.check_each_run else [])], log)
                framework_rows = read_samples(output)
                samples.extend(framework_rows)
                frameworks[key] = (framework_rows[0], framework_embeddings)
            framework, framework_embeddings = frameworks[key]
            passed, error = embedding_error(embeddings, framework_embeddings, args.atol, args.rtol)
            native_ms, framework_ms = float(record['mean_ms']), float(framework['mean_ms'])
            summary.append(dict(model_path=model, model_types=record['model_types'],
                                native_backend=backend, framework_device=device, threads=threads,
                                block_size=record['block_size'], verification='passed' if passed else 'failed',
                                max_abs_error=error, atol=args.atol, rtol=args.rtol,
                                native_mean_ms=native_ms, native_stddev_ms=record['stddev_ms'],
                                framework_mean_ms=framework_ms, framework_stddev_ms=framework['stddev_ms'],
                                native_speedup=framework_ms / native_ms if passed and native_ms > 0 else '',
                                native_embeddings=str(embeddings), framework_embeddings=str(framework_embeddings)))
            print(f'{Path(model).stem}: {backend} / PyG {device}, threads={threads}: '
                  f'{summary[-1]["verification"]}, max error={error:.6g}', flush=True)

    for name, records, fields in [('comparison.csv', summary, list(summary[0])),
                                   ('samples.csv', samples, FIELDS)]:
        with (root / name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    print(f'Comparison saved to {root / "comparison.csv"}; native_speedup = PyG/native compute time')
    return int(any(row['verification'] != 'passed' for row in summary))


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
