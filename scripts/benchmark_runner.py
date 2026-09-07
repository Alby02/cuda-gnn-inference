"""Run native benchmark configurations; preserve commands and diagnostics for Colab."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from workload_io import read_matrix
from result_schema import FIELDS
from benchmark_metadata import environment_metadata, workload_metadata, run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--features', required=True)
    parser.add_argument('--model', nargs='+', required=True)
    parser.add_argument('--backend', nargs='+', choices=['sequential', 'parallel', 'cuda'], default=['sequential'])
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--block-size', nargs='+', type=int, default=[128, 256])
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    environment = environment_metadata(args.native, 'cuda' in args.backend)
    workloads = {model: workload_metadata(args.graph, args.features, model) for model in args.model}
    commands = []
    backends = list(dict.fromkeys(['sequential', *args.backend]))
    for model, backend in itertools.product(args.model, backends):
        settings = args.threads if backend == 'parallel' else args.block_size if backend == 'cuda' else [None]
        for setting in settings:
            name = f'{len(commands):03d}-{Path(model).stem}-{backend}'
            command = [str(Path(args.native).resolve()), '--backend', backend, '--graph', str(Path(args.graph).resolve()),
                       '--features', str(Path(args.features).resolve()), '--model', str(Path(model).resolve()),
                       '--warmups', str(args.warmups), '--repetitions', str(args.repetitions),
                       '--output', str((root / f'{name}.csv').resolve()),
                       '--embeddings', str((root / f'{name}.bin_matrix').resolve())]
            if backend == 'parallel':
                command += ['--threads', str(setting)]
            elif backend == 'cuda':
                command += ['--block-size', str(setting)]
            commands.append((name, command, model, backend, setting))
    failed = False
    references = {}
    with (root / 'runs.jsonl').open('w', encoding='utf-8') as log:
        for name, command, model, backend, setting in commands:
            print(json.dumps(command), flush=True)
            try:
                result = subprocess.run(command, capture_output=True, text=True)
                status, stdout, stderr = result.returncode, result.stdout, result.stderr
            except OSError as error:
                status, stdout, stderr = -1, '', str(error)
            (root / f'{name}.stdout.txt').write_text(stdout, encoding='utf-8')
            (root / f'{name}.stderr.txt').write_text(stderr, encoding='utf-8')
            log.write(json.dumps(dict(name=name, argv=command, returncode=status)) + '\n')
            log.flush()
            failed |= status != 0
            if status != 0:
                continue
            output = read_matrix(root / f'{name}.bin_matrix')
            if backend == 'sequential':
                references[model] = output
                verification = 'reference'
            elif model not in references:
                verification = 'reference_failed'
            else:
                reference = references[model]
                equivalent = ((output.rows, output.cols) == (reference.rows, reference.cols)
                              and len(output.values) == len(reference.values)
                              and all(math.isfinite(actual) and math.isfinite(expected)
                                      and abs(actual - expected) <= args.atol + args.rtol * abs(expected)
                                      for actual, expected in zip(output.values, reference.values)))
                verification = 'passed' if equivalent else 'failed'
            failed |= verification in ('failed', 'reference_failed')
            csv_path = root / f'{name}.csv'
            with csv_path.open(newline='', encoding='utf-8') as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            metadata = run_metadata(args, workloads[model], environment, backend, setting, command)
            with csv_path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for row in rows:
                    row.update(metadata)
                    row.update(verification=verification, atol=args.atol, rtol=args.rtol)
                    writer.writerow(row)
            print(f'{name}: {verification}', flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
