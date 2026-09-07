"""Run and verify fixed-weight GCN/GraphSAGE inference with PyTorch Geometric."""
import argparse
import csv
import importlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time

from result_schema import FIELDS
from workload_io import read_workload, read_matrix, write_matrix


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', required=True)
    p.add_argument('--features', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--mapping', default='pyg_models', help='Importable module with PREPROCESSING and build(tensors)')
    reference = p.add_mutually_exclusive_group(required=True)
    reference.add_argument('--native', help='Native executable used to generate the sequential reference')
    reference.add_argument('--reference', help='Existing native sequential embeddings for these inputs')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--repetitions', type=int, default=10)
    p.add_argument('--atol', type=float, default=1e-4)
    p.add_argument('--rtol', type=float, default=1e-4)
    p.add_argument('--output', required=True)
    p.add_argument('--embeddings')
    p.add_argument('--check-each-run', action='store_true', help='Verify each measured output outside its timing interval')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    start = time.perf_counter()
    workload = read_workload(args.graph, args.features, args.model)
    load_ms = (time.perf_counter() - start) * 1000
    mapping = importlib.import_module(args.mapping)
    # Generate the actual native sequential reference, outside framework timing.
    if args.reference:
        reference = read_matrix(args.reference)
    else:
        with tempfile.TemporaryDirectory(prefix='gnn-reference-') as temporary:
            reference_path = Path(temporary) / 'reference.bin_matrix'
            command = [str(Path(args.native).resolve()), '--backend', 'sequential', '--graph', args.graph,
                       '--features', args.features, '--model', args.model,
                       '--warmups', '1', '--repetitions', '1', '--embeddings', str(reference_path)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            reference = read_matrix(reference_path)
    import torch
    import torch_geometric
    import numpy as np
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def matrix(value):
        return torch.tensor(value.values, dtype=torch.float32).reshape(value.rows, value.cols)

    setup_start = time.perf_counter()
    graph = workload['graph']
    sources = torch.tensor(graph['row_ind'], dtype=torch.int64)
    destinations = torch.repeat_interleave(torch.arange(graph['nodes']), torch.tensor(np.diff(graph['col_ptr']).astype(np.int64)))
    tensors = dict(nodes=graph['nodes'], directed=graph['directed'], edge_index=torch.stack((sources, destinations)),
                   edge_weight=torch.ones(graph['stored_edges'], dtype=torch.float32) if graph['weights'] is None else torch.tensor(graph['weights'], dtype=torch.float32),
                   layers=[dict(kind=l['kind'], activation=l['activation'],
                                **{k: None if l[k] is None else matrix(l[k]) for k in ('w_neigh', 'w_self', 'bias')})
                           for l in workload['layers']])
    model = mapping.build(tensors)
    model.eval()
    x = matrix(workload['features'])
    setup_ms = (time.perf_counter() - setup_start) * 1000
    transfer_start = time.perf_counter()
    model = model.to(device=device, dtype=torch.float32)
    x = x.to(device)
    sync()
    upload_ms = (time.perf_counter() - transfer_start) * 1000
    expected = matrix(reference)

    def verify(output):
        actual = output.detach().cpu()
        if actual.dtype != torch.float32 or actual.shape != expected.shape:
            raise ValueError('Framework output dtype/shape mismatch')
        if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
            raise ValueError('Non-finite framework/reference output')
        actual64, expected64 = actual.to(torch.float64), expected.to(torch.float64)
        if not bool((torch.abs(actual64 - expected64) <= args.atol + args.rtol * torch.abs(expected64)).all()):
            raise ValueError('Framework verification failed; performance records rejected')

    samples = []
    with torch.inference_mode():
        verify(model(x))
        for _ in range(args.warmups):
            output = model(x)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
            event_start, event_stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.repetitions):
            sync()
            begin = time.perf_counter()
            if device.type == 'cuda':
                event_start.record()
            output = model(x)
            if device.type == 'cuda':
                event_stop.record()
                event_stop.synchronize()
                compute_ms = event_start.elapsed_time(event_stop)
            else:
                compute_ms = (time.perf_counter() - begin) * 1000
            samples.append((compute_ms, (time.perf_counter() - begin) * 1000))
            if args.check_each_run:
                verify(output)
        download_start = time.perf_counter()
        host_output = output.detach().cpu()
        sync()
        download_ms = (time.perf_counter() - download_start) * 1000
        verify(host_output)
    mean = statistics.mean(s[0] for s in samples)
    stddev = statistics.pstdev(s[0] for s in samples)
    record = dict(schema_version=1, runner='torch_geometric', backend=args.device, strategy=args.mapping,
                  graph_path=str(Path(args.graph).resolve()), features_path=str(Path(args.features).resolve()),
                  model_path=str(Path(args.model).resolve()),
                  model_types=';'.join(l['kind'] for l in workload['layers']),
                  feature_widths=';'.join(map(str, [workload['features'].cols] + [l['w_neigh'].cols for l in workload['layers']])),
                  nodes=graph['nodes'], stored_edges=graph['stored_edges'], layers=len(workload['layers']),
                  orientation='directed' if graph['directed'] else 'undirected', dtype='float32', seed=workload['seed'],
                  provenance=workload['provenance'], threads=args.threads, warmups=args.warmups, repetitions=args.repetitions,
                  verification='passed', atol=args.atol, rtol=args.rtol, load_ms=load_ms, setup_ms=setup_ms, upload_ms=upload_ms,
                  download_ms=download_ms, reset_ms=0, mean_ms=mean, stddev_ms=stddev,
                  hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                  software=json.dumps(dict(python=platform.python_version(), torch=torch.__version__, torch_geometric=torch_geometric.__version__, numpy=np.__version__, cuda=torch.version.cuda)),
                  command=json.dumps([sys.executable, __file__] + (sys.argv[1:] if argv is None else argv)),
                  timing_policy='eval + inference_mode; immutable input resident; compute=model(x); end_to_end=compute+synchronization; imports/native-reference/load/setup/upload/final download excluded; preprocessing=' + mapping.PREPROCESSING + '; per-sample verification=' + str(args.check_each_run),
                  memory_policy='host peak unmeasured; CUDA max_memory_allocated after warmups, includes resident tensors, excludes reserved allocator memory')
    if device.type == 'cuda':
        record['device_peak_bytes'] = torch.cuda.max_memory_allocated(device)
    # No results file is written until both verification checks have passed.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for i, (compute_ms, elapsed_ms) in enumerate(samples):
            writer.writerow(dict(record, sample=i, compute_ms=compute_ms, end_to_end_ms=elapsed_ms))
    if args.embeddings:
        Path(args.embeddings).parent.mkdir(parents=True, exist_ok=True)
        write_matrix(args.embeddings, *host_output.shape, host_output.reshape(-1).tolist())
    print(f'Verified framework compute: mean={mean:.6g} ms, population stddev={stddev:.6g} ms')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, ImportError, AttributeError, TypeError, subprocess.CalledProcessError) as error:
        print(f'Framework execution failed: {error}', file=sys.stderr)
        if isinstance(error, subprocess.CalledProcessError):
            print(error.stderr, file=sys.stderr)
        raise SystemExit(1)
