"""Download data, generate controlled workloads, verify all backends, and plot timings."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import subprocess
import sys

from compare_framework import embedding_error, main as compare, read_samples
from export_dense_matrix import export_dense_matrix
from export_graph_csc import export_graph_csc


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def save_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def export_graph(root, edges, nodes, directed, weights=None):
    import numpy as np
    from scipy.sparse import csc_matrix
    graph = csc_matrix((np.ones(edges.shape[1]) if weights is None else weights,
                        (edges[0], edges[1])), shape=(nodes, nodes))
    graph.sum_duplicates()
    graph.sort_indices()
    path = root / 'graph.bin_graph'
    export_graph_csc(path, nodes, graph.indptr, graph.indices, directed,
                     None if weights is None else graph.data)
    degrees = np.diff(graph.indptr)
    return path, dict(nodes=nodes, stored_edges=int(graph.nnz),
                      mean_degree=float(degrees.mean()), max_degree=int(degrees.max()),
                      degree_cv=float(degrees.std() / degrees.mean()) if degrees.mean() else 0.0)


def layer_specs(kind, dimensions, seed):
    import numpy as np
    import torch
    rng = np.random.default_rng(seed)
    layers = []
    for index, (left, right) in enumerate(zip(dimensions, dimensions[1:])):
        limit = (6.0 / (left + right)) ** 0.5
        def weight():
            return torch.tensor(rng.uniform(-limit, limit, (left, right)), dtype=torch.float32)
        layers.append(dict(kind=kind, activation='NONE' if index == len(dimensions) - 2 else 'RELU',
                           w_neigh=weight(), w_self=weight() if kind == 'GRAPHSAGE' else None,
                           bias=torch.zeros(1, right)))
    return layers


def export_model(root, kind, layers):
    root.mkdir(parents=True, exist_ok=True)
    lines = ['GNN_MODEL 1', f'layers {len(layers)}']
    for i, layer in enumerate(layers):
        files = []
        for name in ('w_neigh', 'w_self', 'bias'):
            value = layer[name]
            filename = '-' if value is None else f'{i}-{name}.bin_matrix'
            if value is not None:
                export_dense_matrix(root / filename, value.detach().cpu().numpy())
            files.append(filename)
        lines.append(f'layer {kind} {layer["activation"]} ' + ' '.join(f'"{f}"' for f in files))
    path = root / f'{kind.lower()}.model'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return path


def real_workload(args, root):
    directory = root / 'inputs' / args.dataset
    directory.mkdir(parents=True, exist_ok=True)
    if args.dataset == 'ogbn-arxiv':
        from ogb.nodeproppred import NodePropPredDataset
        dataset = NodePropPredDataset(name=args.dataset, root=str(root / 'downloads'))
        data, _ = dataset[0]
        edges, values = data['edge_index'], data['node_feat']
        nodes, classes, directed = data['num_nodes'], dataset.num_classes, True
        source = 'OGBN-Arxiv downloaded directed graph and original features'
    else:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(str(root / 'downloads'), args.dataset)
        data = dataset[0]
        edges, values = data.edge_index.numpy(), data.x.numpy()
        nodes, classes, directed = data.num_nodes, dataset.num_classes, False
        source = 'Planetoid downloaded graph and original features'
    graph, stats = export_graph(directory, edges, nodes, directed)
    features = directory / 'features.bin_matrix'
    export_dense_matrix(features, values)
    models = [str(export_model(directory / kind.lower(), kind,
                                layer_specs(kind, [values.shape[1], args.real_hidden, classes], args.seed)))
              for kind in args.models]
    return dict(tag=args.dataset, graph=str(graph), features=str(features), models=models,
                axes=[], width=values.shape[1], depth=2, skew='', **stats,
                provenance=source + f'; generated Xavier weights (untrained); seed={args.seed}')


def synthetic_workloads(args, root):
    import numpy as np
    baseline = (args.nodes[0], args.widths[0], args.depths[0], args.skews[0])
    configurations = {}
    for axis, values in enumerate((args.nodes, args.widths, args.depths, args.skews)):
        for value in values:
            config = list(baseline)
            config[axis] = value
            configurations.setdefault(tuple(config), []).append(('nodes', 'width', 'depth', 'skew')[axis])
    graph_cache = {}
    for (nodes, width, depth, skew), axes in configurations.items():
        tag = f'n{nodes}-w{width}-d{depth}-s{skew:g}'
        directory = root / 'inputs' / tag
        directory.mkdir(parents=True, exist_ok=True)
        graph_key = (nodes, skew)
        if graph_key not in graph_cache:
            # Fixed unique edge count; destination popularity controls incoming-degree imbalance.
            rng = np.random.default_rng(args.seed)
            probability = np.arange(1, nodes + 1, dtype=float) ** -skew
            probability /= probability.sum()
            probability = probability[rng.permutation(nodes)]
            target = min(nodes * args.degree, nodes * (nodes - 1))
            keys = np.empty(0, dtype=np.int64)
            while keys.size < target:
                count = max(1024, 2 * (target - keys.size))
                src = rng.integers(nodes, size=count)
                dst = rng.choice(nodes, size=count, p=probability)
                keys = np.unique(np.concatenate((keys, (src * nodes + dst)[src != dst])))
            # Randomly trim the final batch, without preferring particular source IDs.
            keys = rng.choice(keys, size=target, replace=False)
            edges = np.stack((keys // nodes, keys % nodes))
            weights = rng.uniform(0.5, 2.0, target).astype(np.float32)
            graph_cache[graph_key] = export_graph(directory, edges, nodes, True, weights)
        graph, stats = graph_cache[graph_key]
        # Keep graph and input fixed when changing depth; prefixes match when widening features.
        rng = np.random.default_rng(args.seed)
        features = directory / 'features.bin_matrix'
        values = rng.standard_normal((nodes, max(args.widths))).astype(np.float32)[:, :width]
        export_dense_matrix(features, values)
        models = [str(export_model(directory / kind.lower(), kind,
                                    layer_specs(kind, [width] * (depth + 1), args.seed)))
                  for kind in args.models]
        yield dict(tag=tag, graph=str(graph), features=str(features), models=models, axes=axes,
                   width=width, depth=depth, skew=skew, **stats,
                   provenance=f'generated Xavier weights (untrained); seed={args.seed}; directed weighted graph')


def audit_outputs(args, directory):
    """Compare every pair for each model, plus independent one-inference native repetitions."""
    runs = [json.loads(line) for line in (directory / 'native' / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    outputs, report = {}, []
    audit = directory / 'repeatability'
    audit.mkdir(exist_ok=True)
    for entry in runs:
        command = entry['argv']
        model = command[command.index('--model') + 1]
        baseline = Path(command[command.index('--embeddings') + 1])
        outputs.setdefault(model, []).append(baseline)
        for index in range(args.repeat_checks):
            repeated = list(command)
            path = audit / f'{entry["name"]}-{index}.bin_matrix'
            for option, value in [('--repetitions', '1'), ('--warmups', '0'),
                                  ('--embeddings', str(path)), ('--output', str(path.with_suffix('.csv')))]:
                repeated[repeated.index(option) + 1] = value
            result = subprocess.run(repeated, capture_output=True, text=True)
            path.with_suffix('.stdout.txt').write_text(result.stdout, encoding='utf-8')
            path.with_suffix('.stderr.txt').write_text(result.stderr, encoding='utf-8')
            if result.returncode:
                raise RuntimeError(f'Repeatability execution failed: {repeated}; {result.stderr}')
            passed, error = embedding_error(path, baseline, args.atol, args.rtol)
            report.append(dict(check='native repeat', model=model, left=str(path), right=str(baseline),
                               passed=passed, max_abs_error=error, command=repeated))
    for record in read_samples(directory / 'comparison.csv'):
        paths = outputs[record['model_path']]
        framework = Path(record['framework_embeddings'])
        if framework not in paths:
            paths.append(framework)
    for model, paths in outputs.items():
        for left, right in itertools.combinations(paths, 2):
            passed, error = embedding_error(left, right, args.atol, args.rtol)
            report.append(dict(check='cross execution', model=model, left=str(left), right=str(right),
                               passed=passed, max_abs_error=error))
    save_json(directory / 'verification.json', report)
    if not all(row['passed'] for row in report):
        raise RuntimeError(f'Output mismatch; see {directory / "verification.json"}')
    return len(report)


def plot_results(root, workloads, samples, comparisons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_tag = {w['tag']: w for w in workloads}
    # Per-sample CSV repeats mean/stddev: keep one measurement summary per configuration.
    measurements = {}
    for row in samples:
        kind = row['model_types'].split(';')[0]
        label = f'{row["runner"]} {row["backend"]}'
        label += f' block={row["block_size"]}' if row['backend'] == 'cuda' and row['runner'] != 'torch_geometric' else f' t={row["threads"]}'
        measurements[(row['workload'], kind, label)] = row
    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    for kind in sorted({key[1] for key in measurements}):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ax, dimension in zip(axes.flat, ('nodes', 'width', 'depth', 'skew')):
            groups = {}
            for (tag, model, label), row in measurements.items():
                workload = by_tag[tag]
                if model == kind and dimension in workload['axes']:
                    groups.setdefault(label, []).append((workload[dimension], float(row['mean_ms']), float(row['stddev_ms'])))
            for label, values in sorted(groups.items()):
                values.sort()
                ax.errorbar([v[0] for v in values], [v[1] for v in values],
                            yerr=[v[2] for v in values], marker='o', capsize=3, label=label)
            ax.set(xlabel=dimension, ylabel='Compute mean (ms)', title=f'{kind}: vary {dimension}')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            if groups:
                ax.legend(fontsize=7)
        fig.suptitle('One variable at a time; error bars = population standard deviation')
        for extension in ('png', 'svg'):
            fig.savefig(plots / f'{kind.lower()}-scaling.{extension}', dpi=160)
        plt.close(fig)
    for workload in workloads:
        rows = [r for r in comparisons if r['workload'] == workload['tag']]
        labels = [f'{r["model_types"].split(";")[0]} {r["native_backend"]}\nt={r["threads"]} block={r["block_size"]}' for r in rows]
        fig, ax = plt.subplots(figsize=(max(7, len(rows)), 5), constrained_layout=True)
        ax.bar(range(len(rows)), [float(r['native_speedup']) for r in rows])
        ax.set_xticks(range(len(rows)), labels, rotation=35, ha='right')
        ax.axhline(1, color='black', linewidth=1)
        ax.set(ylabel='PyG compute / native compute (>1: native faster)', title=workload['tag'])
        for extension in ('png', 'svg'):
            fig.savefig(plots / f'{workload["tag"]}-speedup.{extension}', dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_native = 'builddir/gnn.exe' if Path('builddir/gnn.exe').exists() else 'builddir/gnn'
    parser.add_argument('--native', default=default_native, help='Path to compiled native executable')
    parser.add_argument('--output-dir', default='experiment_results', help='Output directory for results and plots')
    parser.add_argument('--dataset', choices=['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed', 'none'], default='ogbn-arxiv')
    parser.add_argument('--models', nargs='+', choices=['GCN', 'GRAPHSAGE'], default=['GCN', 'GRAPHSAGE'])
    parser.add_argument('--real-hidden', type=int, default=32)
    parser.add_argument('--backend', nargs='+', choices=['sequential', 'parallel', 'cuda'], default=None,
                        help='Execution backends (defaults to all supported by native binary)')
    parser.add_argument('--threads', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--block-size', nargs='+', type=int, default=[256])
    parser.add_argument('--nodes', nargs='+', type=int, default=[1000, 10000])
    parser.add_argument('--widths', nargs='+', type=int, default=[32, 128])
    parser.add_argument('--depths', nargs='+', type=int, default=[2, 8])
    parser.add_argument('--skews', nargs='+', type=float, default=[0, 1, 2])
    parser.add_argument('--degree', type=int, default=8, help='Mean stored incoming degree for synthetic graphs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--repetitions', type=int, default=10)
    parser.add_argument('--repeat-checks', type=int, default=2, help='Extra single-inference native executions per configuration')
    parser.add_argument('--atol', type=float, default=1e-4)
    parser.add_argument('--rtol', type=float, default=1e-4)
    args = parser.parse_args()
    if args.backend is None:
        help_text = subprocess.run([str(Path(args.native).resolve()), '--help'], capture_output=True, text=True).stdout
        modes_line = [l for l in help_text.splitlines() if l.startswith('Modes:') or 'Available modes:' in l]
        available = [b for b in ('sequential', 'parallel', 'cuda') if any(b in l for l in modes_line)]
        args.backend = available if available else ['sequential']
        print(f'Auto-detected backends from {args.native}: {args.backend}', flush=True)
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / 'experiment.json', vars(args))
    workloads, samples, comparisons = [], [], []
    if args.dataset != 'none':
        print(f'Downloading {args.dataset} graph/features and generating seeded model weights', flush=True)
        workloads.append(real_workload(args, root))
    workloads.extend(synthetic_workloads(args, root))
    save_json(root / 'workloads.json', workloads)
    checks = 0
    for workload in workloads:
        print(f'Running {workload["tag"]}', flush=True)
        directory = root / 'results' / workload['tag']
        code = compare(['--native', args.native, '--graph', workload['graph'],
                        '--features', workload['features'], '--model', *workload['models'],
                        '--backend', *args.backend, '--threads', *map(str, args.threads),
                        '--block-size', *map(str, args.block_size), '--warmups', str(args.warmups),
                        '--repetitions', str(args.repetitions), '--atol', str(args.atol),
                        '--rtol', str(args.rtol), '--check-each-run', '--output-dir', str(directory)])
        if code:
            raise RuntimeError(f'Native/framework comparison failed for {workload["tag"]}')
        checks += audit_outputs(args, directory)
        for name, target in [('samples.csv', samples), ('comparison.csv', comparisons)]:
            target.extend(dict(workload=workload['tag'], **row) for row in read_samples(directory / name))
        save_csv(root / 'samples.csv', samples)
        save_csv(root / 'comparison.csv', comparisons)
    plot_results(root, workloads, samples, comparisons)
    save_json(root / 'complete.json', dict(workloads=len(workloads), comparison_pairs=len(comparisons),
                                          repeat_and_pairwise_checks=checks, verified=True))
    print(f'Complete: {len(workloads)} workloads, {checks} repeat/pairwise checks. Plots: {root / "plots"}')


if __name__ == '__main__':
    main()
