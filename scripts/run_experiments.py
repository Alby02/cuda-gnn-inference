"""Download data, generate controlled workloads, verify all backends, and plot timings."""
import argparse
import csv
import itertools
import json
import os
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
    from matplotlib.ticker import LogLocator, NullFormatter

    plt.rcParams.update({
        'axes.grid': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.facecolor': 'white',
        'font.size': 10,
        'grid.alpha': 0.22,
        'legend.frameon': False,
    })

    by_tag = {w['tag']: w for w in workloads}

    def model_name(row):
        return row['model_types'].split(';')[0]

    def display_value(value):
        if value is None or value == '':
            return 'n/a'
        if isinstance(value, float):
            return f'{value:g}'
        return str(value)

    def series_label(row):
        if row['runner'] == 'torch_geometric':
            return ('PyG / CUDA' if row['backend'] == 'cuda'
                    else f'PyG / CPU (t={row["threads"]})')
        if row['backend'] == 'cuda':
            return f'Native / CUDA (b={row["block_size"]})'
        if row['backend'] == 'parallel':
            return f'Native / OpenMP (t={row["threads"]})'
        return 'Native / sequential'

    def series_color(label):
        if 'CUDA' in label and label.startswith('Native'):
            return '#2a9d8f'
        if 'CUDA' in label:
            return '#e76f51'
        if 'OpenMP' in label:
            return '#457b9d'
        if 'PyG / CPU' in label:
            return '#f4a261'
        return '#6c757d'

    def series_style(label):
        """Keep thread/block variants distinguishable in monochrome as well as colour."""
        variants = ('(t=1)', '(t=2)', '(t=4)', '(t=8)', '(t=16)',
                    '(b=128)', '(b=256)', '(b=512)')
        markers = ('o', 's', '^', 'D', 'P', 'o', 's', '^')
        marker = next((markers[index] for index, value in enumerate(variants) if value in label), 'o')
        return {'marker': marker, 'linestyle': '--' if label.startswith('PyG') else '-'}

    def series_order(label):
        families = ('Native / sequential', 'Native / OpenMP', 'Native / CUDA',
                    'PyG / CPU', 'PyG / CUDA')
        return next((index for index, value in enumerate(families) if label.startswith(value)), 99), label

    def clean_log_axis(axis):
        axis.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        axis.xaxis.set_minor_formatter(NullFormatter())

    def save_figure(fig, stem):
        for extension in ('png', 'svg'):
            fig.savefig(plots / f'{stem}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    # Per-sample CSV repeats duplicate the same summary fields. Keep one record
    # per workload/model/execution configuration for plotting.
    measurements = {}
    for row in samples:
        measurements[(row['workload'], model_name(row), series_label(row))] = row

    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    for old_plot in itertools.chain(plots.glob('*.png'), plots.glob('*.svg')):
        old_plot.unlink()

    # One readable dashboard per workload and model. This is useful even for a
    # single workload point, unlike a fake scaling chart with four empty axes.
    for workload in workloads:
        tag = workload['tag']
        for kind in sorted({key[1] for key in measurements if key[0] == tag}):
            entries = sorted(
                ((label, row) for (row_tag, model, label), row in measurements.items()
                 if row_tag == tag and model == kind),
                key=lambda item: series_order(item[0]),
            )
            if not entries:
                continue
            labels = [label for label, _ in entries]
            colors = [series_color(label) for label in labels]
            means = [float(row['mean_ms']) for _, row in entries]
            deviations = [float(row['stddev_ms']) for _, row in entries]
            throughputs = [float(workload['nodes']) * 1000.0 / value for value in means]

            compared = [row for row in comparisons
                        if row['workload'] == tag and model_name(row) == kind]
            speedup_labels = []
            speedups = []
            errors = []
            for row in compared:
                label = ('Native / CUDA (b=' + row['block_size'] + ')'
                         if row['native_backend'] == 'cuda'
                         else 'Native / OpenMP (t=' + row['threads'] + ')'
                         if row['native_backend'] == 'parallel'
                         else 'Native / sequential')
                speedup_labels.append(label)
                speedups.append(float(row['native_speedup']))
                errors.append(max(float(row['max_abs_error']), 1e-12))

            fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
            positions = list(range(len(labels)))
            axes[0, 0].barh(positions, means, xerr=deviations, color=colors, alpha=0.9,
                            error_kw={'capsize': 3, 'elinewidth': 1})
            axes[0, 0].set_yticks(positions, labels)
            axes[0, 0].invert_yaxis()
            axes[0, 0].set_xscale('log')
            clean_log_axis(axes[0, 0])
            axes[0, 0].set(title='Compute latency (lower is better)', xlabel='Mean milliseconds, log scale')

            axes[0, 1].barh(positions, throughputs, color=colors, alpha=0.9)
            axes[0, 1].set_yticks(positions, labels)
            axes[0, 1].invert_yaxis()
            axes[0, 1].set_xscale('log')
            clean_log_axis(axes[0, 1])
            axes[0, 1].set(title='Node throughput (higher is better)', xlabel='Nodes / second, log scale')

            compared_positions = list(range(len(speedup_labels)))
            compared_colors = [series_color(label) for label in speedup_labels]
            axes[1, 0].barh(compared_positions, speedups, color=compared_colors, alpha=0.9)
            axes[1, 0].set_yticks(compared_positions, speedup_labels)
            axes[1, 0].invert_yaxis()
            axes[1, 0].axvline(1.0, color='#222222', linestyle='--', linewidth=1)
            axes[1, 0].set(title='Native speed relative to matching PyG device',
                           xlabel='PyG time / native time (>1 means native is faster)')

            axes[1, 1].barh(compared_positions, errors, color=compared_colors, alpha=0.9)
            axes[1, 1].set_yticks(compared_positions, speedup_labels)
            axes[1, 1].invert_yaxis()
            axes[1, 1].set_xscale('log')
            clean_log_axis(axes[1, 1])
            axes[1, 1].axvline(float(compared[0]['atol']), color='#c1121f', linestyle='--',
                               linewidth=1, label='absolute tolerance')
            axes[1, 1].legend(loc='best')
            axes[1, 1].set(title='Numerical agreement with PyG', xlabel='Maximum absolute error, log scale')

            fig.suptitle(
                f'{kind} — {tag}\n'
                f'{workload["nodes"]:,} nodes · {workload["stored_edges"]:,} stored edges · '
                f'width {display_value(workload["width"])} · '
                f'depth {display_value(workload["depth"])} · '
                f'skew {display_value(workload["skew"])}',
                fontsize=15,
            )
            save_figure(fig, f'{tag}-{kind.lower()}-dashboard')

    # Scaling plots are created only for axes with at least two actual values.
    # Each axis gets its own figure so labels and legends remain legible.
    for kind in sorted({key[1] for key in measurements}):
        for dimension in ('nodes', 'width', 'depth', 'skew'):
            eligible = [w for w in workloads if dimension in w['axes']]
            if len({w[dimension] for w in eligible}) < 2:
                continue
            groups = {}
            sequential = {}
            for (tag, model, label), row in measurements.items():
                workload = by_tag[tag]
                if model != kind or workload not in eligible:
                    continue
                point = (workload[dimension], float(row['mean_ms']), float(row['stddev_ms']))
                groups.setdefault(label, []).append(point)
                if label == 'Native / sequential':
                    sequential[tag] = float(row['mean_ms'])

            fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), constrained_layout=True)
            for label, values in sorted(groups.items(), key=lambda item: series_order(item[0])):
                values.sort(key=lambda value: value[0])
                color = series_color(label)
                style = series_style(label)
                axes[0].errorbar(
                    [value[0] for value in values], [value[1] for value in values],
                    yerr=[value[2] for value in values], linewidth=2,
                    capsize=3, label=label, color=color, **style,
                )
                speedup_points = []
                for value in values:
                    matching = next(w for w in eligible if w[dimension] == value[0])
                    if matching['tag'] in sequential and value[1] > 0:
                        speedup_points.append((value[0], sequential[matching['tag']] / value[1]))
                if speedup_points:
                    axes[1].plot(
                        [value[0] for value in speedup_points],
                        [value[1] for value in speedup_points],
                        linewidth=2, label=label, color=color, **style,
                    )

            axes[0].set_yscale('log')
            axes[0].set(title='Compute latency', xlabel=dimension, ylabel='Mean milliseconds, log scale')
            axes[1].axhline(1.0, color='#222222', linestyle='--', linewidth=1)
            axes[1].set(title='Speedup over native sequential', xlabel=dimension,
                        ylabel='Sequential time / configuration time')
            if dimension == 'nodes':
                axes[0].set_xscale('log')
                axes[1].set_xscale('log')
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='outside lower center', ncol=3)
            fig.suptitle(f'{kind} scaling with {dimension} — one variable changed at a time', fontsize=14)
            save_figure(fig, f'{kind.lower()}-{dimension}-scaling')


def default_cpu_threads():
    count = os.cpu_count() or 1
    threads, t = [], 1
    while t < count:
        threads.append(t)
        t *= 2
    if count not in threads:
        threads.append(count)
    return threads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_native = 'builddir/gnn.exe' if Path('builddir/gnn.exe').exists() else 'builddir/gnn'
    parser.add_argument('--native', default=default_native, help='Path to compiled native executable')
    parser.add_argument('--output-dir', default='experiment_results', help='Output directory for results and plots')
    parser.add_argument('--plots-only', action='store_true',
                        help='Rebuild plots from workloads.json, samples.csv, and comparison.csv in output-dir')
    parser.add_argument('--dataset', choices=['ogbn-arxiv', 'Cora', 'CiteSeer', 'PubMed', 'none'], default='ogbn-arxiv')
    parser.add_argument('--skip-synthetic', action='store_true',
                        help='Run only the selected public dataset (invalid with --dataset none)')
    parser.add_argument('--models', nargs='+', choices=['GCN', 'GRAPHSAGE'], default=['GCN', 'GRAPHSAGE'])
    parser.add_argument('--real-hidden', type=int, default=32)
    parser.add_argument('--backend', nargs='+', choices=['sequential', 'parallel', 'cuda'], default=None,
                        help='Execution backends (defaults to all supported by native binary)')
    parser.add_argument('--threads', nargs='+', type=int, default=default_cpu_threads(),
                        help='Thread counts for parallel CPU backend (default: 2^x scaling up to cpu_count)')
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
    if args.dataset == 'none' and args.skip_synthetic:
        parser.error('--skip-synthetic requires a public --dataset')
    if any(value <= 0 for value in args.nodes):
        parser.error('--nodes values must be positive')
    if any(value <= 0 for value in args.widths):
        parser.error('--widths values must be positive')
    if any(value <= 0 for value in args.depths):
        parser.error('--depths values must be positive')
    if any(value < 0 for value in args.skews):
        parser.error('--skews values must be non-negative')
    if args.degree < 0:
        parser.error('--degree must be non-negative')
    if args.real_hidden <= 0:
        parser.error('--real-hidden must be positive')
    if any(value <= 0 for value in args.threads):
        parser.error('--threads values must be positive')
    if any(value <= 0 for value in args.block_size):
        parser.error('--block-size values must be positive')
    if args.atol < 0 or args.rtol < 0:
        parser.error('--atol and --rtol must be non-negative')
    if args.warmups < 0 or args.repetitions <= 0 or args.repeat_checks < 0:
        parser.error('--warmups/--repeat-checks must be non-negative and --repetitions positive')
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        required = [root / name for name in ('workloads.json', 'samples.csv', 'comparison.csv')]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            parser.error('cannot rebuild plots; missing ' + ', '.join(missing))
        workloads = json.loads((root / 'workloads.json').read_text(encoding='utf-8'))
        samples = read_samples(root / 'samples.csv')
        comparisons = read_samples(root / 'comparison.csv')
        plot_results(root, workloads, samples, comparisons)
        print(f'Rebuilt plots from existing results: {root / "plots"}', flush=True)
        return
    if args.backend is None:
        help_text = subprocess.run([str(Path(args.native).resolve()), '--help'], capture_output=True, text=True).stdout
        modes_line = [l for l in help_text.splitlines() if l.startswith('Modes:') or 'Available modes:' in l]
        available = [b for b in ('sequential', 'parallel', 'cuda') if any(b in l for l in modes_line)]
        args.backend = available if available else ['sequential']
        print(f'Auto-detected backends from {args.native}: {args.backend}', flush=True)
    save_json(root / 'experiment.json', vars(args))
    workloads, samples, comparisons = [], [], []
    if args.dataset != 'none':
        print(f'Downloading {args.dataset} graph/features and generating seeded model weights', flush=True)
        workloads.append(real_workload(args, root))
    if not args.skip_synthetic:
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
