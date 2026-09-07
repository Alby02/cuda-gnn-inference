

import argparse
import itertools
import json
import os

import numpy as np

from synthetic_generator import generate_synthetic_graph
from export_dense_matrix import export_dense_matrix


def count_gcn_messages(csc_mat, num_nodes: int) -> dict:
    
    indptr = csc_mat.indptr
    indices = csc_mat.indices

    stored_edges = int(csc_mat.nnz)
    explicit_self_loops = 0
    for v in range(num_nodes):
        start, end = indptr[v], indptr[v + 1]
        if v in indices[start:end]:
            explicit_self_loops += 1

    effective_messages = stored_edges + (num_nodes - explicit_self_loops)
    return {
        "stored_edges": stored_edges,
        "explicit_self_loops": explicit_self_loops,
        "effective_messages": effective_messages,
    }


def generate_gcn_layer_params(rng: np.random.Generator, dims: list[int]) -> list[np.ndarray]:
    """Generates one Glorot-uniform weight matrix per GCN layer""" 
    weights = []
    for f_in, f_out in zip(dims[:-1], dims[1:]):
        limit = np.sqrt(6.0 / (f_in + f_out))  # Glorot/Xavier uniform bound
        w = rng.uniform(-limit, limit, size=(f_in, f_out)).astype(np.float32)
        weights.append(w)
    return weights


def run_sweep(
    node_sizes: list[int],
    feature_dims: list[int],
    depths: list[int],
    graph_type: str,
    out_root: str,
    seed: int,
    directed: bool,
) -> None:
    manifest_entries = []

    for num_nodes, feature_dim, depth in itertools.product(node_sizes, feature_dims, depths):
        tag = f"{graph_type}_n{num_nodes}_f{feature_dim}_d{depth}"
        out_prefix = os.path.join(out_root, tag, "graph")
        os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    
        config_seed = int(
            np.random.SeedSequence([seed, num_nodes, feature_dim, depth]).generate_state(1)[0]
        )

        stats = generate_synthetic_graph(
            graph_type=graph_type,
            num_nodes=num_nodes,
            feature_dim=feature_dim,
            out_prefix=out_prefix,
            is_directed=directed,
            seed=config_seed,
        )

        message_counts = count_gcn_messages(stats["csc_mat"], stats["num_nodes"])

        layer_dims = [feature_dim] * (depth + 1)
        rng = np.random.default_rng(config_seed + 1)  
        weights = generate_gcn_layer_params(rng, layer_dims)

        weight_filenames = []
        for i, w in enumerate(weights):
            w_filename = f"{out_prefix}_gcn_layer{i}_weight.bin_matrix"
            export_dense_matrix(w_filename, w)
            weight_filenames.append(w_filename)

        model_filename = f"{out_prefix}_gcn.model"
        with open(model_filename, 'w', encoding='utf-8') as model_file:
            model_file.write(f'GNN_MODEL 1\nlayers {depth}\n')
            for i, filename in enumerate(weight_filenames):
                activation = 'NONE' if i == depth - 1 else 'RELU'
                model_file.write(f'layer GCN {activation} "{os.path.basename(filename)}" "-" "-"\n')

        manifest_entries.append({
            "tag": tag,
            "graph_type": graph_type,
            "num_nodes": num_nodes,
            "feature_dim": feature_dim,
            "depth": depth,
            "directed": stats["is_directed"],
            "seed": config_seed,
            "graph_file": stats["graph_filename"],
            "features_file": stats["feats_filename"],
            "gcn_weight_files": weight_filenames,
            "model_file": model_filename,
            "layer_dims": layer_dims,
            **message_counts,
        })

    manifest_path = os.path.join(out_root, "gcn_sweep_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"[+] Wrote {len(manifest_entries)} workload configurations.")
    print(f"[+] Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN synthetic workload sweep (T-DATA-02/04/05)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000],
                        help="Node counts to sweep (BEN-COMP-04: at least one order of magnitude)")
    parser.add_argument("--feature-dims", type=int, nargs="+", default=[32, 128],
                        help="Node feature dimensions to sweep (BEN-COMP-05)")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4],
                        help="Number of GCN layers to sweep (BEN-COMP-06)")
    parser.add_argument("--type", choices=["barabasi_albert", "erdos_renyi", "watts_strogatz"],
                        default="barabasi_albert", help="Synthetic graph family (DATA-SYN-01)")
    parser.add_argument("--out-root", type=str, default="./synth_data/gcn_sweep",
                        help="Root output directory for the sweep")
    parser.add_argument("--seed", type=int, default=42, help="Top-level reproducibility seed")
    parser.add_argument("--directed", action="store_true", help="Generate directed graphs")

    args = parser.parse_args()

    run_sweep(
        node_sizes=args.sizes,
        feature_dims=args.feature_dims,
        depths=args.depths,
        graph_type=args.type,
        out_root=args.out_root,
        seed=args.seed,
        directed=args.directed,
    )