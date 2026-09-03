import argparse
import os

from export_dense_matrix import export_dense_matrix
from export_graph_csc import export_graph_csc

from export_split_indices import export_split_indices



def convert_pyg_dataset(dataset_name: str, out_dir: str):
    # Direct conversion for PyTorch Geometric Planetoid datasets (Cora, CiteSeer, PubMed): since
    # they are citation networks, they are treated as undirected so is_directed=False


    try:
        import numpy as np
        import scipy.sparse as sp
        from torch_geometric.datasets import Planetoid
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"missing Python dependency '{error.name}'; "
            "run 'uv sync' to install the project environment"
        ) from error

    os.makedirs(out_dir, exist_ok=True) #creation of output directory
    dataset = Planetoid(root=f"/tmp/{dataset_name}", name=dataset_name) #download Planetoid dataset
    data = dataset[0] #Planetoid w one graph (entire citation network)

    num_nodes = data.num_nodes
    edge_index = data.edge_index.numpy() # (2, num_edges): [src, dst] -> from PyTorch tensor to numpy array
    src, dst = edge_index[0], edge_index[1]

    csc_mat = sp.csc_matrix((np.ones_like(src), (src, dst)), shape=(num_nodes, num_nodes))
    #planetoid dataset doesn't have edge weights (an edge either exits or not)

    # Save and export graph structure
    export_graph_csc(
        os.path.join(out_dir, f"{dataset_name}.bin_graph"),
        num_nodes=num_nodes,
        col_ptr=csc_mat.indptr,
        row_ind=csc_mat.indices,
        is_directed=False #planetoid have symmetric edge_index (both directions) therefore they are undirected
    )

    # Save and export node features
    if data.x is not None:
        export_dense_matrix(
            os.path.join(out_dir, f"{dataset_name}_feats.bin_matrix"),
            data.x.numpy()
        )

    # if presents, save node labels
    if data.y is not None:
        y_matrix = data.y.numpy().reshape(-1, 1) #from 1D vector of lenght n to matrix nx1 (compatible with export_dense_matrix)
        export_dense_matrix(
            os.path.join(out_dir, f"{dataset_name}_labels.bin_matrix"),
            y_matrix
        )

    #REQUIRED ONLY IF WE WANT TO COMPUTE THE ACCURACY ON A GIVEN SPLIT (OPTIONAL)
    #save from which split the node come from in order to correctly calculate the accuracy
    for split_name, mask_attr in [("train", "train_mask"), ("val", "val_mask"), ("test", "test_mask")]:
        if hasattr(data, mask_attr):
            mask = getattr(data, mask_attr).numpy().astype(np.float32).reshape(-1, 1)
            export_dense_matrix(
                os.path.join(out_dir, f"{dataset_name}_{split_name}_mask.bin_matrix"),
                mask
            )



def convert_ogb_dataset(dataset_name: str, out_dir: str):

# Direct conversion for Open Graph Benchmark datasets, they are not all undirected
# (for example: ogbn-arxiv is a directed citation graph)

    try:
        import numpy as np
        import scipy.sparse as sp
        from ogb.nodeproppred import NodePropPredDataset
    except ModuleNotFoundError as error:
        raise RuntimeError(
            f"missing Python dependency '{error.name}'; "
            "run 'uv sync' to install the project environment"
        ) from error

    os.makedirs(out_dir, exist_ok=True)
    dataset = NodePropPredDataset(name=dataset_name, root=os.path.join(out_dir, "ogb_raw")) #download ogb dataset
    graph, labels = dataset[0] #tuple (graph dictonary, labels array)

    edge_index = graph["edge_index"]  # (2, num_edges): [src, dst]
    node_feat = graph["node_feat"]
    num_nodes = node_feat.shape[0]
    src, dst = edge_index[0], edge_index[1]

    #DEBUG
    print(f"  Nodes={num_nodes}  Edges={edge_index.shape[1]}  FeatureDim={node_feat.shape[1]}")

    csc_mat = sp.csc_matrix((np.ones_like(src, dtype=np.float32), (src, dst)), shape=(num_nodes, num_nodes))

    # Save and export graph structure
    export_graph_csc(
        os.path.join(out_dir, f"{dataset_name}.bin_graph"),
        num_nodes=num_nodes,
        col_ptr=csc_mat.indptr,
        row_ind=csc_mat.indices,
        is_directed=True
    )

    # Save nodes X features
    export_dense_matrix(
        os.path.join(out_dir, f"{dataset_name}_feats.bin_matrix"),
        node_feat
    )

    # Save nodes Y features
    flat_labels = labels.flatten().reshape(-1, 1)
    export_dense_matrix(
        os.path.join(out_dir, f"{dataset_name}_labels.bin_matrix"),
        flat_labels
    )

    #REQUIRED ONLY IF WE WANT TO COMPUTE THE ACCURACY ON A GIVEN SPLIT (OPTIONAL)
    #save from which split the node come from in order to correctly calculate the accuracy
    split_idx = dataset.get_idx_split()
    for split_name, key in [("train", "train"), ("val", "valid"), ("test", "test")]:
        export_split_indices(
            os.path.join(out_dir, f"{dataset_name}_{split_name}_mask.bin_matrix"),
            split_idx[key],
            num_nodes
        )


if __name__ == "__main__": #executed if launched from command line
    parser = argparse.ArgumentParser(description="Custom Graph Format Converter")
    parser.add_argument("--mode", choices=[ "pyg", "ogb"], default="pyg",
                        help="Conversion mode: pyg (Planetoid), ogb (Open Graph Benchmark)")
    parser.add_argument("--input", type=str, default="Cora",
                        help="Dataset name Planetoid/OGB")
    parser.add_argument("--out_prefix", type=str, default="dataset_out", help="Directory/Prefix for binary output files")

    args = parser.parse_args() #read and get arguments from command line

    try:
        if args.mode == "pyg":
            convert_pyg_dataset(args.input, args.out_prefix)
        elif args.mode == "ogb":
            convert_ogb_dataset(args.input, args.out_prefix)
    except RuntimeError as error:
        parser.exit(1, f"converter: {error}\n")
