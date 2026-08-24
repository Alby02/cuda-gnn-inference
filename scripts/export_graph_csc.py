import numpy as np
import struct

def export_graph_csc(
    filename: str,
    num_nodes: int,
    col_ptr: np.ndarray,
    row_ind: np.ndarray,
    is_directed: bool,
    weights: np.ndarray = None,
    edge_features: np.ndarray = None
):
    """
    Export graph in binary format CSC compatible with DirectedCSC / UndirectedCSC  
    """
    #implementation of boolean flag to check whether features and weights are present
    num_edges = len(row_ind)
    has_weights = int(weights is not None and len(weights) > 0)
    has_edge_feats = int(edge_features is not None and edge_features.size > 0)
    edge_feat_dim = edge_features.shape[1] if has_edge_feats else 0

    #type must be uint64 in order for it to be coherent
    col_ptr = np.asarray(col_ptr, dtype=np.uint64)
    row_ind = np.asarray(row_ind, dtype=np.uint64)

    with open(filename, 'wb') as f:
        # Writing header in binary format
        header = struct.pack(
            '<QQBBBQ', #little-endian (Linux like)
            num_nodes,
            num_edges,
            int(is_directed),
            has_weights,
            has_edge_feats,
            edge_feat_dim
        )
        f.write(header)

        # Writing CSC vectors
        col_ptr.tofile(f) #written as uint64 (cotiguous bytes like a C++ array in memory)
        row_ind.tofile(f)

        # Writing Weights and Edge Features (if present)
        if has_weights:
            np.asarray(weights, dtype=np.float32).tofile(f)

        if has_edge_feats:
            np.asarray(edge_features, dtype=np.float32).tofile(f) #matrix written in row-major

    #DEBUG
    print(f"[+] Graph saved successfully in '{filename}' ({num_nodes} nodes, {num_edges} edges).")