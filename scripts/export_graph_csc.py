from __future__ import annotations

import struct
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

def export_graph_csc(
    filename: str,
    num_nodes: int,
    col_ptr: np.ndarray,
    row_ind: np.ndarray,
    is_directed: bool,
    weights: np.ndarray = None
):
    """
    Export graph topology and optional scalar weights in the native CSC format.

    Node and edge features are exported separately with export_dense_matrix().
    """
    import numpy as np

    #implementation of boolean flag to check whether features and weights are present
    num_edges = len(row_ind)
    has_weights = int(weights is not None and len(weights) > 0)

    #type must be uint64 in order for it to be coherent
    col_ptr = np.asarray(col_ptr, dtype=np.uint64)
    row_ind = np.asarray(row_ind, dtype=np.uint64)

    with open(filename, 'wb') as f:
        # Writing header in binary format
        header = struct.pack(
            '<QQBB', #little-endian
            num_nodes,
            num_edges,
            int(is_directed),
            has_weights
        )
        f.write(header)

        # Writing CSC vectors
        col_ptr.tofile(f) #written as uint64 (cotiguous bytes like a C++ array in memory)
        row_ind.tofile(f)

        # Writing weights (if present)
        if has_weights:
            np.asarray(weights, dtype=np.float32).tofile(f)

    #DEBUG
    print(f"[+] Graph saved successfully in '{filename}' ({num_nodes} nodes, {num_edges} edges).")
