import numpy as np
import export_dense_matrix

def export_split_indices(filename: str, indices: np.ndarray, num_nodes: int):
    
    # Export a split (train/val/test) as a Nx1 mask of 0.0/1.0, 
    # reusing the same binary format as export_dense_matrix ('<QQ' header + float32).

    mask = np.zeros((num_nodes, 1), dtype=np.float32)
    mask[indices] = 1.0
    export_dense_matrix(filename, mask)