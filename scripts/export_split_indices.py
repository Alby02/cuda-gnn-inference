from __future__ import annotations

from typing import TYPE_CHECKING

from export_dense_matrix import export_dense_matrix

if TYPE_CHECKING:
    import numpy as np


def export_split_indices(filename: str, indices: np.ndarray, num_nodes: int):
    """Export train/validation/test indices as an N x 1 float32 mask."""
    import numpy as np

    mask = np.zeros((num_nodes, 1), dtype=np.float32)
    mask[indices] = 1.0
    export_dense_matrix(filename, mask)
