import struct
import numpy as np

def export_dense_matrix(filename: str, matrix: np.ndarray):
    """
    Export of dense matrix (for example for node features and weights) per DenseMatrix<float> C++.
    """
    matrix = np.asarray(matrix, dtype=np.float32) #I need float32 in order to be compatible with DenseMatrix
    rows, cols = matrix.shape

    with open(filename, 'wb') as f:
        # Header: rows e cols uint64 - like
        header = struct.pack('<QQ', rows, cols)
        f.write(header)
        matrix.tofile(f)

    #DEBUG
    print(f"[+] Matrix {rows}x{cols} saved successfully in '{filename}'.")
