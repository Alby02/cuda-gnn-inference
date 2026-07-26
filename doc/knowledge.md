# Graph Neural Networks (GNNs) & Graph Data Representations

This document provides foundational knowledge on Graph Neural Networks (GNNs), their message-passing mechanisms, and the sparse graph data structures (CSR and CSC) used for efficient CPU and GPU inference.

---

## 1. Graph Neural Networks (GNNs) Overview

### Core Concept
Graph Neural Networks (GNNs) extend deep learning to graph-structured data $G = (V, E)$, where $V$ is the set of vertices (nodes) and $E$ is the set of edges.

* **Input**: A graph $G$ with node feature matrix $\mathbf{X} \in \mathbb{R}^{|V| \times F_{in}}$, where each node $v \in V$ has an initial feature vector $h_v^{(0)} \in \mathbb{R}^{F_{in}}$.
* **Output**: Updated node embeddings $h_v^{(L)} \in \mathbb{R}^{F_{out}}$ after passing through $L$ GNN layers, suitable for downstream tasks such as node classification or link prediction.

---

### Message-Passing Architecture
GNNs operate via iterative **message passing** (also known as neighborhood aggregation) across graph layers. For layer $l \in \{0, \dots, L-1\}$:

#### 1. Neighborhood Aggregation (Message Passing)
Each node $v \in V$ collects information (messages) from its direct neighbors $\mathcal{N}(v)$:

$$m_v^{(l)} = \text{AGGREGATE}^{(l)} \left( \{ h_u^{(l)} : u \in \mathcal{N}(v) \} \right)$$

Common aggregation operators include:
* **Sum**: $m_v^{(l)} = \sum_{u \in \mathcal{N}(v)} h_u^{(l)}$
* **Mean**: $m_v^{(l)} = \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} h_u^{(l)}$
* **Max**: $m_v^{(l)} = \max_{u \in \mathcal{N}(v)} \left( h_u^{(l)} \right)$

#### 2. Node State Update
The node combines its aggregated neighborhood message $m_v^{(l)}$ (and optionally its own state $h_v^{(l)}$) and applies a linear weight transformation $W^{(l)}$ followed by a non-linear activation function $\sigma$ (e.g., ReLU):

* **GCN Layer (Kipf & Welling)**:
  Uses normalized aggregation with self-loops ($\tilde{\mathcal{N}}(v) = \mathcal{N}(v) \cup \{v\}$):
  $$m_v^{(l)} = \sum_{u \in \tilde{\mathcal{N}}(v)} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} h_u^{(l)}$$
  $$h_v^{(l+1)} = \sigma \left( W^{(l)} m_v^{(l)} \right)$$

* **GraphSAGE Layer (Hamilton et al.)**:
  Uses unnormalized aggregation (e.g., mean/sum) over $\mathcal{N}(v)$ and concatenates self-features:
  $$h_v^{(l+1)} = \sigma \left( W^{(l)} \cdot \text{concat}\left( h_v^{(l)}, m_v^{(l)} \right) \right)$$

---

### Key Properties & Mathematical Insight
1. **Shared Parameters**: The weight matrix $W^{(l)}$ is shared across all nodes $v \in V$ in layer $l$, ensuring parameter efficiency regardless of graph size. However, different layers $l$ have distinct weight matrices $W^{(l)}$.
2. **Order of Operations (Linear Transformation vs. Aggregation)**:
   Since matrix multiplication and summation are linear and associative, linear transformations can be applied either **before** or **after** neighborhood aggregation:
   $$\mathbf{A} (\mathbf{H} \mathbf{W}) = (\mathbf{A} \mathbf{H}) \mathbf{W}$$
   * **Transform-then-Aggregate**: Compute $\tilde{\mathbf{H}}^{(l)} = \mathbf{H}^{(l)} \mathbf{W}^{(l)}$ ($|V| \times F_{out}$), then aggregate. Efficient when output dimension $F_{out} < F_{in}$.
   * **Aggregate-then-Transform**: Aggregate $\mathbf{M}^{(l)} = \mathbf{A} \mathbf{H}^{(l)}$, then multiply by $\mathbf{W}^{(l)}$. Efficient when input dimension $F_{in} < F_{out}$.

---

## 2. Node Feature Matrix Storage

Node features are stored separately from the graph structure in a **dense 2D matrix** of size $N \times F$ (where $N = |V|$ is the number of nodes, and $F$ is the feature vector dimension per node):

$$\mathbf{X} \in \mathbb{R}^{N \times F} = \begin{bmatrix}
x_{0, 0} & x_{0, 1} & \dots & x_{0, F-1} \\
x_{1, 0} & x_{1, 1} & \dots & x_{1, F-1} \\
\vdots & \vdots & \ddots & \vdots \\
x_{N-1, 0} & x_{N-1, 1} & \dots & x_{N-1, F-1}
\end{bmatrix}$$

* In contiguous 1D memory (row-major order), node $u$'s feature vector starts at memory offset `u * F` and spans through `u * F + (F - 1)`.

---

## 3. Edge Feature Matrix Storage

If edges carry multi-dimensional feature vectors $e_{uv} \in \mathbb{R}^{F_e}$ (rather than a single scalar weight), they are stored in a **dense 2D matrix** of size $|E| \times F_e$:

$$\mathbf{E}_{feat} \in \mathbb{R}^{|E| \times F_e} = \begin{bmatrix}
e_{0, 0} & e_{0, 1} & \dots & e_{0, F_e-1} \\
e_{1, 0} & e_{1, 1} & \dots & e_{1, F_e-1} \\
\vdots & \vdots & \ddots & \vdots \\
e_{|E|-1, 0} & e_{|E|-1, 1} & \dots & e_{|E|-1, F_e-1}
\end{bmatrix}$$

* The array index `idx` from `col_ind` / `row_ind` acts as the unique **Edge ID**.
* In row-major 1D memory, edge `idx`'s feature vector starts at memory offset `idx * F_e` and spans through `idx * F_e + (F_e - 1)`.

---

## 4. Sparse Graph Representations: CSR & CSC

Storing adjacency matrices for large graphs as dense $N \times N$ matrices is memory-prohibitive (e.g., a 100,000-node graph requires ~40 GB in dense 32-bit float format). Sparse formats compress the graph by storing only non-zero entries (edges).

---

### Compressed Sparse Row (CSR)

**CSR** compresses the adjacency matrix by row, making it highly efficient for querying **outgoing edges** or iterating over the neighbors of a given node.

#### CSR Structure
CSR represents a graph $G = (V, E)$ using two primary arrays (or three for weighted graphs):

1. **`row_ptr` (Row Pointers)**:
   * Array of length $|V| + 1$.
   * `row_ptr[i]` stores the starting index in `col_ind` for node $i$'s outgoing neighbors.
   * `row_ptr[i+1] - row_ptr[i]` gives the out-degree (number of neighbors) of node $i$.
   * `row_ptr[|V|]` equals total edges $|E|$.

2. **`col_ind` (Column Indices)**:
   * Array of length $|E|$.
   * Contains target neighbor node IDs stored contiguously for each row.

3. **`values`** *(Optional)*:
   * Array of length $|E|$ storing edge weights or features associated with edge $(u, v)$ at index `idx`.

#### CSR Example
Consider a directed graph with 4 nodes ($V = \{0, 1, 2, 3\}$):
* Node 0 $\rightarrow$ Node 1 (w=0.5), Node 2 (w=1.2)
* Node 1 $\rightarrow$ Node 3 (w=0.8)
* Node 2 $\rightarrow$ Node 1 (w=0.3), Node 3 (w=0.9)
* Node 3 $\rightarrow$ (no outgoing edges)

**CSR Arrays**:
* `row_ptr` = `[0, 2, 3, 5, 5]`
* `col_ind` = `[1, 2, 3, 1, 3]`
* `values`  = `[0.5, 1.2, 0.8, 0.3, 0.9]`

#### How Edge Identification Works in CSR
1. **Tail Node (Source Node $u$)**: Identified by the row index in `row_ptr`.
2. **Head Node (Target Node $v$)**: Identified by the value at `col_ind[idx]`.
3. **Edge Weight / Features**: Stored at the **exact same array index `idx`** in `values[idx]`.

For a source node $u$, its outgoing edges are located in the range of array indices:
$$\text{idx} \in [\text{row\_ptr}[u], \text{row\_ptr}[u+1])$$

```c
// Traversal over outgoing neighbors of node u in C/C++:
for (int idx = row_ptr[u]; idx < row_ptr[u + 1]; ++idx) {
    int v = col_ind[idx];       // Head (target) node ID
    float weight = values[idx]; // Scalar weight of edge (u -> v)
    
    // Multi-dimensional edge features (size F_e):
    // Feature vector for edge (u -> v) starts at: edge_features + (idx * F_e)
    for (int k = 0; k < F_e; ++k) {
        float feat_k = edge_features[idx * F_e + k];
    }
}
```

---

### Compressed Sparse Column (CSC)

**CSC** compresses the adjacency matrix by column, making it ideal for querying **incoming edges** (i.e., finding all source nodes pointing to a target destination node).

#### CSC Structure
1. **`col_ptr` (Column Pointers)**:
   * Array of length $|V| + 1$.
   * `col_ptr[j]` stores the starting index in `row_ind` for node $j$'s incoming source nodes.
   * `col_ptr[j+1] - col_ptr[j]` gives the in-degree of node $j$.

2. **`row_ind` (Row Indices)**:
   * Array of length $|E|$.
   * Contains source node IDs stored contiguously for each column.

3. **`values`** *(Optional)*:
   * Array of length $|E|$ storing edge weights or features associated with edge $(u, v)$ at index `idx`.

#### How Edge Identification Works in CSC
1. **Head Node (Target Node $v$)**: Identified by the column index in `col_ptr`.
2. **Tail Node (Source Node $u$)**: Identified by the value at `row_ind[idx]`.
3. **Edge Weight / Features**: Stored at the **exact same array index `idx`** in `values[idx]`.

```c
// Traversal over incoming source nodes of node v in C/C++:
for (int idx = col_ptr[v]; idx < col_ptr[v + 1]; ++idx) {
    int u = row_ind[idx];       // Tail (source) node ID
    float weight = values[idx]; // Weight of edge (u -> v) at the exact same index
}
```

---

### Role in GPU Acceleration (CUDA)

* **Parallel Neighborhood Traversal**: In CUDA kernels, `row_ptr` enables threads to directly access neighbor offsets with $O(1)$ lookup complexity without linear scans.
* **Coalesced Memory Access**: Storing neighbor lists in contiguous memory (`col_ind`) facilitates coalesced memory transactions on GPU warps.
* **SpMM formulation**: Message passing corresponds to Sparse Matrix-Dense Matrix Multiplication ($\text{SpMM}$), where the sparse adjacency matrix in CSR/CSC format is multiplied by the dense node feature matrix $\mathbf{H} \in \mathbb{R}^{|V| \times F}$.
