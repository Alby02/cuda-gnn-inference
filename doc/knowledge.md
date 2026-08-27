# GNN and Graph-Representation Knowledge Base

## 1. Purpose

This document explains the concepts needed to understand and implement the project. It is educational rather than normative.

The binding selections and mathematical conventions are defined in [semantics.md](semantics.md), software responsibilities in [architecture.md](architecture.md), and project requirements in [requirements.md](requirements.md).

## 2. Graphs, features, and adjacency

A graph is $G=(V,E)$, where $V$ is a set of nodes and $E$ is a set of edges. A directed edge:

$$
u \rightarrow v
$$

has source, or tail, $u$ and destination, or head, $v$.

Each node has an input feature vector. Placing these vectors in rows produces:

$$
X \in \mathbb{R}^{N \times F_{in}},
$$

where $N=|V|$ and $F_{in}$ is the input feature dimension. After $L$ GNN layers, the model produces a node embedding/feature matrix:

$$
H^{(L)} \in \mathbb{R}^{N \times F_{out}}.
$$

### 2.1 Project adjacency convention

This project uses:

$$
A_{u,v} = w_{u,v}
$$

for an edge from source $u$ to destination $v$. An absent edge has value zero. With this convention:

- row $u$ contains the edges leaving $u$;
- column $v$ contains the edges entering $v$;
- CSR naturally exposes outgoing destinations; and
- CSC naturally exposes incoming sources.

This is a project convention, not a universal rule. Some mathematical texts and sparse-library APIs store $A_{v,u}$ for the same edge so that $AH$ directly represents incoming aggregation. The chosen convention must therefore be stated whenever graph data crosses a library boundary.

### 2.2 Duplicate entries

A mathematical adjacency matrix has one value at each coordinate $(u,v)$, so it cannot contain two distinct entries for the same ordered pair.

An edge-list or malformed CSR/CSC file can still repeat the same pair physically. Some sparse libraries permit this intermediate representation and later combine repeated values. This project instead defines duplicate ordered pairs as invalid input: converters produce unique pairs and the loader rejects repetitions.

### 2.3 Directed and undirected storage

A directed edge $u \rightarrow v$ is stored once.

An undirected edge $\{u,v\}$ is normally expanded into two adjacency entries:

$$
u \rightarrow v
\quad \text{and} \quad
v \rightarrow u.
$$

This makes the same source-to-target message-passing code usable for both graph orientations. A self-loop $v \rightarrow v$, if explicitly present in the loaded file, is stored once.

Because inference uses the same incoming-neighbor traversal in both cases, orientation can be metadata on one CSC graph type, for example an enum with `DIRECTED` and `UNDIRECTED`. Separate graph classes are useful only if their storage or operations actually differ. If a graph is labelled undirected, validation should ensure that every non-self entry has a reciprocal entry with the same weight. Orientation need not be tested inside an aggregation loop or CUDA kernel.

It is useful to distinguish:

- logical edges, where an undirected pair may count as one; and
- loaded adjacency entries, which are physically present in CSR/CSC; and
- effective GCN messages, which also include any implicit self contributions.

Storage metrics use the number of loaded adjacency entries. A GCN throughput result should state whether it uses loaded entries or effective messages after implicit self contributions are included.

## 3. Message passing

At layer $l$, node $v$ receives information from source nodes with edges into $v$:

$$
\mathcal{N}_{in}(v)
=
\{u \mid (u,v)\in E\}.
$$

A generic incoming aggregation is:

$$
m_v^{(l)}
=
\operatorname{AGGREGATE}
\left(
\{h_u^{(l)} : u\in\mathcal{N}_{in}(v)\}
\right).
$$

The update stage transforms the aggregate into the next embedding:

$$
h_v^{(l+1)}
=
\operatorname{UPDATE}
\left(h_v^{(l)},m_v^{(l)}\right).
$$

Sum, mean, maximum, and attention-weighted sum are common aggregation operators.

### 3.1 Pull and push execution

Message passing can be viewed in two equivalent ways.

**Pull aggregation** assigns work to destination $v$. The worker traverses the incoming sources of $v$, reads their features, and accumulates directly into $v$'s output.

**Push aggregation** assigns work to source $u$ or to its outgoing edges. Contributions are sent to destination nodes. Multiple workers may update the same destination and therefore need atomics, locks, private partial buffers, or a separate reduction.

Pull aggregation is attractive for the vertex-centric implementation because:

- a destination's incoming adjacency is contiguous in CSC;
- one worker can own the destination output row;
- no per-edge message list needs to be materialized; and
- aggregation normally needs no synchronization between destinations.

Pull does not make source feature access contiguous: neighboring source IDs may still be irregular. Its primary advantages here are output ownership and direct accumulation.

## 4. From model semantics to execution

A GNN model applies an ordered sequence of graph-wide layers. Each layer consumes an $N\times F_l$ node matrix and produces an $N\times F_{l+1}$ matrix, so adjacent feature dimensions must agree. The mathematical layer rule does not change when execution moves from sequential C++ to OpenMP or CUDA.

The selected GNN types and their equations are defined in [semantics.md](semantics.md). 

At a high level, the model runner processes layers in dependency order, the layer algorithm requests mathematical operations, and the selected backend maps those operations to hardware work. See the native-engine composition diagram in `architecture.md` for the normative component relationships and data flow.

## 5. Graph Convolutional Networks

GCN is one of the two model workloads selected in `semantics.md`. A GCN layer combines:

1. a shared linear transformation;
2. degree-normalized incoming aggregation;
3. an optional bias; and
4. an optional activation.

Using the project's adjacency convention, one layer is shown below. The matrices in the equation are unpacked immediately afterward.

$$
H^{(l+1)}
=
\sigma_l
\left(
D^{-1/2} \widehat{A}^T D^{-1/2}
H^{(l)}W^{(l)}
+ \mathbf{1}b^{(l)T}
\right).
$$

### 5.1 Meaning of A, effective A, and D

#### Adjacency matrix A

$A$ is the adjacency matrix loaded from the graph file. Under the project convention:

$$
A_{u,v}
=
\begin{cases}
w_{u,v}, & \text{if the loaded graph contains }u\rightarrow v, \\
0, & \text{otherwise}.
\end{cases}
$$

The first index is always the source and the second index is always the destination. Therefore row $u$ describes outgoing edges from $u$, while column $v$ describes incoming edges to $v$.

#### Effective adjacency matrix

$\widehat{A}$ is the adjacency that the GCN operation effectively uses. It equals $A$ except for a missing diagonal entry:

$$
\widehat{A}_{v,v}
=
\begin{cases}
A_{v,v}, & \text{if the loaded graph contains }v\rightarrow v, \\
1, & \text{if that self-loop is absent}.
\end{cases}
$$

Thus $A$ describes the stored graph, while $\widehat{A}$ also describes the implicit self contributions added by the GCN kernel. Constructing $\widehat{A}$ does not require physically changing the CSR/CSC arrays.

#### Degree matrix D

$D$ is a diagonal matrix containing the weighted incoming degree of every node in the effective adjacency:

$$
D
=
\operatorname{diag}(d_0,d_1,\dots,d_{N-1}),
$$

where:

$$
d_v
=
\sum_{u=0}^{N-1}\widehat{A}_{u,v}.
$$

All off-diagonal entries of $D$ are zero. Its inverse square root is also diagonal:

$$
D^{-1/2}
=
\operatorname{diag}
\left(
\frac{1}{\sqrt{d_0}},
\frac{1}{\sqrt{d_1}},
\dots,
\frac{1}{\sqrt{d_{N-1}}}
\right).
$$

The effective self contribution ensures $d_v>0$ for every node, so these divisions are defined.

#### What the two degree-normalization factors do

Define the normalized aggregation matrix:

$$
S
=
D^{-1/2}\widehat{A}^{T}D^{-1/2}.
$$

Its element at destination row $v$ and source column $u$ is:

$$
S_{v,u}
=
\frac{\widehat{A}_{u,v}}{\sqrt{d_vd_u}}.
$$

The right-hand $D^{-1/2}$ scales the source contribution from node $u$ by $1/\sqrt{d_u}$. The left-hand $D^{-1/2}$ scales the completed destination row $v$ by $1/\sqrt{d_v}$.

Therefore, for effective edge $u\rightarrow v$, the normalized message coefficient is:

$$
\widehat{\alpha}_{u,v}
=
\frac{\widehat{w}_{u,v}}{\sqrt{d_ud_v}}.
$$

On directed graphs, this project uses the same weighted incoming-degree vector for both $d_u$ and $d_v$. On undirected graphs, incoming and outgoing degrees are equal.

The exact self-loop, degree, bias, activation, and numerical rules are specified in [semantics.md](semantics.md).

The original spectral GCN formulation is most natural for undirected graphs. Directed graphs admit multiple normalization conventions involving incoming degree, outgoing degree, or separate source and destination factors. [semantics.md](semantics.md) selects one source-to-target convention so that every native executor and strategy uses the same definition.

### 5.2 Transform and aggregate order

The linear operations can be associated in two ways:

$$
\left(D^{-1/2}\widehat{A}^TD^{-1/2}H\right)W
=
D^{-1/2}\widehat{A}^TD^{-1/2}(HW).
$$

Therefore an implementation can:

- aggregate $F_{in}$ features and then transform; or
- transform to $F_{out}$ features and then aggregate.

In exact arithmetic the results are equal. In floating-point arithmetic, summation order may introduce small differences.

The cheaper order can depend on dimensions:

- aggregate then transform may be preferable when $F_{in}<F_{out}$;
- transform then aggregate may be preferable when $F_{out}<F_{in}$.

This is a useful optimization dimension because it changes both dense-matrix work and per-edge feature traffic.

### 5.3 Self-loops

Self-loops allow a node's current representation to participate in its next GCN representation. The loaded file determines whether a self-loop is explicitly stored for each node.

For a GCN layer:

- an explicit self-loop is processed using its loaded weight;
- if the self-loop is absent, the kernel adds an implicit self contribution with weight $1$;
- the implicit contribution is included in degree normalization; and
- the CSR/CSC arrays remain unchanged.

The kernel must detect or be told whether the explicit self-loop exists so that it never applies both an explicit and an implicit contribution.

## 6. Mean-aggregator GraphSAGE

GraphSAGE is the second GNN architecture selected in `semantics.md`. The selected mean-aggregator form keeps a node's own representation separate from the representations received from its neighbors. For destination node $v$, define the non-self incoming neighborhood:

$$
\mathcal{N}^{-}(v)=\{u\mid (u,v)\in E,\ u\ne v\}.
$$

With positive scalar edge weights, the neighbor branch computes:

$$
s_v=\sum_{u\in\mathcal{N}^{-}(v)}w_{u,v},
$$

$$
m_v=
\begin{cases}
\dfrac{1}{s_v}\displaystyle\sum_{u\in\mathcal{N}^{-}(v)}w_{u,v}h_u, & s_v>0,\\[6pt]
0, & s_v=0.
\end{cases}
$$

The layer then combines separate self and neighbor transformations:

$$
h'_v=\sigma\!\left(h_vW_{self}+m_vW_{neigh}+b\right).
$$

When every edge weight is `1`, this is the ordinary arithmetic mean of the non-self incoming neighbors. A node with no such neighbors receives a zero neighbor vector, so its output still contains the self branch and optional bias.

An explicitly stored self-loop is not part of the GraphSAGE neighbor mean: the node already contributes through $h_vW_{self}$. GraphSAGE also does not add the implicit self-loop used by GCN. This distinction belongs to the layer semantics, not to the graph container or executor.

The neighbor transformation may happen before or after the mean because the transformation is linear. The self transformation remains a separate branch. Some frameworks implement the same equation with concatenation and one larger matrix; an adapter may use that form only if it maps the parameters and output exactly.

GCN and mean GraphSAGE therefore share graph traversal machinery but not the same layer rule:

| Property | GCN | Mean GraphSAGE |
|---|---|---|
| Self contribution | One effective normalized self message | Separate self transformation |
| Neighbor aggregation | Degree-normalized weighted sum | Weighted non-self mean |
| Main matrices | One layer matrix | Separate self and neighbor matrices |
| Empty loaded neighborhood | Still has an implicit self message if no explicit one exists | Zero neighbor branch; self branch remains |

## 7. Dense feature storage

Node features use a dense row-major matrix:

$$
H =
\begin{bmatrix}
h_{0,0} & h_{0,1} & \dots & h_{0,F-1} \\
h_{1,0} & h_{1,1} & \dots & h_{1,F-1} \\
\vdots & \vdots & \ddots & \vdots \\
h_{N-1,0} & h_{N-1,1} & \dots & h_{N-1,F-1}
\end{bmatrix}.
$$

The feature vector for node $v$ begins at flat offset:

$$
vF.
$$

Thus the element at row $v$, feature $f$, is located at:

$$
vF+f.
$$

Row-major storage makes all features vectors contuguos one node to the next.
## 8. Sparse and dense graph representations

A dense $N\times N$ adjacency matrix uses $O(N^2)$ storage. A compressed sparse representation stores only $M$ adjacency entries and uses $O(N+M)$ storage.

CSR and CSC contain the same information but group entries along different matrix dimensions.

Canonical rows or columns keep their node indices sorted. This makes duplicate detection deterministic and lets converters emit stable byte-for-byte output for the same input and seed.

### 8.1 Compressed Sparse Row

Under the project convention, CSR stores outgoing adjacency.

It contains:

1. **row_ptr**, with $N+1$ entries;
2. **col_ind**, with $M$ destination-node IDs; and
3. optional **values**, with $M$ scalar edge weights.

For source $u$, its outgoing destinations occupy:

$$
\text{row\_ptr}[u]
\leq e <
\text{row\_ptr}[u+1].
$$

For each such position $e$:

- the source is $u$;
- the destination is $\text{col\_ind}[e]$; and
- the optional weight is $\text{values}[e]$.

CSR is convenient for push-based traversal from sources.

### 8.2 Compressed Sparse Column

Under the project convention, CSC stores incoming adjacency.

It contains:

1. **col_ptr**, with $N+1$ entries;
2. **row_ind**, with $M$ source-node IDs; and
3. optional **values**, with $M$ scalar edge weights.

For destination $v$, its incoming sources occupy:

$$
\text{col\_ptr}[v]
\leq e <
\text{col\_ptr}[v+1].
$$

For each such position $e$:

- the source is $\text{row\_ind}[e]$;
- the destination is $v$; and
- the optional weight is $\text{values}[e]$.

CSC is convenient for destination-owned pull aggregation.

### 8.3 CSR/CSC example

Consider the weighted directed edges:

- $0\rightarrow1$, weight $0.5$;
- $0\rightarrow2$, weight $1.2$;
- $1\rightarrow3$, weight $0.8$;
- $2\rightarrow1$, weight $0.3$; and
- $2\rightarrow3$, weight $0.9$.

The CSR arrays are:

- **row_ptr** = $[0,2,3,5,5]$
- **col_ind** = $[1,2,3,1,3]$
- **values** = $[0.5,1.2,0.8,0.3,0.9]$

The equivalent CSC arrays are:

- **col_ptr** = $[0,0,2,3,5]$
- **row_ind** = $[0,2,0,1,2]$
- **values** = $[0.5,0.3,1.2,0.8,0.9]$

For example, CSC column $1$ occupies positions $[0,2)$, so destination $1$ receives messages from sources $0$ and $2$.

The loaded example omits explicit GCN self-loops to keep the sparse-format conversion easy to inspect. A GCN kernel would add an implicit self contribution for each node while leaving these arrays unchanged.

### 8.4 Sparse versus dense cost

For element widths $b_v$ bytes, a dense adjacency requires approximately:

$$
N^2 b_v
$$

bytes and exposes $N^2$ potential interactions. A CSC graph with offset width $b_o$, index width $b_i$, and optional scalar-weight width $b_w$ requires approximately:

$$
(N+1)b_o + Mb_i + Mb_w,
$$

where the final term is omitted for an unweighted graph.

For example, a `float32` dense adjacency with one million nodes requires about four terabytes before features or workspaces are counted. Sparse storage remains proportional to the actual number of entries. A dense adjacency may still be useful as a small-graph baseline because it offers regular memory access, but it is not a viable large-graph representation.

A fair sparse/dense study records both storage and executed work. Comparing only elapsed time can be misleading if the dense version performs $N^2$ operations while the sparse version performs work proportional to $M$.

## 9. Vertex-centric and edge-centric CPU work

Storage format and work mapping are separate decisions.

### 9.1 Vertex-centric OpenMP

The parallel iteration space contains $N$ destination nodes. A worker:

1. receives one or more destinations;
2. traverses each destination's complete CSC column;
3. accumulates its incoming messages; and
4. exclusively writes that destination's output row.

This normally avoids atomic output updates. Its main weakness is load imbalance when a few nodes have very high degree.

### 9.2 Edge-centric OpenMP

The parallel iteration space contains $M$ flattened adjacency entries. OpenMP creates a fixed team of CPU workers; it does not create one CPU thread per edge.

With static scheduling and $P$ workers, each worker processes approximately $M/P$ edge iterations. The ranges are not constrained to complete CSC columns, so the incoming edges of one high-degree destination may be split across workers.

Multiple workers can then contribute to the same output row. Correct accumulation requires:

- atomic additions;
- locks;
- thread-private partial results followed by reduction; or
- another explicitly correct segmented-reduction method.

If partitions are forced to end only at CSC column boundaries and each output row has one owner, the method becomes an edge-balanced vertex partition rather than a genuinely edge-centric implementation.

Dynamic scheduling can assign new edge chunks to workers as they finish, but very small chunks increase scheduling overhead. Comparing the two mappings reveals the trade-off among load balance, locality, synchronization, and reduction cost.

## 10. CUDA work mappings

Possible CUDA mappings include:

- one thread or warp per destination node;
- one thread per edge;
- two-dimensional mappings over nodes and feature dimensions; and
- segmented or cooperative processing of high-degree nodes.

A thread-per-edge CUDA kernel exposes abundant parallelism but multiple edges may target the same output element, commonly requiring atomic addition or a staged reduction.

### 10.1 Shared memory

Shared memory is useful only when data has enough reuse within a block to repay loading and synchronization costs. Possible reusable values include feature tiles, transformation weights, or partial reductions. Loading irregular neighbor features into shared memory without reuse merely adds a copy and a barrier.

A shared-memory comparison should hold the mathematical workload and global-memory layout constant where possible. It should report the cached values, bytes reserved per block, additional synchronization, occupancy effect, and measured performance.

## 11. Edge weights and edge features

A scalar edge weight $w_{u,v}$ changes the strength of a message and can participate directly in GCN normalization. The GCN contract uses finite, strictly positive stored weights; an omittedself loop edge is replaced  ith an edge with wweight `1`,.

An edge feature:

$$
e_{u,v}\in\mathbb{R}^{F_e}
$$

is a vector that may be consumed by an edge-aware message function or attention mechanism. If stored densely in edge order, edge features form:

$$
E_{feat}\in\mathbb{R}^{M\times F_e}.
$$

The edge position in **col_ind** or **row_ind** serves as the index into the corresponding scalar weight or feature row.

Scalar edge weights are compatible with both required architectures: they participate in GCN normalization and in the GraphSAGE weighted neighbor mean. Dense edge-feature processing belongs toothersegnn layerse.

## 12. External-framework comparison

The required comparison with an established GNN framework is a second implementation of the same workloads, not a replacement for the native engine. It consumes equivalent topology, features, model parameters, and layer configuration and produces results that can be checked against the matching native sequential baseline.

Framework layer names alone do not prove equivalence. The adapter must explicitly map edge orientation, GCN self-loop and normalization behavior, GraphSAGE's non-self mean, parameter layout, bias, activation, dtype, and output node order. A framework default that changes one of these rules creates a different workload.

Performance records also need comparable boundaries. They should document preprocessing and caching, warm-up, synchronization, repetitions, execution mode, software versions, device, dtype, throughput definition, and peak-memory measurement. Unavoidable differences between framework and native measurement boundaries should remain visible in the report.

## 13. Further reading


- [PyTorch Geometric GCNConv documentation](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.GCNConv.html)
- [PyTorch Geometric SAGEConv documentation](https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.SAGEConv.html)
- [PyTorch Geometric message-passing guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html)
- [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
