# Project Semantics
## Selected GNN and Execution Contracts

## 1. Purpose and authority

This document defines the mathematical and data semantics that every sequential CPU, OpenMP, CUDA, and external-framework execution shall follow for the required GCN and mean-aggregator GraphSAGE layers.

[requirements.md](requirements.md) defines the minimum capabilities and implementation counts without selecting concrete GNN families or work mappings. This document records those project selections and defines what correct GCN and GraphSAGE inference results mean. Architecture and implementation details may vary as long as they preserve these contracts within the numerical tolerance defined in Section 9.

### 1.1 Selected GNN architectures

The two selected required architectures are:

1. a normalized Graph Convolutional Network (GCN), defined in Section 6; and
2. mean-aggregator GraphSAGE, defined in Section 7.

This pair is selected because it provides two distinct message-passing rules while allowing meaningful reuse of graph storage and low-level traversal operations:

- GCN uses degree-normalized weighted aggregation with one effective self message;
- GraphSAGE uses a non-self neighbor mean and a separate self branch;
- both support full-batch inference over the same sparse graph and dense node-feature layout; and
- both are available in established external frameworks, allowing equivalent correctness and performance comparisons.

Changing either selected architecture requires updating this document's equations, parameter contracts, fixtures, backend capability checks, workload bundles, and external-framework mapping. The minimum requirement remains at least two GNN architectures.

### 1.2 Selected implementation mappings

The minimum requirement is one or more complete multi-threaded CPU implementations and one or more complete CUDA implementations. The selected project profile uses one work-mapping family for each parallel backend:

| Implementation family | Selected mapping | Rationale |
| --- | --- | --- |
| Sequential CPU | Destination-oriented pull traversal | It is direct, deterministic, and provides a readable numerical baseline for both selected GNN types. |
| Multi-threaded CPU | OpenMP destination/vertex ownership | A worker owns complete destination rows, naturally traverses incoming CSC columns, avoids concurrent aggregation writes, and supports both selected GNN types with the same work-ownership rule. |
| CUDA GPU | Two-dimensional destination/feature mapping with destination-owned aggregation | It exposes parallelism across nodes and feature dimensions, avoids aggregation atomics, fits incoming CSC traversal, and provides one common complete CUDA path for both selected GNN types. |

One selected implementation family may contain multiple kernels or loop operations; “one implementation” refers to one complete end-to-end work-mapping composition, not one kernel for the entire model.

Additional CPU or CUDA mappings are permitted but are not required merely to satisfy the minimum implementation count. An applicable shared-memory experiment may be implemented as a controlled configuration of the selected CUDA mapping. If an additional work mapping is added, it shall be defined here before its results are treated as part of the required project profile.

The selected feature representation remains dense `float32`; no reduced-precision, quantized, or separately compressed feature representation is part of the project profile.

### 1.3 Required strategy rationale

The technical documentation shall state:

- how many complete multi-threaded CPU and CUDA implementations were provided;
- why that number was chosen rather than only the minimum or a larger strategy set;
- why each mapping fits the sparse representation, node-feature layout, selected GNN operations, and target hardware;
- how work is owned or partitioned and where synchronization or reduction is required;
- which alternative mappings were considered; and
- whether measurements supported or contradicted the original rationale.

This explanation is required whether the project implements exactly one or more than one mapping for a parallel backend.

## 2. Notation and tensor shapes

| Symbol | Meaning | Shape |
| --- | --- | --- |
| $N = \lvert V\rvert$ | Number of nodes | Scalar |
| $M$ | Number of adjacency entries stored in the loaded graph | Scalar |
| $\widehat{M}$ | Number of effective GCN messages after missing self-loops are included | Scalar |
| $F_l$ | Feature dimension at layer $l$ | Scalar |
| $H^{(l)}$ | Node feature/embedding matrix at layer $l$ | $N \times F_l$ |
| $W^{(l)}$ | GCN layer weight matrix | $F_l \times F_{l+1}$ |
| $b^{(l)}$ | Optional layer bias | $F_{l+1}$ |
| $W_{self}^{(l)}$ | GraphSAGE self-branch weight matrix | $F_l \times F_{l+1}$ |
| $W_{neigh}^{(l)}$ | GraphSAGE neighbor-branch weight matrix | $F_l \times F_{l+1}$ |
| $A$ | Weighted adjacency matrix using the convention in Section 3 | $N \times N$ |
| $\widehat{A}$ | Effective GCN adjacency after missing self-loops are included | $N \times N$ |

Node features are row vectors. Row $v$ of $H^{(l)}$ is the feature vector of node $v$.

The selected project profile uses:

- node features, weights, biases, edge weights, intermediate values, and outputs use IEEE 754 binary32 (`float32`);
- node and edge indices use an unsigned integer type large enough for the loaded graph; and
- dense matrices use contiguous row-major storage.

## 3. Graph and adjacency convention

### 3.1 Edge direction

The project defines:

$$
A_{u,v} = w_{u,v}
$$

when an edge exists from source node $u$ to destination node $v$. Otherwise, $A_{u,v}=0$.

Therefore:

- matrix row $u$ describes edges leaving source node $u$;
- matrix column $v$ describes edges entering destination node $v$;
- an edge pair is written as $(u,v)$ or $u \rightarrow v$; and
- message passing follows the source-to-target direction.

This convention is binding for project file formats, converters, loaders, tests, and framework adapters. Other libraries may use a transposed sparse-matrix convention, so conversion code shall not infer orientation from the words CSR or CSC alone.

### 3.2 CSR and CSC meaning under this convention

With the adjacency convention above:

- CSR row $u$, represented by `row_ptr[u] ... row_ptr[u+1]`, contains the destination IDs of the outgoing edges from $u$.
- CSC column $v$, represented by `col_ptr[v] ... col_ptr[v+1]`, contains the source IDs of the incoming edges to $v$.

The core aggregation is destination-oriented and uses incoming messages. CSC is therefore the natural pull representation for the architecture.

### 3.3 Undirected graphs

When an undirected graph is supported, each non-self undirected edge $\{u,v\}$ shall be represented by two stored adjacency entries:

$$
(u,v) \quad \text{and} \quad (v,u).
$$

A self-loop $(v,v)$, when present in the loaded graph, is stored once. Storage metrics shall use $M$, the number of stored adjacency entries, rather than the number of logical undirected edges.

## 4. Loaded topology, duplicate edges, and self-loops

### 4.1 Loaded topology

The graph file defines whether each edge, including each self-loop, is explicitly present. Loading shall preserve that topology and shall not add or remove self-loops.

The inference engine shall not permanently modify the loaded CSR/CSC arrays. A model may nevertheless require an implicit message for a missing self-loop, as defined in Section 4.3.

Within each canonical CSR row or CSC column, adjacent node indices shall be sorted. Sorting belongs to an explicit conversion step; loading shall not silently reorder a file that claims to be canonical.

### 4.2 Duplicate edges

The mathematical adjacency matrix has one value for each ordered pair $(u,v)$. Accordingly, the project graph format shall contain at most one stored entry for any ordered pair.

An edge-list or sparse file can physically repeat an ordered pair even though the mathematical model does not permit distinct duplicate edges. Dataset converters shall produce unique ordered pairs, and the loader shall reject duplicate entries rather than silently summing or processing them separately.

### 4.3 Effective GCN self-loop policy

GCN inference requires one effective self message per node:

- If the loaded graph contains $(v,v)$, the kernel shall use that explicit self-loop and its loaded weight.
- If the loaded graph does not contain $(v,v)$, the kernel shall add one implicit self contribution with weight $1$.
- An explicit and an implicit self-loop shall never both be applied to the same node.
- The implicit contribution need not be inserted into the stored CSR/CSC arrays.
- Every executor and strategy shall use the same effective self-loop rule.

Let the effective GCN edge set be:

$$
\widehat{E}
=
E \cup \{(v,v) \mid (v,v)\notin E\}.
$$

Its effective weights are:

$$
\widehat{w}_{u,v}
=
\begin{cases}
w_{u,v}, & (u,v)\in E, \\
1, & u=v \text{ and } (v,v)\notin E.
\end{cases}
$$

Thus, $\widehat{M}=|\widehat{E}|$. The GCN operation processes $M$ explicit entries plus one implicit self message for each node whose self-loop is absent.

## 5. Edge weights and degrees

Each stored adjacency entry has a finite, strictly positive scalar weight $w_{u,v}$. If a dataset does not provide scalar edge weights, its entries have weight $1$. A zero-weight sparse entry carries no message and shall be omitted rather than stored. Missing GCN self-loops use the implicit weight defined in Section 4.3.

For source-to-target aggregation, the weighted incoming degree of node $v$ is:

$$
d_v = \sum_{u:(u,v)\in \widehat{E}} \widehat{w}_{u,v}.
$$

Because the effective GCN topology contains a positive explicit or implicit self-loop for every node, $d_v>0$ for every node.

The normalized effective weight of edge $(u,v)\in\widehat{E}$ is:

$$
\widehat{\alpha}_{u,v}
=
\frac{\widehat{w}_{u,v}}{\sqrt{d_u d_v}}.
$$

For an undirected graph, incoming and outgoing degrees are equal and the equation reduces to the standard symmetric GCN normalization.

The original spectral GCN formulation is most natural on undirected graphs. Several normalization choices are possible for directed graphs; using weighted incoming degree for both endpoints is this project's explicit source-to-target convention, not a claim that it is the only directed-GCN definition.

## 6. GCN layer semantics

For layer $l$, define:

$$
Z^{(l)} = H^{(l)} W^{(l)}.
$$

For each destination node $v$ and output feature $o$, normalized incoming aggregation is:

$$
M^{(l)}_{v,o}
=
\sum_{u:(u,v)\in \widehat{E}}
\widehat{\alpha}_{u,v} Z^{(l)}_{u,o}.
$$

The layer output is:

$$
H^{(l+1)}_{v,o}
=
\sigma_l\left(
M^{(l)}_{v,o} + b^{(l)}_o
\right),
$$

where the bias term is omitted when the layer configuration has no bias.

In matrix form under the project's source-row/destination-column adjacency convention:

$$
H^{(l+1)}
=
\sigma_l\left(
D^{-1/2} \widehat{A}^T D^{-1/2} H^{(l)} W^{(l)}
+ \mathbf{1} b^{(l)T}
\right).
$$

The transpose is necessary because rows of $A$ are sources while rows of $H$ are the node outputs being computed.

### 6.1 Permitted execution orders

Because normalization and aggregation are linear, an implementation may use either:

1. transform then aggregate:
   $$
   D^{-1/2}\widehat{A}^TD^{-1/2}(HW),
   $$
2. aggregate then transform:
   $$
   (D^{-1/2}\widehat{A}^TD^{-1/2}H)W.
   $$

These orders are mathematically equivalent in exact arithmetic. Implementations may select an order according to feature dimensions and hardware behavior, but the selected order shall be recorded for benchmarked variants. Floating-point differences are evaluated using Section 9.

### 6.2 Bias and activation

- Bias is applied after normalized aggregation and linear transformation.
- The activation is applied after bias.
- Supported core activation modes are `NONE` and `RELU`.
- Activation is configured per layer.
- The conventional default is `RELU` for hidden layers and `NONE` for the final embedding layer.
- `SIGMOID` may be supported as an additional option but is not required by the core GCN contract.

## 7. Mean-aggregator GraphSAGE semantics

GraphSAGE uses a neighbor branch and a separate self branch. Define the non-self incoming neighborhood:

$$
\mathcal{N}_{in}^{-}(v)
=
\{u \mid (u,v)\in E,\ u\neq v\}.
$$

An explicitly stored self-loop is excluded from this neighbor set because the self representation is already handled by the self branch. GraphSAGE does not add an implicit graph self-loop.

Let the total non-self incoming weight be:

$$
s_v
=
\sum_{u\in\mathcal{N}_{in}^{-}(v)} w_{u,v}.
$$

The weighted mean neighbor representation is:

$$
m_v^{(l)}
=
\begin{cases}
\displaystyle
\frac{1}{s_v}
\sum_{u\in\mathcal{N}_{in}^{-}(v)}
w_{u,v}h_u^{(l)}, & s_v>0, \\
\mathbf{0}, & s_v=0.
\end{cases}
$$

Because stored weights are strictly positive, $s_v=0$ exactly when no non-self incoming neighbor exists. For an unweighted graph, all stored weights are $1$ and this becomes the ordinary arithmetic mean.

The GraphSAGE layer output is:

$$
h_v^{(l+1)}
=
\sigma_l\left(
h_v^{(l)}W_{self}^{(l)}
+
m_v^{(l)}W_{neigh}^{(l)}
+ b^{(l)}
\right),
$$

where the bias term is omitted when absent. This is equivalent to concatenating self and neighbor representations and multiplying by a vertically concatenated parameter matrix, but the implementation need not materialize the concatenation.

The neighbor mean and its linear transformation may be associated in either order because $W_{neigh}^{(l)}$ is shared across nodes. The self branch remains separate. Bias is applied after both branches are combined, followed by the configured activation. Required activation modes are `NONE` and `RELU`.

## 8. Model composition and mixed layers

The input to the model is:

$$
H^{(0)} = X.
$$

For a model with $L$ layers, layer $l$ consumes $H^{(l)}$ and produces $H^{(l+1)}$. Every layer preserves the node-row association: output row $v$ remains the representation of node $v$.

For a GCN layer, the parameter dimensions satisfy:

$$
W^{(l)} \in \mathbb{R}^{F_l \times F_{l+1}}.
$$

For a GraphSAGE layer, both branch matrices satisfy:

$$
W_{self}^{(l)}, W_{neigh}^{(l)}
\in \mathbb{R}^{F_l \times F_{l+1}}.
$$

For every adjacent pair of supported layers, the first layer's output dimension shall equal the next layer's input dimension. The final result is a dense node embedding/feature matrix:

$$
H^{(L)} \in \mathbb{R}^{N \times F_L}.
$$

The required homogeneous GCN and homogeneous GraphSAGE models may each contain one or more layers. A model may also contain a mixed sequence of supported layer types, but mixed execution is optional. Every additional layer type requires its own normative semantic contract defining:

- required graph, node, and edge data;
- parameter and tensor shapes;
- aggregation and update equations;
- self-node, self-loop, bias, and activation policies; and
- numerical-equivalence rules when they differ from Section 9.

The selected executor and strategy shall support every layer in the model. Unsupported combinations and incompatible consecutive dimensions shall fail validation before inference begins. The mechanism used to select a layer algorithm shall not alter its mathematical result.

Dependent layers execute in model order because layer $l+1$ consumes the complete output of layer $l$. The selected profile parallelizes work within a graph-wide layer; it does not require or select inter-layer concurrency.

Intermediate matrices may be reused or stored in ping-pong buffers, but execution shall not overwrite values still required by the active layer.

## 9. Numerical equivalence

The sequential CPU implementation is the native correctness baseline.

For a baseline value $r$ and compared value $x$, the default element-wise acceptance condition is:

$$
|x-r| \leq \text{atol} + \text{rtol}|r|,
$$

with:

- `atol = 1e-4`;
- `rtol = 1e-4`.

Tests may define stricter tolerances. A looser tolerance requires a documented numerical reason and shall not be selected merely to hide an implementation error.

NaN and infinity fail verification unless the test explicitly expects the same non-finite value. Shape mismatches always fail verification.

## 10. External-framework semantic mapping

The required external-framework runner shall reproduce the same logical graph, node features, parameter values, layer order, and output interpretation as the native engine. Framework defaults shall not silently determine semantics.

For GCN, the mapping explicitly records:

- source-to-target edge direction;
- explicit and missing self-loop behavior;
- weighted incoming-degree normalization;
- weight-matrix orientation;
- bias and activation; and
- numerical precision.

For GraphSAGE, the mapping explicitly records:

- exclusion of stored self-loops from neighbor aggregation;
- weighted mean over non-self incoming neighbors;
- the zero vector for an empty neighbor set;
- separate self and neighbor transformations, including any parameter concatenation/transposition required by the framework;
- bias and activation; and
- numerical precision.

Framework preprocessing or cached normalization may be used only when it produces these same semantics. Framework output is checked against the corresponding native sequential baseline; it does not replace that baseline.

## 11. Worked examples

### 11.1 GCN directed example

Consider three nodes with non-self edges:

$$
0 \rightarrow 1, \qquad 2 \rightarrow 1,
$$

all with weight $1$. Assume that the loaded graph contains no explicit self-loops. During GCN inference, every executor/strategy therefore includes one implicit weight-$1$ self message per node.

The weighted incoming degrees are:

$$
d_0=1, \qquad d_1=3, \qquad d_2=1.
$$

The normalized incoming aggregate for node $1$ is:

$$
M_1
=
\frac{1}{\sqrt{3}}Z_0
+ \frac{1}{3}Z_1
+ \frac{1}{\sqrt{3}}Z_2.
$$

Nodes $0$ and $2$ receive only their self messages and therefore have normalized self-loop coefficient $1$. This example shall be usable as a small semantic test for the sequential executor.

### 11.2 GraphSAGE directed example

Use the same non-self edges:

$$
0 \rightarrow 1, \qquad 2 \rightarrow 1,
$$

with scalar node features:

$$
h_0=2, \qquad h_1=3, \qquad h_2=6.
$$

For node $1$, the mean non-self incoming representation is:

$$
m_1=\frac{h_0+h_2}{2}=4.
$$

Nodes $0$ and $2$ have empty non-self incoming neighborhoods, so $m_0=m_2=0$.

For one output feature, let:

$$
W_{self}=2, \qquad W_{neigh}=1, \qquad b=-1,
$$

with activation `NONE`. The outputs are:

$$
h_0'=3, \qquad h_1'=9, \qquad h_2'=11.
$$

This fixture checks the mean, empty-neighbor rule, separate self branch, bias, and absence of implicit GraphSAGE self-loop insertion.
