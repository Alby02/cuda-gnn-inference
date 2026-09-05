# GNN Inference Engine on CPUs and GPUs — Project Documentation

This document fullfill the requirements for the DOCUMENTATION file mandatory in the deliverables. It is therefore an integrated summary of the already present markdown in this folder, for better meeting the requirements of the documentation. For further details refers to [semantics.md](semantics.md) for binding selections and mathematical conventions, [architecture.md](architecture.md) for software responsibilities, [requirements.md](requirements.md) for project requirements and [knowledge.md](knowledge.md) for explanation regarding GNN.

---

## 1. Main Design Choices
### 1.1 GNN architectures
the project implemented two GNN architectures: GCN and mean-aggregator GraphSAGE. *(`project.md` allows "one or two"; two were selected to ensure even workload distribution across three team members).*
*   **Selected parallel mappings:** Destination-owned/vertex-centric for OpenMP, destination/feature 2D mapping for CUDA (see §2 below for the rationale).

### 1.1 Graph representation and format 
*(from `doc/architecture.md §6.1` and `doc/knowledge.md §8.2`)*

The engine stores graph topology in **Compressed Sparse Column (CSC)** format, representing incoming adjacency: for destination node $v$, its incoming sources occupy `row_ind[col_ptr[v] .. col_ptr[v+1])`. This choice is deliberate rather than the more commonly seen CSR-for-everything convention, because the core GNN operation (message aggregation) is destination-centric: every output row is produced by pulling messages from a node's incoming neighbors. 

CSC exposes exactly that access pattern contiguously, letting one worker (CPU thread or one CUDA thread/block) own and write one destination's output row without needing to coordinate with any other worker.

The project supports both directed and undirected graphs via a `GraphOrientation` enum rather than separate graph classes since the only difference between the two is a validation rule (every non-self undirected edge must have a reciprocal entry with equal weight) — the traversal and layer math are identical regardless of orientation.



### 1.2 Edge weights 

This is explicitly NOT required by the original assignment PDF — the official Project Q1 text only mentions optional dense edge feature vectors for GraphSAGE/attention, never scalar edge weights. Supporting weighted graphs was a deliberate team extension, and should be documented as such so a grader comparing the report against the PDF understands it's an addition.

**Rationale:**
*   The classical spectral GCN normalization $D^{-1/2}\widehat{A}^T D^{-1/2}$ (Kipf & Welling, ICLR 2017) generalizes naturally to weighted adjacency — an unweighted graph is simply the special case where every stored weight equals 1,(`doc/semantics.md §5`).
*   Supporting weights costs extra complexity (one optional array aligned with `row_ind`) and makes the loader directly compatible with weighted public datasets.
*   **Constraint:** Stored weights must be finite and strictly positive (`doc/requirements.md FR-GRAPH-08`); a missing GCN self-loop uses the fixed implicit weight of 1 (`doc/semantics.md §4.3`).

### 1.3 GCN layer parameter generation (model weights) 

Since this project is inference-only, model weights are generated via seeded random initialization:

*   **Reproducibility:** `doc/requirements.md FR-IO-03` explicitly allows parameters to be either loadable or reproducibly generated. `generate_gcn_params.py` (`T-DATA-04`) takes a base seed and derives one distinct-but-reproducible seed per layer, so the exact same parameters can be regenerated in independent runs.
*   **Why weight values don't affect deliverables:** Required comparisons are throughput, memory footprint, and cross-executor numerical equivalence. All of these depend on tensor shapes and the graph's sparsity pattern, not semantic meaning. A randomly initialized weight matrix produces identical FLOP counts and memory traffic.
*   **Out of scope:** Applying a final classifier and reporting node-classification accuracy (`OPT-04`) is explicitly not attempted in the current scope.
*   **Initialization scheme:** A Glorot/Xavier-style uniform distribution, $\pm\sqrt{6/(f_{in}+f_{out})}$, is used purely to keep magnitudes numerically well-scaled (avoiding overflow/underflow).

### 1.4 Memory layout
*(From `doc/architecture.md §6.2`)*

Node features, weights, biases, and intermediate/output matrices use `float32`, stored in contiguous row-major dense format: `element(row, col) = data[row * num_columns + col]`. This keeps a node's full feature vector contiguous in memory, which matters for cache locality during both the dense linear-transform step and the per-node aggregation loop.

### 1.5 Software architecture overview 
*(See `doc/architecture.md §4.2` for the full responsibility table)*

The engine separates four responsibilities: 
1. **Layer algorithms** (`forward_layer`, owns operation order per GNN type).
2. **Executors** (typed hardware operations for Sequential/OpenMP/CUDA).
3. **Strategies** (work partitioning within an executor).
4. **Workspaces** (reusable host/device buffer ownership).

### 1.6 CPU parallelization scheme

#### 1.6.1 Shared strategy rationale 
The selected multi-threaded CPU mapping is OpenMP, **destination/vertex ownership**: the parallel iteration space is the set of destination nodes. A worker thread owns one or more complete destinations and pulls all incoming messages for each from its CSC column. This avoids concurrent writes to any output row entirely — no atomics, locks, or reductions are needed for aggregation.

#### 1.6.2 GCN OpenMP implementation
The GCN OpenMP engine (`OpenMPGCNEngine`, `T-OMPV-01`/`T-OMPV-04`) applies this destination-owned mapping to both stages of a GCN layer: the dense linear transform $Z = H \cdot W$ and the normalized aggregation step itself. Thread count, OpenMP schedule kind (static/dynamic/guided), and chunk size are run-time configurable.

#### 1.6.3 GraphSAGE OpenMP implementation `[TODO — owner: s296248]`


### 1.7 GPU parallelization scheme

#### 1.7.1 Shared strategy rationale 
The selected CUDA mapping is a **two-dimensional destination/feature mapping**: one thread is responsible for exactly one (destination node, output feature) pair, walking that destination's CSC column sequentially. This gives every output element a single logical owner, avoiding aggregation atomics on the GPU.

#### 1.7.2 GCN CUDA implementation
The GCN aggregation kernel and its run-time-configurable launch geometry are implemented. 
*   **Known limitation:** This kernel is not yet integrated into a complete, testable CUDA engine, because it depends on shared CUDA infrastructure not yet available. No correctness or performance comparison for GCN exists yet.

#### 1.7.3 GraphSAGE CUDA implementation `[TODO — owner: s296248]`
To be filled.

### 1.8 Strategies for handling highly skewed degree distributions

#### 1.8.1 Rationale 
Real graphs (and the scale-free synthetic family) have highly non-uniform in-degree distributions. Under static OpenMP scheduling, thread work is partitioned by node index ranges of equal count, not equal edge count, causing severe load imbalance. Using `dynamic` or `guided` scheduling lets idle threads pull new chunks of destination nodes as they finish, trading scheduling overhead for better balance.


---

## 2. Experimental Evaluation

> **Environment Setup:** 
> *   **CPU:** `[TODO]`
> *   **Cores:** `[TODO]`
> *   **RAM:** `[TODO]`
> *   **GPU:** `[TODO]`
> *   **OS/Compiler:** `[TODO]`
>
> 


### 2.1 Sequential vs. multi-core vs. GPU
*   **2.2.1 GCN:** `[TODO]` 
*   **2.2.2 GraphSAGE:** `[TODO — owner: s296248]`

### 2.2 Vertex-centric vs. edge-centric `[TODO — joint]`
Only the vertex-centric mapping is implemented on the GCN side; edge-centric is optional (`F-OMP-ADDITIONAL`). Decide as a team whether to attempt this comparison at all; if not, state explicitly that it was scoped out.

### 2.3 With/without shared memory (CUDA) `[TODO — owner: s296248]`


### 2.4 Different graph and feature sizes
*   **Graph-size scalability:** `[TODO]` 
*   **Feature-dimension sweep:** `[TODO]` 
*   **Model-depth sweep:** `[TODO]` 

### 2.5 Synthetic vs. public graph benchmarks
*   **Synthetic (scale-free):** `[TODO]` 
*   **Public dataset (OGB or Planetoid):** `[TODO s362415]`

---

## 3. Correctness Verification Summary 

*   **GCN sequential vs. OpenMP:** Verified across non-uniform-degree graphs, mixed explicit/implicit self-loops, an isolated zero-in-degree node, and three OpenMP scheduling policies — all agree within `atol=1e-4`, `rtol=1e-4`. 
*   **GCN vs. CUDA:** 
*   **GraphSAGE:** `[TODO — owner: s296248]`


---

## 4. Known Limitations 


---

## 5. References

*   Alvaro Sanchez-Gonzalez, Nicolas Heess, Jost Tobias Springenberg, Josh Merel, Martin Riedmiller, Raia Hadsell, Peter Battaglia, *Graph Networks as Learnable Physics Engines for Inference and Control*, Proceedings
*   Jie Zhou, Ganqu Cui, Shengding Hu, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Lifeng Wang, Changcheng Li, Maosong Sun (2020), *Graph neural networks: A review of methods and applications*, AI Open, Volume 1,  ages 57-81,
*   Open Graph Benchmark: [https://ogb.stanford.edu](https://ogb.stanford.edu)
*   Project internal specs: `doc/architecture.md`, `doc/semantics.md`, `doc/knowledge.md`, `doc/requirements.md`.
