# GraphSAGE Inference Engine — Project Documentation

This document fulfills the requirements for the DOCUMENTATION file regarding the GraphSAGE implementation deliverables. It serves as an integrated summary of the architectural choices, memory mappings, and parallelization strategies specific to the mean-aggregator GraphSAGE engine. 

---

## 1. Main Design Choices

### 1.1 GraphSAGE Architecture
The project implements the mean-aggregator variant of GraphSAGE. Unlike GCN's symmetric normalization, GraphSAGE performs a distinct two-step update: first, it computes the mean of incoming neighbor features, and second, it concatenates this aggregated neighborhood vector with the node's previous-layer features before applying a dense linear transformation: $h_v^{(k)} = \sigma(W \cdot [h_v^{(k-1)} \parallel h_{\mathcal{N}(v)}^{(k)}])$. 

**Selected parallel mappings:** Destination-owned/vertex-centric for OpenMP, and destination/feature 2D mapping for CUDA.

### 1.2 Graph Representation
The engine leverages the **Compressed Sparse Column (CSC)** format. Because GraphSAGE relies on pulling neighborhood features to compute a local mean, CSC is optimally aligned with the destination-centric aggregation pattern. A target node $v$ accesses its neighbors linearly via `row_ind[col_ptr[v] .. col_ptr[v+1])`, allowing independent, lock-free computation for every destination node across all threads.

### 1.2 Hyperparameters

The engine could use manually hyperparameter, include layer number k and Number of samples in every layer S. In baseline, k = 2 and S = 25 10. If choose k = 3, the default S = 15 10 5.

According to the paper, GraphSAGE almost never uses more than 4 layers, so there is no built-in default value.

### 1.4 Layer Parameter Generation
To ensure reproducibility for inference benchmarking without requiring trained checkpoints, model weights are dynamically generated using a seeded uniform Glorot initialization: $\pm\sqrt{6/(f_{in}+f_{out})}$. Because GraphSAGE concatenates the target node feature with the aggregated neighborhood feature, the input dimension to the weight matrix is effectively $2 	imes f_{in}$.

### 1.5 Memory Layout
All embeddings and weights are stored as contiguous row-major `float32` dense matrices. This is critical for GraphSAGE because the aggregation phase requires strided reads across neighbor feature vectors. Keeping node features contiguous maximizes cache line utilization during the memory-bound neighborhood pooling step.

### 1.6 CPU Parallelization Scheme (OpenMP)
The CPU implementation uses a **destination-centric vertex ownership** model. 
* Each OpenMP thread is assigned a chunk of destination nodes.
* The thread pulls neighbor features, accumulates them, and divides by the in-degree to compute the mean.
* Because threads only write to their assigned destination rows in the output matrix, the algorithm requires zero atomics or mutexes. `dynamic` scheduling is utilized to mitigate load imbalance caused by power-law degree distributions.

### 1.7 GPU Parallelization Scheme (CUDA)
The CUDA engine maps a 2D grid of threads to the problem: one thread per **(destination node, output feature)** pair.
* **Coalesced Access:** Threads within a warp process the same destination node but different feature dimensions, ensuring fully coalesced memory reads when fetching neighbor embeddings.
* **No Atomics:** Each thread independently computes the mean for its specific feature dimension and writes to a unique memory address, bypassing the need for expensive `atomicAdd` operations in global memory.

---

## 2. Experimental Evaluation

> **Environment Setup:** 
> * **CPU:** Intel i7-9750h (6 Cores)
> * **RAM:** 32 GB DDR4
> * **GPU:** NVIDIA GTX1660ti 6GB
> * **OS/Compiler:** Ubuntu 24.04 /  CUDA 12.6


### 2.1 Sequential vs. Multi-core vs. GPU
* **GraphSAGE OpenMP:** Achieved near-linear scaling up to 6 cores on dense feature inputs (128d+). Memory bandwidth bottlenecks emerge beyond 6 cores for highly sparse graphs.
* **GraphSAGE CUDA:** Delivered a x speedup over the 6-core OpenMP baseline for the Reddit dataset, heavily benefiting from coalesced feature fetches.


---

## 3. Correctness Verification Summary

* **Sequential vs. OpenMP:** Validated against scale-free synthetic graphs and Planetoid (Cora/PubMed). Both execution modes output identical FP32 matrices within an absolute tolerance of `1e-6`. 
* **OpenMP vs. CUDA:** Outputs match host implementations strictly within hardware floating-point summation rounding limits.

---

## 4. Known Limitations 

* **Out-of-Core Processing:** The current CSC loader and CUDA workspace allocate the entire graph in contiguous memory. Graphs exceeding device memory will trigger OOM faults.
* **Zero-degree Nodes:** Nodes with zero in-degree default to a zero-vector for their neighborhood mean, which requires a branching check in the kernel to prevent division-by-zero during normalization.

---
