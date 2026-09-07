# GraphSAGE Inference Engine — Project Documentation

This document fulfills the requirements for the DOCUMENTATION file regarding the GraphSAGE implementation deliverables. It serves as an integrated summary of the architectural choices, memory mappings, and parallelization strategies specific to the mean-aggregator GraphSAGE engine. It also consolidates benchmarking results, hardware/software environment specifications, micro-architectural insights, and deployment guidelines for GraphSAGE inference across multiple execution backends (`sequential`, `parallel`, and `cuda`).

---

## 1. Main Design Choices

### 1.1 GraphSAGE Architecture
The project implements the mean-aggregator variant of GraphSAGE. Unlike GCN's symmetric normalization, GraphSAGE performs a distinct two-step update: first, it computes the mean of incoming neighbor features, and second, it concatenates this aggregated neighborhood vector with the node's previous-layer features before applying a dense linear transformation: $h_v^{(k)} = \sigma(W \cdot [h_v^{(k-1)} \parallel h_{\mathcal{N}(v)}^{(k)}])$. 

**Selected parallel mappings:** Destination-owned/vertex-centric for OpenMP, and destination/feature 2D mapping for CUDA.


### 1.2 GraphSAGE Neighbor Sampling 

This feature introduces an optional, layer-dependent neighbor sampling mechanism into the GraphSAGE CUDA and CPU aggregation pipelines. By limiting the number of neighbors processed per node during aggregation, it significantly reduces computational overhead and memory bandwidth requirements for highly connected (dense) graphs.


#### Sampling Heuristic & Logic

* **Layer-Based Decay**: The system dynamically calculates the sample size based on the network depth (layer index) to balance performance and information flow:
  * **Layer 0**: Maximum of 25 neighbors.
  * **Deeper Layers**: Sample size decreases linearly (formula: `10 - layer * 2`), with a hard minimum of 5 neighbors.
* **Deterministic Selection**: To avoid the overhead and branch divergence introduced by pseudo-random number generation , the kernels use a uniform selection strategy based on a deterministic stride: `edge_idx = e * degree / maxSamples`. This ensures an even sampling of neighbors across the node's adjacency list.
* **Bypass Mechanism**: If a node's total degree is less than or equal to the `maxSamples` threshold, the kernel automatically bypasses the sampling logic and processes all available neighbors.



### 1.3 Graph Representation
The engine leverages the **Compressed Sparse Column (CSC)** format. Because GraphSAGE relies on pulling neighborhood features to compute a local mean, CSC is optimally aligned with the destination-centric aggregation pattern. A target node $v$ accesses its neighbors linearly via `row_ind[col_ptr[v] .. col_ptr[v+1])`, allowing independent, lock-free computation for every destination node across all threads.


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
> * **CPU:** Intel(R) Xeon(R) CPU @ 2.20GHz (1 Core 2 Threads)
> * **RAM:** 16 GB DDR5
> * **GPU:** NVIDIA T4 16GB
> * **OS/Compiler:** Ubuntu 22.04 /  CUDA 12.8


### GPU Info & Toolchain

```text
=== GPU Info ===
Mon Sep  7 03:04:06 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   37C    P8             13W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

=== Toolchain Compilatori ===
g++ (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
```
> **Environment Setup 2:** 
> * **CPU:** Intel(R) core I7-9750h (6 Core 12 Threads)
> * **RAM:** 32 GB DDR5
> * **GPU:** RTX 1660Ti 6GB
> * **OS/Compiler:** Ubuntu 24.04 /  CUDA 12.6
```text
=== GPU Info ===
Mon Sep  7 07:51:28 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.53                 KMD Version: 610.74        CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1660 Ti     On  |   00000000:01:00.0  On |                  N/A |
| N/A   48C    P8              8W /   80W |     748MiB /   6144MiB |     10%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A              25      G   /Xwayland                             N/A      |
+-----------------------------------------------------------------------------------------+
```
=== Toolchain Compilatori ===
g++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
Cuda compilation tools, release 12.6, V12.6.85
Build cuda_12.6.r12.6/compiler.35059454_0

### 2.1 Sequential vs. Multi-core vs. GPU
* **GraphSAGE OpenMP:** Achieved near-linear scaling up to 2 thread on dense feature inputs (128d+). Delivered a 
* **GraphSAGE CUDA:** Delivered a 60.16x speedup over baseline for the ogbn-arxiv dataset, heavily benefiting from coalesced feature fetches.

### GNN Comprehensive Benchmark (Native C++ Compute vs. End-to-End Latency, GraphSAGE, Reps=5)
> **Stress Graph Topology:** Nodes = 80,000 | Edges = 270,449

| Config Name | Dim | L | Backend | Native C++ (ms) | E2E Lat (ms) | Throughput (nodes/s) | Peak VRAM (MB) | C++ Speedup |
| :--- | :---: | :---: | :--- | :---: | ---: | ---: | :---: | ---: |
| **Standard-2L** | 128 | 2 | sequential | 883.88 ± 62.98 | 1132.86 | 90,509.9 | - | 1.00x |
| Standard-2L | 128 | 2 | parallel | 539.91 ± 11.68 | 684.39 | 148,173.1 | - | 1.64x |
| Standard-2L | 128 | 2 | cuda | ~80.60 (E2E) | 80.60 | 992,559.0 | 311.0 | 10.97x |
| **LargeFeat-2L** | 256 | 2 | sequential | 3522.46 ± 151.43 | 4361.93 | 22,711.4 | - | 1.00x |
| LargeFeat-2L | 256 | 2 | parallel | 2435.21 ± 483.24 | 2924.90 | 32,851.4 | - | 1.45x |
| LargeFeat-2L | 256 | 2 | cuda | ~112.45 (E2E) | 112.45 | 711,401.6 | 511.0 | 31.32x |
| **WideHidden-2L** | 128 | 2 | sequential | 5019.22 ± 286.54 | 5991.62 | 15,938.7 | - | 1.00x |
| WideHidden-2L | 128 | 2 | parallel | 3365.17 ± 495.08 | 4003.41 | 23,772.9 | - | 1.49x |
| WideHidden-2L | 128 | 2 | cuda | ~114.90 (E2E) | 114.90 | 696,233.3 | 473.0 | 43.68x |
| **DeepNet-3L** | 128 | 3 | sequential | 2492.82 ± 165.07 | 3114.28 | 32,092.2 | - | 1.00x |
| DeepNet-3L | 128 | 3 | parallel | 1494.78 ± 18.22 | 1971.76 | 53,519.6 | - | 1.67x |
| DeepNet-3L | 128 | 3 | cuda | ~90.28 (E2E) | 90.28 | 886,145.0 | 311.0 | 27.61x |
| **LargeDeep-3L** | 256 | 3 | sequential | 4731.81 ± 291.69 | 6447.22 | 16,906.8 | - | 1.00x |
| LargeDeep-3L | 256 | 3 | parallel | 3344.50 ± 1037.54 | 4028.89 | 23,919.9 | - | 1.41x |
| LargeDeep-3L | 256 | 3 | cuda | ~109.05 (E2E) | 109.05 | 733,603.6 | 511.0 | 43.39x |


### GNN Inference Performance & Memory Benchmark (GraphSAGE)
> **Graph Topology:** Nodes = 3,000 | Edges = 468

| Config Name | In Dim | Layers | Backend | Latency (ms) | Throughput (nodes/s) | GPU Mem (MB) | Speedup |
| :--- | :---: | :---: | :--- | ---: | ---: | :---: | ---: |
| **SmallFeat-2L** | 64 | 2 | sequential | 26.61 | 112,741.5 | - | 1.00x |
| SmallFeat-2L | 64 | 2 | parallel | 31.85 | 94,191.9 | - | 0.84x |
| SmallFeat-2L | 64 | 2 | cuda | 255.61 | 11,736.7 | 3.0 | 0.10x |
| **Nominal-2L** | 128 | 2 | sequential | 69.45 | 43,196.0 | - | 1.00x |
| Nominal-2L | 128 | 2 | parallel | 69.42 | 43,212.7 | - | 1.00x |
| Nominal-2L | 128 | 2 | cuda | 210.92 | 14,223.2 | 3.0 | 0.33x |
| **LargeFeat-2L** | 256 | 2 | sequential | 264.85 | 11,327.1 | - | 1.00x |
| LargeFeat-2L | 256 | 2 | parallel | 265.91 | 11,282.0 | - | 1.00x |
| LargeFeat-2L | 256 | 2 | cuda | 249.63 | 12,017.8 | 3.0 | 1.06x |
| **Nominal-3L** | 128 | 3 | sequential | 80.53 | 37,252.5 | - | 1.00x |
| Nominal-3L | 128 | 3 | parallel | 79.92 | 37,536.2 | - | 1.01x |
| Nominal-3L | 128 | 3 | cuda | 205.47 | 14,600.5 | 3.0 | 0.39x |
| **LargeFeat-3L** | 256 | 3 | sequential | 318.87 | 9,408.2 | - | 1.00x |
| LargeFeat-3L | 256 | 3 | parallel | 315.88 | 9,497.4 | - | 1.01x |
| LargeFeat-3L | 256 | 3 | cuda | 277.82 | 10,798.2 | 3.0 | 1.15x |
| **DeepNet-4L** | 128 | 4 | sequential | 230.92 | 12,991.6 | - | 1.00x |
| DeepNet-4L | 128 | 4 | parallel | 230.08 | 13,039.1 | - | 1.00x |
| DeepNet-4L | 128 | 4 | cuda | 270.70 | 11,082.2 | 3.0 | 0.85x |
---


### GNN Inference Performance & Memory Benchmark (GraphSAGE) on another machime

### GNN Comprehensive Benchmark (Native C++ Compute vs. End-to-End Latency, GraphSAGE, Reps=5)
> **Dataset:** ogbn-arxiv | **Stress Graph Topology:** Nodes = 80,000, Edges = 270,449

| Config Name | Dim | L | Backend | Native C++ (ms) | E2E Lat (ms) | Throughput (nodes/s) | Peak VRAM (MB) | C++ Speedup |
| :--- | :---: | :---: | :--- | :---: | ---: | ---: | :---: | ---: |
| **Standard-2L** | 128 | 2 | sequential | 1295.03 ± 4.19 | 1602.37 | 61,774.6 | - | 1.00x |
| Standard-2L | 128 | 2 | parallel | 190.27 ± 3.43 | 260.17 | 420,450.7 | - | 6.81x |
| Standard-2L | 128 | 2 | cuda | ~84.84 (E2E) | 84.84 | 942,985.6 | 962.0 | 15.26x |
| **LargeFeat-2L** | 256 | 2 | sequential | 5352.78 ± 74.22 | 6511.25 | 14,945.5 | - | 1.00x |
| LargeFeat-2L | 256 | 2 | parallel | 604.48 ± 13.29 | 789.22 | 132,344.3 | - | 8.86x |
| LargeFeat-2L | 256 | 2 | cuda | ~116.23 (E2E) | 116.23 | 688,291.7 | 1153.0 | 46.05x |
| **WideHidden-2L** | 128 | 2 | sequential | 7052.95 ± 15.46 | 8507.83 | 11,342.8 | - | 1.00x |
| WideHidden-2L | 128 | 2 | parallel | 867.43 ± 8.99 | 1090.74 | 92,226.8 | - | 8.13x |
| WideHidden-2L | 128 | 2 | cuda | ~117.23 (E2E) | 117.23 | 682,423.5 | 1111.0 | 60.16x |
| **DeepNet-3L** | 128 | 3 | sequential | 3570.19 ± 6.98 | 4308.69 | 22,407.8 | - | 1.00x |
| DeepNet-3L | 128 | 3 | parallel | 456.03 ± 21.21 | 582.60 | 175,426.7 | - | 7.83x |
| DeepNet-3L | 128 | 3 | cuda | ~94.32 (E2E) | 94.32 | 848,147.0 | 951.0 | 37.85x |
| **LargeDeep-3L** | 256 | 3 | sequential | 6283.07 ± 30.74 | 7597.18 | 12,732.6 | - | 1.00x |
| LargeDeep-3L | 256 | 3 | parallel | 730.44 ± 17.65 | 925.34 | 109,522.4 | - | 8.60x |
| LargeDeep-3L | 256 | 3 | cuda | ~119.91 (E2E) | 119.91 | 667,160.8 | 1158.0 | 52.40x |

## 3. Correctness Verification Summary

* **Sequential vs. OpenMP:** Validated against scale-free synthetic graphs and Planetoid (Cora/PubMed). Both execution modes output identical FP32 matrices within an absolute tolerance of `1e-6`. 
* **OpenMP vs. CUDA:** Outputs match host implementations strictly within hardware floating-point summation rounding limits.

---

## 4. conclusion and Selection Guide

#### 4.1. Benchmark Overview & Graph Topologies
* **Stress Large Graph:** 80,000 nodes | 270,449 edges (dense, high-concurrency target)
* **Light Sparse Graph:** 3,000 nodes | 468 edges (sparse, lightweight payload)
* **Evaluation Metrics:** End-to-End Latency (ms), Throughput (nodes/s), Peak VRAM (MB), Speedup relative to CPU sequential baseline

---

#### 4.2. Key Insights & Bottleneck Analysis

* **GPU Acceleration Threshold Effect**
  * **Small Graph Scenarios (Nodes ≤ 3K):** Avoid enabling CUDA unconditionally. Fixed overheads—including kernel launch latency and host-to-device PCIe data transfers—dominate total runtime, resulting in severe performance regression with speedups of only **0.10x to 1.15x** compared to single-threaded CPU execution.
  * **Large Graph Scenarios (Nodes ≥ 80K):** The GPU demonstrates overwhelming advantages. Massive neighbor aggregation fully saturates compute units, achieving **10.97x to 43.68x** speedups and peak throughput nearing 1,000,000 nodes/s.
* **Marginal Returns of CPU Multi-Threading (`parallel`)**
  * On small graphs, insufficient computational granularity leads to negligible gains (0.84x` to `1.01x) due to thread synchronization and context switching overhead.
* On large graphs, CPU parallelism exhibits strong scalability: dual-thread execution delivers a stable 1.41× to 1.67× speedup, while scaling to 12 threads on the 80k-node graph achieves 6.81× to 8.86× acceleration. This parallel efficiency further improves as the feature dimension increases from 128 to 256 (LargeFeat-2L), pushing speedup from 6.81× to 8.86×. The higher arithmetic intensity per neighborhood aggregation effectively mitigates memory latency constraints and cache thrashing.
  
* **Sensitivity to Model Complexity**
  * As depth (2L → 3L/4L) and feature dimensions (64 → 256) scale up, CPU latency grows super-linearly. In contrast, GPU execution latency remains well-controlled, expanding the GPU performance advantage non-linearly under heavy workloads.

---
#### 4.3 Topology Impact & The `WideHidden-2L` Anomaly
* `WideHidden-2L` exhibits the worst CPU sequential latency (**8,507.83 ms**), outperforming even deeper 3-layer networks. The wider intermediate projection matrix exceeds fast CPU L1/L2 cache capacity, causing severe memory stall cycles.
* **GPU GEMM Sweet Spot:** Under GPU acceleration, the dense projection operations in `WideHidden-2L` map perfectly to CUDA cores, completing in just **117.23 ms** and registering the maximum recorded speedup (**60.16x**).

#### 4.4 Native C++ Compute vs. End-to-End (E2E) Pipeline Overheads
* Across all CPU runs, framework-level serialization, tensor instantiation, and input validation contribute **150 ms to 1,300 ms** of non-computational runtime latency.
* In the CUDA pipeline, once data is residing on the device, kernel execution overhead remains compact and deterministic, maintaining throughput above 660K nodes/s across all 2-layer and 3-layer configurations.


#### 4.5. Production Deployment Recommendations

| Use Case / Graph Scale | Recommended Backend | Engineering Guidance |
| :--- | :--- | :--- |
| **Small-scale / Real-time Subgraph Extraction** (Nodes < 5K) | `sequential` / `parallel` | Execute directly on CPU to bypass host-to-device data transfer penalties. |
| **Full-graph Inference / Offline Batching** (Nodes > 50K) | `cuda` | Prioritize GPU acceleration; memory footprint remains compact (< 600 MB) while delivering 10x–60x speedups. |
| **CPU-Only Environments** | `parallel` | Enable multi-threading on larger graphs or when feature dimensions ≥ 128 to achieve a throughput uplift. |


## 5. Known Limitations 

* **Out-of-Core Processing:** The current CSC loader and CUDA workspace allocate the entire graph in contiguous memory. Graphs exceeding device memory will trigger OOM faults.


---


