# Requirements Document
## Project: Graph Neural Network Inference on GPUs

### 1. Introduction
The objective of this project is to develop a custom inference engine for Graph Neural Networks (GNNs) targeting both multi-core CPUs and GPUs. The engine must evaluate how different parallelization strategies (vertex parallel, edge parallel, message-passing batching, sparse vs. dense representations) affect performance and scalability on large graphs.

### 2. Functional Requirements
*   **Graph Support:** Must support single, large, static graphs, both directed and undirected.
*   **Feature Support:** Must support node features (attributes) and, optionally, edge features (required for specific architectures like attention mechanisms or certain GraphSAGE variants).
*   **Supported Architectures:** Must implement the forward pass (inference) for at least one or two basic GNN architectures (e.g., **GCN** and/or **GraphSAGE**).
*   **Inference Mode:** Must focus on full-batch inference on a single large graph, rather than mini-batching across many small graphs.
*   **Weights:** Must assume weights and all layer parameters are pre-loaded and fixed. Training logic is not required.
*   **Data Loading:** Must be capable of loading graphs represented in CSR (Compressed Sparse Row) and/or CSC (Compressed Sparse Column) formats, along with dense feature matrices.

### 3. Non-Functional Requirements
*   **Programming Languages:** Must be implemented entirely in **C++** and **CUDA** (for GPU extensions).
*   **Framework Restrictions:** The use of high-level Deep Learning frameworks (e.g., PyTorch, DGL, TensorFlow) is **strictly prohibited** for the core inference implementation. Only custom kernels and implementations are valid.
*   **Environment:** The single-core and multi-threaded CPU code must compile and run on both Unix-like systems and Windows systems. The CUDA GPU version is only required to work on Linux (specifically targeting Google Colab environments).
*   **Memory Efficiency:** Must utilize efficient sparse data structures (like CSR/CSC) for the graph topology to minimize memory footprint.

### 4. Implementation Variants Required
The project must deliver three distinct implementations for comparison:
1.  **Sequential CPU Baseline:** A C/C++ implementation operating sequentially over efficient graph formats.
2.  **Parallel CPU Implementation:** A multi-core implementation utilizing explicit threads, tasks, or OpenMP, exploiting parallelism over nodes, edges, or layers.
3.  **Parallel GPU Implementation (CUDA):** One or more CUDA versions exploring:
    *   Different thread mappings (e.g., thread-per-node, thread-per-edge, thread-per-feature).
    *   Structural optimizations (e.g., utilization of shared memory, coalesced memory access on CSR structures).

### 5. Experimental and Performance Requirements
The engine must include benchmarking capabilities to evaluate and compare performance across the different implementations.
*   **Metrics:** Must measure Throughput (nodes/s or edges/s), Memory Footprint, and Scalability.
*   **Stability:** Each configuration (graph size, feature size, depth, implementation variant) must be run multiple times to report stable timings.
*   **Comparisons:** Must evaluate and compare:
    *   Sequential vs. Multi-core vs. GPU implementations.
    *   Vertex-centric vs. Edge-centric strategies.
    *   Impact of memory optimizations (with/without shared memory, feature compression).
*   **Datasets:** Must benchmark against:
    *   Publicly available datasets (e.g., Open Graph Benchmark (OGB), Planetoid).
    *   Randomly generated synthetic graphs (e.g., Scale-free/Barabási-Albert, Erdos-Rényi, small-world).

### 6. Optional Extensions
*   **Python Bindings:** Provide a Python interface (e.g., using `pybind11`) to allow seamless invocation of the C++/CUDA inference engine from Python, enabling easier integration with existing data loading pipelines and high-level evaluation scripts.
