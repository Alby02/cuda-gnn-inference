# Features List
## Project: Graph Neural Network Inference on GPUs

This document breaks down the requirements into actionable, implementable features for the GNN inference engine.

### 1. Core Architecture & Pipeline Orchestration
*   **F1.1: Core Graph Representation & Factory:** Consolidates the foundational C++ `Graph` interface (for `DirectedGraph` and `UndirectedGraph` supporting CSR and CSC layouts), concrete graph data structures.
*   **F1.2: Engine Interface, DataFactory & Pipeline Orchestrator:** Defines the `Engine` interface (abstracting Sequential, Parallel CPU, and GPU executions), the Engine DataFactory for engine instantiation, and the multi-layer pipeline orchestrator that strings together $L$ layers ($h^{(l+1)} = \text{Layer}(h^{(l)})$) using dependency injection.

### 2. Data Management & Utilities
*   **F2.1: Custom Graph Format Converter (Python):** Script to download and parse OGB and Planetoid datasets, converting them from native formats into a custom CSR/CSC binary or text format.
*   **F2.2: Synthetic Graph Generator (Python/C++):** Utility to generate Erdos-Rényi, Barabási-Albert, and small-world graphs.
*   **F2.3: Graph Loader (C++):** A C++ module that parses the custom CSR/CSC files into in-memory graph structures via the Graph DataFactory.
*   **F2.4: Feature & Weight Loaders (C++):** A C++ module that loads dense feature matrices (input node features $h^{(0)}$ and optional edge features) as well as pre-trained layer weight matrices ($W^{(l)}$ for each layer $l \in [0, L-1]$).

### 3. Core Inference Engine: Sequential CPU (Baseline)
*   **F3.1: Sequential GCN Layer:** Implementation of the GCN aggregation and update steps operating strictly on the CPU using standard C++ loops over the graph structure.
*   **F3.2: Sequential GraphSAGE Layer:** Implementation of the GraphSAGE aggregation (with concatenation) and update steps operating strictly on the CPU.

### 4. Core Inference Engine: Parallel CPU (Multi-core)
*   **F4.1: OpenMP Vertex-Centric Execution:** Parallelization of the aggregation phase where each thread is assigned a subset of nodes and gathers messages from their respective neighbors.
*   **F4.2: OpenMP Edge-Centric Execution:** Parallelization of the aggregation phase where each thread is assigned a subset of edges, accumulating messages into the destination node.

### 5. Core Inference Engine: GPU Acceleration (CUDA - Linux Only)
*   **F5.1: CUDA Memory Manager:** Wrappers for memory operations to seamlessly move the graph structures, node features, and layer weights from host to device memory and retrieve the final embeddings.
*   **F5.2: Thread-per-Node CUDA Kernel:** A baseline GPU implementation where each CUDA thread is responsible for the aggregation and update of exactly one node.
*   **F5.3: Thread-per-Edge CUDA Kernel:** An alternative GPU implementation where each CUDA thread processes one edge, relying on `atomicAdd` to accumulate messages into the destination node's state.
*   **F5.4: Shared Memory Optimization:** An advanced kernel version that utilizes GPU shared memory to cache frequently accessed node features or intermediate weights, reducing global memory bandwidth pressure.
*   **F5.5: Coalesced Memory Access Optimization:** Ensuring that data layouts (e.g., transposing feature matrices) and thread mappings allow for coalesced reads/writes when traversing the graph data structure.

### 6. Verification and Benchmarking
*   **F6.1: Correctness Verifier:** A utility function that compares the final output node embeddings of the Parallel CPU and CUDA implementations against the Sequential CPU baseline to guarantee mathematical equivalence within floating-point tolerances.
*   **F6.2: Performance Profiler:** Integration of high-resolution C++ timers (e.g., `std::chrono`) and CUDA events (`cudaEventRecord`) to accurately measure end-to-end execution time, throughput (nodes/sec, edges/sec), and memory footprint.
*   **F6.3: Automated Benchmarking Suite:** A script that automatically compiles the project, runs it across a sweep of configurations (different graphs, varying thread counts, different kernels), and dumps the results into a CSV file for plotting.

### 7. Optional Extensions
*   **F7.1: Python Bindings (`pybind11`):** Implementation of C++ wrappers using `pybind11` to expose the core graph loading and inference execution functions to Python.
*   **F7.2: Python Test Suite:** A small Python testing script that imports the compiled C++ extension, loads a graph, runs inference, and verifies the output.
