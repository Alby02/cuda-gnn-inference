# Features List
## Project: Graph Neural Network Inference on GPUs

This document breaks down the requirements into actionable, implementable features for the GNN inference engine.

### 1. Architecture & Pipeline Orchestration
*   **F1.1: Core Interfaces:** Define the foundational abstractions for the system, including the `Graph` interface (to be implemented by `DirectedGraph` and `UndirectedGraph` specializations) and the `Engine` interface (to abstract Sequential, Parallel CPU, and GPU executions).
*   **F1.2: Component DataFactory:** Implement a Factory design pattern to abstract the instantiation of Graph objects, Data Loaders, and Inference Engines, enabling clean dependency injection across the project.
*   **F1.3: Multi-layer Pipeline Orchestrator:** The overarching logic to string together $L$ layers, passing the output node embeddings of layer $l$ as the input features for layer $l+1$. This component must rely purely on the interfaces defined in F1.1.

### 2. Data Management & Graph Data Structure
*   **F2.1: Graph Implementations:** Highly efficient in-memory C++ data structures (`DirectedGraph` and `UndirectedGraph`) implementing the `Graph` interface. Must support CSR and CSC layouts.
*   **F2.2: Custom Graph Format Converter (Python):** Script to download and parse OGB and Planetoid datasets, converting them from their native formats into a custom, easy-to-read text or binary CSR/CSC format.
*   **F2.3: Synthetic Graph Generator (Python/C++):** Utility to generate Erdos-Rényi, Barabási-Albert, and small-world graphs.
*   **F2.4: Graph Loader (C++):** A module that parses the custom CSR/CSC files into the in-memory graph data structure via the DataFactory.
*   **F2.5: Feature & Weight Loaders (C++):** A module that loads dense matrices (load matrix 1: node features and edge features, load matrix 2... : pre-trained layer weights needed to be repeated for each layer).

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
