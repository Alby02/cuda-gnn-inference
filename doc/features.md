# Features List
## Project: Graph Neural Network Inference on GPUs

This document breaks down the requirements into actionable, implementable features for the GNN inference engine.

### 1. Data Management & Pre-processing
*   **F1.1: Custom Graph Format Converter (Python):** Script to download and parse OGB and Planetoid datasets, converting them from their native formats into a custom, easy-to-read text or binary CSR format.
*   **F1.2: Synthetic Graph Generator (Python/C++):** Utility to generate Erdos-Rényi, Barabási-Albert, and small-world graphs, exporting them to the custom CSR format.
*   **F1.3: C++ Graph Loader:** A module that parses the custom CSR files into highly efficient in-memory C++ data structures (`row_ptr`, `col_ind`, `values`).
*   **F1.4: C++ Feature/Weight Loader:** A module that loads dense matrices (node features, edge features, and pre-trained layer weights) into contiguous 1D arrays for cache-friendly access.

### 2. Core Inference Engine (Sequential CPU Baseline)
*   **F2.1: Sequential GCN Layer:** Implementation of the GCN aggregation and update steps operating strictly on the CPU using standard C++ loops over the CSR structure.
*   **F2.2: Sequential GraphSAGE Layer:** Implementation of the GraphSAGE aggregation (with concatenation) and update steps operating strictly on the CPU.
*   **F2.3: Multi-layer Pipeline:** The overarching logic to string together $L$ layers, passing the output node embeddings of layer $l$ as the input features for layer $l+1$.

### 3. Parallel Execution (CPU Multi-core)
*   **F3.1: OpenMP Vertex-Centric Execution:** Parallelization of the aggregation phase where each thread is assigned a subset of nodes and gathers messages from their respective neighbors.
*   **F3.2: OpenMP Edge-Centric Execution:** Parallelization of the aggregation phase where each thread is assigned a subset of edges, accumulating messages into the destination node (requires careful handling of race conditions using atomic adds or reduction arrays).

### 4. GPU Acceleration (CUDA - Linux Only)
*   **F4.1: CUDA Memory Manager:** Wrappers for `cudaMalloc`, `cudaMemcpy`, and `cudaFree` to seamlessly move the CSR graph, node features, and layer weights from host to device memory and retrieve the final embeddings.
*   **F4.2: Thread-per-Node CUDA Kernel:** A baseline GPU implementation where each CUDA thread is responsible for the aggregation and update of exactly one node.
*   **F4.3: Thread-per-Edge CUDA Kernel:** An alternative GPU implementation where each CUDA thread processes one edge, relying on `atomicAdd` to accumulate messages into the destination node's state.
*   **F4.4: Shared Memory Optimization:** An advanced kernel version that utilizes GPU shared memory to cache frequently accessed node features or intermediate weights, reducing global memory bandwidth pressure.
*   **F4.5: Coalesced Memory Access Optimization:** Ensuring that data layouts (e.g., transposing feature matrices) and thread mappings allow for coalesced reads/writes when traversing the CSR structure.

### 5. Verification and Benchmarking
*   **F5.1: Correctness Verifier:** A utility function that compares the final output node embeddings of the Parallel CPU and CUDA implementations against the Sequential CPU baseline to guarantee mathematical equivalence within floating-point tolerances.
*   **F5.2: Performance Profiler:** Integration of high-resolution C++ timers (e.g., `std::chrono`) and CUDA events (`cudaEventRecord`) to accurately measure end-to-end execution time, throughput (nodes/sec, edges/sec), and memory footprint.
*   **F5.3: Automated Benchmarking Suite:** A script that automatically compiles the project, runs it across a sweep of configurations (different graphs, varying thread counts, different kernels), and dumps the results into a CSV file for plotting.

### 6. Optional Extensions
*   **F6.1: Python Bindings (`pybind11`):** Implementation of C++ wrappers using `pybind11` to expose the core graph loading and inference execution functions to Python.
*   **F6.2: Python Test Suite:** A small Python testing script that imports the compiled C++ extension, loads a graph, runs inference, and verifies the output.
