# Graph Neural Network Inference Engine on CPUs and GPUs

## Project Description
This project implements a custom inference engine for Graph Neural Networks (GNNs) targeting both multi-core CPUs and GPUs. The goal is to evaluate different parallelization strategies (vertex-centric vs. edge-centric) and memory optimizations (shared memory, coalescing) on large graph datasets.

## Project Documentation
- [Original Text of the Project](doc/project.md)
- [Requirements](doc/requirements.md)
- [Features List](doc/features.md)
- [Environment Instructions](doc/environment.md)

## Workload Division

| Feature / Task | Student ID |
| :--- | :--- |
| **1. Data Management & Pre-processing** | |
| F1.1: Custom Graph Format Converter (Python) | |
| F1.2: Synthetic Graph Generator | |
| F1.3: C++ Graph Loader (CSR/CSC) | |
| F1.4: C++ Feature/Weight Loader | |
| **2. Core Inference Engine (Sequential CPU Baseline)** | |
| F2.1: Sequential GCN Layer | |
| F2.2: Sequential GraphSAGE Layer | |
| F2.3: Multi-layer Pipeline | |
| **3. Parallel Execution (CPU Multi-core)** | |
| F3.1: OpenMP Vertex-Centric Execution | |
| F3.2: OpenMP Edge-Centric Execution | |
| **4. GPU Acceleration (CUDA - Linux Only)** | |
| F4.1: CUDA Memory Manager | |
| F4.2: Thread-per-Node CUDA Kernel | |
| F4.3: Thread-per-Edge CUDA Kernel | |
| F4.4: Shared Memory Optimization | |
| F4.5: Coalesced Memory Access Optimization | |
| **5. Verification, Benchmarking & Optional** | |
| F5.1: Correctness Verifier | |
| F5.2: Performance Profiler & Automated Benchmarking | |
| F6.1: Python Bindings (pybind11) - Optional | |

## Build Instructions
See `doc/environment.md` for details on setting up the cross-platform environment. We use Meson to build the project.

## LICENSE
This project is licensed under the EUPL-1.2 License - see the [LICENSE](LICENSE) file for details