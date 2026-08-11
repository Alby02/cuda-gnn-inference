# Graph Neural Network Inference Engine on CPUs and GPUs

## Project Description
This project implements a custom inference engine for Graph Neural Networks (GNNs) targeting both multi-core CPUs and GPUs. The goal is to evaluate different parallelization strategies (vertex-centric vs. edge-centric) and memory optimizations (shared memory, coalescing) on large graph datasets.

## Project Documentation
- [Original Text of the Project](doc/project.md)
- [Requirements](doc/requirements.md)
- [Features List](doc/features.md)
- [Environment Instructions](doc/environment.md)
- [GNN & Sparse Graph Representations Knowledge Base](doc/knowledge.md)

## Workload Division

| Feature / Task | Student ID |
| :--- | :--- |
| **1. Core Architecture & Pipeline Orchestration** | |
| F1.1: Core Graph Representation | s360540 |
| F1.2: Engine Interface & GNN Model's Layers definition | s360540 |
| | |
| **2. Data Management & Utilities** | |
| F2.1: Custom Graph Format Converter (Python) | |
| F2.2: Synthetic Graph Generator (Python/C++) | |
| F2.3: Graph Loader (C++) | |
| F2.4: Feature & Weight Loaders (C++) | |
| | |
| **3. Core Inference Engine: Sequential CPU (Baseline)** | |
| F3.1: Sequential Layer execution | |
| | |
| **4. Core Inference Engine: Parallel CPU (Multi-core)** | |
| F4.1: OpenMP Vertex-Centric Execution | |
| F4.2: OpenMP Edge-Centric Execution | |
| | |
| **5. Core Inference Engine: GPU Acceleration (CUDA - Linux Only)** | |
| F5.1: CUDA Memory Manager | |
| F5.2: Thread-per-Node CUDA Kernel | |
| F5.3: Thread-per-Edge CUDA Kernel | |
| F5.4: Shared Memory Optimization | |
| F5.5: Coalesced Memory Access Optimization | |
| | |
| **6. Verification and Benchmarking** | |
| F6.1: Correctness Verifier | |
| F6.2: Performance Profiler | |
| F6.3: Automated Benchmarking Suite | |
| | |
| **7. Optional Extensions** | |
| F7.1: Python Bindings (`pybind11`) | |
| F7.2: Python Test Suite | |

## Build Instructions
See `doc/environment.md` for details on setting up the cross-platform environment. We use Meson to build the project.

## LICENSE
This project is licensed under the EUPL-1.2 License - see the [LICENSE](LICENSE) file for details