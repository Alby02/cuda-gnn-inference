# Graph Neural Network Inference Engine on CPUs and GPUs

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![CUDA](https://img.shields.io/badge/CUDA-Toolkit-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Build System](https://img.shields.io/badge/Build-Meson%20%2B%20Ninja-orange.svg)](https://mesonbuild.com/)
[![License: EUPL 1.2](https://img.shields.io/badge/License-EUPL%201.2-blue.svg)](LICENSE)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Alby02/cuda-gnn-inference/blob/main/Colab/Runner.ipynb)

A high-performance, modular full-batch Graph Neural Network (GNN) inference engine implemented from scratch in **C++20**, **OpenMP**, and **CUDA**. It evaluates sequential CPU, multi-core CPU, and NVIDIA GPU execution for GCN and GraphSAGE workloads.

---

## 1. Architectural Overview

The engine adopts a **fully decoupled, modular architecture** that cleanly separates **algorithmic layer logic**, **hardware execution primitives**, and **memory lifecycle management**. This avoids monolithic dispatch engines, eliminates $M \times N$ code duplication, and enables zero-allocation inference loops.

> 📖 **Full Architectural Specification**: See [doc/architecture.md](doc/architecture.md) for in-depth concepts, diagrams, and memory management details.

```mermaid
flowchart TD
    subgraph Client Application
        App["main / Benchmark Runners"]
    end

    subgraph High-Level Orchestration
        Runtime["InferenceRuntime&lt;Executor&gt;<br/>(Coordinates model execution & buffer swaps)"]
        Model["Model&lt;Layers...&gt;<br/>(Ordered stack of heterogeneous layers)"]
    end

    subgraph Algorithmic Layer Graph
        Layers["Layers: GCNLayer | GraphSAGELayer<br/>• Encapsulate parameters (W_neigh, W_self, b)<br/>• Own forward_layer(layer, graph, executor, workspace)"]
    end

    subgraph Hardware Execution & Primitives
        Executor["Executor Concept<br/>• aggregateGCN() [Degree-normalized SpMM]<br/>• aggregateNeighbors() [Mean, Sum, Max]<br/>• rowByColumn() [Dense GEMM projection]<br/>• relu() [Activation]<br/>• add() / biasAdd() [Elementwise & broadcast]"]
        SeqE["SequentialExecutor<br/>(CPU Baseline)"]
        OmpE["ParallelExecutor<br/>(OpenMP Static + SIMD)"]
        CudaE["CudaExecutor<br/>(CUDA 1D Grid-Stride)"]
    end

    subgraph Memory Management & State
        Workspace["Workspace Concept<br/>• Pre-allocated Current & Next buffers<br/>• Scratch & Branch workspaces<br/>• Backend GCN aggregation state<br/>• Zero dynamic allocations in inference loops"]
        CpuW["CpuContext / ParallelCpuContext<br/>(Host buffers + CPU GCN state)"]
        CudaW["CudaWorkspace<br/>(Device buffers + CUDA GCN state + events)"]
    end

    App --> Runtime
    App --> Workspace
    Runtime --> Model
    Runtime --> Executor
    Runtime --> Workspace

    Model -->|Iterates via std::visit| Layers
    Layers -->|"1. Invokes primitive operations"| Executor
    Layers -->|"2. Reads/writes working buffers"| Workspace
    Executor -->|"Reads/writes"| Workspace

    Executor -.-> SeqE
    Executor -.-> OmpE
    Executor -.-> CudaE

    Workspace -.-> CpuW
    Workspace -.-> CudaW
```

### Core Architecture Pillars

1. **`Executor` (Hardware Compute Primitives)**: Exposes typed hardware operations (`rowByColumn`, `aggregateGCN`, `aggregateNeighbors`, `add`, `biasAdd`, `relu`). Implementations: `SequentialExecutor`, `ParallelExecutor` (OpenMP), `CudaExecutor`.
2. **`Workspace` (Memory & State Management)**: Manages pre-allocated intermediate buffers (`current`, `next`, `scratch`, `branch`) and backend GCN aggregation state (`GCNAggregationState` / `CudaGCNAggregationState`), guaranteeing zero dynamic allocations during steady-state inference.
3. **`Layer` (Algorithmic Logic)**: Encapsulates parameters ($W_{\text{neigh}}, W_{\text{self}}, b$) and defines `forward_layer(layer, graph, executor, workspace)` by composing executor primitives.
4. **`Model` (Layer Pipeline Container)**: Holds an ordered sequence of heterogeneous layer descriptors (`std::variant`) and validates adjacent feature dimensions.
5. **`InferenceRuntime` (Execution Orchestrator)**: High-level engine executing the layer pipeline via `std::visit` and exchanging ping-pong buffers (`current` and `next`) via `workspace.swapBuffers()`.

---

## 2. Project Documentation

| Document | Description |
| :--- | :--- |
| [Documentation](doc/documentation_summary.md) | Implementation and design choices summarized in order to meet the derivables requirements. |
| [Modular Architecture Specification](doc/architecture.md) | In-depth technical specification of the 5-pillar architecture, C++20 concepts, and memory models. |
| [Original Project Specification](doc/project.md) | Official course problem definition, required deliverables, and background. |
| [Requirements Document](doc/requirements.md) | Functional, non-functional, architectural, and performance requirements. |
| [Features Specification](doc/features.md) | Granular catalog of all implementable features across all engine components. |
| [Semantic Contract](doc/semantics.md) | Exact GCN/GraphSAGE mathematics, graph conventions, and selected parallel mappings. |
| [Environment Setup Instructions](doc/environment.md) | Setup guide for Windows (MSYS2 UCRT64), Linux, WSL, and Google Colab. |
| [GNN & Graph Knowledge Base](doc/knowledge.md) | Mathematical formulation of GCN/GraphSAGE and sparse graph storage (CSR/CSC). |
| [GCN Specification](doc/GCN.md) | Summarization of GCN logic, implementation and results. |
| [GraphSAGE Specification](doc/GraphSAGE.md) | Summarization of GraphSAGE logic, implementation and results. |


---

## 3. Build & Execution Instructions

### Prerequisites

- **C++ Compiler**: GCC 11+ / Clang 14+ supporting C++20.
- **Build System**: [Meson](https://mesonbuild.com/) (>= 0.60) and [Ninja](https://ninja-build.org/).
- **OpenMP**: For multi-core CPU parallel execution.
- **CUDA Toolkit** (optional, Linux/WSL/Colab): required only for the CUDA backend (`nvcc`).
- **Python 3.10–3.13** and [uv](https://docs.astral.sh/uv/): required for dataset, validation, and benchmark scripts. Dependencies are declared in `pyproject.toml`, locked in `uv.lock`, and mirrored in `python_libraries.txt` for Colab's `pip` setup.

### Quick Start

```bash
# 0. Create the Python tooling environment
uv sync --locked

# 1. Configure the build directory (CUDA is auto-detected)
meson setup builddir

# 2. Compile targets
meson compile -C builddir

# 3. Format and statically analyze the source (Ninja backend)
ninja -C builddir clang-format
ninja -C builddir clang-tidy

# 4. Run one of the modes exposed by the single CLI
./builddir/gnn --backend sequential
./builddir/gnn --backend parallel
./builddir/gnn --backend cuda
```

On MSYS2 UCRT64, CUDA is not supported; configure it explicitly off and use the `.exe` target:

```bash
meson setup builddir -Dcuda=disabled
meson compile -C builddir
./builddir/gnn.exe --backend sequential
./builddir/gnn.exe --backend parallel --threads 4
```

To create a complete, small workload (graph, features, and model parameters) and run it:

```bash
uv run python scripts/make_example_inputs.py synth_data/example
./builddir/gnn --backend sequential \
  --graph synth_data/example/graph.bin_graph \
  --features synth_data/example/features.bin_matrix \
  --model synth_data/example/gcn.model
```

The loader validates the CSC topology, model matrices, and feature row count. The three input options
must be supplied together. `scripts/synthetic_generator.py` is a lower-level graph/features generator;
`scripts/run_experiments.py` creates complete seeded GCN and GraphSAGE workloads and runs the full
native/PyTorch Geometric comparison pipeline.

The `clang-format` and `clang-tidy` targets are generated automatically by
Meson when the corresponding tools and project configuration files are
available (`.clang-format` and `.clang-tidy`).

The sequential mode is always available. The `parallel` and `cuda` modes are
listed by `gnn --help` only when OpenMP and CUDA, respectively, were detected
while configuring the build.

For detailed cross-platform environment setup (including Windows MSYS2 UCRT64 and Google Colab workflows), see [doc/environment.md](doc/environment.md).

---
## 4. Execution parameters

```
Usage: gnn --backend MODE [options]
Modes: sequential parallel cuda   (parallel/cuda only listed if OpenMP/CUDA were found at build time)
```

`--backend MODE` (optional, default: `sequential`)

- `sequential` — Single-threaded CPU run
- `parallel` — Multi-threaded CPU (OpenMP) — only available if OpenMP was found at build time
- `cuda` — CUDA GPU run — only available if CUDA (`nvcc`) was found at build time

Data options — `--graph`, `--features`, and `--model` are **required together**; if any one of the three is missing, the program ignores all three and runs the built-in two-node demo instead:

- `--graph FILE` — binary graph topology in the repository's custom `.bin_graph` format
- `--features FILE` — binary dense node-feature matrix in the custom `.bin_matrix` format
- `--model FILE` — model description (layer types, activations, weight/bias paths); see `src/common/data/model_io.hpp` for the exact format. Complete example bundles can be created with `scripts/make_example_inputs.py`; controlled synthetic benchmark bundles are created by `scripts/run_experiments.py`.

Other options:

- `--help`, `-h` — show CLI usage and exit
- `--output CSV` — path for the machine-readable benchmark results; omit to skip CSV export
- `--embeddings FILE` — path to write the final output embedding matrix; omit to skip
- `--warmups N` — unmeasured warm-up iterations before timing (default: `1`)
- `--repetitions N` — number of measured iterations (default: `10`)
- `--threads N` — OpenMP worker threads, only used by `--backend parallel` (default: `1`)
- `--block-size N` — CUDA threads per block, only used by `--backend cuda` (default: `256`)

**Runtime behaviour to take into account**:

- With no `--graph`/`--features`/`--model` (or with a partial set of the three), the program runs the built-in demo and prints `running demo`.
- If the loaded feature matrix's row count does not match the graph's node count, execution aborts with an error.
- The program prints the mean/stddev compute time and a preview of the result matrix to stdout.

Environment variables that affect execution:

- `OMP_NUM_THREADS` — controls CPU-thread count for OpenMP (used together with `--threads` for `--backend parallel`)
- `CUDA_VISIBLE_DEVICES` — controls which GPUs are visible to the process (useful for `--backend cuda`)

Examples:

- Run built-in demo (no data paths):
```bash
./builddir/gnn --backend sequential
```

- Run with custom data:
```bash
./builddir/gnn --backend cuda --graph path/to/graph.bin_graph --features path/to/features.bin_matrix --model path/to/model.manifest
```

- Run parallel mode with 16 threads:
```bash
export OMP_NUM_THREADS=16
./builddir/gnn --backend parallel --threads 16 --graph path/to/graph.bin_graph --features path/to/features.bin_matrix --model path/to/model.manifest
```

- Export benchmark results and the output embeddings:
```bash
./builddir/gnn --backend sequential --graph g.bin_graph --features f.bin_matrix --model m.manifest --output results.csv --embeddings out.bin_matrix --warmups 3 --repetitions 20
```

- Run the tracked CPU/Python edge-case validation:

```bash
uv run python scripts/check_edge_cases.py --native ./builddir/gnn
```

The checker auto-detects every backend compiled into the executable and tests each one, plus
PyTorch Geometric, against an independent dense Python reference. The fixtures cover empty
neighborhoods, an undirected weighted graph, a mixed multi-layer model, and GraphSAGE without
bias. A machine-readable report is written to `experiment_results/edge-cases/verification.json`.
Use `--backend sequential parallel` (or include `cuda`) to require an explicit backend set.

Run a compact CPU scaling experiment on MSYS2 UCRT64:

```bash
uv run python scripts/run_experiments.py \
  --native ./builddir/gnn.exe --dataset none \
  --nodes 1000 10000 --widths 32 128 --depths 2 4 --skews 0 1 \
  --backend sequential parallel --threads 1 2 4 \
  --warmups 2 --repetitions 5 --repeat-checks 1 \
  --output-dir experiment_results/cpu-scaling
```

The runner changes one workload dimension at a time, saves raw samples and comparisons as CSV,
records workload/configuration metadata as JSON, and emits both PNG and SVG figures. Each
workload/model gets a latency, throughput, speedup, and numerical-error dashboard. A scaling
figure is emitted only when that axis contains at least two measured values, so a single workload
is never presented as a misleading scaling chart. Existing results can be replotted without
rerunning inference:

```bash
uv run python scripts/run_experiments.py \
  --plots-only --output-dir experiment_results/cpu-scaling
```

### Google Colab

Open [`Colab/Runner.ipynb`](Colab/Runner.ipynb), select a GPU runtime, and choose **Runtime → Run all**.
The notebook clones and builds on Colab's local disk, writes persistent results to Google Drive, and
runs the edge-case matrix and a genuine multi-point synthetic sweep by default. It streams progress,
prints verification summaries, and displays the generated dashboards and scaling plots. Cora,
the larger scaling sweep, and the million-node run are opt-in through `RUN_CORA`, `RUN_SCALING`, and
`RUN_MILLION_NODES`; leave them disabled until the quick run passes.



---

## 5. Dataset formats 

The project uses two custom binary formats consumed by `graph::GraphLoader`:

### Graph file: `.bin_graph` (single binary file)

Layout (in byte order little-endian; header format must match Python struct `'<QQBB'`):

1. GraphHeader (packed, total 18 bytes)
   - `uint64_t numNodes` (8 bytes)  
   - `uint64_t numEdges` (8 bytes)  
   - `uint8_t  isDirected` (1 byte) — `0` or `1`  
   - `uint8_t  hasWeights` (1 byte) — `0` or `1`

2. `colPtr` — array of `uint64_t` of length `numNodes + 1` (column/CSR pointer array)  
3. `rowInd` — array of `uint64_t` of length `numEdges` (destination row indices / flattened edges)  
4. `weights` — (optional) array of `float` (IEEE 754 float32) of length `numEdges` — present only if `hasWeights` is 1

Notes:

- The code reads header with `readExact`, then reads `colPtr` and `rowInd` as `uint64_t`.  
- `colPtr` count is computed as `numNodes + 1`.  
- The `GraphHeader` `static_assert` enforces the layout matches Python `'<QQBB'`. Use little-endian writes from Python or tools.  
- The loader returns a `HostGraphCSC` (CSC orientation used internally — but the header/arrays are what the loader expects).  

### Dense matrix file: `.bin_matrix` (used for node features)

Layout (little-endian):

1. `uint64_t rows` (8 bytes)  
2. `uint64_t cols` (8 bytes)  
3. `rows * cols` floats (IEEE 754 float32) in row-major order

Notes:

- The loader reads `rows`, `cols` and then reads exactly `rows*cols` floats into the internal `Matrix<HostBuffer<float>>`.  

## 6. Three-Person Workload Division

The project uses one shared-infrastructure stream and two vertical model streams. All three members write parallel code: `s362415` takes GCN through OpenMP and CUDA, `s296248` does the same for GraphSAGE, and `s360540` implements the common OpenMP/CUDA infrastructure and primitives.

The statuses below combine project-level progress with an audit of the codebase against the
acceptance criteria in [`doc/features.md`](doc/features.md). `Completed`
means the work is fully implemented and integrated in the codebase; `In progress` means work is actively
underway (e.g. final experiments, technical report, presentation); `Optional — unselected in profile` indicates conditional
extensions not selected in the baseline project profile (`semantics.md`).

| Task ID(s) | Task | Student ID | Status |
| :--- | :--- | :---: | :---: |
| T-CON-01 | Directed/undirected canonical CSC graph representation and orientation enum | `s360540` | Completed |
| T-CON-03–T-CON-04 | GCN/GraphSAGE layer descriptors and generic model definition | `s360540` | Completed |
| T-IO-02 | Canonical graph loader and orientation validation | `s362415` | Completed |
| T-DATA-01 | Reproducible synthetic graph generator | `s362415` | Completed |
| T-DATA-03 | Public dataset preparation | `s362415` | Completed |
| T-CON-02 | Storage-generic buffers, matrices, graphs, and CUDA kernel views | `s360540` | Completed |
| T-CON-06–T-CON-08 | Executor contracts, reusable workspaces, and outer backend dispatch | `s360540` | Completed |
| T-CON-05 (GCN) | GCN degree and self-message semantic preparation | `s362415` | Completed |
| T-CON-05 (GraphSAGE) | GraphSAGE neighbor-total semantic preparation | `s296248` | Completed |
| T-IO-01, T-IO-05 | Versioned workload bundle and machine-readable result schema | `s360540` | Completed |
| T-IO-03 | Feature, model, layer, and parameter loading | `s362415` | Completed |
| T-IO-04, T-IO-06 | CLI configuration and unsupported-composition validation | `s360540` | Completed |
| T-SEQ-01 | Shared sequential dense, bias, activation, and branch-combination primitives | `s360540` | Completed |
| T-SEQ-05 | Generic model execution with layer iteration and ping-pong buffers | `s360540` | Completed |
| T-SEQ-06 | Allocation-free steady-state workspace reuse | `s360540` | Completed |
| T-SEQ-02 | Sequential GCN normalized aggregation | `s362415` | Completed |
| T-SEQ-04 (GCN) | Sequential GCN layer execution | `s362415` | Completed |
| T-SEQ-03 | Sequential GraphSAGE unweighted non-self mean | `s296248` | Completed |
| T-SEQ-04 (GraphSAGE) | Sequential GraphSAGE layer execution | `s296248` | Completed |
| T-VER-03, T-VER-05 | Common comparison and invalid-input test infrastructure | `s360540` | Completed |
| T-VER-01, T-VER-04/T-VER-06 (GCN) | GCN fixtures and native/framework backend verification | `s362415` | Completed |
| T-VER-02, T-VER-04/T-VER-06 (GraphSAGE) | GraphSAGE fixtures and native/framework backend verification | `s296248` | Completed |
| T-OMPV-03, T-OMPV-05 | Common OpenMP dense/elementwise operations and workspace | `s360540` | Completed |
| T-OMPV-01, T-OMPV-04 (GCN) | Destination-owned OpenMP GCN and its configurations | `s362415` | Completed |
| T-OMPV-02, T-OMPV-04 (GraphSAGE) | Destination-owned OpenMP GraphSAGE and its configurations | `s296248` | Completed |
| T-OMPE-01–T-OMPE-05 | Conditional additional OpenMP mapping | `s296248` | Completed |
| T-CUDA-01–T-CUDA-04, T-CUDAV-03 | CUDA ownership/runtime and common CUDA operations | `s360540` | Completed |
| T-CUDAV-01, T-CUDAV-04 (GCN) | CUDA GCN aggregation and launch configurations | `s362415` | Completed |
| T-CUDAV-02, T-CUDAV-04 (GraphSAGE) | CUDA GraphSAGE aggregation and launch configurations | `s296248` | Completed |
| T-CUDAA-01–T-CUDAA-04 | Conditional additional CUDA mapping | `s296248` | Completed |
| T-EXP-01–T-EXP-03 | Shared-memory and sparse/dense studies | `s296248` | Completed |
| T-DATA-02, T-DATA-04/T-DATA-05 (GCN) | Synthetic workload ranges and GCN parameters/counts | `s362415` | Completed |
| T-DATA-04/T-DATA-05 (GraphSAGE) | GraphSAGE parameters/counts | `s296248` | Completed |
| T-FRM-01, T-FRM-02, T-FRM-05 | Shared external-framework adapter and measurement boundaries | `s360540` | Completed |
| T-FRM-03, T-FRM-06 (GCN) | GCN framework mapping and comparison | `s362415` | Completed |
| T-FRM-04, T-FRM-06 (GraphSAGE) | GraphSAGE framework mapping and comparison | `s296248` | Completed |
| T-BENCH-01, T-BENCH-02, T-BENCH-07 | Common benchmark runner, timing boundaries, and metadata | `s360540` | Completed |
| T-BENCH-03–T-BENCH-06 (OpenMP/GCN) | OpenMP scaling and GCN benchmark results | `s362415` | Completed |
| T-BENCH-03–T-BENCH-06 (CUDA/GraphSAGE) | CUDA configurations, memory, and GraphSAGE benchmark results | `s296248` | Completed |
| T-DEL-01–T-DEL-02 | Build, CLI, bundle, and run documentation | `s360540` | Completed |
| T-DEL-03 | Comparison plots and visualization scripts | `s362415`, `s296248` | Completed |
| T-DEL-04 | Technical report (architecture, evaluation, analysis) | all members (`s362415`) | Completed |
| T-DEL-05 | Presentation and live demonstration path | all members (`s362415`) | Completed |

Detailed acceptance criteria for every task are in [doc/features.md](doc/features.md#61-three-person-delivery-split).

---

## 7. License

This project is licensed under the European Union Public Licence (EUPL-1.2) - see the [LICENSE](LICENSE) file for details.
