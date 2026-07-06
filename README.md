# GNN Inference Engine (C++/CUDA)

## Project Summary

Graph Neural Networks (GNNs) extend deep learning models to graph-structured data, with applications in recommender systems, social network analysis, cybersecurity, bioinformatics, and industrial recommendation.

The goal of this project is to develop a high-performance inference engine for GNNs targeting multi-core CPUs (C++) and GPUs (CUDA). It evaluates how different parallelization strategies—such as vertex parallel, edge parallel, message-passing batching, and sparse versus dense representations—affect performance and scalability on large graphs.

## Planned Features
- **Multi-core CPU (C++) Implementation:** Baseline and optimized parallel CPU processing.
- **GPU (CUDA) Acceleration:** Massively parallel graph processing leveraging CUDA.
- **Performance Evaluation:** Benchmarking different parallelization and memory representation strategies.

## Development Process
- **Google Colab Optimized:** Designed with a workflow that maximizes the usage of Google Colab's limited GPU time by syncing through Google Drive and Git.

## Getting Started

To efficiently use Google Colab without wasting compute resources, we employ a "connect and disconnect" workflow, keeping all persistent data and code on Google Drive. 

Please refer to the [INSTRUCTIONS.md](INSTRUCTIONS.md) file for a step-by-step guide on how to set up the Colab environment, sync your GitHub repository, and run the CUDA code.

### Local Development & VSCode IntelliSense

To get proper syntax highlighting, tab autocompletion, and IntelliSense in VSCode, your IDE needs a valid C++/CUDA build environment (so that Meson can successfully run and generate the necessary compiler commands). 

If you cannot natively run `meson setup` on your host machine (for example, if you don't have a local CUDA toolkit and compiler installed), **you will not have IntelliSense**. 

To solve this and get the best development experience, we invite you to follow **both** options outlined in the [INSTRUCTIONS.md](INSTRUCTIONS.md):
1. **Option 1 (Google Colab):** Use this for heavy lifting and running your intense GPU workloads.
2. **Option 2 (WSL Fedora 44):** Set this up locally. Even if you only use it to successfully run `meson setup` from WSL, doing so will generate the build files VSCode needs to provide you with full IntelliSense and autocompletion right in your editor!

# LICENSE
This project is licensed under the EUPL-1.2 License - see the [LICENSE](LICENSE) file for details