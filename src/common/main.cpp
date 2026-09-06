#include "execution_modes.hpp"
#include "gnn_build_config.hpp"
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const auto options = gnn::parseOptions(argc, argv);
        if (options.help) {
            std::cout << "Usage: gnn --backend MODE [options]\n"
                      << "Modes: sequential"
#if GNN_HAS_OPENMP
                      << " parallel"
#endif
#if GNN_HAS_CUDA
                      << " cuda"
#endif
                      << "\n\nOptions:\n"
                      << "  --help, -h          Show this help\n"
                      << "  --backend MODE      Execution backend (default: sequential)\n"
                      << "  --graph FILE        Graph topology binary\n"
                      << "  --features FILE     Node-feature matrix\n"
                      << "  --model FILE        Model description and parameter paths\n"
                      << "  --warmups N         Warmup iterations (default: 1)\n"
                      << "  --repetitions N     Measured iterations (default: 10)\n"
                      << "  --output CSV        Benchmark results file\n"
                      << "  --embeddings FILE   Output embedding matrix\n"
#if GNN_HAS_OPENMP
                      << "\nOpenMP options:\n"
                      << "  --threads N         Worker threads (default: 1)\n"
#endif
#if GNN_HAS_CUDA
                      << "\nCUDA options:\n"
                      << "  --block-size N      Threads per block (default: 256)\n"
#endif
                      << "\nGraph, features and model are required together; if any is missing, "
                         "runs the full demo.\n";
            return 0;
        }
        if (options.backend == "sequential")
            return gnn::run_sequential(options);
#if GNN_HAS_OPENMP
        if (options.backend == "parallel")
            return gnn::run_parallel(options);
#endif
#if GNN_HAS_CUDA
        if (options.backend == "cuda")
            return gnn::run_cuda(options);
#endif
        throw std::invalid_argument("Unavailable backend in this build: " + options.backend);
    } catch (const std::exception& error) {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }
}
