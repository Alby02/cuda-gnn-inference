#include "execution_modes.hpp"
#include "gnn_build_config.hpp"

#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void print_usage(std::string_view program) {
    std::cout << "Usage: " << program << " <mode> [graph.bin_graph features.bin_matrix]\n\n"
              << "Available modes:\n"
              << "  sequential  Single-threaded CPU\n";

#if GNN_HAS_OPENMP
    std::cout << "  parallel    Multi-threaded CPU (OpenMP)\n";
#endif

#if GNN_HAS_CUDA
    std::cout << "  cuda        NVIDIA GPU (CUDA)\n";
#endif

    std::cout << "\nWithout data paths, the built-in two-node demo is used.\n"
              << "Every available mode accepts the optional graph and feature paths.\n";
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2 && argc != 4) {
        print_usage(argv[0]);
        return argc == 1 ? 0 : 1;
    }

    const std::string_view mode{argv[1]};
    if (mode == "--help" || mode == "-h") {
        print_usage(argv[0]);
        return 0;
    }
    const std::string_view graphPath = argc == 4 ? argv[2] : "";
    const std::string_view featurePath = argc == 4 ? argv[3] : "";

    try {
        if (mode == "sequential") {
            return gnn::run_sequential(graphPath, featurePath);
        }

#if GNN_HAS_OPENMP
        if (mode == "parallel") {
            return gnn::run_parallel(graphPath, featurePath);
        }
#endif

#if GNN_HAS_CUDA
        if (mode == "cuda") {
            return gnn::run_cuda(graphPath, featurePath);
        }
#endif
    } catch (const std::exception& error) {
        std::cerr << "Execution failed: " << error.what() << '\n';
        return 1;
    }

    std::cerr << "Unknown or unavailable mode: " << mode << "\n\n";
    print_usage(argv[0]);
    return 1;
}
