#include "execution_modes.hpp"
#include "gnn_build_config.hpp"

#include <iostream>
#include <string_view>

namespace {

void print_usage(std::string_view program) {
    std::cout << "Usage: " << program << " <mode>\n\n"
              << "Available modes:\n"
              << "  sequential  Single-threaded CPU\n";

#if GNN_HAS_OPENMP
    std::cout << "  parallel    Multi-threaded CPU (OpenMP)\n";
#endif

#if GNN_HAS_CUDA
    std::cout << "  cuda        NVIDIA GPU (CUDA)\n";
#endif
}

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
        print_usage(argv[0]);
        return argc == 1 ? 0 : 1;
    }

    const std::string_view mode{argv[1]};
    if (mode == "--help" || mode == "-h") {
        print_usage(argv[0]);
        return 0;
    }
    if (mode == "sequential") {
        return gnn::run_sequential();
    }

#if GNN_HAS_OPENMP
    if (mode == "parallel") {
        return gnn::run_parallel();
    }
#endif

#if GNN_HAS_CUDA
    if (mode == "cuda") {
        return gnn::run_cuda();
    }
#endif

    std::cerr << "Unknown or unavailable mode: " << mode << "\n\n";
    print_usage(argv[0]);
    return 1;
}
