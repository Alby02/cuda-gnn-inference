#include "execution_modes.hpp"
#include "cuda_kernels.cuh"

#include <cuda_runtime.h>

#include <iostream>

namespace gnn {

int run_cuda() {
    std::cout << "Hello World from the CUDA GPU mode!\n" << std::flush;
    hello_from_gpu<<<1, 5>>>();

    const cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
        std::cerr << "CUDA execution failed: " << cudaGetErrorString(result) << '\n';
        return 1;
    }
    return 0;
}

} // namespace gnn
