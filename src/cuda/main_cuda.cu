#include "utils.hpp"
#include "cuda_kernels.cuh"

int main() {
    print_hello("CUDA GPU");
    // Launch a kernel with 1 block of 5 threads
    helloFromGPU<<<1, 5>>>();
    cudaDeviceSynchronize();
    return 0;
}
