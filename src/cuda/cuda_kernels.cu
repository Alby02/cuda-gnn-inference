#include "cuda_kernels.cuh"

__global__ void helloFromGPU() {
    // Note: printf is the standard way to print from inside a CUDA device kernel.
    // Standard C++ streams like std::cout are not supported in device code.
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}
