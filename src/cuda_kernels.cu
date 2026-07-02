#include <iostream>
#include <cstdio>
#include "cuda_kernels.h"

__global__ void helloFromGPU() {
    // Note: printf is the standard way to print from inside a CUDA device kernel.
    // Standard C++ streams like std::cout are not supported in device code.
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

void launchHelloFromGPU() {
    std::cout << "Launching CUDA kernel..." << std::endl;

    // Launch a kernel with 1 block of 5 threads
    helloFromGPU<<<1, 5>>>();
    
    // Wait for GPU to finish before exiting
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel finished execution." << std::endl;
}
