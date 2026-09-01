#include "cuda_kernels.cuh"

#include <cstdio>

__global__ void hello_from_gpu() {
    printf("Hello World from GPU thread %d!\n", threadIdx.x);
}
