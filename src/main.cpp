#include <iostream>
#include "cuda_kernels.h"

int main() {
    std::cout << "Hello World from CPU (C++)!" << std::endl;
    
    // Call the wrapper function to launch the CUDA kernel
    launchHelloFromGPU();
    
    return 0;
}
