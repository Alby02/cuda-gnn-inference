#include "utils.hpp"
#include <omp.h>
#include <iostream>

int main() {
    print_hello("Multi-threaded CPU (OpenMP)");
    
    #pragma omp parallel
    {
        #pragma omp single
        std::cout << "Number of OpenMP threads: " << omp_get_num_threads() << std::endl;
    }
    return 0;
}
