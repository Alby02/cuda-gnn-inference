#include <iostream>
#include <omp.h>

int main() {
    std::cout << "Multi-threaded CPU (OpenMP)\n";

#pragma omp parallel
    {
#pragma omp single
        std::cout << "Number of OpenMP threads: " << omp_get_num_threads() << std::endl;
    }
    return 0;
}
