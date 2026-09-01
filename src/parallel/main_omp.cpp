#include "demo.hpp"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"
#include "executor.hpp"

#include <iostream>
#include <omp.h>
#include <utility>

namespace gnn {

int run_parallel() {
    auto workload = demo::makeCpuDemo();
    InferenceRuntime<ParallelExecutor> runtime;

    const Matrix<float> result =
        runtime.run(workload.graph, workload.model, std::move(workload.input));

    std::cout << "Parallel GCN -> GraphSAGE -> GCN output (up to " << omp_get_max_threads()
              << " OpenMP threads):\n";
    demo::printMatrix(result, std::cout);
    return 0;
}

} // namespace gnn
