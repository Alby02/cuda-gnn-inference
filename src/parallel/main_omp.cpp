#include "demo.hpp"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"
#include "executor.hpp"

#include <iostream>
#include <omp.h>
#include <string>
#include <string_view>
#include <utility>

namespace gnn {

int run_parallel(std::string_view graphPath, std::string_view featurePath) {
    const bool usesLoadedData = !graphPath.empty();
    auto workload = usesLoadedData
                        ? demo::loadCpuDemo(std::string(graphPath), std::string(featurePath))
                        : demo::makeCpuDemo();
    const auto nodeCount = workload.graph.getNumNodes();
    const auto inputDimension = workload.input.cols();
    InferenceRuntime<ParallelExecutor> runtime;

    const Matrix<HostBuffer<float>> result =
        runtime.run(workload.graph, workload.model, std::move(workload.input));

    std::cout << (usesLoadedData ? "Loaded-data parallel output"
                                 : "Parallel GCN -> GraphSAGE -> GCN output")
              << " (nodes=" << nodeCount << ", input_features=" << inputDimension
              << ", output_features=" << result.cols() << ", up to " << omp_get_max_threads()
              << " OpenMP threads):\n";
    demo::printMatrix(result, std::cout);
    return 0;
}

} // namespace gnn
