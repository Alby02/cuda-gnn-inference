#include "cuda_executor.cuh"
#include "demo.hpp"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"

#include <iostream>
#include <string>
#include <string_view>

namespace gnn {

int run_cuda(std::string_view graphPath, std::string_view featurePath) {
    const bool usesLoadedData = !graphPath.empty();
    auto workload = usesLoadedData
                        ? demo::loadCpuDemo(std::string(graphPath), std::string(featurePath))
                        : demo::makeCpuDemo();
    const auto nodeCount = workload.graph.getNumNodes();
    const auto inputDimension = workload.input.cols();

    CudaWorkspace workspace;
    const auto deviceGraph = workspace.uploadGraph(workload.graph);
    const auto deviceModel = workspace.uploadModel(workload.model);
    const auto deviceInput = workspace.uploadMatrix(workload.input);

    InferenceRuntime<CudaExecutor> runtime;
    const auto deviceResult = runtime.run(deviceGraph, deviceModel, deviceInput, workspace);
    const Matrix<HostBuffer<float>> result = workspace.downloadMatrix(deviceResult);

    std::cout << (usesLoadedData ? "Loaded-data CUDA output"
                                 : "CUDA GCN -> GraphSAGE -> GCN output")
              << " (nodes=" << nodeCount << ", input_features=" << inputDimension
              << ", output_features=" << result.cols() << "):\n";
    demo::printMatrix(result, std::cout);
    return 0;
}

} // namespace gnn
