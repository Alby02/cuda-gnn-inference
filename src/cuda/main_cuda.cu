/*#include "cuda_executor.cuh"
#include "cuda_model.cuh"
#include "demo.hpp"
#include "device_graph.cuh"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"

#include <iostream>
#include <string>*/
#include <string_view>
//#include <utility>

namespace gnn {

int run_cuda(std::string_view graphPath, std::string_view featurePath) {
    /*const bool usesLoadedData = !graphPath.empty();
    auto workload = usesLoadedData
                        ? demo::loadCpuDemo(std::string(graphPath), std::string(featurePath))
                        : demo::makeCpuDemo();
    const auto nodeCount = workload.graph.getNumNodes();
    const auto inputDimension = workload.input.cols();

    auto deviceGraph = cuda::deviceUploadGraph(workload.graph);
    auto deviceModel = cuda::deviceUploadModel(workload.model);
    auto deviceInput = cuda::deviceUploadMatrix(workload.input);
    InferenceRuntime<CudaExecutor> runtime;
    auto deviceResult = runtime.run(deviceGraph, deviceModel, std::move(deviceInput));
    const Matrix<HostBuffer<float>> result = cuda::deviceDownloadMatrix(deviceResult);

    std::cout << (usesLoadedData ? "Loaded-data CUDA output"
                                 : "CUDA GCN -> GraphSAGE -> GCN output")
              << " (nodes=" << nodeCount << ", input_features=" << inputDimension
              << ", output_features=" << result.cols() << "):\n";
    demo::printMatrix(result, std::cout);*/
    return 0;
}

} // namespace gnn
