#include "benchmark.hpp"
#include "cuda_executor.cuh"
#include "data/model_io.hpp"
#include "demo.hpp"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"
#include <iostream>
namespace gnn {
int run_cuda(const RunOptions& options) {
    const auto loadStart = bench::Clock::now();
    auto workload = io::loadWorkload(options);
    const double loadMs = bench::milliseconds(loadStart);
    const auto setupStart = bench::Clock::now();
    CudaWorkspace workspace;
    const auto preparation = workspace.prepare(workload.data);
    InferenceRuntime<CudaExecutor> runtime{CudaExecutor(static_cast<unsigned>(options.blockSize))};
    cuda::Timer timer;
    const double setupMs = bench::milliseconds(setupStart) - preparation.uploadMs;
    const auto measurements = bench::measure(
        options.warmups, options.repetitions, [&] { workspace.resetInput(); },
        [&] { runtime.run(workspace); }, timer);
    const auto downloadStart = bench::Clock::now();
    const auto& output = workspace.getOutput();
    const double downloadMs = bench::milliseconds(downloadStart);
    bench::Record record;
    record["load_ms"] = bench::number(loadMs);
    record["setup_ms"] = bench::number(setupMs);
    record["upload_ms"] = bench::number(preparation.uploadMs);
    record["download_ms"] = bench::number(downloadMs);
    record["workspace_bytes"] = std::to_string(workspace.capacityBytes());
    bench::writeResults(options.output, record, measurements);
    if (!options.embeddings.empty())
        io::saveMatrix(options.embeddings, output);
    demo::printMatrix(output, std::cout);
    return 0;
}
} // namespace gnn
