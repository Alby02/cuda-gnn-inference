#include "benchmark.hpp"
#include "data/model_io.hpp"
#include "demo.hpp"
#include "execution/runtime.hpp"
#include "execution_modes.hpp"
#include "executor.hpp"
#include <iostream>

namespace gnn {
int run_sequential(const RunOptions& options) {
    const auto loadStart = bench::Clock::now();
    auto workload = io::loadWorkload(options);
    const double loadMs = bench::milliseconds(loadStart);
    const auto setupStart = bench::Clock::now();
    InferenceRuntime<SequentialExecutor> runtime;
    CpuContext workspace;
    workspace.prepare(workload.data);
    const double setupMs = bench::milliseconds(setupStart);
    bench::CpuTimer timer;
    const auto measurements = bench::measure(
        options.warmups, options.repetitions, [&] { workspace.resetInput(); },
        [&] { runtime.run(workspace); }, timer);
    const auto& output = workspace.getOutput();
    bench::Record record;
    record["load_ms"] = bench::number(loadMs);
    record["setup_ms"] = bench::number(setupMs);
    record["upload_ms"] = "0";
    record["download_ms"] = "0";
    record["workspace_bytes"] = std::to_string(workspace.capacityBytes());
    bench::writeResults(options.output, record, measurements);
    if (!options.embeddings.empty())
        io::saveMatrix(options.embeddings, output);
    std::cout << "Compute mean " << measurements.meanMs << " ms; population stddev "
              << measurements.stddevMs << " ms\n";
    demo::printMatrix(output, std::cout);
    return 0;
}
} // namespace gnn
