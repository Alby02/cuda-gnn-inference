#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "data/data_factory.hpp"
#include "execution/runtime.hpp"
#include "gnn/model.hpp"
#include "host_layers.hpp"
#include "parallel/executor.hpp"
#include "sequential/executor.hpp"

using gnn::layers::HostGCN;
using gnn::layers::HostMatrix;

namespace {
bool approxEqual(float a, float b, float atol = 1e-4f, float rtol = 1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}
} // 

int main() {
    constexpr std::uint64_t n = 2000;
    std::vector<std::uint64_t> rowInd;
    rowInd.reserve(n - 1);
    for (std::uint64_t u = 1; u < n; ++u) {
        rowInd.push_back(u);
    }
    std::vector<std::uint64_t> colPtr(n + 1, rowInd.size());
    colPtr[0] = 0;
    auto g = graph::GraphFactory::make(true, n, rowInd.size(), colPtr, rowInd);

    HostMatrix W{1, 1};
    W.data()[0] = 1.0f;
    HostMatrix H{static_cast<std::size_t>(n), 1};
    for (std::size_t i = 0; i < H.size(); ++i) {
        H.data()[i] = 1.0f;
    }

    using DemoModel = gnn::Model<HostGCN>;
    std::vector<DemoModel::LayerVariant> layerVec;
    layerVec.emplace_back(HostGCN{W});
    DemoModel model{std::move(layerVec)};

    gnn::InferenceRuntime<gnn::SequentialExecutor> seqRuntime;
    auto seqOut = seqRuntime.run(g, model, H);

    struct Config {
        const char* name;
        int threads;
        omp_sched_t kind;
        int chunk;
    };
    const std::vector<Config> configs = {
        {"static, default", 4, omp_sched_static, 0},
        {"dynamic, chunk=8", 4, omp_sched_dynamic, 8},
        {"guided, chunk=1", 4, omp_sched_guided, 1},
    };

    for (const auto& cfg : configs) {
        // Build a fresh executor with its own gcnState() per config (the
        // OmpConfig lives inside GCNAggregationStateParallel).
        gnn::ParallelExecutor executor;
        executor.gcnState().setNumThreads(cfg.threads);
        executor.gcnState().setSchedule(cfg.kind, cfg.chunk);

        gnn::InferenceRuntime<gnn::ParallelExecutor> runtime(executor);
        auto out = runtime.run(g, model, H);

        std::size_t mismatches = 0;
        for (std::uint64_t v = 0; v < n; ++v) {
            if (!approxEqual(out(v, 0), seqOut(v, 0))) {
                ++mismatches;
            }
        }
        std::printf("[%-16s] mismatches=%zu (hub=%.5f leaf=%.5f)\n", cfg.name, mismatches,
                    out(0, 0), out(1, 0));
        assert(mismatches == 0);
    }

    std::printf("SUCCESS: OpenMP GCN aggregation matches sequential baseline.\n");
    return 0;
}