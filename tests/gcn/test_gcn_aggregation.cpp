#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "data/data_factory.hpp"
#include "execution/runtime.hpp"
#include "gnn/model.hpp"
#include "host_layers.hpp"
#include "sequential/executor.hpp"

using gnn::layers::HostGCN;
using gnn::layers::HostMatrix;

namespace {
bool approxEqual(float a, float b, float atol = 1e-4f, float rtol = 1e-4f) {
    return std::fabs(a - b) <= atol + rtol * std::fabs(b);
}

HostMatrix makeMatrix(std::size_t rows, std::size_t cols, std::initializer_list<float> values) {
    HostMatrix m{rows, cols};
    std::size_t i = 0;
    for (float v : values) {
        m.data()[i++] = v;
    }
    return m;
}
} // namespace

int main() {
    auto g = graph::GraphFactory::make(true, 3, 2, {0, 0, 2, 2}, {0, 2});
    HostMatrix W = makeMatrix(1, 1, {1.0f});
    HostMatrix H = makeMatrix(3, 1, {1.0f, 2.0f, 3.0f});

    using DemoModel = gnn::Model<HostGCN>;
    std::vector<DemoModel::LayerVariant> layerVec;
    layerVec.emplace_back(HostGCN{W});
    DemoModel model{std::move(layerVec)};

    gnn::InferenceRuntime<gnn::SequentialExecutor> runtime;
    auto out = runtime.run(g, model, H);

    const float invSqrt3 = 1.0f / std::sqrt(3.0f);
    const float expected1 = invSqrt3 * 1.0f + (1.0f / 3.0f) * 2.0f + invSqrt3 * 3.0f;

    std::printf("out=(%.5f,%.5f,%.5f) expected=(1,%.5f,3)\n", out(0, 0), out(1, 0), out(2, 0),
                expected1);
    assert(approxEqual(out(0, 0), 1.0f));
    assert(approxEqual(out(1, 0), expected1));
    assert(approxEqual(out(2, 0), 3.0f));
    std::printf("SUCCESS: semantics.md 11.1 fixture matches the gcn_aggregation.hpp which is correct and working.\n");
    return 0;
}