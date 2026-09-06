#pragma once

#include "cpu_context.hpp"
#include "execution/executor.hpp"
#include "gnn/layer.hpp"

#include <cstddef>
#include <stdexcept>

namespace gnn {

class ParallelExecutor {
public:
    using WorkspaceType = CpuContext;
    using BufferType = WorkspaceType::BufferType;
    using WeightType = Matrix<HostBuffer<float>>;

    using BiasType = HostBuffer<float>;

    void rowByColumn(const BufferType& left, const WeightType& right, BufferType& output) const {
        if (left.cols() != right.rows())
            throw std::invalid_argument("Incompatible matrix product shapes.");
        output.setShape(left.rows(), right.cols());
        forRows(left.rows(), [&](std::size_t row) {
            for (std::size_t col = 0; col < right.cols(); ++col) {
                float value = 0;
                for (std::size_t k = 0; k < left.cols(); ++k)
                    value += left(row, k) * right(k, col);
                output(row, col) = value;
            }
        });
    }
    // Elementwise operations permit exact in-place aliasing.
    void add(const BufferType& left, const BufferType& right, BufferType& output) const {
        output.setShape(left.rows(), left.cols());
        forRows(left.size(),
                [&](std::size_t i) { output.data()[i] = left.data()[i] + right.data()[i]; });
    }
    void biasAdd(BufferType& matrix, const BiasType& bias) const {
        forRows(matrix.size(),
                [&](std::size_t i) { matrix.data()[i] += bias.data()[i % matrix.cols()]; });
    }
    void relu(BufferType& matrix) const {
        forRows(matrix.size(), [&](std::size_t i) {
            if (matrix.data()[i] < 0)
                matrix.data()[i] = 0; // Preserve NaN for verification.
        });
    }

    void aggregateNeighbors(const auto& graph, const BufferType& in_features,
                            BufferType& out_aggregated, auto agg_type) const {

        const std::size_t num_nodes = in_features.rows();
        const std::size_t num_features = in_features.cols();
        const int layer_num = 2;
        const int sample[] = {25, 10};

        out_aggregated.setShape(num_nodes, num_features);
        const int max_samples = sample[layer_num];

#pragma omp parallel for schedule(dynamic, 32)
        for (std::size_t u = 0; u < num_nodes; ++u) {

            auto neighbors = graph.get_neighbors(u);
            const std::size_t deg = neighbors.size();

            std::size_t actual_samples = (max_samples > 0 && max_samples < deg) ? max_samples : deg;

            for (std::size_t f = 0; f < num_features; ++f) {
                out_aggregated(u, f) = 0.0f;
            }

            if (actual_samples == 0) {
                continue;
            }

            if (max_samples > 0 && deg > max_samples) {
                thread_local std::mt19937 rng(1337);
                std::vector<std::size_t> sampled_indices(deg);
                for (std::size_t i = 0; i < deg; ++i)
                    sampled_indices[i] = i;

                std::partial_sort(sampled_indices.begin(), sampled_indices.begin() + actual_samples,
                                  sampled_indices.end(),
                                  [&](std::size_t a, std::size_t b) { return rng() % 2 == 0; });

                for (std::size_t i = 0; i < actual_samples; ++i) {
                    std::size_t v = neighbors[sampled_indices[i]];
                    for (std::size_t f = 0; f < num_features; ++f) {
                        out_aggregated(u, f) += in_features(v, f);
                    }
                }
            } else {
                for (std::size_t i = 0; i < actual_samples; ++i) {
                    std::size_t v = neighbors[i];
                    for (std::size_t f = 0; f < num_features; ++f) {
                        out_aggregated(u, f) += in_features(v, f);
                    }
                }
            }
            for (std::size_t f = 0; f < num_features; ++f) {
                out_aggregated(u, f) /= actual_samples;
            }
        }
    }

private:
    template <typename F> static void forRows(std::size_t count, F operation) {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < count; ++i)
            operation(i);
    }
};

static_assert(Executor<ParallelExecutor>);

} // namespace gnn
