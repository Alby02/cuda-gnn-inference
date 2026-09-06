#pragma once
#include "../cli.hpp"
#include "../demo.hpp"
#include "loader.hpp"
#include "workload.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace gnn::io {
namespace fs = std::filesystem;

class Manifest {
public:
    explicit Manifest(const fs::path& path) : file_(path), path_(path) {
        if (!file_)
            throw std::runtime_error("Cannot open manifest: " + path.string());
    }
    // Python writes the fixed v1 field order; consume labels without revalidating them.
    void skip(std::size_t count = 1) {
        for (std::size_t i = 0; i < count; ++i)
            static_cast<void>(value());
    }
    std::string value() {
        std::string result;
        file_ >> std::quoted(result);
        return result;
    }
    std::string field() {
        skip();
        return value();
    }
    fs::path resolve(const std::string& value) const {
        return path_.parent_path() / fs::path(value);
    }

private:
    std::ifstream file_;
    fs::path path_;
};

inline HostModel loadModel(const fs::path& path) {
    Manifest m(path);
    m.skip(2); // GNN_MODEL 1
    const auto count = std::stoull(m.field());
    std::vector<HostModel::LayerVariant> layers;
    layers.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        m.skip(); // layer
        const auto kind = m.value(), activation = m.value();
        const auto neigh = m.value(), self = m.value(), bias = m.value();
        auto w = graph::GraphLoader::loadDenseMatrix(m.resolve(neigh).string());
        HostBuffer<float> b;
        if (bias != "-") {
            auto matrix = graph::GraphLoader::loadDenseMatrix(m.resolve(bias).string());
            b = HostBuffer<float>(matrix.size());
            std::copy_n(matrix.data(), matrix.size(), b.data());
        }
        if (kind == "GCN") {
            layers.emplace_back(layers::HostGCN(std::move(w), std::move(b),
                                                activation == "RELU"
                                                    ? layers::GCNActivationType::RELU
                                                    : layers::GCNActivationType::NONE));
        } else {
            auto ws = graph::GraphLoader::loadDenseMatrix(m.resolve(self).string());
            layers.emplace_back(layers::HostGraphSAGE(
                std::move(w), std::move(ws), std::move(b), layers::GraphSAGEAggregationType::MEAN,
                activation == "RELU" ? layers::GraphSAGEActivationType::RELU
                                     : layers::GraphSAGEActivationType::NONE));
        }
    }
    return HostModel(std::move(layers));
}

struct Workload {
    HostWorkload data;
    std::string provenance;
    std::string seed;
};

inline Workload loadWorkload(const RunOptions& o) {
    if (o.graph.empty()) {
        std::cout << "running demo\n";
        return {demo::makeCpuDemo(), "built-in fixture", "unspecified"};
    }
    return {{graph::GraphLoader::load(o.graph), graph::GraphLoader::loadDenseMatrix(o.features),
             loadModel(o.model)},
            "explicit input files",
            "unspecified"};
}
inline void saveMatrix(const fs::path& path, const layers::HostMatrix& matrix) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t rows = matrix.rows(), cols = matrix.cols();
    file.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    file.write(reinterpret_cast<const char*>(matrix.data()),
               static_cast<std::streamsize>(matrix.size() * sizeof(float)));
    if (!file)
        throw std::runtime_error("Cannot write output matrix: " + path.string());
}
} // namespace gnn::io
