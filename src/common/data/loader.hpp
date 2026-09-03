#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "data_factory.hpp"
#include "matrix.hpp"

namespace graph {

#pragma pack(push, 1)
struct GraphHeader {
    std::uint64_t numNodes{0};
    std::uint64_t numEdges{0};
    std::uint8_t isDirected{0};
    std::uint8_t hasWeights{0};
};
#pragma pack(pop)

static_assert(sizeof(GraphHeader) == 18, "GraphHeader must match Python '<QQBB' layout");

class GraphLoader {
public:
    static HostGraphCSC load(const std::string& filepath) {
        auto raw = readBinaryGraph(filepath);
        try {
            return GraphFactory::make(raw.header.isDirected != 0, raw.header.numNodes,
                                      raw.header.numEdges, std::move(raw.colPtr),
                                      std::move(raw.rowInd), std::move(raw.weights));
        } catch (const std::invalid_argument& error) {
            throw std::runtime_error("GraphLoader: invalid graph in '" + filepath +
                                     "': " + error.what());
        }
    }

    // Node and edge features are execution data stored as independent dense matrices.
    static gnn::Matrix<gnn::HostBuffer<float>> loadDenseMatrix(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("GraphLoader: could not open matrix file '" + filepath + "'");
        }

        std::uint64_t rows = 0;
        std::uint64_t cols = 0;
        readExact(file, &rows, 1, filepath, "matrix row count");
        readExact(file, &cols, 1, filepath, "matrix column count");
        const auto elementCount = checkedProduct(rows, cols, filepath, "matrix shape");

        gnn::Matrix<gnn::HostBuffer<float>> matrix{checkedSize(rows, filepath, "matrix rows"),
                                                   checkedSize(cols, filepath, "matrix columns")};
        readExact(file, matrix.data(), elementCount, filepath, "matrix values");
        rejectTrailingBytes(file, filepath);
        return matrix;
    }

private:
    // struct used internally to store raw data of a .bin_graph file from readBinaryGraph() to
    // load()
    struct LoadedRawData {
        GraphHeader header;
        std::vector<std::uint64_t> colPtr;
        std::vector<std::uint64_t> rowInd;
        std::vector<float> weights;
    };

    static LoadedRawData readBinaryGraph(const std::string& filepath) {
        std::ifstream file(filepath, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("GraphLoader: could not open graph file '" + filepath + "'");
        }

        GraphHeader header;
        readExact(file, &header, 1, filepath, "graph header");
        if (header.isDirected > 1 || header.hasWeights > 1) {
            throw std::runtime_error("GraphLoader: invalid boolean flag in '" + filepath + "'");
        }

        const auto colPtrCount = checkedAddOne(header.numNodes, filepath, "node count");
        const auto edgeCount = checkedSize(header.numEdges, filepath, "edge count");

        std::vector<std::uint64_t> colPtr(colPtrCount);
        readExact(file, colPtr.data(), colPtr.size(), filepath, "column pointers");

        std::vector<std::uint64_t> rowInd(edgeCount);
        readExact(file, rowInd.data(), rowInd.size(), filepath, "row indices");

        std::vector<float> weights;
        if (header.hasWeights != 0) {
            weights.resize(edgeCount);
            readExact(file, weights.data(), weights.size(), filepath, "edge weights");
        }

        rejectTrailingBytes(file, filepath);
        return {header, std::move(colPtr), std::move(rowInd), std::move(weights)};
    }

    static std::size_t checkedSize(std::uint64_t value, const std::string& filepath,
                                   const char* field) {
        if (value > std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("GraphLoader: " + std::string(field) + " is too large in '" +
                                     filepath + "'");
        }
        return static_cast<std::size_t>(value);
    }

    static std::size_t checkedAddOne(std::uint64_t value, const std::string& filepath,
                                     const char* field) {
        if (value == std::numeric_limits<std::uint64_t>::max()) {
            throw std::runtime_error("GraphLoader: " + std::string(field) + " overflows in '" +
                                     filepath + "'");
        }
        return checkedSize(value + 1, filepath, field);
    }

    static std::size_t checkedProduct(std::uint64_t lhs, std::uint64_t rhs,
                                      const std::string& filepath, const char* field) {
        if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
            throw std::runtime_error("GraphLoader: " + std::string(field) + " overflows in '" +
                                     filepath + "'");
        }
        return checkedSize(lhs * rhs, filepath, field);
    }

    template <typename T>
    static void readExact(std::ifstream& file, T* destination, std::size_t count,
                          const std::string& filepath, const char* section) {
        if (count == 0) {
            return;
        }
        if (count >
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()) / sizeof(T)) {
            throw std::runtime_error("GraphLoader: " + std::string(section) + " is too large in '" +
                                     filepath + "'");
        }
        file.read(reinterpret_cast<char*>(destination),
                  static_cast<std::streamsize>(count * sizeof(T)));
        if (!file) {
            throw std::runtime_error("GraphLoader: truncated " + std::string(section) + " in '" +
                                     filepath + "'");
        }
    }

    static void rejectTrailingBytes(std::ifstream& file, const std::string& filepath) {
        if (file.peek() != std::ifstream::traits_type::eof()) {
            throw std::runtime_error("GraphLoader: trailing bytes in '" + filepath + "'");
        }
    }
};

} // namespace graph
