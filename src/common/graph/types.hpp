#pragma once

#include <cstdint>
#include <cstddef>

namespace graph {

using NodeId = std::uint32_t;
using EdgeId = std::size_t;
using Weight = float;

enum class GraphType {
    Directed,
    Undirected
};

enum class StorageFormat {
    CSR,
    CSC,
    CSR_CSC
};

} // namespace graph
