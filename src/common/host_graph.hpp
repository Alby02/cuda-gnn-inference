#pragma once

#include "data/graph_csc.hpp"
#include "host_buffer.hpp"

#include <cstdint>

namespace graph {

using HostGraphCSC = GraphCSC<gnn::HostBuffer<std::uint64_t>, gnn::HostBuffer<float>>;

} // namespace graph
