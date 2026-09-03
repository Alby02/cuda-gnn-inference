#pragma once

#include "data/matrix.hpp"
#include "gnn/layers/gcn.hpp"
#include "gnn/layers/graph_sage.hpp"
#include "host_buffer.hpp"

namespace gnn::layers {

using HostMatrix = Matrix<HostBuffer<float>>;
using HostGCN = GCNLayer<HostMatrix, HostBuffer<float>>;
using HostGraphSAGE = GraphSAGELayer<HostMatrix, HostBuffer<float>>;

} // namespace gnn::layers
