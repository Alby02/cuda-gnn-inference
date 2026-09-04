#pragma once

#include "data/matrix.hpp"
#include "device_buffer.cuh"

#include <cstddef>

namespace gnn::cuda {

void launchRowByColumn(Matrix<DeviceBuffer<float>> left, Matrix<DeviceBuffer<float>> right,
                       Matrix<DeviceBuffer<float>> output);

} // namespace gnn::cuda
