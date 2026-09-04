#pragma once

#if defined(__CUDACC__)
#define GNN_HOST_DEVICE __host__ __device__
#else
#define GNN_HOST_DEVICE
#endif
