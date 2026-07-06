project('cuda-gnn-inference', 'cpp', 'cuda',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20', 'cuda_std=c++20'])

# Define the executable
executable('gnn_engine',
           'src/main.cpp',
           'src/cuda_kernels.cu')
