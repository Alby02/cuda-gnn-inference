project('cuda-gnn-inference', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20'])

inc_dir = include_directories('src/common')

# 1. Common Files
common_src = files('src/common/utils.cpp')

# 2. Single-threaded (Sequential) CPU part
seq_src = files('src/sequential/main_seq.cpp')
if seq_src.length() > 0
  executable('gnn_seq', common_src + seq_src,
    include_directories : inc_dir)
endif

# 3. Multi-threaded (OpenMP) CPU part
omp_src = files('src/parallel/main_omp.cpp')
omp_dep = dependency('openmp', required : false)
if omp_dep.found() and omp_src.length() > 0
  executable('gnn_omp', common_src + omp_src,
    dependencies : [omp_dep],
    include_directories : inc_dir)
endif

# 4. CUDA GPU part
cuda_src = files(['src/cuda/main_cuda.cu', 'src/cuda/cuda_kernels.cu'])
if add_languages('cuda', required : false, native : false)
  message('CUDA compiler (nvcc) found. Enabling CUDA builds.')
  if cuda_src.length() > 0
    executable('gnn_cuda', common_src + cuda_src,
      override_options : ['cuda_std=c++20'],
      include_directories : inc_dir)
  endif
else
  message('CUDA compiler (nvcc) not found. Skipping CUDA compilation.')
endif
