project('cuda-gnn-inference', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3', 'cpp_std=c++20', 'cpp_eh=default'])

inc_dir = include_directories('src/common')

# The sequential implementation is always available. Optional implementations
# and their dependencies are added to this same executable when detected.
gnn_sources = files(
  'src/common/main.cpp',
  'src/sequential/main_seq.cpp',
)
gnn_dependencies = []

omp_dep = dependency('openmp', required : false)
have_openmp = omp_dep.found()
if have_openmp
  message('OpenMP found. Enabling the parallel CLI mode.')
  gnn_sources += files('src/parallel/main_omp.cpp')
  gnn_dependencies += omp_dep
else
  message('OpenMP not found. The parallel CLI mode will be unavailable.')
endif

have_cuda = add_languages('cuda', required : false, native : false)
if have_cuda
  message('CUDA compiler (nvcc) found. Enabling the CUDA CLI mode.')
  gnn_sources += files(
    'src/cuda/main_cuda.cu',
    'src/cuda/cuda_kernels.cu',
  )
else
  message('CUDA compiler (nvcc) not found. The CUDA CLI mode will be unavailable.')
endif

build_config = configuration_data()
build_config.set10('GNN_HAS_OPENMP', have_openmp)
build_config.set10('GNN_HAS_CUDA', have_cuda)
configure_file(
  output : 'gnn_build_config.hpp',
  configuration : build_config,
)

executable('gnn', gnn_sources,
  dependencies : gnn_dependencies,
  include_directories : inc_dir,
  override_options : ['cuda_std=c++20'],
)
