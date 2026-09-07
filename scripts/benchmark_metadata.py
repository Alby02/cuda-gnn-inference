"""Descriptive metadata for native benchmarks, collected outside the timed process."""
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess
from workload_io import Tokens


def matrix_shape(path):
    with Path(path).open('rb') as stream:
        return struct.unpack('<QQ', stream.read(16))


def workload_metadata(graph_path, feature_path, model_path):
    # Only read headers and model descriptions, not graph arrays or weight values.
    with Path(graph_path).open('rb') as stream:
        nodes, edges, directed, _ = struct.unpack('<QQBB', stream.read(18))
    widths = [matrix_shape(feature_path)[1]]
    model = Tokens(model_path)
    model.skip(2)
    count = int(model.field())
    kinds = []
    for _ in range(count):
        model.skip()  # layer
        kinds.append(model.value())
        model.skip()  # activation
        widths.append(matrix_shape(model.resolve(model.value()))[1])
        model.skip(2)  # self weights and bias
    return dict(graph_path=str(Path(graph_path).resolve()),
                features_path=str(Path(feature_path).resolve()),
                model_path=str(Path(model_path).resolve()),
                model_types=';'.join(kinds), feature_widths=';'.join(map(str, widths)),
                nodes=nodes, stored_edges=edges, layers=count,
                orientation='directed' if directed else 'undirected', dtype='float32',
                seed='unspecified', provenance='explicit input files')


def environment_metadata(native, include_cuda):
    hardware = dict(system=platform.system(), arch=platform.machine(),
                    logical_cpus=os.cpu_count(), cpu=platform.processor())
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith('model name'):
                hardware['cpu'] = line.split(':', 1)[1].strip()
                break
    for name in ('OMP_PROC_BIND', 'OMP_PLACES', 'CUDA_VISIBLE_DEVICES'):
        if name in os.environ:
            hardware[name] = os.environ[name]
    software = dict(python=platform.python_version(), build='release', compiler='unspecified')
    # Use the executable's adjacent Meson build information, not the compiler on PATH.
    compiler_info = Path(native).resolve().parent / 'meson-info' / 'intro-compilers.json'
    if compiler_info.exists():
        software['compiler'] = json.loads(compiler_info.read_text(encoding='utf-8'))['host']
        software['compiler_info'] = str(compiler_info)
    smi = shutil.which('nvidia-smi') if include_cuda else None
    if smi:
        result = subprocess.run([smi, '--query-gpu=index,uuid,name,driver_version', '--format=csv,noheader'],
                                capture_output=True, text=True)
        if result.returncode == 0:
            # Inventory, not a guess about CUDA's selected device ordinal.
            hardware['gpu_inventory'] = result.stdout.strip().splitlines()
    return dict(hardware=json.dumps(hardware), software=json.dumps(software))


def run_metadata(args, model_metadata, environment, backend, setting, command):
    return dict(model_metadata, **environment, schema_version=1, runner='native',
                backend=backend, strategy='destination',
                threads=setting if backend == 'parallel' else 1 if backend == 'sequential' else '',
                schedule='static' if backend == 'parallel' else '',
                block_size=setting if backend == 'cuda' else '',
                warmups=args.warmups, repetitions=args.repetitions,
                command=json.dumps(command),
                memory_policy='workspace capacity only; peak memory unmeasured',
                timing_policy='layer_execution=GCN_transform_then_aggregate; GraphSAGE_weighted_mean_then_transform; load/setup/upload excluded; '
                              'end_to_end=input reset+compute+synchronization; final download separate')
