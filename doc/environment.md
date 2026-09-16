# Environment Setup Instructions

## 1. Windows (MSYS2/UCRT64) - CPU Only
Since MSVC is not supported for this project, you must use MSYS2 on Windows, you will use scoop to manage the environment.
1. Install [Scoop](https://scoop.sh/).
2. Install MSYS2, Python, Meson, Ninja, and uv using Scoop:
   ```powershell
   scoop install msys2 python meson ninja uv
   ```
3. Open the **MSYS2 UCRT64** terminal from PowerShell:
   ```powershell
   ucrt64
   ```
4. Update your package database and core packages:
   ```bash
   pacman -Syu
   ```
5. Install the required toolchain and dependencies:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-toolchain
   ```
6. Set up the Python environment, then configure CUDA explicitly off and build:
   ```bash
   uv sync
   meson setup builddir -Dcuda=disabled
   meson compile -C builddir
   ./builddir/gnn.exe --backend sequential
   ./builddir/gnn.exe --backend parallel --threads 4
   ```
7. Open the project in Visual Studio Code (or your preferred IDE) from the terminal:
   ```bash
   code .
   ```
*(Note: CUDA is not required or supported on Windows for this project. The Meson build will safely skip the CUDA targets.)*

---

## 2. Linux Native Setup Instructions

### CPU Only
If you are using a Linux distribution, you can set up the environment natively, simply install the c++ compiler (g++), python (3.10 or higher), meson, ninja using your package manager, as well as OpenMP.

### Optional CUDA Support
If you have an NVIDIA GPU or simply want to have lsp support for CUDA, you will need to install the CUDA toolkit. You can follow the instructions on the [NVIDIA CUDA Toolkit downloads page](https://developer.nvidia.com/cuda-downloads) to install the toolkit for your specific Linux distribution.

After installation, ensure that you add the CUDA binaries to your `PATH` and the libraries to your `LD_LIBRARY_PATH` so the system can find `nvcc` by doing as follows:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

If you are encountering issues with `nvcc` not being being compatible with your system's GCC version, you can install an older version of GCC (e.g., GCC 15) and set it as the default compiler for `nvcc` in mason by running:

```bash
meson setup builddir -Dcuda_ccbindir='/usr/bin/g++-15'
```
---

## 3. WSL 
If you prefer to compile and run the code locally using WSL (Windows Subsystem for Linux) follow the instructions for Linux Native Setup Instructions above. The only requirements is if you want to use CUDA you must install Ubuntu as distro (required by the NVIDIA CUDA toolkit).

``` Powershell
wsl --install -d Ubuntu
```

---
## 4. Python Dependencies for Scripts Folder (F2.1 / F2.2)

The dataset converter (`scripts/converter.py`) and synthetic graph generator
(`scripts/synthetic_generator.py`) use a repository-local virtual environment managed by
[uv](https://docs.astral.sh/uv/). Create or synchronize `.venv` from `pyproject.toml` and
`uv.lock` with:

```bash
uv sync
```

Run the tools through that managed environment:

```bash
uv run python scripts/synthetic_generator.py --help
uv run python scripts/converter.py --help
```

### Note on NetworKit

The synthetic generator uses [NetworKit](https://networkit.github.io/) to scale graph
generation up to millions of nodes. `uv` normally installs a compatible binary wheel in `.venv`
without modifying the system Python environment. If NetworKit must be built from source, check
that a C++ compiler is discoverable on `PATH` (`g++ --version` or `clang++ --version`).

---

## 5. Google Colab Workflow Instructions (recommended for GPU runs)

Since Google Colab offers limited GPU time (especially on free tiers), it is highly recommended to use a "connect and disconnect" workflow. This means you will only connect to a GPU instance when you actually need to compile and run the CUDA code, and disconnect immediately after to conserve your compute units.

To make this workflow seamless, we use **Google Drive** to store the project files persistently, and **Git** to track changes and pull the code directly from GitHub.

### Development

If you are developing the code, you can do so locally on your machine. You can use any IDE or text editor of your choice. Once you have made changes, you can push them to GitHub. The Colab notebooks will pull the latest changes from GitHub when you run them.

### Two Ways to Run the Notebooks

This repository includes notebook files that you can use to run the code. There are two primary ways to execute them:

#### Method 1: Using the Google Colab Website
1. Download the notebooks file from this repository.
2. Upload it to Google Drive or open it directly in Google Colab.
3. Run the cells directly in your browser.
4. **Disconnect:** When finished, go to `Runtime -> Disconnect and delete runtime` to stop consuming compute resources.

#### Method 2: Using VSCode with the Colab Extension
1. Install the Google Colab extension in VSCode.
2. Connect to the remote IPkernel to bridge your local VSCode with Colab's backend.
3. Run the notebook cells right inside VSCode.
4. **CRITICAL DISCONNECT STEP:** When you are done, you MUST open the command palette (`Ctrl+Shift+P`) and select `"Colab: Remove server"`. This explicitly disconnects the IPkernel and stops the server. Simply closing VSCode might leave the server running in the background and drain your compute units!

### Run the Project (Connect Only When Needed)

1. Open [`Colab/Runner.ipynb`](../Colab/Runner.ipynb) and select a GPU runtime.
2. Click **Run all**. The notebook mounts Drive for persistent results, clones the source into
   `/content` for faster compilation, installs dependencies, builds all detected backends, and runs
   correctness checks plus a small synthetic benchmark.
3. The public-dataset and million-node sections are disabled by default. Enable them only when needed.
4. **Disconnect:** When finished, use `Runtime -> Disconnect and delete runtime`. In VS Code, run
   `Colab: Remove server` from the command palette.

#### Known Issues
- If you are using VSCode every time you disconnect from the kernel if you want to reconnect you will have close and reopen VSCode. This is a known issue with the Colab extension and is not related to this project.
