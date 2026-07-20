# Environment Setup Instructions

## 1. Windows (MSYS2/UCRT64) - CPU Only
Since MSVC is not supported for this project, you must use MSYS2 on Windows, you will use scoop to manage the environment.
1. Install [Scoop](https://scoop.sh/):
2. Install MSYS2, python, meson, and ninja using Scoop:
   ```powershell
   scoop install msys2 python meson ninja
   ```
2. Open the **MSYS2 UCRT64** terminal, from powershell, run:
   ```powershell
   ucrt64
   ```
3. Update your package database and core packages:
   ```bash
   pacman -Syu
   ```
4. Install the required toolchain and dependencies:
   ```bash
   pacman -S mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-llvm-openmp
   ```
5. Set up and build the project:
   ```bash
   meson setup builddir
   meson compile -C builddir
   ```
6. Open the project in Visual Studio Code (or your preferred IDE) buy opening from the terminal:
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

## 4. Google Colab Workflow Instructions (raccomandade for GPU Only)

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

### Set Up repository in Google Drive (Executed Once)

1. Open the "InitialSetup" jupyterNotebook. (If you are using VSCode, connect to the remote Colab kernel)
2. Click the "Run All" button to execute all cells. This will:
    - Mount your Google Drive. (It requires an authorization step where you have to grant all permissions apparently, cherry-picking doesn't seem to work.)
    - Clone the repository into your Drive for persistent storage.
    - Set up the necessary the compiler and build environment for C++/CUDA.
3. You can disconnect after this step, as the repository will remain in your Google Drive.

### Run the Project (Connect Only When Needed)

1. Open the "Runner" jupyterNotebook. (If you are using VSCode, connect to the remote Colab kernel)
2. Click the "Run All" button to execute all cells. This will:
    - Mount your Google Drive.
    - Navigate to the cloned repository.
    - Pull the latest changes from GitHub.
    - Compile the C++/CUDA code.
    - Run the compiled program.
3. **Disconnect:** When finished, go to `Runtime -> Disconnect and delete runtime` to stop consuming compute resources. Or if you are using VSCode, open the command palette (`Ctrl+Shift+P`) and select `"Colab: Remove server"` to explicitly disconnect the IPkernel and stop the server.

#### Known Issues
- If you are using VSCode every time you disconnect from the kernel if you want to reconnect you will have close and reopen VSCode. This is a known issue with the Colab extension and is not related to this project.