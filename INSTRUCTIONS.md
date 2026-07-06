# 1. Google Colab Workflow Instructions

Since Google Colab offers limited GPU time (especially on free tiers), it is highly recommended to use a "connect and disconnect" workflow. This means you will only connect to a GPU instance when you actually need to compile and run the CUDA code, and disconnect immediately after to conserve your compute units.

To make this workflow seamless, we use **Google Drive** to store the project files persistently, and **Git** to track changes and push to GitHub.

## Development

If you are developing the code, you can do so locally on your machine. You can use any IDE or text editor of your choice. Once you have made changes, you can push them to GitHub. The Colab notebooks will pull the latest changes from GitHub when you run them. In this phase you do not need to connect to Colab otherwise you will be consuming your compute units unnecessarily. Connect to Colab only when you want to compile and run the code, then disconnect immediately after to conserve your compute units.

## Two Ways to Run the Notebooks

This repository includes notebook files that you can use to run the code. There are two primary ways to execute them:

### Method 1: Using the Google Colab Website
1. Download the notebooks file from this repository.
2. Upload it to Google Drive or open it directly in Google Colab.
3. Run the cells directly in your browser.
4. **Disconnect:** When finished, go to `Runtime -> Disconnect and delete runtime` to stop consuming compute resources.

### Method 2: Using VSCode with the Colab Extension (Recommended for Local Feel)
1. Install the Google Colab extension in VSCode.
2. Connect to the remote IPkernel to bridge your local VSCode with Colab's backend.
3. Run the notebook cells right inside VSCode.
4. **CRITICAL DISCONNECT STEP:** When you are done, you MUST open the command palette (`Ctrl+Shift+P`) and select `"Colab: Remove server"`. This explicitly disconnects the IPkernel and stops the server. Simply closing VSCode might leave the server running in the background and drain your compute units!

## Set Up repository in Google Drive (Executed Once)

1. Open the "InitialSetup" jupyterNotebook. (If you are using VSCode, connect to the remote Colab kernel)
2. Click the "Run All" button to execute all cells. This will:
    - Mount your Google Drive. (It requires an authorization step where you have to grant all permissions apparently, cherry-picking doesn't seem to work.)
    - Clone the repository into your Drive for persistent storage.
    - Set up the necessary the compiler and build environment for C++/CUDA.
3. You can disconnect after this step, as the repository will remain in your Google Drive.

## Run the Project (Connect Only When Needed)

1. Open the "Runner" jupyterNotebook. (If you are using VSCode, connect to the remote Colab kernel)
2. Click the "Run All" button to execute all cells. This will:
    - Mount your Google Drive.
    - Navigate to the cloned repository.
    - Pull the latest changes from GitHub.
    - Compile the C++/CUDA code.
    - Run the compiled program.
3. **Disconnect:** When finished, go to `Runtime -> Disconnect and delete runtime` to stop consuming compute resources. Or if you are using VSCode, open the command palette (`Ctrl+Shift+P`) and select `"Colab: Remove server"` to explicitly disconnect the IPkernel and stop the server.


### Known Issues
- If you are using VSCode every time you disconnect from the kernel if you want to reconnect you will have close and reopen VSCode. This is a known issue with the Colab extension and is not related to this project.

# 2. WSL Fedora 44 Setup Instructions

If you prefer to compile and run the code locally using WSL (Windows Subsystem for Linux) with Fedora 44, you need to work around the fact that Fedora 44 ships with GCC 16, which is too new for current versions of CUDA (`nvcc`).

Follow these steps to set up a compatible environment and compile the project:

### Step 1: Install the Compatible Compiler
Fedora provides older compilers for compatibility. Install GCC 15:
```bash
sudo dnf install gcc15 gcc15-c++
```

### Step 2: Add CUDA to your Environment
After you installed CUDA you must add it to your `PATH` so the system can find `nvcc`:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Configure Meson
Because `nvcc` does its compiler sanity checks during the initial setup, you must explicitly tell Meson to use the `g++-15` compiler during the setup phase. Run this from the project root:
```bash
meson setup builddir -Dcuda_ccbindir='/usr/bin/g++-15'
```

### Step 4: Compile and Run
Now you can compile the project normally:
```bash
meson compile -C builddir
```

Finally, run the compiled executable (This is useful only if you have an NVIDIA GPU and the necessary drivers installed otherwise it will skip the CUDA parts or fail to run entirely, skip this step if you plan to run the code on Colab):
```bash
./builddir/gnn_engine
```