# HeteroBCC

HeteroBCC directory contains the code for PHEM BCC.

## Prerequisites

Ensure you have the following installed:
- CUDA Toolkit (compatible with your GPU architecture)
- NVIDIA GPU with appropriate compute capability
- GNU Make and a compatible C++ compiler (e.g., `g++` or `nvcc`)

## Compilation

To compile the code, navigate to the project directory and run:
```sh
make -j$(nproc)
```
This will utilize all available CPU cores for faster compilation.

### Architecture-Specific Changes
The default Makefile is set to **sm_80** (Ampere architecture). If your GPU has a different compute capability (e.g., sm_70 for Volta, sm_86 for newer Ampere GPUs), update the **sm_80** flag in the Makefile accordingly:
```makefile
CXXFLAGS = -Xcompiler -pthread -std=c++17 -O3 -arch=sm_XX
```
Replace `sm_XX` with the appropriate compute capability for your GPU.

## Execution

Once compiled, run the executable with:
```sh
./heterobcc <filename> <gpu_share> <batch_size>
```
Two sample graphs are included:
- `artic_graph.txt`
- `input.txt`

Example usage:
```sh
./heterobcc artic_graph.txt 1 100
```

## Notes
- Make sure to check your GPU's compute capability using:
  ```sh
  nvcc --help | grep compute
  ```
- Performance may vary based on architecture and workload.

- If necessary, clean the build using:
  ```sh
  make clean
  ```
  This will remove the compiled `heterobcc` binary.

