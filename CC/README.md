# Heterogeneous Spanning Tree (HAST)

## Overview
Heterogeneous Spanning Tree (HAST) can use both CPU and GPU for computation on large datasets.

---

## Prerequisites
1. **CUDA Toolkit**: Ensure that the CUDA toolkit is installed and properly configured.
2. **Supported GPUs**:
   - A100 (use `sm_80` in the Makefile)
   - L40 (use `sm_89` in the Makefile)
3. **C++ Compiler**: A modern C++ compiler supporting C++17 or later.
4. **Linux Environment**: Tested on Ubuntu (other distributions might work but are not guaranteed).

---

## Building the Project

### Step 1: Modify the Makefile
In the `Makefile`, set the architecture flag `sm_xx` to the appropriate value based on your GPU:
- For **A100**, set `sm_xx = 80`
- For **L40**, set `sm_xx = 89`

### Step 2: Build the Project
Run the following command to build the project:
```bash
make -j$(nproc)
```
This command compiles the code using all available CPU cores.

---

## Running the Application

### Command Syntax
```bash
./hast <datasets> <gpu_share> <batch_size>
```

### Parameters
1. **datasets**: Path to the input dataset. The application supports files in the formats:
   - `.edgelist`
   - `.egr`
2. **gpu_share**: The percentage of GPU memory to allocate for the computation (e.g., `0.8` for 80%).
3. **batch_size**: Size of the batch for processing edges.

### Example
```bash
./hast dataset/graph.edgelist 0.8 1024
```
This command runs HAST on the `graph.edgelist` dataset, allocating 80% of GPU memory and processing edges in batches of size 1024.

---

## Input Formats
- **Edge List (`.edgelist`)**: A plain text file where:
  1. The first line specifies the number of vertices and edges: `numVert numEdges`.
  2. Each subsequent line represents an edge in the format `u v`.
  3. **Duplicates**: For an edge `<u, v>`, the input must also include `<v, u>` to account for undirected graph representation.
- **EGR**: Efficient CST binary representation format for larger graphs.
  1. Egr First 8 Bytes number of nodes = n in long long
  2. Egr Second 8 Bytes number of edges = m in long long
  3. Egr Next (n+1)*8 bytes represerent the Index of csr array all in long long
  4. Eger next m*4 bytes represent the neighbours all in int 
  5. THe CSR needs tp be symmetric
---

## Notes
- Ensure that your dataset is in one of the supported formats.
- For best performance, adjust the `batch_size` parameter based on your system's memory and computational capabilities.

---

## Troubleshooting
- **Compilation Errors**: Double-check the CUDA version and the `sm_xx` setting in the Makefile.
- **Memory Errors**: Reduce `gpu_share` or `batch_size` if running out of memory.
- **Performance Issues**: Experiment with different `batch_size` values and monitor GPU utilization.

---
<!-- 
## Contact
For any issues or questions, feel free to reach me out. -->
