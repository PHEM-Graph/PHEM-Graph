/******************************************************************************
* Functionality: GPU related Utility manager
 ******************************************************************************/

#ifndef CUDA_UTILITY_H
#define CUDA_UTILITY_H

//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------
extern bool checker;
extern bool g_verbose;
extern long maxThreadsPerBlock;

typedef long long ll;
typedef unsigned long long ull;

inline void check_for_error(cudaError_t error, const std::string& message, const std::string& file, int line) noexcept {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " << message << "\n";
        std::cerr << "CUDA Error description: " << cudaGetErrorString(error) << "\n";
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(err, msg) check_for_error(err, msg, __FILE__, __LINE__)

inline void CudaAssert(cudaError_t code, const char* expr, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " in " << file << " at line " << line << " during " << expr << std::endl;
        exit(code);
    }
}

// Define the CUCHECK macro
#define CUCHECK(x) CudaAssert(x, #x, __FILE__, __LINE__)

// Function to print available and total memory
inline void printMemoryInfo(const std::string& message) {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;
    std::cout << message << ": Used GPU memory: " 
        << used_db / (1024.0 * 1024.0) << " MB, Free GPU Memory: " 
        << free_db / (1024.0 * 1024.0) << " MB, Total GPU Memory: " 
        << total_db / (1024.0 * 1024.0) << " MB" << std::endl;
}

inline void cuda_init(int device) {
    // Set the CUDA device
    CUDA_CHECK(cudaSetDevice(device), "Unable to set device ");
    cudaGetDevice(&device); // Get current device

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    cudaFree(0);

    maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
}

inline bool CompareDeviceResults(int *host_data, int *device_data, int num_items, bool verbose, cudaStream_t stream) {
    int *device_data_host = new int[num_items];
    cudaMemcpyAsync(device_data_host, device_data, num_items * sizeof(int), cudaMemcpyDeviceToHost, stream);

    for (int i = 0; i < num_items; i++) {
        if (host_data[i] != device_data_host[i]) {
            if (verbose) {
                printf("Mismatch at %d: Host: %d, Device: %d\n", i, host_data[i], device_data_host[i]);
            }
            delete[] device_data_host;
            return false;
        }
    }

    delete[] device_data_host;
    return true;
}

template<typename T>
inline void DisplayDeviceArray(T *device_data, size_t num_items, cudaStream_t stream = 0) {
    // Allocate memory for host data
    T *host_data = new T[num_items];

    // Copy data from device to host using the specified stream (default is stream 0, which is the default stream)
    cudaMemcpyAsync(host_data, device_data, num_items * sizeof(T), cudaMemcpyDeviceToHost, stream);

    // Wait for the cudaMemcpyAsync to complete
    cudaDeviceSynchronize();

    // Display the data
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << "Data[" << i << "]: " << host_data[i] << "\n";
    }

    // Free host memory
    delete[] host_data;
}

template<typename T>
inline void DisplayDeviceArray(T *device_data, size_t num_items, const std::string& display_name, cudaStream_t stream = 0) {
    std::cout << "\n" << display_name << " starts" << "\n";

    // Allocate memory for host data
    T *host_data = new T[num_items];

    // Copy data from device to host using the specified stream (default is stream 0, which is the default stream)
    cudaMemcpyAsync(host_data, device_data, num_items * sizeof(T), cudaMemcpyDeviceToHost, stream);

    // Wait for the cudaMemcpyAsync to complete
    cudaDeviceSynchronize();

    // Display the data
    for (size_t i = 0; i < num_items; ++i) {
        std::cout << "[" << i << "]: " << host_data[i] << "\n";
    }

    // Free host memory
    delete[] host_data;
}

// Renamed and combined function to display edge list from device arrays
inline void DisplayDeviceEdgeList(const int *device_u, const int *device_v, size_t num_edges, cudaStream_t stream = 0) {
    std::cout << "\n" << "Edge List:" << "\n";
    int *host_u = new int[num_edges];
    int *host_v = new int[num_edges];
    cudaMemcpyAsync(host_u, device_u, num_edges * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_v, device_v, num_edges * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();
    for (size_t i = 0; i < num_edges; ++i) {
        std::cout << " Edge[" << i << "]: (" << host_u[i] << ", " << host_v[i] << ")" << "\n";
    }
    delete[] host_u;
    delete[] host_v;
}

inline void WriteDeviceEdgeList(const int *device_u, const int *device_v, size_t num_vert, size_t num_edges, cudaStream_t stream = 0) {
    std::cout <<"Started writing to file :\n";
    std::string filename = "temp.txt";
    std::ofstream outFile(filename);
    
    if(!outFile) {
        std::cerr <<"Unable to create file.\n";
        exit(0);
    }

    // std::cout << std::endl << "Edge List:" << std::endl;
    int *host_u = new int[num_edges];
    int *host_v = new int[num_edges];

    cudaMemcpyAsync(host_u, device_u, num_edges * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_v, device_v, num_edges * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();

    outFile << num_vert <<" " << num_edges << "\n";
    for (size_t i = 0; i < num_edges; ++i) {
        outFile << host_u[i] <<" " << host_v[i] << "\n";
    }
    delete[] host_u;
    delete[] host_v;

    std::cout <<"Writing to file over.\n";
}

inline void WriteParentArray(const int *d_parent, size_t num_vert, cudaStream_t stream = 0) {
    std::cout <<"Started writing to file :\n";
    std::string filename = "parent_array.txt";
    std::ofstream outFile(filename);
    
    if(!outFile) {
        std::cerr <<"Unable to create file.\n";
        exit(0);
    }

    // std::cout << std::endl << "Edge List:" << std::endl;
    int *h_parent = new int[num_vert];

    cudaMemcpyAsync(h_parent, d_parent, num_vert * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaDeviceSynchronize();

    for (size_t i = 0; i < num_vert; ++i) {
        outFile << h_parent[i] <<", ";
    }

    delete[] h_parent;

    std::cout <<"Writing to file over." << std::endl;
}

// Function to print edge list from std::vector
inline void DisplayEdgeList(const std::vector<int>& u, const std::vector<int>& v) {
    std::cout << "\n" << "Edge List:" << "\n";
    size_t num_edges = u.size();
    for (size_t i = 0; i < num_edges; ++i) {
        std::cout << " Edge[" << i << "]: (" << u[i] << ", " << v[i] << ")" << "\n";
    }
}

template <typename T>
__global__ 
void print_device_array_kernel(T* array, long size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index == 0) { 
        for (int i = 0; i < size; ++i) {
            printf("Array[%d] = %lld\n", i, static_cast<long long>(array[i]));
        }
    }
}

template <typename T>
inline void print_device_array(const T* arr, long size) {
    print_device_array_kernel<<<1, 1>>>(arr, size);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after print_device_array_kernel");
    std::cout << std::endl;
}

// Function to print the edge list
inline void print_edge_list(uint64_t* edge_list, long numEdges) {
    for (long i = 0; i < numEdges; ++i) {
        uint64_t edge = edge_list[i];
        int u = edge >> 32;          // Extract the upper 32 bits
        int v = edge & 0xFFFFFFFF;   // Extract the lower 32 bits
        std::cout << u << " " << v << std::endl;
    }
}

// CUDA kernel to print the edge list from device memory
template <typename T>
__global__ 
void print_device_edge_list_kernel(T* d_edge_list, long numEdges) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for(long i = 0; i < numEdges; ++i) {
            uint64_t edge = d_edge_list[i];
            int u = edge >> 32;          // Extract the upper 32 bits
            int v = edge & 0xFFFFFFFF;   // Extract the lower 32 bits
            printf("%ld: %d %d\n", i, u, v);
        }
    }
}
template <typename T>
inline void print_device_edges(T* d_edge_list, long numEdges) {
    print_device_edge_list_kernel<<<1,1>>>(d_edge_list, numEdges);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize");
}

// Declaration of kernel wrapper functions
void kernelPrintEdgeList(int* d_u_arr, int* d_v_arr, long numEdges, cudaStream_t stream);
void kernelPrintArray(int* d_arr, int numItems, cudaStream_t stream);
void kernelPrintCSRUnweighted(long* d_rowPtr, int* d_colIdx, int numRows, cudaStream_t stream);
#endif // CUDA_UTILITY_H