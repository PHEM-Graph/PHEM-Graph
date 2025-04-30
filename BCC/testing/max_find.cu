#include <cub/cub.cuh> // Include CUB library
#include <iostream>

// CUDA error-checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << " " \
                      << cudaGetErrorString(err) << std::endl;                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Function to find the maximum value using CUB
int findMax(const int *d_in, int num_items) {
    // Allocate memory for the result on the device
    int *d_max = nullptr;
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(int)));

    // Determine temporary device storage requirements
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);

    // Allocate temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    // Run max-reduction
    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);

    // Copy the result back to the host
    int h_max;
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_temp_storage));
    CUDA_CHECK(cudaFree(d_max));

    return h_max; // Return the maximum value
}

int main() {
    // Declare and initialize host input
    const int num_items = 7;                   // Number of elements
    int h_in[num_items] = {18, 6, 7, 15, 3, 0, 9}; // Host input array

    // Allocate and initialize device memory
    int *d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(int) * num_items));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, sizeof(int) * num_items, cudaMemcpyHostToDevice));

    // Call the findMax function
    int max_value = findMax(d_in, num_items);

    // Print the result
    std::cout << "Max value: " << max_value << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));

    return 0;
}
