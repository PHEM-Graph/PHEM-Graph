#include <fstream>
#include <cassert>
#include <random>
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <vector>

#define CUDA_CHECK(call, msg) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " << msg << " (" << cudaGetErrorString(err) << ")" << std::endl; \
            throw std::runtime_error("CUDA Error"); \
        } \
    } while (0)

class undirected_graph {
public:
    undirected_graph(const std::string& filename);

private:
    void readGraphFile();
    void readEdgeList();
    void readECLgraph();
    void csr_to_coo();
    void copyEdgesToGPU();

    std::filesystem::path filepath;
    std::chrono::duration<double> read_duration;

    int numVert = 0;
    long numEdges = 0;
    std::vector<long> vertices;
    std::vector<int> edges;
    std::vector<uint64_t> edges64;
    uint64_t* h_edgelist = nullptr;
    uint64_t* d_edgelist = nullptr;
    float GPU_share = 0.5;
};

undirected_graph::undirected_graph(const std::string& filename) : filepath(filename) {
    try {
        auto start = std::chrono::high_resolution_clock::now();
        readGraphFile();
        copyEdgesToGPU();
        auto end = std::chrono::high_resolution_clock::now();
        read_duration = end - start;
    }   
    catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

void undirected_graph::readGraphFile() {
    if (!std::filesystem::exists(filepath)) {
        throw std::runtime_error("File does not exist: " + filepath.string());
    }

    std::string ext = filepath.extension().string();
    if (ext == ".edges" || ext == ".eg2" || ext == ".txt") {
        readEdgeList();
    }
    else if (ext == ".egr" || ext == ".bin" || ext == ".csr") {
        readECLgraph();
    }
    else {
        throw std::runtime_error("Unsupported graph format: " + ext);
    }
}

void undirected_graph::readEdgeList() {
    std::ifstream inFile(filepath);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filepath.string());
    }
    inFile >> numVert >> numEdges;

    size_t bytes = (numEdges / 2) * sizeof(uint64_t);
    CUDA_CHECK(cudaMallocHost((void**)&h_edgelist, bytes),  "Failed to allocate pinned memory for edgelist");
    edges64.resize(numEdges / 2);

    long ctr = 0;
    int u, v;
    for (long i = 0; i < numEdges; ++i) {
        inFile >> u >> v;
        if (u < v) {
            h_edgelist[ctr] = (static_cast<uint64_t>(u) << 32) | (v);
            ctr++;
        }
    }
    assert(ctr == numEdges / 2);

    numEdges = ctr;
}   

void undirected_graph::readECLgraph() {
    std::ifstream inFile(filepath, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Error opening file: " + filepath.string());
    }

    // Reading sizes
    size_t size;
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    vertices.resize(size);
    inFile.read(reinterpret_cast<char*>(&size), sizeof(size));
    edges.resize(size);

    // Reading data
    inFile.read(reinterpret_cast<char*>(vertices.data()), vertices.size() * sizeof(long));
    inFile.read(reinterpret_cast<char*>(edges.data()), edges.size() * sizeof(int));

    numVert = vertices.size() - 1;
    numEdges = edges.size();

    csr_to_coo();
}

void undirected_graph::csr_to_coo() {
    long num_edges_for_gpu = (long)(GPU_share * (numEdges / 2));
    size_t bytes = (numEdges / 2) * sizeof(uint64_t);
    
    CUDA_CHECK(cudaMallocHost((void**)&h_edgelist, bytes),  "Failed to allocate pinned memory for edgelist");
    
    long ctr = 0;
    for (int i = 0; i < numVert; ++i) {
        for (long j = vertices[i]; j < vertices[i + 1]; ++j) {
            if (i < edges[j]) {
                int u = i;
                int v = edges[j];
                h_edgelist[ctr] = (static_cast<uint64_t>(u) << 32) | (v);
                ctr++;
            }
        }
    }
    assert(ctr == numEdges / 2);
    numEdges = ctr;

    vertices.clear();
    edges.clear();
}

void undirected_graph::copyEdgesToGPU() {
    size_t bytes = numEdges * sizeof(uint64_t);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start), "Failed to create start event");
    CUDA_CHECK(cudaEventCreate(&stop), "Failed to create stop event");

    CUDA_CHECK(cudaMalloc((void**)&d_edgelist, bytes), "Failed to allocate device memory for edgelist");

    CUDA_CHECK(cudaEventRecord(start), "Failed to record start event");
    CUDA_CHECK(cudaMemcpy(d_edgelist, h_edgelist, bytes, cudaMemcpyHostToDevice), "Failed to copy edges to device");
    CUDA_CHECK(cudaEventRecord(stop), "Failed to record stop event");

    CUDA_CHECK(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop), "Failed to compute elapsed time");

    std::cout << "Time to copy edges to GPU: " << milliseconds << " ms" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start), "Failed to destroy start event");
    CUDA_CHECK(cudaEventDestroy(stop), "Failed to destroy stop event");
}


int main(int argc, char** argv) {
    // Example usage:
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph-file>" << std::endl;
        return 1;
    }

    try {
        undirected_graph g(argv[1]);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
