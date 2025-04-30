#include <stdio.h>
#include <stdlib.h>
#include "include/graph.cuh"
#include "include/hbcg_utils.cuh"
#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "include/host_spanning_tree.cuh"
#include "include/GPUSpanningtree.cuh"
#include "include/ExternalSpanningTree.cuh"
#include "include/cuda_utility.cuh"
#include <chrono>

using namespace std;

struct graph_data_host h_input;


struct graph_data d_input;
struct graph_data h_input_gpu;

int Batch_Size;
float GPU_share;

#define tb_size 1024
#define LOCAL_BLOCK_SIZE 100

void print_mem_info() {
    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;
    std::cout << "----------------------------------------\n"
          << "GPU Memory Usage Post-Allocation:\n"
          << "Used:     " << used_db / (1024.0 * 1024.0) << " MB\n"
          << "Free:     " << free_db / (1024.0 * 1024.0) << " MB\n"
          << "Total:    " << total_db / (1024.0 * 1024.0) << " MB\n"
          << "========================================\n\n";
}

int main(int argc, char *argv[]) {
    if (argc != 4){
        printf("Usage: %s <input_file> <GPU_share> <Batch_Size> , ", argv[0]);
        exit(1);
    }

    // printf("Started Reading\n");// -*-

    float time_ms = 0.0;
    GPU_share = atof(argv[2]);
    Batch_Size = atoi(argv[3]);
    undirected_graph G(argv[1]);
    printf("%f,", GPU_share);// -*-

    // printf("Graph read successfully\n");// -*-

    long num_edges_for_gpu = (long)(GPU_share * (G.getNumEdges()));
    long num_edges_for_cpu = (G.getNumEdges()) - num_edges_for_gpu;  

    if((int)GPU_share == 1){
        num_edges_for_gpu = G.getNumEdges();
        num_edges_for_cpu = 0;
    }

    //pinned memory allocation
    cudaMallocHost((void**)&h_input.V, sizeof(int));
    cudaMallocHost((void**)&h_input.E, sizeof(long));
    cudaMallocHost((void**)&h_input.edges_size, sizeof(long));
    cudaMallocHost((void**)&h_input.edge_list_size, sizeof(long));

    //pinned memory for h_input_gpu
    cudaMallocHost((void**)&h_input_gpu.V, sizeof(int));
    cudaMallocHost((void**)&h_input_gpu.E, sizeof(int));
    cudaMallocHost((void**)&h_input_gpu.size, sizeof(long));
    cudaMallocHost((void**)&h_input_gpu.size2, sizeof(long));
    h_input_gpu.edges = G.h_edgelist;

    //cuda Allocating memory for d_input
    cudaMalloc((void**)&d_input.V, sizeof(int));
    cudaMalloc((void**)&d_input.E, sizeof(int));
    cudaMalloc((void**)&d_input.size, sizeof(long));
    cudaMalloc((void**)&d_input.size2, sizeof(long));
    cudaMalloc((void**)&d_input.edges, Batch_Size * sizeof(uint64_t));
    cudaMalloc((void**)&d_input.edges2, Batch_Size * sizeof(uint64_t));
    cudaMalloc((void**)&d_input.label, G.getNumVertices() * sizeof(int));
    cudaMalloc((void**)&d_input.temp_label, G.getNumVertices() * sizeof(int));
    cudaMalloc((void**)&d_input.T1edges, G.getNumVertices() * sizeof(uint64_t));
    CUDA_CHECK(cudaMalloc((void**)&d_input.T2edges, G.getNumVertices() * sizeof(uint64_t) ), "Failed to allocate memory for d_input.T2edges");

    // printf("Allocations done successfully\n");// -*-

    h_input.V[0] = G.getNumVertices();
    h_input.E[0] = G.getNumEdges();
    h_input.edges_size[0] = num_edges_for_cpu;
    h_input.edge_list_size[0] = num_edges_for_gpu;

    h_input_gpu.V[0] = G.getNumVertices();
    h_input_gpu.E[0] = G.getNumEdges(); 

    cudaMemcpy(d_input.V, h_input.V, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_input.E, h_input.E, sizeof(int), cudaMemcpyHostToDevice);


    // printf("Number of vertices: %d\n", h_input.V[0]);// -*-
    // printf("Number of edges: %ld\n", h_input.E[0]);// -*-
    // printf("Number of edges for CPU: %ld\n", h_input.edges_size[0]);// -*-
    // printf("Number of edges for GPU: %ld\n", h_input.edge_list_size[0]);// -*-

    //h_input.edgelist = G.h_edgelist;
    h_input.label = parlay::sequence<int>::from_function(h_input.V[0], [&](size_t i) { return i; });
    h_input.temp_label = parlay::sequence<int>::from_function(h_input.V[0], [&](size_t i) { return i;});
    h_input.sptree = parlay::sequence<uint64_t>::from_function(h_input.V[0], [&](size_t i) { return INT_MAX;});
    h_input.edges = G.edges64;

    #ifdef DEBUG
        printf("Printing edge list given to gpu\n");
        for(long i=0;i<h_input.edge_list_size[0];i++){
            uint64_t edge = h_input_gpu.edges[i];
            int u = edge >> 32;          // Extract the upper 32 bits
            int v = edge & 0xFFFFFFFF;   // Extract the lower 32 bits
            printf("%d %d\n", u, v);
        }
        printf("Printing edge list given to cpu\n");
        for(long i = 0; i < h_input.edges_size[0]; ++i) {
            uint64_t edge = h_input.edges[i];
            int u = edge >> 32;          // Extract the upper 32 bits
            int v = edge & 0xFFFFFFFF;   // Extract the lower 32 bits
            printf("%d %d\n", u, v);
        }
    #endif

    //Step 1 :: Heterogeneous Spanning Tree Algorithm
    //Spanning tree is stored in d_input.T2edges

    time_ms = SpanningTree(&h_input_gpu, &d_input, &h_input);
    // printf("Time taken for spanning tree : %f ms\n", time_ms);
    printf("%f ms,", time_ms);

    #ifdef DEBUG
        int numCC_gpu =0 ;
        int numCC_cpu = 0;
        int* label = (int*)malloc(G.getNumVertices() * sizeof(int));
        cudaMemcpy(label, d_input.label, G.getNumVertices() * sizeof(int), cudaMemcpyDeviceToHost);
        //printf("Printing labels of gpu part\n");
        for(int i = 0; i < G.getNumVertices(); ++i){
            if(label[i] == i){
                numCC_gpu++;
            }
        }
        //printf("Printing labels of cpu part\n");
        for(int i = 0; i < G.getNumVertices(); ++i){
            if(h_input.label[i] == i){
                numCC_cpu++;
            }
        }
        printf("Number of connected components in GPU: %d\n", numCC_gpu);
        printf("Number of connected components in CPU: %d\n", numCC_cpu);

        //printing sptree and T1edges in gpu
        uint64_t* T1 = (uint64_t*)malloc(G.getNumVertices() * sizeof(uint64_t));
        cudaMemcpy(T1, d_input.T1edges, G.getNumVertices() * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        printf("Printing T1edges\n");
        for(int i = 0; i < G.getNumVertices(); i++){
            printf("%d %d\n", T1[i] >> 32, T1[i] & 0xFFFFFFFF);
        }
        printf("Printing sptree\n");
        for(int i = 0; i < G.getNumVertices(); ++i){
            printf("%d %d\n", h_input.sptree[i] >> 32, h_input.sptree[i] & 0xFFFFFFFF);
        }
    #endif

    uint64_t* temp_array;
    cudaMalloc((void**)&temp_array, G.getNumVertices() * sizeof(uint64_t));
    cudaMemcpy(temp_array, h_input.sptree.begin(), G.getNumVertices() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    float extraSptime = gpu_spanning_tree(&d_input, temp_array, G.getNumVertices());
    // printf("extra sp time : %f\n", extraSptime);// -*-
    printf("%f ms,", extraSptime);
    time_ms += extraSptime;
    printf("%f ms,", time_ms);
    
    int numCC =0 ;
    int* label = (int*)malloc(G.getNumVertices() * sizeof(int));
    cudaMemcpy(label, d_input.label, G.getNumVertices() * sizeof(int), cudaMemcpyDeviceToHost);
    //printf("Printing labels of gpu part\n");
    for(int i = 0; i < G.getNumVertices(); ++i){
        if(label[i] == i){
            numCC++;
        }
    }
    printf("CC = %d \n", numCC);

    cudaFree(d_input.edges);
    cudaFree(d_input.edges2);
    cudaFree(d_input.label);
    cudaFree(d_input.temp_label);
    cudaFree(d_input.T1edges);
    cudaFree(label);
    
    return 0;
}
