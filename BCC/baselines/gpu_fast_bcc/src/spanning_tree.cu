#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

#include "euler.cuh"
#include "cuda_utility.cuh"
#include "spanning_tree.cuh"
#include "bcc_memory_utils.cuh"

// #define DEBUG

__global__
void init(uint64_t* d_parentEdge, int* d_componentParent, int* d_rep, int nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < nodes) {
        d_parentEdge[idx] = INT_MAX;
        d_componentParent[idx] = idx;
        d_rep[idx] = idx;
    }
}

__global__ 
void HOOKING(
    long edges, 
    uint64_t* d_edge_list,
    int *rep, 
    int *componentParent, 
    bool isMaxIteration, 
    int *c_flag) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(tid < edges) {
        
        uint64_t i = d_edge_list[tid];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        int rep_u = rep[u];
        int rep_v = rep[v];

        if(rep_u != rep_v) {
            // 2 different components
            *c_flag = true;
            if(isMaxIteration) {
                componentParent[min(rep_u, rep_v)] = max(rep_u, rep_v);
            }
            else {
                componentParent[max(rep_u, rep_v)] = min(rep_u, rep_v);
            }
        }
    }
}

__global__ 
void UPDATE_REP_PARENT(int nodes, int *componentParent, int *rep) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < nodes) {
        if(rep[tid] == tid && componentParent[tid] != -1) {
            rep[tid] = componentParent[tid];
        }
    }
}

/**
 * Needs to be executed before UPDATE_REP_PARENT
 * @d_parentEdge : d_parentEdge[i] --> idx of the edge which connects ith tree to parent of ith tree
*/
__global__ 
void STORE_CROSS_EDGES(
    int edges,
    int *rep, 
    uint64_t* d_edge_list,
    int *componentParent, 
    uint64_t *d_parentEdge) {

    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < edges) {
       
        uint64_t i = d_edge_list[tid];

        int u = i >> 32;  // Extract higher 32 bits
        int v = i & 0xFFFFFFFF; // Extract lower 32 bits
        
        int rep_u = rep[u];
        int rep_v = rep[v];

        if(rep_u == rep_v)
            return;

        // printf("u = %d, v = %d , rep_u = %d, rep_v = %d \n", u, v, rep_u, rep_v);
        if( rep_v == componentParent[rep_u]){
            // u is the representative of the tree
            // v belongs to the parent tree of u

            d_parentEdge[rep_u] = d_edge_list[tid];
        }

        if( rep_u == componentParent[rep_v]) {
            d_parentEdge[rep_v] = d_edge_list[tid];
        }   
    }
}

__global__ 
void SHORTCUTTING(int nodes, int *rep, int *flag) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nodes)
    {
        int prevValue = rep[tid];
        rep[tid] = rep[rep[tid]];
        if (prevValue != rep[tid])
        {
            *flag = 1;
        }
    }
}

void hooking(
    int nodes, long edges,
    uint64_t* d_edge_list,
    int* d_rep,                     // rep[i] --> representative of the tree of which i is a part
    int* d_componentParent,         // componentParent[i] =rep of parent tree of the ith tree
    uint64_t* d_parentEdge,
    int* h_flag,                
    int* d_flag, 
    int* h_shortcutFlag,
    int* d_shortcutFlag) {

    #ifdef DEBUG
        std::cout << "Printing from hooking function:" << std::endl;

        std::cout << "nodes: " << nodes << ", edges: " << edges << std::endl;

        std::cout << "rep array:" << std::endl;
        // print_device_array(d_rep, nodes);

        std::cout << "Edges input to hooking: " << std::endl;
        // print_device_edge_list(d_edge_list, edges);
    #endif

    int num_threads = 1024;

    int num_blocks_edges = (edges + num_threads - 1) / num_threads;
    int num_blocks_vert = (nodes + num_threads - 1) / num_threads;

    *h_flag = 1;
    *h_shortcutFlag = 1;
    int itr_count = 0;
    bool maxIteration = true;

        #ifdef DEBUG
            std::cout << "\n\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n";
            std::cout << "initial values:\n";
            // print rep and components parent
            std::cout << "Rep array:\n";
            print_device_array(d_rep, nodes);
            std::cout << "Components array:\n";
            print_device_array(d_componentParent, nodes);
            std::cout << "Selected Edges array:\n";
            print_device_edges(d_parentEdge, nodes);
            std::cout << "\n\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n";
        #endif

    while(*h_flag) {
        itr_count++;
        *h_flag = false;
        // cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
        CUDA_CHECK(cudaMemcpy(d_flag, h_flag, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy flag to device");
        HOOKING<<<num_blocks_edges, num_threads>>> (
            edges,
            d_edge_list,
            d_rep,
            d_componentParent,
            maxIteration,
            d_flag);

        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after HOOKING");
        CUDA_CHECK(cudaMemcpy(h_flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy flag back to device");

        maxIteration = !maxIteration;

        // !!! This should be done before updating
        STORE_CROSS_EDGES<<<num_blocks_edges, num_threads>>> (
            edges,
            d_rep,
            d_edge_list,
            d_componentParent,
            d_parentEdge
        );
        
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after STORE_CROSS_EDGES");

        // rep[representative] = representative of its parent
        UPDATE_REP_PARENT<<<num_blocks_vert, num_threads>>> (
            nodes,
            d_componentParent,
            d_rep);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after UPDATE_REP_PARENT");
        
        *h_shortcutFlag = true;
        // auto start = std::chrono::high_resolution_clock::now();
        while(*h_shortcutFlag) {
            *h_shortcutFlag = false;
            CUDA_CHECK(cudaMemcpy(d_shortcutFlag, h_shortcutFlag, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy h_shortcutFlag to device");
            SHORTCUTTING <<<num_blocks_vert, num_threads >>> (nodes, d_rep, d_shortcutFlag);
            CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after SHORTCUTTING kernel");
            CUDA_CHECK(cudaMemcpy(h_shortcutFlag, d_shortcutFlag, sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back h_shortcutFlag to host");
        }

        #ifdef DEBUG
        int k = 0;
        std::cout << "\n\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n";
            std::cout << "Printing for " << k++ << " iteration.\n";
            // print rep and components parent
            std::cout << "Rep array:\n";
            print_device_array(d_rep, nodes);
            std::cout << "Components array:\n";
            print_device_array(d_componentParent, nodes);
            std::cout << "Selected Edges array:\n";
            print_device_edges(d_parentEdge, nodes);
            std::cout << "\n\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n";
        #endif
    }

    #ifdef DEBUG
        std::cout << "\n\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n";
        std::cout << "Printing Final Rep array:" << std::endl;
        print_device_array(d_rep, nodes);
        std::cout << std::endl;

        std::cout << "Printing spanning tree edges:" << std::endl;
        print_device_edges(d_parentEdge, nodes);
        std::cout << std::endl;
        std::cout << "\n\n-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_\n\n";
    #endif
}

int construct_spanning_tree(GPU_BCG& g_bcg_ds) {

    uint64_t* d_edgelist 		= 	g_bcg_ds.updated_edgelist; // The input edgelist
    int nodes 					= 	g_bcg_ds.numVert;
    long edges 					= 	g_bcg_ds.numEdges;
    uint64_t* d_parentEdge 		= 	g_bcg_ds.d_parentEdge; // The actual spanning Tree edges (output)
    int* d_componentParent 		= 	g_bcg_ds.d_componentParent;
    int* d_rep 					= 	g_bcg_ds.d_rep;

    int *h_flag 				= 	g_bcg_ds.h_flag;
    int *h_shortcutFlag 		= 	g_bcg_ds.h_shortcutFlag;

    int *d_flag 				= 	g_bcg_ds.d_flag;
    int *d_shortcutFlag 		= 	g_bcg_ds.d_shortcutFlag;

    #ifdef DEBUG
        std::cout << "Actual Edges array:\n";
        print_device_edges(d_edgelist, edges);
    #endif

    int num_threads = 1024;
    int num_blocks_vert = (nodes + num_threads - 1) / num_threads;

    auto start = std::chrono::high_resolution_clock::now();

    init<<<num_blocks_vert, num_threads>>>(d_parentEdge, d_componentParent, d_rep, nodes);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init kernel");

    hooking(
        nodes, edges, 
        d_edgelist, 
        d_rep, 
        d_componentParent, 
        d_parentEdge, 
        h_flag,
        d_flag,
        h_shortcutFlag,
        d_shortcutFlag);


    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Spanning Tree Construction time: " << dur << " ms." << std::endl;

    CUDA_CHECK(cudaMemcpy(h_flag, &d_rep[0], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy flag from device to host");
    int root = *h_flag;
    // std::cout << "Root Value: " << root << std::endl;

    cuda_euler_tour(nodes, root, g_bcg_ds);

    return root;
}
