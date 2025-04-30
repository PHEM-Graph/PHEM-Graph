#include <cuda_runtime.h>
#include <iostream>

#include "bcc_memory_utils.cuh"
#include "fast_bcc.cuh"
#include "cuda_utility.cuh"
#include "spanning_tree.cuh"
#include "sparse_table_min.cuh"
#include "sparse_table_max.cuh"
#include "connected_components.cuh"

#define LOCAL_BLOCK_SIZE 20
// #define DEBUG

__global__
void init_arrays(int* iscutVertex, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		iscutVertex[tid] = 0;
	}
}

__global__
void init_w1_w2(int* w1, int* w2, int numVert, int* first_occ, int* last_occ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        w1[idx] = first_occ[idx];
        w2[idx] = last_occ[idx];
    }
}


__global__
void fill_w1_w2(uint64_t* edge_list, long numEdges, int* w1, int* w2, int* parent , int* first_occ , int* last_occ) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int u = (edge_list[idx] >> 32) & 0xFFFFFFFF;
        int v = edge_list[idx] & 0xFFFFFFFF;
        // if non tree edge
        if(parent[u] != v && parent[v] != u) {
            atomicMin(&w1[v], first_occ[u]);
            atomicMin(&w1[u], first_occ[v]);
            atomicMax(&w2[v], last_occ[u]);
            atomicMax(&w2[u], last_occ[v]);
        }
    }
}
// #define DEBUG

__global__
void compute_a1(int* first_occ, int* last_occ, int numVert , int* w1 , int* a1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        a1[first_occ[idx]] = w1[idx];
        a1[last_occ[idx]] = w1[idx];
    }
}

__global__
void fill_left_right(int* first_occ , int* last_occ, int numVert, int* left, int* right) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        left[idx] = first_occ[idx];
        right[idx] = last_occ[idx];
    }
}


__global__
void mark_fence_back(uint64_t* edge_list, long numEdges, int* low, int* high, int* first_occ, int* last_occ, int* d_parent, int* fg) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        int u = (edge_list[idx] >> 32) & 0xFFFFFFFF;
        int v = edge_list[idx] & 0xFFFFFFFF;
        int f_u = first_occ[u];
        int l_u = last_occ[u];
        int f_v = first_occ[v];
        int l_v = last_occ[v];
        int low_u = low[u];
        int high_u = high[u];
        int low_v = low[v];
        int high_v = high[v];
        if(d_parent[u] != v && d_parent[v] != u) {
            if(f_u <= f_v && l_u>=f_v  || f_v <= f_u && l_v>=f_u) {
                // printf("u = %d , v = %d is a back edge\n", u, v);
                // printf("marking edge %d\n", idx);
                fg[idx] = 0;
            }
            else{
                fg[idx] = 1;
            }
        }
        else{
            if(f_u<=low_v && l_u>=high_v  || f_v<=low_u && l_v>=high_u) {
                //printf("u = %d , v = %d is a fence edge of index %d\n", u, v , idx);
                fg[idx] = 0;
            }
            else{
                fg[idx] = 1;
            }

        }
    }
}

__global__
void mark_fence_back_repair(
    uint64_t* edge_list, 
    long numEdges, 
    int* low, 
    int* high, 
    int* first_occ, 
    int* last_occ, 
    int* d_parent, 
    int* fg, 
    int* d_counter, int root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // if(idx == 0) {
    //     printf("Counter Array from mark_fence_back repair kernel:\n");
    //     for(int i = 0; i < 8; ++i) {
    //         printf("d_counter[%d]: %d\n", i, d_counter[i]);
    //     }
    // }

    if (idx < numEdges) {
        int u = (edge_list[idx] >> 32) & 0xFFFFFFFF;
        int v = edge_list[idx] & 0xFFFFFFFF;
        int f_u = first_occ[u];
        int l_u = last_occ[u];
        int f_v = first_occ[v];
        int l_v = last_occ[v];
        int low_u = low[u];
        int high_u = high[u];
        int low_v = low[v];
        int high_v = high[v];
        if(d_parent[u] != v && d_parent[v] != u) {
            if(f_u <= f_v && l_u>=f_v  || f_v <= f_u && l_v>=f_u) {
                // printf("u = %d , v = %d is a back edge\n", u, v);
                // printf("marking edge %d\n", idx);
                fg[idx] = 0;
            }
            else{
                fg[idx] = 1;
            }
        }
        else{
            if(f_u<=low_v && l_u>=high_v  || f_v<=low_u && l_v>=high_u) {
                // printf("u = %d , v = %d is a fence edge of index %d\n", u, v , idx);
                if(u == d_parent[v] && (d_counter[u] == 1)){
                    // printf("u = %d , v = %d is a fence edge of index %d\n", u, v , idx);
                    fg[idx] = 0;
                }
                else if (v== d_parent[u] && (d_counter[v]==1)){
                    // printf("u = %d , v = %d is a fence edge of index %d\n", u, v , idx);
                    fg[idx] = 0;
                }
                else
                    fg[idx] = 1;
            }
            else {
                fg[idx] = 1;
            }

        }
    }
}

__global__
void give_label_to_root(int* rep, int* parent, int numVert, int root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < numVert) {
        int temp = parent[idx];
        if(idx != temp && temp == root ){
            rep[temp] = rep[idx];
        }
    }
}

__global__
void checking(int* rep, int* parent, int numVert, int* d_flag_for_root ,int root) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // if(idx == 0) {
    //     printf("root flag: %d\n", *d_flag_for_root);
    // }

    if (idx < numVert) {
        int temp = parent[idx];
        if(idx != temp && rep[idx] != rep[temp] && temp == root){
            *d_flag_for_root = 1;            
        }
    }
}

__global__
void mark_cut_vertices(int* rep , int* parent, int* fg,int numVert , int* d_flag_for_root , int root , int* iscutVertex) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert){
        int temp = parent[idx];
        if(temp == root){
            if(d_flag_for_root[0]){ 
            iscutVertex[temp] = true;
            }
        }
        else{
            if(rep[temp] != rep[idx]){
                iscutVertex[temp] = true;
            }
        }
    }
}



__global__
void fill_d_from_to1(uint64_t* edge_list, long numEdges, int* d_from, int* d_to) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numEdges) {
        d_from[idx] = (edge_list[idx] >> 32) & 0xFFFFFFFF;
        d_to[idx] = edge_list[idx] & 0xFFFFFFFF;
    }
}


__global__
void my_copy(int* rep_temp, int* rep, int numVert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        rep[idx] = rep_temp[idx];
    }
}

__global__
void mark_neg_one(int* rep, int* iscutVertex, int numVert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        if(iscutVertex[idx]){
            rep[idx] = numVert + idx+1;
        }
    }
}

__global__
void mark_rem_vertices(int* rep, int numVert, int* prefix_sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        prefix_sum[rep[idx]] = 1;
    }
}


__global__
void update_rep(int* temp_rep, int* rep , int* prefix_sum, int numVert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVert) {
        temp_rep[idx] = prefix_sum[rep[idx]];
    }
}

void assign_tags(const int root, GPU_BCG& g_bcg_ds, bool isLastBatch) {

	int numVert             = 	g_bcg_ds.numVert;
    int numEdges            = 	g_bcg_ds.numEdges;
    int* d_w1               =   g_bcg_ds.d_w1;
    int* d_w2               =   g_bcg_ds.d_w2;
    int* d_first_occ        =   g_bcg_ds.d_first_occ;
    int* d_last_occ         =   g_bcg_ds.d_last_occ;
    uint64_t* d_edgelist    =   g_bcg_ds.updated_edgelist;
    int* d_parent           =   g_bcg_ds.d_parent;
    int* d_a1               =   g_bcg_ds.d_a1;
    int* d_a2               =   g_bcg_ds.d_a2;
    int* d_na1              =   g_bcg_ds.d_na1;
    int* d_na2              =   g_bcg_ds.d_na2;
    int* d_left             =   g_bcg_ds.d_left;
    int* d_right            =   g_bcg_ds.d_right;
    int* d_low              =   g_bcg_ds.d_low;
    int* d_high             =   g_bcg_ds.d_high;
    int* d_fg               =   g_bcg_ds.d_fg;
    int* d_counter          =   g_bcg_ds.d_counter;

    int n_asize = (2 * numVert + LOCAL_BLOCK_SIZE - 1) / LOCAL_BLOCK_SIZE;

    // step 2: Compute w1, w2, low and high using first and last
    init_w1_w2<<<(numVert + 1023) / 1024, 1024>>>(d_w1, d_w2, numVert , d_first_occ , d_last_occ);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    fill_w1_w2<<<(numEdges + 1023) / 1024, 1024>>>(d_edgelist, numEdges, d_w1, d_w2, d_parent , d_first_occ , d_last_occ);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    compute_a1<<<(numVert + 1023) / 1024, 1024>>>(d_first_occ, d_last_occ, numVert , d_w1 , d_a1);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    compute_a1<<<(numVert + 1023) / 1024, 1024>>>(d_first_occ, d_last_occ, numVert , d_w2 , d_a2);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    fill_left_right<<<(numEdges + 1023) / 1024, 1024>>>(d_first_occ , d_last_occ, numVert, d_left, d_right);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    main_min(2*numVert, numVert, d_a1, d_left, d_right, d_low , n_asize , d_na1);
    main_max(2*numVert, numVert, d_a2, d_left, d_right, d_high , n_asize , d_na2);

    // Step 3: Mark Fence and Back Edges using the above 4 tags
    if(!isLastBatch) {
        mark_fence_back<<<(numEdges + 1023) / 1024, 1024>>>(
            d_edgelist, 
            numEdges, 
            d_low, 
            d_high, 
            d_first_occ, 
            d_last_occ, 
            d_parent, 
            d_fg);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");
    }

    else {
        #ifdef DEBUG
            std::cout << "Counter Array before mark_fence_back kernel: " << std::endl;
            print_device_array(d_counter, numVert);
        #endif
        mark_fence_back_repair<<<(numEdges + 1023) / 1024, 1024>>>(
            d_edgelist, 
            numEdges, 
            d_low, 
            d_high, 
            d_first_occ, 
            d_last_occ, 
            d_parent, 
            d_fg,
            d_counter,
            root);
        CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");
    }

}

void finalise_labels(int root, GPU_BCG& g_bcg_ds) {

	int* d_rep 		= 	g_bcg_ds.d_rep;
	int* d_parent 	= 	g_bcg_ds.d_parent;
	int numVert 	= 	g_bcg_ds.numVert;
    int numEdges    =   g_bcg_ds.numEdges;
    int* d_fg       =   g_bcg_ds.d_fg;
	int* iscutVertex =  g_bcg_ds.iscutVertex;


    const long numThreads = 1024;
    int numBlocks = (numVert + numThreads - 1) / numThreads;

	init_arrays<<<numBlocks, numThreads>>>(iscutVertex, numVert);
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

	give_label_to_root<<<(numVert + 1023)/ 1024 , 1024>>>(
        d_rep, d_parent, 
        numVert, 
        root);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    int* d_flag_for_root = g_bcg_ds.d_flag;
    int h_flag_for_root = 0;
    CUDA_CHECK(cudaMemcpy(d_flag_for_root, &h_flag_for_root, sizeof(int), cudaMemcpyHostToDevice), "Failed to copy");

    checking<<<(numVert+ 1023 )/ 1024 , 1024>>>(d_rep, d_parent, numVert, d_flag_for_root, root);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    #ifdef DEBUG
        std::cout << "Flag array after checking kernel: ";
        print_device_array(d_flag_for_root , 1);

        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
        std::cout << "BCC Numbers:" << std::endl;
        print_device_array(d_rep, numVert);
        std::cout << std::endl;
        std::cout<<"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    #endif

    mark_cut_vertices<<<(numEdges + 1023) / 1024, 1024>>>(
        d_rep, d_parent, 
        d_fg, 
        numVert, 
        d_flag_for_root, 
        root, 
        iscutVertex);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize device");

    #ifdef DEBUG
        std::cout << "Cut Vertices info: \n" << std::endl;
        print_device_array(iscutVertex, numVert);
    #endif
}

__global__ 
void update_bcc_flag_kernel(int* d_cut_vertex, int* d_bcc_num, int* d_bcc_flag, int numVert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < numVert) {
        if(!d_cut_vertex[i]) {
            d_bcc_flag[d_bcc_num[i]] = 1;
        }
    }
}

__global__ 
void update_bcc_number_kernel(int* d_cut_vertex, int* d_bcc_num, int* bcc_ps, int* cut_ps, int numVert) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // if(i == 0) update_bcg_num_vert;
    if(i < numVert) {
        if(!d_cut_vertex[i]) {
            d_bcc_num[i] = bcc_ps[d_bcc_num[i]] - 1;
        }
        else
            d_bcc_num[i] = bcc_ps[numVert - 1] + cut_ps[i] - 1;
    }
}

// inclusive prefix_sum
void incl_scan(
    int*& d_in, 
    int*& d_out, 
    int& num_items, 
    void*& d_temp_storage) {

    size_t temp_storage_bytes = 0;
    cudaError_t status;

    status = cub::DeviceScan::InclusiveSum(NULL, temp_storage_bytes, d_in, d_out, num_items);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");
    
    // Run inclusive prefix sum
    status = cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");
}

int update_bcc_numbers(GPU_BCG& g_bcg_ds, int numVert) {
    int* d_bcc_flag     =   g_bcg_ds.d_bcc_flag;
    int* d_bcc_ps       =   g_bcg_ds.d_left;   // reusing few arrays
    int* d_cut_ps       =   g_bcg_ds.d_right; // reusing few arrays
    
    int* d_cut_vertex   =   g_bcg_ds.iscutVertex;
    int* d_bcc_num      =   g_bcg_ds.d_rep;
    
    int numThreads = static_cast<int>(maxThreadsPerBlock);
    size_t numBlocks = (numVert + numThreads - 1) / numThreads;

    init_arrays<<<numBlocks, numThreads>>>(d_bcc_flag, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize init_arrays kernel");

    update_bcc_flag_kernel<<<numBlocks, numThreads>>>(d_cut_vertex, d_bcc_num, d_bcc_flag, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize update_bcc_flag kernel");

    incl_scan(d_bcc_flag,   d_bcc_ps, numVert, g_bcg_ds.d_temp_storage);
    incl_scan(d_cut_vertex, d_cut_ps, numVert, g_bcg_ds.d_temp_storage);

    // pinned memory
    int* h_max_ps_bcc           = g_bcg_ds.h_max_ps_bcc;
    int* h_max_ps_cut_vertex    = g_bcg_ds.h_max_ps_cut_vertex;

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");
    CUDA_CHECK(cudaMemcpy(h_max_ps_bcc, &d_bcc_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back max_ps_bcc.");
    CUDA_CHECK(cudaMemcpy(h_max_ps_cut_vertex, &d_cut_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back max_ps_cut_vertex.");
    
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    std::cout << "max_ps_bcc: " << *h_max_ps_bcc << "\n";
    std::cout << "max_ps_cut_vertex: " << *h_max_ps_cut_vertex << "\n";

    int bcg_num_vert = *h_max_ps_bcc + *h_max_ps_cut_vertex;

    update_bcc_number_kernel<<<numBlocks, numThreads>>>(
        d_cut_vertex, 
        d_bcc_num, 
        d_bcc_ps, 
        d_cut_ps, 
        numVert
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    if(g_verbose) {
        std::cout << "BCC numbers:" << std::endl;
        print_device_array(d_bcc_num, numVert);
        std::cout << "Cut vertex status:" << std::endl;
        print_device_array(d_cut_vertex, numVert);
    }

    return bcg_num_vert;
}

void Fast_BCC(GPU_BCG& g_bcg_ds, const bool isLastBatch) {

    int numVert     =   g_bcg_ds.numVert;
    long numEdges   =   g_bcg_ds.numEdges;
    int* d_fg       =   g_bcg_ds.d_fg;
    int* iscutVertex =  g_bcg_ds.iscutVertex;

    std::cout << "\n***************** Starting CUDA_BCC...*****************\n";

    std::cout << "Input numVert: " << numVert << " and numEdges: " << numEdges << "\n";

    std::cout << "********************************************************************\n";

    // step 0: init cut vertices, cut edges, bcc numbers, d_fg, etc
    if(g_verbose) {
        std::cout << "Actual Edges array:" << std::endl;
        print_device_edges(g_bcg_ds.updated_edgelist, numEdges);
        std::cout << std::endl;
    }
	
    // step 1: construct Spanning Tree
	int root = construct_spanning_tree(g_bcg_ds);

    g_bcg_ds.last_root = root;

    auto start = std::chrono::high_resolution_clock::now();
	// step 2: Assigning Tags
	assign_tags(root, g_bcg_ds, isLastBatch);
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Assigning Tages time: " << dur << " ms." << std::endl;

    start = std::chrono::high_resolution_clock::now();
	// Apply CC
	connected_comp(g_bcg_ds.numEdges, g_bcg_ds.updated_edgelist, g_bcg_ds.d_fg, g_bcg_ds.numVert, g_bcg_ds.d_rep, g_bcg_ds.d_flag);
    end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "CC time: " << dur << " ms." << std::endl;

    if(isLastBatch)
        return;
	
    start = std::chrono::high_resolution_clock::now();
    // finalise the labels
	finalise_labels(root, g_bcg_ds);
	end = std::chrono::high_resolution_clock::now();
    dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "finalising labels time: " << dur << " ms." << std::endl;
}