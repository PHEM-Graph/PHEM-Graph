//---------------------------------------------------------------------
// CUDA & CUB Libraries
//---------------------------------------------------------------------
#include <cub/cub.cuh>
#include <cuda_runtime.h>

//---------------------------------------------------------------------
// CUDA Utility & helper functions
//---------------------------------------------------------------------
#include "timer.hpp"
#include "cuda_utility.cuh"
#include "bcc_memory_utils.cuh"

extern int local_block_size;

#define DEBUG

__global__
void init_bcc_num_kernel(int* d_mapping, int* d_imp_bcc_num, int numVert) {
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < numVert) {
        d_imp_bcc_num[idx] = idx;
        d_mapping[idx] = idx;
    }
}

// Function to allocate temporary storage for cub functions
size_t AllocateTempStorage(void** d_temp_storage, long num_items) {
    size_t temp_storage_bytes = 0;
    size_t required_bytes = 0;

    // Determine the temporary storage requirement for DeviceRadixSort::SortPairs
    cub::DeviceRadixSort::SortPairs(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceScan::InclusiveSum
    cub::DeviceScan::InclusiveSum(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Determine the temporary storage requirement for DeviceSelect::Flagged
    cub::DeviceSelect::Flagged(nullptr, required_bytes, (int*)nullptr, (int*)nullptr, (int*)nullptr, (int*)nullptr, static_cast<int>(num_items));
    temp_storage_bytes = std::max(temp_storage_bytes, required_bytes);

    // Allocate the maximum required temporary storage
    CUDA_CHECK(cudaMalloc(d_temp_storage, temp_storage_bytes), "cudaMalloc failed for temporary storage for CUB operations");

    return temp_storage_bytes;
}

GPU_BCG::GPU_BCG(int vertices, long num_edges, long _batchSize) : numVert(vertices), orig_numVert(vertices), orig_numEdges(num_edges), batchSize(_batchSize) {

    numEdges = num_edges;
    max_allot = numEdges;
    E = numEdges;
    
    Timer myTimer;

    /* ------------------------------ Batch Spanning Tree ds starts ------------------------------ */
    CUDA_CHECK(cudaMalloc(&d_parentEdge, sizeof(uint64_t) * numVert),       "Failed to allocate memory for d_parentEdge");
    CUDA_CHECK(cudaMalloc(&d_componentParent, sizeof(int) * numVert),       "Failed to allocate memory for d_parentEdge");
    CUDA_CHECK(cudaMalloc(&d_rep, sizeof(int) * numVert),                   "Failed to allocate memory for d_rep_hook");

    CUDA_CHECK(cudaMallocHost((void **)&h_flag, sizeof(int)),               "Failed to allocate memory for c_flag");
    CUDA_CHECK(cudaMallocHost((void **)&h_shortcutFlag, sizeof(int)),       "Failed to allocate memory for c_shortcutFlag");
    
    CUDA_CHECK(cudaMalloc((void **)&d_flag, sizeof(int)),                   "Failed to allocate memory for c_flag");
    CUDA_CHECK(cudaMalloc((void **)&d_shortcutFlag, sizeof(int)),           "Failed to allocate memory for c_shortcutFlag");
    /* ------------------------------ Spanning Tree ds ends ------------------------------ */

    /* ------------------------------ BCC ds starts ------------------------------ */
    CUDA_CHECK(cudaMalloc((void**)&updated_edgelist, numEdges * sizeof(uint64_t)), "Failed to allocate original_edgelist array");

    // set these two arrays (one time work)
    long threadsPerBlock = maxThreadsPerBlock;
    size_t blocks = (numVert + threadsPerBlock - 1) / threadsPerBlock;

    /* ------------------------------ step 1: Eulerian Tour ds starts ------------------------------ */

    int euler_edges = 2 * numVert - 2;

    CUDA_CHECK(cudaMalloc((void **)&d_edges_to,     sizeof(int) * euler_edges), "Failed to allocate d_edges_to");
    CUDA_CHECK(cudaMalloc((void **)&d_edges_from,   sizeof(int) * euler_edges), "Failed to allocate d_edges_from");

    CUDA_CHECK(cudaMalloc((void **)&d_index, sizeof(uint64_t) * euler_edges), "Failed to allocate memory for d_index");
    CUDA_CHECK(cudaMalloc((void **)&d_next, sizeof(int) * euler_edges), "Failed to allocate memory for d_next");
    CUDA_CHECK(cudaMalloc((void **)&d_roots, sizeof(int)), "Failed to allocate memory for d_roots");

    CUDA_CHECK(cudaMalloc((void **)&d_first, sizeof(int) * numVert), "Failed to allocate d_first");
    CUDA_CHECK(cudaMalloc((void **)&d_last, sizeof(int) * numVert),  "Failed to allocate d_last");

    CUDA_CHECK(cudaMalloc((void **)&succ, sizeof(int) * euler_edges), "Failed to allocate succ array");
    CUDA_CHECK(cudaMalloc((void **)&devRank, sizeof(int) * euler_edges), "Failed to allocate devRank array");

    // List Ranking Params
    CUDA_CHECK(cudaMallocHost((void **)&notAllDone, sizeof(int)), "Failed to allocate notAllDone");
    CUDA_CHECK(cudaMalloc((void **)&devRankNext, sizeof(ull) * euler_edges), "Failed to allocate devRankNext");
    CUDA_CHECK(cudaMalloc((void **)&devNotAllDone, sizeof(int)), "Failed to allocate devNotAllDone");
    // List Ranking Params end

    CUDA_CHECK(cudaMalloc((void **)&d_parent,       sizeof(int) * numVert), "Failed to allocate memory for d_parent");
    CUDA_CHECK(cudaMalloc((void **)&d_level,        sizeof(int) * numVert),  "Failed to allocate d_level");
    CUDA_CHECK(cudaMalloc((void **)&devW1Sum,       sizeof(int) * euler_edges), "Failed to allocate devW1Sum");
    CUDA_CHECK(cudaMalloc((void **)&d_first_occ,    sizeof(int) * numVert), "Failed to allocate memory for d_first_occ");
    CUDA_CHECK(cudaMalloc((void **)&d_last_occ,     sizeof(int) * numVert), "Failed to allocate memory for d_last_occ");


    /* ------------------------------ Eulerian Tour ds ends ------------------------------ */

    /* ------------------------------------ step 2: BCC ds starts --------------------------------------- */
    CUDA_CHECK(cudaMalloc(&d_w1,    sizeof(int) * numVert), "Failed to allocate memory for d_w1");
    CUDA_CHECK(cudaMalloc(&d_w2,    sizeof(int) * numVert), "Failed to allocate memory for d_w2");
    CUDA_CHECK(cudaMalloc(&d_low,   sizeof(int) * numVert), "Failed to allocate memory for d_low");
    CUDA_CHECK(cudaMalloc(&d_high,  sizeof(int) * numVert), "Failed to allocate memory for d_high");
    CUDA_CHECK(cudaMalloc(&d_left,  sizeof(int) * numVert), "Failed to allocate memory for d_left");
    CUDA_CHECK(cudaMalloc(&d_right, sizeof(int) * numVert), "Failed to allocate memory for d_right");
    // For updating bcc numbers
    CUDA_CHECK(cudaMalloc(&d_bcc_flag, sizeof(int) * numVert), "Failed to allocate memory for d_bcc_flag");

    CUDA_CHECK(cudaMalloc(&iscutVertex, sizeof(int) * numVert), "Failed to allocate memory for iscutVertex");
    CUDA_CHECK(cudaMalloc(&d_a1,    sizeof(int) * 2 * numVert), "Failed to allocate memory for d_a1");
    CUDA_CHECK(cudaMalloc(&d_a2,    sizeof(int) * 2 * numVert), "Failed to allocate memory for d_a2");

    int n_asize = (2*numVert + local_block_size - 1) / local_block_size;

    CUDA_CHECK(cudaMalloc(&d_na1, n_asize * sizeof(int)) , "Failed to allocate memory to d_na1");
    CUDA_CHECK(cudaMalloc(&d_na2, n_asize * sizeof(int)), "Failed to allocate memory to d_na1");
    CUDA_CHECK(cudaMalloc(&d_fg, sizeof(int) * numEdges), "Failed to allocate memory for d_fg");

    // This is for removing self loops and duplicates
    CUDA_CHECK(cudaMalloc((void**)&d_flags, numEdges * sizeof(unsigned char)), "Failed to allocate flag array");
    CUDA_CHECK(cudaMallocHost(&h_num_selected_out, sizeof(long)),   "Failed to allocate pinned memory for h_num_items value");
    CUDA_CHECK(cudaMalloc((void**)&d_num_selected_out, sizeof(long)), "Failed to allocate d_num_selected_out");

    // Repair data-structure
    CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int) * numVert),   "Failed to allocate memory for d_counter");
    CUDA_CHECK(cudaMalloc(&d_isFakeCutVertex, sizeof(int) * numVert),   "Failed to allocate memory for d_isFakeCutVertex");
    /* ------------------------------------ BCC ds ends --------------------------------------- */

    // Determine temporary storage requirements and allocate
    auto temp_s = AllocateTempStorage(&d_temp_storage, euler_edges);

    // pinned memories
    // CUDA_CHECK(cudaMallocHost(&h_flag, sizeof(int)),                        "Failed to allocate pinned memory for flag value");
    CUDA_CHECK(cudaMallocHost(&h_max_ps_bcc, sizeof(int)),                    "Failed to allocate pinned memory for max_ps_bcc value");
    CUDA_CHECK(cudaMallocHost(&h_max_ps_cut_vertex, sizeof(int)),             "Failed to allocate pinned memory for max_ps_cut_vertex value");

    // Initialize GPU memory
    // init();
    auto dur = myTimer.stop();

    size_t free_byte;
    size_t total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte), "Error: cudaMemGetInfo fails");

    double free_db = static_cast<double>(free_byte); 
    double total_db = static_cast<double>(total_byte);
    double used_db = total_db - free_db;

    std::cout << "========================================\n"
          << "       Allocation Details & GPU Memory Usage\n"
          << "========================================\n\n"
          << "Vertices (numVert):             " << numVert << "\n"
          << "Edges (numEdges):               " << numEdges << "\n"
          << "Max Allotment:                  " << max_allot << "\n"
          // << "Temporary Storage:              " << temp_s << " bytes\n"
          << "Device Allocation & Setup Time: " << dur << " ms\n"
          << "Batch Size:                     " << batchSize << "\n"
          << "Total Number of Batches:        " << (orig_numEdges + batchSize - 1) / batchSize << "\n" 
          << "----------------------------------------\n"
          << "GPU Memory Usage Post-Allocation:\n"
          << "Used:     " << used_db / (1024.0 * 1024.0) << " MB\n"
          << "Free:     " << free_db / (1024.0 * 1024.0) << " MB\n"
          << "Total:    " << total_db / (1024.0 * 1024.0) << " MB\n"
          << "========================================\n\n";
}

// GPU_BCG::~GPU_BCG() {
//     // Free allocated memory
//     Timer myTimer;
//     if(g_verbose) {
//         std::cout <<"\nDestructor started\n";
//     }

//     // Copy buffers
//     for (int i = 0; i < 2; ++i) {
//         CUDA_CHECK(cudaFree(d_edge_u[i]),       "Failed to free copy buffer_u");
//         CUDA_CHECK(cudaFree(d_edge_v[i]),       "Failed to free copy buffer_v");
//     }

//     // Original_edge_stream
//     CUDA_CHECK(cudaFree(original_u),            "Failed to free original_u array");
//     CUDA_CHECK(cudaFree(original_v),            "Failed to free original_v array");

//     // data - structures used for creating csr
//     CUDA_CHECK(cudaFree(u_arr_buf),             "Failed to free u_arr array");
//     CUDA_CHECK(cudaFree(v_arr_buf),             "Failed to free v_arr array");
//     CUDA_CHECK(cudaFree(u_arr_alt_buf),         "Failed to free u_arr_alt_buf array");
//     CUDA_CHECK(cudaFree(v_arr_alt_buf),         "Failed to free v_arr_alt_buf array");

//     // csr vertex offset array
//     CUDA_CHECK(cudaFree(d_vertices),            "Failed to free vertices array");

//     // BFS output
//     CUDA_CHECK(cudaFree(d_parent),              "Failed to free parent array");
//     CUDA_CHECK(cudaFree(d_level),               "Failed to free level array");
//     CUDA_CHECK(cudaFree(d_org_parent),          "Failed to free orginal_parent array");
//     CUDA_CHECK(cudaFree(d_child_of_root),       "Failed to free d_child_of_root");

//     // Part of cuda_BCC data-structure
//     CUDA_CHECK(cudaFree(d_isSafe),              "Failed to free d_isSafe");
//     CUDA_CHECK(cudaFree(d_baseVertex),          "Failed to free d_baseVertex");
//     CUDA_CHECK(cudaFree(d_cut_vertex),          "Failed to free d_cut_vertex");
//     CUDA_CHECK(cudaFree(d_imp_bcc_num),         "Failed to free d_imp_bcc_num");
//     CUDA_CHECK(cudaFree(d_isPartofFund),        "Failed to free d_isPartofFund");
//     CUDA_CHECK(cudaFree(d_is_baseVertex),       "Failed to free d_is_baseVertex");
//     CUDA_CHECK(cudaFree(d_nonTreeEdgeId),       "Failed to free d_nonTreeEdgeId");

//     // CC
//     CUDA_CHECK(cudaFree(d_rep),                 "Failed to free d_rep");
//     CUDA_CHECK(cudaFree(d_baseU),               "Failed to free d_baseU");
//     CUDA_CHECK(cudaFree(d_baseV),               "Failed to free d_baseV");

//     // For updating bcc_num
//     CUDA_CHECK(cudaFree(d_bcc_flag),            "Failed to free d_bcc_flag");

//     // BCG DS
//     CUDA_CHECK(cudaFree(d_mapping),             "Failed to free d_mapping");
//     CUDA_CHECK(cudaFree(d_isFakeCutVertex),     "Failed to free d_isFakeCutVertex");

//     // Common flags
//     CUDA_CHECK(cudaFree(d_flag),                "Failed to free d_flag");
//     CUDA_CHECK(cudaFree(d_flags),               "Failed to free d_flags");
//     CUDA_CHECK(cudaFree(d_temp_storage),        "Failed to free d_temp_storage");
//     CUDA_CHECK(cudaFree(d_num_selected_out),    "Failed to free d_num_selected_out");
    
//     // Destroy CUDA streams
//     CUDA_CHECK(cudaStreamDestroy(computeStream),          "Failed to free computeStream");
//     CUDA_CHECK(cudaStreamDestroy(transH2DStream),      "Failed to free transH2DStream");
//     CUDA_CHECK(cudaStreamDestroy(transD2HStream),      "Failed to free transD2HStream");

//     CUDA_CHECK(cudaEventDestroy(event),         "Failed to destroy event");

//     if(g_verbose) {
//         auto dur = myTimer.stop();
//         std::cout <<"Deallocation of device memory took : " << dur << " ms.\n";
//         std::cout <<"Destructor ended\n";
//     }

//     std::cout <<"\n[Process completed]" << std::endl;
// }
