/******************************************************************************
* Functionality: Memory Management
* Handles allocation, deallocation, and initialization of all variables
 ******************************************************************************/

#ifndef BCC_MEM_UTILS_H
#define BCC_MEM_UTILS_H

#include <cuda_runtime.h>
#include "cuda_utility.cuh"

class GPU_BCG {
public:
    GPU_BCG(int, long, long);
    // ~GPU_BCG();

    void init();

    // These two denotes the #f vertices and #f edges in the pbcg graph H_i
    // ____________________________________________________________________ //
    int numVert;
    long numEdges;
    // -------------------------------------------------------------------- //

    int last_root = -1;
    
    long batchSize;
    long E;  // Double the edge count (e,g. for edge (2,3), (3,2) is also counted)
    long max_allot;

    const int orig_numVert;
    const long orig_numEdges;

    /* Pointers for dynamic memory */
    // a. Input edge stream (host pinned memory)
    const uint64_t* h_edgelist;
    
    // b. device arrays
    // 1. Pointers for batch of edges
    uint64_t *d_edgelist;

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ 2. Batch Spanning Tree data-structures_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    uint64_t* d_parentEdge;
    int* d_componentParent;
    int* d_rep;
    
    // pinned memory for flags
    int* h_flag;
    int* h_shortcutFlag;
    
    // respective device flags
    int* d_flag;
    int* d_shortcutFlag;
    // _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ 3. BCC data-structures_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    uint64_t* updated_edgelist;   // This list contains all edges after removing self-loops and duplicate edges
    uint64_t* d_edge_buffer;      // This is an intermediate buffer where we store the previous set of edges plus new batch of edges

    int* d_mapping = NULL;
    int* d_imp_bcc_num = NULL;
    long* h_num_selected_out = NULL;
    long* d_num_selected_out = NULL;
    void *d_temp_storage =  NULL;
    unsigned char *d_flags = NULL;

    // _-_-_-_-_-_-_-_-_-_-_-_-_-_ b. Eulerian Tour data-structures_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    int *d_edges_to;
    int *d_edges_from;

    uint64_t *d_index;
    int *d_next;
    int *d_roots;

    int *d_first;
    int *d_last;

    int *succ;
    int *devRank;

    int *notAllDone;
    ull *devRankNext;
    int *devNotAllDone;

    int *d_parent;
    int *d_level;
    int *devW1Sum;
    int *d_first_occ;
    int *d_last_occ;
    int *iscutVertex;
    // _-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_ c. Assigning Tag starts -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_- //
    int* d_w1;
    int* d_w2;
    int* d_a1;
    int* d_a2;
    int* d_na1;
    int* d_na2;
    int* d_left;
    int* d_right;
    int* d_low;
    int* d_high;
    int* d_fg;

    // making the bcc labels continous
    int* d_bcc_flag;
    int* h_max_ps_bcc;
    int* h_max_ps_cut_vertex;

    // repair
    int* d_counter;
    int* d_isFakeCutVertex;
};

#endif // BCG_MEM_UTILS_H
