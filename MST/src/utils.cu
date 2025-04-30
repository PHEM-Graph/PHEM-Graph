#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/atomic>
#include <random>
#include <stdlib.h>
#include <omp.h>
#include "constants.cpp"
#include <execution>
#define BLOCKSIZE 1024
#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "common/graph.h"
#include "common/speculative_for.h"
#include "common/get_time.h"
#include "algorithm/union_find.h"
#include "MST.h"



void printProgress(double progress) {
  int barWidth = 70;

  std::flush(std::cout);
  std::cout << "[";
  int pos = barWidth * progress;
  for (int i = 0; i < barWidth; ++i) {
    if (i < pos) std::cout << "=";
    else if (i == pos) std::cout << ">";
    else std::cout << " ";
  }
  std::cout << "] " << int(progress * 100.0) << " %\r";
  std::cout.flush();
  std::flush(std::cout);

}

std::random_device rd;     // Only used once to initialise (seed) engine
std::mt19937 rng(rd());    // Random-number engine used (Mersenne-Twister in this case)
std::uniform_int_distribution<int> uni(1,100); // Guaranteed unbiased


edge sentinel_edge{sentinel_vertex, sentinel_vertex, sentinel};
//sentinel_edge.u = sentinel_vertex;
//sentinel_edge.v = sentinel_vertex;
//sentinel_edge.w = sentinel;

vertex hash_weight(vertex a){
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}


inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}



//Functor type for selecting edges where the weight == sentinel
struct NotSentinel{
    CUB_RUNTIME_FUNCTION __forceinline__
    __host__ __device__ bool operator()(const edge &a) const {
        return a.w != sentinel;
    }
};

//Function to remove the edges where src == -1
vertex remove_sentinel_weighted_edges(edge* d_edges, vertex num_edges, edge* d_edges_out){

    vertex *d_num_selected_out;
    cudaMalloc(&d_num_selected_out, sizeof(int));

    //Use cub to remove the edges where src == -1
    //Malloc a temporary storage array on device
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    //Compute the number of bytes of temporary storage needed
    std::cout << "start cub \n";
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_edges, d_edges_out, d_num_selected_out, num_edges, NotSentinel());
    std::cout << "found number of bytes \n";

    //Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    //Run selection
    std::cout << "start selection cub\n ";
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_edges, d_edges_out, d_num_selected_out, num_edges, NotSentinel());
    std::cout << "done selection cub\n ";


    vertex num_selected_out;
    //Set num_selected_out to d_num_selected_out
    //THIS COULD BE BLOCKING
    cudaMemcpy(&num_selected_out, d_num_selected_out, sizeof(vertex), cudaMemcpyDeviceToHost);
    //print num_selected_out
    std::cout << "number of selected out " << num_selected_out << std::endl;

    cudaFree(d_temp_storage);

    return num_selected_out;
}


//Functor type for selecting edges where the src != dst
struct NotSelfLoop{
    CUB_RUNTIME_FUNCTION __forceinline__
    __host__ __device__ bool operator()(const edge &a) const {
        return a.u != a.v;
    }
};

vertex remove_self_loops(edge* d_edges, vertex num_edges, edge* d_edges_out){

    vertex *d_num_selected_out;
    cudaMalloc(&d_num_selected_out, sizeof(vertex));

    //Use cub to remove the edges where src == -1
    //Malloc a temporary storage array on device
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    //Compute the number of bytes of temporary storage needed
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_edges, d_edges_out, d_num_selected_out, num_edges, NotSelfLoop(), 0);

    //Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    //Run selection
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_edges, d_edges_out, d_num_selected_out, num_edges, NotSelfLoop(), 0);

    vertex h_num_selected_out;
    cudaMemcpy(&h_num_selected_out, d_num_selected_out, sizeof(vertex), cudaMemcpyDeviceToHost);

    //Free temporary storage
    cudaFree(d_temp_storage);


    return h_num_selected_out;
}

void getGraphInfo(std::ifstream &file, vertex &num_nodes, large_vertex &num_edges_large){
    std::string line;
    //skip if line starts with %
    while(std::getline(file, line)){
        if(line[0] != '%')
            break;
    }
    std::stringstream ss(line);
    vertex tmp;
    ss >> num_nodes >> tmp >> num_edges_large;
    std::cout << "num nodes " << num_nodes << std::endl;
}


void ompPopulateEdgeArray(std::ifstream &file, edge *edges, vertex num_edges, vertex num_nodes,  bool generate_random_weights=false){
  std::string line;
  std::vector<std::string> string_edges(num_edges);
  //populate string edges  
  for(vertex i=0; i<num_edges; i++){
    std::getline(file, line);
    string_edges[i] = line;
  }
  std::cout << "populated string edges. \n";

  if(generate_random_weights){
    #pragma omp parallel for
    for(vertex i=0; i<num_edges; i++){
      vertex u, v;
      large_vertex w;
      std::string edge_string = string_edges[i];
      std::stringstream ss(edge_string);
      ss >> u >> v;
      edges[i].u = u-1;
      edges[i].v = v-1;
      //w = (hash_weight(v)%10) + 1;
      w = ((u-1)*(v-1)%num_nodes) + 1;
      edges[i].w = w;
    }
  }else{
    #pragma omp parallel for
    for(vertex i=0; i<num_edges; i++){
      vertex u, v;
      weight w;
      std::string edge_string = string_edges[i];
      std::stringstream ss(edge_string);
      ss >> u >> v >> w;
      edges[i].u = u-1;
      edges[i].v = v-1;
      edges[i].w = w;
    }
    
  }  
  string_edges.erase(string_edges.begin(), string_edges.end());
  string_edges.shrink_to_fit();
  return;
}

void populateEdgeArray(std::ifstream &file, edge *edges, vertex num_edges, vertex num_nodes
                       ,bool generate_random_weights = false){

    std::string line;
    vertex u, v;
    weight w;
    std::cout << "----------------------------------------------------------------\n";
    if(generate_random_weights){

      for(vertex i = 0; i < num_edges; i++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v ;
          edges[i].u = u-1;
          edges[i].v = v-1;
          //w = (hash_weight(v)%10) + 1;
      	  w = ((u-1)*(v-1)%num_nodes) + 1;
          edges[i].w = w;
      }
    }
    else{
      for(vertex i = 0; i < num_edges; i++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v >> w;
          edges[i].u = u-1;
          edges[i].v = v-1;
          edges[i].w = w;
      }
    }
}


void populateEdgeChunksLarge(std::ifstream &file, edge **chunks_of_edges, large_vertex num_edges, vertex num_nodes,
                        vertex chunk_size, vertex last_chunk_size, int num_chunks,
                        bool generate_random_weights=false, bool is_weight_float=false){
    std::string line;
    vertex u, v;
    weight w;
    weight w_discard;
    if(generate_random_weights && !is_weight_float){
      std::cout << "generating random weights 10\n";
      for(vertex i = 0; i < num_chunks-1; i++){
        for(vertex j = 0; j < chunk_size; j++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v;
          chunks_of_edges[i][j].u = u - 1;
          chunks_of_edges[i][j].v = v - 1;
          //w = (hash_weight(v)%10) + 1;
      	  w = ((u-1)*(v-1)%num_nodes) + 1;
          chunks_of_edges[i][j].w = w;
          //if((i*chunk_size+j)%100000 == 0){
          //  double prog = double(i*chunk_size+j) / num_edges;
          //  printProgress(prog);
          //}
        }
      }
      for(vertex j = 0; j < last_chunk_size; j++){
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> u >> v;
        chunks_of_edges[num_chunks-1][j].u = u - 1;
        chunks_of_edges[num_chunks-1][j].v = v - 1;
        //w = (hash_weight(v)%10) + 1;
        w = ((u-1)*(v-1)%num_nodes) + 1;
        chunks_of_edges[num_chunks-1][j].w = w;
      }
    }
    if(generate_random_weights && is_weight_float){
      std::cout << "discarding float weights\n";
      for(vertex i = 0; i < num_chunks-1; i++){
        for(vertex j = 0; j < chunk_size; j++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v >> w_discard;
          chunks_of_edges[i][j].u = u - 1;
          chunks_of_edges[i][j].v = v - 1;
          //w = (hash_weight(v)%10) + 1;
          w = ((u-1)*(v-1)%num_nodes) + 1;
          chunks_of_edges[i][j].w = w;
          //if((i*chunk_size+j)%100000 == 0){
          //  double prog = double(i*chunk_size+j) / num_edges;
          //  printProgress(prog);
          //}
        }
      }
      for(vertex j = 0; j < last_chunk_size; j++){
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> u >> v;
        chunks_of_edges[num_chunks-1][j].u = u - 1;
        chunks_of_edges[num_chunks-1][j].v = v - 1;
        //w = (hash_weight(v)%10) + 1;
        w = ((u-1)*(v-1)%num_nodes) + 1;
        chunks_of_edges[num_chunks-1][j].w = w;
      }

    }
    if(!generate_random_weights){
      for(vertex i = 0; i < num_chunks-1; i++){
        for(vertex j = 0; j < chunk_size; j++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v >> w;
          chunks_of_edges[i][j].u = u - 1;
          chunks_of_edges[i][j].v = v - 1;
          chunks_of_edges[i][j].w = w;
          //if((i*chunk_size+j)%100000 == 0){
          //  double prog = (i*chunk_size+j)/num_edges;
          //  printProgress(prog);
          //}
        }
      }
      for(vertex j = 0; j < last_chunk_size; j++){
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> u >> v >> w;
        chunks_of_edges[num_chunks-1][j].u = u - 1;
        chunks_of_edges[num_chunks-1][j].v = v - 1;
        chunks_of_edges[num_chunks-1][j].w = w;
      }

    }

}

void populateEdgeChunks(std::ifstream &file, edge **chunks_of_edges, vertex num_edges, vertex num_nodes,
                        vertex chunk_size, vertex last_chunk_size, int num_chunks,
                        bool generate_random_weights=false, bool is_weight_float=false){
    std::string line;
    vertex u, v;
    weight w;
    weight w_discard;
    if(generate_random_weights && !is_weight_float){
      std::cout << "generating random weights 10\n";
      for(vertex i = 0; i < num_chunks-1; i++){
        for(vertex j = 0; j < chunk_size; j++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v;
          chunks_of_edges[i][j].u = u - 1;
          chunks_of_edges[i][j].v = v - 1;
          //w = (hash_weight(v)%10) + 1;
      	  w = ((u-1)*(v-1)%num_nodes) + 1;
          chunks_of_edges[i][j].w = w;
          //if((i*chunk_size+j)%100000 == 0){
          //  double prog = double(i*chunk_size+j) / num_edges;
          //  printProgress(prog);
          //}
        }
      }
      for(vertex j = 0; j < last_chunk_size; j++){
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> u >> v;
        chunks_of_edges[num_chunks-1][j].u = u - 1;
        chunks_of_edges[num_chunks-1][j].v = v - 1;
        //w = (hash_weight(v)%10) + 1;
        w = ((u-1)*(v-1)%num_nodes) + 1;
        chunks_of_edges[num_chunks-1][j].w = w;
      }
    }
    if(generate_random_weights && is_weight_float){
      std::cout << "discarding float weights\n";
      for(vertex i = 0; i < num_chunks-1; i++){
        for(vertex j = 0; j < chunk_size; j++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v >> w_discard;
          chunks_of_edges[i][j].u = u - 1;
          chunks_of_edges[i][j].v = v - 1;
          //w = (hash_weight(v)%10) + 1;
      	  w = ((u-1)*(v-1)%num_nodes) + 1;
          chunks_of_edges[i][j].w = w;
          //if((i*chunk_size+j)%100000 == 0){
          //  double prog = double(i*chunk_size+j) / num_edges;
          //  printProgress(prog);
          //}
        }
      }
      for(vertex j = 0; j < last_chunk_size; j++){
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> u >> v;
        chunks_of_edges[num_chunks-1][j].u = u - 1;
        chunks_of_edges[num_chunks-1][j].v = v - 1;
        //w = (hash_weight(v)%10) + 1;
      	w = ((u-1)*(v-1)%num_nodes) + 1;
        chunks_of_edges[num_chunks-1][j].w = w;
      }

    }
    if(!generate_random_weights){
      for(vertex i = 0; i < num_chunks-1; i++){
        for(vertex j = 0; j < chunk_size; j++){
          std::getline(file, line);
          std::stringstream ss(line);
          ss >> u >> v >> w;
          chunks_of_edges[i][j].u = u - 1;
          chunks_of_edges[i][j].v = v - 1;
          chunks_of_edges[i][j].w = w;
          //if((i*chunk_size+j)%100000 == 0){
          //  double prog = (i*chunk_size+j)/num_edges;
          //  printProgress(prog);
          //}
        }
      }
      for(vertex j = 0; j < last_chunk_size; j++){
        std::getline(file, line);
        std::stringstream ss(line);
        ss >> u >> v >> w;
        chunks_of_edges[num_chunks-1][j].u = u - 1;
        chunks_of_edges[num_chunks-1][j].v = v - 1;
        chunks_of_edges[num_chunks-1][j].w = w;
      }

    }

}

// Function to compare two vertices and return 1 if they are equal
struct EqualVertex{
    __host__ __device__ int operator()(const vertex& a, const vertex& b) const {
        return (a == b) ? 1 : 0;
    }
};

//Function to check if two edges are the same
__device__ void check_if_edges_are_same(edge a, edge b, bool *same){
    if(a.u == b.u && a.v == b.v && a.w == b.w){
        *same = true;
    }else{
        *same = false;
    }
}

//Custom binary operator to find the minimum weighted edge
struct MinWeightedEdge{
    __host__ __device__ edge operator()(const edge& a, const edge& b) const {
        return (a.w < b.w) ? a : b;
    }
};


//Custom binary operator to find the maximum weighted edge
struct MaxWeightedCompare{
    __host__ __device__ bool operator()(const edge& a, const edge& b) const {
        return (a.w > b.w) ;
    }
};


struct check_if_weight_is_greater{
  __host__ __device__ bool operator()(edge a, edge b) const { return a.w > b.w; }
};

void readGraph(edge* edges, vertex num_nodes, vertex num_edges,
               std::ifstream &file, bool generate_random_weight){

  std::string line; 
  vertex u, v;
  weight w;
  if(!generate_random_weight){
    for(vertex i=0; i<num_edges; i++){
      std::getline(file, line);
      std::stringstream ss(line);
      ss >> u >> v >> w;
      edges[i].u = u-1;
      edges[i].v = v-1;
      edges[i].w = w;
      //if(i%100000 == 0){
      //  double prog = i/num_edges;
      //  printProgress(prog);
      //}
    }
    return;
  }
  if(generate_random_weight){
    for(vertex i=0; i<num_edges; i++){
      std::getline(file, line);
      std::stringstream ss(line);
      ss >> u >> v;
      //w = (hash_weight(v)%10) + 1;
      w = ((u-1)*(v-1)%num_nodes) + 1;
      edges[i].u = u-1;
      edges[i].v = v-1;
      edges[i].w = w;
      //if(i%100000 == 0){
      //  double prog = i/num_edges;
      //  printProgress(prog);
      //}

    }
    return;

  }

}







void reset_edge_array_to_sentinel_cpu(std::vector<edge> &edges, vertex num_edges){
  #pragma omp parallel for num_threads(40)
  for(vertex i=0; i<num_edges;i++){
    edges[i].u = sentinel_vertex;
    edges[i].v = sentinel_vertex;
    edges[i].w = sentinel;
  }
}

void reset_outgoing_edge_array_cpu(std::vector<outgoing_edge_id> &outgoing_edges,
                                   vertex num_nodes, vertex num_edges){
  #pragma omp parallel for num_threads(40)
  for(vertex i=0; i<num_nodes; i++){
    outgoing_edges[i].e_id = num_edges +1;
    outgoing_edges[i].w = sentinel;
  }
}

void copy_representative_arrays_cpu(std::vector<vertex> &representative, std::vector<vertex> &iter_representatives,
                                vertex num_nodes){
  #pragma omp parallel for num_threads(40)
  for(vertex i=0; i<num_nodes; i++){
    representative[i] = iter_representatives[i];
  }
}



//Function to check if two edges are the same
void check_if_edges_are_same_cpu(edge a, edge b, bool *same){
    if(a.u == b.u && a.v == b.v && a.w == b.w){
        *same = true;
    }else{
        *same = false;
    }
}


bool check_if_edge_weight_is_not_sentinel(edge e){
  if(e.w != sentinel)
    return true;
  else {
    return false;
  }
}

bool is_not_self_edge(const edge &e){
  return e.u != e.v;
}


void atomic_min_weight_cpu(outgoing_edge_id *existing_id, weight new_weight){
    if(existing_id->w > new_weight){
      existing_id->w = new_weight;
    }
  return;
}

void atomic_min_eid_cpu(outgoing_edge_id *existing_id, vertex new_eid){
  //#pragma omp atomic
    if(existing_id->e_id > new_eid){
      existing_id->e_id = new_eid;
    }
  return;
}

void populate_min_weight_cpu(std::vector<edge> &edges,
                            std::vector<outgoing_edge_id> &best_index,
                            vertex num_edges, vertex num_nodes){
  #pragma omp parallel for num_threads(40)
  for(int i=0; i<num_edges; i++){
    edge tid_e = edges[i];
    vertex tid_u = tid_e.u;
    vertex tid_v = tid_e.v;
    weight tid_w = tid_e.w;
    if(tid_u != tid_v){
      if(best_index[tid_u].w > tid_w){
        vertex oldValue = 0;
        do {
          oldValue = best_index[tid_u].w;
        } while (!__sync_bool_compare_and_swap(&best_index[tid_u].w, oldValue, std::min(oldValue, tid_w)));

        //#pragma omp atomic write
        //best_index[tid_u].w = std::min(best_index[tid_u].w, tid_w);
      }
      //other vertex
      if(best_index[tid_v].w > tid_w){
        vertex oldValue1 = 0;
        do {
          oldValue1 = best_index[tid_v].w;
        } while (!__sync_bool_compare_and_swap(&best_index[tid_v].w, oldValue1, std::min(oldValue1, tid_w)));

        //#pragma omp atomic write
        //best_index[tid_v].w = std::min(best_index[tid_v].w, tid_w);
        //best_index[tid_v].w = tid_w;
      }
    }
  }
}

void populate_eid_cpu(std::vector<edge> &edges,
                  std::vector<outgoing_edge_id> &best_index,
                  vertex num_edges, vertex num_nodes){
  #pragma omp parallel for num_threads(40)
  for(vertex i=0; i<num_edges; i++){
    edge tid_e = edges[i];
    vertex tid_u = tid_e.u;
    vertex tid_v = tid_e.v;
    if(tid_u != tid_v){
      weight tid_w = tid_e.w;
      weight u_best_w = best_index[tid_u].w;
      weight v_best_w = best_index[tid_v].w;
      if(tid_w == u_best_w){
        //atomic_min_eid_cpu(&best_index[tid_u], i);
        //#pragma omp atomic write
        //best_index[tid_u].e_id = i;


        vertex oldValue = 0;
        do {
          oldValue = best_index[tid_u].e_id;
        } while (!__sync_bool_compare_and_swap(&best_index[tid_u].e_id, oldValue, std::min(oldValue, i)));



      }
      if(tid_w == v_best_w){
        //atomic_min_eid_cpu(&best_index[tid_v], i);
        //#pragma omp atomic write
        //best_index[tid_v].e_id = i;

        vertex oldValue = 0;
        do {
          oldValue = best_index[tid_v].e_id;
        } while (!__sync_bool_compare_and_swap(&best_index[tid_v].e_id, oldValue, std::min(oldValue, i)));


      }
    }
  }



  return;
}


void grafting_edges_cpu(std::vector<outgoing_edge_id> &outgoing_edges,
                        std::vector<edge> &iter_edges, std::vector<edge> &edges,
                        std::vector<edge> &msf, std::vector<vertex> &representative,
                        vertex num_nodes, vertex num_edges){
  #pragma omp parallel for num_threads(40)
  for(int i=0; i<num_nodes; i++){
    outgoing_edge_id tid_outgoing_edge_id = outgoing_edges[i];
    vertex tid_outgoing_e_id = tid_outgoing_edge_id.e_id;
    //weight tid_best_w = tid_outgoing_edge_id.w;
    if(tid_outgoing_e_id < (num_edges) ){
      edge tid_best = iter_edges[tid_outgoing_e_id];
      vertex u = tid_best.u;
      vertex v = tid_best.v;
      if(tid_best.u != sentinel_vertex && u != v){
        edge corresponding_edge_in_edges = edges[tid_outgoing_e_id];
        vertex tid_best_u, tid_best_v;
        vertex corresponding_u, corresponding_v;
        weight corresponding_w;
        corresponding_u = corresponding_edge_in_edges.u;
        corresponding_w = corresponding_edge_in_edges.w;
        if(corresponding_u != sentinel_vertex && corresponding_w != sentinel){
          corresponding_v = corresponding_edge_in_edges.v;
          if(tid_best.u == i){
            tid_best_u = tid_best.u;
            tid_best_v = tid_best.v;
          }else{
            tid_best_u = tid_best.v;
            tid_best_v = tid_best.u;
          }
          // This ensures that tid_best_u == tid

          //find the best outgoing edge for tid_best_v
          outgoing_edge_id best_outgoing_edge_of_v = outgoing_edges[tid_best_v];
          vertex best_outgoing_edge_of_v_e_id = best_outgoing_edge_of_v.e_id;
          edge best_outgoing_edge_of_v_edge = iter_edges[best_outgoing_edge_of_v_e_id];
          //check if tid_best and best_outgoing_edge_of_v are the same
          bool same = false;
          check_if_edges_are_same_cpu(tid_best, best_outgoing_edge_of_v_edge, &same);
          if(same && (tid_best_u < tid_best_v)){
            msf[i].w = sentinel;
          }
          else if(same && (tid_best_u > tid_best_v)){
          //else if((tid_best_u > tid_best_v)){
            representative[i] = tid_best_v;
            msf[i].u = corresponding_u;
            msf[i].v = corresponding_v;
            msf[i].w = corresponding_w;
          }
          else if(!same){
            representative[i] = tid_best_v;
            msf[i].u = corresponding_u;
            msf[i].v = corresponding_v;
            msf[i].w = corresponding_w;
          }
        }
      }
    }
    else{
      //d_representative[tid] = -1;
      msf[i].w = sentinel;
    }
  }
  return;
}



void shortcutting_step_cpu(std::vector<vertex> &representative,
                vertex num_nodes, weight sentinel, std::vector<vertex> &iter_representatives){
  #pragma omp parallel for num_threads(40)
  for(int i=0; i<num_nodes; i++){
    //Find the root of each tid. Break when root is found 
    vertex r = i;
    vertex root = representative[r];
    while(true){
      if(root == representative[root]){
        break;
      }
      root = representative[root];
    }
    //Set the representative of each tid to the root
    iter_representatives[i] = root;
  }
  return;
}

void relabelling_step_cpu(std::vector<edge> &edges, std::vector<vertex> &representative,
                      vertex num_edges, vertex num_nodes){
  #pragma omp parallel for num_threads(40)
  for(vertex i=0; i<num_edges; i++){
    vertex u = edges[i].u;
    vertex v = edges[i].v;
    //if(u == v){
    //  edges[i].w = sentinel;
    //}
    if(u != sentinel_vertex && u != v){
      if(u < num_nodes && v < num_nodes){
        vertex new_u = representative[u];
        vertex new_v = representative[v];
        if(edges[i].u != sentinel_vertex){
          edges[i].u = new_u;
          edges[i].v = new_v;
        }
      }
      else{
        edges[i].u = sentinel_vertex;
        edges[i].v = sentinel_vertex;
        edges[i].w = sentinel;
      }
    }
  }
}


void filtering_step_cpu(std::vector<edge> &new_edges, std::vector<vertex> &representative,
                    vertex num_edges, vertex num_nodes){

  #pragma omp parallel for num_threads(40)
  for(vertex i=0; i<num_edges; i++){
    vertex u = new_edges[i].u;
    weight w = new_edges[i].w;
    if(u != sentinel_vertex && w != sentinel){
      vertex v = new_edges[i].v;
      if(u<num_nodes && v<num_nodes){
        if(representative[u] != representative[v]){
          new_edges[i].u = u;
          new_edges[i].v = v;
          new_edges[i].w = new_edges[i].w;
        }else{
          new_edges[i].u = sentinel_vertex;
          new_edges[i].v = sentinel_vertex;
          new_edges[i].w = sentinel;
        }
      }
    }
  }
}



void boruvka_for_edge_vector_cpu(std::vector<edge> edges, vertex num_nodes,
                                vertex num_edges, std::vector<edge> &final_msf,
                                vertex &number_of_filled_edges,
                                std::vector<vertex> representative_array,     
                                std::vector<vertex> representative_array_iter,     
                                std::vector<edge> iter_edges, std::vector<edge> iter_msf,
                                std::vector<outgoing_edge_id> outgoing_edges,
                                std::string result = "res.txt"){
  //print the graph params
  std::cout << "num nodes " << num_nodes << std::endl;
  std::cout << "num_edges " << num_edges << std::endl;

  int iter = 0;
  vertex offset_msf = 0;


  auto start_cpu = std::chrono::high_resolution_clock::now();
  while(true){
    std::cout << "iter " << iter << std::endl;

    //reset msf array to sentinel
    reset_edge_array_to_sentinel_cpu(iter_msf, num_nodes);
    //reset outgoing edges array
    reset_outgoing_edge_array_cpu(outgoing_edges, num_nodes, num_edges);
    std::cout << "done reset\n";

    //min weight finding step
    populate_min_weight_cpu(iter_edges,
                            outgoing_edges,
                            num_edges, num_nodes);

    std::cout << "done populate min weight\n";
    //populate e_id
    populate_eid_cpu(iter_edges, outgoing_edges,
                    num_edges, num_nodes);

    std::cout << iter <<  "\n";
    //grafting edges
    std::cout << "grafting edges \n";
    grafting_edges_cpu(outgoing_edges,
                      iter_edges, edges,
                      iter_msf, representative_array,
                      num_nodes, num_edges);
    std::cout << "grafted edges \n";
    std::cout << "\n";

    //find number of edges in msf where edge weight is not sentinel

    vertex candidate_edges = std::count_if(std::execution::par, iter_msf.begin(), iter_msf.begin()+num_nodes, 
                              check_if_edge_weight_is_not_sentinel);

    std::cout << "counted \n";

    std::copy_if(std::execution::par, iter_msf.begin(), iter_msf.end(), final_msf.begin()+offset_msf, 
                check_if_edge_weight_is_not_sentinel);
    std::cout << "offset " << offset_msf<<std::endl;
    std::cout << "candidate_edges " << candidate_edges << std::endl;
    std::cout << "completed copy \n";

    offset_msf += candidate_edges;

    //shortcutting_step_cpu
    shortcutting_step_cpu(representative_array,
                num_nodes, sentinel, representative_array_iter);
    std::cout << "completed shortcut \n";


    //copy representative_array_iter to representative_array
    copy_representative_arrays_cpu(representative_array, representative_array_iter,
                               num_nodes);
    std::cout << "completed copy \n";

    //Relabelling steps
    relabelling_step_cpu(iter_edges, representative_array, num_edges, num_nodes);
    std::cout << "completed relabelling\n";

    //filtering steps for edges
    filtering_step_cpu(edges, representative_array, num_edges, num_nodes);
    std::cout << "completed filtering\n";

    //Remove edges in 'edges' where edge weight is sentinel
    //convert edges array to std::vector
    //std::vector<edge> edges_vector(edges, edges+num_edges);
    //std::vector<edge> iter_edges_vector(iter_edges, iter_edges+num_edges);

    //parallel_remove_if(edges, num_edges, num_nodes);
    //std::remove_if(std::execution::par, edges.begin(), 
    //                edges.begin()+num_edges, [](edge e){ 
    //                return e.w == sentinel;});
    std::cout << "completed remove1 \n";
    
    //Remove self loop edges in 'iter_edges'
    //auto iter_edges_start(iter_edges);
    

    //std::remove_if(std::execution::par, iter_edges.begin(), 
    //               iter_edges.begin()+num_edges, [](edge e){return e.u == e.v;});
    //std::cout << "completed remove2 \n";

    vertex new_num_edges=1;
    auto does_non_self_exist = std::find_if(std::execution::par, iter_edges.begin(),
                                        iter_edges.end(), 
                                        is_not_self_edge);
    if(does_non_self_exist == iter_edges.end()){
      std::cout << "no more self edges\n";
      new_num_edges =0;
    }
    std::cout << "INDEX of same edges " << std::distance(iter_edges.begin(), does_non_self_exist) << "\n";

    std::cout << "completed counting new num edges\n";


    //for(int i=0; i<num_edges; i++){
    //  std::cout << iter_edges[i].u << " " << iter_edges[i].v << "\n";
    //}
    std::cout << "Number of non self edges  "  << new_num_edges << "\n";

    //update num_edges
    //num_edges = new_num_edges;
  
    std::cout << "NEW NUM EDGES " << num_edges << "\n";
    if(new_num_edges == 0){
      std::cout << "done in " << iter << " iterations \n";
      std::cout << "filled " << offset_msf << " edges \n";
      number_of_filled_edges = offset_msf;
      break;
    }

    iter++;
    if(iter == 50){
      break;
    }

  }
  auto stop_cpu = std::chrono::high_resolution_clock::now();
  auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(
                      stop_cpu-start_cpu);
  std::cout << "Time to execute " << duration_cpu.count() << " microseconds"<< std::endl;

  //free(iter_edges);

}


void boruvka_for_edge_vector_cpu_with_remove_if(std::vector<edge> edges, vertex num_nodes,
                                vertex num_edges, std::vector<edge> &final_msf,
                                vertex &number_of_filled_edges,
                                std::vector<vertex> representative_array,     
                                std::vector<vertex> representative_array_iter,     
                                std::vector<edge> iter_edges, std::vector<edge> iter_msf,
                                std::vector<outgoing_edge_id> outgoing_edges,
                                std::string result = "res.txt"){
  //print the graph params
  std::cout << "num nodes " << num_nodes << std::endl;
  std::cout << "num_edges " << num_edges << std::endl;

  int iter = 0;
  vertex offset_msf = 0;


  auto start_cpu = std::chrono::high_resolution_clock::now();
  while(true){
    std::cout << "iter " << iter << std::endl;

    //reset msf array to sentinel
    reset_edge_array_to_sentinel_cpu(iter_msf, num_nodes);
    //reset outgoing edges array
    reset_outgoing_edge_array_cpu(outgoing_edges, num_nodes, num_edges);
    std::cout << "done reset\n";

    //min weight finding step
    populate_min_weight_cpu(iter_edges,
                            outgoing_edges,
                            num_edges, num_nodes);

    std::cout << "done populate min weight\n";
    //populate e_id
    populate_eid_cpu(iter_edges, outgoing_edges,
                    num_edges, num_nodes);

    std::cout << iter <<  "\n";
    //grafting edges
    std::cout << "grafting edges \n";
    grafting_edges_cpu(outgoing_edges,
                      iter_edges, edges,
                      iter_msf, representative_array,
                      num_nodes, num_edges);
    std::cout << "grafted edges \n";
    std::cout << "\n";

    //find number of edges in msf where edge weight is not sentinel

    vertex candidate_edges = std::count_if(std::execution::par, iter_msf.begin(), iter_msf.begin()+num_nodes, 
                              check_if_edge_weight_is_not_sentinel);

    std::cout << "counted \n";

    std::copy_if(std::execution::par, iter_msf.begin(), iter_msf.end(), final_msf.begin()+offset_msf, 
                check_if_edge_weight_is_not_sentinel);
    std::cout << "offset " << offset_msf<<std::endl;
    std::cout << "candidate_edges " << candidate_edges << std::endl;
    std::cout << "completed copy \n";

    offset_msf += candidate_edges;

    //shortcutting_step_cpu
    shortcutting_step_cpu(representative_array,
                num_nodes, sentinel, representative_array_iter);
    std::cout << "completed shortcut \n";


    //copy representative_array_iter to representative_array
    copy_representative_arrays_cpu(representative_array, representative_array_iter,
                               num_nodes);
    std::cout << "completed copy \n";

    //Relabelling steps
    relabelling_step_cpu(iter_edges, representative_array, num_edges, num_nodes);
    std::cout << "completed relabelling\n";

    //filtering steps for edges
    filtering_step_cpu(edges, representative_array, num_edges, num_nodes);
    std::cout << "completed filtering\n";

    //Remove edges in 'edges' where edge weight is sentinel
    //convert edges array to std::vector

    std::remove_if(std::execution::par, edges.begin(), 
                    edges.begin()+num_edges, [](edge e){ 
                    return e.w == sentinel;});
    std::cout << "completed remove1 \n";
    
    //Remove self loop edges in 'iter_edges'
    //auto iter_edges_start(iter_edges);
    

    //std::remove_if(std::execution::par, iter_edges.begin(), 
    //               iter_edges.begin()+num_edges, [](edge e){return e.u == e.v;});
    //std::cout << "completed remove2 \n";

    //vertex new_num_edges=1;
    //auto does_non_self_exist = std::find_if(std::execution::par, iter_edges.begin(),
    //                                    iter_edges.end(), 
    //                                    is_not_self_edge);
    //if(does_non_self_exist == iter_edges.end()){
    //  std::cout << "no more self edges\n";
    //  new_num_edges =0;
    //}
    //std::cout << "INDEX of same edges " << std::distance(iter_edges.begin(), does_non_self_exist) << "\n";

    vertex new_num_edges = std::count_if(std::execution::par, iter_edges.begin(),
                                        iter_edges.begin()+num_edges, [](edge e){return e.u != e.v;});
    std::cout << "completed counting new num edges\n";


    //for(int i=0; i<num_edges; i++){
    //  std::cout << iter_edges[i].u << " " << iter_edges[i].v << "\n";
    //}
    std::cout << "Number of non self edges  "  << new_num_edges << "\n";

    //update num_edges
    num_edges = new_num_edges;
  
    std::cout << "NEW NUM EDGES " << num_edges << "\n";
    if(new_num_edges == 0){
      std::cout << "done in " << iter << " iterations \n";
      std::cout << "filled " << offset_msf << " edges \n";
      number_of_filled_edges = offset_msf;
      break;
    }

    iter++;
    if(iter == 50){
      break;
    }

  }
  auto stop_cpu = std::chrono::high_resolution_clock::now();
  auto duration_cpu = std::chrono::duration_cast<std::chrono::microseconds>(
                      stop_cpu-start_cpu);
  std::cout << "Time to execute " << duration_cpu.count() << " microseconds"<< std::endl;

  //free(iter_edges);

}

void kruskal_for_edge_vector_cpu(std::vector<edge> edges,
                                 std::vector<edge> &final_msf,
                                 DSU s, vertex number_of_filled_edges){
  kruskals_mst(edges, s, final_msf, number_of_filled_edges);
  return;
}

std::string header_for_weighted_mtx = R"(%%MatrixMarket matrix coordinate integer general
%-------------------------------------------------------------------------------
%-------------------------------------------------------------------------------
)";












struct indexedEdge {
  vertexId u; vertexId v; edgeId id; edgeWeight w;
  indexedEdge(vertexId u, vertexId v, edgeId id, edgeWeight w)
    : u(u), v(v), id(id), w(w){}
  indexedEdge() {};
};

using reservation = pbbs::reservation<edgeId>;

struct UnionFindStep {
  parlay::sequence<indexedEdge> &E;
  parlay::sequence<reservation> &R;
  unionFind<vertexId> &UF;
  parlay::sequence<bool> &inST;
  UnionFindStep(parlay::sequence<indexedEdge> &E,
		unionFind<vertexId> &UF,
		parlay::sequence<reservation> &R,
		parlay::sequence<bool> &inST) :
    E(E), R(R), UF(UF), inST(inST) {}

  bool reserve(edgeId i) {
    vertexId u = E[i].u = UF.find(E[i].u);
    vertexId v = E[i].v = UF.find(E[i].v);
    if (u != v) {
      R[v].reserve(i);
      R[u].reserve(i);
      return true;
    } else return false;
  }

  bool commit(edgeId i) {
    vertexId u = E[i].u;
    vertexId v = E[i].v;
    if (R[v].check(i)) {
      R[u].checkReset(i); 
      UF.link(v, u); 
      inST[E[i].id] = true;
      return true;}
    else if (R[u].check(i)) {
      UF.link(u, v);
      inST[E[i].id] = true;
      return true; }
    else return false;
  }
};

parlay::sequence<edgeId> mst(wghEdgeArray<vertexId,edgeWeight> &E) { 
  timer t("mst", true);
  size_t m = E.m;
  size_t n = E.n;
  size_t k = std::min<size_t>(5 * n / 4, m);

  // equal edge weights will prioritize the earliest one
  auto edgeLess = [&] (indexedEdge a, indexedEdge b) {
    return (a.w < b.w) || ((a.w == b.w) && (a.id < b.id));};

  // tag each edge with an index
  auto IW = parlay::delayed_seq<indexedEdge>(m, [&] (size_t i) {
      return indexedEdge(E[i].u, E[i].v, i, E[i].weight);});

  auto IW1 = parlay::sort(IW, edgeLess);
  t.next("sort edges");

  parlay::sequence<bool> mstFlags(m, false);
  unionFind<vertexId> UF(n);
  parlay::sequence<reservation> R(n);
  UnionFindStep UFStep1(IW1, UF, R,  mstFlags);
  pbbs::speculative_for<vertexId>(UFStep1, 0, IW1.size(), 20, false);
  t.next("union find loop");

  parlay::sequence<edgeId> mst = parlay::pack_index<edgeId>(mstFlags);
  t.next("pack out results");

  return mst;
}

// Helper function to convert edge* to parlay::sequence<wghEdge>
parlay::sequence<wghEdge<vertexId, vertexId>> convert_to_wghEdges(const edge_pbbs* edges, size_t num_edges) {
    parlay::sequence<wghEdge<vertexId, vertexId>> wgh_edges(num_edges);
    for (size_t i = 0; i < num_edges; ++i) {
        wgh_edges[i] = wghEdge<vertexId, vertexId>(edges[i].u, edges[i].v, edges[i].w);
    }
    return wgh_edges;
}

// Function to convert edge* (array of edge) to wghEdgeArray
wghEdgeArray<vertexId, vertexId> create_wghEdgeArray(const edge_pbbs* edges, size_t num_edges) {
    // 1. Find the maximum vertex ID + 1 to determine 'n' (number of vertices).
    size_t max_vertex = 0;
    for (size_t i = 0; i < num_edges; ++i) {
        max_vertex = std::max(max_vertex, (size_t)std::max(edges[i].u, edges[i].v));
    }
    size_t num_vertices = max_vertex + 1;

    // 2. Convert to parlay::sequence<wghEdge> using the helper function
    parlay::sequence<wghEdge<vertexId, vertexId>> parlay_edges = convert_to_wghEdges(edges, num_edges);

    // 3. Create the wghEdgeArray
    return wghEdgeArray<vertexId, vertexId>(std::move(parlay_edges), num_vertices);
}


// Function to convert parlay::sequence<edgeId> to edge*
edge* get_mst_edges_as_array(wghEdgeArray<vertexId, vertexId>& edge_array, const parlay::sequence<edgeId>& mst_edges, size_t& num_mst_edges) {
    num_mst_edges = mst_edges.size();
    edge* mst_edge_array = new edge[num_mst_edges];

    // Parallel conversion using parlay::parallel_for
    parlay::parallel_for(0, num_mst_edges, [&](size_t i) {
        edgeId edge_index = mst_edges[i];
        wghEdge<vertexId, vertexId> original_edge = edge_array[edge_index];
        mst_edge_array[i].u = static_cast<uint32_t>(original_edge.u);
        mst_edge_array[i].v = static_cast<uint32_t>(original_edge.v);
        mst_edge_array[i].w = original_edge.weight;
    });

    return mst_edge_array;
}

edge* cpu_compute_pbbs(size_t &num_edges_mst, wghEdgeArray<vertexId, vertexId> edge_array){

      parlay::sequence<edgeId> pbbs_res_mst = mst(edge_array);
      std::cout << "done pbbs mst\n";

      edge* mst_array = get_mst_edges_as_array(edge_array, pbbs_res_mst, num_edges_mst);
      return mst_array;
}
