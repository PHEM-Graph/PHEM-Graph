#ifndef UTILS_CU
#define UTILS_CU

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/atomic>
#include <random>
#include <stdlib.h>
#include <omp.h>
#include "constants.h"
#include <execution>
#define BLOCKSIZE 1024
#include "parlay/sequence.h"
#include "parlay/parallel.h"
#include "common/graph.h"
#include "common/speculative_for.h"
#include "common/get_time.h"
#include "algorithm/union_find.h"
#include "algorithm/kth_smallest.h"
#include "MST.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>



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

void getGraphInfo(std::ifstream &file, vertex &num_nodes, large_vertex &num_edges_large) {
  std::string line;
  // Skip comments
  while (std::getline(file, line)) {
    if (line[0] != '%') break;
  }
  
  std::stringstream ss(line);
  vertex tmp;
  ss >> num_nodes >> tmp >> num_edges_large;
  
  // Reset file position if needed
  if (file.tellg() != std::streampos(-1)) {
    file.seekg(0, std::ios::beg);
    // Skip comments again
    while (std::getline(file, line)) {
      if (line[0] != '%') break;
    }
  }
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

void fastPopulateEdgeArray(std::ifstream &file, edge *edges, vertex num_edges, vertex num_nodes, bool generate_random_weights=false) {
  // Read all lines into memory first
  std::vector<std::string> lines(num_edges);
  
  #pragma omp parallel
  {
    // Each thread reads a chunk of the file
    #pragma omp for schedule(dynamic, 1024)
    for (vertex i = 0; i < num_edges; i++) {
      std::string line;
      #pragma omp critical
      {
        std::getline(file, line);
      }
      lines[i] = line;
    }
    
    // Now parse the lines in parallel
    #pragma omp for schedule(dynamic, 1024)
    for (vertex i = 0; i < num_edges; i++) {
      vertex u, v;
      weight w;
      std::stringstream ss(lines[i]);
      
      if (generate_random_weights) {
        ss >> u >> v;
        w = ((u-1)*(v-1)%num_nodes) + 1;
      } else {
        ss >> u >> v >> w;
      }
      
      edges[i].u = u - 1;
      edges[i].v = v - 1;
      edges[i].w = w;
    }
  }
  
  // Clear the vector to free memory
  lines.clear();
  lines.shrink_to_fit();
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
    std::cout << "Starting memory-mapped file loading for edge chunks..." << std::endl;
    
    // We need to get the filename from the file stream
    // Since we can't directly get it, we'll need to use a temporary file
    std::string temp_filename = "temp_mmap_file.txt";
    
    // Save current position in the file
    std::streampos current_pos = file.tellg();
    
    // Rewind to beginning
    file.seekg(0, std::ios::beg);
    
    // Create a temporary file with the same content
    std::ofstream temp_file(temp_filename, std::ios::binary);
    if (!temp_file) {
        std::cerr << "Error creating temporary file for memory mapping" << std::endl;
        return;
    }
    
    // Copy content
    temp_file << file.rdbuf();
    temp_file.close();
    
    // Restore original position
    file.seekg(current_pos);
    
    // Now open the temporary file for memory mapping
    int fd = open(temp_filename.c_str(), O_RDONLY);
    if (fd == -1) {
        std::cerr << "Error opening temporary file for memory mapping: " << strerror(errno) << std::endl;
        return;
    }
    
    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        std::cerr << "Error getting file stats: " << strerror(errno) << std::endl;
        close(fd);
        return;
    }
    
    // Map the file into memory
    char* file_data = (char*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        std::cerr << "Error memory mapping file: " << strerror(errno) << std::endl;
        close(fd);
        return;
    }
    
    std::cout << "File mapped successfully, size: " << sb.st_size << " bytes" << std::endl;
    std::cout << "Processing " << num_edges << " edges for " << num_nodes << " nodes" << std::endl;
    
    // Find the start of the data (after header comments and dimensions line)
    char* data_start = file_data;
    char* file_end = file_data + sb.st_size;
    
    // Skip comment lines (starting with %)
    while (data_start < file_end && *data_start == '%') {
        // Skip to end of line
        while (data_start < file_end && *data_start != '\n') {
            data_start++;
        }
        // Skip the newline character
        if (data_start < file_end) data_start++;
    }
    
    // Skip the dimensions line (first non-comment line)
    while (data_start < file_end && *data_start != '\n') {
        data_start++;
    }
    if (data_start < file_end) data_start++; // Skip the newline
    
    std::cout << "Header skipped, starting to process edge chunks" << std::endl;
    
    // Count the number of newlines to determine line positions
    std::vector<char*> line_starts;
    line_starts.reserve(num_edges);
    
    char* current = data_start;
    line_starts.push_back(current);
    
    while (current < file_end) {
        if (*current == '\n') {
            if (current + 1 < file_end) {
                line_starts.push_back(current + 1);
            }
        }
        current++;
    }
    
    // Make sure we don't try to process more lines than we have
    size_t actual_lines = std::min(line_starts.size(), static_cast<size_t>(num_edges));
    
    std::cout << "Found " << actual_lines << " data lines" << std::endl;
    
    // Process the edge data in parallel for each chunk
    #pragma omp parallel
    {
        // Process each chunk
        #pragma omp for schedule(dynamic, 1)
        for (int chunk_idx = 0; chunk_idx < num_chunks - 1; chunk_idx++) {
            vertex chunk_start = chunk_idx * chunk_size;
            
            // Process each edge in the chunk
            #pragma omp parallel for schedule(dynamic, 1024)
            for (vertex i = 0; i < chunk_size; i++) {
                large_vertex edge_idx = chunk_start + i;
                if (edge_idx >= actual_lines) continue;
                
                char* line = line_starts[edge_idx];
                char* next_line = (edge_idx < line_starts.size() - 1) ? line_starts[edge_idx + 1] : file_end;
                size_t line_length = next_line - line - 1; // -1 to exclude newline
                
                // Skip empty lines
                if (line_length <= 0) continue;
                
                // Parse the line
                vertex u = 0, v = 0;
                weight w = 0;
                
                // Parse u
                while (line < next_line && (*line == ' ' || *line == '\t')) line++; // Skip leading whitespace
                while (line < next_line && *line >= '0' && *line <= '9') {
                    u = u * 10 + (*line - '0');
                    line++;
                }
                
                // Skip whitespace
                while (line < next_line && (*line == ' ' || *line == '\t')) line++;
                
                // Parse v
                while (line < next_line && *line >= '0' && *line <= '9') {
                    v = v * 10 + (*line - '0');
                    line++;
                }
                
                // Parse or generate w
                if (generate_random_weights && !is_weight_float) {
                    w = ((u-1)*(v-1)%num_nodes) + 1;
                } else if (generate_random_weights && is_weight_float) {
                    // Skip whitespace to discard float weight
                    while (line < next_line && (*line == ' ' || *line == '\t')) line++;
                    
                    // Skip over the float value
                    while (line < next_line && ((*line >= '0' && *line <= '9') || *line == '.' || *line == 'e' || *line == 'E' || *line == '+' || *line == '-')) {
                        line++;
                    }
                    
                    w = ((u-1)*(v-1)%num_nodes) + 1;
                } else {
                    // Skip whitespace
                    while (line < next_line && (*line == ' ' || *line == '\t')) line++;
                    
                    // Parse w
                    while (line < next_line && *line >= '0' && *line <= '9') {
                        w = w * 10 + (*line - '0');
                        line++;
                    }
                }
                
                // Store the edge
                chunks_of_edges[chunk_idx][i].u = u - 1;
                chunks_of_edges[chunk_idx][i].v = v - 1;
                chunks_of_edges[chunk_idx][i].w = w;
            }
        }
        
        // Process the last chunk separately (it might have a different size)
        #pragma omp single
        {
            vertex chunk_start = (num_chunks - 1) * chunk_size;
            
            #pragma omp parallel for schedule(dynamic, 1024)
            for (vertex i = 0; i < last_chunk_size; i++) {
                large_vertex edge_idx = chunk_start + i;
                if (edge_idx >= actual_lines) continue;
                
                char* line = line_starts[edge_idx];
                char* next_line = (edge_idx < line_starts.size() - 1) ? line_starts[edge_idx + 1] : file_end;
                size_t line_length = next_line - line - 1; // -1 to exclude newline
                
                // Skip empty lines
                if (line_length <= 0) continue;
                
                // Parse the line
                vertex u = 0, v = 0;
                weight w = 0;
                
                // Parse u
                while (line < next_line && (*line == ' ' || *line == '\t')) line++; // Skip leading whitespace
                while (line < next_line && *line >= '0' && *line <= '9') {
                    u = u * 10 + (*line - '0');
                    line++;
                }
                
                // Skip whitespace
                while (line < next_line && (*line == ' ' || *line == '\t')) line++;
                
                // Parse v
                while (line < next_line && *line >= '0' && *line <= '9') {
                    v = v * 10 + (*line - '0');
                    line++;
                }
                
                // Parse or generate w
                if (generate_random_weights && !is_weight_float) {
                    w = ((u-1)*(v-1)%num_nodes) + 1;
                } else if (generate_random_weights && is_weight_float) {
                    // Skip whitespace to discard float weight
                    while (line < next_line && (*line == ' ' || *line == '\t')) line++;
                    
                    // Skip over the float value
                    while (line < next_line && ((*line >= '0' && *line <= '9') || *line == '.' || *line == 'e' || *line == 'E' || *line == '+' || *line == '-')) {
                        line++;
                    }
                    
                    w = ((u-1)*(v-1)%num_nodes) + 1;
                } else {
                    // Skip whitespace
                    while (line < next_line && (*line == ' ' || *line == '\t')) line++;
                    
                    // Parse w
                    while (line < next_line && *line >= '0' && *line <= '9') {
                        w = w * 10 + (*line - '0');
                        line++;
                    }
                }
                
                // Store the edge
                chunks_of_edges[num_chunks-1][i].u = u - 1;
                chunks_of_edges[num_chunks-1][i].v = v - 1;
                chunks_of_edges[num_chunks-1][i].w = w;
            }
        }
    }
    
    // Clean up
    munmap(file_data, sb.st_size);
    close(fd);
    
    // Remove the temporary file
    std::remove(temp_filename.c_str());
    
    std::cout << "Memory-mapped file processing for edge chunks complete" << std::endl;
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
  parlay::internal::timer t("mst", true);
  size_t m = E.m;
  size_t n = E.n;
  size_t k = std::min<size_t>(5 * n / 4, m);

  // equal edge weights will prioritize the earliest one
  auto edgeLess = [&] (indexedEdge a, indexedEdge b) { 
    return (a.w < b.w) || ((a.w == b.w) && (a.id < b.id));};

  // tag each edge with an index
  auto IW = parlay::delayed_seq<indexedEdge>(m, [&] (size_t i) {
      return indexedEdge(E[i].u, E[i].v, i, E[i].weight);});

  indexedEdge kth = pbbs::approximate_kth_smallest(IW, k, edgeLess);
  t.next("approximate kth smallest");
  
  auto IW1 = parlay::filter(IW, [&] (indexedEdge e) {
      return edgeLess(e, kth);}); //edgeLess(e, kth);});
  t.next("filter those less than kth smallest as prefix");
  
  parlay::sort_inplace(IW1, edgeLess);
  t.next("sort prefix");

  parlay::sequence<bool> mstFlags(m, false);
  unionFind<vertexId> UF(n);
  parlay::sequence<reservation> R(n);
  UnionFindStep UFStep1(IW1, UF, R,  mstFlags);
  pbbs::speculative_for<vertexId>(UFStep1, 0, IW1.size(), 5, false);
  t.next("union find loop on prefix");

  auto IW2 = parlay::filter(IW, [&] (indexedEdge e) {
      return !edgeLess(e, kth) && UF.find(e.u) != UF.find(e.v);});
  t.next("filter those that are not self edges");
  
  parlay::sort_inplace(IW2, edgeLess);
  t.next("sort remaining");

  UnionFindStep UFStep2(IW2, UF, R, mstFlags);
  pbbs::speculative_for<vertexId>(UFStep2, 0, IW2.size(), 5, false);
  t.next("union find loop on remaining");

  parlay::sequence<edgeId> mst = parlay::internal::pack_index<edgeId>(mstFlags);
  t.next("pack out results");

  //cout << "n=" << n << " m=" << m << " nInMst=" << size << endl;
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

void mmapPopulateEdgeArray(const std::string& filename, edge *edges, vertex num_edges, vertex num_nodes, bool generate_random_weights=false) {
  std::cout << "Starting memory-mapped file loading..." << std::endl;
  
  // Open the file
  int fd = open(filename.c_str(), O_RDONLY);
  if (fd == -1) {
    std::cerr << "Error opening file for memory mapping: " << strerror(errno) << std::endl;
    return;
  }
  
  // Get file size
  struct stat sb;
  if (fstat(fd, &sb) == -1) {
    std::cerr << "Error getting file stats: " << strerror(errno) << std::endl;
    close(fd);
    return;
  }
  
  // Map the file into memory
  char* file_data = (char*)mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (file_data == MAP_FAILED) {
    std::cerr << "Error memory mapping file: " << strerror(errno) << std::endl;
    close(fd);
    return;
  }
  
  std::cout << "File mapped successfully, size: " << sb.st_size << " bytes" << std::endl;
  std::cout << "Processing " << num_edges << " edges for " << num_nodes << " nodes" << std::endl;
  
  // Find the start of the data (after header comments and dimensions line)
  char* data_start = file_data;
  char* file_end = file_data + sb.st_size;
  
  // Skip comment lines (starting with %)
  while (data_start < file_end && *data_start == '%') {
    // Skip to end of line
    while (data_start < file_end && *data_start != '\n') {
      data_start++;
    }
    // Skip the newline character
    if (data_start < file_end) data_start++;
  }
  
  // Skip the dimensions line (first non-comment line)
  while (data_start < file_end && *data_start != '\n') {
    data_start++;
  }
  if (data_start < file_end) data_start++; // Skip the newline
  
  std::cout << "Header skipped, starting to process edge data" << std::endl;
  
  // Count the number of newlines to determine line positions
  std::vector<char*> line_starts;
  line_starts.reserve(num_edges);
  
  char* current = data_start;
  line_starts.push_back(current);
  
  while (current < file_end) {
    if (*current == '\n') {
      if (current + 1 < file_end) {
        line_starts.push_back(current + 1);
      }
    }
    current++;
  }
  
  // Make sure we don't try to process more lines than we have
  size_t actual_lines = std::min(line_starts.size(), static_cast<size_t>(num_edges));
  
  std::cout << "Found " << actual_lines << " data lines" << std::endl;
  
  // Process the edge data in parallel
  #pragma omp parallel for schedule(dynamic, 1024)
  for (vertex i = 0; i < actual_lines; i++) {
    if (i >= num_edges) continue;
    
    char* line = line_starts[i];
    char* next_line = (i < line_starts.size() - 1) ? line_starts[i + 1] : file_end;
    size_t line_length = next_line - line - 1; // -1 to exclude newline
    
    // Skip empty lines
    if (line_length <= 0) continue;
    
    // Parse the line
    vertex u = 0, v = 0;
    weight w = 0;
    
    // Parse u
    while (line < next_line && (*line == ' ' || *line == '\t')) line++; // Skip leading whitespace
    while (line < next_line && *line >= '0' && *line <= '9') {
      u = u * 10 + (*line - '0');
      line++;
    }
    
    // Skip whitespace
    while (line < next_line && (*line == ' ' || *line == '\t')) line++;
    
    // Parse v
    while (line < next_line && *line >= '0' && *line <= '9') {
      v = v * 10 + (*line - '0');
      line++;
    }
    
    // Parse or generate w
    if (generate_random_weights) {
      w = ((u-1)*(v-1)%num_nodes) + 1;
    } else {
      // Skip whitespace
      while (line < next_line && (*line == ' ' || *line == '\t')) line++;
      
      // Parse w
      while (line < next_line && *line >= '0' && *line <= '9') {
        w = w * 10 + (*line - '0');
        line++;
      }
    }
    
    // Store the edge
    if(u > v){
      edges[i].u = v - 1;
      edges[i].v = u - 1;
    } else {
      edges[i].u = u - 1;
      edges[i].v = v - 1;
    }
    edges[i].w = w;
    
    // Progress indicator (only from main thread)
    if (i % 1000000 == 0 && omp_get_thread_num() == 0) {
      std::cout << "Processed " << i << " edges (" 
                << (i * 100.0 / num_edges) << "%)" << std::endl;
    }
  }
  
  // Clean up
  munmap(file_data, sb.st_size);
  close(fd);
  
  std::cout << "Memory-mapped file processing complete" << std::endl;
}

#endif // UTILS_CU
