#include "src/new_boruvka.cu"
#include <omp.h>


int main(int argc, char* argv[]){
  std::string filename;
  std::string result_file;
  filename = argv[1];
  bool debug = atoi(argv[2]);
  result_file = argv[3];
  vertex chunk_size = 100;
  bool generate_random_weights = atoi(argv[4]);
  bool use_malloc_managed = atoi(argv[5]);
  bool use_streamline = atoi(argv[6]);
  bool use_cpu_only = atoi(argv[7]);
  bool cpu_gpu_streamline = atoi(argv[8]);
  bool save_weighted_graph = atoi(argv[9]);
  int num_chunks_ideal = atoi(argv[10]);
  
  // Note: We need to move this code inside new_boruvka where we have access to num_edges_large
  // bool use_mmap = (num_edges_large > 10000000);
  // if (use_mmap) {
  //   mmapPopulateEdgeArray(filename, edges, num_edges, num_nodes, generate_random_weights);
  // } else {
  //   fastPopulateEdgeArray(file, edges, num_edges, num_nodes, generate_random_weights);
  // }

  new_boruvka(filename, result_file, debug, chunk_size, num_chunks_ideal,
              generate_random_weights, use_malloc_managed, use_streamline, 
              use_cpu_only, cpu_gpu_streamline, save_weighted_graph);
  std::cout << "Done!" << std::endl;
  std::cout << "Generate random weights " << generate_random_weights << std::endl;
  return 0;
}
