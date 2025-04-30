#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/tuple.h>
#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>
#include <thrust/inner_product.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/discard_iterator.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <limits>
#include <cstdio>
#include <chrono> 
#include "kernels.cu"


void boruvka_for_edge_array(edge* edges, vertex num_nodes, vertex num_edges,
                            edge* final_msf, 
                            vertex& number_of_filled_edges,
                            std::string result = "res.txt"){

  //Print graph params
  std::cout << "chunk num nodes " << num_nodes << std::endl;
  std::cout << "chunk num edges " << num_edges << std::endl;

  //DeviceMallocs start
  //Malloc edge array on device
  edge *d_edges;
  cudaMalloc(&d_edges, num_edges*sizeof(edge));

  //Malloc a new edges array on device to be used in the iterative relabelling step on device
  edge *d_iter_edges;
  cudaMalloc(&d_iter_edges, num_edges*sizeof(edge));

  //Malloc a representative array of size num_nodes on device
  vertex *d_representatives;
  cudaMalloc(&d_representatives, num_nodes*sizeof(vertex));

  //Malloc array for d_msf
  edge *d_msf;
  cudaMalloc(&d_msf, num_nodes*sizeof(edge));

  //Malloc a new representatives array on device to be used in the iterative relabelling step
  vertex *d_new_representatives;
  cudaMalloc(&d_new_representatives, num_nodes*sizeof(vertex));

  //Malloc a representative array to be used for iterations
  vertex *d_iter_representatives;
  cudaMalloc(&d_iter_representatives, num_nodes*sizeof(vertex));

  //Malloc array for outgoing_edges
  outgoing_edge_id *d_outgoing_edges;
  cudaMalloc(&d_outgoing_edges, num_nodes*sizeof(outgoing_edge_id));

  //Malloc a final MSF array to store the edges in the MSF in device
  edge *d_final_msf;
  cudaMalloc(&d_final_msf, num_nodes*sizeof(edge));

  //DeviceMallocs end

  //timer for transfer
  auto start_transfer = std::chrono::high_resolution_clock::now();
  //HtoD Memcpy start
  //Copy edge array to device
  cudaMemcpy(d_edges, edges, num_edges*sizeof(edge), cudaMemcpyHostToDevice);
  //HtoD Memcpy end
  auto stop_transfer = std::chrono::high_resolution_clock::now();

  cudaMemcpy(d_iter_edges, d_edges, num_edges*sizeof(edge), cudaMemcpyDeviceToDevice);


  //populate the representative array with the node id
  thrust::device_ptr<vertex> d_representatives_ptr(d_representatives);
  thrust::sequence(d_representatives_ptr, d_representatives_ptr+num_nodes);
  std::cout << "done intializing" << std::endl;
  //Make a copy of the representative array to use in iterations
  thrust::device_ptr<vertex> d_new_representatives_ptr(d_new_representatives);
  thrust::copy(d_representatives_ptr, d_representatives_ptr+num_nodes, d_new_representatives_ptr);

  //Make a copy of representative array to use in iterations
  thrust::device_ptr<vertex> d_iter_representatives_ptr(d_iter_representatives);
  thrust::copy(d_representatives_ptr, d_representatives_ptr+num_nodes, d_iter_representatives_ptr);


  thrust::device_ptr<edge> d_final_msf_ptr(d_final_msf);


  vertex *d_non_self_loops;
  cudaMalloc(&d_non_self_loops, sizeof(vertex)*2);
  vertex *h_non_self_loops = (vertex *)malloc(2*sizeof(vertex));
  h_non_self_loops[0] = 1;
  cudaMemcpy(d_non_self_loops, h_non_self_loops, 2*sizeof(vertex), cudaMemcpyHostToDevice);


  cudaError err;

  auto start = std::chrono::high_resolution_clock::now();
  auto duration_transfer = std::chrono::duration_cast<std::chrono::microseconds>(stop_transfer 
                          - start_transfer);


  //device_boruvka_exp(d_edges, d_final_msf, 
  //                d_iter_edges, d_msf,
  //                d_representatives, d_new_representatives,
  //                d_iter_representatives,
  //                d_outgoing_edges, d_non_self_loops,
  //                num_nodes, num_edges, 
  //                number_of_filled_edges,
  //                d_final_msf_ptr);
  device_boruvka(d_edges, d_final_msf, 
                  d_iter_edges, d_msf,
                  d_representatives, d_new_representatives,
                  d_iter_representatives,
                  d_outgoing_edges,
                  num_nodes, num_edges, 
                  number_of_filled_edges,
                  d_new_representatives_ptr,
                  d_representatives_ptr,
                  d_iter_representatives_ptr,
                  d_final_msf_ptr);


  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
  std::cout << "Time to execute " << duration.count() << " microseconds"<< std::endl;
  std::cout << "Time to transfer " << duration_transfer.count() << " microseconds"<< std::endl;
  cudaMemcpy(final_msf, d_final_msf, num_nodes*sizeof(edge), cudaMemcpyDeviceToHost);
  err = cudaDeviceSynchronize();
  fprintf(stderr, "Completed memcpy %d: %s.\n", err, cudaGetErrorString(err));
  assert(err == cudaSuccess);

  large_vertex total_weight=0;
  for(vertex i=0; i<num_nodes; i++){
    edge e = final_msf[i];
    total_weight+=e.w;
    if(i<10){
      std::cout << e.u + 1 << " " << e.v + 1 << " " << e.w << std::endl;
      std::cout << edges[i].u + 1 << " " << edges[i].v + 1 << " " << edges[i].w << std::endl;
    }
  }
  std::cout << "Total weight " << total_weight << std::endl;

  //free all cuda allocs

  cudaFree(0);
  cudaFree(d_edges);
  cudaFree(d_representatives);
  cudaFree(d_msf);
  cudaFree(d_new_representatives);
  cudaFree(d_iter_representatives);
  cudaFree(d_outgoing_edges);
  cudaFree(d_final_msf);
  cudaFree(d_iter_edges);

  return;
}



void malloc_managed_monolithic(std::ifstream &input_stream, vertex num_nodes,
                               vertex num_edges, std::string result,
                               bool generate_random_weights, edge* final_msf, 
                               vertex& number_of_filled_edges){

  edge* d_edges;
  cudaError err;
  // Allocate Unified Memory -- accessible from CPU or GPU
  cudaMallocManaged(&d_edges, num_edges*sizeof(edge));
  err = cudaDeviceSynchronize();
  fprintf(stderr, "CUDA Malloc managed %d: %s.\n", err, cudaGetErrorString(err));
  assert(err == cudaSuccess);

  //populate the managed mem
  populateEdgeArray(input_stream, d_edges, num_edges, num_nodes, generate_random_weights);

  //Malloc a new edges array on device to be used in the iterative relabelling step on device
  edge *d_iter_edges;
  cudaMallocManaged(&d_iter_edges, num_edges*sizeof(edge));

  //Malloc a representative array of size num_nodes on device
  vertex *d_representatives;
  cudaMallocManaged(&d_representatives, num_nodes*sizeof(vertex));

  //Malloc array for d_msf
  edge *d_msf;
  cudaMallocManaged(&d_msf, num_nodes*sizeof(edge));

  //Malloc a new representatives array on device to be used in the iterative relabelling step
  vertex *d_new_representatives;
  cudaMallocManaged(&d_new_representatives, num_nodes*sizeof(vertex));

  //Malloc a representative array to be used for iterations
  vertex *d_iter_representatives;
  cudaMallocManaged(&d_iter_representatives, num_nodes*sizeof(vertex));

  //Malloc array for outgoing_edges
  outgoing_edge_id *d_outgoing_edges;
  cudaMallocManaged(&d_outgoing_edges, num_nodes*sizeof(outgoing_edge_id));

  //Malloc a final MSF array to store the edges in the MSF in device
  edge *d_final_msf;
  cudaMallocManaged(&d_final_msf, num_nodes*sizeof(edge));

  //DeviceMallocs end

  //timer for transfer
  auto start_transfer = std::chrono::high_resolution_clock::now();
  //HtoD Memcpy start
  //Copy edge array to device
  //cudaMemcpy(d_edges, edges, num_edges*sizeof(edge), cudaMemcpyHostToDevice);
  //HtoD Memcpy end
  auto stop_transfer = std::chrono::high_resolution_clock::now();
  auto start = std::chrono::high_resolution_clock::now();

  cudaMemcpy(d_iter_edges, d_edges, num_edges*sizeof(edge), cudaMemcpyDeviceToDevice);


  //populate the representative array with the node id
  thrust::device_ptr<vertex> d_representatives_ptr(d_representatives);
  thrust::sequence(d_representatives_ptr, d_representatives_ptr+num_nodes);
  std::cout << "done intializing" << std::endl;
  //Make a copy of the representative array to use in iterations
  thrust::device_ptr<vertex> d_new_representatives_ptr(d_new_representatives);
  thrust::copy(d_representatives_ptr, d_representatives_ptr+num_nodes, d_new_representatives_ptr);

  //Make a copy of representative array to use in iterations
  thrust::device_ptr<vertex> d_iter_representatives_ptr(d_iter_representatives);
  thrust::copy(d_representatives_ptr, d_representatives_ptr+num_nodes, d_iter_representatives_ptr);

  //vertex num_edges_copy = num_edges;
  vertex num_nodes_copy = num_nodes;

  thrust::device_ptr<edge> d_final_msf_ptr(d_final_msf);

  auto duration_transfer = std::chrono::duration_cast<std::chrono::microseconds>(stop_transfer 
                          - start_transfer);


  device_boruvka(d_edges, d_final_msf, 
                  d_iter_edges, d_msf,
                  d_representatives, d_new_representatives,
                  d_iter_representatives,
                  d_outgoing_edges,
                  num_nodes, num_edges, 
                  number_of_filled_edges,
                  d_new_representatives_ptr,
                  d_representatives_ptr,
                  d_iter_representatives_ptr,
                  d_final_msf_ptr);


  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
  std::cout << "Time to execute " << duration.count() << " microseconds"<< std::endl;
  std::cout << "Time to transfer " << duration_transfer.count() << " microseconds"<< std::endl;
  std::cout << "number of num_nodes_copy "<< num_nodes_copy <<std::endl;

  cudaMemcpy(final_msf, d_final_msf, num_nodes_copy*sizeof(edge), cudaMemcpyDeviceToHost);
  //Synchronize
  err = cudaDeviceSynchronize();
  fprintf(stderr, "Completed memcpy %d: %s.\n", err, cudaGetErrorString(err));
  assert(err == cudaSuccess);


  large_vertex total_weight = 0;
  //only for debug remove if the entire msf is to be printed to file.
  result = "res.txt";
  if(result != "res.txt"){
    std::ofstream result_file(result);
    for(vertex i=0; i<num_nodes_copy-1; i++){
      if(final_msf[i].w >0){
        weight unmasked_weight = final_msf[i].w;
        total_weight += unmasked_weight;
        result_file << final_msf[i].u + 1 << " " << final_msf[i].v + 1
                    << " " << unmasked_weight << std::endl;
      }
    }
  }else{
    for(int i=0; i<num_nodes_copy-1; i++){
      if(final_msf[i].w >0){
        total_weight +=  final_msf[i].w;
      }
    }
  }
  //print total weight
  std::cout << "total weight " << total_weight << std::endl;
  std::cout << "number of filled edges " << number_of_filled_edges << std::endl;
  //free all cuda allocs

  cudaFree(d_edges);
  cudaFree(d_representatives);
  cudaFree(d_msf);
  cudaFree(d_new_representatives);
  cudaFree(d_iter_representatives);
  cudaFree(d_outgoing_edges);
  cudaFree(d_final_msf);
  cudaFree(d_iter_edges);

  return;
}


void cpu_boruvka_monolithic(std::string filename, bool generate_random_weights=false){

  std::ifstream file(filename);
  vertex num_nodes, num_edges, number_of_filled_edges;
  large_vertex num_edges_large;
  getGraphInfo(file, num_nodes, num_edges_large);
  if(num_edges_large < sentinel_vertex){
    num_edges = vertex(num_edges_large);
  }
  edge *edges_arr = (edge *)malloc(num_edges*sizeof(edge));
  readGraph(edges_arr, num_nodes, num_edges, file, generate_random_weights);
  std::vector<edge> edges(edges_arr, edges_arr + num_edges);

  std::cout << "read the graph\n";
  //Malloc three representative arrays
  vertex* representative_array_arr = (vertex *)malloc(num_nodes*sizeof(vertex));
  vertex* representative_array_iter_arr = (vertex *)malloc(num_nodes*sizeof(vertex));
  std::vector<vertex> representative_array(representative_array_arr,
                                  representative_array_arr+num_nodes);
  std::vector<vertex> representative_array_iter(representative_array_iter_arr, 
                                      representative_array_iter_arr+num_nodes);


  //Malloc a copy of edges for iterations
  edge* iter_edges_arr = (edge *)malloc(num_edges*sizeof(edge));
  std::vector<edge> iter_edges(iter_edges_arr, iter_edges_arr+num_edges);
  edge* iter_msf_arr = (edge *)malloc(num_nodes*sizeof(edge));
  std::vector<edge> iter_msf(iter_msf_arr, iter_msf_arr+num_nodes);



  edge *final_msf_arr = (edge *)malloc(num_nodes*sizeof(edge));
  std::vector<edge> final_msf(final_msf_arr, final_msf_arr+num_nodes);


  //Populate representative arrays
  #pragma omp parallel for
  for(vertex i=0; i<num_nodes; i++){
    representative_array[i] = i;
    representative_array_iter[i] = i;
  }


  //Malloc outgoing_edge_id array
  outgoing_edge_id* outgoing_edges_arr = (outgoing_edge_id *)malloc(num_nodes*
                                          sizeof(outgoing_edge_id));
  std::vector<outgoing_edge_id> outgoing_edges(outgoing_edges_arr, outgoing_edges_arr+num_nodes);

  std::cout << "in the method \n";

  //Deep copy edges to iter_edges
  std::cout << "\n";
  #pragma omp parallel for
  for(vertex i=0; i<num_edges; i++){
    iter_edges[i].u = edges[i].u;
    iter_edges[i].v = edges[i].v;
    iter_edges[i].w = edges[i].w;
  }




  //for(int i=0; i<num_edges; i++){
  //  std::cout << edges[i].u << " " << edges[i].v << " " << edges[i].w << std::endl;
  //}
  boruvka_for_edge_vector_cpu(edges, num_nodes, num_edges, 
                          final_msf, number_of_filled_edges, representative_array,
                          representative_array_iter, iter_edges, iter_msf, outgoing_edges);
  
  large_vertex w = 0;
  for(vertex i=0; i<num_nodes-1;i++){
    //std::cout << final_msf[i].u +1<< " " << final_msf[i].v+1 << " "
    //  <<final_msf[i].w << "\n";
    w += final_msf[i].w;
  }
  std::cout << "total weight " << w << "\n";

  //Free mallocs
  free(iter_msf_arr);
  free(outgoing_edges_arr);
  free(representative_array_arr);
  free(representative_array_iter_arr);
  free(iter_edges_arr);
}

float convertToFloat(int num) {
    return static_cast<float>(num) + 0.4f;
}

void new_boruvka(std::string filename, std::string result_filename, 
                bool debug, vertex chunk_size, int num_chunks_ideal,
                bool generate_random_weights=false,
                bool use_malloc_managed=false, bool use_streamline=false,
                bool use_cpu_only=false, bool use_cpu_gpu_streamline=false,
                bool save_weighted_graph=false){
    std::cout << "new boruvka " << filename << std::endl;

    vertex num_edges = 0;
    vertex num_nodes = 0;
    large_vertex num_edges_large = 0;



    std::ifstream file(filename);
    cudaFree(0);
    getGraphInfo(file, num_nodes, num_edges_large);
    if(num_edges_large < sentinel_vertex){
      num_edges = vertex(num_edges_large);
    }
    std::cout << "Total num edges " << num_edges_large << std::endl;
    std::cout << "Total num nodes " << num_nodes << std::endl;
    if(num_chunks_ideal > 0){
      float float_num_chunks = convertToFloat(num_chunks_ideal);
      //chunk_size = num_edges / float_num_chunks;
      chunk_size = vertex(num_edges_large / float_num_chunks);
    }

    if(debug){
      int debug_iters = 10;
      std::cout << "chunk \n";
      if(use_cpu_gpu_streamline){
        std::cout << "streamline \n";
        bool is_weight_float = false;
        if (filename.find("MOLIERE_2016") != std::string::npos) 
          is_weight_float = true;

        streamline_boruvka_heterogeneous(file, num_nodes, num_edges_large, chunk_size, 
                          result_filename, generate_random_weights, is_weight_float, debug_iters);
        std::cout << "debug " << debug << std::endl;
        return;
      }
      if(use_streamline){
        std::cout << "streamline \n";
        bool is_weight_float = false;
        if (filename.find("MOLIERE_2016") != std::string::npos) 
          is_weight_float = true;



        streamline_boruvka(file, num_nodes, num_edges_large, chunk_size, 
                          result_filename, generate_random_weights, is_weight_float, debug_iters);
        std::cout << "debug " << debug << std::endl;
        return;
      }

      edge *msf = (edge *)malloc(num_nodes*sizeof(edge));
      //cudaMallocHost((void**)&msf, (num_nodes-1)* sizeof(edge));
      vertex number_of_filled_edges = 0;

      if(use_malloc_managed){
        malloc_managed_monolithic(file, num_nodes, num_edges, 
                                  result_filename, generate_random_weights,
                                  msf, number_of_filled_edges);
        return;
      }

      edge *edges;
      cudaMallocHost((void**)&edges, num_edges * sizeof(edge));
      ompPopulateEdgeArray(file, edges, num_edges, generate_random_weights);
      if(save_weighted_graph){
        std::ofstream outgraph_weighted("weighted.mtx");
        outgraph_weighted << header_for_weighted_mtx;
        outgraph_weighted << num_nodes << " " << num_nodes << " " << num_edges * 2 << "\n";
        for(vertex i=0; i<num_edges; i++){
          edge e = edges[i];
          outgraph_weighted << e.u + 1<< " " << e.v + 1 << " " << e.w << "\n";
          outgraph_weighted << e.v + 1<< " " << e.u + 1 << " " << e.w << "\n";
        }
        outgraph_weighted.close();
        return;
        //for(vertex i=0; i<num_edges; i++){

        //}
      }

      std::cout << "monolithic GPU approach\n";
      file.close();
      boruvka_for_edge_array(edges, num_nodes, num_edges, msf, number_of_filled_edges, result_filename);
      weight total_weight=0;
      for(vertex i=0; i<num_nodes; i++){
        edge e = msf[i];
        total_weight+=e.w;
        //std::cout << e.u + 1 << " " << e.v + 1 << " " << e.w << std::endl;
      }
      std::cout << "Total weight " << total_weight << std::endl;


      //Free all allocs
      cudaFree(edges);
      cudaFree(msf);
      //test_tbb();

      return;
    }

    cudaFree(0);

    if(use_cpu_gpu_streamline){
      std::cout << "streamline \n";
      bool is_weight_float = false;
      if (filename.find("MOLIERE_2016") != std::string::npos) 
        is_weight_float = true;

      streamline_boruvka_heterogeneous(file, num_nodes, num_edges_large, chunk_size, 
                        result_filename, generate_random_weights, is_weight_float);
      return;
    }
    if(use_streamline){
      std::cout << "streamline \n";
      bool is_weight_float = false;
      if (filename.find("MOLIERE_2016") != std::string::npos) 
        is_weight_float = true;

      streamline_boruvka(file, num_nodes, num_edges_large, chunk_size, 
                        result_filename, generate_random_weights, is_weight_float);
      return;
    }

    edge *msf = (edge *)malloc(num_nodes*sizeof(edge));
    //cudaMallocHost((void**)&msf, (num_nodes-1)* sizeof(edge));
    vertex number_of_filled_edges = 0;

    if(use_malloc_managed){
      malloc_managed_monolithic(file, num_nodes, num_edges, 
                                result_filename, generate_random_weights,
                                msf, number_of_filled_edges);
      return;
    }

    edge *edges;
    cudaMallocHost((void**)&edges, num_edges * sizeof(edge));
    ompPopulateEdgeArray(file, edges, num_edges, num_nodes, generate_random_weights);

    if(use_cpu_only){
      //cpu_boruvka_monolithic(filename, generate_random_weights);
      edge_pbbs *edges_for_pbbs = (edge_pbbs *)malloc(num_edges_large*sizeof(edge));
      for(large_vertex i=0;i<num_edges_large; i++){
        edges_for_pbbs[i].u = edges[i].u;
        edges_for_pbbs[i].v = edges[i].v;
        edges_for_pbbs[i].w = edges[i].w;
      }
      wghEdgeArray<vertexId, vertexId> edge_array = create_wghEdgeArray(edges_for_pbbs, num_edges_large);

      size_t num_edges_mst;
      auto start_pbbs= std::chrono::high_resolution_clock::now();
      edge* mst_array = cpu_compute_pbbs(num_edges_mst, edge_array);

      auto stop_pbbs = std::chrono::high_resolution_clock::now();
      auto duration_pbbs = std::chrono::duration_cast<std::chrono::microseconds>(
                                 stop_pbbs - start_pbbs);

      large_vertex total_weight = 0;
      for(large_vertex j=0; j<num_edges_mst; j++){
	total_weight += mst_array[j].w;
      }
      std::cout<< "pbbs execution time " << duration_pbbs.count() << " microseconds\n";
      std::cout << "Total weight " << total_weight << std::endl;
      return;
    }

    std::cout << "monolithic1 GPU approach\n";
    file.close();

    boruvka_for_edge_array(edges, num_nodes, num_edges, msf, number_of_filled_edges, result_filename);
    large_vertex total_weight=0;
    for(vertex i=0; i<num_nodes; i++){
      edge e = msf[i];
      total_weight+=e.w;
    }
    std::cout << "Total weight " << total_weight << std::endl;


    //Free all allocs
    cudaFree(edges);
    cudaFree(msf);

    return;
}
