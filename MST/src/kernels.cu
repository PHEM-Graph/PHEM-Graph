#include "utils.cu"
#include <condition_variable>
#include <mutex>
#include <thread>

std::mutex mtx;
std::condition_variable cv;
// Shared variable
vertex x = 0;

__device__ void find_min_weighted_edge_id(outgoing_edge_id *existing_id, weight new_weight, vertex tid){
    //AtomicMin to replace the existing edge weight with the new edge weight if the new edge weight is smaller
  atomicMin(&existing_id->w, new_weight);
  __threadfence();
  //Make sure that the existing weight matches the new weight
  if(existing_id->w == new_weight){
      atomicExch(&existing_id->e_id, tid);
  }

  return;
}

__device__ void atomic_min_weight(outgoing_edge_id *existing_id, weight new_weight){
  atomicMin(&existing_id->w, new_weight);
  return;
}

__device__ void atomic_min_eid(outgoing_edge_id *existing_id, vertex new_eid){
  atomicMin(&existing_id->e_id, new_eid);
  //existing_id->e_id = new_eid;
  return;
}


__global__ void populate_min_weight(edge* d_edges,
                                    outgoing_edge_id* d_best_index,
                                    vertex num_edges, vertex num_nodes){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_e = d_edges[tid];
    vertex tid_u = tid_e.u;
    if(tid_u == sentinel_vertex){
      return;
    }
    vertex tid_v = tid_e.v;
    weight tid_w = tid_e.w;
    atomic_min_weight(&d_best_index[tid_u], tid_w);
    atomic_min_weight(&d_best_index[tid_v], tid_w);

  }
}

__global__ void populate_eid(edge* d_edges,
                             outgoing_edge_id* d_best_index,
                             vertex num_edges, vertex num_nodes){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_e = d_edges[tid];
    vertex tid_u = tid_e.u;
    if(tid_u == sentinel_vertex){
      return;
    }
    vertex tid_v = tid_e.v;
    weight tid_w = tid_e.w;
    weight u_best_w = d_best_index[tid_u].w;
    weight v_best_w = d_best_index[tid_v].w;
    if(tid_w == u_best_w){
      atomic_min_eid(&d_best_index[tid_u], tid);
    }
    if(tid_w == v_best_w){
      atomic_min_eid(&d_best_index[tid_v], tid);
    }
    return;


  }
}


__global__ void new_grafting_edges(outgoing_edge_id* outgoing_edges,
                        edge* iter_edges, edge* edges,
                        edge* msf, vertex* representative,
                        vertex num_nodes, vertex num_edges,
                        weight sentinel, vertex sentinel_vertex){
  vertex i = blockIdx.x*blockDim.x + threadIdx.x;
  if(i<num_nodes){
    outgoing_edge_id tid_outgoing_edge_id = outgoing_edges[i];
    vertex tid_outgoing_e_id = tid_outgoing_edge_id.e_id;
    //weight tid_best_w = tid_outgoing_edge_id.w;
    if(tid_outgoing_e_id < (num_edges)){
      edge tid_best = iter_edges[tid_outgoing_e_id];
      if(tid_best.u != sentinel_vertex){
        edge corresponding_edge_in_edges = edges[tid_outgoing_e_id];
        vertex tid_best_u, tid_best_v;
        vertex corresponding_u, corresponding_v;
        weight corresponding_w;
        corresponding_u = corresponding_edge_in_edges.u;
        if(corresponding_u != sentinel_vertex){
          corresponding_v = corresponding_edge_in_edges.v;
          corresponding_w = corresponding_edge_in_edges.w;
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
          if(tid_best.u == best_outgoing_edge_of_v_edge.u 
          && tid_best.v == best_outgoing_edge_of_v_edge.v 
          && tid_best.w == best_outgoing_edge_of_v_edge.w)
            same = true;
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
  }
  return;
}

__global__ void new_grafting_edges_exp(outgoing_edge_id* d_outgoing_edges,
                                   edge* d_iter_edges,
                                   edge* d_edges,
                                   edge* d_msf,
                                   vertex* d_representative,
                                   vertex num_nodes,
                                   vertex num_edges,
                                   weight sentinel, vertex sentinel_vertex){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    outgoing_edge_id tid_outgoing_edge_id = d_outgoing_edges[tid];
    vertex tid_outgoing_e_id = tid_outgoing_edge_id.e_id;
    //weight tid_best_w = tid_outgoing_edge_id.w;
    if(tid_outgoing_e_id < (num_edges)){
      edge tid_best= d_iter_edges[tid_outgoing_e_id];
      if(tid_best.u == sentinel_vertex){
        return;
      }
      edge corresponding_edge_in_d_edges = d_edges[tid_outgoing_e_id];
      vertex tid_best_u, tid_best_v;
      vertex corresponding_u, corresponding_v;
      weight corresponding_w;
      corresponding_u = corresponding_edge_in_d_edges.u;
      corresponding_v = corresponding_edge_in_d_edges.v;
      corresponding_w = corresponding_edge_in_d_edges.w;
      if(corresponding_u == sentinel_vertex){
        return;
      }
      if(tid_best.u == tid){
        tid_best_u = tid_best.u;
        tid_best_v = tid_best.v;
      }else{
        tid_best_u = tid_best.v;
        tid_best_v = tid_best.u;
      }
      // This ensures that tid_best_u == tid

      //find the best outgoing edge for tid_best_v
      outgoing_edge_id best_outgoing_edge_of_v = d_outgoing_edges[tid_best_v];
      vertex best_outgoing_edge_of_v_e_id = best_outgoing_edge_of_v.e_id;
      edge best_outgoing_edge_of_v_edge = d_iter_edges[best_outgoing_edge_of_v_e_id];
      //check if tid_best and best_outgoing_edge_of_v are the same
      bool same = false;

      check_if_edges_are_same(tid_best, best_outgoing_edge_of_v_edge, &same);
      if(same && (tid_best_u > tid_best_v)){
      //else if((tid_best_u > tid_best_v)){
        d_representative[tid] = tid_best_v;
        d_msf[tid].u = corresponding_u;
        d_msf[tid].v = corresponding_v;
        d_msf[tid].w = corresponding_w;
      }
      else if(!same){
        d_representative[tid] = tid_best_v;
        d_msf[tid].u = corresponding_u;
        d_msf[tid].v = corresponding_v;
        d_msf[tid].w = corresponding_w;
      }
    }
  }
  return;
}


__global__ void grafting_edges(edge* d_best, vertex* d_representative, vertex num_nodes, 
                                weight sentinel, edge* d_msf){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    if(d_best[tid].w != sentinel && d_best[tid].v != num_nodes){
      vertex u = d_best[tid].u;
      vertex v = d_best[tid].v;
      vertex j = 0;
      if(u == tid){
          j = v;
      }else{
          j = u;
      }
      bool same = false;
      check_if_edges_are_same(d_best[tid], d_best[j], &same);
      if( same && tid<j){
          //do nothing with representative array
          d_msf[tid].u = tid;
          d_msf[tid].v = j;
          d_msf[tid].w = sentinel; 

      }else {
          d_representative[tid] = j;
          d_msf[tid].u = d_best[tid].u;
          d_msf[tid].v = d_best[tid].v;
          d_msf[tid].w = d_best[tid].w;
          ////Add the most significant bit to d_msf[tid].w to indicate that it is in the msf
          //d_msf[tid].w = d_msf[tid].w | (1 << 31);
      }
    } else {
      d_msf[tid].u = tid;
      d_msf[tid].v = num_nodes;
      d_msf[tid].w = sentinel;
    }
  }
}

__global__ void shortcutting_step(vertex* d_representative, vertex num_nodes,
                                  weight sentinel){
    /*Input - d_representative representative array, num_nodes number of nodes
      Output - Compacted parent array: a node is either a root or child of root*/
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    vertex r = tid;
    while(true){
      __threadfence();
      if(d_representative[r] == d_representative[d_representative[r]]){
        break;
      }
      d_representative[r] = d_representative[d_representative[r]];
    }
  }
}

__global__ void new_shortcutting_step(vertex* d_representative,
                vertex num_nodes, weight sentinel, vertex* d_iter_representatives){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    //Find the root of each tid. Break when root is found
    vertex r = tid;
    vertex root = d_representative[r];
    while(true){
      if(root == d_representative[root]){
        break;
      }
      root = d_representative[root];
    }
    //Set the representative of each tid to the root
    d_iter_representatives[tid] = root;
  }
  return;
}

__global__ void relabelling_step(edge* d_edges, vertex* d_representative,
                                 vertex num_edges, vertex num_nodes,
                                 weight sentinel, vertex sentinel_vertex){
    vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid<num_edges){
      vertex u = d_edges[tid].u;
      if(u == sentinel_vertex){
        return;
      }
      vertex v = d_edges[tid].v;
      if(u < num_nodes && v < num_nodes){
        vertex new_u = d_representative[u];
        vertex new_v = d_representative[v];
        if(d_edges[tid].u != sentinel_vertex){
          d_edges[tid].u = new_u;
          d_edges[tid].v = new_v;
        }
      }
      else{
        d_edges[tid].u = sentinel_vertex;
        d_edges[tid].v = sentinel_vertex;
        d_edges[tid].w = sentinel;
      }
    }
}

__global__ void filtering_step(edge* d_new_edges, vertex* d_representative,
                               vertex num_edges, vertex num_nodes,
                               weight sentinel, vertex sentinel_vertex){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    vertex u = d_new_edges[tid].u;
    if(u == sentinel_vertex){
      return;
    }
    vertex v = d_new_edges[tid].v;
    if(u<num_nodes && v<num_nodes){
      if(d_representative[u] != d_representative[v]){
        d_new_edges[tid].u = u;
        d_new_edges[tid].v = v;
        d_new_edges[tid].w = d_new_edges[tid].w;
      }else{
        d_new_edges[tid].u = sentinel_vertex;
        d_new_edges[tid].v = sentinel_vertex;
        d_new_edges[tid].w = sentinel;
      }
    }
  }
}


__global__ void reset_edge_array_to_sentinel(edge* d_edges, vertex num_edges,
                                             weight sentinel){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    d_edges[tid].u = sentinel_vertex;
    d_edges[tid].v = sentinel_vertex;
    d_edges[tid].w = sentinel;
  }
}

__global__ void reset_outgoing_edge_array_to_sentinel(outgoing_edge_id* d_outgoing_edges, 
                                vertex num_nodes, vertex num_edges, weight sentinel_wt){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    d_outgoing_edges[tid].e_id = num_edges+1;
    d_outgoing_edges[tid].w = sentinel_wt;
  }
}

__global__ void reset_outgoing_edge_array_to_sentinel_max(outgoing_edge_id* d_outgoing_edges, 
                                vertex num_nodes, vertex num_edges, weight sentinel_wt){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    d_outgoing_edges[tid].e_id = 0;
    d_outgoing_edges[tid].w = 0;
  }
}



__global__ void set_self_edge_to_sentinel(edge* d_edges, vertex num_edges,
                                           weight sentinel, vertex sentinel_vertex){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    if(d_edges[tid].u == d_edges[tid].v && d_edges[tid].u != sentinel_vertex){
      d_edges[tid].u = sentinel_vertex;
      d_edges[tid].v = sentinel_vertex;
      d_edges[tid].w = sentinel;
    }
  }
}

__global__ void add_one_to_edge_weight(edge* d_edges, vertex num_edges){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    d_edges[tid].w += 1;
  }
}


__global__ void populate_representative_array(vertex* d_representative, vertex num_nodes){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    d_representative[tid] = tid;
  }
}

__global__ void copy_representative_array(vertex* d_representative, 
                      vertex* d_iter_representative, vertex num_nodes){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_nodes){
    d_representative[tid] = d_iter_representative[tid];
  }
}

__global__ void copy_edges_array(edge* d_edges_src, 
                      edge* d_edges_dst, vertex num_edges){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    d_edges_dst[tid].u = d_edges_dst[tid].u;
    d_edges_dst[tid].v = d_edges_dst[tid].v;
    d_edges_dst[tid].w = d_edges_dst[tid].w;
  }
}

__global__ void mark_sentinel_weighted_edges(edge* d_edges, vertex num_edges){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_edge=d_edges[tid];
    if(tid_edge.w == sentinel){
      d_edges[tid].u = sentinel_vertex;
      d_edges[tid].v = sentinel_vertex;
    }
  }
}

__global__ void mark_self_loops(edge* d_iter_edges, vertex num_edges){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_edge=d_iter_edges[tid];
    if(tid_edge.u == tid_edge.v){
      d_iter_edges[tid].u = sentinel_vertex;
      d_iter_edges[tid].v = sentinel_vertex;
      d_iter_edges[tid].w = sentinel;
    }
  }
}

//Idea - use global memory flag.
__global__ void check_if_any_loop(edge* d_iter_edges, vertex num_edges, vertex *non_self_loop){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_edge=d_iter_edges[tid];
    if(tid_edge.u != tid_edge.v){
      if(non_self_loop[0] == 0){
        non_self_loop[0] = 1;
      }
    }
  }
}

// ADDING LOGIC FOR MAX OUTGOING 

__device__ void find_max_weighted_edge_id(outgoing_edge_id *existing_id, weight new_weight, vertex tid){
  //AtomicMax to replace the existing edge weight with the new 
  //edge weight if the new edge weight is bigger
  atomicMax(&existing_id->w, new_weight);
  __threadfence();
  //Make sure that the existing weight matches the new weight
  if(existing_id->w == new_weight){
      atomicExch(&existing_id->e_id, tid);
  }

  return;
}

__device__ void atomic_max_weight(outgoing_edge_id *existing_id, weight new_weight){
  atomicMax(&existing_id->w, new_weight);
  return;
}

__device__ void atomic_max_eid(outgoing_edge_id *existing_id, vertex new_eid){
  atomicMax(&existing_id->e_id, new_eid);
  return;
}


__global__ void populate_max_weight(edge* d_edges,
                                    outgoing_edge_id* d_best_index,
                                    vertex num_edges, vertex num_nodes){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_e = d_edges[tid];
    vertex tid_u = tid_e.u;
    if(tid_u == sentinel_vertex){
      return;
    }
    vertex tid_v = tid_e.v;
    weight tid_w = tid_e.w;
    atomic_max_weight(&d_best_index[tid_u], tid_w);
    atomic_max_weight(&d_best_index[tid_v], tid_w);
  }
}

__global__ void populate_eid_max(edge* d_edges,
                             outgoing_edge_id* d_best_index,
                             vertex num_edges, vertex num_nodes){
  vertex tid = blockIdx.x*blockDim.x + threadIdx.x;
  if(tid<num_edges){
    edge tid_e = d_edges[tid];
    vertex tid_u = tid_e.u;
    if(tid_u == sentinel_vertex){
      return;
    }
    vertex tid_v = tid_e.v;
    weight tid_w = tid_e.w;
    weight u_best_w = d_best_index[tid_u].w;
    weight v_best_w = d_best_index[tid_v].w;
    if(tid_w == u_best_w){
      atomic_max_eid(&d_best_index[tid_u], tid);
    }
    if(tid_w == v_best_w){
      atomic_max_eid(&d_best_index[tid_v], tid);
    }
    return;


  }
}

//Boruvka methods


void device_boruvka(edge *d_edges, edge* d_final_msf, 
                    edge *d_iter_edges, edge *d_msf,
                    vertex *d_representatives, vertex *d_new_representatives,
                    vertex *d_iter_representatives,
                    outgoing_edge_id *d_outgoing_edges,
                    vertex &num_nodes, vertex &num_edges, 
                    vertex &number_of_filled_edges,
                    thrust::device_ptr<vertex> d_new_representatives_ptr,
                    thrust::device_ptr<vertex> d_representatives_ptr,
                    thrust::device_ptr<vertex> d_iter_representatives_ptr,
                    thrust::device_ptr<edge> d_final_msf_ptr,
                    cudaStream_t kernel_stream=0){

    int iter = 0;

    vertex prev_num_edges = num_edges;
    thrust::device_ptr<edge> d_msf_ptr(d_msf);
    thrust::device_ptr<edge> d_edges_ptr(d_edges);
    thrust::device_ptr<edge> d_iter_edges_ptr(d_iter_edges);
    while(true){
        //find the minimum edge for each node
        std::cout << "iter " << iter << std::endl;
        int num_blocks = (num_edges + BLOCKSIZE - 1)/BLOCKSIZE;
        if(num_nodes > num_edges){
            num_blocks = (num_nodes + BLOCKSIZE - 1)/BLOCKSIZE;
        }

        //Reset the d_msf array
        if(iter < 3){
          reset_edge_array_to_sentinel<<<num_blocks, BLOCKSIZE, 
                  0, kernel_stream>>>(d_msf, num_nodes, sentinel);
        }
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (reset msf) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);

        //reset d_outgoing edges to sentinel
        reset_outgoing_edge_array_to_sentinel<<<num_blocks, BLOCKSIZE, 
            0, kernel_stream>>>(d_outgoing_edges, num_nodes, num_edges, sentinel);
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (reset outgoing edges) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);

        //populate minimum edge weight for d_outgoing_edges
        populate_min_weight<<<num_blocks, BLOCKSIZE,
          0, kernel_stream>>>(d_iter_edges, d_outgoing_edges, num_edges, num_nodes);
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (populate min weight) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);

        //populate edge id for d_outgoing_edges
        populate_eid<<<num_blocks, BLOCKSIZE,
          0, kernel_stream>>>(d_iter_edges, d_outgoing_edges, num_edges, num_nodes);
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (populate edge id) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);


        //New grafting step
        new_grafting_edges<<<num_blocks, BLOCKSIZE,
          0, kernel_stream>>>(d_outgoing_edges, 
                            d_iter_edges, d_edges, d_msf,
                            d_representatives, num_nodes, num_edges, 
                            sentinel, sentinel_vertex);
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (new grafting step) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);


        //find the number of edges in d_msf where edge weight is not sentinel
        vertex non_sentinel_edges = thrust::count_if(d_msf_ptr, d_msf_ptr+num_nodes,
                                    [] __device__ (edge e){return e.w != sentinel;});
        std::cout << "number of non sentinel edges " << non_sentinel_edges << std::endl;

        if(non_sentinel_edges == 0){
            std::cout << "number of iterations "<< iter << std::endl;
            break;
        }
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (grafting step) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);

        //Remove edges in d_msf where endge weight is sentinel
        thrust::device_ptr<edge> d_msf_ptr_end = thrust::remove_if(d_msf_ptr, d_msf_ptr+num_nodes, 
                                                  [] __device__ (edge e){return e.w == sentinel;});
        //std::cout << "removed sentinel edges \n";
        //update d_final_msf with the edges in d_msf
        if(number_of_filled_edges + non_sentinel_edges <= num_nodes-1){
            thrust::copy(d_msf_ptr, d_msf_ptr_end, d_final_msf_ptr+number_of_filled_edges);
        }
        else{
            std::cout << "number of iters- filled msf array" << iter << std::endl;
            //break;
        }
        //std::cout << "copied edges to d_final_msf \n";

        //update number of filled edges
        number_of_filled_edges += non_sentinel_edges;
        //synchronize
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (grafting step) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);

        //Shortcutting step
        new_shortcutting_step<<<num_blocks, BLOCKSIZE, 0, kernel_stream>>>(d_representatives, num_nodes,
                             sentinel, d_iter_representatives);
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (short cutting step) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);
        //std::cout << "done shortcutting step \n";

        //set d_representatives to d_iter_representatives
        thrust::copy(d_iter_representatives_ptr, d_iter_representatives_ptr+num_nodes, d_representatives_ptr);

        //Relabelling step
        relabelling_step<<<num_blocks, BLOCKSIZE,
              0, kernel_stream>>>(d_iter_edges, d_representatives, num_edges, 
              num_nodes, sentinel, sentinel_vertex);
        //std::cout << "done relabelling step \n";
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (relabelling step) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);


        //filtering step for d_edges
        filtering_step<<<num_blocks, BLOCKSIZE,
                        0, kernel_stream>>>(d_edges, d_representatives,
                        num_edges, num_nodes, sentinel, sentinel_vertex);
        //std::cout << "done filtering step \n";
        //err = cudaDeviceSynchronize();
        //fprintf(stderr, "CUDA error (filtering step) %d: %s.\n", err, cudaGetErrorString(err));
        //assert(err == cudaSuccess);

        //Remove edges in d_edges where edge weight is sentinel
        thrust::device_ptr<edge> d_edges_end = thrust::remove_if(d_edges_ptr, d_edges_ptr+num_edges, [] __device__ (edge e){return e.w == sentinel;});
        //std::cout << "removed edges \n";

        //Remove self loops in d_iter_edges
        thrust::device_ptr<edge> d_iter_edges_end = thrust::remove_if(d_iter_edges_ptr, d_iter_edges_ptr+num_edges, 
                        [] __device__ (edge e){return e.u == e.v;});
        //std::cout << "removed self loops \n";
        //update num_edges
        num_edges = d_iter_edges_end - d_iter_edges_ptr;


        //find indices where d_representatives[i] == i using thrust



        std::cout << "num edges " << num_edges << std::endl;
        if(num_edges == 0){
            std::cout << "number of iterations "<< iter << std::endl;
            break;
        }

        if(num_edges == prev_num_edges){
            std::cout << "Not converging\n";
            std::cout << "---------------------------------------------------------------\n";
            std::cout << "Iter "<< iter << std::endl;
            break;
        }

        if(num_edges == prev_num_edges){
            std::cout << "number of iterations - break by no change in edges "<< 
                        iter << std::endl;
            break;
        }

        if(iter >=2){
          thrust::fill(thrust::device, d_msf_ptr, d_msf_ptr+num_nodes, sentinel_edge);
        }
        iter++;
        if(iter == 100){
            std::cout << "failed to converge\n";
            break;
        }
        if(iter == 0){
          break;
        }

    }
  return;
}



void device_boruvka_exp(edge *d_edges, edge* d_final_msf, 
                    edge *d_iter_edges, edge *d_msf,
                    vertex *d_representatives, vertex *d_new_representatives,
                    vertex *d_iter_representatives,
                    outgoing_edge_id *d_outgoing_edges,
                    vertex* d_non_self_loops,
                    vertex &num_nodes, vertex &num_edges, 
                    vertex &number_of_filled_edges,
                    thrust::device_ptr<vertex> d_new_representatives_ptr,
                    thrust::device_ptr<vertex> d_representatives_ptr,
                    thrust::device_ptr<vertex> d_iter_representatives_ptr,
                    thrust::device_ptr<edge> d_final_msf_ptr,
                    cudaStream_t kernel_stream=0){

  int iter = 0;

  vertex *h_non_self_loops = (vertex *)malloc(2*sizeof(vertex));
  thrust::device_ptr<edge> d_msf_ptr(d_msf);
  h_non_self_loops[0] = 1;
  int skipIters = 2;
  while(iter != skipIters+1){
    //find the minimum edge for each node
    std::cout << "iter " << iter << std::endl;
    int num_blocks = (num_edges + BLOCKSIZE - 1)/BLOCKSIZE;
    if(num_nodes > num_edges){
        num_blocks = (num_nodes + BLOCKSIZE - 1)/BLOCKSIZE;
    }

    //Reset the d_msf array
    if(iter == 0){
      reset_edge_array_to_sentinel<<<num_blocks, BLOCKSIZE, 
              0, kernel_stream>>>(d_msf, num_nodes, sentinel);
    }
    //err = cudaDeviceSynchronize();
    //fprintf(stderr, "CUDA error (reset msf) %d: %s.\n", err, cudaGetErrorString(err));
    //assert(err == cudaSuccess);

    //reset d_outgoing edges to sentinel
    reset_outgoing_edge_array_to_sentinel<<<num_blocks, BLOCKSIZE, 
        0, kernel_stream>>>(d_outgoing_edges, num_nodes, num_edges, sentinel);
    //err = cudaDeviceSynchronize();
    //fprintf(stderr, "CUDA error (reset outgoing edges) %d: %s.\n", err, cudaGetErrorString(err));
    //assert(err == cudaSuccess);

    //populate minimum edge weight for d_outgoing_edges
    populate_min_weight<<<num_blocks, BLOCKSIZE,
      0, kernel_stream>>>(d_iter_edges, d_outgoing_edges, num_edges, num_nodes);
    //err = cudaDeviceSynchronize();
    //fprintf(stderr, "CUDA error (populate min weight) %d: %s.\n", err, cudaGetErrorString(err));
    //assert(err == cudaSuccess);

    //populate edge id for d_outgoing_edges
    populate_eid<<<num_blocks, BLOCKSIZE,
      0, kernel_stream>>>(d_iter_edges, d_outgoing_edges, num_edges, num_nodes);
    //err = cudaDeviceSynchronize();
    //fprintf(stderr, "CUDA error (populate edge id) %d: %s.\n", err, cudaGetErrorString(err));
    //assert(err == cudaSuccess);


    //New grafting step
    new_grafting_edges_exp<<<num_blocks, BLOCKSIZE,
      0, kernel_stream>>>(d_outgoing_edges, 
                        d_iter_edges, d_edges, d_msf,
                        d_representatives, num_nodes, num_edges, 
                        sentinel, sentinel_vertex);
    //err = cudaDeviceSynchronize();
    //fprintf(stderr, "CUDA error (new grafting step) %d: %s.\n", err, cudaGetErrorString(err));
    //assert(err == cudaSuccess);




    //Shortcutting step
    new_shortcutting_step<<<num_blocks, BLOCKSIZE, 0, kernel_stream>>>(d_representatives, num_nodes,
                         sentinel, d_iter_representatives);
    //err = cudaDeviceSynchronize();
    //fprintf(stderr, "CUDA error (short cutting step) %d: %s.\n", err, cudaGetErrorString(err));
    //assert(err == cudaSuccess);
    //std::cout << "done shortcutting step \n";
    copy_representative_array<<<num_blocks, BLOCKSIZE, 0, kernel_stream>>>(
                      d_representatives, d_iter_representatives, num_nodes);

    //Relabelling step
    relabelling_step<<<num_blocks, BLOCKSIZE,
          0, kernel_stream>>>(d_iter_edges, d_representatives, num_edges, 
          num_nodes, sentinel, sentinel_vertex);

    //filtering step for d_edges
    filtering_step<<<num_blocks, BLOCKSIZE,
                    0, kernel_stream>>>(d_edges, d_representatives,
                    num_edges, num_nodes, sentinel, sentinel_vertex);

    //std::cout << "removed edges \n";
    mark_sentinel_weighted_edges<<<num_blocks, BLOCKSIZE,
                0, kernel_stream>>>(d_edges, num_edges);
    std::cout << "done marking sentinel edges step \n";

    if(iter == skipIters){
      cudaMemcpy(d_non_self_loops, h_non_self_loops, 2*sizeof(vertex), cudaMemcpyHostToDevice);
      check_if_any_loop<<<num_blocks, BLOCKSIZE,
                  0, kernel_stream>>>(d_iter_edges, num_edges, d_non_self_loops);
      cudaMemcpy(h_non_self_loops, d_non_self_loops, 2*sizeof(vertex), cudaMemcpyDeviceToHost);
      std::cout << "NON SELF LOOPS " << h_non_self_loops[0] << "\n";

    }
    mark_self_loops<<<num_blocks, BLOCKSIZE,
                0, kernel_stream>>>(d_iter_edges, num_edges);
    std::cout << "done removing self edges step \n";

    std::cout << "num edges " << num_edges << std::endl;

    if(h_non_self_loops[0] == 0 || iter == skipIters){

      thrust::device_ptr<edge> d_msf_ptr(d_msf);
      vertex non_sentinel_edges = thrust::count_if(d_msf_ptr, d_msf_ptr+num_nodes,
                                  [] __device__ (edge e){return e.w != sentinel;});
      number_of_filled_edges = non_sentinel_edges;
      std::cout << "number of non sentinel edges " << non_sentinel_edges << std::endl;
      std::cout << "no more non self loops \n";
      thrust::copy_if(d_msf_ptr, d_msf_ptr+num_nodes, d_final_msf_ptr,
                        [] __device__ (edge e){return e.w != sentinel;});
      break;
    }



    if(iter > skipIters)
      h_non_self_loops[0] = 0;
    iter++;
    if(iter == 100){
        std::cout << "failed to converge\n";
        break;
    }
    if(iter == 0){
      break;
    }

  }

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
                 d_final_msf_ptr, kernel_stream);
  

  return;
}


void streamline_boruvka(std::ifstream &input_stream, vertex num_nodes, 
                        large_vertex num_edges, vertex chunk_size, 
                        std::string result_filename="res.txt",
                        bool generate_random_weights=false, bool is_weight_float=false,
                        int debug=1){

  std::cout << "streamline chunk boruvka " << std::endl;
  int num_chunks = (num_edges + chunk_size - 1)/chunk_size;
  large_vertex every_chunk_but_last = (num_chunks - 1)*chunk_size;
  vertex last_chunk_size = vertex(num_edges - every_chunk_but_last);
  std::cout << "last chunk size " << last_chunk_size << std::endl;
  std::cout << "other chunk size " << chunk_size << std::endl;

  //create a 2d array of edges of size num_chunks x chunk_size
  edge **chunk_edges; //= (edge **)malloc(num_chunks*sizeof(edge *));
  cudaMallocHost((void**)&chunk_edges, num_chunks*sizeof(edge*));
  for(int i=0; i<num_chunks-1; i++){
    //chunk_edges[i] = (edge *)malloc(chunk_size*sizeof(edge));
    cudaMallocHost((void**)&chunk_edges[i], chunk_size*sizeof(edge));
  }
  cudaMallocHost((void**)&chunk_edges[num_chunks-1], last_chunk_size*sizeof(edge));

  edge *cpu_msf;
  cudaMallocHost((void**)&cpu_msf, (num_nodes-1)*sizeof(edge));
  //populate the 2d array of edges
  std::cout << "generating random weights " << generate_random_weights << std::endl;
  populateEdgeChunksLarge(input_stream, chunk_edges, num_edges, num_nodes, 
      chunk_size, last_chunk_size, num_chunks, generate_random_weights, is_weight_float);

  std::vector<int> execution_times;
  for(int loops=0; loops<debug; loops++){
    edge *final_msf;// = (edge *)malloc(num_nodes*sizeof(edge));
    cudaMallocHost((void**)&final_msf, num_nodes*sizeof(edge));

    //DeviceMallocs start
    //Malloc edge array on device
    edge *d_edges;
    cudaMalloc(&d_edges, (chunk_size+num_nodes+last_chunk_size)*sizeof(edge));

    edge *d_edges_next;
    cudaMalloc(&d_edges_next, (chunk_size+num_nodes+last_chunk_size)*sizeof(edge));

    //Malloc a new edges array on device to be used in the iterative relabelling step on device
    edge *d_iter_edges;
    cudaMalloc(&d_iter_edges, (chunk_size+num_nodes+last_chunk_size)*sizeof(edge));

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

    vertex *d_non_self_loops;
    cudaMalloc(&d_non_self_loops, sizeof(vertex)*2);
    vertex *h_non_self_loops = (vertex *)malloc(2*sizeof(vertex));
    h_non_self_loops[0] = 1;
    cudaMemcpy(d_non_self_loops, h_non_self_loops, 2*sizeof(vertex), cudaMemcpyHostToDevice);

    vertex number_of_filled_edges=0;
    vertex chunk_num_edges = chunk_size;
    vertex next_edges_size = chunk_size;


    //Streams initialize
    int nStreams = num_chunks;
    cudaStream_t stream[nStreams];
    for(int i=0; i<num_chunks+1; i++)
      checkCuda( cudaStreamCreate(&stream[i]) );


    auto start_streamline = std::chrono::high_resolution_clock::now();
    cudaMemcpyAsync(d_edges, chunk_edges[0], chunk_size*sizeof(edge), cudaMemcpyHostToDevice,
                    stream[0]);

    for(int i=1; i<num_chunks; i++){
      std::cout << "in chunk "<< i << "\n";

      //HtoD Memcpy start
      //Copy edge array to device
      if(i < (num_chunks-1)){
        if(i != 1){
          chunk_num_edges = chunk_size+number_of_filled_edges;
          cudaMemcpyAsync(d_edges, d_edges_next, chunk_size*sizeof(edge),
                    cudaMemcpyDeviceToDevice, stream[i-1]);
          cudaMemcpyAsync(d_edges+chunk_size, d_final_msf, number_of_filled_edges*sizeof(edge), 
                    cudaMemcpyDeviceToDevice, stream[i-1]);
          if(number_of_filled_edges == (num_nodes - 1)){
            std::cout << "------------ ----------------------------------------------------------\n";
            //Find the weight of the tree
            thrust::device_vector<edge> thrust_msf(d_final_msf, d_final_msf+(num_nodes-1));
            auto max_edge = thrust::max_element(thrust_msf.begin(), 
                            thrust_msf.end(),
                            MaxWeightedCompare());
            vertex index_of_highest_weight = max_edge - thrust_msf.begin();
            std::cout << index_of_highest_weight << "\n";

          }
          else{
            std::cout << "NOT FILLED THE MSF ARRAY " << number_of_filled_edges 
                      << " edges out of " << num_nodes-1 << " edges filled \n";
          }
        }
        cudaMemcpyAsync(d_iter_edges, d_edges, chunk_num_edges*sizeof(edge), 
                        cudaMemcpyDeviceToDevice, stream[i-1]);
        cudaMemcpyAsync(d_edges_next, chunk_edges[i], chunk_size*sizeof(edge), 
                  cudaMemcpyHostToDevice, stream[i]);
        next_edges_size = chunk_size;
        number_of_filled_edges=0;
      }
      if(i==(num_chunks-1)){
        chunk_num_edges = chunk_size+number_of_filled_edges;
        cudaMemcpyAsync(d_edges, d_edges_next, chunk_size*sizeof(edge),
                  cudaMemcpyDeviceToDevice, stream[i-1]);
        cudaMemcpyAsync(d_edges+chunk_size, d_final_msf, number_of_filled_edges*sizeof(edge), 
                  cudaMemcpyDeviceToDevice, stream[i-1]);
        //cudaMemcpyAsync(d_iter_edges, d_edges, chunk_num_edges*sizeof(edge), 
        //                cudaMemcpyDeviceToDevice, stream[i-1]);
        cudaMemcpyAsync(d_edges_next, chunk_edges[i], last_chunk_size*sizeof(edge), 
                  cudaMemcpyHostToDevice, stream[i]);
 // do stuff here
        std::cout << "number of filled edges in prev msf " << number_of_filled_edges<<"\n";
        chunk_num_edges += last_chunk_size;
        cudaMemcpyAsync(d_edges+chunk_size+number_of_filled_edges, 
                  d_edges_next, last_chunk_size*sizeof(edge),
                  cudaMemcpyDeviceToDevice, stream[i]);
        cudaMemcpyAsync(d_iter_edges, d_edges, chunk_num_edges*sizeof(edge), 
                        cudaMemcpyDeviceToDevice, stream[i]);
        number_of_filled_edges = 0;
      }

      //Code begin to replicate async behaviour. Uncomment to see that thrust is 
      // forcing synchronicity
      int num_blocks = (num_edges + BLOCKSIZE - 1)/BLOCKSIZE;
      //add_one_to_edge_weight<<<num_blocks, BLOCKSIZE, 0, stream[i-1]>>>(d_edges, chunk_size);
      //asyn replicate end
      


      std::cout << "chunk num edges " << chunk_num_edges << " " << chunk_size << "\n";
      //HtoD Memcpy end

      std::cout << "chunk size " << chunk_num_edges << std::endl;
      std::cout << "next_edges_size " << next_edges_size << std::endl;
      std::cout << "num chunks " << num_chunks << std::endl;


      //populate the representative array with the node id
      thrust::device_ptr<vertex> d_representatives_ptr(d_representatives);
      //thrust::sequence(d_representatives_ptr, d_representatives_ptr+num_nodes);
      populate_representative_array<<<num_blocks, BLOCKSIZE, 
            0, stream[i-1]>>>(d_representatives, num_nodes);

      std::cout << "done intializing" << std::endl;
      //Make a copy of the representative array to use in iterations
      thrust::device_ptr<vertex> d_new_representatives_ptr(d_new_representatives);

      //thrust::copy(d_representatives_ptr, d_representatives_ptr+num_nodes, d_new_representatives_ptr);
      cudaMemcpyAsync(d_new_representatives, d_representatives, num_nodes*sizeof(vertex),
                      cudaMemcpyDeviceToDevice, stream[i-1]);

      //Make a copy of representative array to use in iterations
      thrust::device_ptr<vertex> d_iter_representatives_ptr(d_iter_representatives);
      //thrust::copy(thrust::cuda::par.on(stream[i-1]), d_representatives_ptr, 
      //            d_representatives_ptr+num_nodes, d_iter_representatives_ptr);
      cudaMemcpyAsync(d_iter_representatives, d_representatives, num_nodes*sizeof(vertex),
                      cudaMemcpyDeviceToDevice, stream[i-1]);


      //vertex num_edges_copy = num_edges;
      vertex num_nodes_copy = num_nodes;

      thrust::device_ptr<edge> d_final_msf_ptr(d_final_msf);

      cudaError err;
      
      number_of_filled_edges = 0;

      if(i < num_chunks-1){
        device_boruvka_exp(d_edges, d_final_msf, 
                        d_iter_edges, d_msf,
                        d_representatives, d_new_representatives,
                        d_iter_representatives,
                        d_outgoing_edges,
                        d_non_self_loops,
                        num_nodes, chunk_num_edges, 
                        number_of_filled_edges,
                        d_new_representatives_ptr,
                        d_representatives_ptr,
                        d_iter_representatives_ptr,
                        d_final_msf_ptr, stream[i-1]);
      }

      if (i >= num_chunks-1){
        device_boruvka(d_edges, d_final_msf, 
                        d_iter_edges, d_msf,
                        d_representatives, d_new_representatives,
                        d_iter_representatives,
                        d_outgoing_edges,
                        num_nodes, chunk_num_edges, 
                        number_of_filled_edges,
                        d_new_representatives_ptr,
                        d_representatives_ptr,
                        d_iter_representatives_ptr,
                        d_final_msf_ptr, stream[i]);
      }

      //device_boruvka(d_edges, d_final_msf, 
      //                d_iter_edges, d_msf,
      //                d_representatives, d_new_representatives,
      //                d_iter_representatives,
      //                d_outgoing_edges,
      //                num_nodes, chunk_num_edges, 
      //                number_of_filled_edges,
      //                d_new_representatives_ptr,
      //                d_representatives_ptr,
      //                d_iter_representatives_ptr,
      //                d_final_msf_ptr, stream[i-1]);



      if(i == num_chunks-1){
        auto stop_streamline = std::chrono::high_resolution_clock::now();
        auto duration_streamline = std::chrono::duration_cast<std::chrono::microseconds>(
                                   stop_streamline - start_streamline);
        //Break condition for cpu code
        std::cout<< "streamline execution time " << duration_streamline.count() << " microseconds\n";
        execution_times.push_back(duration_streamline.count());
        //Synchronize
        cudaMemcpyAsync(final_msf, d_final_msf, num_nodes_copy*sizeof(edge), 
                        cudaMemcpyDeviceToHost, stream[i-1]);
        err = cudaDeviceSynchronize();
        fprintf(stderr, "Completed memcpy %d: %s.\n", err, cudaGetErrorString(err));
        assert(err == cudaSuccess);
      }
    }
    large_vertex w = 0;
    if(result_filename == "res.txt"){
      for(vertex i=0; i<num_nodes-1; i++){
        w+=final_msf[i].w;
        //std::cout << final_msf[i].u << " " << final_msf[i].v << " " << final_msf[i].w << "\n";
      }
    }
    else{
      std::ofstream result(result_filename);
      for(vertex i=0; i< num_nodes-1;i++){
        w+=final_msf[i].w;
        result << final_msf[i].u+1 << " " << final_msf[i].v+1 
              << " " << final_msf[i].w << "\n";
      }
      result.close();
    }
    std::cout << "final weight by streamline approach " << w << "\n";

    //Free all mallocs
    cudaFree(d_edges);
    cudaFree(d_edges_next);
    cudaFree(d_representatives);
    cudaFree(d_msf);
    cudaFree(d_new_representatives);
    cudaFree(d_iter_representatives);
    cudaFree(d_outgoing_edges);
    cudaFree(d_final_msf);
    cudaFree(d_iter_edges);
  }

  vertex total_time = 0;
  for(int i=0; i<execution_times.size(); i++){
    std::cout << i << "th execution time " << execution_times[i] << std::endl;
    total_time += execution_times[i];
  }
  std::cout << "Average exec time " << total_time / execution_times.size() << std::endl;


  //Free all mallocs
  //cudaFree(d_edges);
  //cudaFree(d_edges_next);
  //cudaFree(d_representatives);
  //cudaFree(d_msf);
  //cudaFree(d_new_representatives);
  //cudaFree(d_iter_representatives);
  //cudaFree(d_outgoing_edges);
  //cudaFree(d_final_msf);
  //cudaFree(d_iter_edges);


  return;
}


void streamline_boruvka_heterogeneous(std::ifstream &input_stream, vertex num_nodes, 
                          large_vertex num_edges, vertex chunk_size, 
                          std::string result_filename="res.txt",
                          bool generate_random_weights=false, 
                          bool is_weight_float=false, int debug=1){

  std::cout << "streamline heterogeneous boruvka " << std::endl;
  std::cout << num_edges << std::endl;
  std::cout << chunk_size << std::endl;
  int num_chunks = (num_edges + chunk_size - 1)/chunk_size;
  large_vertex other_size = (num_chunks-1)*chunk_size;
  large_vertex last_chunk_size = num_edges - other_size;
  std::cout << "streamline heterogeneous boruvka " << std::endl;
  std::cout << "last chunk size " << last_chunk_size << std::endl;

  //create a 2d array of edges of size num_chunks x chunk_size
  edge **chunk_edges; //= (edge **)malloc(num_chunks*sizeof(edge *));
  cudaMallocHost((void**)&chunk_edges, num_chunks*sizeof(edge*));
  std::cout << "streamline heterogeneous boruvka " << std::endl;
  for(int i=0; i<num_chunks-1; i++){
    //chunk_edges[i] = (edge *)malloc(chunk_size*sizeof(edge));
    cudaMallocHost((void**)&chunk_edges[i], chunk_size*sizeof(edge));
  }
  std::cout << "streamline heterogeneous boruvka 1" << std::endl;
  cudaMallocHost((void**)&chunk_edges[num_chunks-1], last_chunk_size*sizeof(edge));
  std::cout << "streamline heterogeneous boruvka 2" << std::endl;

  edge *cpu_msf;
  cudaMallocHost((void**)&cpu_msf, (num_nodes-1)*sizeof(edge));
  //populate the 2d array of edges
  std::cout << "generating random weights " << generate_random_weights << std::endl;
  populateEdgeChunks(input_stream, chunk_edges, num_edges, num_nodes,
      chunk_size, last_chunk_size, num_chunks, generate_random_weights, is_weight_float);
  std::cout << "populated edge chunks\n";

  //Swap the first last_chunk_size edges with the last chunk
  #pragma omp parallel for
  for(vertex i=0; i<last_chunk_size; i++){
    vertex temp_v, temp_u;
    weight temp_w;
    temp_u = chunk_edges[num_chunks-1][i].u;
    temp_v = chunk_edges[num_chunks-1][i].v;
    temp_w = chunk_edges[num_chunks-1][i].w;
    chunk_edges[num_chunks-1][i].u = chunk_edges[0][i].u;
    chunk_edges[num_chunks-1][i].v = chunk_edges[0][i].v;
    chunk_edges[num_chunks-1][i].w = chunk_edges[0][i].w;
    chunk_edges[0][i].u = temp_u;
    chunk_edges[0][i].v = temp_v;
    chunk_edges[0][i].w = temp_w;
  }

  edge *final_msf;// = (edge *)malloc(num_nodes*sizeof(edge));
  cudaMallocHost((void**)&final_msf, num_nodes*sizeof(edge));
  //DeviceMallocs start
  //Malloc edge array on device
  edge *d_edges;
  cudaMalloc(&d_edges, (chunk_size+num_nodes)*sizeof(edge));

  edge *d_edges_next;
  cudaMalloc(&d_edges_next, (chunk_size+num_nodes)*sizeof(edge));

  //Malloc a new edges array on device to be used in the iterative relabelling step on device
  edge *d_iter_edges;
  cudaMalloc(&d_iter_edges, (chunk_size+num_nodes)*sizeof(edge));

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

  vertex *d_non_self_loops;
  cudaMalloc(&d_non_self_loops, sizeof(vertex)*2);
  vertex *h_non_self_loops = (vertex *)malloc(2*sizeof(vertex));
  h_non_self_loops[0] = 1;
  cudaMemcpy(d_non_self_loops, h_non_self_loops, 2*sizeof(vertex), cudaMemcpyHostToDevice);

  vertex number_of_filled_edges=0;
  vertex chunk_num_edges = chunk_size;
  vertex next_edges_size = chunk_size;


  //Streams initialize
  int nStreams = num_chunks;
  cudaStream_t stream[nStreams];
  for(int i=0; i<num_chunks+1; i++)
    checkCuda( cudaStreamCreate(&stream[i]) );

  std::cout << "finished gpu initialization \n";

  //CPU INITIALIZE
  vertex num_edges_cpu_pbbs = last_chunk_size;
  vertex max_edges_cpu = 100000000;
  if(num_edges_cpu_pbbs > max_edges_cpu){
    num_edges_cpu_pbbs = max_edges_cpu;
  }

  edge* edges_final_after_pbbs_filter = (edge *)malloc(last_chunk_size* sizeof(edge));

  edge_pbbs *edges_for_pbbs = (edge_pbbs *)malloc(num_edges_cpu_pbbs*2*sizeof(edge));
  for(vertex i=0;i<num_edges_cpu_pbbs; i++){
    edges_for_pbbs[i*2].u = chunk_edges[num_chunks-1][i].u;
    edges_for_pbbs[i*2].v = chunk_edges[num_chunks-1][i].v;
    edges_for_pbbs[i*2].w = chunk_edges[num_chunks-1][i].w;
    edges_for_pbbs[i*2+1].u = chunk_edges[num_chunks-1][i].v;
    edges_for_pbbs[i*2+1].v = chunk_edges[num_chunks-1][i].u;
    edges_for_pbbs[i*2+1].w = chunk_edges[num_chunks-1][i].w;
  }
  wghEdgeArray<vertexId, vertexId> edge_array = create_wghEdgeArray(edges_for_pbbs, num_edges_cpu_pbbs*2);


  std::cout<<"number of vertices: "<<edge_array.n<<std::endl;
  std::cout<<"number of edges: "<<edge_array.m<<std::endl;
  for(int i = 0; i < 10; i++)
  {
    std::cout<<"(u, v, weight) : ("<<edge_array[i].u<<", "<<edge_array[i].v<<", "<<edge_array[i].weight<<")"<<std::endl;
  }



  size_t num_edges_mst;




  auto start_streamline = std::chrono::high_resolution_clock::now();
  cudaMemcpyAsync(d_edges, chunk_edges[0], chunk_size*sizeof(edge), cudaMemcpyHostToDevice,
                  stream[0]);


  vertex last_chunk_size_final=0;

  // Mutex and condition variable for synchronization
  omp_set_nested(1); // Enable nested parallel regions
  #pragma omp parallel sections
  {
    #pragma omp section
    {
      for(int i=1; i<num_chunks+1; i++){
        std::cout << "in chunk "<< i << "\n";
        //HtoD Memcpy start
        //Copy edge array to device
        if(i < (num_chunks-1)){
          if(i != 1){
            chunk_num_edges = chunk_size+number_of_filled_edges;
            cudaMemcpyAsync(d_edges, d_edges_next, chunk_size*sizeof(edge),
                      cudaMemcpyDeviceToDevice, stream[i-1]);
            cudaMemcpyAsync(d_edges+chunk_size, d_final_msf, number_of_filled_edges*sizeof(edge), 
                      cudaMemcpyDeviceToDevice, stream[i-1]);
            if(number_of_filled_edges == (num_nodes - 1)){
              std::cout << "------------ ----------------------------------------------------------\n";
              //Find the weight of the tree
              thrust::device_vector<edge> thrust_msf(d_final_msf, d_final_msf+(num_nodes-1));
              auto max_edge = thrust::max_element(thrust_msf.begin(), 
                              thrust_msf.end(),
                              MaxWeightedCompare());
              vertex index_of_highest_weight = max_edge - thrust_msf.begin();
              std::cout << index_of_highest_weight << "\n";
            }
            else{
              std::cout << "NOT FILLED THE MSF ARRAY " << number_of_filled_edges 
                        << " edges out of " << num_nodes-1 << " edges filled \n";
            }
          }
          cudaMemcpyAsync(d_iter_edges, d_edges, chunk_num_edges*sizeof(edge), 
                          cudaMemcpyDeviceToDevice, stream[i-1]);
          cudaMemcpyAsync(d_edges_next, chunk_edges[i], chunk_size*sizeof(edge), 
                    cudaMemcpyHostToDevice, stream[i]);
          next_edges_size = chunk_size;
          number_of_filled_edges=0;
        }
        if(i==(num_chunks-1)){
          chunk_num_edges = chunk_size+number_of_filled_edges;
          cudaMemcpyAsync(d_edges, d_edges_next, chunk_size*sizeof(edge),
                    cudaMemcpyDeviceToDevice, stream[i-1]);
          cudaMemcpyAsync(d_edges+chunk_size, d_final_msf, number_of_filled_edges*sizeof(edge), 
                    cudaMemcpyDeviceToDevice, stream[i-1]);
          cudaMemcpyAsync(d_iter_edges, d_edges, chunk_num_edges*sizeof(edge), 
                          cudaMemcpyDeviceToDevice, stream[i-1]);

          std::unique_lock<std::mutex> lock(mtx);
          std::cout << "function_1 is waiting for x to be 1\n";

          // Wait until x is set to 1
          cv.wait(lock, [] { return x != 0; });
          last_chunk_size_final = x;

          std::cout << "function_1 resumed after x is 1\n";
          std::cout << "last chunk size is now " << last_chunk_size_final << 
                       " from "<< last_chunk_size << std::endl;

          cudaMemcpyAsync(d_edges_next, chunk_edges[i], last_chunk_size_final*sizeof(edge), 
                    cudaMemcpyHostToDevice, stream[i]);
          next_edges_size = last_chunk_size ;

          number_of_filled_edges=0;
        }
        if(i==num_chunks){
          std::cout << "number of filled edges in prev msf " << number_of_filled_edges<<"\n";
          chunk_num_edges = last_chunk_size+number_of_filled_edges;
          cudaMemcpyAsync(d_edges, d_edges_next, last_chunk_size*sizeof(edge),
                    cudaMemcpyDeviceToDevice, stream[i-1]);
          cudaMemcpyAsync(d_edges+last_chunk_size, d_final_msf, number_of_filled_edges*sizeof(edge),
                    cudaMemcpyDeviceToDevice, stream[i-1]);
          cudaMemcpyAsync(d_iter_edges, d_edges, chunk_num_edges*sizeof(edge), 
                          cudaMemcpyDeviceToDevice, stream[i-1]);
          number_of_filled_edges = 0;
        }

        //Code begin to replicate async behaviour. Uncomment to see that thrust is 
        // forcing synchronicity
        int num_blocks = (num_edges + BLOCKSIZE - 1)/BLOCKSIZE;
        //add_one_to_edge_weight<<<num_blocks, BLOCKSIZE, 0, stream[i-1]>>>(d_edges, chunk_size);
        //asyn replicate end
        


        std::cout << "chunk num edges " << chunk_num_edges << " " << chunk_size << "\n";
        //HtoD Memcpy end

        std::cout << "chunk size " << chunk_num_edges << std::endl;
        std::cout << "next_edges_size " << next_edges_size << std::endl;
        std::cout << "num chunks " << num_chunks << std::endl;


        //populate the representative array with the node id
        thrust::device_ptr<vertex> d_representatives_ptr(d_representatives);
        //thrust::sequence(d_representatives_ptr, d_representatives_ptr+num_nodes);
        populate_representative_array<<<num_blocks, BLOCKSIZE, 
              0, stream[i-1]>>>(d_representatives, num_nodes);

        std::cout << "done intializing" << std::endl;
        //Make a copy of the representative array to use in iterations
        thrust::device_ptr<vertex> d_new_representatives_ptr(d_new_representatives);

        //thrust::copy(d_representatives_ptr, d_representatives_ptr+num_nodes, d_new_representatives_ptr);
        cudaMemcpyAsync(d_new_representatives, d_representatives, num_nodes*sizeof(vertex),
                        cudaMemcpyDeviceToDevice, stream[i-1]);

        //Make a copy of representative array to use in iterations
        thrust::device_ptr<vertex> d_iter_representatives_ptr(d_iter_representatives);
        //thrust::copy(thrust::cuda::par.on(stream[i-1]), d_representatives_ptr, 
        //            d_representatives_ptr+num_nodes, d_iter_representatives_ptr);
        cudaMemcpyAsync(d_iter_representatives, d_representatives, num_nodes*sizeof(vertex),
                        cudaMemcpyDeviceToDevice, stream[i-1]);


        //vertex num_edges_copy = num_edges;
        vertex num_nodes_copy = num_nodes;

        thrust::device_ptr<edge> d_final_msf_ptr(d_final_msf);

        cudaError err;
        
        number_of_filled_edges = 0;


        if(i < num_chunks-1){
          device_boruvka_exp(d_edges, d_final_msf, 
                          d_iter_edges, d_msf,
                          d_representatives, d_new_representatives,
                          d_iter_representatives,
                          d_outgoing_edges,
                          d_non_self_loops,
                          num_nodes, chunk_num_edges, 
                          number_of_filled_edges,
                          d_new_representatives_ptr,
                          d_representatives_ptr,
                          d_iter_representatives_ptr,
                          d_final_msf_ptr, stream[i-1]);
        }

        if (i >= num_chunks-1){
          device_boruvka(d_edges, d_final_msf, 
                          d_iter_edges, d_msf,
                          d_representatives, d_new_representatives,
                          d_iter_representatives,
                          d_outgoing_edges,
                          num_nodes, chunk_num_edges, 
                          number_of_filled_edges,
                          d_new_representatives_ptr,
                          d_representatives_ptr,
                          d_iter_representatives_ptr,
                          d_final_msf_ptr, stream[i-1]);
        }

        //device_boruvka(d_edges, d_final_msf, 
        //                d_iter_edges, d_msf,
        //                d_representatives, d_new_representatives,
        //                d_iter_representatives,
        //                d_outgoing_edges,
        //                num_nodes, chunk_num_edges, 
        //                number_of_filled_edges,
        //                d_new_representatives_ptr,
        //                d_representatives_ptr,
        //                d_iter_representatives_ptr,
        //                d_final_msf_ptr, stream[i-1]);


        if(i == num_chunks){
          auto stop_streamline = std::chrono::high_resolution_clock::now();
          auto duration_streamline = std::chrono::duration_cast<std::chrono::microseconds>(
                                     stop_streamline - start_streamline);
          //Break condition for cpu code
          std::cout<< "streamline execution time " << duration_streamline.count() << " microseconds\n";
          //Synchronize
          cudaMemcpyAsync(final_msf, d_final_msf, num_nodes_copy*sizeof(edge), 
                          cudaMemcpyDeviceToHost, stream[i-1]);
          err = cudaDeviceSynchronize();
          fprintf(stderr, "Completed memcpy %d: %s.\n", err, cudaGetErrorString(err));
          assert(err == cudaSuccess);
        }
      }
    }
    #pragma omp section
    {
      vertex total_offset = 0;

      auto start_pbbs= std::chrono::high_resolution_clock::now();
      edge* mst_array = cpu_compute_pbbs(num_edges_mst, edge_array);

      auto stop_pbbs = std::chrono::high_resolution_clock::now();
      auto duration_pbbs = std::chrono::duration_cast<std::chrono::microseconds>(
                                 stop_pbbs - start_pbbs);
      auto start_aux_copy= std::chrono::high_resolution_clock::now();
      #pragma omp parallel for
      for(vertex j=0; j<num_edges_mst; j++){
	edges_final_after_pbbs_filter[j].u = mst_array[j].u;
	edges_final_after_pbbs_filter[j].v = mst_array[j].v;
	edges_final_after_pbbs_filter[j].w = mst_array[j].w;
      }

      vertex diff_edges = 0;
      if(last_chunk_size > max_edges_cpu){
	diff_edges = last_chunk_size - max_edges_cpu;
	#pragma omp parallel for
	for(vertex j=0; j<diff_edges; j++){
	  edges_final_after_pbbs_filter[num_edges_mst+j].u = chunk_edges[num_chunks-1][max_edges_cpu+j].u;
	  edges_final_after_pbbs_filter[num_edges_mst+j].v = chunk_edges[num_chunks-1][max_edges_cpu+j].v;
	  edges_final_after_pbbs_filter[num_edges_mst+j].w = chunk_edges[num_chunks-1][max_edges_cpu+j].w;
	}
      }

      total_offset = num_edges_mst + diff_edges;

      #pragma omp parallel for
      for(vertex j=0; j<total_offset; j++){
	chunk_edges[num_chunks-1][j].u = edges_final_after_pbbs_filter[j].u;
	chunk_edges[num_chunks-1][j].v = edges_final_after_pbbs_filter[j].v;
	chunk_edges[num_chunks-1][j].w = edges_final_after_pbbs_filter[j].w;
      }

      auto stop_aux_copy= std::chrono::high_resolution_clock::now();

      auto duration_aux_copy = std::chrono::duration_cast<std::chrono::microseconds>(
                                 stop_aux_copy - start_aux_copy);

      std::cout << last_chunk_size - diff_edges << " cut down to  " << total_offset-diff_edges << std::endl;
      //std::cout << "number of filled final cpu " << number_of_filled_edges_cpu_new << std::endl;
      last_chunk_size_final = total_offset;

      std::cout << "num edges mst " << num_edges_mst << std::endl;
      std::cout << "last_chunk_size" << last_chunk_size << std::endl;
      std::cout << "max edges cpu " << max_edges_cpu<< std::endl;


      std::cout << "updated last_chunk_size_final " << last_chunk_size_final << "\n";
      std::cout<< "pbbs execution time " << duration_pbbs.count() << " microseconds\n";
      std::cout<< "aux copy time " << duration_aux_copy.count() << " microseconds\n";
      std::cout << "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n";
      std::lock_guard<std::mutex> lock(mtx);
      x = last_chunk_size_final;
      std::cout << "set x to last_chunk_size_final\n";
      cv.notify_one();
    }
  }
  free(edges_final_after_pbbs_filter);
  free(edges_for_pbbs);



  large_vertex w = 0;
  if(result_filename == "res.txt"){
    for(vertex i=0; i<num_nodes-1; i++){
      w+=final_msf[i].w;
      //std::cout << final_msf[i].u << " " << final_msf[i].v << " " << final_msf[i].w << "\n";
    }
  }
  else{
    std::ofstream result(result_filename);
    for(vertex i=0; i< num_nodes-1;i++){
      w+=final_msf[i].w;
      result << final_msf[i].u+1 << " " << final_msf[i].v+1 
            << " " << final_msf[i].w << "\n";
    }
    result.close();
  }
  std::cout << "final weight by streamline approach " << w << "\n";


  //Free all mallocs
  cudaFree(d_edges);
  cudaFree(d_edges_next);
  cudaFree(d_representatives);
  cudaFree(d_msf);
  cudaFree(d_new_representatives);
  cudaFree(d_iter_representatives);
  cudaFree(d_outgoing_edges);
  cudaFree(d_final_msf);
  cudaFree(d_iter_edges);


  return;
}
