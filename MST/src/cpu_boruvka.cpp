#include <random>
#include <stdlib.h>
#include <omp.h>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <string_view>
#include <iterator>
//#include <oneapi/tbb.h>

//#include <tbb/parallel_for.h>
//#include <tbb/atomic.h>
#include <algorithm>
#include <iomanip>
#include <execution>
#include "constants.cpp"
#include <chrono> 



vertex hash_weight(vertex a){
  a = (a ^ 61) ^ (a >> 16);
  a = a + (a << 3);
  a = a ^ (a >> 4);
  a = a * 0x27d4eb2d;
  a = a ^ (a >> 15);
  return a;
}

void reset_edge_array_to_sentinel_cpu(edge* edges, vertex num_edges){
  #pragma omp parallel for
  for(vertex i=0; i<num_edges;i++){
    edges[i].u = sentinel_vertex;
    edges[i].v = sentinel_vertex;
    edges[i].w = sentinel;
  }
}

void reset_outgoing_edge_array_cpu(outgoing_edge_id* outgoing_edges,
                                   vertex num_nodes, vertex num_edges){
  #pragma omp parallel for
  for(vertex i=0; i<num_nodes; i++){
    outgoing_edges[i].e_id = num_edges +1;
    outgoing_edges[i].w = sentinel;
  }
}

void copy_representative_arrays_cpu(vertex* representative, vertex* iter_representatives,
                                vertex num_nodes){
  #pragma omp parallel for
  for(vertex i=0; i<num_nodes; i++){
    representative[i] = iter_representatives[i];
  }
}

void getGraphInfo(std::ifstream &file, vertex &num_nodes, vertex &num_edges){
    std::string line;
    //skip if line starts with %
    while(std::getline(file, line)){
        if(line[0] != '%')
            break;
    }
    std::stringstream ss(line);
    vertex tmp, tmp1;
    ss >> num_nodes >> tmp >> num_edges;
}

void readGraph(edge *edges, vertex num_nodes, vertex num_edges,
               std::ifstream &file, bool generate_random_weight){

  std::string line; 
  vertex u, v;
  weight w;
  if(!generate_random_weight){
    for(int i=0; i<num_edges; i++){
      std::getline(file, line);
      std::stringstream ss(line);
      ss >> u >> v >> w;
      edges[i].u = u-1;
      edges[i].v = v-1;
      edges[i].w = w;
    }
    return;
  }
  if(generate_random_weight){
    for(int i=0; i<num_edges; i++){
      std::getline(file, line);
      std::stringstream ss(line);
      ss >> u >> v;
      w = (hash_weight(v)%10) + 1;
      edges[i].u = u-1;
      edges[i].v = v-1;
      edges[i].w = w;
    }
    return;

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

void populate_min_weight_cpu(edge* edges,
                            outgoing_edge_id* best_index,
                            vertex num_edges, vertex num_nodes){
  #pragma omp parallel for
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

void populate_eid_cpu(edge* edges,
                  outgoing_edge_id* best_index,
                  vertex num_edges, vertex num_nodes){
  #pragma omp parallel for
  for(vertex i=0; i<num_edges; i++){
    edge tid_e = edges[i];
    vertex tid_u = tid_e.u;
    vertex tid_v = tid_e.v;
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



  return;
}


void grafting_edges_cpu(outgoing_edge_id* outgoing_edges,
                        edge* iter_edges, edge* edges,
                        edge* msf, vertex* representative,
                        vertex num_nodes, vertex num_edges){
  #pragma omp parallel for
  for(int i=0; i<num_nodes; i++){
    outgoing_edge_id tid_outgoing_edge_id = outgoing_edges[i];
    vertex tid_outgoing_e_id = tid_outgoing_edge_id.e_id;
    //weight tid_best_w = tid_outgoing_edge_id.w;
    if(tid_outgoing_e_id < (num_edges) && tid_outgoing_e_id >= 0){
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



void shortcutting_step_cpu(vertex* representative,
                vertex num_nodes, weight sentinel, vertex* iter_representatives){
  #pragma omp parallel for
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

void relabelling_step_cpu(edge* edges, vertex* representative,
                      vertex num_edges, vertex num_nodes){
  #pragma omp parallel for
  for(vertex i=0; i<num_edges; i++){
    vertex u = edges[i].u;
    vertex v = edges[i].v;
    //if(u == v){
    //  edges[i].w = sentinel;
    //}
    if(u != sentinel_vertex){
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


void filtering_step_cpu(edge* new_edges, vertex* representative,
                    vertex num_edges, vertex num_nodes){

  #pragma omp parallel for
  for(vertex i=0; i<num_edges; i++){
    vertex u = new_edges[i].u;
    if(u != sentinel_vertex){
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



void boruvka_for_edge_array_cpu(edge* edges, vertex num_nodes,
                                vertex num_edges, edge* final_msf,
                                vertex number_of_filled_edges,
                                std::string result = "res.txt"){
  //print the graph params
  std::cout << "num nodes " << num_nodes << std::endl;
  std::cout << "num_edges " << num_edges << std::endl;

  //Malloc three representative arrays
  vertex* representative_array = (vertex *)malloc(num_nodes*sizeof(vertex));
  vertex* representative_array_new = (vertex *)malloc(num_nodes*sizeof(vertex));
  vertex* representative_array_iter = (vertex *)malloc(num_nodes*sizeof(vertex));
  //Malloc a copy of edges for iterations
  edge* iter_edges = (edge *)malloc(num_edges*sizeof(edge));
  edge* iter_msf = (edge *)malloc(num_nodes*sizeof(edge));
  //Malloc outgoing_edge_id array
  outgoing_edge_id* outgoing_edges = (outgoing_edge_id *)malloc(num_nodes*sizeof(outgoing_edge_id));

  //Populate representative arrays
  #pragma omp parallel for
  for(int i=0; i<num_nodes; i++){
    representative_array[i] = i;
    representative_array_new[i] = i;
    representative_array_iter[i] = i;
  }
  //Deep copy edges to iter_edges
  #pragma omp parallel for
  for(int i=0; i<num_edges; i++){
    iter_edges[i].u = edges[i].u;
    iter_edges[i].v = edges[i].v;
    iter_edges[i].w = edges[i].w;
  }

  int iter = 0;
  vertex offset_msf = 0;

  vertex prev_number_of_unique_nodes = num_nodes;
  vertex prev_num_edges = num_edges;

  auto start = std::chrono::high_resolution_clock::now();
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
    //for(int i=0; i<num_nodes; i++){
    //  std::cout << outgoing_edges[i].e_id << " " << outgoing_edges[i].w << "\n";
    //}
    //grafting edges
    std::cout << "grafted edges \n";
    grafting_edges_cpu(outgoing_edges,
                      iter_edges, edges,
                      iter_msf, representative_array,
                      num_nodes, num_edges);
    std::cout << "\n";

    //find number of edges in msf where edge weight is not sentinel
    vertex non_sentinel_edges = 0;
    //for(int i=0; i<num_nodes; i++){
    //  if(iter_msf[i].w != sentinel){
    //    final_msf[offset_msf].u = iter_msf[i].u;
    //    final_msf[offset_msf].v = iter_msf[i].v;
    //    final_msf[offset_msf].w = iter_msf[i].w;
    //    offset_msf++;
    //    non_sentinel_edges++;
    //  }
    //}

    vertex candidate_edges = std::count_if(std::execution::par, iter_msf, iter_msf+num_nodes, 
                              check_if_edge_weight_is_not_sentinel);

    std::cout << "counted \n";

    std::copy_if(std::execution::par, iter_msf, iter_msf+num_nodes, final_msf+offset_msf, 
                check_if_edge_weight_is_not_sentinel);
    std::cout << "completed copy \n";

    offset_msf += candidate_edges;

    //number_of_filled_edges += non_sentinel_edges;
    //std::cout << "NUMBER OF FILLED EDGES " << offset_msf<< "\n";




    //if(offset_msf >= num_nodes-1){
    //  std::cout << "MSF ARRAY OVERFLOW\n";
    //  break;
    //}


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
    std::remove_if(std::execution::par, edges, 
                    edges + num_edges, [](edge e){ 
                    return e.w == sentinel;});
    //parallel_remove_if(edges, num_edges, num_nodes);
    std::cout << "completed remove1 \n";
    
    //Remove self loop edges in 'iter_edges'
    //auto iter_edges_start(iter_edges);
    

    std::remove_if(std::execution::par, iter_edges, 
                   iter_edges+ num_edges, [](edge e){return e.u == e.v;});
    std::cout << "completed remove2 \n";
    vertex new_num_edges = std::count_if(std::execution::par, iter_edges,
                                        iter_edges+num_edges, [](edge e){return e.u != e.v;});
    std::cout << "completed counting new num edges\n";


    //for(int i=0; i<num_edges; i++){
    //  std::cout << iter_edges[i].u << " " << iter_edges[i].v << "\n";
    //}
    std::cout << "Number of non self edges  "  << new_num_edges << "\n";

    //update num_edges
    num_edges = new_num_edges;
  
    std::cout << "NEW NUM EDGES " << num_edges << "\n";
    if(num_edges == 0){
      std::cout << "done in " << iter << " iterations \n";
      break;
    }

    //Update iter_edges and edges array
    //iter_edges[0] = iter_edges_vector[0];
    //edges[0] = edges_vector[0];
    //std::cout << "iter edge " << iter_edges[num_edges-1].u << " " << iter_edges[num_edges-1].v << "\n";
    //std::cout << "edge " << edges[0].u << " " << edges[0].v << "\n";
    //#pragma omp parallel for
    //for(vertex i=0; i<num_edges; i++){
    //  iter_edges[i].u = iter_edges_vector[i].u;
    //  iter_edges[i].v = iter_edges_vector[i].v;
    //  iter_edges[i].w = iter_edges_vector[i].w;
    //  edges[i].u = edges_vector[i].u;
    //  edges[i].v = edges_vector[i].v;
    //  edges[i].w = edges_vector[i].w;
    //}





    if(prev_num_edges == num_edges){
      std::cout << "no change in num edges \n";
      std::cout << "number of non sentinel edges "<< non_sentinel_edges << "\n";
      break;
    }
    iter++;

  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop-start);
  std::cout << "Time to execute " << duration.count() << " microseconds"<< std::endl;

  //Free mallocs
  free(representative_array);
  free(representative_array_new);
  free(representative_array_iter);
  free(iter_msf);
  free(outgoing_edges);
  free(iter_edges);

}

int main(int argc, char* argv[]){
  std::string filename ;//= "../graph.txt";
	std::string result_file;
	filename = argv[1];
  bool generate_random_weight = atoi(argv[2]);
  std::ifstream file(filename);
  vertex num_nodes, num_edges, number_of_filled_edges;
  getGraphInfo(file, num_nodes, num_edges);
  edge *edges = (edge *)malloc(num_edges*sizeof(edge));
  readGraph(edges, num_nodes, num_edges, file, generate_random_weight);
  edge *final_msf = (edge *)malloc(num_nodes*sizeof(edge));

  //for(int i=0; i<num_edges; i++){
  //  std::cout << edges[i].u << " " << edges[i].v << " " << edges[i].w << std::endl;
  //}
  boruvka_for_edge_array_cpu(edges, num_nodes, num_edges, 
                          final_msf, number_of_filled_edges);
  
  weight w = 0;
  for(vertex i=0; i<num_nodes-1;i++){
    //std::cout << final_msf[i].u +1<< " " << final_msf[i].v+1 << " "
    //  <<final_msf[i].w << "\n";
    w += final_msf[i].w;
  }
  std::cout << "total weight " << w << "\n";

  //test_tbb();

  //Testing remove_if logic
  //std::string str1{"Text with some   spaces"};
 
  //auto noSpaceEnd = std::remove(str1.begin(), str1.end(), ' ');
 
  //// The spaces are removed from the string only logically.
  //// Note, we use view, the original string is still not shrunk:
 
  //str1.erase(noSpaceEnd, str1.end());
 
  //// The spaces are removed from the string physically.
  //std::cout << str1 << " size: " << str1.size() << '\n';
 
  //std::string str2 = "Text\n with\tsome \t  whitespaces\n\n";
  //str2.erase(std::remove_if(str2.begin(), 
  //                          str2.end(),
  //                          [](unsigned char x) { return std::isspace(x); }),
  //           str2.end());
  //std::cout << str2 << '\n';

  return 0;
}
