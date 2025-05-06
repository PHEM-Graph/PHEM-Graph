#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <limits>
#include <vector>
#include <algorithm>

typedef uint32_t vertex;
typedef uint32_t weight;
typedef uint64_t large_vertex;

const weight sentinel = std::numeric_limits<weight>::max();
const vertex sentinel_vertex = std::numeric_limits<vertex>::max();

struct edge{
  vertex u;
  vertex v;
  weight w;
};

struct outgoing_edge_id{
  vertex e_id;
  weight w;
};

struct weighted_parent{
  vertex parent;
  weight w;
};

class DSU {
  private:
    std::vector<vertex> parent;
    std::vector<vertex> rank;
  
  public:
    DSU(vertex n) {
      parent.resize(n);
      rank.resize(n);
      for (vertex i = 0; i < n; i++) {
        parent[i] = i;
        rank[i] = 0;
      }
    }
  
    vertex find(vertex x) {
      if (parent[x] != x) {
        parent[x] = find(parent[x]);
      }
      return parent[x];
    }
  
    void unite(vertex x, vertex y) {
      vertex xset = find(x);
      vertex yset = find(y);
  
      if (xset == yset) {
        return;
      }
  
      if (rank[xset] < rank[yset]) {
        parent[xset] = yset;
      } else if (rank[xset] > rank[yset]) {
        parent[yset] = xset;
      } else {
        parent[yset] = xset;
        rank[xset]++;
      }
    }
};

// Function to compare edges by weight
bool compare_by_weight(const edge& a, const edge& b) {
  return a.w < b.w;
}

// Kruskal's algorithm to find MST
void kruskals_mst(std::vector<edge>& graph, DSU& s, std::vector<edge>& mst, vertex& mst_size) {
  // Sort edges in increasing order of weight
  std::sort(graph.begin(), graph.end(), compare_by_weight);
  
  mst_size = 0;
  
  for (const auto& e : graph) {
    vertex u = e.u;
    vertex v = e.v;
    
    vertex set_u = s.find(u);
    vertex set_v = s.find(v);
    
    // If including this edge doesn't cause a cycle, include it in the MST
    if (set_u != set_v) {
      mst[mst_size++] = e;
      s.unite(set_u, set_v);
    }
  }
}

#endif // CONSTANTS_H 