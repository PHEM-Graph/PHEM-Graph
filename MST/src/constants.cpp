typedef uint32_t vertex;
//typedef uint64_t vertex;
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
	vertex* parent; 
	vertex* rank; 

public: 
	DSU(vertex n) 
	{ 
		parent = new vertex[n]; 
		rank = new vertex[n]; 

		for (vertex i = 0; i < n; i++) { 
			parent[i] = std::numeric_limits<vertex>::max(); 
			rank[i] = 1; 
		} 
	} 

	// Find function 
	vertex find(vertex i) 
	{ 
		if (parent[i] == std::numeric_limits<vertex>::max()) 
			return i; 

		return parent[i] = find(parent[i]); 
	} 

	// Union function 
	void unite(vertex x, vertex y) 
	{ 
		vertex s1 = find(x); 
		vertex s2 = find(y); 

		if (s1 != s2) { 
			if (rank[s1] < rank[s2]) { 
				parent[s1] = s2; 
			} 
			else if (rank[s1] > rank[s2]) { 
				parent[s2] = s1; 
			} 
			else { 
				parent[s2] = s1; 
				rank[s1] += 1; 
			} 
		} 
	} 
}; 

bool compare_by_weight(const edge& lhs, const edge& rhs){
  return lhs.w < rhs.w;
}

void kruskals_mst(std::vector<edge> graph, DSU s,
                  std::vector<edge> &final_msf, 
                  vertex &number_of_filled_edges_cpu) 
{ 
	// Sort all edges 
  std::sort(graph.begin(), graph.end(), compare_by_weight); 

	// Initialize the DSU 
	//DSU s(V); 
	int ans = 0; 
  std::cout << "Following are the edges in the "
			"constructed MST"
		<< std::endl; 
	for (auto edge : graph) { 
		vertex w = edge.w; 
		vertex x = edge.u; 
		vertex y = edge.v; 

		// Take this edge in MST if it does 
		// not forms a cycle 
		if (s.find(x) != s.find(y)) { 
			s.unite(x, y); 
			ans += w; 
      number_of_filled_edges_cpu++;

      final_msf.push_back({x, y, w});
      //std::cout << x << " -- " << y << " == " << w << std::endl; 
		} 
	} 
  std::cout << "Minimum Cost Spanning Tree: " << ans << std::endl; 
} 
 
class dsu { 
  std::vector<vertex> parent; 
	std::vector<vertex> rank; 

public: 
	dsu(vertex n) : parent(n, std::numeric_limits<vertex>::max()), rank(n, 1)
	{ 
		//parent = new vertex[n]; 
		//rank = new vertex[n]; 

		//for (vertex i = 0; i < n; i++) { 
		//	parent[i] = std::numeric_limits<vertex>::max(); 
		//	rank[i] = 1; 
		//} 
	} 

	// Find function 
	vertex find(vertex i) 
	{ 
		if (parent[i] == std::numeric_limits<vertex>::max()) 
			return i; 

		return parent[i] = find(parent[i]); 
	} 

	// Union function 
	void unite(vertex x, vertex y) 
	{ 
		vertex s1 = find(x); 
		vertex s2 = find(y); 

		if (s1 != s2) { 
			if (rank[s1] < rank[s2]) { 
				parent[s1] = s2; 
			} 
			else if (rank[s1] > rank[s2]) { 
				parent[s2] = s1; 
			} 
			else { 
				parent[s2] = s1; 
				rank[s1] += 1; 
			} 
		} 
	} 
}; 



void kruskals_mst_new(std::vector<edge> graph, dsu s,
                  std::vector<edge> &final_msf, 
                  vertex &number_of_filled_edges_cpu) 
{ 
	// Sort all edges 
  std::sort(graph.begin(), graph.end(), compare_by_weight); 

	// Initialize the DSU 
	//DSU s(V); 
	int ans = 0; 
  std::cout << "Following are the edges in the "
			"constructed MST"
		<< std::endl; 
	for (auto edge : graph) { 
		vertex w = edge.w; 
		vertex x = edge.u; 
		vertex y = edge.v; 

		// Take this edge in MST if it does 
		// not forms a cycle 
		if (s.find(x) != s.find(y)) { 
			s.unite(x, y); 
			ans += w; 
      number_of_filled_edges_cpu++;

      final_msf.push_back({x, y, w});
      //std::cout << x << " -- " << y << " == " << w << std::endl; 
		} 
	} 
  std::cout << "Minimum Cost Spanning Tree: " << ans << std::endl; 
} 
