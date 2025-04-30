#include <set>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#include "cuda_utility.cuh"
#include "connected_components.cuh"

__global__
void initialise(int* parent, int n) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		parent[tid] = tid;
	}
}

// select_flag is your flag
__global__ 
void hooking(long numEdges, uint64_t* d_edgelist, int* d_rep, int* d_flag, int* select_flag, int itr_no) 
{
	long tid = blockDim.x * blockIdx.x + threadIdx.x;
	#ifdef DEBUG
		if(tid == 0) {
			printf("\nIteration number = %d", itr_no);
			printf("\nFlag inside kernel before start of iteration = %d", *d_flag);
		}
	#endif

	if(tid < numEdges) {

		if(select_flag[tid] == 1) {

		uint64_t i = d_edgelist[tid];

        int edge_u = i >> 32;  // Extract higher 32 bits
        int edge_v = i & 0xFFFFFFFF; // Extract lower 32 bits

		//printf("edge_u = %d, edge_v = %d\n", edge_u, edge_v);

		int comp_u = d_rep[edge_u];
		int comp_v = d_rep[edge_v];

		if(comp_u != comp_v) 
		{
			*d_flag = 1;
			int max = (comp_u > comp_v) ? comp_u : comp_v;
			int min = (comp_u < comp_v) ? comp_u : comp_v;

			if(itr_no%2) {
				d_rep[min] = max;
			}
			else { 
				d_rep[max] = min;
			}
		}

	}
	}
}

__global__ 
void short_cutting(int n, int* d_parent) {
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < n) {
		if(d_parent[tid] != tid) {
			d_parent[tid] = d_parent[d_parent[tid]];
		}
	}	
}

// u_arr, v_arr and d_select_flag are device arrays
void connected_comp(long numEdges, uint64_t* d_edgelist, int* d_select_flag, int numVert, int* d_rep, int* d_flag) {

    const long numThreads = 1024;
    long numBlocks = (numVert + numThreads - 1) / numThreads;

    std::cout << "numVert: " << numVert << " and numEdges: " << numEdges << std::endl;
    std::cout << "In CC, numBlocks: " << numBlocks << " and numThreads: " << numThreads << std::endl;

    CUDA_CHECK(cudaDeviceSynchronize(), "No carrying forward error");

    cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max blocks (grid size x): " << prop.maxGridSize[0] << std::endl;

	initialise<<<numBlocks, numThreads>>>(d_rep, numVert);
	cudaError_t err = cudaGetLastError();
	CUDA_CHECK(err, "Error in launching initialise kernel");

	int flag = 1;
	int iteration = 0;

	const long numBlocks_hooking = (numEdges + numThreads - 1) / numThreads;
	const long numBlocks_updating_parent = (numVert + numThreads - 1) / numThreads;

	while(flag) {
		flag = 0;
		iteration++;
		CUDA_CHECK(cudaMemcpy(d_flag, &flag, sizeof(int),cudaMemcpyHostToDevice), "Unable to copy the flag to device");

		// select_flag is your flag
		hooking<<<numBlocks_hooking, numThreads>>> (numEdges, d_edgelist, d_rep, d_flag, d_select_flag, iteration);
		err = cudaGetLastError();
		CUDA_CHECK(err, "Error in launching hooking kernel");
		
		#ifdef DEBUG
			std::vector<int> host_rep(numVert);
			cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost);
			// Printing the data
			std::cout << "\niteration num : "<< iteration << std::endl;
			std::cout << "d_rep : ";
			for (int i = 0; i < numVert; i++) {
			    std::cout << host_rep[i] << " ";
			}
			std::cout << std::endl;
		#endif

		// This module can be replaced with optimised pointer jumping module
		for(int i = 0; i < std::ceil(std::log2(numVert)); ++i) {
			short_cutting<<<numBlocks_updating_parent, numThreads>>> (numVert, d_rep);
			err = cudaGetLastError();
			CUDA_CHECK(err, "Error in launching short_cutting kernel");
		}

		CUDA_CHECK(cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back flag to host");
	}

	if(g_verbose) {
		//std::cout <<"Number of iteration: " << iteration << std::endl;
		std::vector<int> host_rep(numVert);
		CUDA_CHECK(cudaMemcpy(host_rep.data(), d_rep, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back rep array");
		std::set<int> num_comp(host_rep.begin(), host_rep.end());

		std::cout <<"numComp after removing fence and back edges = " << num_comp.size() << std::endl;
		std::cout <<"numBCC = " << num_comp.size() - 1 << std::endl;
	}
}
