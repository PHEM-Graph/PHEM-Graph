//---------------------------------------------------------------------
// Standard Libraries
//---------------------------------------------------------------------
#include <iostream>  
#include <random> 

//---------------------------------------------------------------------
// Graph data-structure
//---------------------------------------------------------------------
#include "graph.cuh" 

//---------------------------------------------------------------------
// Utility Functions
//---------------------------------------------------------------------
#include "timer.hpp"             
#include "utility.hpp"           
#include "cuda_utility.cuh"
#include "CommandLineParser.cuh" 

//---------------------------------------------------------------------
// Biconnected Components (BCC) related functions
//---------------------------------------------------------------------
#include "fast_bcc.cuh"
#include "bcc_memory_utils.cuh"

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool    checker             = false;
bool    g_verbose           = false;  // Whether to display i/o to console
int     local_block_size    = 20;
long    maxThreadsPerBlock  = 0;

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------
int main(int argc, char* argv[]) {
    std::ios_base::sync_with_stdio(false);

    // Initialize command line parser and retrieve parsed arguments.
    CommandLineParser cmdParser(argc, argv);
    const auto& args = cmdParser.getArgs();
        
    if (args.error) {
        std::cerr << CommandLineParser::help_msg << std::endl;
        exit(EXIT_FAILURE);
    }

    try {
        std::string filename = args.inputFile;
        std::cout <<"\nReading " << get_file_extension(filename) << " file." << std::endl << std::endl;
        undirected_graph g(filename);

        int     num_vert    =   g.getNumVertices();
        long    num_edges   =   g.getNumEdges() / 2;
        
        // Initialize device
        cuda_init(args.cudaDevice);
        g_verbose = args.verbose;
        checker   = args.g_checker_mode;

        // print basic_stats
        g.basic_stats(maxThreadsPerBlock, g_verbose, checker);

        if(g_verbose) {
            std::cout <<"\n\nEdgeList:\n";
            g.print_edgelist();
        }
        // Create External - BCC object
        GPU_BCG gpu_bcg(num_vert, num_edges, args.batchSize);

        gpu_bcg.h_edgelist = g.getList();

        std::cout << "Starting External BCC:" << std::endl;
        // start external_bcc
        int start = 0;

        CUDA_CHECK(cudaMemcpy(
            gpu_bcg.updated_edgelist, 
            gpu_bcg.h_edgelist + start, 
            num_edges * sizeof(uint64_t), 
            cudaMemcpyHostToDevice), 
        "Memcpy error");

        Timer myTimer;
        Fast_BCC(gpu_bcg, false);
        auto dur = myTimer.stop();

        std::cout << "GPU-based Fast BCC finished in : " << dur << " ms." << std::endl;

        std::vector<int> host_rep(gpu_bcg.numVert);
        CUDA_CHECK(cudaMemcpy(host_rep.data(), gpu_bcg.d_rep, gpu_bcg.numVert * sizeof(int), cudaMemcpyDeviceToHost), "Unable to copy back rep array");
        std::set<int> num_comp(host_rep.begin(), host_rep.end());

        int root_freq[1];
        int root = gpu_bcg.last_root;
        size_t final_count = num_comp.size();
        CUDA_CHECK(cudaMemcpy(root_freq , gpu_bcg.d_counter + root , sizeof(int) , cudaMemcpyDeviceToHost), "Failed to copy");
        std::cout<<"root's value : "<<root_freq[0]<<"\n";
        if(root_freq[0] == 1) {
            std::cout<<"final bcc's count : "<<final_count-1<<"\n";
        }
        else{
            std::cout<<"final bcc's count : "<<final_count<<"\n";
        }

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// ====[ End of Main Code ]====