
Example Usage: 

PHEM Run             - ./mst /path/to/graph.mtx 0 out.txt 1 0 0 0 1 0 4

GPU Streams only run - ./mst /path/to/graph.mtx 0 out.txt 1 0 1 0 0 0 4

PBBS impl only run   - ./mst /path/to/graph.mtx 0 out.txt 1 0 0 1 0 0 4

Unified Memory run   - ./mst /path/to/graph.mtx 0 out.txt 1 1 0 0 0 0 4

This program requires following parameters in the following format. ./mst '/path/to/graph.mtx' 'Debug (1/0)' '/path/to/output.txt' 'require random weights (1/0)' 'Use managed memory (1/0)' 'Use GPU streams (1/0)' 'Use CPU only version (1/0)' 'Use PHEM model (1/0)' 'Number of Chunks' 'require debug (1/0)'

