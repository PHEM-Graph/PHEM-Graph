#ifndef CUDA_LIST_RANKING_CUH
#define CUDA_LIST_RANKING_CUH

#include "cuda_utility.cuh"

void CudaSimpleListRank(int *devRank, int N, int *devNext, int *notAllDone, ull *devRankNext, int *devNotAllDone);

#endif // CUDA_LIST_RANKING_CUH
