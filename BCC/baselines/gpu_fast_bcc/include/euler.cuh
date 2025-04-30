#ifndef EULER_TOUR_CUH
#define EULER_TOUR_CUH

#include "cuda_utility.cuh"
#include "bcc_memory_utils.cuh"

void cuda_euler_tour(int N, int root, GPU_BCG& g_bcg_ds);

#endif // EULER_TOUR_CUH