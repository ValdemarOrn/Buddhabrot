
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void init_stuff(curandState* state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(1337, idx, 0, &state[idx]);
}

__global__ void make_rand(curandState *state, float *randArray) 
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	randArray[idx] = curand_uniform(&state[idx]);
}

void host_function()
{
	int nThreads = 100;
	int nBlocks = 1;
	curandState* d_state;
	float* randArray;

	cudaMalloc(&d_state, nThreads * nBlocks);
	cudaMalloc(&randArray, nThreads * nBlocks);

	init_stuff<<<1, 1>>> (d_state);
	make_rand<<<1, 1>>> (d_state, randArray);
	cudaFree(d_state);
}