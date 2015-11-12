#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Cuda.h>
#include <curand.h>
#include <curand_kernel.h>

struct Settings
{
	int Width;
	int Height;
	int Iterations;
	float XMin;
	float XMax;
	float YMin;
	float YMax;
	float NxFactor;
	float NyFactor;
};

__device__ Settings globalSettings;

__device__ void IncreasePixel(Settings* settings, unsigned int* arr, float x, float y)
{
	if (x >= settings->XMax || x < settings->XMin)
		return;
	if (y >= settings->YMax || y < settings->YMin)
		return;

	int nx = (int)((x - settings->XMin) * settings->NxFactor);
	int ny = (int)((y - settings->YMin) * settings->NyFactor);
	int idx = nx + ny * settings->Width;
	atomicAdd(&arr[idx], 1);
}

__global__ void Init(curandState* state)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(idx, 0, 0, &state[idx]);
}

__global__ void SetSettings(Settings* newSettings)
{
	globalSettings = *newSettings;
}

__global__ void RunBuddha(unsigned int* array, curandState *state)
{
	Settings settings = globalSettings;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float x = curand_uniform(&state[idx]) * 2 * (settings.XMax - settings.XMin) + settings.XMin;
	float y = curand_uniform(&state[idx]) * 2 * (settings.YMax - settings.YMin) + settings.YMin;

	float zr = 0.0;
	float zi = 0.0;
	float cr = x;
	float ci = y;

	// check for escape
	for (int i = 0; i < settings.Iterations; i++)
	{
		float zzr = zr * zr - zi * zi;
		float zzi = zr * zi + zi * zr;
		zr = zzr + cr;
		zi = zzi + ci;

		if ((zr * zr + zi * zi) > 4)
			break;
	}

	if ((zr * zr + zi * zi) > 4) // did escape
	{
		zr = 0;
		zi = 0;
		for (int i = 0; i < settings.Iterations; i++)
		{
			float zzr = zr * zr - zi * zi;
			float zzi = zr * zi + zi * zr;
			zr = zzr + cr;
			zi = zzi + ci;

			if ((zr * zr + zi * zi) > 14)
				break;

			IncreasePixel(&settings, array, zr, zi);
			IncreasePixel(&settings, array, zr, -zi);
		}
	}
}
