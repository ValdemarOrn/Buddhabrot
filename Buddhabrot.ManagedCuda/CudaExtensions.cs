using System.Collections.Generic;
using System.Runtime.InteropServices;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;

namespace LowProfile.Cuda
{
	public static class CudaExtensions
	{
		public static void Launch(this CudaKernel kernel, int gridDimX, int blockDimX, params object[] parameters)
		{
			kernel.GridDimensions = new dim3(gridDimX, 1, 1);
			kernel.BlockDimensions = new dim3(blockDimX, 1, 1);
			kernel.Run(parameters);
		}

		public static CUdeviceptr AllocateMemoryFor<T>(this CudaContext context, T item)
		{
			var size = Marshal.SizeOf(item);
			var ptr = context.AllocateMemory(size);
			return ptr;
		}

		public static CUdeviceptr AllocateMemoryFor<T>(this CudaContext context, T[] item)
		{
			var size = Marshal.SizeOf(item[0]);
			var ptr = context.AllocateMemory(size * item.Length);
			return ptr;
		}

		public static CUdeviceptr AllocateMemoryFor<T>(this CudaContext context, List<T> item)
		{
			var size = Marshal.SizeOf(item[0]);
			var ptr = context.AllocateMemory(size * item.Count);
			return ptr;
		}
	}
}