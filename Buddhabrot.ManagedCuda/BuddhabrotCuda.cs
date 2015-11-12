using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LowProfile.Cuda;
using ManagedCuda;

namespace Buddhabrot.ManagedCuda
{
	class BuddhabrotCuda
	{
		private const int SizeOfCurandState = 32 * 4;

		private readonly Settings settings;
		private readonly long? maxSamples;

		public bool IsStopping;
		
		public BuddhabrotCuda(int width, int height, double xMin, double xMax, int iterations, long? maxSamples)
		{
			var aspectRatio = width / (double)height;
			this.maxSamples = maxSamples;

			double xSize = xMax - xMin;
			double ySize = xSize / aspectRatio;
			var yMin = -ySize / 2;
			var yMax = ySize / 2;

			var nxFactor = 1 / xSize * width;
			var nyFactor = 1 / ySize * height;

			settings = new Settings
			{
				Height = height,
				Iterations = iterations,
				NxFactor = (float)nxFactor,
				NyFactor = (float)nyFactor,
				Width = width,
				XMax = (float)xMax,
				XMin = (float)xMin,
				YMax = (float)yMax,
				YMin = (float)yMin
			};
		}

		public uint[] Run()
		{
			var ptx = @"C:\Src\_Tree\SmallPrograms\Buddhabrot\Buddhabrot.Cuda70\x64\Release\Buddhabrot.ptx";

			var context = new CudaContext();
			var module = new CudaModuleHelper(context, ptx);

			var init = module.GetKernel("Init");
			var setSettings = module.GetKernel("SetSettings");
			var runBuddha = module.GetKernel("RunBuddha");

			var nBlocks = 4196;
			var nThreads = 256;

			var dSettings = context.AllocateMemoryFor(settings);
			context.CopyToDevice(dSettings, settings);

			var array = new uint[settings.Width * settings.Height];
			var dState = context.AllocateMemory(nThreads * nBlocks * SizeOfCurandState);
			var dArray = context.AllocateMemoryFor(array);
			context.CopyToDevice(dArray, array);

			init.Launch(nBlocks, nThreads, dState);
			setSettings.Launch(1, 1, dSettings);

			Console.WriteLine("Starting...");
			var sw = Stopwatch.StartNew();
			long i = 0;

			while (!IsStopping)
			{
				runBuddha.Launch(nBlocks, nThreads, dArray, dState);
				
				double count = (++i * nBlocks * nThreads);
				if (i % 5 == 0)
				{
					Console.WriteLine("Generated {0:0.0} Million samples in {1:0.000} sec", count / 1000000.0, sw.ElapsedMilliseconds / 1000.0);
				}

				if (maxSamples.HasValue && count >= maxSamples)
					break;
			}

			context.CopyToHost(array, dArray);
			return array;
		}
	}
}
