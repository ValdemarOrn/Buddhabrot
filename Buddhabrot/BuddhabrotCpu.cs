using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace Buddhabrot
{
	class BuddhabrotCpu
	{
		public volatile bool IsStopping;

		private readonly int threadCount;
		private readonly int width;
		private readonly int iterations;
		private readonly double[][] arrays;
		private readonly double xMin = -2;
		private readonly double xMax = 2;
		private readonly double yMin = -2;
		private readonly double yMax = 2;

		private readonly double nxFactor;
		private readonly double nyFactor;
		private readonly long? maxSamples;
		private long totalSamples;
		private Stopwatch sw;

		public BuddhabrotCpu(int width, int height, double xMin, double xMax, int iterations, long? maxSamples)
		{
			var aspectRatio = width / (double)height;
			threadCount = Environment.ProcessorCount;
			this.width = width;
			this.iterations = iterations;
			this.xMin = xMin;
			this.xMax = xMax;
			this.maxSamples = maxSamples;
			this.totalSamples = 0L;
			
			double xSize = xMax - xMin;
			double ySize = xSize / aspectRatio;
			yMin = -ySize / 2;
			yMax = ySize / 2;

			arrays = new double[threadCount][];
			for (int i = 0; i < threadCount; i++)
			{
				arrays[i] = new double[width * height];
			}

			nxFactor = 1 / xSize * width;
			nyFactor = 1 / ySize * height;
		}

		public double[][] Run()
		{
			sw = Stopwatch.StartNew();
			var tasks = Enumerable
				.Range(0, threadCount)
				.Select(thread => Task.Run(() => Run(arrays[thread], new Random(thread))))
				.ToArray();

			Task.WaitAll(tasks);
			return arrays;
		}

		private void Run(double[] array, Random random)
		{
			long n = 0;
			while (!IsStopping)
			{
				// all points in mandelbrot set are between -2...2
				var x = random.NextDouble() * 2 * (xMax - xMin) + xMin;
				var y = random.NextDouble() * 2 * (yMax - yMin) + yMin;

				var zr = 0.0;
				var zi = 0.0;
				var cr = x;
				var ci = y;

				// check for escape
				for (var i = 0; i < iterations; i++)
				{
					var zzr = zr * zr - zi * zi;
					var zzi = zr * zi + zi * zr;
					zr = zzr + cr;
					zi = zzi + ci;

					if ((zr * zr + zi * zi) > 4)
						break;
				}

				if ((zr * zr + zi * zi) > 4) // did escape
				{
					zr = 0;
					zi = 0;
					for (var i = 0; i < iterations; i++)
					{
						var zzr = zr * zr - zi * zi;
						var zzi = zr * zi + zi * zr;
						zr = zzr + cr;
						zi = zzi + ci;

						if ((zr * zr + zi * zi) > 14)
							break;

						IncreasePixel(array, zr, zi);
						IncreasePixel(array, zr, -zi);
					}
				}

				n++;

				if (n % 10000000 == 0)
				{
					Interlocked.Add(ref totalSamples, n);
					n = 0;
					Console.WriteLine("Calculated {0:0.0} Million samples in {1:0.000} sec", totalSamples / 1000000.0, sw.ElapsedMilliseconds / 1000.0);
					if (maxSamples.HasValue && totalSamples >= maxSamples)
						break;
				}
			}
		}

		private void IncreasePixel(double[] arr, double x, double y)
		{
			if (x >= xMax || x < xMin)
				return;
			if (y >= yMax || y < yMin)
				return;

			var nx = (int)((x - xMin) * nxFactor);
			var ny = (int)((y - yMin) * nyFactor);
			var idx = nx + ny * width;
			arr[idx]++;
		}
	}
}
