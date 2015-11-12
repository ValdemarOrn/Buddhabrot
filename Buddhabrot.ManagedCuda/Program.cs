using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace Buddhabrot.ManagedCuda
{
	class Program
	{
		#region Startup Code

		static void Main(string[] args)
		{
			if (args.Length == 0)
				args = "-file C:\\dev\\CudaBuddha\\brotCuda-200000.bmp -w 2160 -h 3840 -it 200000 -xmin -2.0 -xmax 1.1 -samples 50000000".Split();

			var file = Get<string>("file", args);
			var width = Get<int?>("w", args) ?? 1920;
			var height = Get<int?>("h", args) ?? 1080;
			var iter = Get<int?>("it", args) ?? 100;
			var xmin = Get<double?>("xmin", args) ?? -2;
			var xmax = Get<double?>("xmax", args) ?? 2;
			var samples = Get<long?>("samples", args);

			uint[] array = null;
			var brot = new BuddhabrotCuda(width, height, xmin, xmax, iter, samples);
			var process = Task.Run(() => array = brot.Run());

			if (!samples.HasValue)
			{
				Console.WriteLine("Press Enter to stop...");
				Console.ReadLine();
				brot.IsStopping = true;
			}

			process.Wait();
			SaveImage(file, width, height, array);
			Console.WriteLine("Quitting...");
		}

		/// <summary>
		/// Parse command line arguments
		/// </summary>
		private static T Get<T>(string flag, string[] args)
		{
			var value = args.SkipWhile(x => !x.StartsWith("-" + flag)).Skip(1).Take(1).FirstOrDefault();
			if (value == null)
				return default(T);

			if (typeof(T) == typeof(string))
			{
				return (T)(object)value;
			}
			if (typeof(T) == typeof(double?))
			{
				double output;
				var ok = double.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out output);
				return ok ? (T)(object)output : (T)(object)null;
			}
			if (typeof(T) == typeof(int?))
			{
				int output;
				var ok = int.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out output);
				return ok ? (T)(object)output : (T)(object)null;
			}
			if (typeof(T) == typeof(long?))
			{
				long output;
				var ok = long.TryParse(value, NumberStyles.Any, CultureInfo.InvariantCulture, out output);
				return ok ? (T)(object)output : (T)(object)null;
			}
			if (typeof(T) == typeof(bool?))
			{
				var isTrue = value.ToLower() == "true" || value == "1";
				return (T)(object)isTrue;
			}
			return default(T);
		}

		#endregion

		public static void SaveImage(string filename, int width, int height, uint[] array)
		{
			var totalMatrix = array.Select(x => (double)x).ToArray();
			// Used to normalize the intensity of the pixels
			var limit = GetNormalizer(totalMatrix);

			const int ch = 3;
			var imageData = new byte[width * height * ch];
			for (var y = 0; y < height; y++)
			{
				Parallel.For(0, width, x =>
				{
					var val = totalMatrix[x + y * width] / limit * 256;
					if (val > 255)
						val = 255;

					imageData[3 * (x + y * width) + 0] = (byte)(val);
					imageData[3 * (x + y * width) + 1] = (byte)(val);
					imageData[3 * (x + y * width) + 2] = (byte)(val);
				});
			}

			using (var bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
			{
				var bmData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadWrite, bitmap.PixelFormat);
				var pNative = bmData.Scan0;
				Marshal.Copy(imageData, 0, pNative, width * height * ch);
				bitmap.UnlockBits(bmData);
				bitmap.Save(filename);
			}
		}

		private static double GetNormalizer(double[] totalMatrix, double sammpleThreshold = 0.01, double threshold = 0.9995)
		{
			var rand = new Random();
			var sampleCount = (int)(totalMatrix.Length * sammpleThreshold);

			if (sampleCount < 1000)
				sampleCount = 1000;
			if (sampleCount > totalMatrix.Length)
				sampleCount = totalMatrix.Length;

			var samples = Enumerable.Range(0, sampleCount)
				.Select(x => totalMatrix[rand.Next(0, totalMatrix.Length)])
				.OrderBy(x => x)
				.ToArray();

			// pull out the 1% brightest pixels
			var sampleThreshold = samples[(int)(samples.Length * 0.99)];
			var brightSamples = new List<double>();
			for (int i = 0; i < totalMatrix.Length; i++)
			{
				var sample = totalMatrix[i];
				if (sample > sampleThreshold)
					brightSamples.Add(sample);
			}

			var values = brightSamples.OrderBy(x => x).ToArray();
			var k = values.Length - (int)(values.Length * threshold);
			var limit = values[values.Length - k - 1];
			return limit;
		}
	}
}
