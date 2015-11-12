using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Buddhabrot.ManagedCuda
{
	public struct Settings
	{
		public int Width;
		public int Height;
		public int Iterations;
		public float XMin;
		public float XMax;
		public float YMin;
		public float YMax;
		public float NxFactor;
		public float NyFactor;
	};
}
