using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using ManagedCuda;
using ManagedCuda.BasicTypes;

namespace LowProfile.Cuda
{
	public class CudaModuleHelper
	{
		private readonly string[] functionNames;

		public CUmodule Module { get; private set; }
		public CudaContext Context { get; private set; }
		public string PtxFile { get; private set; }

		public CudaModuleHelper(CudaContext context, string file)
		{
			Context = context;
			Module = context.LoadModule(file);
			PtxFile = file;
			functionNames = File.ReadAllLines(file)
				.Where(x => x.Contains("// .globl"))
				.Select(x => x.Replace("// .globl", "").Trim())
				.ToArray();
		}

		public CudaKernel GetKernel(string name, bool isStrongName = false)
		{
			if (!isStrongName)
				name = GetStrongName(name);

			var kernel = new CudaKernel(name, Module, Context);
			return kernel;
		}

		private string GetStrongName(string name)
		{
			var matches = functionNames.Where(x => x.Contains(name)).ToArray();
			if (matches.Length == 0)
				throw new Exception("Unable to find a match kernel with name " + name);
			if (matches.Length > 1)
				throw new Exception("Found multiple kernels that match name " + name + " - Please provide Strong Name");

			return matches.Single();
		}
	}
}
