using System;
using System.Collections.Generic;

namespace QuoteRunner
{
	/// <summary>
	/// Orchestrates strategy execution and order routing.
	/// </summary>
	public sealed class OrchestratorAgent
	{
		public void RunStrategies(IEnumerable<object> strategies)
		{
			foreach (var strategy in strategies)
			{
				Console.WriteLine($"Running strategy: {strategy}");
				// TODO: Execute strategy logic
			}
		}
	}
}
