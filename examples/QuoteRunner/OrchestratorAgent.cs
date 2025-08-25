// Agent: OrchestratorAgent (example)
// Role: Example orchestration logic for strategy and agent coordination.
// Integration: Used in QuoteRunner for demo and testing.
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
				// Execute strategy logic (implemented below)
				try
				{
					// Prefer the BotCore.IStrategy contract if present
					if (strategy is BotCore.IStrategy s)
					{
						// Minimal, self-contained context for the example runner
						var ctx = new BotCore.StrategyContext(
							accountId: 0,
							contractIds: new Dictionary<string, string>(),
							getBars: _ => Array.Empty<BotCore.Models.Bar>()
						)
						{
							Log = (lvl, strat, msg) => Console.WriteLine($"[{lvl}] {strat}: {msg}")
						};

						// Drive one “bar tick” for a demo symbol
						var symbol = "ES";
						var bars = Array.Empty<BotCore.Models.Bar>(); // supply real bars in your app
						var signals = s.OnBar(symbol, bars, ctx);
						foreach (var sig in signals)
							Console.WriteLine($"Signal => {sig.Strategy} {sig.Symbol} {sig.Side} x{sig.Size} @{sig.LimitPrice}");
					}
					else
					{
						// Fallback: try common method names
						var t = strategy.GetType();
						var run = t.GetMethod("Run") ?? t.GetMethod("Execute");
						run?.Invoke(strategy, null);
					}
				}
				catch (Exception ex)
				{
					Console.Error.WriteLine($"Strategy error: {ex}");
				}
			}
		}
	}
}
