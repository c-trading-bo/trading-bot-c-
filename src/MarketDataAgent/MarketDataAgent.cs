// Agent: MarketDataAgent (standalone)
// Role: Standalone market data streaming and aggregation.
// Integration: Used by strategies and orchestrator for data feeds.
using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using BotCore.Models;

namespace MarketDataAgent
{
	/// <summary>
	/// Handles market data aggregation and bar generation for strategies.
	/// </summary>
	public sealed class MarketDataAgent
	{
		private readonly ILogger<MarketDataAgent> _log;
		private readonly Dictionary<string, List<Bar>> _barsBySymbol = new();
		private readonly Dictionary<string, BarAggregator> _aggregators = new();

		public MarketDataAgent(ILogger<MarketDataAgent> log)
		{
			_log = log;
		}

		public void OnTrade(string symbol, JsonElement tradeJson)
		{
			_log.LogInformation($"Trade event for {symbol}: {tradeJson}");
			var aggregator = GetOrCreateAggregator(symbol);
			aggregator.OnBar += bar =>
			{
				if (!_barsBySymbol.TryGetValue(symbol, out var bars))
					_barsBySymbol[symbol] = bars = new List<Bar>();
				bars.Add(bar);
				_log.LogInformation($"Bar aggregated for {symbol}: O={bar.Open} H={bar.High} L={bar.Low} C={bar.Close} V={bar.Volume}");
			};
			aggregator.OnTrade(tradeJson);
		}

		public void OnQuote(string symbol, JsonElement quoteJson)
		{
			_log.LogInformation($"Quote event for {symbol}: {quoteJson}");
			var aggregator = GetOrCreateAggregator(symbol);
			aggregator.OnBar += bar =>
			{
				if (!_barsBySymbol.TryGetValue(symbol, out var bars))
					_barsBySymbol[symbol] = bars = new List<Bar>();
				bars.Add(bar);
				_log.LogInformation($"Bar aggregated for {symbol}: O={bar.Open} H={bar.High} L={bar.Low} C={bar.Close} V={bar.Volume}");
			};
			aggregator.OnQuote(quoteJson);
		}

		public IReadOnlyList<Bar> GetBars(string symbol)
		{
			return _barsBySymbol.TryGetValue(symbol, out var bars) ? bars : new List<Bar>();
		}

		private BarAggregator GetOrCreateAggregator(string symbol)
		{
			if (!_aggregators.TryGetValue(symbol, out var agg))
			{
				agg = new BarAggregator(60); // 60s bars by default
				_aggregators[symbol] = agg;
			}
			return agg;
		}
	}
}
