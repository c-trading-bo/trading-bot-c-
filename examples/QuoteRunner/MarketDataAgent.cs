// Agent: MarketDataAgent (example)
// Role: Example market data streaming and aggregation.
// Integration: Used in QuoteRunner for demo and testing.
using System;
using System.Collections.Generic;
using System.Text.Json;

namespace QuoteRunner
{
	/// <summary>
	/// Handles market data aggregation and bar generation for strategies.
	/// </summary>
	public sealed class MarketDataAgent
	{
		private readonly Dictionary<string, List<JsonElement>> _barsBySymbol = new();

		public void OnTrade(string symbol, JsonElement tradeJson)
		{
			if (!_barsBySymbol.ContainsKey(symbol))
				_barsBySymbol[symbol] = new List<JsonElement>();
			_barsBySymbol[symbol].Add(tradeJson);
		}

		public IReadOnlyList<JsonElement> GetBars(string symbol)
		{
			return _barsBySymbol.TryGetValue(symbol, out var bars) ? bars : new List<JsonElement>();
		}
	}
}
