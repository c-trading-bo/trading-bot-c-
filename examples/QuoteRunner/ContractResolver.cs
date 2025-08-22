using System;
using System.Collections.Generic;

namespace QuoteRunner
{
	/// <summary>
	/// Resolves contract IDs for trading symbols.
	/// </summary>
	public static class ContractResolver
	{
		private static readonly Dictionary<string, string> _contracts = new()
		{
			{ "ES", "CON.F.US.EP.U25" },
			{ "NQ", "CON.F.US.NQ.U25" }
		};

		public static string? GetContractId(string symbol)
		{
			return _contracts.TryGetValue(symbol, out var id) ? id : null;
		}
	}
}
