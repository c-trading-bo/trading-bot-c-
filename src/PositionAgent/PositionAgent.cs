// Agent: PositionAgent (standalone)
// Role: Standalone position tracking and management.
// Integration: Used for portfolio state and P&L tracking.
using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace PositionAgent
{
	/// <summary>
	/// Tracks open/closed positions and PnL for the account.
	/// </summary>
	public sealed class PositionAgent
	{
		private readonly ILogger<PositionAgent> _log;
		private readonly List<object> _positions = new(); // Replace object with your Position model

		public PositionAgent(ILogger<PositionAgent> log)
		{
			_log = log;
		}

		public void AddPosition(object position)
		{
			_positions.Add(position);
			_log.LogInformation($"Position added: {position}");
		}

		public void ClosePosition(object position)
		{
			_positions.Remove(position);
			_log.LogInformation($"Position closed: {position}");
		}

		public IEnumerable<object> GetOpenPositions() => _positions;
		public void OnOrder(JsonElement je)
		{
			try
			{
				var symbol = je.TryGetProperty("symbol", out var s) && s.ValueKind == JsonValueKind.String ? s.GetString() : "?";
				var sideStr = je.TryGetProperty("side", out var sd) && sd.ValueKind == JsonValueKind.String ? sd.GetString() : "";
				var qty = je.TryGetProperty("qty", out var q) && q.ValueKind == JsonValueKind.Number && q.TryGetInt32(out var qi) ? qi : 0;
				var status = je.TryGetProperty("status", out var st) && st.ValueKind == JsonValueKind.String ? st.GetString() : "";
				_log.LogInformation("[PositionAgent] Order evt: {Symbol} {Side} qty={Qty} status={Status}", symbol, sideStr, qty, status);
			}
			catch { }
		}
	}
}
