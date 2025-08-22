// Agent: PositionAgent
// Role: Tracks and manages open positions, P&L, and position state.
// Integration: Used by orchestrator and risk agents for portfolio management.
using System;
using System.Collections.Generic;
using BotCore.Models;
using Microsoft.Extensions.Logging;

namespace BotCore
{
	/// <summary>
	/// Tracks open/closed positions and PnL for the account.
	/// </summary>
	public sealed class PositionAgent
	{
		private readonly ILogger<PositionAgent> _log;
		private readonly List<Bar> _positions = new(); // Replace Bar with your Position model if available

		public PositionAgent(ILogger<PositionAgent> log)
		{
			_log = log;
		}

		public void AddPosition(Bar position)
		{
			_positions.Add(position);
			_log.LogInformation($"Position added: {position}");
		}

		public void ClosePosition(Bar position)
		{
			_positions.Remove(position);
			_log.LogInformation($"Position closed: {position}");
		}

		public IEnumerable<Bar> GetOpenPositions() => _positions;

		public decimal GetTotalPnL()
		{
			// Example: sum PnL from all positions
			decimal total = 0m;
			foreach (var pos in _positions)
			{
				total += pos.Close - pos.Open;
			}
			return total;
		}
	}
}
