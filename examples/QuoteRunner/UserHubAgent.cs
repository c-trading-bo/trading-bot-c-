// Agent: UserHubAgent (example)
// Role: Example implementation of SignalR user hub connection and event handling.
// Integration: Used in QuoteRunner for demo and testing.
using System;
using System.Text.Json;

namespace QuoteRunner
{
	/// <summary>
	/// Handles user hub events and logging for the QuoteRunner example.
	/// </summary>
	public sealed class UserHubAgent
	{
		public void OnOrder(object order)
		{
			Console.WriteLine($"ORDER => {JsonSerializer.Serialize(order)}");
		}

		public void OnTrade(object trade)
		{
			Console.WriteLine($"TRADE => {JsonSerializer.Serialize(trade)}");
		}
	}
}
