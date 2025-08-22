// Agent: OrchestratorAgent
// Role: Main orchestration logic for strategy execution, agent coordination, and trading flow.
// Integration: Coordinates all other agents and manages bot lifecycle.
using System;

namespace BotCore
{
	/// <summary>
	/// Provides shared orchestration logic or interfaces for the bot.
	/// </summary>
	public static class OrchestratorAgent
	{
		public static void PrintStatus(string message)
		{
			Console.WriteLine($"[Orchestrator] {message}");
		}
	}
}
