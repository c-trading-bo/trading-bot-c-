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
