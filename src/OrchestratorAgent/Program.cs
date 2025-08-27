using System;
using System.Threading.Tasks;

namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            // Testing/CI helper: exit immediately when requested to avoid hanging without credentials
            var quick = Environment.GetEnvironmentVariable("BOT_QUICK_EXIT");
            if (!string.IsNullOrEmpty(quick) && quick.Trim().Equals("1", StringComparison.OrdinalIgnoreCase))
                return;

            // If no credentials are present, avoid long-running network calls and just exit.
            var hasAnyCred = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_JWT"))
                          || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME"))
                          || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY"));
            if (!hasAnyCred)
            {
                await Task.Delay(10);
                return;
            }

            // Placeholder main until full hosting pipeline is wired here.
            // In production runs, this should initialize orchestrator services and web endpoints.
            await Task.Delay(10);
        }
    }
}
