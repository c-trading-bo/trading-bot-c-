using System;
using System.Threading.Tasks;

namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            var urls = Environment.GetEnvironmentVariable("ASPNETCORE_URLS") ?? "http://localhost:5000";
            Console.WriteLine($"[Orchestrator] Starting (urls={urls}) …");

            // Testing/CI helper: exit immediately when requested to avoid hanging without credentials
            var quick = Environment.GetEnvironmentVariable("BOT_QUICK_EXIT");
            if (!string.IsNullOrEmpty(quick) && quick.Trim().Equals("1", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine("[Orchestrator] BOT_QUICK_EXIT=1 → exiting immediately.");
                return;
            }

            // If no credentials are present, avoid long-running network calls and just exit with a clear message.
            var hasAnyCred = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_JWT"))
                          || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME"))
                          || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY"));
            if (!hasAnyCred)
            {
                Console.WriteLine("[Orchestrator] No credentials detected (TOPSTEPX_JWT / TOPSTEPX_USERNAME / TOPSTEPX_API_KEY). Exiting cleanly.");
                await Task.Delay(50);
                return;
            }

            // Placeholder main until full hosting pipeline is wired here.
            // In production, this should initialize orchestrator services and web endpoints.
            Console.WriteLine("[Orchestrator] Credentials detected. Stub build does not start long-running services yet. Exiting.");
            await Task.Delay(100);
        }
    }
}
