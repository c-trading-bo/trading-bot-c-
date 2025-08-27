using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

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

            // Minimal hosting: start health endpoint and keep process running.
            try
            {
                var prefix = urls.EndsWith("/") ? urls : urls + "/";

                // Read env
                var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
                var accountIdStr = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
                long.TryParse(accountIdStr, out var accountId);
                var primarySymbol = Environment.GetEnvironmentVariable("PRIMARY_SYMBOL") ?? "ES";

                // Lightweight logger factory
                using var loggerFactory = LoggerFactory.Create(builder =>
                {
                    builder.AddConsole();
                    builder.SetMinimumLevel(LogLevel.Information);
                });
                var log = loggerFactory.CreateLogger("Orchestrator");

                // Wire StatusService and ApiClient for Preflight
                var status = new SupervisorAgent.StatusService(loggerFactory.CreateLogger<SupervisorAgent.StatusService>())
                {
                    AccountId = accountId
                };
                var http = new HttpClient();
                var api = new BotCore.ApiClient(http, loggerFactory.CreateLogger<BotCore.ApiClient>(), apiBase);
                var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (!string.IsNullOrWhiteSpace(jwt)) api.SetJwt(jwt);

                var pf = new OrchestratorAgent.Health.Preflight(api, status, new OrchestratorAgent.Health.Preflight.TradingProfileConfig(), accountId);
                var dst = new OrchestratorAgent.Health.DstGuard("America/Chicago");
                var mode = new OrchestratorAgent.Ops.ModeController(stickyLive: false);

                OrchestratorAgent.Health.HealthzServer.StartWithMode(pf, dst, mode, primarySymbol, prefix);
                Console.WriteLine($"[Orchestrator] Health endpoint ready at {prefix}healthz (mode={ (mode.IsLive ? "LIVE" : "SHADOW") })");

                // Keep running until Ctrl+C
                using var cts = new System.Threading.CancellationTokenSource();
                Console.CancelKeyPress += (s, e) => { e.Cancel = true; cts.Cancel(); };
                try { await Task.Delay(System.Threading.Timeout.Infinite, cts.Token); }
                catch (TaskCanceledException) { }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[Orchestrator] Fatal error: {ex.Message}");
                Environment.ExitCode = 1;
            }
        }
    }
}
