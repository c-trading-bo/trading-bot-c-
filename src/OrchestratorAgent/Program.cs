using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore;
using SupervisorAgent;

namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            var loggerFactory = LoggerFactory.Create(b =>
            {
                b.AddConsole();
                b.SetMinimumLevel(LogLevel.Information);
            });
            var log = loggerFactory.CreateLogger("Orchestrator");

            using var http = new HttpClient { BaseAddress = new Uri(Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com") };
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { e.Cancel = true; cts.Cancel(); };

            // Load credentials
            string? jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            string? userName = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            string? apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            long accountId = long.TryParse(Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID"), out var id) ? id : 0L;

            // Try to obtain JWT if not provided
            if (string.IsNullOrWhiteSpace(jwt) && !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
            {
                try
                {
                    var auth = new TopstepAuthAgent(http);
                    log.LogInformation("Fetching JWT using login key for {User}…", userName);
                    jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token);
                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);
                }
                catch (Exception ex)
                {
                    log.LogWarning(ex, "Failed to obtain JWT using TOPSTEPX_USERNAME/TOPSTEPX_API_KEY");
                }
            }

            var status = new StatusService(loggerFactory.CreateLogger<StatusService>()) { AccountId = accountId };

            if (!string.IsNullOrWhiteSpace(jwt) && accountId > 0)
            {
                try
                {
                    var userHub = new BotCore.UserHubAgent(loggerFactory.CreateLogger<BotCore.UserHubAgent>(), status);
                    await userHub.ConnectAsync(jwt!, accountId, cts.Token);
                    log.LogInformation("Bot launched. Press Ctrl+C to exit.");
                    // Keep running until cancelled
                    try { await Task.Delay(Timeout.Infinite, cts.Token); } catch (OperationCanceledException) { }
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    log.LogError(ex, "Unhandled exception while running bot");
                }
            }
            else
            {
                log.LogWarning("Missing TOPSTEPX_JWT and/or TOPSTEPX_ACCOUNT_ID. Set them in .env.local or environment. Process will stay alive for 60 seconds to verify launch.");
                try { await Task.Delay(TimeSpan.FromSeconds(60), cts.Token); } catch (OperationCanceledException) { }
            }
        }
    }
}



