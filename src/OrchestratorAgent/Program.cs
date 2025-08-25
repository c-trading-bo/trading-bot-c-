using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore;
using SupervisorAgent;
using Microsoft.AspNetCore.SignalR.Client;
using System.Text.Json;

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

            // Load configuration from environment
            string apiBase = http.BaseAddress!.ToString().TrimEnd('/');
            string rtcBase = (Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com").TrimEnd('/');
            string symbol  = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL") ?? "ES";

            // Load credentials
            string? jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            string? userName = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            string? apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            long accountId = long.TryParse(Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID"), out var id) ? id : 0L;

            log.LogInformation("Env config: API={Api}  RTC={Rtc}  Symbol={Sym}  AccountId={Acc}  HasJWT={HasJwt}  HasLoginKey={HasLogin}", apiBase, rtcBase, symbol, accountId, !string.IsNullOrWhiteSpace(jwt), !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey));

            // Try to obtain JWT if not provided
            if (string.IsNullOrWhiteSpace(jwt) && !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
            {
                try
                {
                    var auth = new TopstepAuthAgent(http);
                    log.LogInformation("Fetching JWT using login key for {User}…", userName);
                    jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token);
                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);
                    log.LogInformation("Obtained JWT via loginKey for {User}.", userName);
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

                    // Wire Market hub for real-time quotes/trades (two contracts)
                    Func<string> getJwt = () => Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? jwt!;
                    var market1 = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), getJwt);
                    var market2 = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), getJwt);
                    using (var m1Cts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token))
                    using (var m2Cts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token))
                    {
                        m1Cts.CancelAfter(TimeSpan.FromSeconds(15));
                        await market1.StartAsync("CON.F.US.EP.U25", m1Cts.Token);
                        m2Cts.CancelAfter(TimeSpan.FromSeconds(15));
                        await market2.StartAsync("CON.F.US.ENQ.U25", m2Cts.Token);
                    }
                    status.Set("market.state", $"{market1.Connection.ConnectionId}|{market2.Connection.ConnectionId}");
                    market1.OnQuote += (_, __) => status.Set("last.quote", DateTimeOffset.UtcNow);
                    market2.OnQuote += (_, __) => status.Set("last.quote", DateTimeOffset.UtcNow);
                    market1.OnTrade += (_, __) => status.Set("last.trade", DateTimeOffset.UtcNow);
                    market2.OnTrade += (_, __) => status.Set("last.trade", DateTimeOffset.UtcNow);
                    market1.OnDepth += (_, __) => status.Set("last.depth", DateTimeOffset.UtcNow);
                    market2.OnDepth += (_, __) => status.Set("last.depth", DateTimeOffset.UtcNow);

                    var quickExit = string.Equals(Environment.GetEnvironmentVariable("BOT_QUICK_EXIT"), "1", StringComparison.Ordinal);
                    log.LogInformation(quickExit ? "Bot launched (quick-exit). Verifying startup then exiting..." : "Bot launched. Press Ctrl+C to exit.");
                    // Keep running until cancelled (or quick short delay when BOT_QUICK_EXIT=1)
                    if (quickExit)
                    {
                        try { await Task.Delay(TimeSpan.FromSeconds(2), cts.Token); } catch (OperationCanceledException) { }
                    }
                    else
                    {
                        try { await Task.Delay(Timeout.Infinite, cts.Token); } catch (OperationCanceledException) { }
                    }
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    log.LogError(ex, "Unhandled exception while running bot");
                }
            }
            else
            {
                var quickExit = string.Equals(Environment.GetEnvironmentVariable("BOT_QUICK_EXIT"), "1", StringComparison.Ordinal);
                if (quickExit)
                {
                    log.LogWarning("Missing TOPSTEPX_JWT and/or TOPSTEPX_ACCOUNT_ID. Quick-exit mode: waiting 2s to verify launch then exiting.");
                    try { await Task.Delay(TimeSpan.FromSeconds(2), cts.Token); } catch (OperationCanceledException) { }
                }
                else
                {
                    log.LogWarning("Missing TOPSTEPX_JWT and/or TOPSTEPX_ACCOUNT_ID. Set them in .env.local or environment. Process will stay alive for 60 seconds to verify launch.");
                    try { await Task.Delay(TimeSpan.FromSeconds(60), cts.Token); } catch (OperationCanceledException) { }
                }
            }
        }
    }
}



