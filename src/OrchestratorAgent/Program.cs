
using System.Net.Http;
using Microsoft.Extensions.Logging;
using BotCore;                   // ApiClient
using BotCore.Strategy;           // AllStrategies
using OrchestratorAgent;
using TopstepAuthAgent;           // TopstepAuthAgent
using BotCore.Models;
using BotCore.Risk;
using System.Text.Json;
#nullable enable


namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            // Helper for JWT
            string CurrentJwt() => Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? string.Empty;

            // Example handlers (log a few fields to prove flow)
            void WireHandlers(MarketHubClient c, string root)
            {
                c.OnQuote += (cid, json) =>
                {
                    // Optional: inspect json for last/price fields and print
                    if (json.ValueKind == JsonValueKind.Object)
                    {
                        if (json.TryGetProperty("last", out var p) && p.ValueKind == JsonValueKind.Number)
                            Console.WriteLine($"[{root}] Quote last={p}");
                    }
                };
                c.OnTrade += (cid, json) => { /* feed your bar aggregator here */ };
                c.OnDepth += (cid, json) => { /* optional */ };
            }

            var marketClients = new List<MarketHubClient>();
            // Removed legacy candidatesBySymbol logic
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

            string apiBase   = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
            string? userName = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME") ?? Environment.GetEnvironmentVariable("TSX_USERNAME");
            string? apiKey   = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY") ?? Environment.GetEnvironmentVariable("TSX_API_KEY");
            string? jwt      = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            string? acctStr  = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID") ?? Environment.GetEnvironmentVariable("TSX_ACCOUNT_ID") ?? Environment.GetEnvironmentVariable("ACCOUNT_ID");

            int accountId = 0;
            if (string.IsNullOrWhiteSpace(acctStr) || !int.TryParse(acctStr, out accountId) || accountId <= 0)
            {
                Console.Error.WriteLine("ERROR: TOPSTEPX_ACCOUNT_ID missing or invalid. Tried TOPSTEPX_ACCOUNT_ID, TSX_ACCOUNT_ID, ACCOUNT_ID. Set one in your environment.");
                Environment.Exit(1);
            }
            if (string.IsNullOrWhiteSpace(userName) || string.IsNullOrWhiteSpace(apiKey))
            {
                Console.Error.WriteLine("ERROR: TOPSTEPX_USERNAME/TSX_USERNAME and TOPSTEPX_API_KEY/TSX_API_KEY are required. Set them in your environment.");
                Environment.Exit(1);
            }

            using var http = new HttpClient();
            using var apiHttp = new HttpClient();
            using var loggerFactory = LoggerFactory.Create(b => b
                .AddSimpleConsole(o => { o.TimestampFormat = "HH:mm:ss "; o.SingleLine = true; })
                .SetMinimumLevel(LogLevel.Information));

            var log = loggerFactory.CreateLogger("Orchestrator");
            var authLog = loggerFactory.CreateLogger<TopstepAuthAgent.TopstepAuthAgent>();

            // --- AUTH ---
            var auth = new TopstepAuthAgent.TopstepAuthAgent(http, authLog, apiBase);
            if (string.IsNullOrWhiteSpace(jwt))
            {
                log.LogInformation("Logging in with loginKey…");
                jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token);
            }
            else
            {
                log.LogInformation("Validating existing JWT…");
                try { jwt = await auth.ValidateAsync(jwt!, cts.Token); }
                catch { jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token); }
            }
            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);

            // --- EVAL MODE POLICY + PNL ---
            var policy = EvalPolicy.FromEnv();
            var pnl = new PnLTracker(policy);
            await using var userHub = new UserHubAgent(loggerFactory.CreateLogger<UserHubAgent>(), pnl);
            await userHub.ConnectAsync(jwt!, accountId, cts.Token);
            var guard = new EvalGuard(policy, pnl);

            // --- API client ---
            var api = new ApiClient(apiHttp, loggerFactory.CreateLogger<ApiClient>(), apiBase);
            api.SetJwt(jwt!);

            // --- Resolve ES/NQ current contractId ---
            var roots = new[] { "ES", "NQ" };
            var contractIds = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (var r in roots)
            {
                try
                {
                    log.LogInformation("Resolving contract for symbol: {Root}", r);
                    var cid = await api.ResolveContractIdAsync(r, cts.Token);
                    contractIds[r] = cid;
                    log.LogInformation("Resolved {Root} -> {ContractId}", r, cid);
                }
                catch (Exception ex)
                {
                    log.LogWarning("Failed to resolve contract for {Root}: {Msg}", r, ex.Message);
                }
            }

            // Only proceed with the ones we actually resolved:

            var bars = new BotCore.BarsRegistry(maxKeep: 1000);

            void WireSymbol(string root, string contractId)
            {
                var mc  = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), CurrentJwt);
                var agg = new BarAggregator(60);

                var printedSchema = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                void Peek(string tag, JsonElement json)
                {
                    if (printedSchema.Contains(tag)) return;
                    printedSchema.Add(tag);
                    if (json.ValueKind == JsonValueKind.Object)
                    {
                        var keys = string.Join(",", json.EnumerateObject().Select(p => p.Name));
                        Console.WriteLine($"[{root}] {tag} keys: {keys}");
                    }
                    else if (json.ValueKind == JsonValueKind.Array && json.GetArrayLength() > 0 && json[0].ValueKind == JsonValueKind.Object)
                    {
                        var keys = string.Join(",", json[0].EnumerateObject().Select(p => p.Name));
                        Console.WriteLine($"[{root}] {tag}[0] keys: {keys}");
                    }
                }

                mc.OnTrade += (_, json) =>
                {
                    Peek("Trade", json);
                    agg.OnTrade(json);
                };
                agg.OnBar += bar =>
                {
                    bars.Append(root, bar);
                    Console.WriteLine($"[{root}] {bar.Start:HH:mm:ss} Open={bar.Open} High={bar.High} Low={bar.Low} Close={bar.Close} Volume={bar.Volume}");
                    // TODO: run your strategy here (EMA, etc.) once you see bars flowing
                };

                marketClients.Add(mc);
                _ = mc.StartAsync(contractId, cts.Token);
            }

            foreach (var kv in contractIds)
                WireSymbol(kv.Key, kv.Value);

            Console.WriteLine("Market hubs started. Waiting for bars… Press Ctrl+C to exit.");
            await Task.Delay(Timeout.Infinite, cts.Token);

            // (optional) dispose on shutdown
            foreach (var mc in marketClients)
                await mc.DisposeAsync();

            // If none resolved, exit cleanly:
            // Removed legacy strategy and order flow logic. All strategy is now event-driven via BarsRegistry and EmaCrossStrategy.

            // --- 4. Config file check ---
            var configPath = "src/BotCore/Config/high_win_rate_profile.json";
            if (!System.IO.File.Exists(configPath))
            {
                log.LogWarning($"Config file missing: {configPath}");
            }
            else
            {
                log.LogInformation($"Config file found: {configPath}");
            }

            // --- 5. Unit test check ---
            var testPath = "tests/BotTests/AllStrategiesTests.cs";
            if (!System.IO.File.Exists(testPath))
            {
                log.LogWarning($"Unit test file missing: {testPath}");
            }
            else
            {
                log.LogInformation($"Unit test file found: {testPath}");
            }

            log.LogInformation("Eval account orchestration online. Press Ctrl+C to exit.");
            await Task.Delay(Timeout.Infinite, cts.Token);
        }
    }
}
