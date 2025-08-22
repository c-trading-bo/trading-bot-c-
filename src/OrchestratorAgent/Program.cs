
using System.Net.Http;
using Microsoft.Extensions.Logging;
using BotCore;                   // ApiClient
using SupervisorAgent;           // Supervisor integration
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
            // Build status board and supervisor
            var status = new SupervisorAgent.StatusService(loggerFactory.CreateLogger<SupervisorAgent.StatusService>())
            {
                AccountId = accountId,
                Contracts = new() { ["ES"] = contractIds.ContainsKey("ES") ? contractIds["ES"] : "", ["NQ"] = contractIds.ContainsKey("NQ") ? contractIds["NQ"] : "" },
            };

            var supervisor = new SupervisorAgent.SupervisorAgent(
                loggerFactory.CreateLogger<SupervisorAgent.SupervisorAgent>(),
                http, apiBase, jwt!, accountId,
                null, // marketHub placeholder
                userHub,
                status,
                new SupervisorAgent.SupervisorAgent.Config
                {
                    LiveTrading = true,                 // set false to dry-run
                    BarSeconds = 60,                    // 1-minute bars
                    Symbols = new[] { "ES", "NQ" },   // extend as needed
                    UseQuotes = true,                   // subscribe to bid/ask
                    DefaultBracket = new SupervisorAgent.SupervisorAgent.BracketConfig
                    {
                        StopTicks = 12,                 // protective stop (ticks)
                        TargetTicks = 18,               // take-profit (ticks)
                        BreakevenAfterTicks = 8,        // move stop to BE after this
                        TrailTicks = 6                  // trailing after BE
                    }
                }
            );

            await supervisor.RunAsync(cts.Token);
            // Integrate BotSupervisor (runs in parallel, does not block login/auth or SignalR)
            var botSupervisor = new OrchestratorAgent.BotSupervisor(
                loggerFactory.CreateLogger<OrchestratorAgent.BotSupervisor>(),
                http, apiBase, jwt!, accountId,
                null, // marketHub placeholder
                userHub,
                status,
                new OrchestratorAgent.BotSupervisor.Config
                {
                    LiveTrading = true,
                    BarSeconds = 60,
                    Symbols = new[] { "ES", "NQ" },
                    UseQuotes = true,
                    DefaultBracket = new OrchestratorAgent.BotSupervisor.BracketConfig
                    {
                        StopTicks = 12,
                        TargetTicks = 18,
                        BreakevenAfterTicks = 8,
                        TrailTicks = 6
                    }
                }
            );
            _ = botSupervisor.RunAsync(cts.Token); // fire-and-forget, does not block
            // ——— END UPGRADE ———

            var bars = new BotCore.BarsRegistry(maxKeep: 1000);

            void WireSymbol(string root, string contractId)
            {
                var mc  = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), CurrentJwt);
                    var agg = new BotCore.FootprintBarAggregator();

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

                mc.OnTrade += (contractId, json) =>
                {
                    // Parse GatewayTrade payload
                    if (json.ValueKind != System.Text.Json.JsonValueKind.Object) return;
                    if (!json.TryGetProperty("symbolId", out var symbolProp) ||
                        !json.TryGetProperty("price", out var priceProp) ||
                        !json.TryGetProperty("volume", out var volumeProp) ||
                        !json.TryGetProperty("type", out var typeProp) ||
                        !json.TryGetProperty("timestamp", out var tsProp)) return;
                    var symbolId = symbolProp.GetString();
                    var price = priceProp.GetDecimal();
                    var volume = volumeProp.GetDecimal();
                    var sideFlag = typeProp.GetInt32(); // 0=Buy, 1=Sell
                    var iso = tsProp.GetString();
                    var ts = DateTimeOffset.Parse(iso!, System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.AdjustToUniversal);
                    if (BotCore.TradeDeduper.Seen(symbolId!, price, volume, ts)) return;
                    Console.WriteLine($"[{symbolId}] {contractId} {ts:HH:mm:ss.fff} {price} x {volume} {(sideFlag==0 ? "B" : "S")}");
                    agg.OnTrade(contractId, symbolId!, ts, price, volume, sideFlag);
                };
                agg.BarClosed += (contractId, symbolId, bar) =>
                {
                    Console.WriteLine($"[{symbolId}] 1m {bar}");
                    // Convert FootprintBar to Bar
                    var barModel = new BotCore.Models.Bar {
                        Start = bar.OpenTime.UtcDateTime,
                        Ts = bar.OpenTime.ToUnixTimeMilliseconds(),
                        Symbol = symbolId,
                        Open = bar.O,
                        High = bar.H,
                        Low = bar.L,
                        Close = bar.C,
                        Volume = (int)bar.V
                    };

                    // Tick-aware strategy engine and smart order router
                    var strategyAgent = new StrategyAgent.StrategyAgent(new BotCore.Config.TradingProfileConfig());
                    var risk = new BotCore.Risk.RiskEngine();
                    var router = new OrchestratorAgent.OrderRouter(loggerFactory.CreateLogger<OrchestratorAgent.OrderRouter>(), apiHttp, apiBase, jwt!, accountId);

                    var barsList = new List<BotCore.Models.Bar> { barModel };
                    var snapshot = new BotCore.Config.MarketSnapshot { Symbol = symbolId, UtcNow = DateTime.UtcNow };
                    var signals = strategyAgent.RunAll(snapshot, barsList, risk);
                    foreach (var sig in signals)
                    {
                        var side = sig.Side.Equals("BUY", StringComparison.OrdinalIgnoreCase) ? BotCore.SignalSide.Long : BotCore.SignalSide.Short;
                        var normalized = new BotCore.StrategySignal { Strategy = sig.StrategyId, Symbol = symbolId, Side = side, Size = Math.Max(1, sig.Size), LimitPrice = sig.Entry };
                        _ = router.RouteAsync(normalized, contractId, cts.Token);
                    }

                    // Diagnostics per strategy
                    var diag = StrategyDiagnostics.Explain(new BotCore.Config.TradingProfileConfig(), new BotCore.Config.StrategyDef(), snapshot);
                    status.Set("strategies", diag);
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
