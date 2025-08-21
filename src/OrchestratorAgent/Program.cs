#nullable enable
using System.Net.Http;
using Microsoft.Extensions.Logging;
using BotCore;                   // ApiClient
using BotCore.Strategy;           // AllStrategies
using OrchestratorAgent;
using TopstepAuthAgent;           // TopstepAuthAgent
using BotCore.Models;
using BotCore.Risk;
using System.Text.Json;

namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            var candidatesBySymbol = new Dictionary<string, List<Candidate>>();
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
            var marketHubs = new Dictionary<string, BotCore.MarketHubClient>();
            var barsBySymbol = new Dictionary<string, List<Bar>>();
            foreach (var kv in contractIds)
            {
                var root = kv.Key;
                var contractId = kv.Value;
                var mhub = new BotCore.MarketHubClient(jwt!);
                await mhub.StartAsync(contractId, cts.Token);
                marketHubs[root] = mhub;
                mhub.OnLastQuote += price => log.LogInformation($"[{root}] LastQuote={price}");
                mhub.OnTrade += (price, vol) => log.LogInformation($"[{root}] Trade {price} x {vol}");

                // --- 1. Live bar fetching ---
                var barsJson = await BotCore.MarketHubClient.FetchHistoricalBarsAsync(jwt!, contractId, DateTime.UtcNow.AddDays(-2), DateTime.UtcNow, cts.Token);
                var bars = new List<Bar>();
                using var doc = JsonDocument.Parse(barsJson);
                if (doc.RootElement.TryGetProperty("bars", out var arr))
                {
                    foreach (var el in arr.EnumerateArray())
                    {
                        bars.Add(new Bar
                        {
                            Ts = el.TryGetProperty("ts", out var ts) ? ts.GetInt64() : 0,
                            Open = el.TryGetProperty("open", out var o) ? o.GetDecimal() : 0m,
                            High = el.TryGetProperty("high", out var h) ? h.GetDecimal() : 0m,
                            Low = el.TryGetProperty("low", out var l) ? l.GetDecimal() : 0m,
                            Close = el.TryGetProperty("close", out var c) ? c.GetDecimal() : 0m,
                            Volume = el.TryGetProperty("volume", out var v) ? v.GetInt32() : 0,
                            Symbol = root
                        });
                    }
                }
                barsBySymbol[root] = bars;
            }

            // If none resolved, exit cleanly:
            if (contractIds.Count == 0)
            {
                log.LogError("No contracts resolved (eval=live:false). Make sure your API key has market data access and retry.");
                return;
            }

            // --- 2. StrategyAgent wiring (using AllStrategies) ---
            foreach (var s in contractIds.Keys)
            {
                var bars = barsBySymbol[s];
                var env = new Env { atr = bars.Count > 0 ? (decimal?)Math.Abs(bars[^1].High - bars[^1].Low) : null, volz = 1.0m };
                var levels = new Levels();
                var risk = new RiskEngine();
                var candidates = AllStrategies.generate_candidates(s, env, levels, bars, risk);
                candidatesBySymbol[s] = candidates;
                foreach (var cand in candidates)
                {
                    log.LogInformation($"Strategy {cand.strategy_id} {cand.symbol} {cand.side} entry={cand.entry} stop={cand.stop} t1={cand.t1} R~{cand.expR}");
                }
            }

            // --- 3. Order flow connection ---
            // --- Find the single best signal across all symbols ---
            Candidate? bestSignal = null;
            string? bestSignalSymbol = null;
            decimal bestR = decimal.MinValue;
            foreach (var s in contractIds.Keys)
            {
                if (!candidatesBySymbol.ContainsKey(s)) continue;
                foreach (var cand in candidatesBySymbol[s])
                {
                    if (guard.CanOpen(s, (int)cand.qty, out var reason))
                    {
                        if (cand.expR > bestR)
                        {
                            bestR = cand.expR;
                            bestSignal = cand;
                            bestSignalSymbol = s;
                        }
                    }
                }
            }
            if (bestSignal != null && bestSignalSymbol != null)
            {
                // TODO: Implement order placement logic here, e.g. call api.PlaceOrderAsync or similar
                log.LogInformation($"Best signal ready for order: {bestSignal.strategy_id} {bestSignal.symbol} {bestSignal.side} entry={bestSignal.entry}");
            }
            else
            {
                log.LogWarning("No suitable signal found for trading.");
            }

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
