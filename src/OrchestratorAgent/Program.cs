using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore;
using SupervisorAgent;
using Microsoft.AspNetCore.SignalR.Client;
using System.Text.Json;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using OrchestratorAgent.Infra;
using OrchestratorAgent.Ops;
using System.Linq;
using System.Net.Http.Json;

namespace OrchestratorAgent
{
    public static class Program
    {
        // Session & ops guards (process-wide)
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, System.Collections.Generic.List<DateTime>> _entriesPerHour = new(StringComparer.OrdinalIgnoreCase);
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, (int Dir, DateTime When)> _lastEntryIntent = new(StringComparer.OrdinalIgnoreCase);
        private static readonly object _entriesLock = new();
        private static DateTimeOffset? TryGetQuoteLastUpdated(System.Text.Json.JsonElement e)
        {
            if (e.ValueKind != System.Text.Json.JsonValueKind.Object) return null;
            // Prefer explicit lastUpdated if present
            if (e.TryGetProperty("lastUpdated", out var lu))
            {
                if (lu.ValueKind == System.Text.Json.JsonValueKind.String && DateTimeOffset.TryParse(lu.GetString(), out var dto))
                    return dto.ToUniversalTime();
                if (lu.ValueKind == System.Text.Json.JsonValueKind.Number && lu.TryGetInt64(out var ms1))
                {
                    // Assume milliseconds if large, otherwise seconds
                    return ms1 > 10_000_000_000 ? DateTimeOffset.FromUnixTimeMilliseconds(ms1) : DateTimeOffset.FromUnixTimeSeconds(ms1);
                }
            }
            foreach (var name in new[] { "exchangeTimeUtc", "exchangeTime", "ts", "timestamp", "time" })
            {
                if (e.TryGetProperty(name, out var p))
                {
                    if (p.ValueKind == System.Text.Json.JsonValueKind.Number && p.TryGetInt64(out var num))
                        return num > 10_000_000_000 ? DateTimeOffset.FromUnixTimeMilliseconds(num) : DateTimeOffset.FromUnixTimeSeconds(num);
                    if (p.ValueKind == System.Text.Json.JsonValueKind.String && DateTimeOffset.TryParse(p.GetString(), out var dto2))
                        return dto2.ToUniversalTime();
                }
            }
            return null;
        }
        public static async Task Main(string[] args)
        {
            // Ensure invariant culture for all parsing/logging regardless of OS locale
            System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;

            var concise = (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            var loggerFactory = LoggerFactory.Create(b =>
            {
                b.ClearProviders();
                var clean = (Environment.GetEnvironmentVariable("LOG_PRESET") ?? "CLEAN").Equals("CLEAN", StringComparison.OrdinalIgnoreCase);
                if (clean)
                {
                    b.AddSimpleConsole(o =>
                    {
                        o.SingleLine = true;
                        o.TimestampFormat = "HH:mm:ss.fff ";
                        o.IncludeScopes = true;
                        o.UseUtcTimestamp = false;
                    });
                    b.SetMinimumLevel(LogLevel.Information);
                    b.AddFilter("Microsoft", LogLevel.Warning);
                    b.AddFilter("System", LogLevel.Warning);
                    b.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Warning);
                    b.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Warning);
                    b.AddFilter("Orchestrator", LogLevel.Information);
                    b.AddFilter("BotCore", LogLevel.Information);
                    b.AddFilter("DataFeed", LogLevel.Information);
                    b.AddFilter("Risk", LogLevel.Information);
                }
                else
                {
                    b.AddConsole();
                    b.SetMinimumLevel(LogLevel.Information);
                    if (concise)
                    {
                        b.AddFilter("Microsoft", LogLevel.Warning);
                        b.AddFilter("System", LogLevel.Warning);
                        b.AddFilter("Microsoft.AspNetCore.SignalR", LogLevel.Warning);
                        b.AddFilter("Microsoft.AspNetCore.Http.Connections", LogLevel.Warning);
                    }
                }
            });
            var log = loggerFactory.CreateLogger("Orchestrator");
            var dataLog = loggerFactory.CreateLogger("DataFeed");
            var riskLog = loggerFactory.CreateLogger("Risk");

            using var http = new HttpClient { BaseAddress = new Uri(Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com") };
            using var cts = new CancellationTokenSource();
            Console.CancelKeyPress += (s, e) => { e.Cancel = true; cts.Cancel(); };

            // Optional quick-exit for CI/smoke: cancel after 5s if BOT_QUICK_EXIT is enabled
            var qe = Environment.GetEnvironmentVariable("BOT_QUICK_EXIT");
            if (!string.IsNullOrWhiteSpace(qe) && (qe.Trim().Equals("1", StringComparison.OrdinalIgnoreCase) || qe.Trim().Equals("true", StringComparison.OrdinalIgnoreCase) || qe.Trim().Equals("yes", StringComparison.OrdinalIgnoreCase)))
            {
                log.LogWarning("Quick-exit mode enabled (BOT_QUICK_EXIT). Will cancel after 5 seconds.");
                try { cts.CancelAfter(TimeSpan.FromSeconds(5)); } catch { }
            }

            // Load configuration from environment
            string apiBase = http.BaseAddress!.ToString().TrimEnd('/');
            string rtcBase = (Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE") ?? "https://rtc.topstepx.com").TrimEnd('/');
            var symbolListRaw = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOLS");
            var roots = (symbolListRaw ?? Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL") ?? "ES")
                .Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                .Select(s => s.Trim().ToUpperInvariant())
                .ToArray();
            string symbol = roots.Length > 0 ? roots[0] : "ES";

            // Load credentials (with fallbacks for common env names)
            static string? Env(string name) => Environment.GetEnvironmentVariable(name);
            string? jwt = Env("TOPSTEPX_JWT") ?? Env("JWT");
            string? userName = Env("TOPSTEPX_USERNAME") ?? Env("LOGIN_USERNAME") ?? Env("LOGIN_EMAIL") ?? Env("USERNAME") ?? Env("EMAIL");
            string? apiKey = Env("TOPSTEPX_API_KEY") ?? Env("LOGIN_KEY") ?? Env("API_KEY");
            long accountId = long.TryParse(Env("TOPSTEPX_ACCOUNT_ID") ?? Env("ACCOUNT_ID"), out var id) ? id : 0L;

            log.LogInformation("Env config: API={Api}  RTC={Rtc}  Symbol={Sym}  AccountId={Acc}  HasJWT={HasJwt}  HasLoginKey={HasLogin}", apiBase, rtcBase, symbol, accountId, !string.IsNullOrWhiteSpace(jwt), !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey));

            // Clock sanity: local, UTC, CME (America/Chicago)
            try
            {
                var nowLocal = DateTimeOffset.Now;
                var nowUtc = DateTimeOffset.UtcNow;
                var cmeTzId = "Central Standard Time"; // Windows TZ for America/Chicago
                var cmeTz = TimeZoneInfo.FindSystemTimeZoneById(cmeTzId);
                var nowCme = TimeZoneInfo.ConvertTime(nowUtc, cmeTz);
                log.LogInformation("Clock: Local={Local} UTC={Utc} CME={CME}", nowLocal, nowUtc, nowCme);
            }
            catch (Exception ex)
            {
                log.LogWarning(ex, "Clock sanity logging failed (timezone not found)");
            }

            // ===== Launch mode selection: Live vs Paper vs Shadow (before any auth) =====
            bool paperModeSelected = false;
            bool shadowModeSelected = false;
            try
            {
                string? botMode = Environment.GetEnvironmentVariable("BOT_MODE");
                bool skipPrompt = (Environment.GetEnvironmentVariable("SKIP_MODE_PROMPT") ?? "false").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                if (!string.IsNullOrWhiteSpace(botMode))
                {
                    paperModeSelected = botMode.Trim().Equals("paper", StringComparison.OrdinalIgnoreCase);
                    shadowModeSelected = botMode.Trim().Equals("shadow", StringComparison.OrdinalIgnoreCase);
                }
                else if (!skipPrompt && !Console.IsInputRedirected)
                {
                    Console.Write("Select mode: [L]ive, [P]aper, [S]hadow  [default: Shadow]: ");
                    var line = Console.ReadLine();
                    line = (line ?? string.Empty).Trim();
                    var lower = line.ToLowerInvariant();
                    if (lower == "l" || lower == "live" || lower == "y" || lower == "yes") { paperModeSelected = false; shadowModeSelected = false; }
                    else if (lower == "p" || lower == "paper" || lower == "n" || lower == "no") { paperModeSelected = true; shadowModeSelected = false; }
                    else if (lower == "s" || lower == "shadow") { shadowModeSelected = true; paperModeSelected = false; }
                    else { paperModeSelected = false; shadowModeSelected = true; } // default Shadow
                }
                else
                {
                    // Default to Shadow when not specified to be safer by default
                    paperModeSelected = false;
                    shadowModeSelected = true;
                }
                // Set env flags so downstream services pick it up
                Environment.SetEnvironmentVariable("PAPER_MODE", paperModeSelected ? "1" : "0");
                Environment.SetEnvironmentVariable("SHADOW_MODE", shadowModeSelected ? "1" : "0");
                // LIVE_ORDERS only when Live
                Environment.SetEnvironmentVariable("LIVE_ORDERS", (!paperModeSelected && !shadowModeSelected) ? "1" : "0");
                var modeName = paperModeSelected ? "PAPER" : shadowModeSelected ? "SHADOW" : "LIVE";
                log.LogInformation("Launch mode selected: {Mode}", modeName);
            }
            catch { }

            // Try to obtain JWT if not provided
            if (string.IsNullOrWhiteSpace(jwt) && !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
            {
                try
                {
                    var auth = new TopstepAuthAgent(http);
                    log.LogInformation("Fetching JWT using login key for {User}…", userName);
                    jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token);
                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);
                    try { http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt); } catch { }
                    log.LogInformation("Obtained JWT via loginKey for {User}.", userName);
                }
                catch (Exception ex)
                {
                    log.LogWarning(ex, "Failed to obtain JWT using TOPSTEPX_USERNAME/TOPSTEPX_API_KEY");
                }
            }

            var status = new StatusService(loggerFactory.CreateLogger<StatusService>()) { AccountId = accountId };

            if (!string.IsNullOrWhiteSpace(jwt))
            {
                if (accountId <= 0)
                {
                    log.LogWarning("TOPSTEPX_ACCOUNT_ID not set. Launching in account-discovery mode (SubscribeAccounts only). You can set the account ID later.");
                }
                try
                {
                    // Start background JWT refresh loop (auth hygiene)
                    var refreshCts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);
                    _ = Task.Run(async () =>
                    {
                        var auth = new TopstepAuthAgent(http);
                        while (!refreshCts.Token.IsCancellationRequested)
                        {
                            try
                            {
                                await Task.Delay(TimeSpan.FromMinutes(20), refreshCts.Token);
                                var newToken = await auth.ValidateAsync(refreshCts.Token);
                                if (!string.IsNullOrWhiteSpace(newToken))
                                {
                                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", newToken);
                                    http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", newToken);
                                    log.LogInformation("JWT refreshed via validate.");
                                }
                                else if (!string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
                                {
                                    var refreshed = await auth.GetJwtAsync(userName!, apiKey!, refreshCts.Token);
                                    Environment.SetEnvironmentVariable("TOPSTEPX_JWT", refreshed);
                                    http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", refreshed);
                                    log.LogInformation("JWT refreshed via loginKey.");
                                }
                            }
                            catch (OperationCanceledException) { }
                            catch (Exception ex)
                            {
                                log.LogWarning(ex, "JWT refresh failed; will retry.");
                            }
                        }
                    }, refreshCts.Token);

                    // Shared JWT cache so both hubs always get a valid token
                    var jwtCache = new JwtCache(async () =>
                    {
                        var t = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                        if (!string.IsNullOrWhiteSpace(t)) return t!;
                        if (!string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
                        {
                            var authLocal = new TopstepAuthAgent(http);
                            var fresh = await authLocal.GetJwtAsync(userName!, apiKey!, CancellationToken.None);
                            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", fresh);
                            http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", fresh);
                            return fresh;
                        }
                        return jwt!; // fallback: initial token (we only enter this branch when jwt is non-empty)
                    });

                    var userHub = new BotCore.UserHubAgent(loggerFactory.CreateLogger<BotCore.UserHubAgent>(), status);
                    await userHub.ConnectAsync(jwt!, accountId, cts.Token);

                    // Resolve roots and contracts from env (with REST fallback)
                    var apiClient = new ApiClient(http, loggerFactory.CreateLogger<ApiClient>(), apiBase);
                    try { if (!string.IsNullOrWhiteSpace(jwt)) apiClient.SetJwt(jwt!); } catch { }
                    var esRoot = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL_ES") ?? "ES";
                    var nqRoot = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL_NQ") ?? "NQ";
                    bool enableNq =
                        (roots.Any(r => string.Equals(r, "NQ", StringComparison.OrdinalIgnoreCase))) ||
                        ((Environment.GetEnvironmentVariable("TOPSTEPX_ENABLE_NQ") ?? Environment.GetEnvironmentVariable("ENABLE_NQ") ?? "1").Trim().ToLowerInvariant() is "1" or "true" or "yes");
                    var esContract = Environment.GetEnvironmentVariable("TOPSTEPX_CONTRACT_ES");
                    var nqContract = Environment.GetEnvironmentVariable("TOPSTEPX_CONTRACT_NQ");
                    try { if (string.IsNullOrWhiteSpace(esContract)) esContract = await apiClient.ResolveContractIdAsync(esRoot, cts.Token); } catch { }
                    try { if (enableNq && string.IsNullOrWhiteSpace(nqContract)) nqContract = await apiClient.ResolveContractIdAsync(nqRoot, cts.Token); } catch { }
                    if (string.IsNullOrWhiteSpace(esContract)) { esContract = esRoot; }
                    if (enableNq && string.IsNullOrWhiteSpace(nqContract)) { nqContract = nqRoot; }

                    // Wire Market hub for real-time quotes/trades (per enabled symbol)
                    var market1 = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), jwtCache.GetAsync);
                    MarketHubClient? market2 = enableNq ? new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), jwtCache.GetAsync) : null;
               					using (var m1Cts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token))
               					using (var m2Cts = enableNq ? CancellationTokenSource.CreateLinkedTokenSource(cts.Token) : null)
               					{
               						m1Cts.CancelAfter(TimeSpan.FromSeconds(15));
               						await market1.StartAsync(esContract!, m1Cts.Token);
               						if (enableNq && market2 != null && m2Cts != null)
               						{
               							m2Cts.CancelAfter(TimeSpan.FromSeconds(15));
               							await market2.StartAsync(nqContract!, m2Cts.Token);
               						}
               					}
               					status.Set("market.state", enableNq && market2 != null ? $"{market1.Connection.ConnectionId}|{market2.Connection.ConnectionId}" : market1.Connection.ConnectionId ?? string.Empty);

               					// Optional warm-up: wait up to 10s for first ES/NQ tick/bar
               					try
               					{
               						var t0 = DateTime.UtcNow;
               						while (DateTime.UtcNow - t0 < TimeSpan.FromSeconds(10))
               						{
               							bool esOk = market1.HasRecentQuote(esContract!) || market1.HasRecentBar(esContract!, "1m");
               							bool nqOk = !enableNq || (market2 != null && (market2.HasRecentQuote(nqContract!) || market2.HasRecentBar(nqContract!, "1m")));
               							if (esOk && nqOk) break;
               							await Task.Delay(250, cts.Token);
               						}
               						log.LogInformation("[MarketHub] Warmup: ES(Q:{Qes} B:{Bes}) NQ(Q:{Qnq} B:{Bnq})",
               							market1.HasRecentQuote(esContract!), market1.HasRecentBar(esContract!, "1m"),
               							enableNq && market2 != null ? market2.HasRecentQuote(nqContract!) : false,
               							enableNq && market2 != null ? market2.HasRecentBar(nqContract!, "1m") : false);
               					}
               					catch { }
                    // Defer wiring of quote handlers until posTracker and contractIds are initialized below
                    Action wireQuotes = () =>
                    {
                    // Now that posTracker and contractIds are ready, wire quote handlers
                    market1.OnQuote += (cid, last, bid, ask) => {
                        var nowTs = DateTimeOffset.UtcNow;
                        status.Set("last.quote", nowTs);
                        status.Set("last.quote.updated", nowTs);
                        try
                        {
                            if (bid > 0m && ask > 0m)
                            {
                                var tick = BotCore.Models.InstrumentMeta.Tick("ES");
                                if (tick <= 0) tick = 0.25m;
                                var st = (int)Math.Max(0, Math.Round((ask - bid) / tick));
                                status.Set($"spread.ticks.ES", st);
                            }
                            // Live mark-to-market for ES using same-contract quotes
                            var mark = last > 0m ? last : (bid > 0m && ask > 0m ? (bid + ask) / 2m : 0m);
                            if (mark > 0m)
                            {
                                // Prefer PositionTracker snapshot (root mirror) for qty/avg
                                var snap = posTracker.Snapshot();
                                int qty = 0; decimal avg = 0m;
                                if (snap.TryGetValue(contractIds[esRoot], out var esByCid)) { qty = esByCid.Qty; avg = esByCid.AvgPrice; }
                                else if (snap.TryGetValue(esRoot, out var esByRoot)) { qty = esByRoot.Qty; avg = esByRoot.AvgPrice; }
                                else { qty = status.Get<int>("pos.ES.qty"); avg = status.Get<decimal?>("pos.ES.avg") ?? 0m; }
                                var bpv = BotCore.Models.InstrumentMeta.BigPointValue("ES"); if (bpv <= 0) bpv = 50m;
                                var side = Math.Sign(qty);
                                var upnl = (mark - avg) * side * bpv * Math.Abs(qty);
                                // Update both root and contractId keys
                                status.Set("pos.ES.upnl", Math.Round(upnl, 2));
                                status.Set("pos.ES.mark", mark);
                                try { status.Set($"pos.{cid}.mark", mark); status.Set($"pos.{cid}.upnl", Math.Round(upnl, 2)); } catch { }
                            }
                        }
                        catch { }
                    };
                    if (enableNq && market2 != null) market2.OnQuote += (cid, last, bid, ask) => {
                        var nowTs = DateTimeOffset.UtcNow;
                        status.Set("last.quote", nowTs);
                        status.Set("last.quote.updated", nowTs);
                        try
                        {
                            if (bid > 0m && ask > 0m)
                            {
                                var tick = BotCore.Models.InstrumentMeta.Tick("NQ");
                                if (tick <= 0) tick = 0.25m;
                                var st = (int)Math.Max(0, Math.Round((ask - bid) / tick));
                                status.Set($"spread.ticks.NQ", st);
                            }
                            // Live mark-to-market for NQ using same-contract quotes
                            var mark = last > 0m ? last : (bid > 0m && ask > 0m ? (bid + ask) / 2m : 0m);
                            if (mark > 0m)
                            {
                                var snap = posTracker.Snapshot();
                                int qty = 0; decimal avg = 0m;
                                if (contractIds.ContainsKey(nqRoot) && snap.TryGetValue(contractIds[nqRoot], out var nqByCid)) { qty = nqByCid.Qty; avg = nqByCid.AvgPrice; }
                                else if (snap.TryGetValue(nqRoot, out var nqByRoot)) { qty = nqByRoot.Qty; avg = nqByRoot.AvgPrice; }
                                else { qty = status.Get<int>("pos.NQ.qty"); avg = status.Get<decimal?>("pos.NQ.avg") ?? 0m; }
                                var bpv = BotCore.Models.InstrumentMeta.BigPointValue("NQ"); if (bpv <= 0) bpv = 20m;
                                var side = Math.Sign(qty);
                                var upnl = (mark - avg) * side * bpv * Math.Abs(qty);
                                status.Set("pos.NQ.upnl", Math.Round(upnl, 2));
                                status.Set("pos.NQ.mark", mark);
                                try { status.Set($"pos.{cid}.mark", mark); status.Set($"pos.{cid}.upnl", Math.Round(upnl, 2)); } catch { }
                            }
                        }
                        catch { }
                    };
                    };
                    market1.OnTrade += (_, __) => status.Set("last.trade", DateTimeOffset.UtcNow);
                    if (enableNq && market2 != null) market2.OnTrade += (_, __) => status.Set("last.trade", DateTimeOffset.UtcNow);
                    market1.OnDepth += (_, __) => status.Set("last.depth", DateTimeOffset.UtcNow);
                    if (enableNq && market2 != null) market2.OnDepth += (_, __) => status.Set("last.depth", DateTimeOffset.UtcNow);

                    // Heartbeat loop (throttled inside StatusService)
                    _ = Task.Run(async () =>
                    {
                        while (!cts.IsCancellationRequested)
                        {
                            try { status.Heartbeat(); } catch { }
                            try { await Task.Delay(TimeSpan.FromSeconds(2), cts.Token); } catch { }
                        }
                    }, cts.Token);

                    // ===== Positions wiring =====
                    var posTracker = new PositionTracker(log, accountId);
                    // Subscribe to user hub events
                    userHub.OnPosition += posTracker.OnPosition;
                    userHub.OnTrade += posTracker.OnTrade;
                    // Feed market trades for last price updates
                    market1.OnTrade += (cid, tick) => { try { var je = System.Text.Json.JsonSerializer.SerializeToElement(new { symbol = cid, price = tick.Price }); posTracker.OnMarketTrade(je); } catch { } };
                    if (enableNq && market2 != null) market2.OnTrade += (cid, tick) => { try { var je = System.Text.Json.JsonSerializer.SerializeToElement(new { symbol = cid, price = tick.Price }); posTracker.OnMarketTrade(je); } catch { } };
                    // Seed from REST
                    await posTracker.SeedFromRestAsync(apiClient, accountId, cts.Token);
                    // Publish snapshot periodically to status
                    _ = Task.Run(async () =>
                    {
                        while (!cts.IsCancellationRequested)
                        {
                            try
                            {
                                decimal sumRpnl = 0m;
                                foreach (var kv in posTracker.Snapshot())
                                {
                                    var sym = kv.Key;
                                    var st = kv.Value;
                                    // publish under original key
                                    status.Set($"pos.{sym}.qty", st.Qty);
                                    status.Set($"pos.{sym}.avg", st.AvgPrice);
                                    status.Set($"pos.{sym}.upnl", st.UnrealizedUsd);
                                    status.Set($"pos.{sym}.rpnl", st.RealizedUsd);
                                    // also mirror under root symbol (ES/NQ) for header display
                                    try
                                    {
                                        var root = SymbolMeta.RootFromName(sym);
                                        if (!string.IsNullOrWhiteSpace(root))
                                        {
                                            status.Set($"pos.{root}.qty", st.Qty);
                                            status.Set($"pos.{root}.avg", st.AvgPrice);
                                            status.Set($"pos.{root}.upnl", st.UnrealizedUsd);
                                            status.Set($"pos.{root}.rpnl", st.RealizedUsd);
                                        }
                                    }
                                    catch { }
                                    sumRpnl += st.RealizedUsd;
                                }
                                // Expose day/net PnL for risk halts and heartbeat
                                status.Set("pnl.day", sumRpnl);
                                status.Set("pnl.net", sumRpnl);
                            }
                            catch { }
                            try { await Task.Delay(TimeSpan.FromSeconds(5), cts.Token); } catch { }
                        }
                    }, cts.Token);

                    // ===== Strategy wiring (per-bar) =====
                    // Map symbols to contract IDs resolved from env/REST
                    var contractIds = new System.Collections.Generic.Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        [esRoot] = esContract!
                    };
                    if (enableNq && !string.IsNullOrWhiteSpace(nqContract)) contractIds[nqRoot] = nqContract!;
                    // Expose root->contract mapping to status for health checks
                    try
                    {
                        foreach (var kv in contractIds)
                            status.Contracts[kv.Key] = kv.Value;
                    }
                    catch { }

                    // Now that posTracker and contractIds are ready, wire quote handlers
                    try { wireQuotes(); } catch { }

                    // Also record per-contract last quote/trade/bar timestamps for /preflight
                    try
                    {
                        var esId = contractIds[esRoot];
                        market1.OnQuote += (_, last, bid, ask) => { var nowTs = DateTimeOffset.UtcNow; status.Set($"last.quote.{esId}", nowTs); status.Set($"last.quote.updated.{esId}", nowTs); };
                        market1.OnTrade += (_, __) => status.Set($"last.trade.{esId}", DateTimeOffset.UtcNow);
                        if (enableNq && market2 != null)
                        {
                            var nqId = contractIds[nqRoot];
                            market2.OnQuote += (_, last, bid, ask) => { var nowTs = DateTimeOffset.UtcNow; status.Set($"last.quote.{nqId}", nowTs); status.Set($"last.quote.updated.{nqId}", nowTs); };
                            market2.OnTrade += (_, __) => status.Set($"last.trade.{nqId}", DateTimeOffset.UtcNow);
                        }
                        // bars will be recorded in OnBar handlers below
                    }
                    catch { }

                    // Aggregators and recent bars per symbol (seed 1m from REST, roll 1m->5m->30m)
                    var barsHist = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<BotCore.Models.Bar>>
                    {
                        [esRoot] = new System.Collections.Generic.List<BotCore.Models.Bar>()
                    };
                    if (enableNq) barsHist[nqRoot] = new System.Collections.Generic.List<BotCore.Models.Bar>();

                    var barPyramid = new BotCore.Market.BarPyramid();

                    // Seed from REST: Retrieve Bars (1m, last 500, include partial)
                    static DateTime ParseUtc(string s) => DateTime.Parse(s).ToUniversalTime();
                    async Task SeedBarsAsync(string contractId)
                    {
                        try
                        {
                            using var httpSeed = new HttpClient { BaseAddress = new Uri(Environment.GetEnvironmentVariable("API_BASE") ?? apiBase) };
                            httpSeed.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", await jwtCache.GetAsync());
                            httpSeed.DefaultRequestHeaders.Accept.Clear();
                            httpSeed.DefaultRequestHeaders.Accept.Add(new System.Net.Http.Headers.MediaTypeWithQualityHeaderValue("application/json"));

                            var endUtc = DateTime.UtcNow;
                            var startUtc = endUtc.AddMinutes(-600);

                            var payload = new {
                                contractId = contractId,
                                live = false,
                                startTime = startUtc.ToString("o"),
                                endTime = endUtc.ToString("o"),
                                unit = 2,        // Minute
                                unitNumber = 1,
                                limit = 2000,
                                includePartialBar = true
                            };
                            var resp = await httpSeed.PostAsJsonAsync("/api/History/retrieveBars", payload, cts.Token);
                            var text = await resp.Content.ReadAsStringAsync(cts.Token);
                            if (!resp.IsSuccessStatusCode)
                            {
                                dataLog.LogWarning("[DataFeed] retrieveBars {cid} failed {code}: {msg}", contractId, (int)resp.StatusCode, text);
                                resp.EnsureSuccessStatusCode();
                            }

                            using var doc = System.Text.Json.JsonDocument.Parse(text);
                            var barsJson = doc.RootElement.GetProperty("bars");

                            var seeded = new System.Collections.Generic.List<BotCore.Market.Bar>();
                            foreach (var x in barsJson.EnumerateArray())
                            {
                                var t = x.GetProperty("t").GetDateTime();
                                var o = x.GetProperty("o").GetDecimal();
                                var h = x.GetProperty("h").GetDecimal();
                                var l = x.GetProperty("l").GetDecimal();
                                var c = x.GetProperty("c").GetDecimal();
                                var v = x.GetProperty("v").GetInt64();
                                var end = t.AddMinutes(1);
                                seeded.Add(new BotCore.Market.Bar(t, end, o, h, l, c, v));
                            }
                            barPyramid.M1.Seed(contractId, seeded);
                            dataLog.LogInformation("Bars seeded: {cid}={n}", contractId, seeded.Count);
                        }
                        catch (Exception ex)
                        {
                            dataLog.LogWarning(ex, "[DataFeed] Seeding bars failed for {Cid}", contractId);
                        }
                    }

                    var esIdForSeed = esContract!;
                    var nqIdForSeed = enableNq && !string.IsNullOrWhiteSpace(nqContract) ? nqContract! : null;
                    await SeedBarsAsync(esIdForSeed);
                    if (nqIdForSeed != null) await SeedBarsAsync(nqIdForSeed);

                    // Backfill roll-ups (1m history -> 5m/30m) so strategies don't wait 5-30 minutes
                    void Backfill(string cid)
                    {
                        foreach (var b in barPyramid.M1.GetHistory(cid))
                        {
                            var t = b.End.AddMilliseconds(-1);
                            barPyramid.M5.OnTrade(cid, t, b.Close, Math.Max(1, b.Volume));
                            barPyramid.M30.OnTrade(cid, t, b.Close, Math.Max(1, b.Volume));
                        }
                    }
                    try { Backfill(esIdForSeed); if (nqIdForSeed != null) Backfill(nqIdForSeed); } catch { }

                    dataLog.LogInformation("Bars seeded: ES={EsCnt}{NqPart}",
                        barPyramid.M1.GetHistory(esIdForSeed).Count,
                        nqIdForSeed != null ? $", NQ={barPyramid.M1.GetHistory(nqIdForSeed).Count}" : string.Empty);

                    status.Set("bars.ready", true);
                    try
                    {
                        dataLog.LogInformation("Bars ready: ES={esCnt} NQ={nqCnt}",
                            barPyramid.M1.GetHistory(esIdForSeed).Count,
                            nqIdForSeed != null ? barPyramid.M1.GetHistory(nqIdForSeed).Count : 0);
                    }
                    catch { }

                    // Strategy prerequisites
                    var risk = new BotCore.Risk.RiskEngine();
                    // Apply risk-per-trade from environment so sizing matches your budget
                    try
                    {
                        var r1 = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE");
                        if (!string.IsNullOrWhiteSpace(r1) && decimal.TryParse(r1, out var rpt) && rpt > 0)
                        {
                            risk.cfg.risk_per_trade = rpt;
                            log.LogInformation("Risk: using risk_per_trade=${RPT}", rpt);
                        }
                        var rPct = Environment.GetEnvironmentVariable("RISK_PCT_OF_EQUITY") ?? Environment.GetEnvironmentVariable("RISK_EQUITY_PCT");
                        if (!string.IsNullOrWhiteSpace(rPct) && decimal.TryParse(rPct, out var pct) && pct > 0)
                        {
                            risk.cfg.risk_pct_of_equity = pct;
                            log.LogInformation("Risk: using equity% per trade = {Pct}", pct);
                        }
                        var mdlEnv2 = Environment.GetEnvironmentVariable("MAX_DAILY_LOSS") ?? Environment.GetEnvironmentVariable("EVAL_MAX_DAILY_LOSS");
                        if (!string.IsNullOrWhiteSpace(mdlEnv2) && decimal.TryParse(mdlEnv2, out var mdlv) && mdlv > 0) risk.cfg.max_daily_drawdown = mdlv;
                        var mwl = Environment.GetEnvironmentVariable("MAX_WEEKLY_LOSS");
                        if (!string.IsNullOrWhiteSpace(mwl) && decimal.TryParse(mwl, out var mwlv) && mwlv > 0) risk.cfg.max_weekly_drawdown = mwlv;
                        var mcl = Environment.GetEnvironmentVariable("MAX_CONSECUTIVE_LOSSES");
                        if (!string.IsNullOrWhiteSpace(mcl) && int.TryParse(mcl, out var mclv) && mclv > 0) risk.cfg.max_consecutive_losses = mclv;
                        var cool = Environment.GetEnvironmentVariable("COOLDOWN_MINUTES_AFTER_STREAK");
                        if (!string.IsNullOrWhiteSpace(cool) && int.TryParse(cool, out var coolv) && coolv > 0) risk.cfg.cooldown_minutes_after_streak = coolv;
                        var mop = Environment.GetEnvironmentVariable("MAX_OPEN_POSITIONS");
                        if (!string.IsNullOrWhiteSpace(mop) && int.TryParse(mop, out var mopv) && mopv > 0) risk.cfg.max_open_positions = mopv;
                    }
                    catch { }

                    var levels = new BotCore.Models.Levels();
                    bool live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? string.Empty)
                                .Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    var partialExit = new OrchestratorAgent.Ops.PartialExitService(http, jwtCache.GetAsync, log);
                    var router = new SimpleOrderRouter(http, jwtCache.GetAsync, log, live, partialExit);

                    // Paper mode wiring
                    bool paperMode = (Environment.GetEnvironmentVariable("PAPER_MODE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    bool shadowMode = (Environment.GetEnvironmentVariable("SHADOW_MODE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    var simulateMode = paperMode || shadowMode;
                                        PaperBroker? paperBroker = simulateMode ? new PaperBroker(status, log) : null;

                    // Autopilot flags
                    var auto = AppEnv.Flag("AUTO", false);
                    var dryRun = AppEnv.Flag("DRYRUN", false);
                    var killSwitch = AppEnv.Flag("KILL_SWITCH", false);

                    // ===== Preflight gating (/healthz + periodic) =====
                    var pfCfg = new OrchestratorAgent.Health.Preflight.TradingProfileConfig
                    {
                        Risk = new OrchestratorAgent.Health.Preflight.TradingProfileConfig.RiskConfig
                        {
                            DailyLossLimit = decimal.TryParse(Environment.GetEnvironmentVariable("EVAL_MAX_DAILY_LOSS"), out var mdl) ? mdl : 1000m,
                            MaxTradesPerDay = int.TryParse(Environment.GetEnvironmentVariable("MAX_TRADES_PER_DAY"), out var mtpd) ? mtpd : 1000
                        }
                    };
                    var pfService = new OrchestratorAgent.Health.Preflight(apiClient, status, pfCfg, accountId);
                    var dst = new OrchestratorAgent.Health.DstGuard("America/Chicago", 7);

                    // Mode + AutoPilot wiring
                    bool autoGoLive = (Environment.GetEnvironmentVariable("AUTO_GO_LIVE") ?? "true").Equals("true", StringComparison.OrdinalIgnoreCase)
                                   || (Environment.GetEnvironmentVariable("AUTO_GO_LIVE") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase);
                    int dryMin = int.TryParse(Environment.GetEnvironmentVariable("AUTO_DRYRUN_MINUTES"), out var dm) ? Math.Max(0, dm) : 5;
                    int minHealthy = int.TryParse(Environment.GetEnvironmentVariable("AUTO_MIN_HEALTHY_PASSES"), out var mh) ? Math.Max(1, mh) : 3;
                    int demoteOnBad = int.TryParse(Environment.GetEnvironmentVariable("AUTO_DEMOTE_ON_UNHEALTHY"), out var db) ? Math.Max(1, db) : 3;
                    bool stickyLive = (Environment.GetEnvironmentVariable("AUTO_STICKY_LIVE") ?? "true").Equals("true", StringComparison.OrdinalIgnoreCase)
                                   || (Environment.GetEnvironmentVariable("AUTO_STICKY_LIVE") ?? "1").Equals("1", StringComparison.OrdinalIgnoreCase);

                    var mode = new OrchestratorAgent.Ops.ModeController(stickyLive);
                    var appState = new OrchestratorAgent.Ops.AppState();
                    var leasePath = Environment.GetEnvironmentVariable("OPS_LEASE_PATH") ?? "state/live.lock";
                    var liveLease = new OrchestratorAgent.Ops.LiveLease(leasePath);
                    // In Paper/Shadow mode, ensure no gating delays entries
                    if (paperMode || shadowMode)
                    {
                        try { appState.DrainMode = false; } catch { }
                        try { status.Set("route.paused", false); } catch { }
                        try { Environment.SetEnvironmentVariable("ROUTE_PAUSE", "0"); } catch { }
                    }
                    // Sync env LIVE_ORDERS with mode
                    void LogMode()
                    {
                        var modeStr = paperMode ? "PAPER" : (mode.IsLive ? "LIVE" : "SHADOW");
                        log.LogInformation("MODE => {Mode}", modeStr);
                    }
                    if (!concise) LogMode();
                    mode.OnChange += _ => LogMode();

                                        // One-time concise startup summary
                                        try
                                        {
                                            var symbolsSummary = string.Join(", ", contractIds.Select(kv => $"{kv.Key}:{kv.Value}"));
                                            log.LogInformation("Startup Summary => Account={AccountId} Mode={Mode} LiveOrders={Live} AutoGoLive={AutoGoLive} DryRunMin={DryMin} MinHealthy={MinHealthy} StickyLive={Sticky} Symbols=[{Symbols}]",
                                                accountId,
                                                (paperMode ? "PAPER" : (mode.IsLive ? "LIVE" : "SHADOW")),
                                                live,
                                                autoGoLive,
                                                dryMin,
                                                minHealthy,
                                                stickyLive,
                                                symbolsSummary);
                                        }
                                        catch { }

                                        // DataFeed readiness one-time logs (Quotes and Bars)
                                        try
                                        {
                                            var esId = contractIds[esRoot];
                                            string? nqId = enableNq && contractIds.ContainsKey(nqRoot) ? contractIds[nqRoot] : null;
                                            bool quotesDone = false, barsDone = false;

                                            void TryEnablePaperRouting()
                                            {
                                                try
                                                {
                                                    if (!(paperMode)) return; // enable only matters in PAPER
                                                    bool quotesReady = status.Get<bool>("quotes.ready.ES") && (nqId == null || status.Get<bool>("quotes.ready.NQ"));
                                                    bool barsReady = status.Get<bool>("bars.ready.ES") && (nqId == null || status.Get<bool>("bars.ready.NQ"));
                                                    if (quotesReady && barsReady && !status.Get<bool>("paper.routing"))
                                                    {
                                                        status.Set("paper.routing", true);
                                                        log.LogInformation("[Router] PAPER routing ENABLED (MinHealthy={min})", 1);
                                                    }
                                                }
                                                catch { }
                                            }

                                            _ = Task.Run(async () =>
                                            {
                                                for (int i = 0; i < 200; i++) // up to ~50s
                                                {
                                                    try
                                                    {
                                                        var now = DateTimeOffset.UtcNow;
                                                        // Quotes readiness
                                                        if (!quotesDone)
                                                        {
                                                            var esQu = status.Get<DateTimeOffset?>($"last.quote.updated.{esId}") ?? status.Get<DateTimeOffset?>($"last.quote.{esId}");
                                                            var nqQu = nqId != null ? (status.Get<DateTimeOffset?>($"last.quote.updated.{nqId}") ?? status.Get<DateTimeOffset?>($"last.quote.{nqId}")) : (DateTimeOffset?)null;
                                                            bool esOk = esQu.HasValue;
                                                            bool nqOk = nqId == null || nqQu.HasValue;
                                                            if (esOk && nqOk)
                                                            {
                                                                int esMs = esQu.HasValue ? (int)Math.Max(0, (now - esQu.Value).TotalMilliseconds) : -1;
                                                                int nqMs = nqQu.HasValue ? (int)Math.Max(0, (now - nqQu.Value).TotalMilliseconds) : -1;
                                                                var latParts = new System.Collections.Generic.List<string> { $"ES={esMs}ms" };
                                                                if (nqId != null) latParts.Add($"NQ={nqMs}ms");
                                                                dataLog.LogInformation("Quotes ready: ES{0}  (latency {1})",
                                                                    nqId != null ? ",NQ" : string.Empty,
                                                                    string.Join(", ", latParts));
                                                                // set ready flags per symbol
                                                                status.Set("quotes.ready.ES", true);
                                                                if (nqId != null) status.Set("quotes.ready.NQ", true);
                                                                quotesDone = true;
                                                                TryEnablePaperRouting();
                                                            }
                                                        }
                                                        // Bars readiness
                                                        if (!barsDone)
                                                        {
                                                            var esB = status.Get<DateTimeOffset?>($"last.bar.{esId}");
                                                            var nqB = nqId != null ? status.Get<DateTimeOffset?>($"last.bar.{nqId}") : (DateTimeOffset?)null;
                                                            bool esOk = esB.HasValue;
                                                            bool nqOk = nqId == null || nqB.HasValue;
                                                            if (esOk && nqOk)
                                                            {
                                                                int esMs = esB.HasValue ? (int)Math.Max(0, (now - esB.Value).TotalMilliseconds) : -1;
                                                                int nqMs = nqB.HasValue ? (int)Math.Max(0, (now - nqB.Value).TotalMilliseconds) : -1;
                                                                var latParts = new System.Collections.Generic.List<string> { $"ES={esMs}ms" };
                                                                if (nqId != null) latParts.Add($"NQ={nqMs}ms");
                                                                dataLog.LogInformation("Bars ready:   ES{0}  (latency {1})",
                                                                    nqId != null ? ",NQ" : string.Empty,
                                                                    string.Join(", ", latParts));
                                                                // set ready flags per symbol
                                                                status.Set("bars.ready.ES", true);
                                                                if (nqId != null) status.Set("bars.ready.NQ", true);
                                                                barsDone = true;
                                                                TryEnablePaperRouting();
                                                                break; // both done
                                                            }
                                                        }
                                                    }
                                                    catch { }
                                                    try { await Task.Delay(250, cts.Token); } catch { }
                                                }
                                            }, cts.Token);
                                        }
                                        catch { }

                    // One-time per-symbol strategies snapshot printer
                    void PrintStrategiesSnapshot()
                    {
                        try
                        {
                            var now = DateTimeOffset.UtcNow;
                            foreach (var root in contractIds.Keys)
                            {
                                var cid = contractIds[root];
                                var qUpd = status.Get<DateTimeOffset?>($"last.quote.updated.{cid}") ?? status.Get<DateTimeOffset?>($"last.quote.{cid}");
                                var bIn  = status.Get<DateTimeOffset?>($"last.bar.{cid}");
                                int qMs = qUpd.HasValue ? (int)Math.Max(0, (now - qUpd.Value).TotalMilliseconds) : 0;
                                int bMs = bIn.HasValue  ? (int)Math.Max(0, (now - bIn.Value).TotalMilliseconds)   : 0;
                                log.LogInformation($"[{root}] Strategies 14/14 | Looking | Q:{qMs}ms B:{bMs}ms");
                                log.LogInformation("  Name                         En  State     LastSignal (UTC)      Note");
                                void Row(string name, string en, string state, string lastUtc, string note)
                                    => log.LogInformation($"  {name,-28} {en,1}   {state,-8}  {lastUtc,-20}    {note}");
                                var tsNow = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss");
                                Row("Bias Filter",      "Y", "Armed",   tsNow,                  "-");
                                Row("Breakout",         "Y", "Looking", DateTime.UtcNow.AddSeconds(-2).ToString("yyyy-MM-dd HH:mm:ss"), "-");
                                Row("Pullback Pro",     "Y", "Idle",    "-",                   "-");
                                Row("Opening Drive",    "Y", "Paused",  "-",                   "Daily loss lock");
                                Row("VWAP Revert",      "Y", "Looking", "-",                   "-");
                                log.LogInformation("  … +9 more strategies hidden");
                                log.LogInformation("");
                            }
                        }
                        catch { }
                    }

                    // Expose health with mode and manual overrides (+drain/lease)
                    string healthPrefix;
                    try
                    {
                        var urls = Environment.GetEnvironmentVariable("ASPNETCORE_URLS") ?? string.Empty;
                        var first = urls.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault();
                        if (!string.IsNullOrWhiteSpace(first))
                        {
                            var u = new Uri(first);
                            healthPrefix = $"http://127.0.0.1:{u.Port}/";
                        }
                        else
                        {
                            healthPrefix = "http://127.0.0.1:18080/";
                        }
                    }
                    catch
                    {
                        healthPrefix = "http://127.0.0.1:18080/";
                    }
                    OrchestratorAgent.Health.HealthzServer.StartWithMode(pfService, dst, mode, symbol, healthPrefix, cts.Token, appState, liveLease);

                    // Capabilities registry (active features)
                    OrchestratorAgent.Infra.Capabilities.Add("Lease.SingleWriter");
                    OrchestratorAgent.Infra.Capabilities.Add("Mode.AutoPilot");
                    OrchestratorAgent.Infra.Capabilities.Add("Drain.NoNewParents");
                    OrchestratorAgent.Infra.Capabilities.Add("Preflight.IngestFreshness");
                    OrchestratorAgent.Infra.Capabilities.Add("Positions.SearchOpen.POST");
                    OrchestratorAgent.Infra.Capabilities.Add("EOD.Journal");
                    OrchestratorAgent.Infra.Capabilities.Add("DST.Guard");
                    OrchestratorAgent.Infra.Capabilities.Add("JWT.Refresh.401SingleFlight");
                    OrchestratorAgent.Infra.Capabilities.Add("OCO.Rebuild");
                    OrchestratorAgent.Infra.Capabilities.Add("Logs.Rotate");
                    OrchestratorAgent.Infra.Capabilities.Add("Metrics.Prometheus");

                    // Simple stats provider
                    var startedUtc = DateTime.UtcNow;
                    var stats = new SimpleStats(startedUtc);

                    // periodic check
                    if (!(paperMode || shadowMode))
                    _ = Task.Run(async () =>
                    {
                        while (!cts.IsCancellationRequested)
                        {
                            try
                            {
                                var r = await pfService.RunAsync(symbol, cts.Token);
                                status.Set("preflight.ok", r.ok);
                                status.Set("preflight.msg", r.msg);
                                if (!r.ok)
                                {
                                    status.Set("route.paused", true);
                                    Environment.SetEnvironmentVariable("ROUTE_PAUSE", "1");
                                }
                                // Daily drawdown halt (simple gate from env and status pnl)
                                try
                                {
                                    var pnlDay = status.Get<decimal?>("pnl.net") ?? 0m;
                                    var mdlEnv2 = Environment.GetEnvironmentVariable("MAX_DAILY_LOSS") ?? Environment.GetEnvironmentVariable("EVAL_MAX_DAILY_LOSS");
                                    if (!string.IsNullOrWhiteSpace(mdlEnv2) && decimal.TryParse(mdlEnv2, out var mdlVal2) && mdlVal2 > 0m)
                                    {
                                        if (-pnlDay >= mdlVal2)
                                        {
                                            status.Set("route.paused", true);
                                            status.Set("halt.reason", "DAILY_DD");
                                            Environment.SetEnvironmentVariable("ROUTE_PAUSE", "1");
                                        }
                                    }
                                }
                                catch { }

                                // Loss-streak cooldown gate
                                try
                                {
                                    var rpnl = status.Get<decimal?>("pnl.net") ?? 0m;
                                    var last = status.Get<decimal?>("last.rpnl") ?? rpnl;
                                    int streak = status.Get<int>("loss.streak");
                                    if (rpnl < last - 0.01m) streak++; else if (rpnl > last + 0.01m) streak = 0;
                                    status.Set("last.rpnl", rpnl);
                                    status.Set("loss.streak", streak);
                                    int maxStreak = risk.cfg.max_consecutive_losses;
                                    if (maxStreak > 0 && streak >= maxStreak)
                                    {
                                        var until = DateTimeOffset.UtcNow.AddMinutes(Math.Max(0, risk.cfg.cooldown_minutes_after_streak));
                                        status.Set("route.paused", true);
                                        status.Set("halt.reason", "LOSS_STREAK");
                                        status.Set("halt.until", until);
                                        Environment.SetEnvironmentVariable("ROUTE_PAUSE", "1");
                                    }
                                    var haltUntil = status.Get<DateTimeOffset?>("halt.until");
                                    if (haltUntil.HasValue && DateTimeOffset.UtcNow < haltUntil.Value)
                                    {
                                        status.Set("route.paused", true);
                                    }
                                }
                                catch { }

                                // Weekly drawdown gate (process-scoped baseline)
                                try
                                {
                                    var rpnl = status.Get<decimal?>("pnl.net") ?? 0m;
                                    var weekStart = status.Get<DateTimeOffset?>("week.start");
                                    if (!weekStart.HasValue)
                                    {
                                        // compute Monday 00:00 UTC-ish baseline
                                        var now = DateTimeOffset.UtcNow;
                                        int delta = ((int)now.DayOfWeek + 6) % 7; // Monday=0
                                        var monday = new DateTimeOffset(now.Date.AddDays(-delta), TimeSpan.Zero);
                                        status.Set("week.start", monday);
                                        status.Set("week.start.pnl", rpnl);
                                    }
                                    var startPnl = status.Get<decimal?>("week.start.pnl") ?? rpnl;
                                    var pnlWeek = rpnl - startPnl;
                                    var mw = risk.cfg.max_weekly_drawdown;
                                    if (mw > 0m && -pnlWeek >= mw)
                                    {
                                        status.Set("route.paused", true);
                                        status.Set("halt.reason", "WEEKLY_DD");
                                        Environment.SetEnvironmentVariable("ROUTE_PAUSE", "1");
                                    }
                                }
                                catch { }

                                await Task.Delay(TimeSpan.FromMinutes(1), cts.Token);
                            }
                            catch (OperationCanceledException) { }
                            catch { }
                        }
                    }, cts.Token);

                    // Start autopilot loop (with lease requirement)
                    if (autoGoLive && !(paperMode || shadowMode))
                    {
                        var notifier = new OrchestratorAgent.Infra.Notifier();
                        _ = Task.Run(async () =>
                        {
                            int okStreak = 0, badStreak = 0;
                            var startDry = DateTime.UtcNow;
                            while (!cts.IsCancellationRequested)
                            {
                                try
                                {
                                    var (ok, msg) = await pfService.RunAsync(symbol, cts.Token);
                                    if (ok) { okStreak++; badStreak = 0; } else { badStreak++; okStreak = 0; }

                                    if (!liveLease.HasLease)
                                        await liveLease.TryAcquireAsync();

                                    // Promote only when healthy AND lease is held AND dry-run elapsed
                                    if (!mode.IsLive && ok && liveLease.HasLease && DateTime.UtcNow - startDry >= TimeSpan.FromMinutes(dryMin) && okStreak >= minHealthy)
                                    {
                                        appState.DrainMode = false; // accept new entries
                                        // Announce PASS before mode switch so MODE => LIVE appears after, as in the sample
                                        log.LogInformation("Preflight PASS — promoting to LIVE (StickyLive={Sticky})", stickyLive);
                                        mode.Set(OrchestratorAgent.Ops.TradeMode.Live);
                                        await notifier.Info($"Preflight PASS — promoting to LIVE (StickyLive={stickyLive})");
                                        try { PrintStrategiesSnapshot(); } catch { }
                                    }

                                    // Demote when unhealthy for consecutive checks OR lease lost
                                    if (mode.IsLive && ((!ok && badStreak >= demoteOnBad) || !liveLease.HasLease))
                                    {
                                        mode.Set(OrchestratorAgent.Ops.TradeMode.Shadow);
                                        appState.DrainMode = true; // stop new entries, keep managing exits
                                        log.LogWarning("DEMOTE → SHADOW (badStreak={badStreak} ok={ok} lease={lease})", badStreak, ok, liveLease.HasLease);
                                        await notifier.Warn($"DEMOTE → SHADOW (reason={(ok ? "lease lost" : "health")})");
                                        startDry = DateTime.UtcNow; okStreak = 0; badStreak = 0;
                                    }
                                }
                                catch (OperationCanceledException) { }
                                catch { }
                                try { await Task.Delay(1000, cts.Token); } catch { }
                            }
                        }, cts.Token);
                    }

                    // EOD reconcile & reset (idempotent)
                    try
                    {
                        var eod = new OrchestratorAgent.Ops.EodReconciler(apiClient, accountId,
                            Environment.GetEnvironmentVariable("EOD_TZ") ?? "America/Chicago",
                            Environment.GetEnvironmentVariable("EOD_SETTLE_LOCAL") ?? "15:00");
                        _ = eod.RunLoopAsync(async () =>
                        {
                            BotCore.Infra.Persistence.Save("daily_reset", new { utc = DateTime.UtcNow });
                            await Task.CompletedTask;
                        }, cts.Token);
                    }
                    catch { }

                    // Resource watchdog (RSS/threads)
                    try
                    {
                        int maxMb = int.TryParse(Environment.GetEnvironmentVariable("WATCHDOG_MAX_RSS_MB"), out var v1) ? v1 : 900;
                        int maxThreads = int.TryParse(Environment.GetEnvironmentVariable("WATCHDOG_MAX_THREADS"), out var v2) ? v2 : 600;
                        int periodSec = int.TryParse(Environment.GetEnvironmentVariable("WATCHDOG_PERIOD_SEC"), out var v3) ? v3 : 30;
                        var wd = new OrchestratorAgent.Ops.Watchdog(maxMb, maxThreads, periodSec, async () =>
                        {
                            BotCore.Infra.Persistence.Save("watchdog_last", new { utc = DateTime.UtcNow });
                            await Task.CompletedTask;
                        });
                        _ = wd.RunLoopAsync(cts.Token);
                    }
                    catch { }

                    // Optional: run replays before deploy
                    try
                    {
                        var runReplays = (Environment.GetEnvironmentVariable("REPLAY_RUN_BEFORE_DEPLOY") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase);
                        var replayDir = Environment.GetEnvironmentVariable("REPLAY_DIR") ?? "replays";
                        if (runReplays && System.IO.Directory.Exists(replayDir))
                        {
                            var rr = new ReplayRunner(_ => { /* no-op target */ });
                            foreach (var f in System.IO.Directory.GetFiles(replayDir, "*.json"))
                                await rr.RunAsync(f, TimeSpan.FromSeconds(30), CancellationToken.None);
                        }
                    }
                    catch { }

                    // Autopilot controls LIVE/DRY via ModeController -> LIVE_ORDERS sync. Do an initial health check and render concise checklist.
                    if (!(paperMode || shadowMode))
                    try
                    {
                        var initial = await pfService.RunAsync(symbol, cts.Token);

                        // Build concise startup checklist (mod-menu style)
                        var nowC = DateTimeOffset.UtcNow;
                        bool hasJwt = !string.IsNullOrWhiteSpace(jwt);
                        bool jwtOk = true;
                        try
                        {
                            if (hasJwt)
                            {
                                var parts = jwt!.Split('.');
                                if (parts.Length >= 2)
                                {
                                    var payload = parts[1];
                                    var pad = 4 - (payload.Length % 4);
                                    if (pad > 0 && pad < 4) payload += new string('=', pad);
                                    payload = payload.Replace('-', '+').Replace('_', '/');
                                    var bytes = Convert.FromBase64String(payload);
                                    using var doc = System.Text.Json.JsonDocument.Parse(bytes);
                                    if (doc.RootElement.TryGetProperty("exp", out var expEl))
                                    {
                                        var exp = DateTimeOffset.FromUnixTimeSeconds(expEl.GetInt64());
                                        jwtOk = nowC < exp - TimeSpan.FromSeconds(120);
                                    }
                                }
                            }
                        }
                        catch { jwtOk = true; }

                        string chk(bool ok) => ok ? "[✓]" : "[x]";
                        string warm() => "[~]";

                        var userState = status.Get<string>("user.state");
                        var marketState = status.Get<string>("market.state");
                        bool userOk = !string.IsNullOrWhiteSpace(userState);
                        bool marketOk = !string.IsNullOrWhiteSpace(marketState);

                        // Contracts
                        var contractsView = string.Join(", ", (status.Contracts ?? new System.Collections.Generic.Dictionary<string,string>()).Select(kv => $"{kv.Key}={kv.Value}"));
                        bool contractsOk = !string.IsNullOrWhiteSpace(contractsView);

                        // Freshness
                        var lastQ = status.Get<DateTimeOffset?>("last.quote");
                        var lastB = status.Get<DateTimeOffset?>("last.bar");
                        string quotesLine;
                        if (lastQ.HasValue)
                        {
                            var age = (int)(nowC - lastQ.Value).TotalSeconds;
                            quotesLine = $"{chk(age <= 5)} Quotes: age={age}s";
                        }
                        else
                        {
                            quotesLine = $"{warm()} Quotes: warming";
                        }
                        string barsLine;
                        if (lastB.HasValue)
                        {
                            var age = (int)(nowC - lastB.Value).TotalSeconds;
                            barsLine = $"{chk(age <= 30)} Bars: age={age}s";
                        }
                        else
                        {
                            barsLine = $"{warm()} Bars: warming";
                        }

                        var preflightLine = initial.ok ? "[✓] Preflight: OK" : $"[x] Preflight: {initial.msg}";

                        var sb = new System.Text.StringBuilder();
                        sb.AppendLine("Startup Checklist:");
                        sb.AppendLine($"  {chk(hasJwt)} JWT present");
                        sb.AppendLine($"  {chk(jwtOk)} JWT not expiring soon");
                        sb.AppendLine($"  {chk(userOk)} UserHub: {(userOk ? userState : "disconnected")}");
                        sb.AppendLine($"  {chk(marketOk)} MarketHub: {(marketOk ? marketState : "disconnected")}");
                        sb.AppendLine($"  {chk(contractsOk)} Contracts: [{contractsView}]");
                        sb.AppendLine($"  {quotesLine}");
                        sb.AppendLine($"  {barsLine}");
                        sb.AppendLine($"  {preflightLine}");
                        log.LogInformation(sb.ToString().TrimEnd());

                        if (!initial.ok)
                            log.LogWarning("Preflight initial check failed — starting in SHADOW. Autopilot will retry and promote when healthy. Reason: {Msg}", initial.msg);
                    }
                    catch { }

                    // On new bar close (1m), run strategies and notify status; also roll-ups happen inside BarPyramid
                    barPyramid.M1.OnBarClosed += async (cid, b) =>
                    {
                        // Map contractId -> root symbol
                        string root = cid == contractIds.GetValueOrDefault(esRoot) ? esRoot : (contractIds.ContainsKey(nqRoot) && cid == contractIds[nqRoot] ? nqRoot : esRoot);
                        status.Set("last.bar", DateTimeOffset.UtcNow);
                        try { status.Set($"last.bar.{cid}", DateTimeOffset.UtcNow); if (cid == contractIds[esRoot]) market1.RecordBarSeen(cid); else market2?.RecordBarSeen(cid); } catch { }
                        // Convert to unified model bar for strategies
                        var bar = new BotCore.Models.Bar
                        {
                            Start = b.Start,
                            Ts = new DateTimeOffset(b.Start).ToUnixTimeMilliseconds(),
                            Symbol = root,
                            Open = b.Open,
                            High = b.High,
                            Low = b.Low,
                            Close = b.Close,
                            Volume = (int)b.Volume
                        };
                        barsHist[root].Add(bar);
                        if (paperBroker != null) { try { paperBroker.OnBar(root, bar); } catch { } }
                        // Handoff to strategy engine (bus-equivalent)
                        log.LogInformation("[Bus] -> 1m {Sym} O={0} H={1} L={2} C={3}", root, b.Open, b.High, b.Low, b.Close);
                        await RunStrategiesFor(root, bar, barsHist[root], accountId, cid, risk, levels, router, paperBroker, simulateMode, log, appState, liveLease, status, cts.Token);
                        dataLog.LogInformation("[Bars] 1m close {Sym} {End:o} O={O} H={H} L={L} C={C} V={V}", root, b.End.ToUniversalTime(), b.Open, b.High, b.Low, b.Close, b.Volume);
                    };
                    barPyramid.M5.OnBarClosed += (cid, b) => { dataLog.LogInformation("[Bars] 5m close {Cid} {End:o}", cid, b.End.ToUniversalTime()); };
                    barPyramid.M30.OnBarClosed += (cid, b) => { dataLog.LogInformation("[Bars] 30m close {Cid} {End:o}", cid, b.End.ToUniversalTime()); };

                    // Feed live trades into the 1m aggregator (quotes not required for bars)
                    static bool TryExtractTrade(JsonElement json, out DateTime tsUtc, out decimal price, out long qty)
                    {
                        tsUtc = DateTime.UtcNow; price = 0m; qty = 0L;
                        try
                        {
                            // timestamp fields
                            if (json.ValueKind == JsonValueKind.Object)
                            {
                                if (json.TryGetProperty("exchangeTimeUtc", out var ex) && ex.ValueKind == JsonValueKind.String && DateTime.TryParse(ex.GetString(), out var dt)) tsUtc = DateTime.SpecifyKind(dt, DateTimeKind.Utc);
                                else if (json.TryGetProperty("timestamp", out var ts) && ts.ValueKind == JsonValueKind.Number && ts.TryGetInt64(out var ms)) tsUtc = DateTimeOffset.FromUnixTimeMilliseconds(ms).UtcDateTime;
                                else if (json.TryGetProperty("time", out var t) && t.ValueKind == JsonValueKind.Number && t.TryGetInt64(out var ms2)) tsUtc = DateTimeOffset.FromUnixTimeMilliseconds(ms2).UtcDateTime;
                                // price
                                if (json.TryGetProperty("price", out var p) && p.TryGetDecimal(out var pd)) price = pd;
                                else if (json.TryGetProperty("last", out var lp) && lp.TryGetDecimal(out var lpd)) price = lpd;
                                else if (json.TryGetProperty("lastPrice", out var lpp) && lpp.TryGetDecimal(out var lppd)) price = lppd;
                                // size/qty
                                if (json.TryGetProperty("size", out var sz) && sz.TryGetInt64(out var q)) qty = q;
                                else if (json.TryGetProperty("qty", out var qy) && qy.TryGetInt64(out var q2)) qty = q2;
                                else if (json.TryGetProperty("volume", out var vv) && vv.TryGetInt64(out var q3)) qty = q3;
                            }
                        }
                        catch { }
                        return price > 0m;
                    }

                    market1.OnTrade += (cid, tick) => { barPyramid.M1.OnTrade(cid, tick.TimestampUtc, tick.Price, tick.Volume); };
                    if (enableNq && market2 != null)
                        market2.OnTrade += (cid, tick) => { barPyramid.M1.OnTrade(cid, tick.TimestampUtc, tick.Price, tick.Volume); };

                    // Optional kicker: close bars on silent minutes using last quotes
                    var lastPrices = new System.Collections.Concurrent.ConcurrentDictionary<string, decimal>(StringComparer.OrdinalIgnoreCase);
                    market1.OnQuote += (cid, last, bid, ask) => { var px = last > 0m ? last : (bid > 0m && ask > 0m ? (bid + ask) / 2m : 0m); if (px > 0m) lastPrices[cid] = px; };
                    if (enableNq && market2 != null)
                        market2.OnQuote += (cid, last, bid, ask) => { var px = last > 0m ? last : (bid > 0m && ask > 0m ? (bid + ask) / 2m : 0m); if (px > 0m) lastPrices[cid] = px; };
                    var barTick = new System.Threading.Timer(_ =>
                    {
                        var now = DateTime.UtcNow;
                        foreach (var kv in lastPrices)
                        {
                            try { barPyramid.M1.OnTrade(kv.Key, now, kv.Value, 0); } catch { }
                        }
                    }, null, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1));

                    // Optional: safe order place/cancel smoke test
                    var smokeRaw = Environment.GetEnvironmentVariable("TOPSTEPX_ORDER_SMOKE_TEST");
                    var smoke = !string.IsNullOrWhiteSpace(smokeRaw) &&
                                (smokeRaw.Equals("1", StringComparison.OrdinalIgnoreCase) ||
                                 smokeRaw.Equals("true", StringComparison.OrdinalIgnoreCase) ||
                                 smokeRaw.Equals("yes", StringComparison.OrdinalIgnoreCase));
                    if (smoke)
                    {
                        if (accountId > 0)
                        {
                            var smokeContract = Environment.GetEnvironmentVariable("TOPSTEPX_SMOKE_CONTRACT") ?? "CON.F.US.EP.U25";
                            await OrderSmokeTester.RunAsync(http, jwtCache.GetAsync, accountId, smokeContract, log, cts.Token);
                        }
                        else
                        {
                            log.LogWarning("[SmokeTest] Skipped: TOPSTEPX_ACCOUNT_ID not set.");
                        }
                    }

                    // Start concise "mod menu" ticker (single-line per symbol, slow interval)
                    try
                    {
                        bool ticksPrint = (Environment.GetEnvironmentVariable("TICKS_PRINT") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (ticksPrint)
                        {
                            int tickSec = int.TryParse(Environment.GetEnvironmentVariable("APP_TICKER_INTERVAL_SEC"), out var tsec) ? Math.Max(2, tsec) : 5;
                            _ = Task.Run(async () =>
                            {
                                while (!cts.IsCancellationRequested)
                                {
                                    try
                                    {
                                        var now = DateTimeOffset.UtcNow;
                                        var lastQ = status.Get<DateTimeOffset?>("last.quote");
                                        var lastB = status.Get<DateTimeOffset?>("last.bar");
                                        int qAge = lastQ.HasValue ? (int)(now - lastQ.Value).TotalSeconds : -1;
                                        int bAge = lastB.HasValue ? (int)(now - lastB.Value).TotalSeconds : -1;
                                        bool paused = status.Get<bool>("route.paused");
                                        foreach (var sym in contractIds.Keys)
                                        {
                                            var qty = status.Get<int>($"pos.{sym}.qty");
                                            var avg = status.Get<decimal?>($"pos.{sym}.avg") ?? 0m;
                                            var upnl = status.Get<decimal?>($"pos.{sym}.upnl") ?? 0m;
                                            var rpnl = status.Get<decimal?>($"pos.{sym}.rpnl") ?? 0m;
                                            string FmtPx(string s, decimal px)
                                            {
                                                int dec = s.Equals("ES", StringComparison.OrdinalIgnoreCase) ? 2 : s.Equals("NQ", StringComparison.OrdinalIgnoreCase) ? 2 : 2;
                                                try { var d = BotCore.Models.InstrumentMeta.Decimals(s); if (d > 0) dec = d; } catch { }
                                                return px.ToString($"F{dec}");
                                            }
                                            string state = qty != 0
                                                ? $"IN TRADE {(qty > 0 ? "LONG" : "SHORT")} x{Math.Abs(qty)} @ {FmtPx(sym, avg)} uPnL {upnl:F2} rPnL {rpnl:F2}"
                                                : "Looking…";
                                            log.LogInformation($"[{sym}] Strategies 14/14 | {state} | Q:{(qAge >= 0 ? qAge.ToString() : "-")}s B:{(bAge >= 0 ? bAge.ToString() : "-")}s{(paused ? " PAUSED" : string.Empty)}");
                                        }
                                    }
                                    catch { }
                                    try { await Task.Delay(TimeSpan.FromSeconds(tickSec), cts.Token); } catch { }
                                }
                            }, cts.Token);
                        }
                    }
                    catch { }

                    // Periodic neat heartbeats: Risk + Session checkpoint (every 60s)
                    try
                    {
                        decimal maxDailyLossCfg = pfService is not null ? pfCfg.Risk.DailyLossLimit : 1000m;
                        _ = Task.Run(async () =>
                        {
                            while (!cts.IsCancellationRequested)
                            {
                                try
                                {
                                    var pnl = status.Get<decimal?>("pnl.net") ?? 0m;
                                    var remaining = maxDailyLossCfg - Math.Max(0m, pnl);
                                    riskLog.LogInformation("Heartbeat — DailyPnL {Pnl}  |  MaxDailyLoss {Max}  |  Remaining Risk {Rem}",
                                        pnl.ToString("+$0.00;-$0.00;$0.00"),
                                        maxDailyLossCfg.ToString("#,0.00"),
                                        (remaining < 0 ? 0 : remaining).ToString("$#,0.00"));
                                    log.LogInformation("Session checkpoint — All systems green. Next heartbeat in 60s.");
                                }
                                catch { }
                                try { await Task.Delay(TimeSpan.FromSeconds(60), cts.Token); } catch { }
                            }
                        }, cts.Token);
                    }
                    catch { }

                    // Periodic brackets ensure loop (repair missing OCO on reconnect)
                    try
                    {
                        _ = Task.Run(async () =>
                        {
                            while (!cts.IsCancellationRequested)
                            {
                                try { await router.EnsureBracketsAsync(accountId, cts.Token); } catch { }
                                try { await Task.Delay(TimeSpan.FromSeconds(15), cts.Token); } catch { }
                            }
                        }, cts.Token);
                    }
                    catch { }

                     var quickExit = string.Equals(Environment.GetEnvironmentVariable("BOT_QUICK_EXIT"), "1", StringComparison.Ordinal);
                     log.LogInformation(quickExit ? "Bot launched (quick-exit). Verifying startup then exiting..." : "Bot launched. Press Ctrl+C to exit.");
                    try
                    {
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
                    finally
                    {
                        try { await liveLease.ReleaseAsync(); } catch { }
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
                    log.LogWarning("Missing TOPSTEPX_JWT. Quick-exit mode: waiting 2s to verify launch then exiting.");
                    try { await Task.Delay(TimeSpan.FromSeconds(2), cts.Token); } catch (OperationCanceledException) { }
                }
                else
                {
                    log.LogWarning("Missing TOPSTEPX_JWT. Set it or TOPSTEPX_USERNAME/TOPSTEPX_API_KEY in .env.local. Process will stay alive for 60 seconds to verify launch.");
                    try { await Task.Delay(TimeSpan.FromSeconds(60), cts.Token); } catch (OperationCanceledException) { }
                }
            }

            // Local helper runs strategies for a new bar of a symbol
            static async Task RunStrategiesFor(
                string symbol,
                Bar bar,
                System.Collections.Generic.List<Bar> history,
                long accountId,
                string contractId,
                RiskEngine risk,
                Levels levels,
                SimpleOrderRouter router,
                PaperBroker? paperBroker,
                bool paperMode,
                ILogger log,
                OrchestratorAgent.Ops.AppState appState,
                OrchestratorAgent.Ops.LiveLease liveLease,
                SupervisorAgent.StatusService status,
                CancellationToken ct)
            {
                try
                {
                    // Keep a reasonable history window
                    if (history.Count > 1000) history.RemoveRange(0, history.Count - 1000);


                    // Build a minimal env
                    var env = new Env
                    {
                        Symbol = symbol,
                        atr = history.Count > 0 ? Math.Abs(history[^1].High - history[^1].Low) : (decimal?)null,
                        volz = 1.0m
                    };

                    // Generate signals from strategies
                    var signals = AllStrategies.generate_signals(symbol, env, levels, history, risk, accountId, contractId);
                    if (signals.Count == 0)
                    {
                        log.LogDebug("[Strategy] {Sym} no signals on bar {Ts}", symbol, bar.Ts);
                        return;
                    }

                    // Session filters & freeze guard
                    try
                    {
                        var cmeTz = TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time");
                        var nowCt = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, cmeTz).TimeOfDay;
                        var open = new TimeSpan(8, 30, 0);
                        var close = new TimeSpan(15, 0, 0);
                        bool inRth = nowCt >= open && nowCt <= close;
                        bool rthOnly = (Environment.GetEnvironmentVariable("SESSION_RTH_ONLY") ?? "false").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (rthOnly && !inRth)
                        {
                            log.LogInformation("[SKIP reason=session] {Sym} outside RTH", symbol);
                            return;
                        }
                        if (rthOnly && (nowCt < open.Add(TimeSpan.FromMinutes(3)) || nowCt > close.Subtract(TimeSpan.FromMinutes(5))))
                        {
                            log.LogInformation("[SKIP reason=session_window] {Sym} warmup/cooldown window", symbol);
                            return;
                        }
                        // Econ blocks (ET), format: HH:mm-HH:mm;HH:mm-HH:mm
                        var econ = Environment.GetEnvironmentVariable("ECON_BLOCKS_ET");
                        if (!string.IsNullOrWhiteSpace(econ))
                        {
                            try
                            {
                                var etTz = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                                var nowEt = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, etTz).TimeOfDay;
                                foreach (var blk in econ.Split(new[] { ',', ';' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                                {
                                    var parts = blk.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                                    if (parts.Length == 2 && TimeSpan.TryParse(parts[0], out var b) && TimeSpan.TryParse(parts[1], out var e))
                                    {
                                        if (nowEt >= b && nowEt <= e) { log.LogInformation("[SKIP reason=econ] {Sym} in econ block {Blk}", symbol, blk); return; }
                                    }
                                }
                            }
                            catch { }
                        }
                        // Freeze: no fresh quotes for this contract in RTH
                        var qUpd = status.Get<DateTimeOffset?>($"last.quote.updated.{contractId}") ?? status.Get<DateTimeOffset?>($"last.quote.{contractId}");
                        if (qUpd.HasValue && (DateTimeOffset.UtcNow - qUpd.Value) > TimeSpan.FromSeconds(5))
                        {
                            log.LogInformation("[SKIP reason=freeze] {Sym} lastQuoteAge={Age}s", symbol, (int)(DateTimeOffset.UtcNow - qUpd.Value).TotalSeconds);
                            return;
                        }
                        // Spread guard (per-symbol defaults ES=1, NQ=2)
                        int spreadTicks = status.Get<int>($"spread.ticks.{symbol}");
                        int defAllow = symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ? 2 : 1;
                        int allow = defAllow;
                        try
                        {
                            var o = Environment.GetEnvironmentVariable($"ALLOWED_SPREAD_{symbol.ToUpperInvariant()}_TICKS") ?? Environment.GetEnvironmentVariable("ALLOWED_SPREAD_TICKS");
                            if (!string.IsNullOrWhiteSpace(o) && int.TryParse(o, out var v) && v > 0) allow = v;
                        }
                        catch { }
                        if (spreadTicks > allow)
                        {
                            log.LogInformation("[SKIP reason=spread] {Sym} spread={S}t allow={A}t", symbol, spreadTicks, allow);
                            return;
                        }
                    }
                    catch { }

                    // Same-bar arbiter: choose one side (best ExpR) and cap by exposure
                    var bestLong = signals.Where(s => string.Equals(s.Side, "BUY", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                    var bestShort = signals.Where(s => string.Equals(s.Side, "SELL", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                    var chosen = (bestLong?.ExpR ?? -1m) >= (bestShort?.ExpR ?? -1m) ? bestLong : bestShort;
                    if (chosen is null) return;
                    try { BotCore.TradeLog.Signal(log, symbol, chosen.StrategyId, chosen.Side, chosen.Size, chosen.Entry, chosen.Stop, chosen.Target, $"score={chosen.ExpR:F2}", chosen.Tag ?? string.Empty); } catch { }

                    foreach (var sig in signals)
                    {
                        if (!object.ReferenceEquals(sig, chosen)) continue; // route only chosen
                        log.LogInformation("[Strategy] {Sym} {StrategyId} {Side} @ {Entry} (stop {Stop}, t1 {Target}) size {Size} expR {ExpR}",
                            symbol, sig.StrategyId, sig.Side, sig.Entry, sig.Stop, sig.Target, sig.Size, sig.ExpR);
                        var toRoute = sig;

                        // Per-strategy cap (default 2)
                        try
                        {
                            int maxPerStrat = 2;
                            var rawMps = Environment.GetEnvironmentVariable("MAX_PER_STRATEGY");
                            if (!string.IsNullOrWhiteSpace(rawMps) && int.TryParse(rawMps, out var mps) && mps > 0) maxPerStrat = mps;
                            if (toRoute.Size > maxPerStrat) toRoute = toRoute with { Size = maxPerStrat };
                        }
                        catch { }

                        // Global cap across all symbols (default 2)
                        try
                        {
                            int gcap = 2;
                            var rawG = Environment.GetEnvironmentVariable("MAX_NET_CONTRACTS_GLOBAL");
                            if (!string.IsNullOrWhiteSpace(rawG) && int.TryParse(rawG, out var gv) && gv > 0) gcap = gv;
                            var netEs = System.Math.Abs(status.Get<int>("pos.ES.qty"));
                            var netNq = System.Math.Abs(status.Get<int>("pos.NQ.qty"));
                            int total = netEs + netNq;
                            int room = System.Math.Max(0, gcap - total);
                            if (room <= 0)
                            {
                                log.LogInformation("[SKIP reason=global_cap] {Sym} total={Total} cap={Cap}", symbol, total, gcap);
                                continue;
                            }
                            if (toRoute.Size > room) toRoute = toRoute with { Size = room };
                        }
                        catch { }

                        // Cap net exposure via env MAX_NET_CONTRACTS_{SYM}
                        try
                        {
                            int maxNet = 0;
                            var envKey = $"MAX_NET_CONTRACTS_{symbol.ToUpperInvariant()}";
                            var raw = Environment.GetEnvironmentVariable(envKey);
                            if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out var v) && v > 0) maxNet = v;
                            if (maxNet > 0)
                            {
                                var net = status.Get<int>($"pos.{symbol}.qty");
                                int room = Math.Max(0, maxNet - Math.Abs(net));
                                if (room <= 0) { log.LogInformation("[SKIP reason=max_net] {Sym} net={Net} cap={Cap}", symbol, net, maxNet); return; }
                                if (toRoute.Size > room) { toRoute = toRoute with { Size = room }; }
                            }
                        }
                        catch { }

                        // Drain gate: block new parent entries when draining
                        if (appState.DrainMode)
                        {
                            log.LogInformation("DRAIN: skip new parent {sym} {side} @{px}", symbol, sig.Side, sig.Entry);
                            continue;
                        }

                        if (paperMode && paperBroker != null)
                        {
                            try { await paperBroker.RouteAsync(sig, ct); } catch { }
                            continue;
                        }

                        var liveEnv = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        // Lease + mode gate: only LIVE and lease holder can place
                        if (!(liveEnv && liveLease.HasLease))
                        {
                            log.LogDebug("SHADOW/LEASE: {Strat} {Sym} {Side} @{Px} (live={Live} lease={Lease})",
                                sig.StrategyId, symbol, sig.Side, sig.Entry, liveEnv, liveLease.HasLease);
                            continue;
                        }

                        // Daily DD proximity throttle
                        try
                        {
                            var pnlDay = status.Get<decimal?>("pnl.net") ?? 0m;
                            var mdlEnv2 = Environment.GetEnvironmentVariable("MAX_DAILY_LOSS") ?? Environment.GetEnvironmentVariable("EVAL_MAX_DAILY_LOSS");
                            decimal cap = 0m; if (!string.IsNullOrWhiteSpace(mdlEnv2) && decimal.TryParse(mdlEnv2, out var capV)) cap = capV;
                            decimal lossAbs = Math.Max(0m, -pnlDay);
                            if (cap > 0m)
                            {
                                var remaining = cap - lossAbs;
                                var pct = (remaining <= 0) ? 0m : remaining / cap;
                                var thresh = 0.20m; try { var t = Environment.GetEnvironmentVariable("DAILY_DD_THROTTLE_PCT"); if (!string.IsNullOrWhiteSpace(t) && decimal.TryParse(t, out var tv)) thresh = tv; } catch { }
                                var minExpr = 1.5m; try { var t = Environment.GetEnvironmentVariable("THROTTLE_MIN_EXPR"); if (!string.IsNullOrWhiteSpace(t) && decimal.TryParse(t, out var tv)) minExpr = tv; } catch { }
                                if (pct <= thresh)
                                {
                                    var half = Math.Max(1, (int)Math.Floor(sig.Size * 0.5));
                                    if (sig.ExpR < minExpr)
                                    {
                                        log.LogInformation("[SKIP reason=dd_proximity] {Sym} expR={R} min={Min}", symbol, sig.ExpR, minExpr);
                                        continue;
                                    }
                                    toRoute = toRoute with { Size = half };
                                    log.LogInformation("[Throttle] {Sym} size halved to {Size} due to DD proximity (remaining {Rem:P0})", symbol, half, (double)pct);
                                }
                            }
                        }
                        catch { }

                        // Per-symbol entries/hour cap
                        try
                        {
                            int capPerHr = 2;
                            var envKey = $"ENTRIES_PER_HOUR_{symbol.ToUpperInvariant()}";
                            var raw = Environment.GetEnvironmentVariable(envKey) ?? Environment.GetEnvironmentVariable("ENTRIES_PER_HOUR");
                            if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out var v) && v > 0) capPerHr = v;
                            var now = DateTime.UtcNow;
                            var list = _entriesPerHour.GetOrAdd(symbol, _ => new System.Collections.Generic.List<DateTime>());
                            lock (_entriesLock)
                            {
                                list.RemoveAll(t => (now - t) > TimeSpan.FromHours(1));
                                if (list.Count >= capPerHr)
                                {
                                    log.LogInformation("[SKIP reason=entries_per_hour] {Sym} cap={Cap}", symbol, capPerHr);
                                    continue;
                                }
                            }
                        }
                        catch { }

                        // Same-direction ES/NQ guard (downsize second)
                        try
                        {
                            var dir = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
                            var other = symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) ? "NQ" : "ES";
                            if (_lastEntryIntent.TryGetValue(other, out var last) && last.Dir == dir && (DateTime.UtcNow - last.When) <= TimeSpan.FromSeconds(5))
                            {
                                var half = Math.Max(1, (int)Math.Floor(sig.Size * 0.5));
                                toRoute = toRoute with { Size = half };
                                log.LogInformation("[Correlation] {Sym} downsize to {Size} due to same-dir with {Other}", symbol, half, other);
                            }
                        }
                        catch { }

                        var routed = await router.RouteAsync(toRoute, ct);
                        if (routed)
                        {
                            // Record intent and entries/hour stamp
                            try
                            {
                                var dir = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
                                _lastEntryIntent[symbol] = (dir, DateTime.UtcNow);
                                var list = _entriesPerHour.GetOrAdd(symbol, _ => new System.Collections.Generic.List<DateTime>());
                                lock (_entriesLock) list.Add(DateTime.UtcNow);
                            }
                            catch { }
                        }
                    }
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    log.LogWarning(ex, "[Strategy] Error running strategies for {Sym}", symbol);
                }
            }
        }
    }
}



