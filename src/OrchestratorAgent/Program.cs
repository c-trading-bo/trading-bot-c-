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

namespace OrchestratorAgent
{
    public static class Program
    {
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
                b.AddConsole();
                b.SetMinimumLevel(LogLevel.Information);
                if (concise)
                {
                    b.AddFilter("Microsoft", LogLevel.Warning);
                    b.AddFilter("System", LogLevel.Warning);
                    b.AddFilter("Microsoft.AspNetCore.SignalR", LogLevel.Warning);
                    b.AddFilter("Microsoft.AspNetCore.Http.Connections", LogLevel.Warning);
                }
            });
            var log = loggerFactory.CreateLogger("Orchestrator");

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
                    market1.OnQuote += (_, json) => { var ts = TryGetQuoteLastUpdated(json) ?? DateTimeOffset.UtcNow; status.Set("last.quote", DateTimeOffset.UtcNow); status.Set("last.quote.updated", ts); };
                    if (enableNq && market2 != null) market2.OnQuote += (_, json) => { var ts = TryGetQuoteLastUpdated(json) ?? DateTimeOffset.UtcNow; status.Set("last.quote", DateTimeOffset.UtcNow); status.Set("last.quote.updated", ts); };
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
                    market1.OnTrade += (_, json) => posTracker.OnMarketTrade(json);
                    if (enableNq && market2 != null) market2.OnTrade += (_, json) => posTracker.OnMarketTrade(json);
                    // Seed from REST
                    await posTracker.SeedFromRestAsync(apiClient, accountId, cts.Token);
                    // Publish snapshot periodically to status
                    _ = Task.Run(async () =>
                    {
                        while (!cts.IsCancellationRequested)
                        {
                            try
                            {
                                foreach (var kv in posTracker.Snapshot())
                                {
                                    var sym = kv.Key;
                                    var st = kv.Value;
                                    status.Set($"pos.{sym}.qty", st.Qty);
                                    status.Set($"pos.{sym}.avg", st.AvgPrice);
                                    status.Set($"pos.{sym}.upnl", st.UnrealizedUsd);
                                    status.Set($"pos.{sym}.rpnl", st.RealizedUsd);
                                }
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

                    // Also record per-contract last quote/trade/bar timestamps for /preflight
                    try
                    {
                        var esId = contractIds[esRoot];
                        market1.OnQuote += (_, json) => { var ts = TryGetQuoteLastUpdated(json) ?? DateTimeOffset.UtcNow; status.Set($"last.quote.{esId}", DateTimeOffset.UtcNow); status.Set($"last.quote.updated.{esId}", ts); };
                        market1.OnTrade += (_, __) => status.Set($"last.trade.{esId}", DateTimeOffset.UtcNow);
                        if (enableNq && market2 != null)
                        {
                            var nqId = contractIds[nqRoot];
                            market2.OnQuote += (_, json) => { var ts = TryGetQuoteLastUpdated(json) ?? DateTimeOffset.UtcNow; status.Set($"last.quote.{nqId}", DateTimeOffset.UtcNow); status.Set($"last.quote.updated.{nqId}", ts); };
                            market2.OnTrade += (_, __) => status.Set($"last.trade.{nqId}", DateTimeOffset.UtcNow);
                        }
                        // bars will be recorded in OnBar handlers below
                    }
                    catch { }

                    // Aggregators and recent bars per symbol
                    int barSeconds = int.TryParse(Environment.GetEnvironmentVariable("TOPSTEPX_BAR_SECONDS") ?? Environment.GetEnvironmentVariable("BAR_SECONDS"), out var bs) ? Math.Max(5, bs) : 60;
                    var bars = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<BotCore.Models.Bar>>
                    {
                        [esRoot] = new System.Collections.Generic.List<BotCore.Models.Bar>()
                    };
                    if (enableNq) bars[nqRoot] = new System.Collections.Generic.List<BotCore.Models.Bar>();
                    var aggES = new BotCore.BarAggregator(barSeconds) { Symbol = esRoot };
                    BotCore.BarAggregator? aggNQ = enableNq ? new BotCore.BarAggregator(barSeconds) { Symbol = nqRoot } : null;

                    // Strategy prerequisites
                    var risk = new BotCore.Risk.RiskEngine();
                    var levels = new BotCore.Models.Levels();
                    bool live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? string.Empty)
                                .Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    var router = new SimpleOrderRouter(http, jwtCache.GetAsync, log, live);

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
                    // Sync env LIVE_ORDERS with mode
                    void LogMode()
                    {
                        log.LogInformation("MODE => {Mode}", mode.IsLive ? "LIVE" : "SHADOW");
                    }
                    LogMode();
                    mode.OnChange += _ => LogMode();

                                        // One-time concise startup summary
                                        try
                                        {
                                            var symbolsSummary = string.Join(", ", contractIds.Select(kv => $"{kv.Key}:{kv.Value}"));
                                            log.LogInformation("Startup Summary => Account={AccountId} Mode={Mode} LiveOrders={Live} AutoGoLive={AutoGoLive} DryRunMin={DryMin} MinHealthy={MinHealthy} StickyLive={Sticky} Symbols=[{Symbols}]",
                                                accountId,
                                                mode.IsLive ? "LIVE" : "SHADOW",
                                                live,
                                                autoGoLive,
                                                dryMin,
                                                minHealthy,
                                                stickyLive,
                                                symbolsSummary);
                                        }
                                        catch { }

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
                                await Task.Delay(TimeSpan.FromMinutes(1), cts.Token);
                            }
                            catch (OperationCanceledException) { }
                            catch { }
                        }
                    }, cts.Token);

                    // Start autopilot loop (with lease requirement)
                    if (autoGoLive)
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
                                        mode.Set(OrchestratorAgent.Ops.TradeMode.Live);
                                        log.LogInformation("PROMOTE → LIVE (okStreak={okStreak}) lease={holder}", okStreak, liveLease.HolderId);
                                        await notifier.Info($"PROMOTE → LIVE (lease={liveLease.HolderId})");
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

                    // On new bar, run strategies and (optionally) route orders
                    aggES.OnBar += async bar =>
                    {
                        status.Set("last.bar", DateTimeOffset.UtcNow);
                        // per-contract bar stamp for preflight ingest age
                        try
                        {
                            var esId = contractIds[esRoot]; status.Set($"last.bar.{esId}", DateTimeOffset.UtcNow);
                            market1.RecordBarSeen(esId);
                        }
                        catch { }
                        bars[esRoot].Add(bar);
                        await RunStrategiesFor(esRoot, bar, bars[esRoot], accountId, contractIds[esRoot], risk, levels, router, log, appState, liveLease, cts.Token);
                    };
                    if (enableNq && aggNQ != null && market2 != null)
                    {
                        aggNQ.OnBar += async bar =>
                        {
                            status.Set("last.bar", DateTimeOffset.UtcNow);
                            // per-contract bar stamp for preflight ingest age
                            try
                            {
                                var nqId = contractIds[nqRoot]; status.Set($"last.bar.{nqId}", DateTimeOffset.UtcNow);
                                market2.RecordBarSeen(nqId);
                            }
                            catch { }
                            bars[nqRoot].Add(bar);
                            await RunStrategiesFor(nqRoot, bar, bars[nqRoot], accountId, contractIds[nqRoot], risk, levels, router, log, appState, liveLease, cts.Token);
                        };
                    }

                    // Feed market data → aggregators
                    market1.OnTrade += (_, json) => aggES.OnTrade(json);
                    market1.OnQuote += (_, json) => aggES.OnQuote(json);
                    if (enableNq && market2 != null && aggNQ != null)
                    {
                        market2.OnTrade += (_, json) => aggNQ.OnTrade(json);
                        market2.OnQuote += (_, json) => aggNQ.OnQuote(json);
                    }

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
                                    foreach (var sym in roots)
                                    {
                                        var qty = status.Get<int>($"pos.{sym}.qty");
                                        var avg = status.Get<decimal?>($"pos.{sym}.avg") ?? 0m;
                                        var upnl = status.Get<decimal?>($"pos.{sym}.upnl") ?? 0m;
                                        var rpnl = status.Get<decimal?>($"pos.{sym}.rpnl") ?? 0m;
                                        string state = qty != 0
                                            ? $"IN TRADE {(qty > 0 ? "LONG" : "SHORT")} x{Math.Abs(qty)} @ {avg:F2} uPnL {upnl:F2} rPnL {rpnl:F2}"
                                            : "Looking…";
                                        log.LogInformation($"[{sym}] Signals=14 | {state} | Q:{(qAge >= 0 ? qAge.ToString() : "-")}s B:{(bAge >= 0 ? bAge.ToString() : "-")}s{(paused ? " PAUSED" : string.Empty)}");
                                    }
                                }
                                catch { }
                                try { await Task.Delay(TimeSpan.FromSeconds(tickSec), cts.Token); } catch { }
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
                ILogger log,
                OrchestratorAgent.Ops.AppState appState,
                OrchestratorAgent.Ops.LiveLease liveLease,
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

                    foreach (var sig in signals)
                    {
                        log.LogInformation("[Strategy] {Sym} {StrategyId} {Side} @ {Entry} (stop {Stop}, t1 {Target}) size {Size} expR {ExpR}",
                            symbol, sig.StrategyId, sig.Side, sig.Entry, sig.Stop, sig.Target, sig.Size, sig.ExpR);

                        // Drain gate: block new parent entries when draining
                        if (appState.DrainMode)
                        {
                            log.LogInformation("DRAIN: skip new parent {sym} {side} @{px}", symbol, sig.Side, sig.Entry);
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

                        await router.RouteAsync(sig, ct);
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



