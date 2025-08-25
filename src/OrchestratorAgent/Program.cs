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

namespace OrchestratorAgent
{
    public static class Program
    {
        public static async Task Main(string[] args)
        {
            // Ensure invariant culture for all parsing/logging regardless of OS locale
            System.Globalization.CultureInfo.DefaultThreadCurrentCulture = System.Globalization.CultureInfo.InvariantCulture;
            System.Globalization.CultureInfo.DefaultThreadCurrentUICulture = System.Globalization.CultureInfo.InvariantCulture;

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
            string symbol = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL") ?? "ES";

            // Load credentials
            string? jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            string? userName = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
            string? apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
            long accountId = long.TryParse(Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID"), out var id) ? id : 0L;

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

                    // Wire Market hub for real-time quotes/trades (two contracts)
                    var market1 = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), jwtCache.GetAsync);
                    var market2 = new MarketHubClient(loggerFactory.CreateLogger<MarketHubClient>(), jwtCache.GetAsync);
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

                    // ===== Positions wiring =====
                    var apiClient = new ApiClient(http, loggerFactory.CreateLogger<ApiClient>(), apiBase);
                    var posTracker = new PositionTracker(log, accountId);
                    // Subscribe to user hub events
                    userHub.OnPosition += posTracker.OnPosition;
                    userHub.OnTrade += posTracker.OnTrade;
                    // Feed market trades for last price updates
                    market1.OnTrade += (_, json) => posTracker.OnMarketTrade(json);
                    market2.OnTrade += (_, json) => posTracker.OnMarketTrade(json);
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
                    // Map symbols to contract IDs
                    var contractIds = new System.Collections.Generic.Dictionary<string, string>
                    {
                        ["ES"] = "CON.F.US.EP.U25",
                        ["NQ"] = "CON.F.US.ENQ.U25"
                    };
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
                        var esId = contractIds["ES"]; var nqId = contractIds["NQ"];
                        market1.OnQuote += (_, __) => status.Set($"last.quote.{esId}", DateTimeOffset.UtcNow);
                        market1.OnTrade += (_, __) => status.Set($"last.trade.{esId}", DateTimeOffset.UtcNow);
                        market2.OnQuote += (_, __) => status.Set($"last.quote.{nqId}", DateTimeOffset.UtcNow);
                        market2.OnTrade += (_, __) => status.Set($"last.trade.{nqId}", DateTimeOffset.UtcNow);
                        // bars will be recorded in OnBar handlers below
                    }
                    catch { }

                    // Aggregators and recent bars per symbol
                    var bars = new System.Collections.Generic.Dictionary<string, System.Collections.Generic.List<BotCore.Models.Bar>>
                    {
                        ["ES"] = new System.Collections.Generic.List<BotCore.Models.Bar>(),
                        ["NQ"] = new System.Collections.Generic.List<BotCore.Models.Bar>()
                    };
                    var aggES = new BotCore.BarAggregator(60) { Symbol = "ES" };
                    var aggNQ = new BotCore.BarAggregator(60) { Symbol = "NQ" };

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
                    // Sync env LIVE_ORDERS with mode
                    void SyncLiveEnv()
                    {
                        Environment.SetEnvironmentVariable("LIVE_ORDERS", mode.IsLive ? "1" : "0");
                        log.LogInformation("MODE => {Mode}", mode.IsLive ? "LIVE" : "SHADOW");
                    }
                    SyncLiveEnv();
                    mode.OnChange += _ => SyncLiveEnv();

                    // Expose health with mode and manual overrides
                    OrchestratorAgent.Health.HealthzServer.StartWithMode(pfService, dst, mode, symbol, "http://127.0.0.1:18080/", cts.Token);

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

                    // Start autopilot loop
                    if (autoGoLive)
                    {
                        var notifier = new OrchestratorAgent.Infra.Notifier();
                        var nwrap = new NotifierAdapter(notifier);
                        var pilot = new OrchestratorAgent.Ops.AutoPilot(pfService, mode, stats, nwrap, symbol, minHealthy, demoteOnBad, TimeSpan.FromMinutes(dryMin));
                        _ = pilot.RunAsync(cts.Token);
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

                    // Autopilot controls LIVE/DRY via ModeController -> LIVE_ORDERS sync. Do an initial health check for logging.
                    try
                    {
                        var initial = await pfService.RunAsync(symbol, cts.Token);
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
                            var esId = contractIds["ES"]; status.Set($"last.bar.{esId}", DateTimeOffset.UtcNow);
                            market1.RecordBarSeen(esId);
                        }
                        catch { }
                        bars["ES"].Add(bar);
                        await RunStrategiesFor("ES", bar, bars["ES"], accountId, contractIds["ES"], risk, levels, router, log, cts.Token);
                    };
                    aggNQ.OnBar += async bar =>
                    {
                        status.Set("last.bar", DateTimeOffset.UtcNow);
                        // per-contract bar stamp for preflight ingest age
                        try
                        {
                            var nqId = contractIds["NQ"]; status.Set($"last.bar.{nqId}", DateTimeOffset.UtcNow);
                            market2.RecordBarSeen(nqId);
                        }
                        catch { }
                        bars["NQ"].Add(bar);
                        await RunStrategiesFor("NQ", bar, bars["NQ"], accountId, contractIds["NQ"], risk, levels, router, log, cts.Token);
                    };

                    // Feed market data → aggregators
                    market1.OnTrade += (_, json) => aggES.OnTrade(json);
                    market1.OnQuote += (_, json) => aggES.OnQuote(json);
                    market2.OnTrade += (_, json) => aggNQ.OnTrade(json);
                    market2.OnQuote += (_, json) => aggNQ.OnQuote(json);

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
                        var liveEnv = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (!liveEnv)
                        {
                            log.LogDebug("SHADOW: {Strat} {Sym} {Side} @{Px}", sig.StrategyId, symbol, sig.Side, sig.Entry);
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



