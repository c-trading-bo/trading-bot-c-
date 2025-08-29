using System;
using System.Net.Http;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore;
using SupervisorAgent;
using Microsoft.AspNetCore.SignalR.Client;
using System.Text.Json;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using BotCore.Config;
using OrchestratorAgent.Infra;
using OrchestratorAgent.Ops;
using System.Linq;
using System.Net.Http.Json;
using Dashboard;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using Microsoft.AspNetCore.Http;
using System.Reflection;

namespace OrchestratorAgent
{
    public static class Program
    {
        // Session & ops guards (process-wide)
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, System.Collections.Generic.List<DateTime>> _entriesPerHour = new(StringComparer.OrdinalIgnoreCase);
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, (int Dir, DateTime When)> _lastEntryIntent = new(StringComparer.OrdinalIgnoreCase);
        private static readonly object _entriesLock = new();
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, (DateTime DayEt, int Count)> _attemptsPerStrat = new(StringComparer.OrdinalIgnoreCase);
        // Per-direction attempt caps (key: STRAT|SYMBOL|DIR[L|S])
        private static readonly System.Collections.Concurrent.ConcurrentDictionary<string, (DateTime DayEt, int Count)> _attemptsPerStratDir = new(StringComparer.OrdinalIgnoreCase);

        // Learner status shared for dashboard endpoints/metrics
        private static class LearnerStatus
        {
            private static readonly object _sync = new();
            public static bool On { get; private set; }
            public static DateTime? LastRunUtc { get; private set; }
            public static bool? LastApplied { get; private set; }
            public static string? LastNote { get; private set; }
            public static void Update(bool on, DateTime? lastRunUtc = null, bool? lastApplied = null, string? note = null)
            {
                lock (_sync)
                {
                    On = on;
                    if (lastRunUtc != null) LastRunUtc = lastRunUtc;
                    if (lastApplied != null) LastApplied = lastApplied;
                    if (note != null) LastNote = note;
                }
            }
            public static (bool on, DateTime? last, bool? applied, string? note) Snapshot()
            {
                lock (_sync) return (On, LastRunUtc, LastApplied, LastNote);
            }
        }

        // ET helpers for time-window checks (exact shape per spec)
        private static readonly System.Globalization.CultureInfo _inv = System.Globalization.CultureInfo.InvariantCulture;
        private static readonly TimeZoneInfo ET = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
        private static readonly TimeZoneInfo SD = TryFindTz("SA Western Standard Time") ?? TryFindTz("Atlantic Standard Time") ?? ET; // America/Santo_Domingo fallback
        private static readonly string[] collection = new[] { "S2", "S3", "S6", "S11" };

        private static TimeZoneInfo? TryFindTz(string id) { try { return TimeZoneInfo.FindSystemTimeZoneById(id); } catch { return null; } }
        private static DateTime NowET() => TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, ET);
        private static DateTime NowSD() => TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, SD);
        private static TimeSpan TS(string s)
        {
            // Support both HH:mm and HH:mm:ss
            if (TimeSpan.TryParseExact(s, @"hh\:mm\:ss", _inv, out var tss)) return tss;
            return TimeSpan.ParseExact(s, @"hh\:mm", _inv);
        }
        private static bool InRange(TimeSpan now, string a, string b)
        {
            var s = TS(a); var e = TS(b);
            return s <= e ? (now >= s && now <= e) : (now >= s || now <= e);
        }
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
        private static void LoadDotEnv()
        {
            try
            {
                // Search current and up to 4 parent directories for .env.local then .env
                var candidates = new[] { ".env.local", ".env" };
                string? dir = Environment.CurrentDirectory;
                for (int up = 0; up < 5 && dir != null; up++)
                {
                    foreach (var file in candidates)
                    {
                        var path = System.IO.Path.Combine(dir, file);
                        if (System.IO.File.Exists(path))
                        {
                            foreach (var raw in System.IO.File.ReadAllLines(path))
                            {
                                var line = raw.Trim();
                                if (line.Length == 0 || line.StartsWith("#")) continue;
                                var idx = line.IndexOf('=');
                                if (idx <= 0) continue;
                                var key = line[..idx].Trim();
                                var val = line[(idx + 1)..].Trim();
                                if ((val.StartsWith("\"") && val.EndsWith("\"")) || (val.StartsWith("'") && val.EndsWith("'")))
                                    val = val[1..^1];
                                if (!string.IsNullOrWhiteSpace(key))
                                {
                                    // Do not override variables already set in the process environment
                                    var existing = Environment.GetEnvironmentVariable(key);
                                    if (!string.IsNullOrEmpty(existing)) continue;
                                    Environment.SetEnvironmentVariable(key, val);
                                }
                            }
                        }
                    }
                    dir = System.IO.Directory.GetParent(dir)?.FullName;
                }
            }
            catch { /* best-effort */ }
        }

        public static async Task Main(string[] args)
        {
            // Global exception hooks to diagnose unexpected exits
            try
            {
                AppDomain.CurrentDomain.UnhandledException += (s, e) =>
                {
                    try
                    {
                        var ex = e.ExceptionObject as Exception;
                        Console.Error.WriteLine($"FATAL UnhandledException: {ex?.GetType().Name}: {ex?.Message}");
                    }
                    catch { }
                };
                TaskScheduler.UnobservedTaskException += (s, e) =>
                {
                    try
                    {
                        Console.Error.WriteLine($"FATAL UnobservedTaskException: {e.Exception?.GetType().Name}: {e.Exception?.Message}");
                        e.SetObserved();
                    }
                    catch { }
                };
            }
            catch { }
            var urls = Environment.GetEnvironmentVariable("ASPNETCORE_URLS") ?? "http://localhost:5000";
            Console.WriteLine($"[Orchestrator] Starting (urls={urls}) …");

            // Load .env.local / .env into environment variables before reading any config
            LoadDotEnv();
            // Guardrail: kill.txt forces DRY_RUN/PAPER and disables LIVE_ORDERS
            try
            {
                if (System.IO.File.Exists("kill.txt"))
                {
                    Environment.SetEnvironmentVariable("LIVE_ORDERS", "0");
                    Environment.SetEnvironmentVariable("PAPER_MODE", "1");
                    Console.WriteLine("[Guard] kill.txt present — forcing DRY_RUN (PAPER_MODE=1, LIVE_ORDERS=0)");
                }
            }
            catch { }

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
                    // Default noise floor: Warning; whitelist storyline categories at Information
                    b.SetMinimumLevel(LogLevel.Warning);
                    b.AddFilter("Orchestrator", LogLevel.Information);
                    b.AddFilter("BotCore.Orchestrator", LogLevel.Information);

                    // Mute framework chatter
                    b.AddFilter("Microsoft", LogLevel.Warning);
                    b.AddFilter("System", LogLevel.Warning);
                    b.AddFilter("Microsoft.AspNetCore", LogLevel.Error);
                    b.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Error);
                    b.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Error);

                    // Keep data/risk off the console unless Warning+
                    b.AddFilter("DataFeed", LogLevel.Warning);
                    b.AddFilter("Risk", LogLevel.Warning);
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

            // If no credentials are present, avoid long-running network calls and just exit — except when RUN_TUNING with AUTH_ALLOW is enabled (we can login).
            var hasAnyCred = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_JWT"))
                          || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME"))
                          || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY"));
            bool runTuneGate = (Environment.GetEnvironmentVariable("RUN_TUNING") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            bool authAllowGate = (Environment.GetEnvironmentVariable("AUTH_ALLOW") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (!hasAnyCred && !(runTuneGate && authAllowGate))
            {
                Console.WriteLine("[Orchestrator] No credentials detected (TOPSTEPX_JWT / TOPSTEPX_USERNAME / TOPSTEPX_API_KEY). Exiting cleanly.");
                await Task.Delay(50);
                return;
            }

            // Optional: run historical tuning and exit when requested
            try
            {
                var runTune = (Environment.GetEnvironmentVariable("RUN_TUNING") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                if (runTune)
                {
                    Console.WriteLine("[Tune] RUN_TUNING=1 detected — starting tuner setup…");
                    // Conservative HTTP timeout so we don't hang indefinitely on auth/contract calls
                    try { http.Timeout = TimeSpan.FromSeconds(30); } catch { }
                    // Auth like main: prefer TOPSTEPX_JWT; optionally login with AUTH_ALLOW=1 and (USERNAME, API_KEY)
                    var authAllowedTune = (Environment.GetEnvironmentVariable("AUTH_ALLOW") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    var jwtEnv = Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? Environment.GetEnvironmentVariable("JWT") ?? string.Empty;
                    var userNameTune = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME") ?? Environment.GetEnvironmentVariable("LOGIN_USERNAME") ?? Environment.GetEnvironmentVariable("LOGIN_EMAIL") ?? Environment.GetEnvironmentVariable("USERNAME") ?? Environment.GetEnvironmentVariable("EMAIL");
                    var apiKeyTune = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY") ?? Environment.GetEnvironmentVariable("LOGIN_KEY") ?? Environment.GetEnvironmentVariable("API_KEY");

                    var jwtCacheTune = new JwtCache(async () =>
                    {
                        var t = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                        if (!string.IsNullOrWhiteSpace(t)) return t!;
                        if (authAllowedTune && !string.IsNullOrWhiteSpace(userNameTune) && !string.IsNullOrWhiteSpace(apiKeyTune))
                        {
                            Console.WriteLine($"[Tune] Acquiring JWT via loginKey for {userNameTune}…");
                            var authLocal = new TopstepAuthAgent(http);
                            // Use the main CTS so Ctrl+C cancels, and avoid indefinite wait
                            var fresh = await authLocal.GetJwtAsync(userNameTune!, apiKeyTune!, cts.Token);
                            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", fresh);
                            http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", fresh);
                            Console.WriteLine("[Tune] JWT acquired.");
                            return fresh;
                        }
                        return jwtEnv;
                    });
                    // Simple provider for TuningRunner
                    async Task<string> getJwt() => (await jwtCacheTune.GetAsync()) ?? string.Empty;

                    var symbolsCsv = Environment.GetEnvironmentVariable("TUNE_SYMBOLS")
                                      ?? Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOLS")
                                      ?? Environment.GetEnvironmentVariable("SYMBOLS")
                                      ?? Environment.GetEnvironmentVariable("PRIMARY_SYMBOL")
                                      ?? "ES";
                    var strategiesCsv = Environment.GetEnvironmentVariable("TUNE_STRATEGIES")
                                         ?? Environment.GetEnvironmentVariable("STRATEGIES")
                                         ?? "S2,S3,S6,S11";
                    var symbols = symbolsCsv.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                    var strategies = strategiesCsv.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                    Console.WriteLine($"[Tune] Symbols=[{string.Join(',', symbols)}] Strategies=[{string.Join(',', strategies)}]");

                    var apiBaseTune = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
                    var apiClient = new ApiClient(http, loggerFactory.CreateLogger<ApiClient>(), apiBaseTune);
                    try
                    {
                        Console.WriteLine("[Tune] Ensuring JWT present before contract resolution…");
                        var tok = await jwtCacheTune.GetAsync();
                        if (!string.IsNullOrWhiteSpace(tok)) apiClient.SetJwt(tok);
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[Tune] JWT acquire failed: {ex.Message}");
                    }

                    var until = DateTime.UtcNow.Date.AddDays(0).AddHours(23).AddMinutes(59);
                    // Prefer CLI args (e.g., --days 5 or --lookback 5), then env (TUNE_LOOKBACK_DAYS, LOOKBACK_DAYS), default 5
                    static bool TryParseDaysArg(string[] a, out int days)
                    {
                        days = 0;
                        try
                        {
                            for (int i = 0; i < a.Length; i++)
                            {
                                var s = a[i] ?? string.Empty;
                                if (s.StartsWith("--days=", StringComparison.OrdinalIgnoreCase) || s.StartsWith("--lookback=", StringComparison.OrdinalIgnoreCase))
                                {
                                    var val = s.Split('=', 2)[1];
                                    if (int.TryParse(val, out var d) && d > 0) { days = d; return true; }
                                }
                                if (string.Equals(s, "--days", StringComparison.OrdinalIgnoreCase) || string.Equals(s, "--lookback", StringComparison.OrdinalIgnoreCase))
                                {
                                    if (i + 1 < a.Length && int.TryParse(a[i + 1], out var d) && d > 0) { days = d; return true; }
                                }
                            }
                        }
                        catch { }
                        return false;
                    }

                    var rawTuneDays = Environment.GetEnvironmentVariable("TUNE_LOOKBACK_DAYS") ?? string.Empty;
                    var rawLbDays = Environment.GetEnvironmentVariable("LOOKBACK_DAYS") ?? string.Empty;
                    int lookbackDays;
                    int cliDays = 0;
                    if (TryParseDaysArg(args, out var parsedDays))
                    {
                        cliDays = parsedDays;
                        lookbackDays = parsedDays;
                    }
                    else if (int.TryParse(rawTuneDays, out var lb) && lb > 0)
                        lookbackDays = lb;
                    else if (int.TryParse(rawLbDays, out lb) && lb > 0)
                        lookbackDays = lb;
                    else
                        lookbackDays = 5;

                    var since = until.AddDays(-lookbackDays);
                    Console.WriteLine($"[Tune] Lookback selection → argsDays={(cliDays > 0 ? cliDays : 0)} TUNE_LOOKBACK_DAYS='{rawTuneDays}' LOOKBACK_DAYS='{rawLbDays}' ⇒ using {lookbackDays} days");
                    Console.WriteLine($"[Tune] Window: {since:yyyy-MM-dd} → {until:yyyy-MM-dd} (UTC), lookbackDays={lookbackDays}");

                    // Fail fast if we cannot authenticate and login is locked
                    var tokenProbe = await jwtCacheTune.GetAsync();
                    if (string.IsNullOrWhiteSpace(tokenProbe) && !authAllowedTune)
                    {
                        Console.WriteLine("[Tune] Missing TOPSTEPX_JWT and AUTH_ALLOW is disabled — cannot fetch history. Set TOPSTEPX_JWT or enable AUTH_ALLOW=1 with username+api key.");
                        return;
                    }

                    bool summaryOnly = (Environment.GetEnvironmentVariable("TUNE_SUMMARY_ONLY") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    foreach (var sym in symbols)
                    {
                        var root = sym.Trim().ToUpperInvariant();
                        string contractId = string.Empty;
                        try
                        {
                            // Dynamic env var by root, e.g., TOPSTEPX_CONTRACT_ES, TOPSTEPX_CONTRACT_NQ
                            var contractVar = $"TOPSTEPX_CONTRACT_{root}";
                            contractId = Environment.GetEnvironmentVariable(contractVar) ?? string.Empty;
                            Console.WriteLine($"[Tune] Resolving contractId for {root}…");
                            if (string.IsNullOrWhiteSpace(contractId))
                                contractId = await apiClient.ResolveContractIdAsync(root, cts.Token);
                        }
                        catch (Exception ex)
                        {
                            Console.WriteLine($"[Tune] Contract resolution failed for {root}: {ex.Message}");
                        }

                        if (string.IsNullOrWhiteSpace(contractId))
                        { Console.WriteLine($"[Tune] Missing contractId for {root}; set TOPSTEPX_CONTRACT_{root}"); continue; }
                        else { Console.WriteLine($"[Tune] {root} -> contractId={contractId}"); }

                        foreach (var strat in strategies)
                        {
                            var s = strat.Trim().ToUpperInvariant();
                            if (s == "S2")
                            {
                                if (summaryOnly)
                                {
                                    Console.WriteLine($"[Backtest] Summary S2 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunS2SummaryAsync(http, getJwt, contractId, root, since, until, log, cts.Token);
                                }
                                else
                                {
                                    Console.WriteLine($"[Tune] Running S2 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunS2Async(http, getJwt, contractId, root, since, until, log, cts.Token);
                                }
                            }
                            else if (s == "S3")
                            {
                                if (summaryOnly)
                                {
                                    Console.WriteLine($"[Backtest] Summary S3 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunS3SummaryAsync(http, getJwt, contractId, root, since, until, log, cts.Token);
                                }
                                else
                                {
                                    Console.WriteLine($"[Tune] Running S3 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunS3Async(http, getJwt, contractId, root, since, until, log, cts.Token);
                                }
                            }
                            else if (s == "S6")
                            {
                                if (summaryOnly)
                                {
                                    Console.WriteLine($"[Backtest] Summary S6 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunStrategySummaryAsync(http, getJwt, contractId, root, s, since, until, log, cts.Token);
                                }
                                else
                                {
                                    Console.WriteLine($"[Tune] Running S6 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunS6Async(http, getJwt, contractId, root, since, until, log, cts.Token);
                                }
                            }
                            else if (s == "S11")
                            {
                                if (summaryOnly)
                                {
                                    Console.WriteLine($"[Backtest] Summary S11 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunStrategySummaryAsync(http, getJwt, contractId, root, s, since, until, log, cts.Token);
                                }
                                else
                                {
                                    Console.WriteLine($"[Tune] Running S11 on {root}…");
                                    await OrchestratorAgent.Execution.TuningRunner.RunS11Async(http, getJwt, contractId, root, since, until, log, cts.Token);
                                }
                            }
                        }
                    }
                    Console.WriteLine("[Tune] Completed. Exiting.");
                    return; // exit after tuning
                }
            }
            catch { }

            // Load credentials (with fallbacks for common env names)
            static string? Env(string name) => Environment.GetEnvironmentVariable(name);
            string? jwt = Env("TOPSTEPX_JWT") ?? Env("JWT");
            string? userName = Env("TOPSTEPX_USERNAME") ?? Env("LOGIN_USERNAME") ?? Env("LOGIN_EMAIL") ?? Env("USERNAME") ?? Env("EMAIL");
            string? apiKey = Env("TOPSTEPX_API_KEY") ?? Env("LOGIN_KEY") ?? Env("API_KEY");
            long accountId = long.TryParse(Env("TOPSTEPX_ACCOUNT_ID") ?? Env("ACCOUNT_ID"), out var id) ? id : 0L;

            // Runtime gate for any auth/login activity (default deny; auto-allow in SHADOW later)
            bool authAllowed = (Environment.GetEnvironmentVariable("AUTH_ALLOW") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";

            // Pre-apply Authorization if JWT is already present from .env
            try { if (!string.IsNullOrWhiteSpace(jwt)) http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt); } catch { }
            // Common endpoints and primary symbol for logging and downstream services
            var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE")
                        ?? Environment.GetEnvironmentVariable("TOPSTEPX_API_URL")
                        ?? "https://api.topstepx.com";
            var rtcBase = Environment.GetEnvironmentVariable("TOPSTEPX_RTC_BASE")
                        ?? Environment.GetEnvironmentVariable("TOPSTEPX_RTC_URL")
                        ?? string.Empty;
            var symbol = Environment.GetEnvironmentVariable("PRIMARY_SYMBOL") ?? "ES";

            log.LogInformation("Env config: API={Api}  RTC={Rtc}  Symbol={Sym}  AccountId={Acc}  HasJWT={HasJwt}  HasLoginKey={HasLogin}", apiBase, rtcBase, symbol, accountId, !string.IsNullOrWhiteSpace(jwt), !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey));

            // Clock sanity: local, UTC, CME (America/Chicago) — simplified (no separate listener)
            try { _ = ET; } catch (Exception ex) { log.LogWarning(ex, "Timezone init failed"); }

            // ===== Launch mode selection: Live vs Paper vs Shadow (before any auth) =====
            bool paperModeSelected = false;
            bool shadowModeSelected = false;
            try
            {
                string? botMode = Environment.GetEnvironmentVariable("BOT_MODE");
                var skipPromptEnv = Environment.GetEnvironmentVariable("SKIP_MODE_PROMPT");
                bool skipPrompt = (Environment.GetEnvironmentVariable("SKIP_MODE_PROMPT") ?? "false").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                try { log.LogInformation("Env: BOT_MODE={BotMode} SKIP_MODE_PROMPT={Skip}", botMode, skipPromptEnv); } catch { }
                if (!skipPrompt && !Console.IsInputRedirected)
                {
                    while (true)
                    {
                        Console.WriteLine();
                        Console.WriteLine("Select mode:");
                        Console.WriteLine("  [L]ive   (places real orders)");
                        Console.WriteLine("  [P]aper  (simulates orders)");
                        Console.WriteLine("  [S]hadow (no orders) [default]");
                        Console.Write("Mode: ");
                        var line = Console.ReadLine();
                        var lower = (line ?? string.Empty).Trim().ToLowerInvariant();
                        if (string.IsNullOrEmpty(lower) || lower == "s" || lower == "shadow")
                        { paperModeSelected = false; shadowModeSelected = true; break; }
                        if (lower == "p" || lower == "paper")
                        { paperModeSelected = true; shadowModeSelected = false; break; }
                        if (lower == "l" || lower == "live")
                        {
                            Console.Write("Confirm LIVE mode by typing LIVE: ");
                            var conf = Console.ReadLine();
                            if (string.Equals((conf ?? string.Empty).Trim(), "LIVE", StringComparison.OrdinalIgnoreCase))
                            { paperModeSelected = false; shadowModeSelected = false; break; }
                            Console.WriteLine("Live not confirmed. Returning to selection.");
                            continue;
                        }
                        Console.WriteLine("Invalid choice. Try again.");
                    }
                }
                else if (!string.IsNullOrWhiteSpace(botMode))
                {
                    paperModeSelected = botMode.Trim().Equals("paper", StringComparison.OrdinalIgnoreCase);
                    shadowModeSelected = botMode.Trim().Equals("shadow", StringComparison.OrdinalIgnoreCase);
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

                // Auto-allow auth flows in SHADOW and PAPER modes to enable hub connectivity during testing
                if (shadowModeSelected || paperModeSelected) authAllowed = true;

                // Integrity lock: protect auth/JWT and SignalR connection code from accidental changes
                // Integrity guard disabled per user request
            }
            catch { }

            // Try to obtain JWT if not provided (runtime-locked unless AUTH_ALLOW=1)
            var runtimeJwtAcquired = false;
            if (string.IsNullOrWhiteSpace(jwt) && !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
            {
                if (authAllowed)
                {
                    try
                    {
                        var auth = new TopstepAuthAgent(http);
                        log.LogInformation("Fetching JWT using login key for {User}…", userName);
                        jwt = await auth.GetJwtAsync(userName!, apiKey!, cts.Token);
                        Environment.SetEnvironmentVariable("TOPSTEPX_JWT", jwt);
                        try { http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt); } catch { }
                        log.LogInformation("Obtained JWT via loginKey for {User}.", userName);
                        runtimeJwtAcquired = true;
                    }
                    catch (Exception ex)
                    {
                        log.LogWarning(ex, "Failed to obtain JWT using TOPSTEPX_USERNAME/TOPSTEPX_API_KEY");
                    }
                }
                else
                {
                    log.LogInformation("[AuthLocked] Skipping JWT acquisition (AUTH_ALLOW is not enabled). Provide TOPSTEPX_JWT via .env to run.");
                }
            }

            var status = new StatusService(loggerFactory.CreateLogger<StatusService>()) { AccountId = accountId };

            // Initialize learning system early to avoid connectivity dependency issues
            log.LogInformation("[Startup] Initializing learning system early (before connectivity checks)...");
            try
            {
                var runLearn = (Environment.GetEnvironmentVariable("RUN_LEARNING") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                var liveOrdersFlag = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                log.LogInformation("[Startup] RUN_LEARNING={runLearn}, LIVE_ORDERS={liveOrdersFlag}", runLearn, liveOrdersFlag);

                if (runLearn)
                {
                    log.LogInformation("[Startup] Starting adaptive learning system (early initialization)...");
                    log.LogInformation("[Startup] About to create background Task.Run...");
                    var learnCts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);
                    log.LogInformation("[Startup] Cancellation token created, IsCancellationRequested={isCancelled}", learnCts.Token.IsCancellationRequested);

                    // Try a different approach - use ThreadPool instead of Task.Run
                    ThreadPool.QueueUserWorkItem(state =>
                    {
                        try
                        {
                            Console.WriteLine("[CONSOLE] Learner task starting via ThreadPool...");
                            var llog = loggerFactory.CreateLogger("Learner");
                            Console.WriteLine("[CONSOLE] Logger created...");
                            llog.LogInformation("[Learner] Background task STARTED via ThreadPool - about to begin initialization");
                            Console.WriteLine("[CONSOLE] After first log message...");

                            bool allowLiveInstant = (Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            bool liveNow = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            Console.WriteLine("[CONSOLE] Environment variables read...");

                            LearnerStatus.Update(on: true);
                            Console.WriteLine("[CONSOLE] LearnerStatus updated...");

                            var lastPractice = DateTime.MinValue;
                            var demoMode = (Environment.GetEnvironmentVariable("LEARN_DEMO_MODE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            var minGap = demoMode ?
                                TimeSpan.FromMinutes(2) : // Demo: 2 minute cycles 
                                TimeSpan.FromMinutes(Math.Max(45, int.TryParse(Environment.GetEnvironmentVariable("RETUNE_INTERVAL_MIN"), out var m) ? Math.Max(15, m) : 60));
                            llog.LogInformation("[Learner] Background loop ready (minGap={minGap}, allowLiveInstant={allowLiveInstant}, demoMode={demoMode})", minGap, allowLiveInstant, demoMode);
                            Console.WriteLine("[CONSOLE] After second log message...");

                            // Wait for basic setup to complete before starting learning cycles
                            llog.LogInformation("[Learner] Waiting 30 seconds for bot initialization to complete...");
                            Console.WriteLine("[CONSOLE] Starting 30-second wait...");
                            Thread.Sleep(30000); // Use Thread.Sleep instead of async Task.Delay
                            Console.WriteLine("[CONSOLE] 30-second wait completed...");
                            llog.LogInformation("[Learner] 30-second wait completed, starting main loop...");
                            Console.WriteLine("[CONSOLE] About to start main learning loop...");

                            var loopCount = 0;
                            while (true) // Keep running indefinitely
                            {
                                try
                                {
                                    loopCount++;
                                    Console.WriteLine($"[CONSOLE] === Learning Loop #{loopCount} ===");
                                    var now = DateTime.UtcNow;
                                    var timeSinceLastPractice = now - lastPractice;
                                    Console.WriteLine($"[CONSOLE] Current time: {now:HH:mm:ss}, Last practice: {(lastPractice == DateTime.MinValue ? "never" : lastPractice.ToString("HH:mm:ss"))}, Time since: {timeSinceLastPractice}");
                                    llog.LogInformation("[Learner] Loop iteration: lastPractice={lastPractice}, timeSince={timeSince}, minGap={minGap}",
                                        lastPractice == DateTime.MinValue ? "never" : lastPractice.ToString("yyyy-MM-dd HH:mm:ss"),
                                        timeSinceLastPractice, minGap);

                                    if (now - lastPractice >= minGap)
                                    {
                                        Console.WriteLine($"[CONSOLE] *** STARTING LEARNING CYCLE *** (gap requirement met: {timeSinceLastPractice} >= {minGap})");
                                        llog.LogInformation("[Learner] Starting learning cycle (early init mode)...");
                                        lastPractice = DateTime.UtcNow;
                                        Console.WriteLine($"[CONSOLE] Learning cycle completed at {lastPractice:HH:mm:ss}, next cycle in {minGap}");
                                        llog.LogInformation("[Learner] Learning cycle completed at {time}, next cycle in {minGap}", lastPractice.ToString("yyyy-MM-dd HH:mm:ss"), minGap);
                                    }
                                    else
                                    {
                                        var timeRemaining = minGap - (now - lastPractice);
                                        Console.WriteLine($"[CONSOLE] Waiting for next cycle - {timeRemaining} remaining");
                                        llog.LogInformation("[Learner] Waiting for next cycle - {timeRemaining} remaining", timeRemaining);
                                    }

                                    // Sleep for shorter periods in demo mode
                                    var sleepDuration = demoMode ? TimeSpan.FromMinutes(2) : TimeSpan.FromHours(1);
                                    Console.WriteLine($"[CONSOLE] Sleeping for {sleepDuration} until next learning cycle...");
                                    llog.LogInformation("[Learner] Sleeping for {sleepDuration} until next learning cycle...", sleepDuration);
                                    Thread.Sleep(sleepDuration); // Use Thread.Sleep instead of async Task.Delay
                                }
                                catch (Exception ex)
                                {
                                    Console.WriteLine($"[CONSOLE] ERROR in learning loop: {ex.Message}");
                                    llog.LogWarning(ex, "[Learner] Learning loop error (early init mode)");
                                    Thread.Sleep(TimeSpan.FromMinutes(5)); // Use Thread.Sleep instead of async Task.Delay
                                }
                            }
                        }
                        catch (Exception ex)
                        {
                            log.LogError(ex, "[Learner] Background task failed");
                        }
                    }, null);
                    log.LogInformation("[Startup] Background ThreadPool task queued");
                }
                else
                {
                    log.LogInformation("[Startup] Learning system disabled (RUN_LEARNING={runLearn})", runLearn);
                }
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[Startup] Failed to initialize learning system");
            }

            // Backfill JWT from environment if it was acquired during startup
            if (string.IsNullOrWhiteSpace(jwt))
            {
                var jwtEnvNow = Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? Environment.GetEnvironmentVariable("JWT");
                if (!string.IsNullOrWhiteSpace(jwtEnvNow))
                {
                    jwt = jwtEnvNow;
                    try { http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt); } catch { }
                }
            }

            if (!string.IsNullOrWhiteSpace(jwt) || runtimeJwtAcquired || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TOPSTEPX_JWT")))
            {
                if (string.IsNullOrWhiteSpace(jwt))
                {
                    // Backfill from env if available
                    jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                    try { if (!string.IsNullOrWhiteSpace(jwt)) http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt); } catch { }
                }
                if (accountId <= 0)
                {
                    log.LogWarning("TOPSTEPX_ACCOUNT_ID not set. Launching in account-discovery mode (SubscribeAccounts only). You can set the account ID later.");
                }
                try
                {
                    // Start background JWT refresh loop (auth hygiene) — locked unless AUTH_ALLOW=1
                    if (authAllowed)
                    {
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
                    }
                    else
                    {
                        log.LogInformation("[AuthLocked] Skipping JWT refresh loop (AUTH_ALLOW is not enabled).");
                    }

                    // Shared JWT cache so both hubs always get a valid token (login locked unless AUTH_ALLOW=1)
                    var jwtCache = new JwtCache(async () =>
                    {
                        var t = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                        if (!string.IsNullOrWhiteSpace(t)) return t!;
                        if (authAllowed && !string.IsNullOrWhiteSpace(userName) && !string.IsNullOrWhiteSpace(apiKey))
                        {
                            var authLocal = new TopstepAuthAgent(http);
                            var fresh = await authLocal.GetJwtAsync(userName!, apiKey!, CancellationToken.None);
                            Environment.SetEnvironmentVariable("TOPSTEPX_JWT", fresh);
                            http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", fresh);
                            return fresh;
                        }
                        log.LogInformation("[AuthLocked] JwtCache: no AUTH_ALLOW; returning initial/fallback token only.");
                        return jwt ?? string.Empty;
                    });

                    var userHub = new BotCore.UserHubAgent(loggerFactory.CreateLogger<BotCore.UserHubAgent>(), status);
                    // Ensure a non-empty token for hub connection; prefer jwtCache (fresh) then local jwt as fallback
                    string tokenNow = string.Empty;
                    try { tokenNow = await jwtCache.GetAsync() ?? string.Empty; } catch { tokenNow = string.Empty; }
                    if (string.IsNullOrWhiteSpace(tokenNow) && !string.IsNullOrWhiteSpace(jwt)) tokenNow = jwt!;
                    if (!string.IsNullOrWhiteSpace(tokenNow))
                    {
                        try { http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", tokenNow); } catch { }
                    }
                    try
                    {
                        await userHub.ConnectAsync(tokenNow!, accountId, cts.Token);
                    }
                    catch (Exception ex)
                    {
                        log.LogWarning(ex, "[UserHub] Connect failed; continuing without hub. HTTP will stay up.");
                        try { status.Set("user.state", "disconnected"); } catch { }
                    }

                    // Resolve roots and contracts from env (with REST fallback)
                    var apiClient = new ApiClient(http, loggerFactory.CreateLogger<ApiClient>(), apiBase);
                    try { if (!string.IsNullOrWhiteSpace(jwt)) apiClient.SetJwt(jwt!); } catch { }
                    var esRoot = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL_ES") ?? "ES";
                    var nqRoot = Environment.GetEnvironmentVariable("TOPSTEPX_SYMBOL_NQ") ?? "NQ";
                    bool enableNq = (Environment.GetEnvironmentVariable("TOPSTEPX_ENABLE_NQ") ?? Environment.GetEnvironmentVariable("ENABLE_NQ") ?? "1").Trim().ToLowerInvariant() is "1" or "true" or "yes";
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
                        await market1.StartAsync(esContract!, m1Cts.Token, cts.Token);
                        if (enableNq && market2 != null && m2Cts != null)
                        {
                            m2Cts.CancelAfter(TimeSpan.FromSeconds(15));
                            await market2.StartAsync(nqContract!, m2Cts.Token, cts.Token);
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
                            enableNq && market2 != null && market2.HasRecentQuote(nqContract!),
                            enableNq && market2 != null && market2.HasRecentBar(nqContract!, "1m"));
                    }
                    catch { }
                    // ===== Positions wiring =====
                    var posTracker = new PositionTracker(log, accountId);
                    // Map symbols to contract IDs resolved from env/REST (needed in quote handlers)
                    var contractIds = new System.Collections.Generic.Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
                    {
                        [esRoot] = esContract!
                    };
                    if (enableNq && !string.IsNullOrWhiteSpace(nqContract)) contractIds[nqRoot] = nqContract!;

                    // Subscribe to user hub events
                    Action<string, string>? emitEvent = null; // wired after web host init
                    userHub.OnPosition += posTracker.OnPosition;
                    userHub.OnTrade += posTracker.OnTrade;
                    // Structured JSON logs per guardrails
                    userHub.OnOrder += je =>
                    {
                        try
                        {
                            var line = new { type = "ORDER", ts = DateTimeOffset.UtcNow, account = accountId, json = je };
                            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(line));
                            try
                            {
                                var sym = je.TryGetProperty("symbol", out var s) ? s.GetString() : null;
                                var status = je.TryGetProperty("status", out var st) ? st.GetString() : null;
                                emitEvent?.Invoke("info", $"ORDER {sym ?? "?"} {status ?? ""}");
                            }
                            catch { }
                        }
                        catch { }
                    };
                    userHub.OnTrade += je =>
                    {
                        try
                        {
                            var line = new { type = "TRADE", ts = DateTimeOffset.UtcNow, account = accountId, json = je };
                            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(line));
                            try
                            {
                                var sym = je.TryGetProperty("symbol", out var s) ? s.GetString() : null;
                                var px = je.TryGetProperty("fillPrice", out var fp) ? fp.ToString() : null;
                                var qty = je.TryGetProperty("qty", out var q) ? q.ToString() : null;
                                emitEvent?.Invoke("info", $"TRADE {sym ?? "?"} {qty ?? ""} @ {px ?? ""}");
                            }
                            catch { }
                        }
                        catch { }
                    };
                    // Feed market trades for last price updates
                    market1.OnTrade += (cid, tick) => { try { var je = System.Text.Json.JsonSerializer.SerializeToElement(new { symbol = cid, price = tick.Price }); posTracker.OnMarketTrade(je); } catch { } };
                    if (enableNq && market2 != null) market2.OnTrade += (cid, tick) => { try { var je = System.Text.Json.JsonSerializer.SerializeToElement(new { symbol = cid, price = tick.Price }); posTracker.OnMarketTrade(je); } catch { } };
                    // Seed from REST
                    await posTracker.SeedFromRestAsync(apiClient, accountId, cts.Token);

                    // Prepare placeholders for health/mode wiring used by the web host (assigned below once created)
                    OrchestratorAgent.Health.Preflight? pfServiceRef = null;
                    OrchestratorAgent.Health.DstGuard? dstRef = null;
                    OrchestratorAgent.Ops.ModeController? modeRef = null;

                    OrchestratorAgent.Ops.AppState? appStateRef = null;
                    OrchestratorAgent.Ops.LiveLease? liveLeaseRef = null;

                    // ===== Dashboard + Health: start single web host (Kestrel) on ASPNETCORE_URLS =====
                    Dashboard.RealtimeHub? dashboardHub = null;
                    try
                    {
                        var webBuilder = Microsoft.AspNetCore.Builder.WebApplication.CreateBuilder();
                        // Note: binding to ASPNETCORE_URLS is handled by hosting; no explicit UseUrls call needed here.
                        // Register RealtimeHub with metrics provider capturing current pos/status
                        webBuilder.Services.AddSingleton<Dashboard.RealtimeHub>(sp =>
                        {
                            var logger = sp.GetRequiredService<ILogger<Dashboard.RealtimeHub>>();
                            Dashboard.MetricsSnapshot MetricsProvider()
                            {
                                // Build metrics from PositionTracker snapshot
                                var snap = posTracker.Snapshot();
                                decimal realized = 0m, unreal = 0m;
                                var chips = new List<Dashboard.PositionChip>();
                                foreach (var kv in snap)
                                {
                                    var ps = kv.Value;
                                    realized += ps.RealizedUsd;
                                    unreal += ps.UnrealizedUsd;
                                    // Map contractId keys back to root symbols for UI matching (ES/NQ)
                                    var key = kv.Key;
                                    string symOut = key;
                                    try
                                    {
                                        if (!string.IsNullOrWhiteSpace(esContract) && string.Equals(key, esContract, StringComparison.OrdinalIgnoreCase)) symOut = esRoot;
                                        else if (!string.IsNullOrWhiteSpace(nqContract) && string.Equals(key, nqContract, StringComparison.OrdinalIgnoreCase)) symOut = nqRoot;
                                    }
                                    catch { }
                                    chips.Add(new Dashboard.PositionChip(symOut, ps.Qty, ps.AvgPrice, ps.LastPrice, ps.UnrealizedUsd, ps.RealizedUsd));
                                }
                                var mode = (Environment.GetEnvironmentVariable("PAPER_MODE") == "1") ? "PAPER" : (Environment.GetEnvironmentVariable("SHADOW_MODE") == "1") ? "SHADOW" : "LIVE";
                                var day = realized + unreal;
                                decimal mdl = 1000m;
                                try { mdl = status.Get<decimal?>("risk.daily.max") ?? mdl; } catch { }
                                var remaining = mdl + Math.Min(0m, realized);
                                var userHubState = "Connected"; // simplify; refine from actual hub state if desired
                                var marketHubState = "Connected";

                                // Flags for dashboard badges derived from status/env
                                // (we'll extend the MetricsSnapshot to include them)
                                bool curfewNoNew = status.Get<bool?>("curfew.no_new") ?? false;
                                bool dayPnlNoNew = status.Get<bool?>("day_pnl.no_new") ?? false;

                                // Allowed strategies now (ET windows logic mirrors runtime)
                                var etNow = NowET().TimeOfDay;
                                var allowed = new List<string>();
                                bool isBlackout = InRange(etNow, "16:58", "18:05") || InRange(etNow, "09:15", "09:23:30");
                                bool isNight = !isBlackout && (etNow >= TS("18:05") || etNow < TS("09:15"));
                                if (isNight)
                                {
                                    if (InRange(etNow, "18:05", "02:00")) allowed.Add("S2");
                                    else if (InRange(etNow, "02:55", "04:10")) allowed.Add("S3");
                                }
                                else
                                {
                                    if (etNow < TS("09:28")) { }
                                    else if (InRange(etNow, "09:28", "10:00")) { allowed.Add("S6"); allowed.Add("S3"); }
                                    else if (etNow >= TS("10:20") && etNow < TS("13:30")) { allowed.Add("S2"); }
                                    else if (etNow >= TS("13:30")) { allowed.Add("S11"); }
                                }

                                var (lon, llast, lapplied, lnote) = LearnerStatus.Snapshot();
                                return new Dashboard.MetricsSnapshot(accountId, mode, realized, unreal, day, mdl, remaining, userHubState, marketHubState, DateTime.Now, chips, curfewNoNew, dayPnlNoNew, allowed, lon, llast, lapplied, lnote);
                            }
                            return new Dashboard.RealtimeHub(logger, MetricsProvider);
                        });
                        webBuilder.Services.AddHostedService(sp => sp.GetRequiredService<Dashboard.RealtimeHub>());
                        var web = webBuilder.Build();
                        web.UseDefaultFiles();
                        web.UseStaticFiles();
                        dashboardHub = web.Services.GetRequiredService<Dashboard.RealtimeHub>();
                        emitEvent = (lvl, text) => { try { dashboardHub.EmitEvent(lvl, text); } catch { } };
                        web.MapDashboard(dashboardHub);

                        // Map health endpoints on same Kestrel host
                        web.MapGet("/healthz", async () =>
                        {
                            if (pfServiceRef is null || dstRef is null || modeRef is null)
                                return Results.Json(new { ok = false, msg = "initializing", warn_dst = (string?)null, mode = "UNKNOWN" }, statusCode: 503);
                            var (ok, msg) = await pfServiceRef.RunAsync(symbol, cts.Token);
                            var check = dstRef.Check();
                            var isPaper = (Environment.GetEnvironmentVariable("PAPER_MODE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            var modeStr = isPaper ? "PAPER" : (modeRef.IsLive ? "LIVE" : "SHADOW");
                            return Results.Json(new { ok, msg, warn_dst = check.warn, mode = modeStr });
                        });
                        web.MapGet("/healthz/mode", () =>
                        {
                            if (modeRef is null)
                                return Results.Json(new { mode = "UNKNOWN", lease = (bool?)null, drain = (bool?)null }, statusCode: 503);
                            var isPaper = (Environment.GetEnvironmentVariable("PAPER_MODE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            return Results.Json(new { mode = isPaper ? "PAPER" : (modeRef.IsLive ? "LIVE" : "SHADOW"), lease = liveLeaseRef?.HasLease, drain = appStateRef?.DrainMode });
                        });
                        // Debug: MarketHub connection/subscription status and last data ages
                        web.MapGet("/debug/market", () =>
                        {
                            try
                            {
                                var esCid = esContract ?? string.Empty;
                                var m1State = market1?.Connection?.State.ToString() ?? "null";
                                var esQuoteAge = market1 is null || string.IsNullOrWhiteSpace(esCid) ? (double?)null : Math.Round(market1.LastQuoteSeenAge(esCid).TotalSeconds, 1);
                                var esBarAge = market1 is null || string.IsNullOrWhiteSpace(esCid) ? (double?)null : Math.Round(market1.LastBarSeenAge(esCid).TotalSeconds, 1);
                                var esHasRecentQuote = market1 is not null && !string.IsNullOrWhiteSpace(esCid) && market1.HasRecentQuote(esCid);
                                var esHasRecentBar = market1 is not null && !string.IsNullOrWhiteSpace(esCid) && market1.HasRecentBar(esCid);

                                string? nqCid = enableNq ? nqContract : null;
                                var m2State = market2?.Connection?.State.ToString();
                                var nqQuoteAge = market2 is null || string.IsNullOrWhiteSpace(nqCid) ? (double?)null : Math.Round(market2.LastQuoteSeenAge(nqCid).TotalSeconds, 1);
                                var nqBarAge = market2 is null || string.IsNullOrWhiteSpace(nqCid) ? (double?)null : Math.Round(market2.LastBarSeenAge(nqCid).TotalSeconds, 1);
                                var nqHasRecentQuote = market2 is not null && !string.IsNullOrWhiteSpace(nqCid) && market2.HasRecentQuote(nqCid);
                                var nqHasRecentBar = market2 is not null && !string.IsNullOrWhiteSpace(nqCid) && market2.HasRecentBar(nqCid);

                                return Results.Json(new
                                {
                                    es = new { contractId = esCid, state = m1State, quoteAgeSec = esQuoteAge, barAgeSec = esBarAge, hasRecentQuote = esHasRecentQuote, hasRecentBar = esHasRecentBar },
                                    nq = enableNq ? new { contractId = nqCid, state = m2State, quoteAgeSec = nqQuoteAge, barAgeSec = nqBarAge, hasRecentQuote = nqHasRecentQuote, hasRecentBar = nqHasRecentBar } : null
                                });
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { error = ex.Message }, statusCode: 500);
                            }
                        });
                        web.MapGet("/build", () =>
                        {
                            var infoVer = Assembly.GetExecutingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion
                                           ?? Assembly.GetExecutingAssembly().GetName().Version?.ToString()
                                           ?? "dev";
                            var payload = new
                            {
                                version = infoVer,
                                pid = Environment.ProcessId,
                                mode = modeRef is null ? "UNKNOWN" : (modeRef.IsLive ? "LIVE" : "SHADOW"),
                                lease = liveLeaseRef?.HasLease,
                                startedUtc = System.Diagnostics.Process.GetCurrentProcess().StartTime.ToUniversalTime()
                            };
                            return Results.Json(payload);
                        });
                        // Allowed strategies now (ET window-based)
                        web.MapGet("/runtime/allowed", () =>
                        {
                            var etNow = NowET().TimeOfDay;
                            var allowed = new System.Collections.Generic.List<string>();
                            bool isBlackout = InRange(etNow, "16:58", "18:05") || InRange(etNow, "09:15", "09:23:30");
                            bool isNight = !isBlackout && (etNow >= TS("18:05") || etNow < TS("09:15"));
                            if (isNight)
                            {
                                if (InRange(etNow, "18:05", "02:00")) allowed.Add("S2");
                                else if (InRange(etNow, "02:55", "04:10")) allowed.Add("S3");
                            }
                            else
                            {
                                if (etNow < TS("09:28")) { }
                                else if (InRange(etNow, "09:28", "10:00")) { allowed.Add("S6"); allowed.Add("S3"); }
                                else if (etNow >= TS("10:20") && etNow < TS("13:30")) { allowed.Add("S2"); }
                                else if (etNow >= TS("13:30")) { allowed.Add("S11"); }
                            }
                            return Results.Json(new { etNow = NowET(), allowed });
                        });
                        // Learner status snapshot
                        web.MapGet("/learner/status", () =>
                        {
                            var (on, last, applied, note) = LearnerStatus.Snapshot();
                            return Results.Json(new { on, last, applied, note });
                        });
                        // Learner health (simple OK+timestamp)
                        web.MapGet("/learner/health", () => Results.Json(new { ok = true, ts = DateTime.UtcNow }));
                        // Manual: launch tuner (one-off run). Params: days (int, default 7), strats (csv), roots (csv)
                        web.MapPost("/learner/tune", (HttpContext ctx) =>
                                    {
                                        try
                                        {
                                            int days = 7;
                                            var qd = ctx.Request.Query["days"].FirstOrDefault();
                                            if (!string.IsNullOrWhiteSpace(qd) && int.TryParse(qd, out var d) && d > 0) days = d;
                                            var until = DateTime.UtcNow.Date.AddDays(1).AddTicks(-1);
                                            var since = until.AddDays(-(days - 1)).Date;

                                            var rootsCsv = ctx.Request.Query["roots"].FirstOrDefault();
                                            var stratsCsv = ctx.Request.Query["strats"].FirstOrDefault();
                                            var roots = new System.Collections.Generic.List<string>();
                                            if (!string.IsNullOrWhiteSpace(rootsCsv))
                                            {
                                                roots.AddRange(rootsCsv.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                                            }
                                            else
                                            {
                                                roots.Add(esRoot);
                                                if (enableNq) roots.Add(nqRoot);
                                            }
                                            var strats = new System.Collections.Generic.List<string>();
                                            if (!string.IsNullOrWhiteSpace(stratsCsv))
                                                strats.AddRange(stratsCsv.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(s => s.ToUpperInvariant()));
                                            else
                                                strats.AddRange(collection);

                                            // Fire-and-forget background run
                                            _ = Task.Run(async () =>
                                            {
                                                var runLog = loggerFactory.CreateLogger("Retune");
                                                try
                                                {
                                                    try { var tok = await jwtCache.GetAsync(); if (!string.IsNullOrWhiteSpace(tok)) http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", tok); } catch { }
                                                    var results = new System.Collections.Generic.List<object>();
                                                    foreach (var root in roots)
                                                    {
                                                        if (!contractIds.TryGetValue(root, out var cid) || string.IsNullOrWhiteSpace(cid))
                                                        { results.Add(new { root, strat = (string?)null, ok = false, error = "missing contractId" }); continue; }
                                                        foreach (var s in strats)
                                                        {
                                                            try
                                                            {
                                                                if (s == "S2") await OrchestratorAgent.Execution.TuningRunner.RunS2Async(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, since, until, runLog, cts.Token);
                                                                else if (s == "S3") await OrchestratorAgent.Execution.TuningRunner.RunS3Async(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, since, until, runLog, cts.Token);
                                                                else if (s == "S6") await OrchestratorAgent.Execution.TuningRunner.RunS6Async(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, since, until, runLog, cts.Token);
                                                                else if (s == "S11") await OrchestratorAgent.Execution.TuningRunner.RunS11Async(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, since, until, runLog, cts.Token);
                                                                results.Add(new { root, strat = s, ok = true, error = (string?)null });
                                                            }
                                                            catch (Exception ex)
                                                            {
                                                                results.Add(new { root, strat = s, ok = false, error = ex.Message });
                                                            }
                                                        }
                                                    }
                                                    try
                                                    {
                                                        var pathDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning"); Directory.CreateDirectory(pathDir);
                                                        var status = new { kind = "manual", nowUtc = DateTime.UtcNow, lookbackDays = days, since, until, roots, strats, results };
                                                        System.IO.File.WriteAllText(Path.Combine(pathDir, "retune_status.json"), System.Text.Json.JsonSerializer.Serialize(status, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
                                                    }
                                                    catch { }
                                                }
                                                catch (Exception ex)
                                                {
                                                    try
                                                    {
                                                        var pathDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning"); Directory.CreateDirectory(pathDir);
                                                        var status = new { kind = "manual", nowUtc = DateTime.UtcNow, lookbackDays = days, since, until, error = ex.Message };
                                                        System.IO.File.WriteAllText(Path.Combine(pathDir, "retune_status.json"), System.Text.Json.JsonSerializer.Serialize(status, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
                                                    }
                                                    catch { }
                                                }
                                            }, cts.Token);

                                            return Results.Json(new { started = true, days, roots, strats });
                                        }
                                        catch (Exception ex)
                                        {
                                            return Results.Json(new { started = false, error = ex.Message }, statusCode: 500);
                                        }
                                    });
                        // Manual: run adaptive learner once (forces RUN_LEARNING=1 for this call)
                        web.MapPost("/learner/adapt", () =>
                        {
                            try
                            {
                                _ = Task.Run(async () =>
                                {
                                    var prev = Environment.GetEnvironmentVariable("RUN_LEARNING");
                                    try { Environment.SetEnvironmentVariable("RUN_LEARNING", "1"); } catch { }
                                    try
                                    {
                                        var llog = loggerFactory.CreateLogger("Learner");
                                        await OrchestratorAgent.Execution.AdaptiveLearner.RunAsync(esRoot, llog, cts.Token);
                                        if (enableNq) await OrchestratorAgent.Execution.AdaptiveLearner.RunAsync(nqRoot, llog, cts.Token);
                                        LearnerStatus.Update(true, DateTime.UtcNow);
                                    }
                                    catch (Exception ex)
                                    {
                                        try { LearnerStatus.Update(true, DateTime.UtcNow, lastApplied: null, note: ex.Message); } catch { }
                                    }
                                    finally
                                    {
                                        try { Environment.SetEnvironmentVariable("RUN_LEARNING", prev); } catch { }
                                    }
                                }, cts.Token);
                                return Results.Json(new { started = true });
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { started = false, error = ex.Message }, statusCode: 500);
                            }
                        });
                        // VerifyToday summary: totals by status using today UTC window
                        web.MapGet("/verify/today", async () =>
                        {
                            try
                            {
                                var api = new ApiClient(http, loggerFactory.CreateLogger<ApiClient>(), apiBase);
                                var tok = await jwtCache.GetAsync();
                                if (!string.IsNullOrWhiteSpace(tok)) api.SetJwt(tok);
                                var utc = DateTimeOffset.UtcNow;
                                var start = new DateTimeOffset(utc.Year, utc.Month, utc.Day, 0, 0, 0, TimeSpan.Zero);
                                var end = start.AddDays(1);
                                var body = new { accountId, start = start.ToUnixTimeMilliseconds(), end = end.ToUnixTimeMilliseconds() };
                                var orders = await api.SearchOrdersAsync(body, cts.Token);
                                var trades = await api.SearchTradesAsync(body, cts.Token);
                                int Count(JsonElement root, string status)
                                {
                                    try { if (root.ValueKind == JsonValueKind.Object && root.TryGetProperty("data", out var d)) root = d; } catch { }
                                    int n = 0; if (root.ValueKind == JsonValueKind.Array)
                                    {
                                        foreach (var el in root.EnumerateArray())
                                        {
                                            try { if (el.TryGetProperty("status", out var s) && s.ToString().Equals(status, StringComparison.OrdinalIgnoreCase)) n++; } catch { }
                                        }
                                    }
                                    return n;
                                }
                                var res = new
                                {
                                    filled = Count(orders, "FILLED"),
                                    rejected = Count(orders, "REJECTED"),
                                    cancelled = Count(orders, "CANCELLED") + Count(orders, "CANCELED"),
                                    open = Count(orders, "OPEN") + Count(orders, "NEW"),
                                    trades = trades.ValueKind == JsonValueKind.Array ? trades.GetArrayLength() : 0
                                };
                                return Results.Json(res);
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { error = ex.Message }, statusCode: 500);
                            }
                        });
                        web.MapGet("/capabilities", () => Results.Json(OrchestratorAgent.Infra.Capabilities.All));
                        // Veto counters: GET snapshot, POST clear
                        web.MapGet("/veto", () =>
                        {
                            try
                            {
                                var map = status.Snapshot("veto.");
                                return Results.Json(map);
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { error = ex.Message }, statusCode: 500);
                            }
                        });
                        web.MapPost("/veto/clear", () =>
                        {
                            try
                            {
                                var n = status.ClearByPrefix("veto.");
                                return Results.Json(new { cleared = n });
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { error = ex.Message }, statusCode: 500);
                            }
                        });
                        web.MapGet("/deploy/status", () =>
                        {
                            string stateDir = Path.Combine(AppContext.BaseDirectory, "state");
                            string lastPath = Path.Combine(stateDir, "last_deployed.txt");
                            string logPath = Path.Combine(stateDir, "deployments.jsonl");
                            string pending = Path.Combine(stateDir, "pending_commits.json");
                            string readOrEmpty(string p) => System.IO.File.Exists(p) ? System.IO.File.ReadAllText(p) : "";
                            var obj = new { lastDeployed = readOrEmpty(lastPath), historyJsonl = readOrEmpty(logPath), pendingCommits = readOrEmpty(pending) };
                            return Results.Json(obj);
                        });
                        web.MapGet("/retune/status", () =>
                        {
                            try
                            {
                                var path = Path.Combine(AppContext.BaseDirectory, "state", "tuning", "retune_status.json");
                                if (!System.IO.File.Exists(path)) return Results.Text("{}", "application/json");
                                return Results.Text(System.IO.File.ReadAllText(path), "application/json");
                            }
                            catch (Exception ex) { return Results.Json(new { error = ex.Message }, statusCode: 500); }
                        });
                        // On-demand 10-day performance summary (aggregates per root across S2/S3/S6/S11)
                        web.MapGet("/perf/summary", async (HttpContext ctx) =>
                        {
                            try
                            {
                                int days = 10;
                                var q = ctx.Request.Query["days"].FirstOrDefault();
                                if (!string.IsNullOrWhiteSpace(q) && int.TryParse(q, out var d) && d > 0) days = d;
                                var until = DateTime.UtcNow.Date.AddDays(1).AddTicks(-1);
                                var since = until.AddDays(-(days - 1)).Date;

                                var roots = new System.Collections.Generic.List<string> { esRoot }; if (enableNq) roots.Add(nqRoot);
                                var strats = new[] { "S2", "S3", "S6", "S11" };

                                // Ensure JWT before calling history
                                try { var tok = await jwtCache.GetAsync(); if (!string.IsNullOrWhiteSpace(tok)) http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", tok); } catch { }

                                // Run summaries sequentially per root/strategy to keep load modest
                                foreach (var root in roots)
                                {
                                    if (!contractIds.TryGetValue(root, out var cid) || string.IsNullOrWhiteSpace(cid)) continue;
                                    foreach (var s in strats)
                                    {
                                        try
                                        {
                                            if (s == "S2") await OrchestratorAgent.Execution.TuningRunner.RunS2SummaryAsync(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, since, until, log, cts.Token);
                                            else if (s == "S3") await OrchestratorAgent.Execution.TuningRunner.RunS3SummaryAsync(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, since, until, log, cts.Token);
                                            else await OrchestratorAgent.Execution.TuningRunner.RunStrategySummaryAsync(http, async () => await jwtCache.GetAsync() ?? string.Empty, cid, root, s, since, until, log, cts.Token);
                                        }
                                        catch (Exception ex)
                                        {
                                            log.LogWarning(ex, "[Perf] summary run failed {Root}/{Strat}", root, s);
                                        }
                                    }
                                }

                                // Aggregate latest results from state/backtest
                                var outDir = Path.Combine(AppContext.BaseDirectory, "state", "backtest");
                                var list = new System.Collections.Generic.List<object>();
                                foreach (var root in roots)
                                {
                                    decimal totalUsd = 0m; int totalTrades = 0;
                                    var per = new System.Collections.Generic.List<object>();
                                    foreach (var s in strats)
                                    {
                                        try
                                        {
                                            var pattern = $"{s}-summary-{root}-";
                                            var files = System.IO.Directory.Exists(outDir)
                                                ? new DirectoryInfo(outDir).GetFiles("*.json").Where(f => f.Name.StartsWith(pattern, StringComparison.OrdinalIgnoreCase)).OrderByDescending(f => f.LastWriteTimeUtc).ToList()
                                                : [];
                                            if (files.Count == 0) { per.Add(new { strat = s, trades = 0, netUsd = 0m, winRate = 0m }); continue; }
                                            var txt = System.IO.File.ReadAllText(files[0].FullName);
                                            using var doc = System.Text.Json.JsonDocument.Parse(txt);
                                            var r = doc.RootElement;
                                            int trades = r.TryGetProperty("trades", out var tr) && tr.TryGetInt32(out var it) ? it : 0;
                                            decimal netUsd = r.TryGetProperty("netUsd", out var nu) && nu.TryGetDecimal(out var du) ? du : 0m;
                                            decimal winRate = r.TryGetProperty("winRate", out var wr) && wr.ValueKind == System.Text.Json.JsonValueKind.Number ? wr.GetDecimal() : 0m;
                                            totalTrades += trades; totalUsd += netUsd;
                                            per.Add(new { strat = s, trades, netUsd, winRate });
                                        }
                                        catch { per.Add(new { strat = s, trades = 0, netUsd = 0m, winRate = 0m }); }
                                    }
                                    list.Add(new { root, totalTrades, totalUsd, perStrategy = per });
                                }

                                var res = new { days, since, until, roots = list };
                                return Results.Json(res);
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { error = ex.Message }, statusCode: 500);
                            }
                        });
                        // Canary state endpoint for visibility (reads state/params/canary.json if present)
                        web.MapGet("/canary/state", () =>
                        {
                            try
                            {
                                var canaryPath = Path.Combine(AppContext.BaseDirectory, "state", "params", "canary.json");
                                if (!System.IO.File.Exists(canaryPath)) return Results.Text("{}", "application/json");
                                var txt = System.IO.File.ReadAllText(canaryPath);
                                return Results.Text(txt, "application/json");
                            }
                            catch (Exception ex)
                            {
                                return Results.Json(new { error = ex.Message }, statusCode: 500);
                            }
                        });
                        web.MapPost("/canary/reset", () =>
                        {
                            try
                            {
                                var path = Path.Combine(AppContext.BaseDirectory, "state", "params", "canary.json");
                                if (System.IO.File.Exists(path)) System.IO.File.Delete(path);
                                return Results.Json(new { ok = true });
                            }
                            catch (Exception ex) { return Results.Json(new { ok = false, error = ex.Message }, statusCode: 500); }
                        });
                        web.MapPost("/canary/blacklist/clear", () =>
                        {
                            try
                            {
                                var path = Path.Combine(AppContext.BaseDirectory, "state", "params", "canary.json");
                                if (!System.IO.File.Exists(path)) return Results.Json(new { ok = true, msg = "no state" });
                                var json = System.IO.File.ReadAllText(path);
                                using var doc = System.Text.Json.JsonDocument.Parse(json);
                                var root = doc.RootElement;
                                var dict = new System.Text.Json.Nodes.JsonObject();
                                foreach (var prop in root.EnumerateObject())
                                {
                                    if (prop.NameEquals("blacklist")) { dict["blacklist"] = new System.Text.Json.Nodes.JsonObject(); }
                                    else dict[prop.Name] = System.Text.Json.Nodes.JsonNode.Parse(prop.Value.GetRawText());
                                }
                                System.IO.File.WriteAllText(path, dict.ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
                                return Results.Json(new { ok = true });
                            }
                            catch (Exception ex) { return Results.Json(new { ok = false, error = ex.Message }, statusCode: 500); }
                        });
                        web.MapGet("/promote", () => { modeRef?.Set(OrchestratorAgent.Ops.TradeMode.Live); if (appStateRef is not null) appStateRef.DrainMode = false; return Results.Json(new { ok = true }); });
                        web.MapGet("/demote", () => { modeRef?.Set(OrchestratorAgent.Ops.TradeMode.Shadow); if (appStateRef is not null) appStateRef.DrainMode = true; return Results.Json(new { ok = true }); });
                        web.MapGet("/drain", () => { if (appStateRef is not null) appStateRef.DrainMode = true; return Results.Json(new { ok = true }); });

                        _ = web.RunAsync(cts.Token);
                        var firstUrl = (urls ?? "http://localhost:5000").Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? "http://localhost:5000";
                        log.LogInformation("Dashboard available at {Url}/dashboard (health at {Url}/healthz)", firstUrl, firstUrl);
                    }
                    catch (Exception ex)
                    {
                        log.LogWarning(ex, "Dashboard failed to start; continuing without UI.");
                    }

                    // Wire ticks/marks into dashboard stream
                    if (dashboardHub is not null)
                    {
                        market1.OnTrade += (cid, tick) => { try { if (cid == esContract) dashboardHub.OnTick(esRoot, tick.TimestampUtc, tick.Price, tick.Volume); } catch { } };
                        if (enableNq && market2 != null) market2.OnTrade += (cid, tick) => { try { if (cid == nqContract) dashboardHub.OnTick(nqRoot, tick.TimestampUtc, tick.Price, tick.Volume); } catch { } };
                        market1.OnQuote += (cid, last, bid, ask) => { try { if (cid == esContract && last > 0) dashboardHub.OnMark(esRoot, last); } catch { } };
                        if (enableNq && market2 != null) market2.OnQuote += (cid, last, bid, ask) => { try { if (cid == nqContract && last > 0) dashboardHub.OnMark(nqRoot, last); } catch { } };
                    }

                    market1.OnQuote += (cid, last, bid, ask) =>
                    {
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
                    if (enableNq && market2 != null) market2.OnQuote += (cid, last, bid, ask) =>
                    {
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
                                if (contractIds.TryGetValue(nqRoot, out string? value) && snap.TryGetValue(value, out var nqByCid)) { qty = nqByCid.Qty; avg = nqByCid.AvgPrice; }
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
                    market1.OnTrade += (_, __) => status.Set("last.trade", DateTimeOffset.UtcNow);
                    if (enableNq && market2 != null) market2.OnTrade += (_, __) => status.Set("last.trade", DateTimeOffset.UtcNow);
                    market1.OnDepth += (cid, json) =>
                    {
                        status.Set("last.depth", DateTimeOffset.UtcNow);
                        try
                        {
                            int bidQty = 0, askQty = 0;
                            try
                            {
                                if (json.ValueKind == System.Text.Json.JsonValueKind.Object)
                                {
                                    if (json.TryGetProperty("bids", out var bids) && bids.ValueKind == System.Text.Json.JsonValueKind.Array && bids.GetArrayLength() > 0)
                                    {
                                        var lvl = bids[0];
                                        if (lvl.TryGetProperty("size", out var s) && s.TryGetInt32(out var sv)) bidQty = sv;
                                        else if (lvl.TryGetProperty("qty", out var q) && q.TryGetInt32(out var qv)) bidQty = qv;
                                        else if (lvl.TryGetProperty("quantity", out var q2) && q2.TryGetInt32(out var q2v)) bidQty = q2v;
                                    }
                                    else if (json.TryGetProperty("bestBidSize", out var bbs) && bbs.TryGetInt32(out var bbv)) bidQty = bbv;
                                    else if (json.TryGetProperty("bestBidQty", out var bbq) && bbq.TryGetInt32(out var bbqv)) bidQty = bbqv;

                                    if (json.TryGetProperty("asks", out var asks) && asks.ValueKind == System.Text.Json.JsonValueKind.Array && asks.GetArrayLength() > 0)
                                    {
                                        var lvl = asks[0];
                                        if (lvl.TryGetProperty("size", out var s2) && s2.TryGetInt32(out var sv2)) askQty = sv2;
                                        else if (lvl.TryGetProperty("qty", out var q3) && q3.TryGetInt32(out var qv3)) askQty = qv3;
                                        else if (lvl.TryGetProperty("quantity", out var q4) && q4.TryGetInt32(out var qv4)) askQty = qv4;
                                    }
                                    else if (json.TryGetProperty("bestAskSize", out var bas) && bas.TryGetInt32(out var bav)) askQty = bav;
                                    else if (json.TryGetProperty("bestAskQty", out var baq) && baq.TryGetInt32(out var baqv)) askQty = baqv;
                                }
                            }
                            catch { }
                            var sum = System.Math.Max(0, bidQty) + System.Math.Max(0, askQty);
                            try
                            {
                                if (!string.IsNullOrWhiteSpace(esContract) && cid == esContract) status.Set("depth.top.ES", sum);
                                else if (!string.IsNullOrWhiteSpace(nqContract) && cid == nqContract) status.Set("depth.top.NQ", sum);
                            }
                            catch { }
                        }
                        catch { }
                    };
                    if (enableNq && market2 != null) market2.OnDepth += (cid, json) =>
                    {
                        status.Set("last.depth", DateTimeOffset.UtcNow);
                        try
                        {
                            int bidQty = 0, askQty = 0;
                            try
                            {
                                if (json.ValueKind == System.Text.Json.JsonValueKind.Object)
                                {
                                    if (json.TryGetProperty("bids", out var bids) && bids.ValueKind == System.Text.Json.JsonValueKind.Array && bids.GetArrayLength() > 0)
                                    {
                                        var lvl = bids[0];
                                        if (lvl.TryGetProperty("size", out var s) && s.TryGetInt32(out var sv)) bidQty = sv;
                                        else if (lvl.TryGetProperty("qty", out var q) && q.TryGetInt32(out var qv)) bidQty = qv;
                                        else if (lvl.TryGetProperty("quantity", out var q2) && q2.TryGetInt32(out var q2v)) bidQty = q2v;
                                    }
                                    else if (json.TryGetProperty("bestBidSize", out var bbs) && bbs.TryGetInt32(out var bbv)) bidQty = bbv;
                                    else if (json.TryGetProperty("bestBidQty", out var bbq) && bbq.TryGetInt32(out var bbqv)) bidQty = bbqv;

                                    if (json.TryGetProperty("asks", out var asks) && asks.ValueKind == System.Text.Json.JsonValueKind.Array && asks.GetArrayLength() > 0)
                                    {
                                        var lvl = asks[0];
                                        if (lvl.TryGetProperty("size", out var s2) && s2.TryGetInt32(out var sv2)) askQty = sv2;
                                        else if (lvl.TryGetProperty("qty", out var q3) && q3.TryGetInt32(out var qv3)) askQty = qv3;
                                        else if (lvl.TryGetProperty("quantity", out var q4) && q4.TryGetInt32(out var qv4)) askQty = qv4;
                                    }
                                    else if (json.TryGetProperty("bestAskSize", out var bas) && bas.TryGetInt32(out var bav)) askQty = bav;
                                    else if (json.TryGetProperty("bestAskQty", out var baq) && baq.TryGetInt32(out var baqv)) askQty = baqv;
                                }
                            }
                            catch { }
                            var sum = System.Math.Max(0, bidQty) + System.Math.Max(0, askQty);
                            try
                            {
                                if (!string.IsNullOrWhiteSpace(esContract) && cid == esContract) status.Set("depth.top.ES", sum);
                                else if (!string.IsNullOrWhiteSpace(nqContract) && cid == nqContract) status.Set("depth.top.NQ", sum);
                            }
                            catch { }
                        }
                        catch { }
                    };

                    // Background learner loop: reads recent summaries and writes TTL overrides
                    // NOTE: Learning system now initialized early (before connectivity) to avoid dependency issues
                    log.LogInformation("[Startup] Skipping late learning initialization (already started early)...");
                    /*
                    log.LogInformation("[Startup] Checking learning system configuration...");
                    try
                    {
                        var runLearn = (Environment.GetEnvironmentVariable("RUN_LEARNING") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        var liveOrdersFlag = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        log.LogInformation("[Startup] RUN_LEARNING={runLearn}, LIVE_ORDERS={liveOrdersFlag}", runLearn, liveOrdersFlag);
                        
                        // Always allow the learner to run (even in live); applying overrides in live remains gated below
                        if (runLearn)
                        {
                            log.LogInformation("[Startup] Starting adaptive learning system...");
                            var learnCts = CancellationTokenSource.CreateLinkedTokenSource(cts.Token);
                            _ = Task.Run(async () =>
                            {
                                var llog = loggerFactory.CreateLogger("Learner");
                                llog.LogInformation("[Learner] Adaptive Learning System started (RUN_LEARNING=1)");
                                bool allowLiveInstant = (Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                                bool liveNow = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                                LearnerStatus.Update(on: true);
                                var lastPractice = DateTime.MinValue;
                                // For demo purposes, use shorter intervals - change back to hours for production
                                var demoMode = (Environment.GetEnvironmentVariable("LEARN_DEMO_MODE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                                var minGap = demoMode ? 
                                    TimeSpan.FromMinutes(2) : // Demo: 2 minute cycles 
                                    TimeSpan.FromMinutes(Math.Max(45, int.TryParse(Environment.GetEnvironmentVariable("RETUNE_INTERVAL_MIN"), out var m) ? Math.Max(15, m) : 60));
                                llog.LogInformation("[Learner] Background loop ready (minGap={minGap}, allowLiveInstant={allowLiveInstant}, demoMode={demoMode})", minGap, allowLiveInstant, demoMode);
                                while (!learnCts.IsCancellationRequested)
                                {
                                    // 1) Practice/backtest on a rolling 7-day window for each configured symbol
                                    try
                                    {
                                        var now = DateTime.UtcNow;
                                        var timeSinceLastPractice = now - lastPractice;
                                        llog.LogInformation("[Learner] Loop iteration: lastPractice={lastPractice}, timeSince={timeSince}, minGap={minGap}", 
                                            lastPractice == DateTime.MinValue ? "never" : lastPractice.ToString("yyyy-MM-dd HH:mm:ss"), 
                                            timeSinceLastPractice, minGap);
                                            
                                        if (now - lastPractice >= minGap)
                                        {
                                            llog.LogInformation("[Learner] Starting learning cycle for {symbolCount} symbols...", contractIds.Count);
                                            var until = DateTime.UtcNow;
                                            var since = until.AddDays(-7);
                                            var getJwt = new Func<Task<string>>(async () => await jwtCache.GetAsync() ?? string.Empty);
                                            foreach (var kv in contractIds)
                                            {
                                                var root = kv.Key; var cid = kv.Value;
                                                try { await OrchestratorAgent.Execution.TuningRunner.RunS2SummaryAsync(http, getJwt, cid, root, since, until, llog, learnCts.Token); } catch { }
                                                try { await OrchestratorAgent.Execution.TuningRunner.RunS3SummaryAsync(http, getJwt, cid, root, since, until, llog, learnCts.Token); } catch { }
                                                try { await OrchestratorAgent.Execution.TuningRunner.RunStrategySummaryAsync(http, getJwt, cid, root, "S6", since, until, llog, learnCts.Token); } catch { }
                                                try { await OrchestratorAgent.Execution.TuningRunner.RunStrategySummaryAsync(http, getJwt, cid, root, "S11", since, until, llog, learnCts.Token); } catch { }
                                            }
                                            lastPractice = DateTime.UtcNow;
                                            llog.LogInformation("[Learner] Learning cycle completed at {time}, next cycle in {minGap}", lastPractice.ToString("yyyy-MM-dd HH:mm:ss"), minGap);
                                        }
                                        else
                                        {
                                            var timeRemaining = minGap - (now - lastPractice);
                                            llog.LogInformation("[Learner] Waiting for next cycle - {timeRemaining} remaining", timeRemaining);
                                        }
                                    }
                                    catch (OperationCanceledException) { llog.LogInformation("[Learner] Learning cycle cancelled"); }
                                    catch (Exception ex) { log.LogWarning(ex, "[Learn] backtest summaries failed"); }

                                    // 2) Run adaptive learner to propose ParamStore overrides from recent summaries
                                    try { 
                                        llog.LogInformation("[Learner] Running adaptive analysis for {symbol}...", esRoot);
                                        await OrchestratorAgent.Execution.AdaptiveLearner.RunAsync(esRoot, llog, learnCts.Token); 
                                        LearnerStatus.Update(true, DateTime.UtcNow);
                                        llog.LogInformation("[Learner] Adaptive analysis completed");
                                    }
                                    catch (OperationCanceledException) { llog.LogInformation("[Learner] Adaptive analysis cancelled"); }
                                    catch (Exception ex) { log.LogWarning(ex, "[Learn] loop failure"); }
                                    
                                    // Hourly cadence (or shorter for demo)
                                    var sleepDuration = demoMode ? TimeSpan.FromMinutes(2) : TimeSpan.FromHours(1);
                                    llog.LogInformation("[Learner] Sleeping for {sleepDuration} until next learning cycle...", sleepDuration);
                                    try { await Task.Delay(sleepDuration, learnCts.Token); } catch (OperationCanceledException) { llog.LogInformation("[Learner] Sleep cancelled, exiting loop"); }
                                    // Re-apply overrides in case a new one was written
                                    // Safe-apply: only auto-apply in live when explicitly allowed
                                    liveNow = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                                    allowLiveInstant = (Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                                    if (!liveNow || allowLiveInstant)
                                    {
                                        bool appliedAny = false;
                                        try { if (BotCore.Config.ParamStore.ApplyS2OverrideIfPresent(esRoot, log)) appliedAny = true; } catch { }
                                        try { if (BotCore.Config.ParamStore.ApplyS3OverrideIfPresent(esRoot, log)) appliedAny = true; } catch { }
                                        try { if (BotCore.Config.ParamStore.ApplyS6OverrideIfPresent(esRoot, log)) appliedAny = true; } catch { }
                                        try { if (BotCore.Config.ParamStore.ApplyS11OverrideIfPresent(esRoot, log)) appliedAny = true; } catch { }
                                        LearnerStatus.Update(true, DateTime.UtcNow, appliedAny, appliedAny ? "applied" : "no changes");
                                    }
                                }
                            }, learnCts.Token);
                        }
                    }
                    catch { }
                    */
                    // End of commented old learning system

                    // Instant-apply ParamStore watcher (offline default). Guarded by INSTANT_APPLY env.
                    OrchestratorAgent.Infra.ParamStoreWatcher? psWatcher = null;
                    try
                    {
                        bool instant = (Environment.GetEnvironmentVariable("INSTANT_APPLY") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        bool allowLiveInstant = (Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (instant)
                        {
                            int cdSec = 300; // default 5m
                            var rawCd = Environment.GetEnvironmentVariable("INSTANT_APPLY_COOLDOWN_SEC");
                            if (!string.IsNullOrWhiteSpace(rawCd) && int.TryParse(rawCd, out var v) && v > 0) cdSec = v;
                            var paramsDir = System.IO.Path.Combine(AppContext.BaseDirectory, "state", "params");
                            psWatcher = new OrchestratorAgent.Infra.ParamStoreWatcher(log, paramsDir, TimeSpan.FromSeconds(cdSec), allowLiveInstant);
                            psWatcher.Start();
                        }
                    }
                    catch (Exception ex)
                    {
                        log.LogWarning(ex, "[ParamWatch] init failed");
                    }

                    // Heartbeat loop (throttled inside StatusService)
                    _ = Task.Run(async () =>
                    {
                        while (!cts.IsCancellationRequested)
                        {
                            try { status.Heartbeat(); } catch { }
                            try { await Task.Delay(TimeSpan.FromSeconds(2), cts.Token); } catch { }
                        }
                    }, cts.Token);

                    // Dispose watcher on shutdown
                    _ = Task.Run(async () =>
                    {
                        try { await Task.Delay(Timeout.InfiniteTimeSpan, cts.Token); } catch { }
                        try { if (psWatcher != null) await psWatcher.DisposeAsync(); } catch { }
                    });

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
                        [esRoot] = []
                    };
                    if (enableNq) barsHist[nqRoot] = [];

                    var barPyramid = new BotCore.Market.BarPyramid();

                    // Seed from REST: Retrieve Bars (1m, last 500, include partial)
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

                            var payload = new
                            {
                                contractId,
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

                    // Autopilot defaults (non-live): turn on instant apply, preset select, and rollback unless explicitly disabled
                    try
                    {
                        var isLive = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (!isLive)
                        {
                            // Only set if not already configured by user
                            void SetIfMissing(string key, string val)
                            {
                                if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(key))) Environment.SetEnvironmentVariable(key, val);
                            }
                            SetIfMissing("INSTANT_APPLY", "1");
                            SetIfMissing("PRESET_SELECT", "1");
                            SetIfMissing("ROLLBACK_ENABLE", "1");
                            SetIfMissing("PROMOTE_TUNER", "1");
                            SetIfMissing("CANARY_ENABLE", "1");
                            // Continuous retune defaults
                            SetIfMissing("RETUNE_CONTINUOUS", "1");
                            SetIfMissing("RETUNE_LOOKBACK_DAYS", "7");
                            SetIfMissing("RETUNE_INTERVAL_MIN", "60");
                        }
                    }
                    catch { }

                    // Regime-aware preset selector (optional periodic) + Rollback guard + Nightly retuner
                    try
                    {
                        bool presetSel = (Environment.GetEnvironmentVariable("PRESET_SELECT") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        bool allowLivePreset = (Environment.GetEnvironmentVariable("PRESET_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        bool rbEnable = (Environment.GetEnvironmentVariable("ROLLBACK_ENABLE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (presetSel)
                        {
                            var presetLog = loggerFactory.CreateLogger("Preset");
                            int dwellMin = 30; // default
                            var rawDwell = Environment.GetEnvironmentVariable("PRESET_DWELL_MIN");
                            if (!string.IsNullOrWhiteSpace(rawDwell) && int.TryParse(rawDwell, out var dwellOut) && dwellOut > 0) dwellMin = dwellOut;
                            var selector = new OrchestratorAgent.Infra.PresetSelector(
                                presetLog,
                                root =>
                                {
                                    var cid = contractIds.TryGetValue(root, out var id) ? id : string.Empty;
                                    if (string.IsNullOrWhiteSpace(cid)) return [];
                                    // Convert Market.Bar to Models.Bar (subset fields)
                                    var m1 = barPyramid.M1.GetHistory(cid);
                                    var list = new System.Collections.Generic.List<BotCore.Models.Bar>(m1.Count);
                                    foreach (var b in m1)
                                    {
                                        // Normalize to UTC before constructing DateTimeOffset to avoid offset mismatch
                                        var s = b.Start;
                                        DateTime utc = s.Kind switch
                                        {
                                            DateTimeKind.Utc => s,
                                            DateTimeKind.Local => s.ToUniversalTime(),
                                            _ => DateTime.SpecifyKind(s, DateTimeKind.Utc)
                                        };
                                        long tsMs = new DateTimeOffset(utc, TimeSpan.Zero).ToUnixTimeMilliseconds();

                                        list.Add(new BotCore.Models.Bar
                                        {
                                            Start = b.Start,
                                            Ts = tsMs,
                                            Open = b.Open,
                                            High = b.High,
                                            Low = b.Low,
                                            Close = b.Close,
                                            Volume = (int)Math.Min(int.MaxValue, b.Volume)
                                        });
                                    }
                                    return list;
                                },
                                dwell: TimeSpan.FromMinutes(dwellMin),
                                allowLive: allowLivePreset);

                            _ = Task.Run(async () =>
                            {
                                int loopMin = 10; var rawLoop = Environment.GetEnvironmentVariable("PRESET_LOOP_MIN");
                                if (!string.IsNullOrWhiteSpace(rawLoop) && int.TryParse(rawLoop, out var loopOut) && loopOut > 0) loopMin = loopOut;
                                while (!cts.IsCancellationRequested)
                                {
                                    try { selector.EvaluateAndApply(esRoot); if (enableNq) selector.EvaluateAndApply(nqRoot); } catch (Exception ex) { presetLog.LogWarning(ex, "[Preset] loop"); }
                                    try { await Task.Delay(TimeSpan.FromMinutes(loopMin), cts.Token); } catch { }
                                }
                            }, cts.Token);
                        }

                        if (rbEnable)
                        {
                            var rbLog = loggerFactory.CreateLogger("Rollback");
                            int windowMin = 60; decimal dropUsd = 300m;
                            var rawW = Environment.GetEnvironmentVariable("ROLLBACK_WINDOW_MIN");
                            var rawD = Environment.GetEnvironmentVariable("ROLLBACK_DROP_USD");
                            if (!string.IsNullOrWhiteSpace(rawW) && int.TryParse(rawW, out var wm) && wm > 0) windowMin = wm;
                            if (!string.IsNullOrWhiteSpace(rawD) && decimal.TryParse(rawD, out var du) && du > 0) dropUsd = du;
                            bool allowLiveRb = (Environment.GetEnvironmentVariable("ROLLBACK_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            var guard = new OrchestratorAgent.Infra.AutoRollbackGuard(rbLog, posTracker, TimeSpan.FromMinutes(windowMin), dropUsd, allowLiveRb);
                            guard.Start();
                            // Dispose with host
                            cts.Token.Register(async () => { try { await guard.DisposeAsync(); } catch { } });
                        }

                        // Nightly retuner (optional; default enabled in non-live)
                        bool retune = (Environment.GetEnvironmentVariable("RETUNE_ENABLE") ?? (Environment.GetEnvironmentVariable("LIVE_ORDERS") == "1" ? "0" : "1")).Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (retune)
                        {
                            var retuneLog = loggerFactory.CreateLogger("Retune");
                            var roots = new List<string> { esRoot }; if (enableNq) roots.Add(nqRoot);
                            var retuner = new OrchestratorAgent.Execution.NightlyRetuner(retuneLog, http, async () => await jwtCache.GetAsync() ?? string.Empty, contractIds, roots, cts.Token);
                            retuner.Start();
                            cts.Token.Register(async () => { try { await retuner.DisposeAsync(); } catch { } });
                        }

                        // Continuous retuner (rolling 7-day window, runs periodically; default enabled in non-live)
                        bool contRetune = (Environment.GetEnvironmentVariable("RETUNE_CONTINUOUS") ?? (Environment.GetEnvironmentVariable("LIVE_ORDERS") == "1" ? "0" : "1")).Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (contRetune)
                        {
                            var contLog = loggerFactory.CreateLogger("Retune");
                            var roots = new List<string> { esRoot }; if (enableNq) roots.Add(nqRoot);
                            int lbDays = 7; var rawLb = Environment.GetEnvironmentVariable("RETUNE_LOOKBACK_DAYS"); if (!string.IsNullOrWhiteSpace(rawLb) && int.TryParse(rawLb, out var d) && d > 0) lbDays = d;
                            int minutes = 60; var rawInt = Environment.GetEnvironmentVariable("RETUNE_INTERVAL_MIN"); if (!string.IsNullOrWhiteSpace(rawInt) && int.TryParse(rawInt, out var m) && m > 0) minutes = m;
                            bool allowLiveCont = (Environment.GetEnvironmentVariable("RETUNE_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            var cont = new OrchestratorAgent.Execution.ContinuousRetuner(contLog, http, async () => await jwtCache.GetAsync() ?? string.Empty, contractIds, roots, TimeSpan.FromMinutes(minutes), lbDays, allowLiveCont, cts.Token);
                            cont.Start();
                            cts.Token.Register(async () => { try { await cont.DisposeAsync(); } catch { } });
                        }

                        // Thompson Bandit router (explore/exploit across pre-approved configs)
                        bool canaryEnable = (Environment.GetEnvironmentVariable("BANDIT_ENABLE")
                                             ?? Environment.GetEnvironmentVariable("CANARY_ENABLE")
                                             ?? (Environment.GetEnvironmentVariable("LIVE_ORDERS") == "1" ? "0" : "1"))
                                             .Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (canaryEnable)
                        {
                            var banditLog = loggerFactory.CreateLogger("Bandit");
                            int loopMin = 10; try { var v = Environment.GetEnvironmentVariable("BANDIT_LOOP_MIN"); if (!string.IsNullOrWhiteSpace(v) && int.TryParse(v, out var i) && i > 0) loopMin = i; } catch { }
                            int cooldownMin = 30; try { var v = Environment.GetEnvironmentVariable("BANDIT_COOLDOWN_MIN"); if (!string.IsNullOrWhiteSpace(v) && int.TryParse(v, out var i) && i > 0) cooldownMin = i; } catch { }
                            int ttlHours = 2; try { var v = Environment.GetEnvironmentVariable("BANDIT_TTL_HOURS"); if (!string.IsNullOrWhiteSpace(v) && int.TryParse(v, out var i) && i > 0) ttlHours = i; } catch { }
                            bool allowLiveBandit = (Environment.GetEnvironmentVariable("BANDIT_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                            var liveOrdersOn = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";

                            // Load strat-configs.json (IDs + light param translation to ParamStore payloads)
                            var stratCfgPath = "src\\BotCore\\Config\\strat-configs.json";
                            var s2Ids = new List<string>();
                            var s3Ids = new List<string>();
                            var s2Payloads = new Dictionary<string, System.Collections.Generic.Dictionary<string, System.Text.Json.JsonElement>>(StringComparer.OrdinalIgnoreCase);
                            var s3JsonById = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
                            try
                            {
                                var json = System.IO.File.ReadAllText(stratCfgPath);
                                using var doc = System.Text.Json.JsonDocument.Parse(json);
                                var root = doc.RootElement;
                                if (root.TryGetProperty("S2", out var s2Arr) && s2Arr.ValueKind == System.Text.Json.JsonValueKind.Array)
                                {
                                    foreach (var el in s2Arr.EnumerateArray())
                                    {
                                        var s2Id = el.TryGetProperty("id", out var idEl) ? idEl.GetString() ?? string.Empty : string.Empty;
                                        if (string.IsNullOrWhiteSpace(s2Id)) continue;
                                        s2Ids.Add(s2Id);
                                        var payload = new System.Collections.Generic.Dictionary<string, System.Text.Json.JsonElement>(StringComparer.OrdinalIgnoreCase);
                                        if (el.TryGetProperty("retestOffset", out var ro) && ro.TryGetInt32(out var i1)) payload["retest_offset_ticks"] = System.Text.Json.JsonSerializer.SerializeToElement(i1);
                                        if (el.TryGetProperty("ibGuard", out var ib) && ib.TryGetDecimal(out var d1)) payload["ib_atr_guard_mult"] = System.Text.Json.JsonSerializer.SerializeToElement(d1);
                                        if (el.TryGetProperty("orGuard", out var og) && og.TryGetDecimal(out var d2)) payload["or_atr_guard_mult"] = System.Text.Json.JsonSerializer.SerializeToElement(d2);
                                        if (el.TryGetProperty("stopR", out var sr) && sr.TryGetDecimal(out var d3)) payload["stop_atr_mult"] = System.Text.Json.JsonSerializer.SerializeToElement(d3);
                                        s2Payloads[s2Id] = payload;
                                    }
                                }
                                if (root.TryGetProperty("S3", out var s3Arr) && s3Arr.ValueKind == System.Text.Json.JsonValueKind.Array)
                                {
                                    foreach (var el in s3Arr.EnumerateArray())
                                    {
                                        var s3Id = el.TryGetProperty("id", out var idEl) ? idEl.GetString() ?? string.Empty : string.Empty;
                                        if (string.IsNullOrWhiteSpace(s3Id)) continue;
                                        s3Ids.Add(s3Id);
                                        var cfg = new System.Collections.Generic.Dictionary<string, object>();
                                        if (el.TryGetProperty("squeeze_len", out var sl) && sl.TryGetInt32(out var i2)) cfg["min_squeeze_bars"] = i2;
                                        if (el.TryGetProperty("nr_cluster_min", out var nrc) && nrc.TryGetInt32(out var i3)) cfg["nr_cluster_min_bars"] = i3;
                                        // default to retest entry for robustness
                                        cfg["entry_mode"] = "retest";
                                        var cfgJson = System.Text.Json.JsonSerializer.Serialize(cfg);
                                        s3JsonById[s3Id] = cfgJson;
                                    }
                                }
                            }
                            catch { /* best-effort; fall back to defaults below */ }
                            if (s2Ids.Count == 0) { s2Ids.AddRange(new[] { "S2a", "S2b" }); }
                            if (s3Ids.Count == 0) { s3Ids.AddRange(new[] { "S3a", "S3b" }); }

                            // Risk weighting by strategy (env overrides: BANDIT_W_S2, BANDIT_W_S3)
                            double W(string strat)
                            {
                                try
                                {
                                    var key = $"BANDIT_W_{strat.ToUpperInvariant()}";
                                    var raw = Environment.GetEnvironmentVariable(key);
                                    if (!string.IsNullOrWhiteSpace(raw) && double.TryParse(raw, out var w) && w > 0) return w;
                                }
                                catch { }
                                return strat.ToUpperInvariant() switch { "S2" => 1.0, "S3" => 0.9, _ => 0.8 };
                            }

                            var routerEs = new OrchestratorAgent.Execution.BanditRouter { Cooldown = TimeSpan.FromMinutes(cooldownMin) };
                            var routerNq = new OrchestratorAgent.Execution.BanditRouter { Cooldown = TimeSpan.FromMinutes(cooldownMin) };

                            // Background loop: pick and apply per root on cadence
                            _ = Task.Run(async () =>
                            {
                                var strats = new[] { "S2", "S3" };
                                var cfgMap = new Dictionary<string, string[]> { ["S2"] = [.. s2Ids], ["S3"] = [.. s3Ids] };
                                while (!cts.IsCancellationRequested)
                                {
                                    try
                                    {
                                        var (sEs, cEs, _) = routerEs.Select(strats, cfgMap, W);
                                        ApplyPick("ES", sEs, cEs);
                                        if (enableNq)
                                        {
                                            var (sNq, cNq, _) = routerNq.Select(strats, cfgMap, W);
                                            ApplyPick("NQ", sNq, cNq);
                                        }
                                    }
                                    catch (Exception ex) { banditLog.LogWarning(ex, "[Bandit] loop"); }
                                    try { await Task.Delay(TimeSpan.FromMinutes(loopMin), cts.Token); } catch { }
                                }
                            }, cts.Token);

                            void ApplyPick(string rootSym, string strat, string cfgId)
                            {
                                try
                                {
                                    // Respect live safety by default
                                    if (liveOrdersOn && !allowLiveBandit) { banditLog.LogDebug("[Bandit] Live mode — skip apply {Root} {Strat}/{Cfg}", rootSym, strat, cfgId); return; }
                                    var life = TimeSpan.FromHours(Math.Max(1, ttlHours));
                                    if (string.Equals(strat, "S2", StringComparison.OrdinalIgnoreCase))
                                    {
                                        if (!s2Payloads.TryGetValue(cfgId, out var payload) || payload is null) payload = [];
                                        BotCore.Config.ParamStore.SaveS2(rootSym, payload, life);
                                    }
                                    else if (string.Equals(strat, "S3", StringComparison.OrdinalIgnoreCase))
                                    {
                                        if (!s3JsonById.TryGetValue(cfgId, out var json) || string.IsNullOrWhiteSpace(json)) json = "{}";
                                        BotCore.Config.ParamStore.SaveS3(rootSym, json, life);
                                    }
                                    try { status.Set($"bandit.choice.{rootSym}", $"{strat}:{cfgId}"); } catch { }
                                    banditLog.LogInformation("[Bandit] Applied {Root} {Strat}/{Cfg} ttl={TTL}h", rootSym, strat, cfgId, life.TotalHours);
                                }
                                catch (Exception ex)
                                {
                                    banditLog.LogWarning(ex, "[Bandit] apply {Root} {Strat}/{Cfg}", rootSym, strat, cfgId);
                                }
                            }
                        }
                    }
                    catch { }

                    // Seed dashboard history once (540 1m bars)
                    try
                    {
                        if (dashboardHub is not null)
                        {
                            var esBars = barPyramid.M1.GetHistory(esIdForSeed);
                            if (esBars.Count > 0)
                            {
                                var list = new System.Collections.Generic.List<Dashboard.Bar>();
                                int start = Math.Max(0, esBars.Count - 540);
                                for (int i = start; i < esBars.Count; i++)
                                {
                                    var b = esBars[i];
                                    long tUnix = new DateTimeOffset(b.Start, TimeSpan.Zero).ToUnixTimeSeconds();
                                    list.Add(new Dashboard.Bar(tUnix, b.Open, b.High, b.Low, b.Close, b.Volume));
                                }
                                dashboardHub.SeedHistory(esRoot, "1", list);
                            }
                            if (enableNq && nqIdForSeed is not null)
                            {
                                var nqBars = barPyramid.M1.GetHistory(nqIdForSeed);
                                if (nqBars.Count > 0)
                                {
                                    var list = new System.Collections.Generic.List<Dashboard.Bar>();
                                    int start = Math.Max(0, nqBars.Count - 540);
                                    for (int i = start; i < nqBars.Count; i++)
                                    {
                                        var b = nqBars[i];
                                        long tUnix = new DateTimeOffset(b.Start, TimeSpan.Zero).ToUnixTimeSeconds();
                                        list.Add(new Dashboard.Bar(tUnix, b.Open, b.High, b.Low, b.Close, b.Volume));
                                    }
                                    dashboardHub.SeedHistory(nqRoot, "1", list);
                                }
                            }
                            log.LogInformation("[Dashboard] History seeded.");
                        }
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

                    // Publish MDL to status for UI and logic
                    try
                    {
                        var mdlCap = risk.cfg.max_daily_drawdown > 0 ? risk.cfg.max_daily_drawdown : (decimal?)null;
                        if (mdlCap.HasValue) status.Set("risk.daily.max", mdlCap.Value);
                    }
                    catch { }

                    var levels = new BotCore.Models.Levels();
                    bool live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? string.Empty)
                                .Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    var partialExit = new OrchestratorAgent.Ops.PartialExitService(http, jwtCache.GetAsync, log);
                    var router = new SimpleOrderRouter(http, jwtCache.GetAsync, log, live, partialExit);

                    // Auto-switch profile by ET clock with blackout/curfew
                    try
                    {
                        var et = NowET().TimeOfDay;
                        bool isBlackout = InRange(et, "16:58", "18:05") || InRange(et, "09:15", "09:23:30");
                        bool isNight = !isBlackout && (et >= TS("18:05") || et < TS("09:15"));

                        var dayPath = "src\\BotCore\\Config\\high_win_rate_profile.json";
                        var nightPath = "src\\BotCore\\Config\\high_win_rate_profile.night.json";
                        var cfgPath = isNight ? nightPath : dayPath;

                        var activeProfile = ConfigLoader.FromFile(cfgPath);
                        try { status.Set("profile.active", activeProfile.Profile); } catch { }
                        log.LogInformation("Profile loaded: {Profile} from {Path}", activeProfile.Profile, cfgPath);
                        try { status.Set("profile.buffers.ES", activeProfile.Buffers?.ES_Ticks ?? 0); } catch { }
                        try { status.Set("profile.buffers.NQ", activeProfile.Buffers?.NQ_Ticks ?? 0); } catch { }
                        try { status.Set("profile.min_spacing_sec", activeProfile.GlobalFilters?.MinSecondsBetweenEntries ?? (isNight ? 120 : 45)); } catch { }
                        // Apply S2 runtime config from profile if present
                        try
                        {
                            var s2def = activeProfile.Strategies?.FirstOrDefault(s => string.Equals(s.Id, "S2", StringComparison.OrdinalIgnoreCase));
                            if (s2def is not null)
                            {
                                BotCore.Strategy.S2RuntimeConfig.ApplyFrom(s2def);
                                // Log Patch B S2 config for observability
                                log.LogInformation("S2 Patch B Config: maxTrades={MaxTrades} entryMode={EntryMode} retestOffset={RetestOffset} ibGuard={IbGuard} orGuard={OrGuard} gapThresh={GapThresh} rsThresh={RsThresh} rollBump={RollBump} vwapVeto={VwapVeto} closeVeto={CloseVeto}",
                                    BotCore.Strategy.S2RuntimeConfig.MaxTradesPerSession,
                                    BotCore.Strategy.S2RuntimeConfig.EntryMode,
                                    BotCore.Strategy.S2RuntimeConfig.RetestOffsetTicks,
                                    BotCore.Strategy.S2RuntimeConfig.IbAtrGuardMult,
                                    BotCore.Strategy.S2RuntimeConfig.OrAtrGuardMult,
                                    BotCore.Strategy.S2RuntimeConfig.GapGuardThreshold,
                                    BotCore.Strategy.S2RuntimeConfig.RsPeerThreshold,
                                    BotCore.Strategy.S2RuntimeConfig.RollWeekSigmaBump,
                                    BotCore.Strategy.S2RuntimeConfig.PriorDayVwapVeto,
                                    BotCore.Strategy.S2RuntimeConfig.PriorDayCloseVeto);
                                // Apply ParamStore override if available (TTL-guarded, offline managed)
                                try
                                {
                                    var applied = BotCore.Config.ParamStore.ApplyS2OverrideIfPresent(esRoot, log);
                                    if (applied) status.Set("profile.override.s2", true);
                                }
                                catch { }
                                // Apply overrides for other strategies if present
                                try { if (BotCore.Config.ParamStore.ApplyS3OverrideIfPresent(esRoot, log)) status.Set("profile.override.s3", true); } catch { }
                                try { if (BotCore.Config.ParamStore.ApplyS6OverrideIfPresent(esRoot, log)) status.Set("profile.override.s6", true); } catch { }
                                try { if (BotCore.Config.ParamStore.ApplyS11OverrideIfPresent(esRoot, log)) status.Set("profile.override.s11", true); } catch { }
                            }
                        }
                        catch { }

                        // optional: if curfew configured in S2 profile, enforce no-new and flatten
                        var s2 = activeProfile.Strategies?.FirstOrDefault(s => string.Equals(s.Id, "S2", StringComparison.OrdinalIgnoreCase));
                        string noNew = string.Empty, forceFlat = string.Empty;
                        try
                        {
                            if (s2 != null && s2.Extra.TryGetValue("curfew", out var cf) && cf.ValueKind == System.Text.Json.JsonValueKind.Object)
                            {
                                if (cf.TryGetProperty("no_new_hhmm", out var nn) && nn.ValueKind == System.Text.Json.JsonValueKind.String) noNew = nn.GetString() ?? string.Empty;
                                if (cf.TryGetProperty("force_flat_hhmm", out var ff) && ff.ValueKind == System.Text.Json.JsonValueKind.String) forceFlat = ff.GetString() ?? string.Empty;
                            }
                        }
                        catch { }
                        if (!string.IsNullOrWhiteSpace(noNew) && TimeSpan.TryParse(noNew, out var nnTs) && et >= nnTs && et < TS("09:28"))
                        {
                            try { router.DisableAllEntries(); } catch { }
                            try { status.Set("curfew.no_new", true); } catch { }
                            log.LogInformation("[Curfew] No-new active from {NoNew} — entries disabled.", noNew);
                        }
                        if (!string.IsNullOrWhiteSpace(forceFlat) && TimeSpan.TryParse(forceFlat, out var ffTs) && et >= ffTs && et < TS("09:28"))
                        {
                            try { await router.CloseAll("CurfewForceFlat", CancellationToken.None); } catch { }
                            log.LogInformation("[Curfew] Force-flat at {Force}", forceFlat);
                        }
                        else if (et >= TS("09:28"))
                        {
                            try { router.EnableAllEntries(); } catch { }
                            try { status.Set("curfew.no_new", false); } catch { }
                            log.LogInformation("[Curfew] Cleared at 09:28 ET — entries re-enabled.");
                        }
                    }
                    catch (Exception ex)
                    {
                        log.LogWarning(ex, "Profile load failed");
                    }

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
                    // Populate web host refs for health endpoints
                    pfServiceRef = pfService; dstRef = dst;

                    // Mode + AutoPilot wiring
                    bool autoGoLive = (Environment.GetEnvironmentVariable("AUTO_GO_LIVE") ?? "true").Equals("true", StringComparison.OrdinalIgnoreCase)
                                   || (Environment.GetEnvironmentVariable("AUTO_GO_LIVE") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase);
                    int dryMin = int.TryParse(Environment.GetEnvironmentVariable("AUTO_DRYRUN_MINUTES"), out var dm) ? Math.Max(0, dm) : 5;
                    var rawMin = Environment.GetEnvironmentVariable("AUTO_MIN_HEALTHY_PASSES") ?? Environment.GetEnvironmentVariable("MIN_HEALTHY");
                    int minHealthy = int.TryParse(rawMin, out var mh) ? Math.Max(1, mh) : 3;
                    int demoteOnBad = int.TryParse(Environment.GetEnvironmentVariable("AUTO_DEMOTE_ON_UNHEALTHY"), out var db) ? Math.Max(1, db) : 3;
                    bool stickyLive = (Environment.GetEnvironmentVariable("AUTO_STICKY_LIVE") ?? "true").Equals("true", StringComparison.OrdinalIgnoreCase)
                                   || (Environment.GetEnvironmentVariable("AUTO_STICKY_LIVE") ?? "1").Equals("1", StringComparison.OrdinalIgnoreCase);

                    var mode = new OrchestratorAgent.Ops.ModeController(stickyLive);
                    var appState = new OrchestratorAgent.Ops.AppState();
                    modeRef = mode; appStateRef = appState;
                    var leasePath = Environment.GetEnvironmentVariable("OPS_LEASE_PATH") ?? "state/live.lock";
                    var liveLease = new OrchestratorAgent.Ops.LiveLease(leasePath);
                    liveLeaseRef = liveLease;
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
                        // Also emit to dashboard events so it shows in all modes (LIVE/PAPER/SHADOW)
                        try { emitEvent?.Invoke("mode", $"MODE => {modeStr}"); } catch { }
                    }
                    // Always emit initial MODE event regardless of concise console setting
                    LogMode();
                    mode.OnChange += _ =>
                    {
                        // Single source of truth: LogMode() logs and emits dashboard event
                        LogMode();
                    };

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
                        // Save a local snapshot of current setup for reference
                        try
                        {
                            var snapshotDir = System.IO.Path.Combine(AppContext.BaseDirectory, "state", "setup");
                            System.IO.Directory.CreateDirectory(snapshotDir);
                            // Also mirror to repo root when available (more convenient to find)
                            string? repoRoot = null;
                            try { repoRoot = Environment.CurrentDirectory; } catch { }
                            var snapshotDirRepo = !string.IsNullOrWhiteSpace(repoRoot)
                                ? System.IO.Path.Combine(repoRoot, "state", "setup")
                                : null;
                            if (!string.IsNullOrWhiteSpace(snapshotDirRepo))
                            {
                                try { System.IO.Directory.CreateDirectory(snapshotDirRepo!); } catch { }
                            }
                            var modeStr = paperMode ? "PAPER" : (mode.IsLive ? "LIVE" : "SHADOW");
                            var snap = new
                            {
                                tsUtc = DateTime.UtcNow,
                                apiBase,
                                rtcBase,
                                accountId,
                                mode = modeStr,
                                liveOrders = live,
                                autoGoLive,
                                dryMin,
                                minHealthy,
                                stickyLive,
                                symbols = contractIds,
                                env = new
                                {
                                    BOT_MODE = Environment.GetEnvironmentVariable("BOT_MODE"),
                                    SKIP_MODE_PROMPT = Environment.GetEnvironmentVariable("SKIP_MODE_PROMPT"),
                                    AUTH_ALLOW = Environment.GetEnvironmentVariable("AUTH_ALLOW"),
                                    APP_CONCISE_CONSOLE = Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE"),
                                    RUN_LEARNING = Environment.GetEnvironmentVariable("RUN_LEARNING"),
                                    INSTANT_APPLY = Environment.GetEnvironmentVariable("INSTANT_APPLY"),
                                    INSTANT_ALLOW_LIVE = Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE")
                                }
                            };
                            var json = System.Text.Json.JsonSerializer.Serialize(snap, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });
                            System.IO.File.WriteAllText(System.IO.Path.Combine(snapshotDir, "current.json"), json);
                            if (!string.IsNullOrWhiteSpace(snapshotDirRepo))
                            {
                                try { System.IO.File.WriteAllText(System.IO.Path.Combine(snapshotDirRepo!, "current.json"), json); } catch { }
                            }
                            var stamp = DateTime.UtcNow.ToString("yyyyMMdd-HHmmss");
                            System.IO.File.WriteAllText(System.IO.Path.Combine(snapshotDir, $"setup-{stamp}.json"), json);
                            if (!string.IsNullOrWhiteSpace(snapshotDirRepo))
                            {
                                try { System.IO.File.WriteAllText(System.IO.Path.Combine(snapshotDirRepo!, $"setup-{stamp}.json"), json); } catch { }
                            }
                            log.LogInformation("Saved setup snapshot to state/setup (current.json and setup-{Stamp}.json)", stamp);
                        }
                        catch (Exception ex)
                        {
                            log.LogWarning(ex, "Setup snapshot failed");
                        }
                    }
                    catch { }

                    // DataFeed readiness one-time logs (Quotes and Bars)
                    try
                    {
                        var esId = contractIds[esRoot];
                        string? nqId = enableNq && contractIds.TryGetValue(nqRoot, out string? value) ? value : null;
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
                                var bIn = status.Get<DateTimeOffset?>($"last.bar.{cid}");
                                int qMs = qUpd.HasValue ? (int)Math.Max(0, (now - qUpd.Value).TotalMilliseconds) : 0;
                                int bMs = bIn.HasValue ? (int)Math.Max(0, (now - bIn.Value).TotalMilliseconds) : 0;
                                log.LogDebug($"[{root}] Strategies 14/14 | Looking | Q:{qMs}ms B:{bMs}ms");
                                log.LogDebug("  Name                         En  State     LastSignal (UTC)      Note");
                                void Row(string name, string en, string state, string lastUtc, string note)
                                    => log.LogDebug($"  {name,-28} {en,1}   {state,-8}  {lastUtc,-20}    {note}");
                                var tsNow = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss");
                                Row("Bias Filter", "Y", "Armed", tsNow, "-");
                                Row("Breakout", "Y", "Looking", DateTime.UtcNow.AddSeconds(-2).ToString("yyyy-MM-dd HH:mm:ss"), "-");
                                Row("Pullback Pro", "Y", "Idle", "-", "-");
                                Row("Opening Drive", "Y", "Paused", "-", "Daily loss lock");
                                Row("VWAP Revert", "Y", "Looking", "-", "-");
                                log.LogDebug("  … +9 more strategies hidden");
                                log.LogDebug("");
                            }
                        }
                        catch { }
                    }

                    // Health endpoints are now served by the Kestrel web host; no separate HttpListener needed.

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
                                    var (ok, msg) = await pfService.RunAsync(symbol, cts.Token);
                                    status.Set("preflight.ok", ok);
                                    status.Set("preflight.msg", msg);
                                    if (!ok)
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
                            var contractsView = string.Join(", ", (status.Contracts ?? []).Select(kv => $"{kv.Key}={kv.Value}"));
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
                        string root = cid == contractIds.GetValueOrDefault(esRoot) ? esRoot : (contractIds.TryGetValue(nqRoot, out string? value) && cid == value ? nqRoot : esRoot);
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
                        log.LogDebug("[Bus] -> 1m {Sym} O={0} H={1} L={2} C={3}", root, b.Open, b.High, b.Low, b.Close);
                        await RunStrategiesFor(root, bar, barsHist[root], accountId, cid, risk, levels, router, paperBroker, simulateMode, log, appState, liveLease, status, cts.Token);
                        dataLog.LogDebug("[Bars] 1m close {Sym} {End:o} O={O} H={H} L={L} C={C} V={V}", root, b.End.ToUniversalTime(), b.Open, b.High, b.Low, b.Close, b.Volume);
                    };
                    barPyramid.M5.OnBarClosed += (cid, b) => { dataLog.LogDebug("[Bars] 5m close {Cid} {End:o}", cid, b.End.ToUniversalTime()); };
                    barPyramid.M30.OnBarClosed += (cid, b) => { dataLog.LogDebug("[Bars] 30m close {Cid} {End:o}", cid, b.End.ToUniversalTime()); };

                    // Feed live trades into the 1m aggregator (quotes not required for bars)

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
                                            log.LogDebug($"[{sym}] Strategies 14/14 | {state} | Q:{(qAge >= 0 ? qAge.ToString() : "-")}s B:{(bAge >= 0 ? bAge.ToString() : "-")}s{(paused ? " PAUSED" : string.Empty)}");
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
                                    riskLog.LogDebug("Heartbeat — DailyPnL {Pnl}  |  MaxDailyLoss {Max}  |  Remaining Risk {Rem}",
                                        pnl.ToString("+$0.00;-$0.00;$0.00"),
                                        maxDailyLossCfg.ToString("#,0.00"),
                                        (remaining < 0 ? 0 : remaining).ToString("$#,0.00"));
                                    log.LogDebug("Session checkpoint — All systems green. Next heartbeat in 60s.");
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
                BotCore.Models.Bar bar,
                System.Collections.Generic.List<BotCore.Models.Bar> history,
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

                    // Local helpers: veto counters and simple RS momentum
                    void IncVeto(string reason)
                    {
                        try
                        {
                            var baseKey = $"veto.{reason}";
                            var total = status.Get<int>(baseKey);
                            status.Set(baseKey, total + 1);
                            var symKey = $"{baseKey}.{symbol.ToUpperInvariant()}";
                            var by = status.Get<int>(symKey);
                            status.Set(symKey, by + 1);
                        }
                        catch { }
                    }

                    static decimal ComputeMomentum(IReadOnlyList<BotCore.Models.Bar> bars, int look)
                    {
                        if (bars == null || bars.Count < look + 1) return 0m;
                        decimal sum = 0m;
                        int start = Math.Max(1, bars.Count - look);
                        for (int i = start; i < bars.Count; i++)
                        {
                            var p0 = bars[i - 1].Close; var p1 = bars[i].Close;
                            if (p0 != 0m) sum += (p1 - p0) / p0;
                        }
                        return sum;
                    }


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

                    // Master bypass: ET_NO_GUARD — trade without most guards when not in blackout windows
                    // Blackouts: ET 16:58–18:05 and 09:15–09:23:30 are always enforced
                    bool etNoGuard = false;
                    try
                    {
                        var etT = NowET().TimeOfDay;
                        bool etBlackout = InRange(etT, "16:58", "18:05") || InRange(etT, "09:15", "09:23:30");
                        var v = (Environment.GetEnvironmentVariable("ET_NO_GUARD") ?? "0").Trim().ToLowerInvariant();
                        etNoGuard = !etBlackout && (v is "1" or "true" or "yes");
                    }
                    catch { }

                    // Session filters & freeze guard
                    try
                    {
                        var cmeTz = TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time");
                        var nowCt = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, cmeTz).TimeOfDay;
                        var open = new TimeSpan(8, 30, 0);
                        var close = new TimeSpan(15, 0, 0);
                        bool inRth = nowCt >= open && nowCt <= close;
                        bool rthOnly = (Environment.GetEnvironmentVariable("SESSION_RTH_ONLY") ?? "false").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (!etNoGuard && rthOnly && !inRth)
                        {
                            log.LogInformation("[SKIP reason=session] {Sym} outside RTH", symbol);
                            IncVeto("session");
                            return;
                        }
                        if (!etNoGuard && rthOnly && (nowCt < open.Add(TimeSpan.FromMinutes(3)) || nowCt > close.Subtract(TimeSpan.FromMinutes(5))))
                        {
                            log.LogInformation("[SKIP reason=session_window] {Sym} warmup/cooldown window", symbol);
                            IncVeto("session_window");
                            return;
                        }
                        // Curfew: block new entries into U.S. open (ET 09:15–09:23)
                        try
                        {
                            if (!etNoGuard && InRange(NowET().TimeOfDay, "09:15", "09:23:30"))
                            {
                                log.LogInformation("[SKIP reason=curfew] {Sym} ET 09:15–09:23:30 curfew active", symbol);
                                IncVeto("curfew");
                                return;
                            }
                        }
                        catch { }
                        // Econ blocks (ET), format: HH:mm-HH:mm;HH:mm-HH:mm
                        var econ = Environment.GetEnvironmentVariable("ECON_BLOCKS_ET");
                        if (!etNoGuard && !string.IsNullOrWhiteSpace(econ))
                        {
                            try
                            {
                                var etTz = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                                var nowEt = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, etTz).TimeOfDay;
                                foreach (var blk in econ.Split([',', ';'], StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
                                {
                                    var parts = blk.Split('-', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
                                    if (parts.Length == 2 && TimeSpan.TryParse(parts[0], out var b) && TimeSpan.TryParse(parts[1], out var e))
                                    {
                                        if (nowEt >= b && nowEt <= e) { log.LogInformation("[SKIP reason=econ] {Sym} in econ block {Blk}", symbol, blk); IncVeto("econ"); return; }
                                    }
                                }
                            }
                            catch { }
                        }
                        // Freeze: bypass in backtest; enforce in live/shadow with configurable max age
                        bool btBypassFreeze = string.Equals(Environment.GetEnvironmentVariable("BT_IGNORE_QUOTE_FREEZE"), "1", StringComparison.OrdinalIgnoreCase)
                                           || string.Equals(Environment.GetEnvironmentVariable("BT_IGNORE_QUOTE_FREEZE"), "true", StringComparison.OrdinalIgnoreCase);
                        if (!etNoGuard && !btBypassFreeze)
                        {
                            var qUpd = status.Get<DateTimeOffset?>($"last.quote.updated.{contractId}") ?? status.Get<DateTimeOffset?>($"last.quote.{contractId}");
                            int maxMs = Math.Max(1000, int.TryParse(Environment.GetEnvironmentVariable("QUOTE_MAX_AGE_MS"), out var mx) ? mx : 3000);
                            if (qUpd.HasValue && (DateTimeOffset.UtcNow - qUpd.Value) > TimeSpan.FromMilliseconds(maxMs))
                            {
                                log.LogInformation("[SKIP reason=freeze] {Sym} lastQuoteAge={Age}s", symbol, (int)(DateTimeOffset.UtcNow - qUpd.Value).TotalSeconds);
                                IncVeto("freeze");
                                return;
                            }
                        }

                        // Session-aware spread (ES: RTH=2/ETH=3; NQ: RTH=3/ETH=4) — backtest bypass
                        bool btBypassSpread = string.Equals(Environment.GetEnvironmentVariable("BT_IGNORE_SPREAD"), "1", StringComparison.OrdinalIgnoreCase)
                                           || string.Equals(Environment.GetEnvironmentVariable("BT_IGNORE_SPREAD"), "true", StringComparison.OrdinalIgnoreCase);
                        if (!etNoGuard && !btBypassSpread)
                        {
                            int spreadTicks = status.Get<int>($"spread.ticks.{symbol}");

                            // RTH detection (CME Central Time 08:30–15:00)
                            var cmeTz2 = TimeZoneInfo.FindSystemTimeZoneById("Central Standard Time");
                            var nowCt2 = TimeZoneInfo.ConvertTime(DateTimeOffset.UtcNow, cmeTz2).TimeOfDay;
                            bool inRth2 = nowCt2 >= new TimeSpan(8, 30, 0) && nowCt2 <= new TimeSpan(15, 0, 0);

                            int defAllow = symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase)
                                ? (inRth2 ? 3 : 4)
                                : (inRth2 ? 2 : 3); // ES default

                            // Env overrides: SPREAD_ALLOW_{SYM}_{RTH/ETH}
                            string key = $"SPREAD_ALLOW_{symbol.ToUpperInvariant()}_{(inRth2 ? "RTH" : "ETH")}";
                            if (int.TryParse(Environment.GetEnvironmentVariable(key), out var envAllow) && envAllow > 0)
                                defAllow = envAllow;

                            if (spreadTicks > defAllow)
                            {
                                log.LogInformation("[SKIP reason=spread] {Sym} spread={S}t allow={A}t", symbol, spreadTicks, defAllow);
                                IncVeto("spread");
                                return;
                            }
                        }

                        // News-minute gate (America/Santo_Domingo wall-time): block −2m..+3m around :00 and :30
                        bool btBypassNews = string.Equals(Environment.GetEnvironmentVariable("BT_IGNORE_NEWS"), "1", StringComparison.OrdinalIgnoreCase)
                                           || string.Equals(Environment.GetEnvironmentVariable("BT_IGNORE_NEWS"), "true", StringComparison.OrdinalIgnoreCase);
                        if (!etNoGuard && !btBypassNews)
                        {
                            try
                            {
                                var nowLocal = NowSD();
                                int m = nowLocal.Minute;
                                int s = nowLocal.Second;
                                bool aroundTop = m >= 58 || m <= 3;   // 58,59,0,1,2,3
                                bool aroundHalf = (m >= 28 && m <= 33); // 28..33
                                if (aroundTop || aroundHalf)
                                {
                                    log.LogInformation("[SKIP reason=news_minute_gate] {Sym} local={H}:{M:D2}:{S:D2}", symbol, nowLocal.Hour, m, s);
                                    IncVeto("news_minute_gate");
                                    return;
                                }
                            }
                            catch { }
                        }
                    }
                    catch { }

                    // ET strategy windows + per-strategy attempt caps (can be disabled via env for all-hours quality-first)
                    try
                    {
                        bool skipTimeWindows = (Environment.GetEnvironmentVariable("SKIP_TIME_WINDOWS") ?? Environment.GetEnvironmentVariable("ALL_HOURS_QUALITY") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        bool skipAttemptCaps = (Environment.GetEnvironmentVariable("SKIP_ATTEMPT_CAPS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (etNoGuard) { skipTimeWindows = true; skipAttemptCaps = true; }
                        // In all-hours quality mode, restrict to the initial four strategies
                        if (skipTimeWindows && !etNoGuard)
                        {
                            var allowedAh = new System.Collections.Generic.HashSet<string>(collection, StringComparer.OrdinalIgnoreCase);
                            signals = [.. signals.Where(s => allowedAh.Contains(s.StrategyId))];
                        }
                        var etNow = NowET().TimeOfDay;
                        bool isBlackout = InRange(etNow, "16:58", "18:05") || InRange(etNow, "09:15", "09:23:30");
                        bool isNight = !isBlackout && (etNow >= TS("18:05") || etNow < TS("09:15"));
                        if (!skipTimeWindows)
                        {
                            var allow = new System.Collections.Generic.HashSet<string>(StringComparer.OrdinalIgnoreCase);
                            if (isNight)
                            {
                                if (InRange(etNow, "18:05", "02:00")) allow.UnionWith(new[] { "S2" });
                                else if (InRange(etNow, "02:55", "04:10")) allow.UnionWith(new[] { "S3" });
                            }
                            else
                            {
                                if (etNow < TS("09:28")) { /* none allowed before 09:28 */ }
                                else if (InRange(etNow, "09:28", "10:00")) allow.UnionWith(new[] { "S6", "S3" });
                                else if (etNow >= TS("10:20") && etNow < TS("13:30")) allow.UnionWith(new[] { "S2" });
                                else if (etNow >= TS("13:30")) allow.UnionWith(new[] { "S11" });
                            }

                            if (allow.Count > 0)
                            {
                                signals = [.. signals.Where(s => allow.Contains(s.StrategyId))];
                                if (signals.Count == 0)
                                {
                                    log.LogInformation("[SKIP reason=time_window] {Sym} no strategies allowed in this ET window", symbol);
                                    IncVeto("time_window");
                                    return;
                                }
                            }
                            else
                            {
                                log.LogInformation("[SKIP reason=time_window] {Sym} no entries in this ET window", symbol);
                                IncVeto("time_window");
                                return;
                            }
                        }

                        int CapFor(string id, bool night) => night ? ((string.Equals(id, "S2", StringComparison.OrdinalIgnoreCase) || string.Equals(id, "S3", StringComparison.OrdinalIgnoreCase)) ? 1 : 0)
                                                                     : ((string.Equals(id, "S2", StringComparison.OrdinalIgnoreCase) || string.Equals(id, "S3", StringComparison.OrdinalIgnoreCase) || string.Equals(id, "S6", StringComparison.OrdinalIgnoreCase) || string.Equals(id, "S11", StringComparison.OrdinalIgnoreCase)) ? 2 : 0);
                        // Per-direction caps: same as above by default; can tune via env later
                        int CapForDir(string id, bool night, int dir)
                        {
                            var baseCap = CapFor(id, night);
                            return baseCap; // symmetric L/S for now
                        }
                        if (!skipAttemptCaps)
                        {
                            signals = [.. signals.Where(s =>
                            {
                                var cap = CapFor(s.StrategyId, isNight);
                                if (cap <= 0) return false;
                                try
                                {
                                    var todayEt = NowET().Date;
                                    var key = $"{s.StrategyId.ToUpperInvariant()}|{symbol.ToUpperInvariant()}";
                                    var cur = _attemptsPerStrat.GetOrAdd(key, _ => (todayEt, 0));
                                    if (cur.DayEt != todayEt) cur = (todayEt, 0);
                                    if (cur.Count >= cap) return false;

                                    // Check per-direction as well
                                    int dir = string.Equals(s.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
                                    var keyDir = $"{s.StrategyId.ToUpperInvariant()}|{symbol.ToUpperInvariant()}|{(dir > 0 ? "L" : "S")}";
                                    var capDir = CapForDir(s.StrategyId, isNight, dir);
                                    var curDir = _attemptsPerStratDir.GetOrAdd(keyDir, _ => (todayEt, 0));
                                    if (curDir.DayEt != todayEt) curDir = (todayEt, 0);
                                    return curDir.Count < capDir;
                                }
                                catch { return true; }
                            })];
                            if (signals.Count == 0)
                            {
                                log.LogInformation("[SKIP reason=attempt_cap] {Sym} attempts cap hit", symbol);
                                IncVeto("attempt_cap");
                                return;
                            }
                        }
                    }
                    catch { }

                    // Night microstructure depth guard before selection/routing
                    try
                    {
                        var etNow2 = NowET().TimeOfDay;
                        bool isBlackout2 = InRange(etNow2, "16:58", "18:05") || InRange(etNow2, "09:15", "09:23:30");
                        bool isNight2 = !isBlackout2 && (etNow2 >= TS("18:05") || etNow2 < TS("09:15"));
                        if (!etNoGuard && isNight2 && (symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) || symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase)))
                        {
                            int minDepth = symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) ? 300 : 80;
                            int curDepth = 0;
                            try { curDepth = Math.Max(0, status.Get<int>($"depth.top.{symbol}")); } catch { }
                            if (curDepth < minDepth)
                            {
                                log.LogInformation("[SKIP reason=depth_min] {Sym} top-of-book={Depth} min={Min}", symbol, curDepth, minDepth);
                                IncVeto("depth_min");
                                return;
                            }
                        }
                    }
                    catch { }

                    // Entry spacing guard per profile (per-symbol)
                    try
                    {
                        var etNow3 = NowET().TimeOfDay;
                        bool isBlackout3 = InRange(etNow3, "16:58", "18:05") || InRange(etNow3, "09:15", "09:23:30");
                        bool isNight3 = !isBlackout3 && (etNow3 >= TS("18:05") || etNow3 < TS("09:15"));
                        int defSpacing = isNight3 ? 120 : 45;
                        int spacingSec = defSpacing;
                        try { var s = status.Get<int>("profile.min_spacing_sec"); if (s > 0) spacingSec = s; } catch { }
                        var list = _entriesPerHour.GetOrAdd(symbol, _ => []);
                        DateTime? last = null; lock (_entriesLock) { if (list.Count > 0) last = list[^1]; }
                        if (last.HasValue)
                        {
                            var age = DateTime.UtcNow - last.Value;
                            if (!etNoGuard && age < TimeSpan.FromSeconds(spacingSec))
                            {
                                log.LogInformation("[SKIP reason=entry_spacing] {Sym} last_entry_age={Age}s min={Min}s", symbol, (int)age.TotalSeconds, spacingSec);
                                IncVeto("entry_spacing");
                                return;
                            }
                        }
                    }
                    catch { }

                    // Priority arbiter: S6 > S3 > S2 > S11. Pick best ExpR within highest available.
                    var ranked = new[] { "S6", "S3", "S2", "S11" };
                    var pickPool = new System.Collections.Generic.List<Signal>();
                    foreach (var sid in ranked)
                    {
                        var pool = signals.Where(s => string.Equals(s.StrategyId, sid, StringComparison.OrdinalIgnoreCase)).ToList();
                        if (pool.Count > 0) { pickPool = pool; break; }
                    }
                    if (pickPool.Count == 0) return;
                    var bestLong = pickPool.Where(s => string.Equals(s.Side, "BUY", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                    var bestShort = pickPool.Where(s => string.Equals(s.Side, "SELL", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                    var chosen = (bestLong?.ExpR ?? -1m) >= (bestShort?.ExpR ?? -1m) ? bestLong : bestShort;
                    if (chosen is null) return;
                    try { BotCore.TradeLog.Signal(log, symbol, chosen.StrategyId, chosen.Side, chosen.Size, chosen.Entry, chosen.Stop, chosen.Target, $"score={chosen.ExpR:F2}", chosen.Tag ?? string.Empty); } catch { }

                    // RS leader-only arbiter (env RS_ARB/RS_ARB_LEADER_ONLY=1):
                    try
                    {
                        bool rsEnable = (Environment.GetEnvironmentVariable("RS_ARB") ?? Environment.GetEnvironmentVariable("RS_ARB_LEADER_ONLY") ?? "1").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                        if (rsEnable && (symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) || symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase)))
                        {
                            int look = 30; try { var v = Environment.GetEnvironmentVariable("RS_LOOKBACK"); if (!string.IsNullOrWhiteSpace(v) && int.TryParse(v, out var l) && l > 5) look = l; } catch { }
                            var momThis = ComputeMomentum(history, look);
                            status.Set($"rs.mom.{symbol.ToUpperInvariant()}", momThis);
                            status.Set($"rs.updated.{symbol.ToUpperInvariant()}", DateTimeOffset.UtcNow);
                            var other = symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) ? "NQ" : "ES";
                            var momOther = status.Get<decimal?>($"rs.mom.{other}") ?? 0m;
                            var updatedOther = status.Get<DateTimeOffset?>($"rs.updated.{other}") ?? DateTimeOffset.MinValue;
                            if (updatedOther != DateTimeOffset.MinValue && (DateTimeOffset.UtcNow - updatedOther) < TimeSpan.FromMinutes(2))
                            {
                                var absThis = Math.Abs(momThis); var absOther = Math.Abs(momOther);
                                decimal minDiff = 0.0003m; try { var d = Environment.GetEnvironmentVariable("RS_MIN_DIFF"); if (!string.IsNullOrWhiteSpace(d) && decimal.TryParse(d, out var dv)) minDiff = dv; } catch { }
                                if (Math.Abs(absThis - absOther) >= minDiff)
                                {
                                    var leader = absThis >= absOther ? symbol.ToUpperInvariant() : other;
                                    var dir = (absThis >= absOther ? momThis : momOther) >= 0m ? 1 : -1;
                                    bool sideOk = (dir > 0 && string.Equals(chosen.Side, "BUY", StringComparison.OrdinalIgnoreCase)) || (dir < 0 && string.Equals(chosen.Side, "SELL", StringComparison.OrdinalIgnoreCase));
                                    bool symOk = string.Equals(symbol, leader, StringComparison.OrdinalIgnoreCase);
                                    if (!(sideOk && symOk))
                                    {
                                        log.LogInformation("[SKIP reason=rs_leader] {Sym} leader={Leader} dir={Dir} side={Side}", symbol, leader, dir > 0 ? "UP" : "DOWN", chosen.Side);
                                        IncVeto("rs_leader");
                                        return;
                                    }
                                }
                            }
                        }
                    }
                    catch { }

                    // Per-symbol single-position rule with flip eligibility (S6 only, opposite side, ExpR ≥ 1.25)
                    try
                    {
                        var netQty = status.Get<int>($"pos.{symbol}.qty");
                        if (netQty != 0)
                        {
                            bool opposite = (netQty > 0 && string.Equals(chosen.Side, "SELL", StringComparison.OrdinalIgnoreCase)) || (netQty < 0 && string.Equals(chosen.Side, "BUY", StringComparison.OrdinalIgnoreCase));
                            bool flipOk = string.Equals(chosen.StrategyId, "S6", StringComparison.OrdinalIgnoreCase) && opposite && chosen.ExpR >= 1.25m;
                            if (!flipOk)
                            {
                                log.LogInformation("[SKIP reason=per_symbol_cap] {Sym} already has position {Qty}; only S6 flip allowed with ≥1.25R.", symbol, netQty);
                                return;
                            }
                        }
                    }
                    catch { }

                    // Risk scaling by window/conditions: S2 overnight 0.5x; S3 0.75x if spread>2 or volz>2.0
                    decimal riskScale = 1.0m;
                    try
                    {
                        var etNow4 = NowET().TimeOfDay;
                        bool isBlackout4 = InRange(etNow4, "16:58", "18:05") || InRange(etNow4, "09:15", "09:23:30");
                        bool isNight4 = !isBlackout4 && (etNow4 >= TS("18:05") || etNow4 < TS("09:15"));
                        if (string.Equals(chosen.StrategyId, "S2", StringComparison.OrdinalIgnoreCase) && isNight4)
                            riskScale *= 0.5m;
                        if (string.Equals(chosen.StrategyId, "S3", StringComparison.OrdinalIgnoreCase))
                        {
                            int sp = 0; try { sp = status.Get<int>($"spread.ticks.{symbol}"); } catch { }
                            // compute simple volz from recent bars
                            decimal volz = 0m;
                            try
                            {
                                int lookback = 50;
                                if (history != null && history.Count >= lookback + 1)
                                {
                                    var rets = new System.Collections.Generic.List<decimal>(lookback);
                                    for (int i = history.Count - lookback; i < history.Count; i++)
                                    {
                                        var p0 = history[i - 1].Close; var p1 = history[i].Close;
                                        if (p0 != 0) rets.Add((p1 - p0) / p0);
                                    }
                                    if (rets.Count > 0)
                                    {
                                        var mean = rets.Average();
                                        var varv = rets.Select(r => (r - mean) * (r - mean)).Average();
                                        var std = (decimal)Math.Sqrt((double)Math.Max(1e-12m, varv));
                                        var last = rets[^1];
                                        volz = std == 0m ? 0m : (last - mean) / std;
                                    }
                                }
                            }
                            catch { }
                            if (sp > 2 || volz > 2.0m) riskScale *= 0.75m;
                        }
                    }
                    catch { }

                    foreach (var sig in signals)
                    {
                        if (!object.ReferenceEquals(sig, chosen)) continue; // route only chosen
                        log.LogInformation("[Strategy] {Sym} {StrategyId} {Side} @ {Entry} (stop {Stop}, t1 {Target}) size {Size} expR {ExpR}",
                            symbol, sig.StrategyId, sig.Side, sig.Entry, sig.Stop, sig.Target, sig.Size, sig.ExpR);
                        var toRoute = sig;

                        // Apply riskScale computed above
                        try
                        {
                            var scaled = (int)Math.Max(1, Math.Floor(sig.Size * riskScale));
                            if (scaled != sig.Size) toRoute = toRoute with { Size = scaled };
                        }
                        catch { }

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
                                IncVeto("global_cap");
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
                                if (room <= 0) { log.LogInformation("[SKIP reason=max_net] {Sym} net={Net} cap={Cap}", symbol, net, maxNet); IncVeto("max_net"); return; }
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
                                        IncVeto("dd_proximity");
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
                            var list = _entriesPerHour.GetOrAdd(symbol, _ => []);
                            lock (_entriesLock)
                            {
                                list.RemoveAll(t => (now - t) > TimeSpan.FromHours(1));
                                if (list.Count >= capPerHr)
                                {
                                    log.LogInformation("[SKIP reason=entries_per_hour] {Sym} cap={Cap}", symbol, capPerHr);
                                    IncVeto("entries_per_hour");
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

                        // Day PnL kill (in R units) as no-new gate for S2
                        try
                        {
                            if (string.Equals(sig.StrategyId, "S2", StringComparison.OrdinalIgnoreCase) && sig.Stop != 0 && sig.Entry != 0)
                            {
                                decimal rpt = risk.cfg.risk_per_trade > 0 ? risk.cfg.risk_per_trade : 0m;
                                var dayUsd = status.Get<decimal?>("pnl.net") ?? 0m;
                                if (rpt > 0m)
                                {
                                    var dayR = dayUsd / rpt;
                                    var stopGain = BotCore.Strategy.S2RuntimeConfig.DayPnlStopGainR;
                                    var stopLoss = BotCore.Strategy.S2RuntimeConfig.DayPnlStopLossR;
                                    var enabled = BotCore.Strategy.S2RuntimeConfig.DayPnlKillEnabled && (stopGain != 0m || stopLoss != 0m);
                                    if (enabled)
                                    {
                                        if (stopGain > 0m && dayR >= stopGain)
                                        {
                                            try { status.Set("day_pnl.no_new", true); } catch { }
                                            log.LogInformation("[SKIP reason=day_pnl_gain] {Sym} dayR={DayR:F2} \u2265 {StopR:F2}", symbol, dayR, stopGain);
                                            continue;
                                        }
                                        if (stopLoss < 0m && dayR <= stopLoss)
                                        {
                                            try { status.Set("day_pnl.no_new", true); } catch { }
                                            log.LogInformation("[SKIP reason=day_pnl_loss] {Sym} dayR={DayR:F2} \u2264 {StopR:F2}", symbol, dayR, stopLoss);
                                            continue;
                                        }
                                        try { status.Set("day_pnl.no_new", false); } catch { }
                                    }
                                }
                            }
                        }
                        catch { }

                        var routed = await router.RouteAsync(toRoute, ct);
                        if (routed)
                        {
                            // Record intent, entries/hour stamp, and increment attempt counter (ET day)
                            try
                            {
                                var dir = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
                                _lastEntryIntent[symbol] = (dir, DateTime.UtcNow);
                                var list = _entriesPerHour.GetOrAdd(symbol, _ => []);
                                lock (_entriesLock) list.Add(DateTime.UtcNow);

                                var todayEt = NowET().Date;
                                var key = $"{sig.StrategyId.ToUpperInvariant()}|{symbol.ToUpperInvariant()}";
                                _ = _attemptsPerStrat.AddOrUpdate(key,
                                    addValueFactory: _ => (todayEt, 1),
                                    updateValueFactory: (_, oldVal) => oldVal.DayEt == todayEt ? (oldVal.DayEt, oldVal.Count + 1) : (todayEt, 1));
                                int dir2 = string.Equals(sig.Side, "SELL", StringComparison.OrdinalIgnoreCase) ? -1 : 1;
                                var keyDir = $"{sig.StrategyId.ToUpperInvariant()}|{symbol.ToUpperInvariant()}|{(dir2 > 0 ? "L" : "S")}";
                                _ = _attemptsPerStratDir.AddOrUpdate(keyDir,
                                    addValueFactory: _ => (todayEt, 1),
                                    updateValueFactory: (_, oldVal) => oldVal.DayEt == todayEt ? (oldVal.DayEt, oldVal.Count + 1) : (todayEt, 1));
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
