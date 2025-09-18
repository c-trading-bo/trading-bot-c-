using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Config;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Execution;

/// <summary>
/// Multi-strategy offline learner: reads recent backtest summaries under state/backtest and proposes
/// bounded overrides for S2/S3/S6/S11. Applies short TTL in PAPER/SHADOW when RUN_LEARNING=1.
/// In LIVE, only applies if INSTANT_ALLOW_LIVE=1.
/// </summary>
public static class AdaptiveLearner
{
    public sealed record Summary(string strategy, string symbol, DateTime start, DateTime end, int trades, int wins, int losses, decimal netUsd, decimal winRate, decimal avgR, decimal maxDrawdownUsd);

    public static async Task RunAsync(string symbolRoot, ILogger log, CancellationToken ct)
    {
        try
        {
            var run = (Environment.GetEnvironmentVariable("RUN_LEARNING") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (!run) { log.LogInformation("[Learn] RUN_LEARNING disabled."); return; }
            var liveOrders = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            var allowLive = (Environment.GetEnvironmentVariable("INSTANT_ALLOW_LIVE") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (liveOrders && !allowLive) { log.LogInformation("[Learn] Live mode — skipping apply (INSTANT_ALLOW_LIVE=0)"); return; }

            var summaries = LoadRecentSummaries(symbolRoot, days: EnvInt("LEARN_LOOKBACK_DAYS", 7));
            if (summaries.Count == 0) { log.LogInformation("[Learn] No summaries for {Sym}", symbolRoot); return; }

            // Group by strategy and compute metrics
            var byStrat = summaries.GroupBy(s => s.strategy, StringComparer.OrdinalIgnoreCase).ToDictionary(g => g.Key.ToUpperInvariant(), g => g.ToList());

            // Strict regression gate (default ON): block proposals if current window underperforms saved best
            try
            {
                var totalScore = byStrat.Values
                    .SelectMany(v => v)
                    .GroupBy(_ => 1)
                    .Select(g => Score(
                        BetaPosteriorMean(g.Sum(x => x.wins), g.Sum(x => x.losses), 7, 3),
                        SafeAvg(g.Select(x => x.avgR)),
                        g.Sum(x => x.netUsd),
                        g.Max(x => x.maxDrawdownUsd)))
                    .FirstOrDefault();

                var strict = EnvBool("LEARN_STRICT", true);
                if (strict)
                {
                    var best = LoadBest(symbolRoot);
                    if (best.HasValue && totalScore < best.Value.score)
                    {
                        log.LogInformation("[Learn] Strict: window score {Score:F0} < best {Best:F0} — skipping proposals", totalScore, best.Value.score);
                        return;
                    }
                }
            }
            catch { /* best-effort guard */ }
            foreach (var kv in byStrat)
            {
                var strat = kv.Key; var list = kv.Value;
                var t = list.Sum(x => x.trades);
                var w = list.Sum(x => x.wins);
                var l = list.Sum(x => x.losses);
                var net = list.Sum(x => x.netUsd);
                var dd = list.Count > 0 ? list.Max(x => x.maxDrawdownUsd) : 0m;
                var wrObs = (t > 0) ? (decimal)w / Math.Max(1, t) : 0m;
                var wrPost = BetaPosteriorMean(w, l, 7, 3); // mild optimistic prior
                var avgR = SafeAvg(list.Select(x => x.avgR));
                var score = Score(wrPost, avgR, net, dd);
                log.LogInformation("[Learn] {Strat} t={T} w={W} l={L} wr~{Wr:P1} avgR={AvgR:F2} net=${Net:F0} dd=${DD:F0} score={Score:F1}", strat, t, w, l, wrPost, avgR, net, dd, score);

                // Propose bounded overrides per strategy
                switch (strat)
                {
                    case "S2": ProposeS2(symbolRoot, t, wrPost, net, dd, log); break;
                    case "S3": ProposeS3(symbolRoot, t, wrPost, net, dd, log); break;
                    case "S6": ProposeS6(symbolRoot, t, wrPost, net, dd, log); break;
                    case "S11": ProposeS11(symbolRoot, t, wrPost, net, dd, log); break;
                }
            }

            // Persist best-of score for regression check
            try
            {
                var totalScore = byStrat.Values.SelectMany(v => v).GroupBy(s => 1).Select(g => Score(BetaPosteriorMean(g.Sum(x => x.wins), g.Sum(x => x.losses), 7, 3), SafeAvg(g.Select(x => x.avgR)), g.Sum(x => x.netUsd), g.Max(x => x.maxDrawdownUsd))).FirstOrDefault();
                SaveBest(symbolRoot, totalScore);
            }
            catch { /* best-effort */ }
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "[Learn] AdaptiveLearner failed");
        }
        await Task.CompletedTask.ConfigureAwait(false);
    }

    private static void ProposeS2(string root, int trades, decimal wr, decimal net, decimal dd, ILogger log)
    {
        var life = TimeSpan.FromDays(EnvInt("LEARN_TTL_DAYS", 1));
        var extra = new Dictionary<string, JsonElement>(); bool write;
        if (trades == 0 || wr < 0.42m)
        {
            write = true;
            extra["sigma_enter"] = JsonSerializer.SerializeToElement(trades == 0 ? 1.6m : 2.2m);
            extra["atr_enter"] = JsonSerializer.SerializeToElement(trades == 0 ? 0.8m : 1.2m);
            if (wr < 0.42m) extra["vwap_slope_guard"] = JsonSerializer.SerializeToElement(new { max_sigma_per_min = 0.10m });
        }
        else if (trades > 140 || net < -1500m)
        {
            write = true;
            extra["sigma_enter"] = JsonSerializer.SerializeToElement(2.2m);
            extra["atr_enter"] = JsonSerializer.SerializeToElement(1.2m);
            extra["adr_guard"] = JsonSerializer.SerializeToElement(new { len = 14, max_used = 0.60m, warn_used = 0.50m });
        }
        if (write) { ParamStore.SaveS2(root, extra, life); log.LogInformation("[Learn] S2 override TTL={TTL}d for {Root}", life.TotalDays, root); }
    }

    private static void ProposeS3(string root, int trades, decimal wr, decimal net, decimal dd, ILogger log)
    {
        var life = TimeSpan.FromDays(EnvInt("LEARN_TTL_DAYS", 1));
        if (trades == 0)
        {
            var cfg = new { width_rank_enter = 0.50m, min_squeeze_bars = 3, entry_mode = "retest", retest_backoff_ticks = 2, min_volume = 0 };
            ParamStore.SaveS3(root, JsonSerializer.Serialize(cfg), life);
            log.LogInformation("[Learn] S3 relax TTL={TTL}d for {Root}", life.TotalDays, root);
        }
        else if (trades > 100 || net < -1200m || wr < 0.44m)
        {
            var cfg = new { width_rank_enter = 0.25m, min_squeeze_bars = 6, entry_mode = "retest", retest_backoff_ticks = 4 };
            ParamStore.SaveS3(root, JsonSerializer.Serialize(cfg), life);
            log.LogInformation("[Learn] S3 throttle TTL={TTL}d for {Root}", life.TotalDays, root);
        }
    }

    private static void ProposeS6(string root, int trades, decimal wr, decimal net, decimal dd, ILogger log)
    {
        var life = TimeSpan.FromDays(EnvInt("LEARN_TTL_DAYS", 1));
        var extra = new Dictionary<string, JsonElement>(); bool write;
        if (trades == 0)
        {
            write = true; extra["min_atr"] = JsonSerializer.SerializeToElement(0.6m); extra["stop_mult"] = JsonSerializer.SerializeToElement(2.2m); extra["target_mult"] = JsonSerializer.SerializeToElement(4.0m);
        }
        else if (trades > 70 || net < -900m || wr < 0.46m)
        {
            write = true; extra["min_atr"] = JsonSerializer.SerializeToElement(0.9m); extra["stop_mult"] = JsonSerializer.SerializeToElement(2.5m); extra["target_mult"] = JsonSerializer.SerializeToElement(3.5m);
        }
        if (write) { ParamStore.SaveS6(root, extra, life); log.LogInformation("[Learn] S6 override TTL={TTL}d for {Root}", life.TotalDays, root); }
    }

    private static void ProposeS11(string root, int trades, decimal wr, decimal net, decimal dd, ILogger log)
    {
        var life = TimeSpan.FromDays(EnvInt("LEARN_TTL_DAYS", 1));
        var extra = new Dictionary<string, JsonElement>(); bool write;
        if (trades == 0)
        {
            write = true; extra["min_atr"] = JsonSerializer.SerializeToElement(0.8m); extra["stop_mult"] = JsonSerializer.SerializeToElement(3.2m); extra["target_mult"] = JsonSerializer.SerializeToElement(6.0m);
        }
        else if (trades > 70 || net < -1300m || wr < 0.44m)
        {
            write = true; extra["min_atr"] = JsonSerializer.SerializeToElement(1.1m); extra["stop_mult"] = JsonSerializer.SerializeToElement(3.5m); extra["target_mult"] = JsonSerializer.SerializeToElement(5.0m);
        }
        if (write) { ParamStore.SaveS11(root, extra, life); log.LogInformation("[Learn] S11 override TTL={TTL}d for {Root}", life.TotalDays, root); }
    }

    private static List<Summary> LoadRecentSummaries(string symbolRoot, int days)
    {
        var dir = Path.Combine(AppContext.BaseDirectory, "state", "backtest");
        var outList = new List<Summary>();
        try
        {
            if (!Directory.Exists(dir)) return outList;
            var cutoff = DateTime.UtcNow.AddDays(-days);
            foreach (var path in Directory.EnumerateFiles(dir, $"*-summary-{symbolRoot}-*.json"))
            {
                try
                {
                    var text = File.ReadAllText(path);
                    var s = JsonSerializer.Deserialize<Summary>(text);
                    if (s is null) continue;
                    if (s.end.ToUniversalTime() < cutoff) continue;
                    outList.Add(s);
                }
                catch { /* ignore malformed/partial */ }
            }
        }
        catch { /* ignore IO */ }
        return outList;
    }

    private static void SaveBest(string symbolRoot, decimal score)
    {
        try
        {
            var dir = Path.Combine(AppContext.BaseDirectory, "state", "learn");
            Directory.CreateDirectory(dir);
            var path = Path.Combine(dir, $"best-{symbolRoot}.json");
            var payload = new { asOfUtc = DateTime.UtcNow, score };
            File.WriteAllText(path, JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch { /* best-effort */ }
    }

    private static (DateTime asOfUtc, decimal score)? LoadBest(string symbolRoot)
    {
        try
        {
            var path = Path.Combine(AppContext.BaseDirectory, "state", "learn", $"best-{symbolRoot}.json");
            if (!File.Exists(path)) return null;
            var json = File.ReadAllText(path);
            using var doc = JsonDocument.Parse(json);
            var r = doc.RootElement;
            var asOf = r.TryGetProperty("asOfUtc", out var a) && a.ValueKind == JsonValueKind.String && DateTime.TryParse(a.GetString(), out var dt)
                ? dt.ToUniversalTime() : DateTime.UtcNow;
            var sc = r.TryGetProperty("score", out var s) && s.TryGetDecimal(out var v) ? v : 0m;
            return (asOf, sc);
        }
        catch { return null; }
    }

    private static decimal Score(decimal wrPosterior, decimal avgR, decimal netUsd, decimal ddUsd)
    {
        // WR dominates, avgR secondary, net/DD for tie-break; simple bounded function
        var wrPts = wrPosterior * 100m; // 0..100
        var rPts = Math.Clamp(avgR * 10m, -20m, 20m);
        var netPts = Math.Clamp(netUsd / 1000m, -30m, 30m);
        var ddPenalty = Math.Clamp(ddUsd / 2000m, 0m, 30m);
        return wrPts + rPts + netPts - ddPenalty;
    }

    private static decimal SafeAvg(IEnumerable<decimal> vals)
    {
        var list = vals.Where(v => v != 0m).ToList();
        if (list.Count == 0) return 0m;
        return list.Average();
    }

    private static decimal BetaPosteriorMean(int wins, int losses, int a, int b)
    {
        var w = Math.Max(0, wins); var l = Math.Max(0, losses); var aa = Math.Max(1, a); var bb = Math.Max(1, b);
        return (decimal)(w + aa) / (decimal)(w + l + aa + bb);
    }

    private static int EnvInt(string key, int def)
    {
        try { var v = Environment.GetEnvironmentVariable(key); if (!string.IsNullOrWhiteSpace(v) && int.TryParse(v, out var i)) return i; }
        catch { }
        return def;
    }

    private static bool EnvBool(string key, bool def)
    {
        try
        {
            var v = (Environment.GetEnvironmentVariable(key) ?? string.Empty).Trim().ToLowerInvariant();
            if (v is "1" or "true" or "yes") return true;
            if (v is "0" or "false" or "no") return false;
        }
        catch { }
        return def;
    }
}
