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
/// Minimal offline learner: reads recent backtest summaries under state/backtest and proposes
/// bounded S2 overrides. Applies with a short TTL in PAPER/SHADOW only when RUN_LEARNING=1.
/// </summary>
public static class AdaptiveLearner
{
    public sealed record Summary(string strategy, string symbol, DateTime start, DateTime end, int trades, int wins, int losses, decimal netUsd, decimal winRate, decimal avgR, decimal maxDrawdownUsd);

    public static async Task RunAsync(string symbolRoot, ILogger log, CancellationToken ct)
    {
        try
        {
            var run = (Environment.GetEnvironmentVariable("RUN_LEARNING") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            var liveOrders = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (!run)
            {
                log.LogInformation("[Learn] RUN_LEARNING disabled.");
                return;
            }
            if (liveOrders)
            {
                log.LogInformation("[Learn] Live orders enabled — skipping automatic apply.");
                return;
            }

            var summaries = LoadRecentSummaries(symbolRoot, days: 7);
            var s2 = summaries.Where(s => string.Equals(s.strategy, "S2", StringComparison.OrdinalIgnoreCase)).ToList();
            var s3 = summaries.Where(s => string.Equals(s.strategy, "S3", StringComparison.OrdinalIgnoreCase)).ToList();
            var s6 = summaries.Where(s => string.Equals(s.strategy, "S6", StringComparison.OrdinalIgnoreCase)).ToList();
            var s11 = summaries.Where(s => string.Equals(s.strategy, "S11", StringComparison.OrdinalIgnoreCase)).ToList();
            if (s2.Count == 0)
            {
                log.LogInformation("[Learn] No S2 summaries found for {Sym}", symbolRoot);
                return;
            }

            // Objective gate: compute a simple score over the window and block regressive proposals
            decimal SumNet(List<Summary> list) => list.Sum(x => x.netUsd);
            decimal MaxDd(List<Summary> list) => list.Count == 0 ? 0m : list.Max(x => x.maxDrawdownUsd);
            var netAll = SumNet(s2) + SumNet(s3) + SumNet(s6) + SumNet(s11);
            var ddAll = new[] { MaxDd(s2), MaxDd(s3), MaxDd(s6), MaxDd(s11) }.Max();
            var score = netAll - 0.25m * ddAll; // profit minus risk penalty
            var best = LoadBest(symbolRoot);
            if (best is not null && score < best.Value.score)
            {
                log.LogInformation("[Learn] Regressive window (score {Score:F0} < best {Best:F0}) — skipping proposals", score, best.Value.score);
                return;
            }

            var trades = s2.Sum(x => x.trades);
            var net = s2.Sum(x => x.netUsd);
            var dd = s2.Count > 0 ? s2.Max(x => x.maxDrawdownUsd) : 0m;
            log.LogInformation("[Learn] S2 window: trades={Trades} net=${Net:F2} maxDD=${DD:F0}", trades, net, dd);

            var extra = new Dictionary<string, JsonElement>();
            bool propose = false;

            if (trades == 0)
            {
                // Zero-trade relaxer: nudge sigma/atr down within safe bounds
                propose = true;
                extra["sigma_enter"] = JsonSerializer.SerializeToElement(1.6m);
                extra["atr_enter"] = JsonSerializer.SerializeToElement(0.8m);
                extra["min_volume"] = JsonSerializer.SerializeToElement(0);
                log.LogInformation("[Learn] Proposing relaxed S2 (zero-trade window).");
            }
            else if (trades > 150 || net < -2000m)
            {
                // Over-trade throttler: tighten entry and enable ADR guard
                propose = true;
                extra["sigma_enter"] = JsonSerializer.SerializeToElement(2.2m);
                extra["atr_enter"] = JsonSerializer.SerializeToElement(1.2m);
                extra["adr_guard"] = JsonSerializer.SerializeToElement(new { len = 14, max_used = 0.60m, warn_used = 0.50m });
                extra["vwap_slope_guard"] = JsonSerializer.SerializeToElement(new { max_sigma_per_min = 0.10m });
                log.LogInformation("[Learn] Proposing throttled S2 (over-trade/losing window).");
            }

            if (propose)
            {
                // Apply with a short TTL (1 day) so it auto-rolls back without renewal
                ParamStore.SaveS2(symbolRoot, extra, ttl: TimeSpan.FromDays(1));
                log.LogInformation("[Learn] Wrote S2 override with TTL 1d for {Sym}.", symbolRoot);
            }

            // --- S3: relax when starved; throttle when over-trading/losing ---
            if (s3.Count > 0)
            {
                var t = s3.Sum(x => x.trades);
                var n = s3.Sum(x => x.netUsd);
                var dmax = s3.Max(x => x.maxDrawdownUsd);
                log.LogInformation("[Learn] S3 window: trades={Trades} net=${Net:F2} maxDD=${DD:F0}", t, n, dmax);

                if (t == 0)
                {
                    var cfg = new
                    {
                        // widen entry tolerance and ease gates
                        width_rank_enter = 0.55m,
                        min_squeeze_bars = 3,
                        min_volume = 0,
                        attempt_cap = new { RTH = 2, overnight = 1 },
                        rs_filter = new { enabled = false, window_bars = 60, threshold = 0.10m, directional_only = false },
                        entry_mode = "retest",
                        retest_backoff_ticks = 2
                    };
                    var json = JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = false });
                    ParamStore.SaveS3(symbolRoot, json, TimeSpan.FromDays(1));
                    log.LogInformation("[Learn] Wrote S3 relax override (zero-trade) TTL 1d for {Sym}", symbolRoot);
                }
                else if (t > 120 || n < -1500m)
                {
                    var cfg = new
                    {
                        width_rank_enter = 0.25m,
                        min_squeeze_bars = 6,
                        attempt_cap = new { RTH = 1, overnight = 0 },
                        rs_filter = new { enabled = true, window_bars = 80, threshold = 0.20m, directional_only = true },
                        entry_mode = "retest",
                        retest_backoff_ticks = 4
                    };
                    var json = JsonSerializer.Serialize(cfg, new JsonSerializerOptions { WriteIndented = false });
                    ParamStore.SaveS3(symbolRoot, json, TimeSpan.FromDays(1));
                    log.LogInformation("[Learn] Wrote S3 throttle override TTL 1d for {Sym}", symbolRoot);
                }
            }

            // --- S6: ATR threshold and risk tweaks ---
            if (s6.Count > 0)
            {
                var t = s6.Sum(x => x.trades);
                var n = s6.Sum(x => x.netUsd);
                var extra6 = new Dictionary<string, JsonElement>();
                bool write6 = false;
                if (t == 0)
                {
                    write6 = true;
                    extra6["min_atr"] = JsonSerializer.SerializeToElement(0.6m);
                    extra6["stop_mult"] = JsonSerializer.SerializeToElement(2.2m);
                    extra6["target_mult"] = JsonSerializer.SerializeToElement(4.0m);
                    log.LogInformation("[Learn] Proposing S6 relax (zero-trade).");
                }
                else if (t > 80 || n < -1000m)
                {
                    write6 = true;
                    extra6["min_atr"] = JsonSerializer.SerializeToElement(0.9m);
                    extra6["stop_mult"] = JsonSerializer.SerializeToElement(2.5m);
                    extra6["target_mult"] = JsonSerializer.SerializeToElement(3.5m);
                    log.LogInformation("[Learn] Proposing S6 throttle (over-trade/losing).");
                }
                if (write6)
                {
                    ParamStore.SaveS6(symbolRoot, extra6, TimeSpan.FromDays(1));
                    log.LogInformation("[Learn] Wrote S6 override TTL 1d for {Sym}", symbolRoot);
                }
            }

            // --- S11: ATR threshold and risk tweaks ---
            if (s11.Count > 0)
            {
                var t = s11.Sum(x => x.trades);
                var n = s11.Sum(x => x.netUsd);
                var extra11 = new Dictionary<string, JsonElement>();
                bool write11 = false;
                if (t == 0)
                {
                    write11 = true;
                    extra11["min_atr"] = JsonSerializer.SerializeToElement(0.8m);
                    extra11["stop_mult"] = JsonSerializer.SerializeToElement(3.2m);
                    extra11["target_mult"] = JsonSerializer.SerializeToElement(6.0m);
                    log.LogInformation("[Learn] Proposing S11 relax (zero-trade).");
                }
                else if (t > 80 || n < -1500m)
                {
                    write11 = true;
                    extra11["min_atr"] = JsonSerializer.SerializeToElement(1.1m);
                    extra11["stop_mult"] = JsonSerializer.SerializeToElement(3.5m);
                    extra11["target_mult"] = JsonSerializer.SerializeToElement(5.0m);
                    log.LogInformation("[Learn] Proposing S11 throttle (over-trade/losing).");
                }
                if (write11)
                {
                    ParamStore.SaveS11(symbolRoot, extra11, TimeSpan.FromDays(1));
                    log.LogInformation("[Learn] Wrote S11 override TTL 1d for {Sym}", symbolRoot);
                }
            }

            // Persist best score snapshot (weekly best-of)
            try { SaveBest(symbolRoot, score); } catch { }
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "[Learn] AdaptiveLearner failed");
        }
        await Task.CompletedTask;
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
                catch (Exception)
                {
                    // Best-effort parsing; ignore malformed or partial files
                }
            }
        }
        catch (Exception)
        {
            // Best-effort directory scan; ignore IO errors
        }
        return outList;
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
            var asOf = r.TryGetProperty("asOfUtc", out var a) && a.ValueKind == JsonValueKind.String && DateTime.TryParse(a.GetString(), out var dt) ? dt.ToUniversalTime() : DateTime.UtcNow;
            var sc = r.TryGetProperty("score", out var s) && s.TryGetDecimal(out var v) ? v : 0m;
            return (asOf, sc);
        }
        catch { return null; }
    }

    private static void SaveBest(string symbolRoot, decimal score)
    {
        var dir = Path.Combine(AppContext.BaseDirectory, "state", "learn");
        Directory.CreateDirectory(dir);
        var path = Path.Combine(dir, $"best-{symbolRoot}.json");
        var payload = new { asOfUtc = DateTime.UtcNow, score };
        var json = JsonSerializer.Serialize(payload, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(path, json);
    }
}
