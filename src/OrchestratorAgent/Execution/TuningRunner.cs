using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Config;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using Microsoft.Extensions.Logging;

namespace OrchestratorAgent.Execution;

public static class TuningRunner
{
    public sealed record Param(string Key, decimal? D = null, int? I = null, bool? B = null, string? S = null)
    {
        public void Apply(Dictionary<string, JsonElement> extra)
        {
            if (D.HasValue) extra[Key] = JsonSerializer.SerializeToElement(D.Value);
            else if (I.HasValue) extra[Key] = JsonSerializer.SerializeToElement(I.Value);
            else if (B.HasValue) extra[Key] = JsonSerializer.SerializeToElement(B.Value);
            else if (S != null) extra[Key] = JsonSerializer.SerializeToElement(S);
        }
    }

    // Helper: parse comma-separated decimals from env with invariant culture and sane fallbacks
    private static decimal[] ParseDecArray(string? raw, decimal[] defaults)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(raw)) return defaults;
            var parts = raw.Split(new char[] { ',', ';', '|' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var list = new List<decimal>(parts.Length);
            foreach (var p in parts)
            {
                if (decimal.TryParse(p, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out var d))
                    list.Add(d);
            }
            return list.Count > 0 ? [.. list] : defaults;
        }
        catch { return defaults; }
    }

    private static int[] ParseIntArray(string? raw, int[] defaults)
    {
        try
        {
            if (string.IsNullOrWhiteSpace(raw)) return defaults;
            var parts = raw.Split(new char[] { ',', ';', '|' }, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            var list = new List<int>(parts.Length);
            foreach (var p in parts)
            {
                if (int.TryParse(p, out var i)) list.Add(i);
            }
            return list.Count > 0 ? [.. list] : defaults;
        }
        catch { return defaults; }
    }

    public sealed record TrialConfig(List<Param> Params)
    {
        public StrategyDef BuildStrategyDef()
        {
            var def = new StrategyDef { Id = "S2", Name = "Session VWAP Mean-Reversion", Enabled = true, Family = "meanrev" };
            foreach (var p in Params) p.Apply(def.Extra);
            // Enable nested guards when relevant keys present
            if (Params.Any(p => p.Key.StartsWith("adr_guard.") || p.Key.Equals("adr_guard")))
            {
                var obj = new Dictionary<string, object?>();
                int len = 14; decimal maxUsed = 0.75m; decimal warn = 0.60m;
                var lenP = Params.FirstOrDefault(p => p.Key == "adr_guard.len"); if (lenP?.I is int i1) len = i1;
                var maxP = Params.FirstOrDefault(p => p.Key == "adr_guard.max_used"); if (maxP?.D is decimal d1) maxUsed = d1;
                var warnP = Params.FirstOrDefault(p => p.Key == "adr_guard.warn_used"); if (warnP?.D is decimal d2) warn = d2;
                def.Extra["adr_guard"] = JsonSerializer.SerializeToElement(new { len, max_used = maxUsed, warn_used = warn });
            }
            if (Params.Any(p => p.Key.StartsWith("vwap_slope_guard.") || p.Key.Equals("vwap_slope_guard")))
            {
                decimal msm = 0.12m;
                var p = Params.FirstOrDefault(x => x.Key == "vwap_slope_guard.max_sigma_per_min");
                if (p?.D is decimal d) msm = d;
                def.Extra["vwap_slope_guard"] = JsonSerializer.SerializeToElement(new { max_sigma_per_min = msm });
            }
            if (Params.Any(p => p.Key.StartsWith("z_decelerate.") || p.Key.Equals("z_decelerate")))
            {
                int need = 2;
                var p = Params.FirstOrDefault(x => x.Key == "z_decelerate.need");
                if (p?.I is int i) need = i;
                def.Extra["z_decelerate"] = JsonSerializer.SerializeToElement(new { need });
            }
            return def;
        }
        public override string ToString() => string.Join(", ", Params.Select(p => p.Key + "=" + (p.D?.ToString() ?? p.I?.ToString() ?? p.B?.ToString() ?? p.S ?? "")));
    }

    public sealed record TrialResult(TrialConfig Config, int Trades, int Wins, int Losses, decimal NetUsd, decimal WinRate, decimal AvgR, decimal MaxDrawdownUsd)
    {
        public override string ToString() => $"trades={Trades} win%={WinRate:P1} net=${NetUsd:F2} avgR={AvgR:F2} maxDD=${MaxDrawdownUsd:F0} :: {Config}";
    }

    public static async Task RunS2Async(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        // 1) Fetch 1m bars for the requested window via /api/History/retrieveBars
        log.LogInformation("[Tune:S2] Fetching bars for {Cid} {From:u} → {To:u}…", contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Tune:S2] Bars fetched: {N}", bars.Count);
        if (bars.Count < 120)
        {
            log.LogWarning("[Tune] Not enough bars: {N}", bars.Count);
            return;
        }

        // 2) Risk setup (sizing)
        var risk = new RiskEngine();
        try
        {
            var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE");
            if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v;
        }
        catch { }

        // 3) Parameter grid (keep small for speed; expand later)
        var grids = new List<TrialConfig>();
        decimal[] sigmaEnter = ParseDecArray(Environment.GetEnvironmentVariable("S2_GRID_SIGMA_ENTER"), [1.8m, 2.0m, 2.2m]);
        decimal[] atrEnter = ParseDecArray(Environment.GetEnvironmentVariable("S2_GRID_ATR_ENTER"), [0.8m, 1.0m, 1.2m]);
        int[] retestTicks = ParseIntArray(Environment.GetEnvironmentVariable("S2_GRID_RETEST_TICKS"), [0, 1, 2]);
        decimal[] vwapSlopeMax = ParseDecArray(Environment.GetEnvironmentVariable("S2_GRID_VWAP_SLOPE_MAX"), [0.10m, 0.12m, 0.15m]);
        decimal[] adrUsedMax = ParseDecArray(Environment.GetEnvironmentVariable("S2_GRID_ADR_USED_MAX"), [0.0m, 0.60m, 0.75m]); // 0 disables guard

        foreach (var se in sigmaEnter)
            foreach (var ae in atrEnter)
                foreach (var rt in retestTicks)
                    foreach (var vs in vwapSlopeMax)
                        foreach (var am in adrUsedMax)
                        {
                            var ps = new List<Param>
            {
                new("sigma_enter", se),
                new("atr_enter", ae),
                new("entry_mode", S: "retest"),
                new("retest_offset_ticks", I: rt),
                new("vwap_slope_guard.max_sigma_per_min", vs),
            };
                            if (am > 0)
                            {
                                ps.Add(new("adr_guard.len", I: 14));
                                ps.Add(new("adr_guard.max_used", am));
                                ps.Add(new("adr_guard.warn_used", 0.60m));
                            }
                            grids.Add(new TrialConfig(ps));
                        }

        // 4) Run trials
        var results = new List<TrialResult>(grids.Count);
        int trialIndex = 0;
        foreach (var cfg in grids)
        {
            trialIndex++;
            if (trialIndex == 1 || trialIndex % Math.Max(1, grids.Count / 10) == 0)
                log.LogInformation("[Tune:S2] Trial {Idx}/{Total}…", trialIndex, grids.Count);
            var def = cfg.BuildStrategyDef();
            S2RuntimeConfig.ApplyFrom(def);

            var env = new Env { Symbol = symbolRoot };
            var levels = new Levels();

            // Backtest loop: single open trade at a time
            Trade? open = null!;
            var equity = new List<decimal> { 0m };
            int wins = 0, losses = 0, trades = 0;

            var history = new List<Bar>(Math.Min(5000, bars.Count));
            foreach (var b in bars)
            {
                // Update env ATR crudely from last bar's range
                try { env.atr = Math.Abs(b.High - b.Low); } catch { }

                // Manage open trade
                if (open != null)
                {
                    if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                    {
                        trades++;
                        if (won) wins++; else losses++;
                        var lastEq = equity[^1]; equity.Add(lastEq + pnlUsd);
                        open = null!;
                    }
                }

                // Append bar and generate on rolling history for indicator context
                history.Add(b);
                var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
                var s2 = signals.Where(s => string.Equals(s.StrategyId, "S2", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                if (s2 != null && open == null)
                {
                    open = new Trade(s2.Side, s2.Entry, s2.Stop, s2.Target, s2.Size);
                }
            }

            // Score
            var net = equity[^1];
            var maxDd = MaxDrawdown(equity);
            var wr = trades > 0 ? (decimal)wins / trades : 0m;
            var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
            results.Add(new TrialResult(cfg, trades, wins, losses, net, wr, avgR, maxDd));
        }

        // 5) Pick best by explicit profit objective (risk-aware)
        var weights = BotCore.Objectives.ProfitObjective.FromEnv();
        var scored = results
            .Select(r => new
            {
                Res = r,
                Score = BotCore.Objectives.ProfitObjective.Score(
                    new BotCore.Objectives.ProfitObjective.Metrics(r.NetUsd, r.MaxDrawdownUsd, r.Trades, r.WinRate, r.AvgR),
                    weights)
            })
            .OrderByDescending(x => x.Score)
            .ToList();
        var best = scored.FirstOrDefault(x => x.Score > decimal.MinusOne)?.Res
                   ?? results.OrderByDescending(r => r.NetUsd).First();
        log.LogInformation("[Tune] Best (objective): {Best}", best.ToString());

        // 6) Persist tuned profile copy under state/tuning
        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"S2-best-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(best, new JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);

            // Also emit a tuned profile copy if the base profile exists
            var basePath = "src\\BotCore\\Config\\high_win_rate_profile.json";
            if (File.Exists(basePath))
            {
                var profile = ConfigLoader.FromFile(basePath);
                var s2def = profile.Strategies.FirstOrDefault(s => string.Equals(s.Id, "S2", StringComparison.OrdinalIgnoreCase));
                if (s2def != null)
                {
                    foreach (var p in best.Config.Params) p.Apply(s2def.Extra);
                    var tunedPath = Path.Combine(outDir, "high_win_rate_profile.tuned.json");
                    var json = JsonSerializer.Serialize(profile, new JsonSerializerOptions { WriteIndented = true });
                    await File.WriteAllTextAsync(tunedPath, json, ct).ConfigureAwait(false);
                    log.LogInformation("[Tune] Wrote tuned profile: {Path}", tunedPath);
                }
            }
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "[Tune] Persist tuned output failed");
        }

        // Optional: promote best S2 into ParamStore for immediate runtime use
        try
        {
            var promote = (Environment.GetEnvironmentVariable("PROMOTE_TUNER") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (promote)
            {
                int ttlHours = 24;
                var rawH = Environment.GetEnvironmentVariable("PROMOTE_TTL_HOURS");
                var rawD = Environment.GetEnvironmentVariable("PROMOTE_TTL_DAYS");
                if (!string.IsNullOrWhiteSpace(rawH) && int.TryParse(rawH, out var th) && th > 0) ttlHours = th;
                else if (!string.IsNullOrWhiteSpace(rawD) && int.TryParse(rawD, out var td) && td > 0) ttlHours = td * 24;

                var def = best.Config.BuildStrategyDef();
                ParamStore.SaveS2(symbolRoot, def.Extra, TimeSpan.FromHours(ttlHours));
                log.LogInformation("[Tune:S2] PROMOTE_TUNER on — wrote S2 override (TTL={Ttl}h) to ParamStore for {Sym}", ttlHours, symbolRoot);
            }
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "[Tune:S2] Promote-to-ParamStore failed");
        }
    }

    // Lightweight single-run backtest of S2 using the current configured profile; prints trades and PnL
    public static async Task RunS2SummaryAsync(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        log.LogInformation("[Backtest:S2] Fetching bars for {Cid} {From:u} → {To:u}…", contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Backtest:S2] Bars fetched: {N}", bars.Count);
        if (bars.Count < 120)
        {
            log.LogWarning("[Backtest:S2] Not enough bars: {N}", bars.Count);
            return;
        }

        // Risk
        var risk = new RiskEngine();
        try
        {
            var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE");
            if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v;
        }
        catch { }

        // Apply S2 from profile if available
        try
        {
            var basePath = "src\\BotCore\\Config\\high_win_rate_profile.json";
            if (File.Exists(basePath))
            {
                var profile = ConfigLoader.FromFile(basePath);
                var s2def = profile.Strategies.FirstOrDefault(s => string.Equals(s.Id, "S2", StringComparison.OrdinalIgnoreCase));
                if (s2def != null) S2RuntimeConfig.ApplyFrom(s2def);
            }
        }
        catch { }

        // Optional: relax S2 guards for summary runs to diagnose zero-trade cases
        try
        {
            var relax = (Environment.GetEnvironmentVariable("TUNE_RELAX") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (relax)
            {
                var def = new StrategyDef
                {
                    Name = "S2",
                    Enabled = true,
                    Extra = []
                };
                // Build a simple relaxed config payload
                using var doc = System.Text.Json.JsonDocument.Parse("{\n  \"min_volume\": 0,\n  \"sigma_enter\": 1.5,\n  \"atr_enter\": 0.8,\n  \"confirm_lookback\": 2,\n  \"retest_offset_ticks\": 0,\n  \"volz\": { \"min\": -5, \"max\": 5 }\n}");
                foreach (var p in doc.RootElement.EnumerateObject())
                    def.Extra[p.Name] = p.Value.Clone();
                S2RuntimeConfig.ApplyFrom(def);
                log.LogInformation("[Backtest:S2] Applied TUNE_RELAX overrides.");
            }
        }
        catch { }

        var env = new Env { Symbol = symbolRoot };
        var levels = new Levels();
        Trade? open = null!;
        var equity = new List<decimal> { 0m };
        int wins = 0, losses = 0, trades = 0;
        var history = new List<Bar>(Math.Min(5000, bars.Count));
        int barCounter = 0;

        foreach (var b in bars)
        {
            try { env.atr = Math.Abs(b.High - b.Low); } catch { }
            if (open != null)
            {
                if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                {
                    trades++;
                    if (won) wins++; else losses++;
                    var lastEq = equity[^1]; equity.Add(lastEq + pnlUsd);
                    open = null!;
                }
            }
            history.Add(b);
            var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
            var s2 = signals.Where(s => string.Equals(s.StrategyId, "S2", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
            if (s2 != null && open == null)
            {
                open = new Trade(s2.Side, s2.Entry, s2.Stop, s2.Target, s2.Size);
            }
            barCounter++;
            if (barCounter % 200 == 0)
            {
                log.LogInformation("[Backtest:S2] Progress {Bars} bars → trades={Trades} net=${Net:F2}", barCounter, trades, equity[^1]);
            }
        }

        var net = equity[^1];
        var maxDd = MaxDrawdown(equity);
        var wr = trades > 0 ? (decimal)wins / trades : 0m;
        var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
        log.LogInformation("[Backtest:S2] Done symbol={Sym} days={Days} trades={Trades} win%={Wr:P1} net=${Net:F2} avgR={AvgR:F2} maxDD=${DD:F0}", symbolRoot, (utcEnd - utcStart).TotalDays, trades, wr, net, avgR, maxDd);

        // Persist compact summary JSON
        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "backtest");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"S2-summary-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            var summary = new
            {
                strategy = "S2",
                symbol = symbolRoot,
                start = utcStart,
                end = utcEnd,
                trades,
                wins,
                losses,
                netUsd = net,
                winRate = wr,
                avgR,
                maxDrawdownUsd = maxDd
            };
            await File.WriteAllTextAsync(outPath, System.Text.Json.JsonSerializer.Serialize(summary, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);
            log.LogInformation("[Backtest:S2] Wrote summary: {Path}", outPath);
        }
        catch { }
    }

    public static async Task RunS3Async(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        // 1) Fetch 1m bars for the requested window
        log.LogInformation("[Tune:S3] Fetching bars for {Cid} {From:u} → {To:u}…", contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Tune:S3] Bars fetched: {N}", bars.Count);
        if (bars.Count < 200)
        {
            log.LogWarning("[Tune:S3] Not enough bars: {N}", bars.Count);
            return;
        }

        // 2) Risk setup (sizing)
        var risk = new RiskEngine();
        try
        {
            var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE");
            if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v;
        }
        catch { }

        // 3) Parameter grid for S3 (keep compact for speed)
        var grids = new List<TrialConfig>();
        decimal[] widthRankEnter = ParseDecArray(Environment.GetEnvironmentVariable("S3_GRID_WIDTH_RANK"), [0.10m, 0.15m, 0.20m]);
        int[] minSqueezeBars = ParseIntArray(Environment.GetEnvironmentVariable("S3_GRID_MIN_SQZ_BARS"), [5, 6, 8]);
        decimal[] confirmBreakMult = ParseDecArray(Environment.GetEnvironmentVariable("S3_GRID_CONFIRM_MULT"), [0.10m, 0.15m, 0.20m]);
        decimal[] stopAtrMult = ParseDecArray(Environment.GetEnvironmentVariable("S3_GRID_STOP_ATR_MULT"), [1.0m, 1.1m, 1.2m]);
        int[] retestBackoffTicks = ParseIntArray(Environment.GetEnvironmentVariable("S3_GRID_RETEST_BACKOFF"), [1, 2]);
        decimal[] rsThreshold = ParseDecArray(Environment.GetEnvironmentVariable("S3_GRID_RS_THRESHOLD"), [0.05m, 0.10m]);

        foreach (var wre in widthRankEnter)
            foreach (var msb in minSqueezeBars)
                foreach (var cbm in confirmBreakMult)
                    foreach (var sam in stopAtrMult)
                        foreach (var rbt in retestBackoffTicks)
                            foreach (var rst in rsThreshold)
                            {
                                var ps = new List<Param>
            {
                new("width_rank_enter", wre),
                new("min_squeeze_bars", I: msb),
                new("confirm_break_mult", cbm),
                new("stop_atr_mult", sam),
                new("entry_mode", S: "retest"),
                new("retest_backoff_ticks", I: rbt),
                new("rs_filter.threshold", rst),
                new("rs_filter.enabled", B: true),
            };
                                grids.Add(new TrialConfig(ps));
                            }

        // 4) Run trials
        var results = new List<TrialResult>(grids.Count);
        int trialIndex = 0;
        foreach (var cfg in grids)
        {
            trialIndex++;
            if (trialIndex == 1 || trialIndex % Math.Max(1, grids.Count / 10) == 0)
                log.LogInformation("[Tune:S3] Trial {Idx}/{Total}…", trialIndex, grids.Count);
            // Build transient S3 JSON and apply
            var s3Json = BuildS3ConfigJson(cfg);
            try { BotCore.Strategy.S3Strategy.ApplyTuningJson(s3Json); } catch { }

            var env = new Env { Symbol = symbolRoot };
            var levels = new Levels();
            Trade? open = null!;
            var equity = new List<decimal> { 0m };
            int wins = 0, losses = 0, trades = 0;
            var history = new List<Bar>(Math.Min(5000, bars.Count));
            foreach (var b in bars)
            {
                try { env.atr = Math.Abs(b.High - b.Low); } catch { }
                if (open != null)
                {
                    if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                    {
                        trades++;
                        if (won) wins++; else losses++;
                        var lastEq = equity[^1]; equity.Add(lastEq + pnlUsd);
                        open = null!;
                    }
                }
                history.Add(b);
                var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
                var s3 = signals.Where(s => string.Equals(s.StrategyId, "S3", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                if (s3 != null && open == null)
                {
                    open = new Trade(s3.Side, s3.Entry, s3.Stop, s3.Target, s3.Size);
                }
            }
            var net = equity[^1];
            var maxDd = MaxDrawdown(equity);
            var wr = trades > 0 ? (decimal)wins / trades : 0m;
            var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
            results.Add(new TrialResult(cfg, trades, wins, losses, net, wr, avgR, maxDd));
        }

        // 5) Pick best by explicit profit objective (risk-aware)
        var weights = BotCore.Objectives.ProfitObjective.FromEnv();
        var scored = results
            .Select(r => new
            {
                Res = r,
                Score = BotCore.Objectives.ProfitObjective.Score(
                    new BotCore.Objectives.ProfitObjective.Metrics(r.NetUsd, r.MaxDrawdownUsd, r.Trades, r.WinRate, r.AvgR),
                    weights)
            })
            .OrderByDescending(x => x.Score)
            .ToList();
        var best = scored.FirstOrDefault(x => x.Score > decimal.MinusOne)?.Res
                   ?? results.OrderByDescending(r => r.NetUsd).First();
        log.LogInformation("[Tune:S3] Best (objective): {Best}", best.ToString());

        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"S3-best-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(best, new JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);

            // Emit a tuned S3-StrategyConfig JSON to use via S3_CONFIG_PATH
            var tunedCfgPath = Path.Combine(outDir, $"S3-StrategyConfig.tuned.{symbolRoot}.json");
            await File.WriteAllTextAsync(tunedCfgPath, BuildS3ConfigJson(best.Config), ct).ConfigureAwait(false);
            log.LogInformation("[Tune:S3] Wrote tuned S3 config: {Path}", tunedCfgPath);
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "[Tune:S3] Persist tuned output failed");
        }

        // Optional: promote best S3 into ParamStore for immediate runtime use
        try
        {
            var promote = (Environment.GetEnvironmentVariable("PROMOTE_TUNER") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (promote)
            {
                int ttlHours = 24;
                var rawH = Environment.GetEnvironmentVariable("PROMOTE_TTL_HOURS");
                var rawD = Environment.GetEnvironmentVariable("PROMOTE_TTL_DAYS");
                if (!string.IsNullOrWhiteSpace(rawH) && int.TryParse(rawH, out var th) && th > 0) ttlHours = th;
                else if (!string.IsNullOrWhiteSpace(rawD) && int.TryParse(rawD, out var td) && td > 0) ttlHours = td * 24;

                var json = BuildS3ConfigJson(best.Config);
                ParamStore.SaveS3(symbolRoot, json, TimeSpan.FromHours(ttlHours));
                log.LogInformation("[Tune:S3] PROMOTE_TUNER on — wrote S3 override (TTL={Ttl}h) to ParamStore for {Sym}", ttlHours, symbolRoot);
            }
        }
        catch (Exception ex)
        {
            log.LogWarning(ex, "[Tune:S3] Promote-to-ParamStore failed");
        }
    }

    // Lightweight single-run backtest of S3 using current defaults; prints trades and PnL
    public static async Task RunS3SummaryAsync(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        log.LogInformation("[Backtest:S3] Fetching bars for {Cid} {From:u} → {To:u}…", contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Backtest:S3] Bars fetched: {N}", bars.Count);
        if (bars.Count < 200)
        {
            log.LogWarning("[Backtest:S3] Not enough bars: {N}", bars.Count);
            return;
        }

        var risk = new RiskEngine();
        try
        {
            var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE");
            if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v;
        }
        catch { }

        // Optionally apply S3 tuned config for summary backtests
        try
        {
            var s3cfg = Environment.GetEnvironmentVariable("S3_CONFIG_PATH");
            if (!string.IsNullOrWhiteSpace(s3cfg) && File.Exists(s3cfg))
            {
                var json = await File.ReadAllTextAsync(s3cfg, ct).ConfigureAwait(false).ConfigureAwait(false);
                try { BotCore.Strategy.S3Strategy.ApplyTuningJson(json); } catch { }
            }
        }
        catch { }

        // Optional: relax S3 guards for summary runs
        try
        {
            var relax = (Environment.GetEnvironmentVariable("TUNE_RELAX") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (relax)
            {
                var relaxed = new System.Collections.Generic.Dictionary<string, object?>
                {
                    { "min_volume", 0 },
                    { "width_rank_enter", 0.10m },
                    { "min_squeeze_bars", 3 },
                    { "confirm_break_mult", 0.10m },
                    { "retest_backoff_ticks", 1 },
                    { "entry_mode", "retest" }
                };
                var rs = new System.Collections.Generic.Dictionary<string, object?>
                {
                    { "enabled", false },
                    { "threshold", 0.05m }
                };
                relaxed.Add("rs_filter", rs);
                var json = System.Text.Json.JsonSerializer.Serialize(relaxed);
                try { BotCore.Strategy.S3Strategy.ApplyTuningJson(json); } catch { }
                log.LogInformation("[Backtest:S3] Applied TUNE_RELAX overrides.");
            }
        }
        catch { }

        var env = new Env { Symbol = symbolRoot };
        var levels = new Levels();
        Trade? open = null!;
        var equity = new List<decimal> { 0m };
        int wins = 0, losses = 0, trades = 0;
        var history = new List<Bar>(Math.Min(5000, bars.Count));
        int barCounter = 0;

        // Optional per-guard diagnostics
        var dbg = Environment.GetEnvironmentVariable("S3_DEBUG_REASONS");
        var dbgOn = string.Equals(dbg, "1", StringComparison.OrdinalIgnoreCase) || string.Equals(dbg, "true", StringComparison.OrdinalIgnoreCase);
        if (dbgOn)
        {
            try { BotCore.Strategy.S3Strategy.ResetDebugCounters(); } catch { }
        }

        foreach (var b in bars)
        {
            try { env.atr = Math.Abs(b.High - b.Low); } catch { }
            if (open != null)
            {
                if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                {
                    trades++; if (won) wins++; else losses++;
                    var lastEq = equity[^1]; equity.Add(lastEq + pnlUsd);
                    open = null!;
                }
            }
            history.Add(b);
            var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
            var s3 = signals.Where(s => string.Equals(s.StrategyId, "S3", StringComparison.OrdinalIgnoreCase))
                            .OrderByDescending(s => s.ExpR)
                            .FirstOrDefault();
            if (s3 != null && open == null)
            {
                open = new Trade(s3.Side, s3.Entry, s3.Stop, s3.Target, s3.Size);
            }
            barCounter++;
            if (barCounter % 200 == 0)
                log.LogInformation("[Backtest:S3] Progress {Bars} bars → trades={Trades} net=${Net:F2}", barCounter, trades, equity[^1]);
        }

        var net = equity[^1];
        var maxDd = MaxDrawdown(equity);
        var wr = trades > 0 ? (decimal)wins / trades : 0m;
        var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
        log.LogInformation("[Backtest:S3] Done symbol={Sym} days={Days} trades={Trades} win%={Wr:P1} net=${Net:F2} avgR={AvgR:F2} maxDD=${DD:F0}", symbolRoot, (utcEnd - utcStart).TotalDays, trades, wr, net, avgR, maxDd);

        // If requested, print top reject reasons to help diagnose zero-trade runs
        if (dbgOn)
        {
            try
            {
                var counts = BotCore.Strategy.S3Strategy.GetDebugCounters();
                if (counts != null && counts.Count > 0)
                {
                    var top = counts.OrderByDescending(kv => kv.Value).Take(12).ToList();
                    log.LogInformation("[Backtest:S3] Top blockers ({Count} unique):", counts.Count);
                    foreach (var kv in top)
                        log.LogInformation("[Backtest:S3]   {Reason} = {Hits}", kv.Key, kv.Value);
                }
                else
                {
                    log.LogInformation("[Backtest:S3] No reject counters recorded.");
                }
            }
            catch { }
        }

        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "backtest");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"S3-summary-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            var summary = new { strategy = "S3", symbol = symbolRoot, start = utcStart, end = utcEnd, trades, wins, losses, netUsd = net, winRate = wr, avgR, maxDrawdownUsd = maxDd };
            await File.WriteAllTextAsync(outPath, System.Text.Json.JsonSerializer.Serialize(summary, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);
            log.LogInformation("[Backtest:S3] Wrote summary: {Path}", outPath);
        }
        catch { }
    }

    // Grid tuning for S6 (trend/open bias) using S6RuntimeConfig knobs
    public static async Task RunS6Async(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        log.LogInformation("[Tune:S6] Fetching bars for {Cid} {From:u} → {To:u}…", contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Tune:S6] Bars fetched: {N}", bars.Count);
        if (bars.Count < 200) { log.LogWarning("[Tune:S6] Not enough bars: {N}", bars.Count); return; }

        var risk = new RiskEngine();
        try { var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE"); if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v; } catch { }

        var grids = new List<(decimal MinAtr, decimal StopMult, decimal TargetMult)>();
        decimal[] minAtr = ParseDecArray(Environment.GetEnvironmentVariable("S6_GRID_MIN_ATR"), [0.6m, 0.8m, 1.0m]);
        decimal[] stopM = ParseDecArray(Environment.GetEnvironmentVariable("S6_GRID_STOP_MULT"), [2.0m, 2.5m, 3.0m]);
        decimal[] targM = ParseDecArray(Environment.GetEnvironmentVariable("S6_GRID_TGT_MULT"), [3.5m, 4.0m, 5.0m]);
        foreach (var ma in minAtr) foreach (var sm in stopM) foreach (var tm in targM) grids.Add((ma, sm, tm));

        var results = new List<TrialResult>(grids.Count);
        int trialIndex = 0;
        foreach (var (MinAtr, StopMult, TargetMult) in grids)
        {
            trialIndex++;
            if (trialIndex == 1 || trialIndex % Math.Max(1, grids.Count / 10) == 0)
                log.LogInformation("[Tune:S6] Trial {Idx}/{Total}…", trialIndex, grids.Count);
            var def = new StrategyDef { Id = "S6", Name = "S6-Override", Enabled = true };
            def.Extra["min_atr"] = JsonSerializer.SerializeToElement(MinAtr);
            def.Extra["stop_mult"] = JsonSerializer.SerializeToElement(StopMult);
            def.Extra["target_mult"] = JsonSerializer.SerializeToElement(TargetMult);
            S6RuntimeConfig.ApplyFrom(def);

            var env = new Env { Symbol = symbolRoot };
            var levels = new Levels();
            Trade? open = null!;
            var equity = new List<decimal> { 0m };
            int wins = 0, losses = 0, trades = 0;
            var history = new List<Bar>(Math.Min(5000, bars.Count));
            foreach (var b in bars)
            {
                try { env.atr = Math.Abs(b.High - b.Low); } catch { }
                if (open != null)
                {
                    if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                    { trades++; if (won) wins++; else losses++; equity.Add(equity[^1] + pnlUsd); open = null; }
                }
                history.Add(b);
                var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
                var s6 = signals.Where(s => string.Equals(s.StrategyId, "S6", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                if (s6 != null && open == null) open = new Trade(s6.Side, s6.Entry, s6.Stop, s6.Target, s6.Size);
            }
            var net = equity[^1]; var maxDd = MaxDrawdown(equity); var wr = trades > 0 ? (decimal)wins / trades : 0m; var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
            var cfg = new TrialConfig([new("min_atr", MinAtr), new("stop_mult", StopMult), new("target_mult", TargetMult)]);
            results.Add(new TrialResult(cfg, trades, wins, losses, net, wr, avgR, maxDd));
        }

        var weights = BotCore.Objectives.ProfitObjective.FromEnv();
        var scored = results.Select(r => new { Res = r, Score = BotCore.Objectives.ProfitObjective.Score(new BotCore.Objectives.ProfitObjective.Metrics(r.NetUsd, r.MaxDrawdownUsd, r.Trades, r.WinRate, r.AvgR), weights) })
                            .OrderByDescending(x => x.Score).ToList();
        var best = scored.FirstOrDefault(x => x.Score > decimal.MinusOne)?.Res ?? results.OrderByDescending(r => r.NetUsd).First();
        log.LogInformation("[Tune:S6] Best (objective): {Best}", best.ToString());

        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"S6-best-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(best, new JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);
        }
        catch (Exception ex) { log.LogWarning(ex, "[Tune:S6] Persist tuned output failed"); }

        try
        {
            var promote = (Environment.GetEnvironmentVariable("PROMOTE_TUNER") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (promote)
            {
                int ttlHours = 24;
                var rawH = Environment.GetEnvironmentVariable("PROMOTE_TTL_HOURS"); var rawD = Environment.GetEnvironmentVariable("PROMOTE_TTL_DAYS");
                if (!string.IsNullOrWhiteSpace(rawH) && int.TryParse(rawH, out var th) && th > 0) ttlHours = th; else if (!string.IsNullOrWhiteSpace(rawD) && int.TryParse(rawD, out var td) && td > 0) ttlHours = td * 24;
                var def = new StrategyDef { Id = "S6", Enabled = true };
                foreach (var p in best.Config.Params) p.Apply(def.Extra);
                ParamStore.SaveS6(symbolRoot, def.Extra, TimeSpan.FromHours(ttlHours));
                log.LogInformation("[Tune:S6] PROMOTE_TUNER on — wrote S6 override (TTL={Ttl}h) to ParamStore for {Sym}", ttlHours, symbolRoot);
            }
        }
        catch (Exception ex) { log.LogWarning(ex, "[Tune:S6] Promote-to-ParamStore failed"); }
    }

    // Grid tuning for S11 using S11RuntimeConfig knobs
    public static async Task RunS11Async(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        log.LogInformation("[Tune:S11] Fetching bars for {Cid} {From:u} → {To:u}…", contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Tune:S11] Bars fetched: {N}", bars.Count);
        if (bars.Count < 200) { log.LogWarning("[Tune:S11] Not enough bars: {N}", bars.Count); return; }

        var risk = new RiskEngine();
        try { var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE"); if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v; } catch { }

        var grids = new List<(decimal MinAtr, decimal StopMult, decimal TargetMult)>();
        decimal[] minAtr = ParseDecArray(Environment.GetEnvironmentVariable("S11_GRID_MIN_ATR"), [0.8m, 1.0m, 1.2m]);
        decimal[] stopM = ParseDecArray(Environment.GetEnvironmentVariable("S11_GRID_STOP_MULT"), [3.0m, 3.5m, 4.0m]);
        decimal[] targM = ParseDecArray(Environment.GetEnvironmentVariable("S11_GRID_TGT_MULT"), [5.0m, 6.0m, 7.0m]);
        foreach (var ma in minAtr) foreach (var sm in stopM) foreach (var tm in targM) grids.Add((ma, sm, tm));

        var results = new List<TrialResult>(grids.Count);
        int trialIndex = 0;
        foreach (var (MinAtr, StopMult, TargetMult) in grids)
        {
            trialIndex++;
            if (trialIndex == 1 || trialIndex % Math.Max(1, grids.Count / 10) == 0)
                log.LogInformation("[Tune:S11] Trial {Idx}/{Total}…", trialIndex, grids.Count);
            var def = new StrategyDef { Id = "S11", Name = "S11-Override", Enabled = true };
            def.Extra["min_atr"] = JsonSerializer.SerializeToElement(MinAtr);
            def.Extra["stop_mult"] = JsonSerializer.SerializeToElement(StopMult);
            def.Extra["target_mult"] = JsonSerializer.SerializeToElement(TargetMult);
            S11RuntimeConfig.ApplyFrom(def);

            var env = new Env { Symbol = symbolRoot };
            var levels = new Levels();
            Trade? open = null!;
            var equity = new List<decimal> { 0m };
            int wins = 0, losses = 0, trades = 0;
            var history = new List<Bar>(Math.Min(5000, bars.Count));
            foreach (var b in bars)
            {
                try { env.atr = Math.Abs(b.High - b.Low); } catch { }
                if (open != null)
                {
                    if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                    { trades++; if (won) wins++; else losses++; equity.Add(equity[^1] + pnlUsd); open = null; }
                }
                history.Add(b);
                var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
                var s11 = signals.Where(s => string.Equals(s.StrategyId, "S11", StringComparison.OrdinalIgnoreCase)).OrderByDescending(s => s.ExpR).FirstOrDefault();
                if (s11 != null && open == null) open = new Trade(s11.Side, s11.Entry, s11.Stop, s11.Target, s11.Size);
            }
            var net = equity[^1]; var maxDd = MaxDrawdown(equity); var wr = trades > 0 ? (decimal)wins / trades : 0m; var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
            var cfg = new TrialConfig([new("min_atr", MinAtr), new("stop_mult", StopMult), new("target_mult", TargetMult)]);
            results.Add(new TrialResult(cfg, trades, wins, losses, net, wr, avgR, maxDd));
        }

        var weights = BotCore.Objectives.ProfitObjective.FromEnv();
        var scored = results.Select(r => new { Res = r, Score = BotCore.Objectives.ProfitObjective.Score(new BotCore.Objectives.ProfitObjective.Metrics(r.NetUsd, r.MaxDrawdownUsd, r.Trades, r.WinRate, r.AvgR), weights) })
                            .OrderByDescending(x => x.Score).ToList();
        var best = scored.FirstOrDefault(x => x.Score > decimal.MinusOne)?.Res ?? results.OrderByDescending(r => r.NetUsd).First();
        log.LogInformation("[Tune:S11] Best (objective): {Best}", best.ToString());

        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"S11-best-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(best, new JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);
        }
        catch (Exception ex) { log.LogWarning(ex, "[Tune:S11] Persist tuned output failed"); }

        try
        {
            var promote = (Environment.GetEnvironmentVariable("PROMOTE_TUNER") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (promote)
            {
                int ttlHours = 24;
                var rawH = Environment.GetEnvironmentVariable("PROMOTE_TTL_HOURS"); var rawD = Environment.GetEnvironmentVariable("PROMOTE_TTL_DAYS");
                if (!string.IsNullOrWhiteSpace(rawH) && int.TryParse(rawH, out var th) && th > 0) ttlHours = th; else if (!string.IsNullOrWhiteSpace(rawD) && int.TryParse(rawD, out var td) && td > 0) ttlHours = td * 24;
                var def = new StrategyDef { Id = "S11", Enabled = true };
                foreach (var p in best.Config.Params) p.Apply(def.Extra);
                ParamStore.SaveS11(symbolRoot, def.Extra, TimeSpan.FromHours(ttlHours));
                log.LogInformation("[Tune:S11] PROMOTE_TUNER on — wrote S11 override (TTL={Ttl}h) to ParamStore for {Sym}", ttlHours, symbolRoot);
            }
        }
        catch (Exception ex) { log.LogWarning(ex, "[Tune:S11] Promote-to-ParamStore failed"); }
    }
    // Generic single-run backtest for any strategy id using current defaults; prints trades and PnL
    public static async Task RunStrategySummaryAsync(HttpClient http, Func<Task<string>> getJwt, string contractId, string symbolRoot, string strategyId, DateTime utcStart, DateTime utcEnd, ILogger log, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(strategyId)) return;
        log.LogInformation("[Backtest:{Strat}] Fetching bars for {Cid} {From:u} → {To:u}…", strategyId, contractId, utcStart, utcEnd);
        var bars = await FetchBarsAsync(http, getJwt, contractId, utcStart, utcEnd, ct).ConfigureAwait(false).ConfigureAwait(false);
        log.LogInformation("[Backtest:{Strat}] Bars fetched: {N}", strategyId, bars.Count);
        if (bars.Count < 200)
        {
            log.LogWarning("[Backtest:{Strat}] Not enough bars: {N}", strategyId, bars.Count);
            return;
        }

        var risk = new RiskEngine();
        try
        {
            var rpt = Environment.GetEnvironmentVariable("RISK_PER_TRADE_USD") ?? Environment.GetEnvironmentVariable("RISK_PER_TRADE");
            if (!string.IsNullOrWhiteSpace(rpt) && decimal.TryParse(rpt, out var v) && v > 0) risk.cfg.risk_per_trade = v;
        }
        catch { }

        var env = new Env { Symbol = symbolRoot };
        var levels = new Levels();
        Trade? open = null!;
        var equity = new List<decimal> { 0m };
        int wins = 0, losses = 0, trades = 0;
        var history = new List<Bar>(Math.Min(5000, bars.Count));
        int barCounter = 0;

        foreach (var b in bars)
        {
            try { env.atr = Math.Abs(b.High - b.Low); } catch { }
            if (open != null)
            {
                if (TryExit(ref open, b, symbolRoot, out var pnlUsd, out var won))
                {
                    trades++; if (won) wins++; else losses++;
                    var lastEq = equity[^1]; equity.Add(lastEq + pnlUsd);
                    open = null!;
                }
            }
            history.Add(b);
            var signals = AllStrategies.generate_signals(symbolRoot, env, levels, history, risk, 0, contractId);
            var pick = signals.Where(s => string.Equals(s.StrategyId, strategyId, StringComparison.OrdinalIgnoreCase))
                              .OrderByDescending(s => s.ExpR)
                              .FirstOrDefault();
            if (pick != null && open == null)
            {
                open = new Trade(pick.Side, pick.Entry, pick.Stop, pick.Target, pick.Size);
            }
            barCounter++;
            if (barCounter % 200 == 0)
                log.LogInformation("[Backtest:{Strat}] Progress {Bars} bars → trades={Trades} net=${Net:F2}", strategyId, barCounter, trades, equity[^1]);
        }

        var net = equity[^1];
        var maxDd = MaxDrawdown(equity);
        var wr = trades > 0 ? (decimal)wins / trades : 0m;
        var avgR = ComputeAvgR(equity, risk.cfg.risk_per_trade);
        log.LogInformation("[Backtest:{Strat}] Done symbol={Sym} days={Days} trades={Trades} win%={Wr:P1} net=${Net:F2} avgR={AvgR:F2} maxDD=${DD:F0}", strategyId, symbolRoot, (utcEnd - utcStart).TotalDays, trades, wr, net, avgR, maxDd);

        try
        {
            var outDir = Path.Combine(AppContext.BaseDirectory, "state", "backtest");
            Directory.CreateDirectory(outDir);
            var outPath = Path.Combine(outDir, $"{strategyId}-summary-{symbolRoot}-{DateTime.UtcNow:yyyyMMdd-HHmm}.json");
            var summary = new { strategy = strategyId, symbol = symbolRoot, start = utcStart, end = utcEnd, trades, wins, losses, netUsd = net, winRate = wr, avgR, maxDrawdownUsd = maxDd };
            await File.WriteAllTextAsync(outPath, System.Text.Json.JsonSerializer.Serialize(summary, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }), ct).ConfigureAwait(false);
            log.LogInformation("[Backtest:{Strat}] Wrote summary: {Path}", strategyId, outPath);
        }
        catch { }
    }

    private static string BuildS3ConfigJson(TrialConfig cfg)
    {
        // Base object with only overridden keys; S3 loader supplies defaults for missing keys
        using var ms = new MemoryStream();
        using (var jw = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = true }))
        {
            jw.WriteStartObject();
            jw.WritePropertyName("Strategies");
            jw.WriteStartArray();
            jw.WriteStartObject();

            // Flat keys
            foreach (var p in cfg.Params)
            {
                if (p.Key.StartsWith("rs_filter.", StringComparison.OrdinalIgnoreCase)) continue; // handled below
                if (p.Key.Contains('.')) continue; // skip other nested for now
                if (p.D.HasValue) { jw.WritePropertyName(p.Key); jw.WriteNumberValue(p.D.Value); }
                else if (p.I.HasValue) { jw.WritePropertyName(p.Key); jw.WriteNumberValue(p.I.Value); }
                else if (p.B.HasValue) { jw.WritePropertyName(p.Key); jw.WriteBooleanValue(p.B.Value); }
                else if (p.S is not null) { jw.WritePropertyName(p.Key); jw.WriteStringValue(p.S); }
            }

            // rs_filter nested
            var hasRs = cfg.Params.Any(p => p.Key.StartsWith("rs_filter.", StringComparison.OrdinalIgnoreCase));
            if (hasRs)
            {
                jw.WritePropertyName("rs_filter");
                jw.WriteStartObject();
                // defaults
                bool enabled = true; int window = 60; decimal threshold = 0.10m; bool directional = true;
                foreach (var p in cfg.Params)
                {
                    if (!p.Key.StartsWith("rs_filter.", StringComparison.OrdinalIgnoreCase)) continue;
                    var k = p.Key["rs_filter.".Length..];
                    switch (k)
                    {
                        case "enabled": if (p.B.HasValue) enabled = p.B.Value; break;
                        case "window_bars": if (p.I.HasValue) window = p.I.Value; break;
                        case "threshold": if (p.D.HasValue) threshold = p.D.Value; break;
                        case "directional_only": if (p.B.HasValue) directional = p.B.Value; break;
                    }
                }
                jw.WriteBoolean("enabled", enabled);
                jw.WriteNumber("window_bars", window);
                jw.WriteNumber("threshold", threshold);
                jw.WriteBoolean("directional_only", directional);
                // minimal peers map
                jw.WritePropertyName("peers");
                jw.WriteStartObject();
                jw.WriteString("ES", "NQ");
                jw.WriteString("NQ", "ES");
                jw.WriteEndObject();
                jw.WriteEndObject();
            }

            jw.WriteEndObject(); // strategy object
            jw.WriteEndArray();
            jw.WriteEndObject();
        }
        return System.Text.Encoding.UTF8.GetString(ms.ToArray());
    }

    private static async Task<List<Bar>> FetchBarsAsync(HttpClient http, Func<Task<string>> getJwt, string contractId, DateTime utcStart, DateTime utcEnd, CancellationToken ct)
    {
        try
        {
            http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", await getJwt()).ConfigureAwait(false);
        }
        catch { }
        var payload = new
        {
            contractId,
            live = false,
            startTime = utcStart.ToString("o"),
            endTime = utcEnd.ToString("o"),
            unit = 2,        // Minute
            unitNumber = 1,
            limit = 20000,
            includePartialBar = false
        };
        using var resp = await http.PostAsJsonAsync("/api/History/retrieveBars", payload, ct).ConfigureAwait(false).ConfigureAwait(false);
        resp.EnsureSuccessStatusCode();
        var text = await resp.Content.ReadAsStringAsync(ct).ConfigureAwait(false).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(text);
        var arr = doc.RootElement.GetProperty("bars");
        var list = new List<Bar>(arr.GetArrayLength());
        foreach (var x in arr.EnumerateArray())
        {
            var t = x.GetProperty("t").GetDateTime();
            var o = x.GetProperty("o").GetDecimal();
            var h = x.GetProperty("h").GetDecimal();
            var l = x.GetProperty("l").GetDecimal();
            var c = x.GetProperty("c").GetDecimal();
            var v = x.GetProperty("v").GetInt64();
            list.Add(new Bar
            {
                Start = t,
                Ts = new DateTimeOffset(t).ToUnixTimeMilliseconds(),
                Symbol = contractId,
                Open = o,
                High = h,
                Low = l,
                Close = c,
                Volume = (int)v
            });
        }
        return list;
    }

    private sealed class Trade(string side, decimal entry, decimal stop, decimal target, int size)
    {
        public string Side { get; } = side; public decimal Entry { get; } = entry; public decimal Stop { get; } = stop; public decimal Target { get; } = target; public int Size { get; } = size;
    }

    private static bool TryExit(ref Trade? tr, Bar b, string root, out decimal pnlUsd, out bool won)
    {
        pnlUsd = 0m; won = false; if (tr == null) return false;
        var pv = InstrumentMeta.PointValue(root);
        // Conservative: stop-first within the bar
        if (string.Equals(tr.Side, "BUY", StringComparison.OrdinalIgnoreCase))
        {
            if (b.Low <= tr.Stop)
            { pnlUsd = (tr.Stop - tr.Entry) * pv * tr.Size; won = false; tr = null; return true; }
            if (b.High >= tr.Target)
            { pnlUsd = (tr.Target - tr.Entry) * pv * tr.Size; won = true; tr = null; return true; }
        }
        else
        {
            if (b.High >= tr.Stop)
            { pnlUsd = (tr.Entry - tr.Stop) * pv * tr.Size; won = false; tr = null; return true; }
            if (b.Low <= tr.Target)
            { pnlUsd = (tr.Entry - tr.Target) * pv * tr.Size; won = true; tr = null; return true; }
        }
        return false;
    }

    private static decimal MaxDrawdown(List<decimal> eq)
    {
        decimal peak = 0m, dd = 0m;
        foreach (var x in eq)
        {
            if (x > peak) peak = x;
            var d = peak - x; if (d > dd) dd = d;
        }
        return dd;
    }

    private static decimal ComputeAvgR(List<decimal> eq, decimal rpt)
    {
        if (rpt <= 0m) return 0m;
        if (eq.Count <= 1) return 0m;
        // crude: average per-trade PnL divided by RPT
        var pnl = eq[^1] - eq[0];
        var trades = eq.Count - 1;
        return (pnl / trades) / rpt;
    }
}
