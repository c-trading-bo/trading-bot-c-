using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using BotCore.Config;
using Microsoft.Extensions.Logging;
using OrchestratorAgent.Configuration;

namespace OrchestratorAgent.Infra;

public sealed class CanarySelector(ILogger log, Func<string, HashSet<string>> getTags, PositionTracker pos, TimeSpan dwell, TimeSpan window, decimal epsilon, bool allowLive, int minPlays = 2, decimal minEpsilon = 0.05m, double decayHalfLifeHours = 6d) : IAsyncDisposable
{
    private readonly ILogger _log = log;
    private readonly Func<string, HashSet<string>> _getTags = getTags; // by root
    private readonly PositionTracker _pos = pos;
    private readonly TimeSpan _dwell = dwell <= TimeSpan.Zero ? TimeSpan.FromMinutes(45) : dwell;
    private readonly TimeSpan _window = window <= TimeSpan.Zero ? TimeSpan.FromMinutes(60) : window;
    private readonly decimal _eps0 = Math.Clamp(epsilon, 0m, 1m);
    private readonly decimal _minEps = Math.Clamp(minEpsilon, 0m, 1m);
    private readonly int _minPlays = Math.Max(0, minPlays);
    private readonly double _halfLifeHours = decayHalfLifeHours <= 0 ? 6d : decayHalfLifeHours;
    private readonly DateTime _startUtc = DateTime.UtcNow;
    private readonly bool _allowLive = allowLive;
    private readonly CancellationTokenSource _cts = new();
    private Task? _loop;

    private sealed class ArmState
    {
        public int plays { get; set; }
        public decimal totalReward { get; set; }
        public double mean { get; set; }
        public double m2 { get; set; }
    }
    private sealed record Experiment(string root, string regime, string arm, DateTime appliedUtc, decimal baselineRealized);
    private sealed class State(Dictionary<string, Dictionary<string, CanarySelector.ArmState>> stats, List<CanarySelector.Experiment> running, Dictionary<string, DateTime> lastApplied, Dictionary<string, List<string>> blacklist)
    {
        public Dictionary<string, Dictionary<string, ArmState>> stats { get; set; } = stats; public List<Experiment> running { get; set; } = running; public Dictionary<string, DateTime> lastApplied { get; set; } = lastApplied; public Dictionary<string, List<string>> blacklist { get; set; } = blacklist; public Dictionary<string, object>? meta { get; set; }
    }

    private readonly string _path = Path.Combine(AppContext.BaseDirectory, "state", "params", "canary.json");
    private const string CanaryIdKey = "_canary_id";
    private const string Trend = "trend";
    private const string HighVol = "high_vol";
    private const string LowVol = "low_vol";
    private const string MinAtr = "min_atr";
    private const string StopMult = "stop_mult";
    private const string TargetMult = "target_mult";

    public void Start()
    {
        if (_loop != null) return;
        Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
        _loop = Task.Run(async () =>
        {
            while (!_cts.IsCancellationRequested)
            {
                try { EvaluateOnce(); } catch (Exception ex) { _log.LogWarning(ex, "[Canary] evaluate"); }
                try { await Task.Delay(TimeSpan.FromMinutes(5), _cts.Token).ConfigureAwait(false); }
                catch (OperationCanceledException) { /* normal on shutdown */ }
                catch (Exception)
                {
                    // non-fatal; continue loop
                }
            }
        }, _cts.Token);
    }

    private void EvaluateOnce()
    {
        if (!_allowLive)
        {
            var live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (live) { _log.LogDebug("[Canary] Live mode — skipping"); return; }
        }

        var state = Load();
        var now = DateTime.UtcNow;

        // Close finished experiments and update rewards
        var stillRunning = new List<Experiment>();
        foreach (var exp in state.running)
        {
            if (now - exp.appliedUtc >= _window)
            {
                var realized = RealizedForRoot(exp.root);
                var reward = realized - exp.baselineRealized;
                var key = Key(exp.root, exp.regime);
                if (!state.stats.TryGetValue(key, out var arms))
                {
                    arms = new Dictionary<string, ArmState>(StringComparer.OrdinalIgnoreCase);
                    state.stats[key] = arms;
                }
                var s = arms.GetValueOrDefault(exp.arm) ?? new ArmState { plays = 0, totalReward = 0m, mean = 0d, m2 = 0d };
                // Update running mean/variance (Welford)
                int nNew = s.plays + 1;
                double x = (double)reward;
                double meanOld = s.mean;
                double meanNew = meanOld + (x - meanOld) / nNew;
                double m2New = s.m2 + (x - meanOld) * (x - meanNew);
                arms[exp.arm] = new ArmState { plays = nNew, totalReward = s.totalReward + reward, mean = meanNew, m2 = m2New };
                _log.LogInformation("[Canary] Result {Root}/{Regime}/{Arm}: reward=${Reward:F2}", exp.root, exp.regime, exp.arm, reward);

                // Promote or blacklist after updating stats
                try
                {
                    var nowS = arms[exp.arm];
                    var avg = nowS.plays > 0 ? (nowS.totalReward / nowS.plays) : 0m;
                    int minPlaysPromote = EnvInt("CANARY_PROMOTE_MIN_PLAYS", 3);
                    decimal minUsdPromote = EnvDec("CANARY_PROMOTE_MIN_USD", 100m);
                    double z = (double)EnvDec("CANARY_PROMOTE_Z", 1.64m); // ~90% conf by default
                    double se = (nowS.plays > 1) ? Math.Sqrt(Math.Max(0d, nowS.m2 / (nowS.plays - 1))) / Math.Sqrt(nowS.plays) : double.PositiveInfinity;
                    double lower = nowS.mean - z * se;
                    int ttlH = EnvInt("CANARY_PROMOTE_TTL_HOURS", 6);
                    bool promoteEnable = EnvFlag("CANARY_PROMOTE_ENABLE", true) && !_allowLive; // default true in non-live
                    if (promoteEnable && nowS.plays >= minPlaysPromote && avg >= minUsdPromote && lower >= (double)minUsdPromote)
                    {
                        _log.LogInformation("[Canary] Promote {Root}/{Regime}/{Arm}: avg=${Avg:F2} plays={Plays} ttl={Ttl}h", exp.root, exp.regime, exp.arm, avg, nowS.plays, ttlH);
                        ApplyBundle(exp.root, exp.regime, exp.arm, TimeSpan.FromHours(Math.Max(1, ttlH)));
                        // Hint for nightly retuner
                        try
                        {
                            var hintsDir = Path.Combine(AppContext.BaseDirectory, "state", "tuning"); Directory.CreateDirectory(hintsDir);
                            var line = new { ts = DateTime.UtcNow, evt = "canary_promote", exp.root, exp.regime, exp.arm, nowS.plays, avg, conf_lower = lower, ttl_h = ttlH };
                            File.AppendAllText(Path.Combine(hintsDir, "hints.jsonl"), JsonSerializer.Serialize(line) + Environment.NewLine);
                        }
                        catch { /* best-effort hint log */ }
                    }

                    int minPlaysBl = EnvInt("CANARY_BLACKLIST_MIN_PLAYS", 3);
                    decimal dropBl = EnvDec("CANARY_BLACKLIST_DROP_USD", 150m);
                    if (nowS.plays >= minPlaysBl && avg <= -Math.Abs(dropBl))
                    {
                        if (!state.blacklist.TryGetValue(key, out var bl)) { bl = []; state.blacklist[key] = bl; }
                        if (!bl.Contains(exp.arm, StringComparer.OrdinalIgnoreCase)) { bl.Add(exp.arm); _log.LogInformation("[Canary] Blacklist {Key}/{Arm} avg=${Avg:F2}", key, exp.arm, avg); }
                    }
                }
                catch (Exception)
                {
                    // best-effort: promotion/blacklist is advisory
                }
            }
            else stillRunning.Add(exp);
        }
        try { state.running.Clear(); state.running.AddRange(stillRunning); }
        catch (Exception) { /* ignore; will be overwritten on next save */ }

        // For each root, possibly launch a new experiment if dwell passed
        foreach (var root in new[] { "ES", "NQ" })
        {
            var last = state.lastApplied.GetValueOrDefault(root, DateTime.MinValue);
            if (now - last < _dwell) continue;
            var tags = _getTags(root);
            if (tags.Count == 0) continue;
            var regime = Regime(tags);
            var key = Key(root, regime);
            var arms = ArmsFor(tags);
            // Exclude blacklisted arms for this key; if all excluded, ignore blacklist to avoid deadlock
            if (state.blacklist.TryGetValue(key, out var blList) && blList.Count > 0)
            {
                var filtered = arms.Where(a => !blList.Contains(a, StringComparer.OrdinalIgnoreCase)).ToList();
                if (filtered.Count > 0) arms = filtered;
            }
            if (arms.Count == 0) continue;
            string chosen;
            var rand = new Random();
            // Prioritize unplayed/underplayed arms until min plays reached
            if (!state.stats.TryGetValue(key, out var stLocal) || stLocal.Count == 0)
            {
                chosen = arms[rand.Next(arms.Count)];
            }
            else
            {
                var minSeen = stLocal.Values.Select(v => v.plays).DefaultIfEmpty(0).Min();
                var underplayed = arms.Where(a => (!stLocal.TryGetValue(a, out var s)) || s.plays <= Math.Max(minSeen, _minPlays - 1)).ToList();
                if (underplayed.Count > 0)
                {
                    chosen = underplayed[rand.Next(underplayed.Count)];
                }
                else if ((decimal)rand.NextDouble() < EffectiveEpsilon(now))
                {
                    chosen = arms[rand.Next(arms.Count)]; // explore
                }
                else
                {
                    if (!state.stats.TryGetValue(key, out var st) || st == null)
                    {
                        st = new Dictionary<string, ArmState>(StringComparer.OrdinalIgnoreCase);
                        state.stats[key] = st;
                    }
                    chosen = arms.OrderByDescending(a => Avg(st, a)).First();
                }
            }
            ApplyBundle(root, regime, chosen);
            try { state.lastApplied[root] = now; } catch (Exception) { /* best-effort timestamp */ }
            try { state.running.Add(new Experiment(root, regime, chosen, now, RealizedForRoot(root))); } catch (Exception) { /* best-effort add */ }
            _log.LogInformation("[Canary] Applied {Root}/{Regime}/{Arm}", root, regime, chosen);
        }
        // Ephemeral meta for dashboard
        try
        {
            // Read current thresholds for visibility
            int minPlaysPromote = EnvInt("CANARY_PROMOTE_MIN_PLAYS", 3);
            decimal minUsdPromote = EnvDec("CANARY_PROMOTE_MIN_USD", 100m);
            double z = (double)EnvDec("CANARY_PROMOTE_Z", 1.64m);
            int minPlaysBl = EnvInt("CANARY_BLACKLIST_MIN_PLAYS", 3);
            decimal dropBl = EnvDec("CANARY_BLACKLIST_DROP_USD", 150m);
            state.meta = new Dictionary<string, object>
            {
                ["nowUtc"] = DateTime.UtcNow,
                ["epsilon"] = EffectiveEpsilon(now),
                ["dwellMinutes"] = (int)_dwell.TotalMinutes,
                ["windowMinutes"] = (int)_window.TotalMinutes,
                ["minPlays"] = _minPlays,
                ["minEpsilon"] = _minEps,
                ["halfLifeHours"] = _halfLifeHours,
                ["allowLive"] = _allowLive,
                ["promoteZ"] = z,
                ["promoteMinPlays"] = minPlaysPromote,
                ["promoteMinUsd"] = minUsdPromote,
                ["blacklistMinPlays"] = minPlaysBl,
                ["blacklistDropUsd"] = dropBl
            };
        }
        catch (Exception) { /* best-effort */ }

        Save(state);
    }

    private static string Key(string root, string regime) => $"{root}:{regime}";
    private static decimal Avg(Dictionary<string, ArmState> st, string arm)
    { return st.TryGetValue(arm, out var s) && s.plays > 0 ? s.totalReward / s.plays : 0m; }

    private decimal EffectiveEpsilon(DateTime now)
    {
        var hours = (now - _startUtc).TotalHours;
        var halfLives = hours / _halfLifeHours;
        var factor = Math.Pow(0.5, Math.Max(0d, halfLives));
        var eps = (decimal)(factor) * _eps0;
        if (eps < _minEps) eps = _minEps;
        return eps;
    }

    private static void ApplyBundle(string root, string regime, string arm, TimeSpan? ttl = null)
    {
        var life = ttl ?? TimeSpan.FromHours(2);
        // two families per regime: conservative vs aggressive; encode via arm id
        // S2
        var s2 = new Dictionary<string, JsonElement>();
        if (arm.Contains("aggr", StringComparison.OrdinalIgnoreCase))
        {
            s2["sigma_enter"] = JsonSerializer.SerializeToElement(2.2m);
            s2["atr_enter"] = JsonSerializer.SerializeToElement(1.2m);
        }
        else
        {
            s2["sigma_enter"] = JsonSerializer.SerializeToElement(1.8m);
            s2["atr_enter"] = JsonSerializer.SerializeToElement(1.0m);
        }
        s2[CanaryIdKey] = JsonSerializer.SerializeToElement($"{regime}:{arm}");
        ParamStore.SaveS2(root, s2, life);

        // S3
        var s3Cfg = new Dictionary<string, object>();
        if (arm.Contains(Trend, StringComparison.OrdinalIgnoreCase))
        {
            s3Cfg["width_rank_enter"] = arm.Contains("aggr", StringComparison.OrdinalIgnoreCase) ? 0.20m : 0.25m;
            s3Cfg["entry_mode"] = "retest";
            s3Cfg["min_squeeze_bars"] = 5;
        }
        else
        {
            s3Cfg["width_rank_enter"] = arm.Contains("aggr", StringComparison.OrdinalIgnoreCase) ? 0.35m : 0.45m;
            s3Cfg["entry_mode"] = "breakout";
            s3Cfg["min_squeeze_bars"] = 3;
        }
        s3Cfg[CanaryIdKey] = $"{regime}:{arm}";
        ParamStore.SaveS3(root, JsonSerializer.Serialize(s3Cfg), life);

        // S6
        var s6 = new Dictionary<string, JsonElement>();
        if (arm.Contains(Trend, StringComparison.OrdinalIgnoreCase))
        {
            s6[MinAtr] = JsonSerializer.SerializeToElement(0.8m);
            s6[StopMult] = JsonSerializer.SerializeToElement((decimal)MLParameterProvider.GetPositionSizeMultiplier());
            s6[TargetMult] = JsonSerializer.SerializeToElement(arm.Contains("aggr", StringComparison.OrdinalIgnoreCase) ? 5.0m : 4.0m);
        }
        else
        {
            s6[MinAtr] = JsonSerializer.SerializeToElement(1.0m);
            s6[StopMult] = JsonSerializer.SerializeToElement(2.0m);
            s6[TargetMult] = JsonSerializer.SerializeToElement(3.5m);
        }
        s6[CanaryIdKey] = JsonSerializer.SerializeToElement($"{regime}:{arm}");
        ParamStore.SaveS6(root, s6, life);

        // S11
        var s11 = new Dictionary<string, JsonElement>();
        if (arm.Contains(HighVol, StringComparison.OrdinalIgnoreCase))
        {
            s11[MinAtr] = JsonSerializer.SerializeToElement(1.2m);
            s11[StopMult] = JsonSerializer.SerializeToElement(3.5m);
            s11[TargetMult] = JsonSerializer.SerializeToElement(6.0m);
        }
        else
        {
            s11[MinAtr] = JsonSerializer.SerializeToElement(0.9m);
            s11[StopMult] = JsonSerializer.SerializeToElement(3.0m);
            s11[TargetMult] = JsonSerializer.SerializeToElement(5.0m);
        }
        s11[CanaryIdKey] = JsonSerializer.SerializeToElement($"{regime}:{arm}");
        ParamStore.SaveS11(root, s11, life);
    }

    private static List<string> ArmsFor(HashSet<string> tags)
    {
        var list = new List<string>();
        var regime = Regime(tags);
        if (regime == Trend) { list.Add("trend_cons"); list.Add("trend_aggr"); }
        else { list.Add("range_cons"); list.Add("range_aggr"); }
        if (tags.Contains(HighVol)) list = [.. list.Select(x => HighVol + "_" + x)];
        else if (tags.Contains(LowVol)) list = [.. list.Select(x => LowVol + "_" + x)];
        else list = [.. list.Select(x => "mid_vol_" + x)];
        return list;
    }

    private static string Regime(HashSet<string> tags) => tags.Contains(Trend) ? Trend : "range";

    private State Load()
    {
        try
        {
            if (File.Exists(_path))
            {
                var json = File.ReadAllText(_path);
                var st = JsonSerializer.Deserialize<State>(json);
                if (st != null) return st;
            }
        }
        catch (Exception) { /* best-effort load */ }
        return new State([], [], [], []);
    }

    private void Save(State st)
    {
        try
        {
            File.WriteAllText(_path, JsonSerializer.Serialize(st, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception) { /* best-effort save */ }
    }

    private decimal RealizedForRoot(string root)
    {
        try
        {
            var snap = _pos.Snapshot();
            decimal sum;
            foreach (var kv in snap)
            {
                var r = OrchestratorAgent.SymbolMeta.RootFromName(kv.Key);
                if (string.Equals(r, root, StringComparison.OrdinalIgnoreCase)) sum += kv.Value.RealizedUsd;
            }
            return sum;
        }
        catch { return 0m; }
    }

    public async ValueTask DisposeAsync()
    {
        try { await _cts.CancelAsync().ConfigureAwait(false); } catch { }
        try { if (_loop != null) await _loop.ConfigureAwait(false); } catch { }
        _cts.Dispose();
    }

    private static bool EnvFlag(string key, bool defaultOn = false)
    {
        var v = Environment.GetEnvironmentVariable(key);
        if (string.IsNullOrWhiteSpace(v)) return defaultOn; v = v.Trim().ToLowerInvariant();
        return v is "1" or "true" or "yes";
    }
    private static int EnvInt(string key, int def)
    {
        var v = Environment.GetEnvironmentVariable(key); return int.TryParse(v, out var i) ? i : def;
    }
    private static decimal EnvDec(string key, decimal def)
    {
        var v = Environment.GetEnvironmentVariable(key); return decimal.TryParse(v, out var d) ? d : def;
    }
}
