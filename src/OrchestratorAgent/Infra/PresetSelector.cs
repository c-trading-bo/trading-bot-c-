using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using BotCore.Config;
using BotCore.Models;
using Microsoft.Extensions.Logging;
using OrchestratorAgent.Configuration;

namespace OrchestratorAgent.Infra;

public sealed class PresetSelector(ILogger log, Func<string, IReadOnlyList<Bar>> getBars, TimeSpan dwell, bool allowLive)
{
    private readonly ILogger _log = log;
    private readonly Func<string, IReadOnlyList<Bar>> _getBars = getBars;
    private readonly TimeSpan _dwell = dwell <= TimeSpan.Zero ? TimeSpan.FromMinutes(30) : dwell;
    private readonly bool _allowLive = allowLive;
    private readonly Dictionary<string, (string presetId, DateTime appliedUtc)> _last = new(StringComparer.OrdinalIgnoreCase);

    public void EvaluateAndApply(string symbolRoot)
    {
        if (!_allowLive)
        {
            var live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Trim().ToLowerInvariant() is "1" or "true" or "yes";
            if (live) { _log.LogDebug("[Preset] Live mode — skipping"); return; }
        }

        var bars = _getBars(symbolRoot);
        if (bars == null || bars.Count < 80) { _log.LogDebug("[Preset] Not enough bars for {Sym}", symbolRoot); return; }
        var tags = TagsFromBars(bars);
        _log.LogInformation("[Preset] {Sym} regime tags: {Tags}", symbolRoot, string.Join(",", tags));

        // Apply for each strategy
        ApplyForS2(symbolRoot, tags);
        ApplyForS3(symbolRoot, tags);
        ApplyForS6(symbolRoot, tags);
        ApplyForS11(symbolRoot, tags);
    }

    private void ApplyForS2(string root, HashSet<string> tags)
    {
        var id = "S2:" + string.Join("+", tags.OrderBy(x => x));
        if (!CanApply("S2", root, id)) return;
        var extra = new Dictionary<string, JsonElement>();
        if (tags.Contains("high_vol"))
        {
            extra["sigma_enter"] = JsonSerializer.SerializeToElement(2.2m);
            extra["atr_enter"] = JsonSerializer.SerializeToElement(1.2m);
            extra["attempt_cap"] = JsonSerializer.SerializeToElement(new { RTH = 1, overnight = 0 });
        }
        else if (tags.Contains("low_vol"))
        {
            extra["sigma_enter"] = JsonSerializer.SerializeToElement(1.6m);
            extra["atr_enter"] = JsonSerializer.SerializeToElement(0.8m);
            extra["min_volume"] = JsonSerializer.SerializeToElement(0);
        }
        else // mid_vol
        {
            extra["sigma_enter"] = JsonSerializer.SerializeToElement(1.9m);
            extra["atr_enter"] = JsonSerializer.SerializeToElement(1.0m);
        }
        ParamStore.SaveS2(root, extra, TimeSpan.FromHours(2));
        MarkApplied("S2", root, id);
        _log.LogInformation("[Preset] Applied S2 preset {Id} for {Root}", id, root);
    }

    private void ApplyForS3(string root, HashSet<string> tags)
    {
        var id = "S3:" + string.Join("+", tags.OrderBy(x => x));
        if (!CanApply("S3", root, id)) return;
        var cfg = new Dictionary<string, object>();
        if (tags.Contains("trend"))
        {
            cfg["width_rank_enter"] = 0.25m;
            cfg["min_squeeze_bars"] = 5;
            cfg["entry_mode"] = "retest";
            cfg["retest_backoff_ticks"] = 3;
            cfg["rs_filter"] = new { enabled = true, window_bars = 80, threshold = 0.18m, directional_only = true };
        }
        else // range
        {
            cfg["width_rank_enter"] = 0.45m;
            cfg["min_squeeze_bars"] = 3;
            cfg["entry_mode"] = "breakout";
            cfg["rs_filter"] = new { enabled = false, window_bars = 60, threshold = 0.10m, directional_only = false };
        }
        var json = JsonSerializer.Serialize(cfg);
        ParamStore.SaveS3(root, json, TimeSpan.FromHours(2));
        MarkApplied("S3", root, id);
        _log.LogInformation("[Preset] Applied S3 preset {Id} for {Root}", id, root);
    }

    private void ApplyForS6(string root, HashSet<string> tags)
    {
        var id = "S6:" + string.Join("+", tags.OrderBy(x => x));
        if (!CanApply("S6", root, id)) return;
        var extra = new Dictionary<string, JsonElement>();
        if (tags.Contains("trend"))
        {
            extra["min_atr"] = JsonSerializer.SerializeToElement(0.8m);
            extra["stop_mult"] = JsonSerializer.SerializeToElement((decimal)MLParameterProvider.GetPositionSizeMultiplier());
            extra["target_mult"] = JsonSerializer.SerializeToElement(4.5m);
        }
        else
        {
            extra["min_atr"] = JsonSerializer.SerializeToElement(1.0m);
            extra["stop_mult"] = JsonSerializer.SerializeToElement(2.0m);
            extra["target_mult"] = JsonSerializer.SerializeToElement(3.5m);
        }
        ParamStore.SaveS6(root, extra, TimeSpan.FromHours(2));
        MarkApplied("S6", root, id);
        _log.LogInformation("[Preset] Applied S6 preset {Id} for {Root}", id, root);
    }

    private void ApplyForS11(string root, HashSet<string> tags)
    {
        var id = "S11:" + string.Join("+", tags.OrderBy(x => x));
        if (!CanApply("S11", root, id)) return;
        var extra = new Dictionary<string, JsonElement>();
        if (tags.Contains("high_vol"))
        {
            extra["min_atr"] = JsonSerializer.SerializeToElement(1.2m);
            extra["stop_mult"] = JsonSerializer.SerializeToElement(3.5m);
            extra["target_mult"] = JsonSerializer.SerializeToElement(6.0m);
        }
        else
        {
            extra["min_atr"] = JsonSerializer.SerializeToElement(0.9m);
            extra["stop_mult"] = JsonSerializer.SerializeToElement(3.0m);
            extra["target_mult"] = JsonSerializer.SerializeToElement(5.0m);
        }
        ParamStore.SaveS11(root, extra, TimeSpan.FromHours(2));
        MarkApplied("S11", root, id);
        _log.LogInformation("[Preset] Applied S11 preset {Id} for {Root}", id, root);
    }

    private bool CanApply(string strat, string root, string presetId)
    {
        var key = strat + ":" + root;
        if (_last.TryGetValue(key, out var prev))
        {
            var since = DateTime.UtcNow - prev.appliedUtc;
            if (since < _dwell)
            {
                // deny if switching presets too quickly; allow re-apply of same preset to refresh TTL
                if (!string.Equals(prev.presetId, presetId, StringComparison.Ordinal))
                {
                    _log.LogDebug("[Preset] Dwell active for {Key} ({Remain}s)", key, (int)(_dwell - since).TotalSeconds);
                    return false;
                }
            }
        }
        return true;
    }

    private void MarkApplied(string strat, string root, string id)
    {
        _last[strat + ":" + root] = (id, DateTime.UtcNow);
    }

    private static HashSet<string> TagsFromBars(IReadOnlyList<Bar> bars)
    {
        // Using last 120 bars of 1m data
        int n = Math.Min(120, bars.Count);
        var seg = bars.Skip(bars.Count - n).ToList();
        // returns
        var rets = new List<decimal>(n - 1);
        for (int i = 1; i < seg.Count; i++)
        {
            var prev = seg[i - 1].Close;
            var curr = seg[i].Close;
            if (prev > 0) rets.Add((curr - prev) / prev);
        }
        var mean = rets.Count > 0 ? rets.Average() : 0m;
        var std = (decimal)Math.Sqrt((double)(rets.Select(r => (r - mean) * (r - mean)).DefaultIfEmpty(0m).Average()));
        // use std as a proxy for volatility; map to tags
        var tags = new HashSet<string>();
        if (std < 0.0006m) tags.Add("low_vol");
        else if (std > 0.0012m) tags.Add("high_vol");
        else tags.Add("mid_vol");
        // simple trend detection: slope of linear fit by time index
        decimal slope;
        if (seg.Count > 5)
        {
            int m = seg.Count;
            var xs = Enumerable.Range(0, m).Select(i => (decimal)i).ToArray();
            var ys = seg.Select(b => b.Close).ToArray();
            var xMean = xs.Average(); var yMean = ys.Average();
            var num; var den;
            for (int i; i < m; i++) { var dx = xs[i] - xMean; num += dx * (ys[i] - yMean); den += dx * dx; }
            slope = den != 0 ? num / den : 0m;
        }
        if (Math.Abs(slope) > 0.25m) tags.Add("trend"); else tags.Add("range");
        return tags;
    }
}
