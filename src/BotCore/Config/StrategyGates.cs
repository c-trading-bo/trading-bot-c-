using System.Collections.Concurrent;

namespace BotCore.Config;

public static class StrategyGates
{
    // per-symbol throttle store (UTC seconds)
    private static readonly ConcurrentDictionary<string, DateTime> _lastEntryUtc = new(StringComparer.OrdinalIgnoreCase);

    // In AlwaysOn mode, enforce hard spread guard; otherwise use RS gate
    public static bool PassesGlobal(TradingProfileConfig cfg, BotCore.Models.MarketSnapshot snap)
    {
        if (cfg is null) throw new ArgumentNullException(nameof(cfg));
        if (snap is null) throw new ArgumentNullException(nameof(snap));
        
        if (cfg.AlwaysOn.Enabled)
        {
            var gf = cfg.GlobalFilters;
            var hardMax = Math.Max(gf.SpreadTicksMaxBo, gf.SpreadTicksMax);
            if (snap.SpreadTicks > hardMax) return false; // must-have microstructure guard
            return true; // soft checks handled by ScoreWeight/SizeScale
        }
        // Legacy: rs + basic global checks still apply when not AlwaysOn
        return PassesRSGate(cfg, snap);
    }

    public static bool PassesRSGate(TradingProfileConfig cfg, BotCore.Models.MarketSnapshot snap)
    {
        if (cfg is null) throw new ArgumentNullException(nameof(cfg));
        if (snap is null) throw new ArgumentNullException(nameof(snap));
        
        var z = snap.Z5mReturnDiff;
        var abs = Math.Abs(z);
        var mid = cfg.RsGate.ThresholdMid;
        var hi = cfg.RsGate.ThresholdHigh;
        // trade only when regime is active but not chaotic
        var ok = abs >= mid && abs < hi;
        if (ok && cfg.RsGate.AlignWithBias)
        {
            var dir = Math.Sign(z);
            var bias = Math.Sign((double)snap.Bias);
            if (dir != 0 && bias != 0 && dir != bias) ok = false;
        }
        return ok;
    }

    // Convert environment into a size multiplier (never zero)
    public static decimal SizeScale(TradingProfileConfig cfg, BotCore.Models.MarketSnapshot snap)
    {
        if (cfg is null) throw new ArgumentNullException(nameof(cfg));
        if (snap is null) throw new ArgumentNullException(nameof(snap));
        
        decimal scale = 1.0m;
        if (cfg.News.BoostOnMajorNews && (snap.IsMajorNewsNow || snap.IsHoliday))
            scale *= cfg.News.SizeBoostOnNews;

        var gf = cfg.GlobalFilters;
        // Soft size penalties: wideness of spread and weak volume
        if (snap.SpreadTicks > gf.SpreadTicksMaxBo)
            scale *= 0.70m; // very wide → small size
        else if (snap.SpreadTicks > gf.SpreadTicksMax)
            scale *= 0.85m; // slightly wide → trim

        if (snap.VolumePct5m > 0 && snap.VolumePct5m < gf.VolumePctMinMr)
            scale *= 0.85m; // thin tape → trim size a bit

        return Math.Clamp(scale, cfg.AlwaysOn.SizeFloor, cfg.AlwaysOn.SizeCap);
    }

    // News/holiday score weight, biased by family (deterministic; no RNG)
    public static decimal ScoreWeight(TradingProfileConfig cfg, BotCore.Models.MarketSnapshot snap, string family)
    {
        if (cfg is null) throw new ArgumentNullException(nameof(cfg));
        if (snap is null) throw new ArgumentNullException(nameof(snap));
        if (family is null) throw new ArgumentNullException(nameof(family));
        
        decimal w = 1.0m;

        // Deterministic "explosive" detection: explicit flags or Z-based fallback
        bool explosive = snap.IsMajorNewsNow || snap.IsHoliday || (snap.IsMajorNewsSoonWithinSec > 0 && snap.IsMajorNewsSoonWithinSec <= 60);
        if (!explosive)
        {
            var zAbs = Math.Abs(snap.Z5mReturnDiff);
            explosive = zAbs >= cfg.RsGate.ThresholdHigh; // fallback: strong short-term regime
        }

        if (cfg.News.BoostOnMajorNews && explosive)
        {
            if (family.Equals("breakout", StringComparison.OrdinalIgnoreCase))
                w *= cfg.News.BreakoutScoreBoost; // > 1
            else if (family.Equals("meanrev", StringComparison.OrdinalIgnoreCase))
                w *= cfg.News.MeanRevScoreBias;   // typically <= 1 (soft bias)
        }

        // Soft penalties from global filters (AlwaysOn-friendly)
        var gf = cfg.GlobalFilters;
        bool breakout = family.Equals("breakout", StringComparison.OrdinalIgnoreCase) || family.Equals("trend", StringComparison.OrdinalIgnoreCase);

        // Spread
        if (snap.SpreadTicks > gf.SpreadTicksMaxBo)
            w *= 0.60m;
        else if (snap.SpreadTicks > gf.SpreadTicksMax)
            w *= 0.80m;

        // Volume
        if (breakout)
        {
            if (snap.VolumePct5m > 0 && snap.VolumePct5m < gf.VolumePctMinBo) w *= 0.75m;
            if (snap.AggIAbs < gf.AggIMinBo) w *= 0.80m;
        }
        else
        {
            if (snap.VolumePct5m > 0 && snap.VolumePct5m < gf.VolumePctMinMr) w *= 0.85m;
            if (snap.AggIAbs > gf.AggIMaxMrAbs && gf.AggIMaxMrAbs > 0) w *= 0.85m;
        }

        // Over-extended signal bar ATR
        if (snap.SignalBarAtrMult > gf.SignalBarMaxAtrMult) w *= 0.70m;

        return w;
    }

    // Legacy blocking filters preserved for non-always-on scenarios
    public static bool PassesGlobalFilters(TradingProfileConfig cfg, StrategyDef s, BotCore.Models.MarketSnapshot snap)
    {
        if (cfg is null) throw new ArgumentNullException(nameof(cfg));
        if (s is null) throw new ArgumentNullException(nameof(s));
        if (snap is null) throw new ArgumentNullException(nameof(snap));
        
        if (cfg.AlwaysOn.Enabled) return true; // never block in AlwaysOn
        var gf = cfg.GlobalFilters;
        var name = s.Name.ToLowerInvariant();
        var breakout = name.Contains("breakout") || name.Contains("trend") || name.Contains("squeeze") || name.Contains("drive");
        var spreadMax = breakout ? gf.SpreadTicksMaxBo : gf.SpreadTicksMax;
        if (snap.SpreadTicks > spreadMax) return false;
        if (snap.SignalBarAtrMult > gf.SignalBarMaxAtrMult) return false;
        var volMin = breakout ? gf.VolumePctMinBo : gf.VolumePctMinMr;
        if (snap.VolumePct5m < volMin) return false;
        if (breakout)
        {
            if (snap.AggIAbs < gf.AggIMinBo) return false;
        }
        else
        {
            if (snap.AggIAbs > gf.AggIMaxMrAbs) return false;
        }
        if (s.Extra.TryGetValue("filters", out var filters))
        {
            if (filters.TryGetProperty("adx_min", out var adxMinEl) &&
                adxMinEl.TryGetInt32(out var adxMin) && snap.Adx5m < adxMin) return false;
            if (filters.TryGetProperty("ema9_over_ema21_5m", out var emaReqEl) &&
                emaReqEl.ValueKind == System.Text.Json.JsonValueKind.True && !snap.Ema9Over21_5m) return false;
            if (filters.TryGetProperty("spread_ticks_max", out var stmEl) &&
                stmEl.TryGetInt32(out var perStratSpread) && snap.SpreadTicks > perStratSpread) return false;
        }
        if (!string.IsNullOrWhiteSpace(s.SessionWindowEt) && !TimeWindows.IsNowWithinEt(s.SessionWindowEt!, snap.UtcNow))
            return false;
        if (!string.IsNullOrWhiteSpace(s.FlatByEt))
        {
            // Allow only before flat-by time in ET (use existing IsNowWithinEt helper)
            var window = $"00:00-{s.FlatByEt}";
            if (!TimeWindows.IsNowWithinEt(window, snap.UtcNow)) return false;
        }

        // Optional news lockout via env switch
        var newsLock = Environment.GetEnvironmentVariable("NEWS_LOCKOUT_ACTIVE");
        if (!string.IsNullOrWhiteSpace(newsLock) && (newsLock.Equals("1", StringComparison.OrdinalIgnoreCase) || newsLock.Equals("true", StringComparison.OrdinalIgnoreCase)))
            return false;

        // Optional throttle per symbol
        if (gf.MinSecondsBetweenEntries > 0 && !string.IsNullOrWhiteSpace(snap.Symbol))
        {
            var now = DateTime.UtcNow;
            if (_lastEntryUtc.TryGetValue(snap.Symbol, out var lastUtc))
            {
                if ((now - lastUtc).TotalSeconds < gf.MinSecondsBetweenEntries)
                    return false;
            }
            _lastEntryUtc[snap.Symbol] = now; // record attempt time
        }

        return true;
    }
}
