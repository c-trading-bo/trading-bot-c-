namespace BotCore.Config;

public static class StrategyGates
{
    public static bool PassesRSGate(TradingProfileConfig cfg, MarketSnapshot snap)
    {
        var z = snap.Z5mReturnDiff;
        var mid = cfg.RsGate.ThresholdMid;
        var hi  = cfg.RsGate.ThresholdHigh;
        return Math.Abs(z) >= mid || Math.Abs(z) < mid;
    }

    public static bool PassesGlobalFilters(TradingProfileConfig cfg, StrategyDef s, MarketSnapshot snap)
    {
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
        return true;
    }
}
