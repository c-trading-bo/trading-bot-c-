using System;

namespace BotCore.Objectives;

/// <summary>
/// Risk-aware profit objective used for offline tuning/backtests.
/// Keeps the order path pure; no runtime trading decisions depend on this directly.
/// Configure weights via environment variables:
///   OBJ_NET_W, OBJ_DD_W, OBJ_WR_W, OBJ_AVGR_W, OBJ_MIN_TRADES, OBJ_MAX_DD_LIMIT_USD
/// </summary>
public static class ProfitObjective
{
    public sealed record ProfitMetrics(decimal NetUsd, decimal MaxDrawdownUsd, int Trades, decimal WinRate, decimal AvgR);

    public sealed record Weights(
        decimal NetUsdWeight,
        decimal DrawdownWeight,
        decimal WinRateWeight,
        decimal AvgRWeight,
        int MinTrades,
        decimal MaxDrawdownLimitUsd)
    {
        public static Weights Defaults => new(
            NetUsdWeight: 1.0m,
            DrawdownWeight: 1.0m,
            WinRateWeight: 0.5m,
            AvgRWeight: 0.25m,
            MinTrades: 15,
            MaxDrawdownLimitUsd: 10000m
        );
    }

    public static Weights FromEnv()
    {
        var w = Weights.Defaults;
        try
        {
            w = w with
            {
                NetUsdWeight = Parse("OBJ_NET_W", w.NetUsdWeight),
                DrawdownWeight = Parse("OBJ_DD_W", w.DrawdownWeight),
                WinRateWeight = Parse("OBJ_WR_W", w.WinRateWeight),
                AvgRWeight = Parse("OBJ_AVGR_W", w.AvgRWeight),
                MinTrades = (int)Parse("OBJ_MIN_TRADES", w.MinTrades),
                MaxDrawdownLimitUsd = Parse("OBJ_MAX_DD_LIMIT_USD", w.MaxDrawdownLimitUsd)
            };
        }
        catch { }
        return w;
    }

    public static decimal Score(ProfitMetrics m, Weights w)
    {
        if (m.Trades < w.MinTrades) return decimal.MinusOne; // unviable
        if (m.MaxDrawdownUsd > w.MaxDrawdownLimitUsd) return decimal.MinusOne; // violates risk cap
        // Simple linear score: reward profit and quality, penalize drawdown
        // WinRate and AvgR are small decimals; scale modestly so profit still dominates.
        var wrScaled = m.WinRate * 100m; // 0..100
        var avgRScaled = m.AvgR * 100m;  // rough normalization
        var score = (m.NetUsd * w.NetUsdWeight)
                  - (m.MaxDrawdownUsd * w.DrawdownWeight)
                  + (wrScaled * w.WinRateWeight)
                  + (avgRScaled * w.AvgRWeight);
        return score;
    }

    private static decimal Parse(string name, decimal fallback)
        => decimal.TryParse(Environment.GetEnvironmentVariable(name), out var v) ? v : fallback;
}
