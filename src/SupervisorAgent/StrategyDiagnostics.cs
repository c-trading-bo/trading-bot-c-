#nullable enable
using System.Text.Json;
using BotCore.Config;

namespace SupervisorAgent
{
    public static class StrategyDiagnostics
    {
        public sealed record Check(string Name, bool Pass, string Detail);
        public sealed record Report(string Strategy, string Symbol, List<Check> Checks, string Verdict);

    public static Report Explain(TradingProfileConfig cfg, StrategyDef def, BotCore.Models.MarketSnapshot snap)
        {
            var checks = new List<Check>();

            // Example gates — mirror your StrategyGates
            checks.Add(new("SessionWindow",
                string.IsNullOrWhiteSpace(def.SessionWindowEt) || TimeWindows.IsNowWithinEt(def.SessionWindowEt!, snap.UtcNow),
                def.SessionWindowEt ?? "any"));

            if (def.Extra.TryGetValue("filters", out var filters))
            {
                if (filters.TryGetProperty("adx_min", out var adxMinEl) && adxMinEl.TryGetInt32(out var adxMin))
                    checks.Add(new("ADX>=min", snap.Adx5m >= adxMin, $"ADX5m={snap.Adx5m} min={adxMin}"));
                if (filters.TryGetProperty("ema9_over_ema21_5m", out var emaReqEl) && emaReqEl.ValueKind == JsonValueKind.True)
                    checks.Add(new("EMA9>EMA21@5m", snap.Ema9Over21_5m, $"Ema9Over21={snap.Ema9Over21_5m}"));
                if (filters.TryGetProperty("spread_ticks_max", out var stmEl) && stmEl.TryGetInt32(out var stm))
                    checks.Add(new("Spread<=max", snap.SpreadTicks <= stm, $"Spread={snap.SpreadTicks} max={stm}"));
            }

            var verdict = checks.All(c => c.Pass) ? "ARMED" : "BLOCKED";
            return new(def.Name, snap.Symbol, checks, verdict);
        }
    }
}
