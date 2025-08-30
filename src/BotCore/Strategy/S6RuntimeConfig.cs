using System.Text.Json;
using BotCore.Config;

namespace BotCore.Strategy;

public static class S6RuntimeConfig
{
    public static decimal MinAtr { get; private set; } = 0.7m;
    public static decimal StopAtrMult { get; private set; } = 2.0m;
    public static decimal TargetAtrMult { get; private set; } = 4.0m;

    public static void ApplyFrom(StrategyDef def)
    {
        if (def.Extra.TryGetValue("min_atr", out var ma) && ma.TryGetDecimal(out var d1)) MinAtr = d1;
        if (def.Extra.TryGetValue("stop_mult", out var sm) && sm.TryGetDecimal(out var d2)) StopAtrMult = d2;
        if (def.Extra.TryGetValue("target_mult", out var tm) && tm.TryGetDecimal(out var d3)) TargetAtrMult = d3;
    }
}
