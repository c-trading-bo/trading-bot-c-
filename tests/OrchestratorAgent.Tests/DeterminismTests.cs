using System;
using System.Collections.Generic;
using System.Linq;
using BotCore.Models;
using BotCore.Risk;
using OrchestratorAgent;
using Xunit;

public class DeterminismTests
{
    [Fact]
    public void DeterministicOutputs()
    {
        var risk = new RiskEngine();
        var levels = new Levels();
        var bars = MakeBars("ES");
        long accountId = 12345;
        string contractId = "CONTRACT";

        var a = BotCore.Strategy.AllStrategies.generate_signals("ES", MakeEnv("ES", bars), levels, bars, risk, accountId, contractId);
        var b = BotCore.Strategy.AllStrategies.generate_signals("ES", MakeEnv("ES", bars), levels, bars, risk, accountId, contractId);

        var aH = a.Select(H).ToList();
        var bH = b.Select(H).ToList();
        Assert.Equal(aH, bH);

        static string H(BotCore.Models.Signal s) => $"{s.StrategyId}|{s.Side}|{s.Entry:F2}|{s.Stop:F2}|{s.Target:F2}|{s.Score:F3}|{s.Size}";
    }

    private static List<Bar> MakeBars(string symbol)
    {
        var list = new List<Bar>();
        var start = DateTime.UtcNow.Date.AddHours(14); // arbitrary fixed-ish start today
        decimal px = 5000m;
        var rnd = new Random(42);
        for (int i = 0; i < 120; i++)
        {
            var o = px;
            var h = o + (decimal)rnd.NextDouble() * 2m;
            var l = o - (decimal)rnd.NextDouble() * 2m;
            var c = l + ((decimal)rnd.NextDouble() * (h - l));
            list.Add(new Bar
            {
                Symbol = symbol,
                Start = start.AddMinutes(i),
                Ts = new DateTimeOffset(start.AddMinutes(i)).ToUnixTimeMilliseconds(),
                Open = o,
                High = Math.Max(Math.Max(o,h), Math.Max(l,c)),
                Low = Math.Min(Math.Min(o,h), Math.Min(l,c)),
                Close = c,
                Volume = 1000 + i
            });
            px = c;
        }
        return list;
    }

    private static BotCore.Models.Env MakeEnv(string symbol, IList<Bar> bars)
    {
        return new BotCore.Models.Env
        {
            Symbol = symbol,
            atr = bars.Count > 0 ? Math.Abs(bars[^1].High - bars[^1].Low) : (decimal?)null,
            volz = 1.0m
        };
    }
}
