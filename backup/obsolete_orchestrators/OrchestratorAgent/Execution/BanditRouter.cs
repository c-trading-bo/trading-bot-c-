using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Execution
{
    public sealed class BanditRouter
    {
        private readonly Random _rng = new();
        private DateTime _lastPick = DateTime.MinValue;
        private ((string Strat, string Cfg) StratCfg, DateTime When)? _lastChoice;
        private readonly Dictionary<(string Strat, string Cfg), (double A, double B)> _post = [];
        public TimeSpan Cooldown { get; set; } = TimeSpan.FromMinutes(30);

        public (string Strat, string Cfg, Dictionary<(string Strat, string Cfg), double> Weights)
            Select(string[] strats, Dictionary<string, string[]> cfgMap, Func<string, double> riskW)
        {
            if ((DateTime.UtcNow - _lastPick) < Cooldown && _lastChoice is not null)
                return (_lastChoice.Value.StratCfg.Strat, _lastChoice.Value.StratCfg.Cfg, CurrentWeights(strats, cfgMap, riskW));

            var samples = new List<((string Strat, string Cfg) Key, double Sample)>();
            foreach (var s in strats)
            {
                var w = Math.Max(1e-3, riskW(s));
                foreach (var c in cfgMap[s])
                {
                    var ab = _post.TryGetValue((s, c), out var xy) ? xy : (7, 3);
                    // Thompson sample scaled by risk weight
                    double sample = SampleBeta(ab.A, ab.B) * w;
                    samples.Add(((s, c), sample));
                }
            }
            var pick = samples.OrderByDescending(x => x.Sample).First().Key;
            _lastChoice = (pick, DateTime.UtcNow); _lastPick = DateTime.UtcNow;
            return (pick.Strat, pick.Cfg, CurrentWeights(strats, cfgMap, riskW));
        }

        static Dictionary<(string Strat, string Cfg), double> CurrentWeights(string[] strats, Dictionary<string, string[]> cfgMap, Func<string, double> riskW)
        {
            var w = new Dictionary<(string Strat, string Cfg), double>();
            foreach (var s in strats)
                foreach (var c in cfgMap[s]) w[(s, c)] = 1.0;
            return w; // placeholder for UI exposure if desired
        }

        double SampleBeta(double a, double b)
        {
            double x = SampleGamma(a), y = SampleGamma(b);
            return x / (x + y);
        }
        double SampleGamma(double shape)
        {
            // Marsaglia-Tsang for k>1; k<=1 boost
            double d = shape < 1 ? shape + 1 : shape;
            double c = d - 1.0 / 3.0;
            while (true)
            {
                double x, v, u;
                do { x = Normal01(); v = 1 + x / Math.Sqrt(9 * c); } while (v <= 0);
                v = v * v * v; u = _rng.NextDouble();
                if (u < 1 - 0.0331 * (x * x) * (x * x)) break;
                if (Math.Log(u) < 0.5 * x * x + c * (1 - v + Math.Log(v))) break;
            }
            double y = c * (1 + Normal01() / Math.Sqrt(9 * c));
            double outv = c * Math.Pow(y / c, 3);
            if (shape < 1) outv *= Math.Pow(_rng.NextDouble(), 1.0 / shape);
            return outv;
        }
        double Normal01()
        {
            double u1 = 1.0 - _rng.NextDouble();
            double u2 = 1.0 - _rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
