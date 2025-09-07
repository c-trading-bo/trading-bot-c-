using System;
using System.Collections.Generic;

namespace OrchestratorAgent.Execution
{
    // Maintains Beta(a,b) win-rate priors per (Strat, Cfg, Regime, Session)
    public sealed class BayesianPriors
    {
        readonly Dictionary<(string strat, string cfg, string regime, string session), (double a, double b)> _map =
            new(StringTupleComparer.OrdinalIgnoreCase);

        public (double a, double b) Get(string s, string c, string reg, string ses)
            => _map.TryGetValue((s, c, reg, ses), out var ab) ? ab : (7, 3); // ~70% prior

        public void Observe(string s, string c, string reg, string ses, bool win)
        {
            var k = (s, c, reg, ses);
            var ab = Get(s, c, reg, ses);
            if (win) ab.a += 1;
            else ab.b += 1;
            _map[k] = ab;
        }

        public double Mean(string s, string c, string reg, string ses)
        {
            var (a, b) = Get(s, c, reg, ses);
            return a / (a + b);
        }

        public double Sample(string s, string c, string reg, string ses, Random rng)
        {
            var (a, b) = Get(s, c, reg, ses);
            return SampleBeta(a, b, rng);
        }

        static double SampleBeta(double a, double b, Random rng)
        {
            double g1 = SampleGamma(a, rng), g2 = SampleGamma(b, rng);
            return g1 / (g1 + g2);
        }

        static double SampleGamma(double k, Random r)
        {
            // Marsaglia-Tsang
            double d = k - 1.0 / 3.0, c = 1.0 / Math.Sqrt(9 * d);
            while (true)
            {
                double x, n;
                do
                {
                    x = Normal01(r);
                    n = 1 + c * x;
                } while (n <= 0);

                n = n * n * n;
                double u = r.NextDouble();
                if (u < 1 - 0.0331 * x * x * x * x) return d * n;
                if (Math.Log(u) < 0.5 * x * x + d * (1 - n + Math.Log(n))) return d * n;
            }
        }

        static double Normal01(Random r)
        {
            double u = r.NextDouble(), v = r.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v);
        }
    }

    sealed class StringTupleComparer : IEqualityComparer<(string, string, string, string)>
    {
        public static readonly StringTupleComparer OrdinalIgnoreCase = new();

        public bool Equals((string, string, string, string) x, (string, string, string, string) y)
            => string.Equals(x.Item1, y.Item1, StringComparison.OrdinalIgnoreCase)
            && string.Equals(x.Item2, y.Item2, StringComparison.OrdinalIgnoreCase)
            && string.Equals(x.Item3, y.Item3, StringComparison.OrdinalIgnoreCase)
            && string.Equals(x.Item4, y.Item4, StringComparison.OrdinalIgnoreCase);

        public int GetHashCode((string, string, string, string) obj)
            => HashCode.Combine(
                obj.Item1.ToLowerInvariant(),
                obj.Item2.ToLowerInvariant(),
                obj.Item3.ToLowerInvariant(),
                obj.Item4.ToLowerInvariant());
    }
}
