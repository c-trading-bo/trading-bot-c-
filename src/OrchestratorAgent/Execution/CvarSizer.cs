using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Execution
{
    // Chooses position multiplier so that tail loss (CVaR) at level q does not exceed targetR.
    internal sealed class CvarSizer
    {
        readonly Queue<double> _r = new();
        readonly int _max = 500;
        readonly double _q;         // e.g., 0.95
        readonly double _targetR;   // e.g., 0.75 R

        public double MinMult { get; set; } = 0.5;
        public double MaxMult { get; set; } = 1.5;
        public double Step { get; set; } = 0.25;

        public CvarSizer(double q = 0.95, double targetR = 0.75)
        {
            _q = q;
            _targetR = targetR;
        }

        public void Observe(double rMultiple)
        {
            _r.Enqueue(Math.Max(-3, Math.Min(3, rMultiple)));
            while (_r.Count > _max)
                _r.Dequeue();
        }

        public double Recommend(double baseMult)
        {
            if (_r.Count < 50)
                return Math.Clamp(baseMult, MinMult, MaxMult);

            var arr = _r.ToArray().OrderBy(x => x).ToArray();
            int idx = (int)Math.Floor((1 - _q) * arr.Length);
            idx = Math.Max(0, Math.Min(idx, arr.Length - 1));
            double var = arr[idx];

            // Expected shortfall (CVaR) ≈ mean of tail below VaR
            var tail = arr.Take(idx + 1);
            double cvar = tail.Any() ? tail.Average() : var;

            // Choose smallest multiplier that keeps |cvar| <= targetR
            double m = baseMult;
            while (m > MinMult && Math.Abs(cvar * m) > _targetR)
                m -= Step;
            while (m < MaxMult && Math.Abs(cvar * m) < _targetR * 0.6)
                m += Step; // scale up cautiously

            return Math.Clamp(m, MinMult, MaxMult);
        }
    }
}
