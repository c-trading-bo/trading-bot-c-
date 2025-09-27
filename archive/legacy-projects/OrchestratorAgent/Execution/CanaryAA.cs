using System;

namespace OrchestratorAgent.Execution
{
    // Sends a fraction of opportunities to a shadow arm; promotes if statistically better.
    internal sealed class CanaryAA
    {
        readonly Random _rng = new();
        readonly double _ratio;  // 0.10
        readonly double _pmax;   // 0.10
        int _nA = 0, _nB = 0, _wA = 0, _wB;

        public CanaryAA(double ratio = 0.10, double pmax = 0.10)
        {
            _ratio = ratio;
            _pmax = pmax;
        }

        public bool ToShadow() => _rng.NextDouble() < _ratio;

        public void Observe(bool isShadow, bool win)
        {
            if (isShadow)
            {
                _nB++;
                if (win) _wB++;
            }
            else
            {
                _nA++;
                if (win) _wA++;
            }
        }

        public bool ShouldPromote()
        {
            if (_nA < 20 || _nB < 20)
                return false; // need samples

            // two-proportion z-test
            double pA = (double)_wA / _nA;
            double pB = (double)_wB / _nB;
            double p = (double)(_wA + _wB) / (_nA + _nB);
            double z = (pB - pA) / Math.Sqrt(p * (1 - p) * (1.0 / _nA + 1.0 / _nB));

            // one-sided p-value
            double pval = 0.5 * (1 - Erf(z / Math.Sqrt(2)));
            return pval <= _pmax; // promote shadow if significantly better
        }

        public void Reset()
        {
            _nA = _nB = _wA = _wB;
        }

        public (double winRateA, double winRateB, int samplesA, int samplesB) GetStats()
        {
            return (
                _nA > 0 ? (double)_wA / _nA : 0,
                _nB > 0 ? (double)_wB / _nB : 0,
                _nA,
                _nB
            );
        }

        static double Erf(double x)
        {
            // Abramowitz-Stegun approximation
            double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
            double tau = t * Math.Exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
            return x >= 0 ? 1 - tau : tau - 1;
        }
    }
}
