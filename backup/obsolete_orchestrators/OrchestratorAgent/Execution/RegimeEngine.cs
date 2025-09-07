using System;
using System.Collections.Generic;
using System.Linq;

namespace OrchestratorAgent.Execution
{
    public enum Regime { Range = 0, Trend = 1, Vol = 2 }

    // 3-state HMM for market regime detection
    public sealed class RegimeEngine
    {
        readonly int _states = 3;
        readonly double[,] _A;      // transition matrix
        readonly double[][] _mu;    // means per state per feature
        readonly double[][] _sigma; // stds per state per feature
        readonly Queue<double[]> _buf = new();
        readonly int _maxBuf = 2000;
        readonly int _minHoldBars;
        int _holdCount = 0;
        Regime _last = Regime.Range;

        public RegimeEngine(int minHoldBars = 10)
        {
            _minHoldBars = Math.Max(1, minHoldBars);
            _A = new double[_states, _states];
            _mu = new double[_states][];
            _sigma = new double[_states][];

            for (int i = 0; i < _states; i++)
            {
                _mu[i] = new double[3];
                _sigma[i] = new double[] { 1, 1, 1 };
            }

            // Start with a sticky prior
            for (int i = 0; i < _states; i++)
                for (int j = 0; j < _states; j++)
                    _A[i, j] = i == j ? 0.90 : 0.05;
        }

        // features per bar: [ret, volZ, dirStrength]
        public Regime UpdateAndInfer(double ret, double volZ, double dirStrength)
        {
            var x = new double[] { ret, volZ, dirStrength };
            _buf.Enqueue(x);
            while (_buf.Count > _maxBuf) _buf.Dequeue();

            // crude online EM step (moment updates)
            var post = Posterior(x);
            for (int k = 0; k < _states; k++)
            {
                for (int d = 0; d < 3; d++)
                {
                    _mu[k][d] = 0.98 * _mu[k][d] + 0.02 * post[k] * x[d];
                    double diff = x[d] - _mu[k][d];
                    _sigma[k][d] = Math.Sqrt(0.98 * _sigma[k][d] * _sigma[k][d] + 0.02 * post[k] * diff * diff);
                }
            }

            // sticky argmax
            var r = ArgMax(post);
            if (r != _last && _holdCount < _minHoldBars)
            {
                _holdCount++;
                r = _last;
            }
            else
            {
                _holdCount = 0;
                _last = r;
            }
            return r;
        }

        double[] Posterior(double[] x)
        {
            double[] p = new double[_states];
            for (int k = 0; k < _states; k++)
            {
                // diag Gaussian likelihood times self-transition weight
                double like = 1.0;
                for (int d = 0; d < 3; d++)
                {
                    double s = Math.Max(1e-3, _sigma[k][d]);
                    double z = (x[d] - _mu[k][d]) / s;
                    like *= Math.Exp(-0.5 * z * z) / s;
                }
                p[k] = like * (0.7); // encourage self-persistence via scaling
            }

            // normalize
            double sum = p.Sum();
            if (sum <= 0)
                return Enumerable.Repeat(1.0 / _states, _states).ToArray();

            for (int k = 0; k < _states; k++)
                p[k] /= sum;
            return p;
        }

        Regime ArgMax(double[] p) => (Regime)Array.IndexOf(p, p.Max());
    }
}
