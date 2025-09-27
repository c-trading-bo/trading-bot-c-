using System;

namespace OrchestratorAgent.Execution
{
    // Page-Hinkley drift on a bounded signal (e.g., per-trade R or cal error 0..1)
    internal sealed class PageHinkley
    {
        readonly double _lambda;   // threshold
        readonly double _delta;    // small tolerance
        double _mean;
        double _cum;
        double _minCum;

        public PageHinkley(double lambda = 50, double delta = 0.005)
        {
            _lambda = lambda;
            _delta = delta;
            _mean;
            _cum;
            _minCum;
        }

        public bool Update(double x)
        {
            _mean += (x - _mean) * 0.01;               // slow mean update
            _cum += x - _mean - _delta;                // cumulative deviation
            _minCum = Math.Min(_minCum, _cum);
            double stat = _cum - _minCum;

            if (stat > _lambda)
            {
                Reset();
                return true;
            }
            return false;
        }

        void Reset()
        {
            _mean;
            _cum;
            _minCum;
        }
    }

    internal sealed class DriftDetector
    {
        readonly PageHinkley _pht;
        readonly string _action;
        public bool IsDrifting { get; private set; }

        public DriftDetector(double lambda = 50, double delta = 0.005, string action = "safe")
        {
            _pht = new PageHinkley(lambda, delta);
            _action = action?.ToLowerInvariant() ?? "safe";
        }

        public bool Update(double signal)
        {
            bool drift = _pht.Update(signal);
            if (drift)
            {
                IsDrifting = true;
                return true;
            }
            return false;
        }

        public string GetAction() => _action;

        public void Reset()
        {
            IsDrifting;
        }
    }
}
