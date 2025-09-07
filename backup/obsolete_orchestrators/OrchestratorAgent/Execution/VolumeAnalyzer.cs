using System;

namespace OrchestratorAgent.Execution
{
    /// <summary>
    /// NEW FEATURE: Volume Analysis Component
    /// Added while bot is running - demonstrates live ML component injection
    /// </summary>
    public class VolumeAnalyzer
    {
        private decimal[] _recentVolumes = new decimal[20];
        private int _index = 0;

        public void UpdateVolume(decimal volume)
        {
            _recentVolumes[_index] = volume;
            _index = (_index + 1) % _recentVolumes.Length;
            Console.WriteLine($"[VOLUME] Updated: {volume:F0} contracts");
        }

        public decimal GetVolumeSignal()
        {
            // Calculate volume momentum
            var recent = 0m;
            var older = 0m;

            for (int i = 0; i < 10; i++)
            {
                recent += _recentVolumes[i];
                older += _recentVolumes[i + 10];
            }

            if (older == 0) return 1.0m;

            var volumeRatio = recent / older;
            Console.WriteLine($"[VOLUME] Recent/Older ratio: {volumeRatio:F2}");

            // Return signal strength (1.0 = neutral, >1.0 = bullish volume, <1.0 = bearish)
            return volumeRatio;
        }

        public bool IsVolumeSpike()
        {
            var current = _recentVolumes[(_index - 1 + _recentVolumes.Length) % _recentVolumes.Length];
            var average = 0m;

            for (int i = 1; i < _recentVolumes.Length; i++)
            {
                average += _recentVolumes[(_index - 1 - i + _recentVolumes.Length) % _recentVolumes.Length];
            }
            average /= (_recentVolumes.Length - 1);

            var spike = current > average * 2.0m;
            if (spike)
            {
                Console.WriteLine($"[VOLUME] SPIKE DETECTED! Current: {current:F0}, Average: {average:F0}");
            }

            return spike;
        }
    }
}
