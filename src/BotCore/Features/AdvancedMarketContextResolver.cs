using Microsoft.Extensions.Logging;
using BotCore.Market;
using Zones;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Features;

/// <summary>
/// Advanced market context resolver for sophisticated market analysis
/// Provides VIX proxy, volume z-score, spread analysis, and momentum divergence detection
/// </summary>
public sealed class AdvancedMarketContextResolver : IFeatureResolver
{
    private readonly ILogger<AdvancedMarketContextResolver> _logger;
    private readonly IFeatureBus _featureBus;
    private readonly ConcurrentDictionary<string, MarketContextBuffer> _buffers = new();
    private readonly ConcurrentDictionary<string, double> _latestFeatures = new();
    
    private const int BufferSize = 100;
    private const int VolumeZScorePeriod = 20;
    private const int VolatilityPeriod = 14;

    private static readonly string[] FeatureKeys = new[]
    {
        "market.vix_proxy",
        "market.volume_zscore",
        "market.spread_quality",
        "market.momentum_divergence",
        "market.volatility_expansion",
        "market.volatility_contraction"
    };

    public AdvancedMarketContextResolver(
        ILogger<AdvancedMarketContextResolver> logger,
        IFeatureBus featureBus)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _featureBus = featureBus ?? throw new ArgumentNullException(nameof(featureBus));
    }

    public Task OnBarAsync(string symbol, object barData, CancellationToken cancellationToken = default)
    {
        try
        {
            if (barData is not Bar bar)
            {
                _logger.LogWarning("[ADVANCED-CONTEXT] Invalid bar data type for {Symbol}", symbol);
                return Task.CompletedTask;
            }

            var buffer = _buffers.GetOrAdd(symbol, _ => new MarketContextBuffer(BufferSize));
            buffer.AddBar(bar);

            if (buffer.BarCount < 20)
            {
                return Task.CompletedTask;
            }

            var now = DateTime.UtcNow;

            var vixProxy = CalculateVixProxy(buffer);
            _featureBus.Publish(symbol, now, "market.vix_proxy", vixProxy);
            _latestFeatures[$"{symbol}::market.vix_proxy"] = vixProxy;

            var volumeZScore = CalculateVolumeZScore(buffer);
            _featureBus.Publish(symbol, now, "market.volume_zscore", volumeZScore);
            _latestFeatures[$"{symbol}::market.volume_zscore"] = volumeZScore;

            var spreadQuality = CalculateSpreadQuality(bar);
            _featureBus.Publish(symbol, now, "market.spread_quality", spreadQuality);
            _latestFeatures[$"{symbol}::market.spread_quality"] = spreadQuality;

            var momentumDivergence = CalculateMomentumDivergence(buffer);
            _featureBus.Publish(symbol, now, "market.momentum_divergence", momentumDivergence);
            _latestFeatures[$"{symbol}::market.momentum_divergence"] = momentumDivergence;

            var (expansion, contraction) = CalculateVolatilityRegime(buffer);
            _featureBus.Publish(symbol, now, "market.volatility_expansion", expansion);
            _featureBus.Publish(symbol, now, "market.volatility_contraction", contraction);
            _latestFeatures[$"{symbol}::market.volatility_expansion"] = expansion;
            _latestFeatures[$"{symbol}::market.volatility_contraction"] = contraction;

            _logger.LogTrace("[ADVANCED-CONTEXT] {Symbol}: VIX={VIX:F2}, VolZ={VolZ:F2}, Spread={Spread:F2}",
                symbol, vixProxy, volumeZScore, spreadQuality);

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ADVANCED-CONTEXT] Error processing bar for {Symbol}", symbol);
            throw;
        }
    }

    public Task<double?> TryGetAsync(string symbol, string featureKey, CancellationToken cancellationToken = default)
    {
        var key = $"{symbol}::{featureKey}";
        if (_latestFeatures.TryGetValue(key, out var value))
        {
            return Task.FromResult<double?>(value);
        }
        return Task.FromResult<double?>(null);
    }

    public string[] GetAvailableFeatureKeys() => FeatureKeys;

    private double CalculateVixProxy(MarketContextBuffer buffer)
    {
        var bars = buffer.GetRecentBars(VolatilityPeriod);
        if (bars.Count < 2) return 0.0;

        var trueRanges = new List<double>();
        for (int i = 1; i < bars.Count; i++)
        {
            var high = (double)bars[i].High;
            var low = (double)bars[i].Low;
            var prevClose = (double)bars[i - 1].Close;
            var tr = Math.Max(high - low, Math.Max(Math.Abs(high - prevClose), Math.Abs(low - prevClose)));
            trueRanges.Add(tr);
        }

        var avgTr = trueRanges.Average();
        var currentClose = (double)bars[^1].Close;
        
        return currentClose > 0 ? (avgTr / currentClose) * 100.0 : 0.0;
    }

    private double CalculateVolumeZScore(MarketContextBuffer buffer)
    {
        var bars = buffer.GetRecentBars(VolumeZScorePeriod);
        if (bars.Count < 2) return 0.0;

        var volumes = bars.Select(b => (double)b.Volume).ToList();
        var currentVolume = volumes[^1];
        var historicalVolumes = volumes.Take(volumes.Count - 1).ToList();

        var mean = historicalVolumes.Average();
        var stdDev = Math.Sqrt(historicalVolumes.Average(v => Math.Pow(v - mean, 2)));

        return stdDev > 0 ? (currentVolume - mean) / stdDev : 0.0;
    }

    private double CalculateSpreadQuality(Bar bar)
    {
        var midPrice = ((double)bar.High + (double)bar.Low) / 2.0;
        if (midPrice == 0) return 0.0;

        var spread = (double)(bar.High - bar.Low);
        var spreadPct = spread / midPrice;

        return Math.Max(0.0, Math.Min(1.0, 1.0 - (spreadPct * 100.0)));
    }

    private double CalculateMomentumDivergence(MarketContextBuffer buffer)
    {
        var bars = buffer.GetRecentBars(10);
        if (bars.Count < 5) return 0.0;

        var priceChange = (double)(bars[^1].Close - bars[0].Close);
        var priceDirection = Math.Sign(priceChange);

        var recentVolume = bars.TakeLast(3).Average(b => (double)b.Volume);
        var earlierVolume = bars.Take(3).Average(b => (double)b.Volume);
        var volumeTrend = recentVolume > earlierVolume ? 1 : -1;

        return priceDirection != 0 && priceDirection != volumeTrend ? 1.0 : 0.0;
    }

    private (double expansion, double contraction) CalculateVolatilityRegime(MarketContextBuffer buffer)
    {
        var bars = buffer.GetRecentBars(20);
        if (bars.Count < 10) return (0.0, 0.0);

        var recentVol = bars.TakeLast(5).Average(b => (double)(b.High - b.Low));
        var historicalVol = bars.Take(15).Average(b => (double)(b.High - b.Low));

        if (historicalVol == 0) return (0.0, 0.0);

        var ratio = recentVol / historicalVol;

        var expansion = ratio > 1.2 ? Math.Min(1.0, (ratio - 1.0)) : 0.0;
        var contraction = ratio < 0.8 ? Math.Min(1.0, (1.0 - ratio)) : 0.0;

        return (expansion, contraction);
    }

    private class MarketContextBuffer
    {
        private readonly List<Bar> _bars;
        private readonly int _maxSize;

        public MarketContextBuffer(int maxSize)
        {
            _maxSize = maxSize;
            _bars = new List<Bar>();
        }

        public void AddBar(Bar bar)
        {
            lock (_bars)
            {
                _bars.Add(bar);
                if (_bars.Count > _maxSize)
                {
                    _bars.RemoveAt(0);
                }
            }
        }

        public List<Bar> GetRecentBars(int count)
        {
            lock (_bars)
            {
                return _bars.TakeLast(Math.Min(count, _bars.Count)).ToList();
            }
        }

        public int BarCount
        {
            get
            {
                lock (_bars)
                {
                    return _bars.Count;
                }
            }
        }
    }
}
