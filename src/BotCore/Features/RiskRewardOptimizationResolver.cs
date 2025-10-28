using Microsoft.Extensions.Logging;
using BotCore.Market;
using BotCore.Risk;
using Zones;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Features;

/// <summary>
/// BACKTEST PARITY: Risk-reward optimization resolver
/// Calculates dynamic stop/target distances based on market conditions
/// Ensures historical backtesting uses realistic risk management
/// </summary>
public sealed class RiskRewardOptimizationResolver : IFeatureResolver
{
    private readonly ILogger<RiskRewardOptimizationResolver> _logger;
    private readonly IFeatureBus _featureBus;
    private readonly ConcurrentDictionary<string, RiskRewardBuffer> _buffers = new();
    private readonly ConcurrentDictionary<string, double> _latestFeatures = new();
    
    private const int AtrPeriod = 14;
    private const int BufferSize = 50;

    private static readonly string[] FeatureKeys = new[]
    {
        "risk.optimal_stop_atr",      // Optimal stop distance in ATR units
        "risk.optimal_target_atr",    // Optimal target distance in ATR units
        "risk.risk_reward_ratio",     // Calculated R:R ratio
        "risk.dynamic_position_size", // Risk-adjusted position size
        "risk.max_risk_amount"        // Maximum risk per trade
    };

    public RiskRewardOptimizationResolver(
        ILogger<RiskRewardOptimizationResolver> logger,
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
                _logger.LogWarning("[RISK-REWARD-RESOLVER] Invalid bar data type for {Symbol}", symbol);
                return Task.CompletedTask;
            }

            var buffer = _buffers.GetOrAdd(symbol, _ => new RiskRewardBuffer(BufferSize));
            buffer.AddBar(bar);

            if (buffer.BarCount < AtrPeriod)
            {
                return Task.CompletedTask;
            }

            var now = DateTime.UtcNow;

            // Calculate ATR for dynamic stop/target sizing
            var atr = CalculateATR(buffer);
            
            // Get volatility regime
            var volatilityRegime = GetVolatilityRegime(buffer);
            
            // Calculate optimal stop and target based on volatility
            var (optimalStopAtr, optimalTargetAtr) = CalculateOptimalStopTarget(volatilityRegime, atr);
            
            // Calculate risk-reward ratio
            var riskRewardRatio = optimalTargetAtr / Math.Max(0.1, optimalStopAtr);
            
            // Calculate dynamic position size based on risk
            var dynamicPositionSize = CalculateDynamicPositionSize(volatilityRegime, atr);
            
            // Calculate max risk amount (percentage of account)
            var maxRiskAmount = CalculateMaxRiskAmount(volatilityRegime);

            // Publish features
            _featureBus.Publish(symbol, now, "risk.optimal_stop_atr", optimalStopAtr);
            _featureBus.Publish(symbol, now, "risk.optimal_target_atr", optimalTargetAtr);
            _featureBus.Publish(symbol, now, "risk.risk_reward_ratio", riskRewardRatio);
            _featureBus.Publish(symbol, now, "risk.dynamic_position_size", dynamicPositionSize);
            _featureBus.Publish(symbol, now, "risk.max_risk_amount", maxRiskAmount);

            // Cache latest features
            var key = symbol;
            _latestFeatures[$"{key}::risk.optimal_stop_atr"] = optimalStopAtr;
            _latestFeatures[$"{key}::risk.optimal_target_atr"] = optimalTargetAtr;
            _latestFeatures[$"{key}::risk.risk_reward_ratio"] = riskRewardRatio;
            _latestFeatures[$"{key}::risk.dynamic_position_size"] = dynamicPositionSize;
            _latestFeatures[$"{key}::risk.max_risk_amount"] = maxRiskAmount;

            _logger.LogTrace("[RISK-REWARD-RESOLVER] {Symbol}: Stop={Stop:F2}ATR, Target={Target:F2}ATR, R:R={RR:F2}",
                symbol, optimalStopAtr, optimalTargetAtr, riskRewardRatio);

            return Task.CompletedTask;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[RISK-REWARD-RESOLVER] Error processing bar for {Symbol}", symbol);
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

    private double CalculateATR(RiskRewardBuffer buffer)
    {
        var bars = buffer.GetRecentBars(AtrPeriod);
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

        return trueRanges.Average();
    }

    private VolatilityRegime GetVolatilityRegime(RiskRewardBuffer buffer)
    {
        var bars = buffer.GetRecentBars(20);
        if (bars.Count < 10) return VolatilityRegime.Normal;

        var recentVol = bars.TakeLast(5).Average(b => (double)(b.High - b.Low));
        var historicalVol = bars.Take(15).Average(b => (double)(b.High - b.Low));

        if (historicalVol == 0) return VolatilityRegime.Normal;

        var ratio = recentVol / historicalVol;

        if (ratio > 1.5) return VolatilityRegime.High;
        if (ratio < 0.7) return VolatilityRegime.Low;
        return VolatilityRegime.Normal;
    }

    private (double stopAtr, double targetAtr) CalculateOptimalStopTarget(VolatilityRegime regime, double atr)
    {
        // Adjust stop/target based on volatility regime
        return regime switch
        {
            VolatilityRegime.Low => (1.0, 2.0),      // Tighter stops, closer targets in low vol
            VolatilityRegime.Normal => (1.5, 3.0),   // Standard 1:2 risk-reward
            VolatilityRegime.High => (2.5, 5.0),     // Wider stops, wider targets in high vol
            _ => (1.5, 3.0)
        };
    }

    private double CalculateDynamicPositionSize(VolatilityRegime regime, double atr)
    {
        // Reduce position size in high volatility
        return regime switch
        {
            VolatilityRegime.Low => 1.2,      // 120% normal size in low vol
            VolatilityRegime.Normal => 1.0,   // 100% normal size
            VolatilityRegime.High => 0.7,     // 70% normal size in high vol
            _ => 1.0
        };
    }

    private double CalculateMaxRiskAmount(VolatilityRegime regime)
    {
        // Maximum risk per trade as percentage of account
        return regime switch
        {
            VolatilityRegime.Low => 0.015,     // 1.5% in low vol
            VolatilityRegime.Normal => 0.010,  // 1.0% in normal vol
            VolatilityRegime.High => 0.005,    // 0.5% in high vol
            _ => 0.010
        };
    }

    private enum VolatilityRegime
    {
        Low,
        Normal,
        High
    }

    private class RiskRewardBuffer
    {
        private readonly List<Bar> _bars;
        private readonly int _maxSize;

        public RiskRewardBuffer(int maxSize)
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
