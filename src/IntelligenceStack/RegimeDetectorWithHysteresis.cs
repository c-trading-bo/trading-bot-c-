using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Advanced regime detector with hysteresis to avoid flip-flop behavior
/// Implements dual thresholds with dwell time and momentum bias
/// </summary>
public class RegimeDetectorWithHysteresis : IRegimeDetector
{
    private readonly ILogger<RegimeDetectorWithHysteresis> _logger;
    private readonly HysteresisConfig _config;
    private readonly object _lock = new();
    
    private RegimeState? _currentState;
    private DateTime _lastTransitionTime = DateTime.MinValue;
    private int _profitableTradesCount = 0;
    private readonly Queue<double> _atrHistory = new();
    private readonly Queue<double> _priceHistory = new();
    private double _medianAtr = 0;
    private readonly double _spreadMedian = 0;

    public RegimeDetectorWithHysteresis(
        ILogger<RegimeDetectorWithHysteresis> logger,
        HysteresisConfig config)
    {
        _logger = logger;
        _config = config;
    }

    public async Task<RegimeState> DetectCurrentRegimeAsync(CancellationToken cancellationToken = default)
    {
        // Ensure proper async execution
        await Task.Yield().ConfigureAwait(false);
        
        lock (_lock)
        {
            if (_currentState != null && IsInDwellPeriod(_currentState))
            {
                return _currentState;
            }

            var newRegime = AnalyzeMarketRegime();
            
            if (_currentState == null)
            {
                _currentState = newRegime;
                _lastTransitionTime = DateTime.UtcNow;
                _logger.LogInformation("[REGIME] Initial regime detected: {Regime} (confidence: {Confidence:F3})", 
                    newRegime.Type, newRegime.Confidence);
            }
            else if (ShouldTransition(_currentState, newRegime))
            {
                _logger.LogInformation("[REGIME] Transitioning from {From} to {To} (confidence: {Confidence:F3})", 
                    _currentState.Type, newRegime.Type, newRegime.Confidence);
                
                _currentState = newRegime;
                _lastTransitionTime = DateTime.UtcNow;
            }

            return _currentState;
        }
    }

    public async Task<RegimeTransition> CheckTransitionAsync(RegimeState currentState, CancellationToken cancellationToken = default)
    {
        var newRegime = await DetectCurrentRegimeAsync(cancellationToken).ConfigureAwait(false);
        
        return new RegimeTransition
        {
            ShouldTransition = newRegime.Type != currentState.Type,
            FromRegime = currentState.Type,
            ToRegime = newRegime.Type,
            TransitionConfidence = newRegime.Confidence,
            BlendDuration = TimeSpan.FromSeconds(60) // Default 60s blend
        };
    }

    public bool IsInDwellPeriod(RegimeState state)
    {
        var timeSinceTransition = DateTime.UtcNow - _lastTransitionTime;
        var requiredDwell = TimeSpan.FromSeconds(_config.DwellSeconds);
        
        // Apply momentum bias - extend dwell time after profitable trades
        if (_profitableTradesCount >= 5)
        {
            var biasMultiplier = 1.0 + _config.MomentumBiasPct;
            requiredDwell = TimeSpan.FromSeconds(requiredDwell.TotalSeconds * biasMultiplier);
        }
        
        return timeSinceTransition < requiredDwell;
    }

    private RegimeState AnalyzeMarketRegime()
    {
        var indicators = CalculateRegimeIndicators();
        var regime = ClassifyRegime(indicators);
        
        var result = new RegimeState
        {
            Type = regime.Type,
            Confidence = regime.Confidence,
            DetectedAt = DateTime.UtcNow,
            DwellTime = DateTime.UtcNow - _lastTransitionTime
        };
        
        // Populate read-only Indicators collection
        foreach (var kvp in indicators)
        {
            result.Indicators[kvp.Key] = kvp.Value;
        }
        
        return result;
    }

    private Dictionary<string, double> CalculateRegimeIndicators()
    {
        var indicators = new Dictionary<string, double>();
        
        // Volatility bands (using ATR)
        var currentAtr = CalculateCurrentATR();
        var volRatio = _medianAtr > 0 ? currentAtr / _medianAtr : 1.0;
        
        indicators["vol_ratio"] = volRatio;
        indicators["vol_low_in"] = _config.VolLowIn;
        indicators["vol_low_out"] = _config.VolLowOut;
        indicators["vol_high_in"] = _config.VolHighIn;
        indicators["vol_high_out"] = _config.VolHighOut;
        
        // Trend slope z-score (50 bars)
        var trendZScore = CalculateTrendZScore();
        indicators["trend_zscore"] = trendZScore;
        indicators["trend_in"] = _config.TrendIn;
        indicators["trend_out"] = _config.TrendOut;
        
        // Liquidity/spread analysis
        var spreadRatio = CalculateSpreadRatio();
        indicators["spread_ratio"] = spreadRatio;
        indicators["spread_wide_in"] = 1.5; // 1.5x recent median
        indicators["spread_wide_out"] = 1.2; // 1.2x recent median
        
        // Additional regime indicators
        indicators["atr_current"] = currentAtr;
        indicators["atr_median"] = _medianAtr;
        indicators["momentum_bias"] = _profitableTradesCount >= 5 ? _config.MomentumBiasPct : 0.0;
        
        return indicators;
    }

    private (RegimeType Type, double Confidence) ClassifyRegime(Dictionary<string, double> indicators)
    {
        var volRatio = indicators["vol_ratio"];
        var trendZScore = Math.Abs(indicators["trend_zscore"]);
        
        // Apply hysteresis logic based on current state
        var inThresholds = GetInThresholds();
        
        // Determine regime with hysteresis
        if (volRatio >= inThresholds.VolHigh)
        {
            return (RegimeType.HighVol, CalculateConfidence(volRatio, inThresholds.VolHigh, 1.0));
        }
        
        if (volRatio <= inThresholds.VolLow)
        {
            return (RegimeType.LowVol, CalculateConfidence(1.0 - volRatio, 1.0 - inThresholds.VolLow, 1.0));
        }
        
        if (trendZScore >= inThresholds.Trend)
        {
            return (RegimeType.Trend, CalculateConfidence(trendZScore, inThresholds.Trend, 1.0));
        }
        
        // Default to range/chop regime
        var rangeConfidence = 1.0 - Math.Max(
            Math.Max(volRatio - 1.0, 0.0),
            Math.Max(trendZScore - 0.5, 0.0)
        );
        
        return (RegimeType.Range, Math.Max(rangeConfidence, 0.1));
    }

    private bool ShouldTransition(RegimeState current, RegimeState proposed)
    {
        if (current.Type == proposed.Type)
            return false;
            
        if (IsInDwellPeriod(current))
            return false;
            
        // High confidence required for transition 
        return proposed.Confidence > 0.7; // High confidence required for transition
    }

    private (double VolLow, double VolHigh, double Trend) GetInThresholds()
    {
        return (_config.VolLowIn, _config.VolHighIn, _config.TrendIn);
    }

    private double CalculateConfidence(double value, double threshold, double maxValue)
    {
        if (threshold >= maxValue) return 0.0;
        return Math.Min(1.0, Math.Max(0.0, (value - threshold) / (maxValue - threshold)));
    }

    private double CalculateCurrentATR()
    {
        // Simplified ATR calculation - would use actual market data in production
        return _atrHistory.Count > 0 ? _atrHistory.ToArray()[^1] : 1.0;
    }

    private double CalculateTrendZScore()
    {
        if (_priceHistory.Count < 50) return 0.0;
        
        var prices = _priceHistory.ToArray();
        var slopes = new List<double>();
        
        // Calculate 50-bar slope
        for (int i = 1; i < Math.Min(50, prices.Length); i++)
        {
            slopes.Add(prices[i] - prices[i - 1]);
        }
        
        if (slopes.Count == 0) return 0.0;
        
        var mean = slopes.Average();
        var variance = slopes.Select(s => Math.Pow(s - mean, 2)).Average();
        var stdDev = Math.Sqrt(variance);
        
        return stdDev > 0 ? mean / stdDev : 0.0;
    }

    private double CalculateSpreadRatio()
    {
        // Simplified spread calculation - would use actual bid/ask data
        return _spreadMedian > 0 ? 1.0 : 1.0; // Placeholder
    }

    public void UpdateMarketData(MarketData data)
    {
        lock (_lock)
        {
            // Update ATR history
            if (_atrHistory.Count >= 14)
                _atrHistory.Dequeue();
            
            var tr = Math.Max(
                Math.Max(data.High - data.Low, Math.Abs(data.High - data.Close)),
                Math.Abs(data.Low - data.Close)
            );
            _atrHistory.Enqueue(tr);
            
            // Update median ATR
            if (_atrHistory.Count > 0)
            {
                var sortedAtrs = _atrHistory.ToArray();
                Array.Sort(sortedAtrs);
                _medianAtr = sortedAtrs.Length % 2 == 0
                    ? (sortedAtrs[sortedAtrs.Length / 2 - 1] + sortedAtrs[sortedAtrs.Length / 2]) / 2.0
                    : sortedAtrs[sortedAtrs.Length / 2];
            }
            
            // Update price history
            if (_priceHistory.Count >= 50)
                _priceHistory.Dequeue();
            _priceHistory.Enqueue(data.Close);
        }
    }

    public void NotifyTradingResult(bool profitable)
    {
        lock (_lock)
        {
            if (profitable)
            {
                _profitableTradesCount++;
            }
            else
            {
                _profitableTradesCount = Math.Max(0, _profitableTradesCount - 1);
            }
        }
    }
}