using Microsoft.Extensions.Logging;
using BotCore.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace BotCore.ML
{
    /// <summary>
    /// Simple feature container for ML model input
    /// </summary>
    public class SimpleFeatureSnapshot
    {
        public string Symbol { get; set; } = "";
        public string Strategy { get; set; } = "";
        public DateTime Timestamp { get; set; }
        public float Price { get; set; }
        public float Atr { get; set; }
        public float Volume { get; set; }
        public float Rsi { get; set; } = 50f;
        public float Ema20 { get; set; }
        public float Ema50 { get; set; }
        public float SignalStrength { get; set; }
        public float Volatility { get; set; }
        
        public Dictionary<string, float> ToDict()
        {
            return new Dictionary<string, float>
            {
                ["price"] = Price,
                ["atr"] = Atr,
                ["volume"] = Volume,
                ["rsi"] = Rsi,
                ["ema20"] = Ema20,
                ["ema50"] = Ema50,
                ["signal_strength"] = SignalStrength,
                ["volatility"] = Volatility
            };
        }
    }

    /// <summary>
    /// ML model manager that integrates ONNX models with strategy execution.
    /// Provides position sizing, signal filtering, and execution quality predictions.
    /// </summary>
    public sealed class StrategyMlModelManager : IDisposable
    {
        private readonly ILogger _logger;
        private readonly string _modelsPath;
        private bool _disposed;

        // Model file paths
        private readonly string _rlSizerPath;
        private readonly string _metaClassifierPath;
        private readonly string _execQualityPath;

        public bool IsEnabled => Environment.GetEnvironmentVariable("RL_ENABLED") == "1";

        public StrategyMlModelManager(ILogger logger)
        {
            _logger = logger;
            _modelsPath = Path.Combine(AppContext.BaseDirectory, "models", "rl");
            
            _rlSizerPath = Path.Combine(_modelsPath, "latest_rl_sizer.onnx");
            _metaClassifierPath = Path.Combine(_modelsPath, "latest_meta_classifier.onnx");
            _execQualityPath = Path.Combine(_modelsPath, "latest_exec_quality.onnx");

            _logger.LogInformation("[ML-Manager] Initialized - RL enabled: {Enabled}", IsEnabled);
        }

        /// <summary>
        /// Get ML-optimized position size multiplier for a strategy signal
        /// </summary>
        public decimal GetPositionSizeMultiplier(
            string strategyId,
            string symbol,
            decimal price,
            decimal atr,
            decimal score,
            decimal qScore,
            IList<Bar> bars)
        {
            try
            {
                if (!IsEnabled || !File.Exists(_rlSizerPath))
                {
                    return 1.0m; // Default multiplier
                }

                // For now, use rule-based logic until ONNX integration is complete
                // TODO: Implement actual ONNX model inference
                
                // Simple rule-based position sizing based on quality and volatility
                decimal multiplier = 1.0m;
                
                // Quality-based adjustment
                if (qScore > 0.8m) multiplier += 0.25m;      // High quality signals get larger size
                else if (qScore < 0.4m) multiplier -= 0.25m; // Low quality signals get smaller size
                
                // Score-based adjustment
                if (score > 2.0m) multiplier += 0.15m;       // High score signals
                else if (score < 1.0m) multiplier -= 0.15m;  // Low score signals
                
                // ATR-based volatility adjustment
                if (bars.Any())
                {
                    var avgAtr = CalculateAverageAtr(bars, 14);
                    if (avgAtr > atr * 1.5m) multiplier -= 0.2m; // High volatility = smaller size
                    else if (avgAtr < atr * 0.7m) multiplier += 0.1m; // Low volatility = slightly larger
                }
                
                // Clamp to reasonable range for safety
                multiplier = Math.Clamp(multiplier, 0.25m, 2.0m);
                
                _logger.LogDebug("[ML-Manager] Position multiplier for {Strategy}-{Symbol}: {Multiplier:F2} (qScore: {QScore:F2}, score: {Score:F2})",
                    strategyId, symbol, multiplier, qScore, score);
                
                return multiplier;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML-Manager] Error getting position size multiplier for {Strategy}-{Symbol}", 
                    strategyId, symbol);
                return 1.0m; // Fallback to default
            }
        }

        /// <summary>
        /// Check if a signal should be filtered out by ML meta-classifier
        /// </summary>
        public bool ShouldAcceptSignal(
            string strategyId,
            string symbol,
            decimal price,
            decimal score,
            decimal qScore,
            IList<Bar> bars)
        {
            try
            {
                if (!IsEnabled)
                {
                    return true; // Accept all signals when ML disabled
                }

                // For now, use simple rule-based filtering
                // TODO: Implement ONNX meta-classifier when available
                
                // Basic quality gates
                if (qScore < 0.3m) return false; // Very low quality signals
                if (score < 0.5m) return false; // Very low score signals
                
                // Volume validation
                if (bars.Any())
                {
                    var latest = bars.Last();
                    if (latest.Volume < 100) return false; // Very low volume
                }

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML-Manager] Error in signal filtering for {Strategy}-{Symbol}", 
                    strategyId, symbol);
                return true; // Default to accepting signal
            }
        }

        /// <summary>
        /// Get execution quality score for order routing decisions
        /// </summary>
        public decimal GetExecutionQualityScore(
            string symbol,
            Side side,
            decimal price,
            decimal spread,
            decimal volume)
        {
            try
            {
                if (!IsEnabled)
                {
                    return 0.8m; // Default good execution quality
                }

                // For now, use simple rule-based scoring
                // TODO: Implement ONNX execution quality predictor when available
                
                decimal qualityScore = 1.0m;
                
                // Penalize wide spreads
                if (spread > price * 0.001m) // > 0.1%
                {
                    qualityScore -= 0.2m;
                }
                
                // Penalize low volume
                if (volume < 1000)
                {
                    qualityScore -= 0.3m;
                }
                
                return Math.Clamp(qualityScore, 0.1m, 1.0m);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML-Manager] Error calculating execution quality for {Symbol}", symbol);
                return 0.8m; // Default score
            }
        }

        private SimpleFeatureSnapshot CreateFeatureSnapshot(
            string strategyId,
            string symbol,
            decimal price,
            decimal atr,
            decimal score,
            decimal qScore,
            IList<Bar> bars)
        {
            var features = new SimpleFeatureSnapshot
            {
                Symbol = symbol,
                Strategy = strategyId,
                Timestamp = DateTime.UtcNow,
                Price = (float)price,
                Atr = (float)atr,
                SignalStrength = (float)score
            };

            if (bars.Any())
            {
                var latest = bars.Last();
                features.Volume = (float)latest.Volume;
                
                // Calculate technical indicators
                if (bars.Count >= 14)
                {
                    features.Rsi = (float)CalculateRsi(bars, 14);
                }
                
                if (bars.Count >= 20)
                {
                    features.Ema20 = (float)CalculateEma(bars, 20);
                }
                
                if (bars.Count >= 50)
                {
                    features.Ema50 = (float)CalculateEma(bars, 50);
                }
                
                // Calculate volatility
                if (bars.Count >= 10)
                {
                    var returns = bars.Skip(1).Select((b, i) => 
                        Math.Log((double)(b.Close / bars[i].Close))).ToList();
                    features.Volatility = (float)(returns.StandardDeviation() * Math.Sqrt(252));
                }
            }

            return features;
        }

        private static decimal CalculateAverageAtr(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return 1m;
            
            var trs = new List<decimal>();
            for (int i = 1; i < bars.Count; i++)
            {
                var tr = Math.Max(bars[i].High - bars[i].Low,
                    Math.Max(Math.Abs(bars[i].High - bars[i - 1].Close),
                        Math.Abs(bars[i].Low - bars[i - 1].Close)));
                trs.Add(tr);
            }
            
            return trs.TakeLast(period).Average();
        }

        private static decimal CalculateEma(IList<Bar> bars, int period)
        {
            if (bars.Count < period) return bars.Last().Close;
            
            var multiplier = 2m / (period + 1);
            var ema = bars[0].Close;
            
            for (int i = 1; i < bars.Count; i++)
            {
                ema = (bars[i].Close * multiplier) + (ema * (1 - multiplier));
            }
            
            return ema;
        }

        private static decimal CalculateRsi(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return 50m;
            
            var gains = 0m;
            var losses = 0m;
            
            for (int i = bars.Count - period; i < bars.Count; i++)
            {
                var change = bars[i].Close - bars[i - 1].Close;
                if (change > 0) gains += change;
                else losses -= change;
            }
            
            var avgGain = gains / period;
            var avgLoss = losses / period;
            
            if (avgLoss == 0) return 100m;
            
            var rs = avgGain / avgLoss;
            return 100m - (100m / (1 + rs));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _logger.LogInformation("[ML-Manager] Disposed");
        }
    }

    /// <summary>
    /// Extension methods for strategy integration
    /// </summary>
    public static class StrategyMlExtensions
    {
        /// <summary>
        /// Get the ML strategy type for a given strategy ID
        /// </summary>
        public static MultiStrategyRlCollector.StrategyType GetStrategyType(string strategyId)
        {
            return Strategy.StrategyMlIntegration.GetStrategyType(strategyId);
        }
    }
}

/// <summary>
/// Extension methods for statistical calculations
/// </summary>
public static class StatisticsExtensions
{
    public static double StandardDeviation(this IEnumerable<double> values)
    {
        var valueList = values.ToList();
        if (valueList.Count < 2) return 0.0;
        
        var mean = valueList.Average();
        var variance = valueList.Select(v => Math.Pow(v - mean, 2)).Average();
        return Math.Sqrt(variance);
    }
}