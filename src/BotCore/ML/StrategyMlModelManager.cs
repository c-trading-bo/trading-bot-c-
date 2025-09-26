using Microsoft.Extensions.Logging;
using BotCore.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Globalization;

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
    /// Enhanced with memory management capabilities.
    /// </summary>
    public sealed class StrategyMlModelManager : IDisposable
    {
        private readonly ILogger _logger;
        private readonly string _modelsPath;
        private readonly IMLMemoryManager? _memoryManager;
        private readonly OnnxModelLoader? _onnxLoader;
        private bool _disposed;

        // Model file paths
        private readonly string _rlSizerPath;
        private readonly string _metaClassifierPath;
        private readonly string _execQualityPath;

        public static bool IsEnabled => Environment.GetEnvironmentVariable("RL_ENABLED") == "1";

        public StrategyMlModelManager(ILogger logger, IMLMemoryManager? memoryManager = null, OnnxModelLoader? onnxLoader = null)
        {
            _logger = logger;
            _memoryManager = memoryManager;
            _onnxLoader = onnxLoader;
            _modelsPath = Path.Combine(AppContext.BaseDirectory, "models");

            // Use your actual trained models instead of fake "latest_" paths
            _rlSizerPath = Path.Combine(_modelsPath, "rl", "cvar_ppo_agent.onnx");
            _metaClassifierPath = Path.Combine(_modelsPath, "rl_model.onnx"); 
            _execQualityPath = Path.Combine(_modelsPath, "rl", "test_cvar_ppo.onnx");

            _logger.LogInformation("[ML-Manager] Initialized - RL enabled: {Enabled}, Memory management: {MemoryEnabled}", 
                IsEnabled, _memoryManager != null);
        }
        
        /// <summary>
        /// Load real ONNX model using OnnxModelLoader instead of fake data
        /// </summary>
        private async Task<T?> LoadModelDirectAsync<T>(string modelPath) where T : class
        {
            try
            {
                _logger.LogInformation("[ML-Manager] Loading real ONNX model: {ModelPath}", modelPath);
                
                if (!File.Exists(modelPath))
                {
                    _logger.LogWarning("[ML-Manager] Model file not found: {ModelPath}", modelPath);
                    return null;
                }

                // Use real ONNX loader if available, otherwise fall back to direct loading
                if (_onnxLoader != null)
                {
                    var session = await _onnxLoader.LoadModelAsync(modelPath, validateInference: true).ConfigureAwait(false);
                    if (session == null)
                    {
                        _logger.LogError("[ML-Manager] Failed to load ONNX model: {ModelPath}", modelPath);
                        return null;
                    }
                    
                    _logger.LogInformation("[ML-Manager] Successfully loaded real ONNX model: {ModelPath}", modelPath);
                    return session as T;
                }
                else
                {
                    _logger.LogWarning("[ML-Manager] No ONNX loader available, model loading disabled for safety");
                    return null;
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ML-Manager] Error loading real ONNX model: {ModelPath}", modelPath);
                return null;
            }
        }
        
        /// <summary>
        /// Get model version from file metadata or timestamp
        /// </summary>
        private string GetModelVersion(string modelPath)
        {
            try
            {
                if (File.Exists(modelPath))
                {
                    var lastWrite = File.GetLastWriteTime(modelPath);
                    return lastWrite.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture);
                }
            }
            catch
            {
                // Ignore errors
            }
            
            return "unknown";
        }

        /// <summary>
        /// Get memory usage statistics from memory manager
        /// </summary>
        public MLMemoryManager.MemorySnapshot? GetMemorySnapshot()
        {
            return _memoryManager?.GetMemorySnapshot();
        }

        /// <summary>
        /// Get ML-optimized position size multiplier for a strategy signal
        /// </summary>
        public async Task<decimal> GetPositionSizeMultiplierAsync(
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

                // 🚀 USE REAL ONNX MODEL FOR POSITION SIZING
                if (_onnxLoader != null)
                {
                    try
                    {
                        // Load model asynchronously with proper await
                        var session = await _onnxLoader.LoadModelAsync(_rlSizerPath, validateInference: false).ConfigureAwait(false);
                        if (session != null)
                        {
                            // Create simple feature array for the model
                            var features = new float[] { (float)price, (float)atr, (float)score, (float)qScore };
                            var inputTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(features, new int[] { 1, features.Length });
                            
                            var inputs = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
                            {
                                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("features", inputTensor)
                            };

                            // Run inference with real ML model
                            using var results = session.Run(inputs);
                            var output = results.FirstOrDefault()?.AsEnumerable<float>()?.FirstOrDefault() ?? 1.0f;
                            
                            // Convert to decimal and clamp for safety
                            decimal multiplier = Math.Clamp((decimal)output, 0.25m, 2.0m);

                            _logger.LogInformation("[ML-Manager] 🧠 REAL ONNX position sizing: {Strategy}-{Symbol} = {Multiplier:F2} (qScore: {QScore:F2}, score: {Score:F2})",
                                strategyId, symbol, multiplier, qScore, score);

                            return multiplier;
                        }
                    }
                    catch (Exception modelEx)
                    {
                        _logger.LogWarning(modelEx, "[ML-Manager] Failed to load ONNX model, using fallback");
                    }
                }
                
                _logger.LogWarning("[ML-Manager] ONNX model not available, using fallback multiplier");
                return 1.0m;
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
        public async Task<bool> ShouldAcceptSignalAsync(
            string strategyId,
            string symbol,
            decimal price,
            decimal score,
            decimal qScore,
            IList<Bar> bars)
        {
            if (bars is null) throw new ArgumentNullException(nameof(bars));
            
            try
            {
                if (!IsEnabled)
                {
                    return true; // Accept all signals when ML disabled
                }

                // 🚀 USE REAL ONNX META-CLASSIFIER MODEL FOR SIGNAL FILTERING
                if (_onnxLoader != null && File.Exists(_metaClassifierPath))
                {
                    try
                    {
                        // Load meta-classifier model asynchronously
                        var session = await _onnxLoader.LoadModelAsync(_metaClassifierPath, validateInference: false).ConfigureAwait(false);
                        if (session != null)
                        {
                            // Create feature array for classification
                            var features = new float[] { (float)price, (float)score, (float)qScore, bars.Count > 0 ? (float)bars.Last().Volume : 0f };
                            var inputTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(features, new int[] { 1, features.Length });
                            
                            var inputs = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
                            {
                                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("features", inputTensor)
                            };

                            // Run real ML classification
                            using var results = session.Run(inputs);
                            var probability = results.FirstOrDefault()?.AsEnumerable<float>()?.FirstOrDefault() ?? 0.5f;
                            
                            bool shouldAccept = probability > 0.5f;

                            _logger.LogInformation("[ML-Manager] 🧠 REAL ONNX signal filter: {Strategy}-{Symbol} = {Accept} (prob: {Probability:F3})",
                                strategyId, symbol, shouldAccept ? "ACCEPT" : "REJECT", probability);

                            return shouldAccept;
                        }
                    }
                    catch (Exception modelEx)
                    {
                        _logger.LogWarning(modelEx, "[ML-Manager] Failed to load meta-classifier, using basic rules");
                    }
                }

                // Fallback to basic quality gates
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
        public async Task<decimal> GetExecutionQualityScoreAsync(
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

                // 🚀 USE REAL ONNX EXECUTION QUALITY PREDICTOR
                if (_onnxLoader != null && File.Exists(_execQualityPath))
                {
                    try
                    {
                        // Load execution quality model asynchronously
                        var session = await _onnxLoader.LoadModelAsync(_execQualityPath, validateInference: false).ConfigureAwait(false);
                        if (session != null)
                        {
                            // Create feature array for quality prediction
                            var features = new float[] { (float)price, (float)spread, (float)volume, (float)(spread/price) };
                            var inputTensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(features, new int[] { 1, features.Length });
                            
                            var inputs = new List<Microsoft.ML.OnnxRuntime.NamedOnnxValue>
                            {
                                Microsoft.ML.OnnxRuntime.NamedOnnxValue.CreateFromTensor("features", inputTensor)
                            };

                            // Run real ML quality prediction
                            using var results = session.Run(inputs);
                            var mlQualityScore = results.FirstOrDefault()?.AsEnumerable<float>()?.FirstOrDefault() ?? 0.8f;
                            
                            decimal finalScore = Math.Clamp((decimal)mlQualityScore, 0.1m, 1.0m);

                            _logger.LogInformation("[ML-Manager] 🧠 REAL ONNX execution quality: {Price} = {Quality:F3} (spread: {Spread}, volume: {Volume})",
                                price, finalScore, spread, volume);

                            return finalScore;
                        }
                    }
                    catch (Exception modelEx)
                    {
                        _logger.LogWarning(modelEx, "[ML-Manager] Failed to load execution quality model, using fallback");
                    }
                }

                // Fallback to rule-based scoring
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

            _memoryManager?.Dispose();
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
}