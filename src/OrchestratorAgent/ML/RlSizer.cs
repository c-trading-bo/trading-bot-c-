using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace OrchestratorAgent.ML
{
    /// <summary>
    /// Default sizer configuration with fallback values
    /// Used when no ISizerConfig is provided to RlSizer
    /// </summary>
    internal class DefaultSizerConfig : ISizerConfig
    {
        public double GetPpoLearningRate() => 3e-4;
        public double GetCqlAlpha() => 0.2;
        public double GetMetaCostWeight(string costType) => 0.2;
        public double GetPositionSizeMultiplierBaseline() => 1.0;
        public double GetMinPositionSizeMultiplier() => 0.1;
        public double GetMaxPositionSizeMultiplier() => 2.5;
        public double GetExplorationRate() => 0.05;
        public double GetWeightFloor() => 0.10;
        public int GetModelRefreshIntervalMinutes() => 120;
    }
    /// <summary>
    /// Feature snapshot for ML model inference. Contains market data features
    /// extracted at a point in time for a specific symbol and strategy.
    /// </summary>
    internal sealed class FeatureSnapshot
    {
        public Dictionary<string, float> Features { get; } = new();

        public string Symbol { get; set; } = string.Empty;
        public string Strategy { get; set; } = string.Empty;
        public string Session { get; set; } = string.Empty;
        public string Regime { get; set; } = string.Empty;
        public DateTime Timestamp { get; set; }

        // Common technical indicators
        public float Price { get; set; }
        public float Atr { get; set; }
        public float Rsi { get; set; }
        public float Ema20 { get; set; }
        public float Ema50 { get; set; }
        public float Volume { get; set; }
        public float Spread { get; set; }
        public float Volatility { get; set; }

        // Market microstructure
        public float BidAskImbalance { get; set; }
        public float OrderBookImbalance { get; set; }
        public float TickDirection { get; set; }

        // Strategy-specific
        public float SignalStrength { get; set; }
        public float PriorWinRate { get; set; }
        public float AvgRMultiple { get; set; }

        // Risk factors
        public float DrawdownRisk { get; set; }
        public float NewsImpact { get; set; }
        public float LiquidityRisk { get; set; }

        // Symbol-specific features for multi-symbol learning
        public float IsES => Symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) ? 1.0f : 0.0f;
        public float IsNQ => Symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ? 1.0f : 0.0f;

        public FeatureSnapshot()
        {
            // Initialize with default feature values
            AddFeature("price", 0f);
            AddFeature("atr", 0f);
            AddFeature("rsi", 50f);
            AddFeature("ema20", 0f);
            AddFeature("ema50", 0f);
            AddFeature("volume", 0f);
            AddFeature("spread", 0f);
            AddFeature("volatility", 0f);
            AddFeature("bid_ask_imbalance", 0f);
            AddFeature("order_book_imbalance", 0f);
            AddFeature("tick_direction", 0f);
            AddFeature("signal_strength", 0f);
            AddFeature("prior_win_rate", 0.5f);
            AddFeature("avg_r_multiple", 0f);
            AddFeature("drawdown_risk", 0f);
            AddFeature("news_impact", 0f);
            AddFeature("liquidity_risk", 0f);
            // Symbol-specific features
            AddFeature("is_es", 0f);
            AddFeature("is_nq", 0f);
        }

        public void AddFeature(string name, float value)
        {
            Features[name] = value;
        }

        public Dictionary<string, float> ToDict()
        {
            var dict = new Dictionary<string, float>(Features);

            // Sync properties to dictionary
            dict["price"] = Price;
            dict["atr"] = Atr;
            dict["rsi"] = Rsi;
            dict["ema20"] = Ema20;
            dict["ema50"] = Ema50;
            dict["volume"] = Volume;
            dict["spread"] = Spread;
            dict["volatility"] = Volatility;
            dict["bid_ask_imbalance"] = BidAskImbalance;
            dict["order_book_imbalance"] = OrderBookImbalance;
            dict["tick_direction"] = TickDirection;
            dict["signal_strength"] = SignalStrength;
            dict["prior_win_rate"] = PriorWinRate;
            dict["avg_r_multiple"] = AvgRMultiple;
            dict["drawdown_risk"] = DrawdownRisk;
            dict["news_impact"] = NewsImpact;
            dict["liquidity_risk"] = LiquidityRisk;

            // Add symbol-specific features
            dict["is_es"] = IsES;
            dict["is_nq"] = IsNQ;

            return dict;
        }

        public void FromDict(Dictionary<string, float> dict)
        {
            Features.Clear();
            foreach (var kv in dict)
            {
                Features[kv.Key] = kv.Value;
            }

            // Sync dictionary to properties
            Price = dict.GetValueOrDefault("price", 0f);
            Atr = dict.GetValueOrDefault("atr", 0f);
            Rsi = dict.GetValueOrDefault("rsi", 50f);
            Ema20 = dict.GetValueOrDefault("ema20", 0f);
            Ema50 = dict.GetValueOrDefault("ema50", 0f);
            Volume = dict.GetValueOrDefault("volume", 0f);
            Spread = dict.GetValueOrDefault("spread", 0f);
            Volatility = dict.GetValueOrDefault("volatility", 0f);
            BidAskImbalance = dict.GetValueOrDefault("bid_ask_imbalance", 0f);
            OrderBookImbalance = dict.GetValueOrDefault("order_book_imbalance", 0f);
            TickDirection = dict.GetValueOrDefault("tick_direction", 0f);
            SignalStrength = dict.GetValueOrDefault("signal_strength", 0f);
            PriorWinRate = dict.GetValueOrDefault("prior_win_rate", 0.5f);
            AvgRMultiple = dict.GetValueOrDefault("avg_r_multiple", 0f);
            DrawdownRisk = dict.GetValueOrDefault("drawdown_risk", 0f);
            NewsImpact = dict.GetValueOrDefault("news_impact", 0f);
            LiquidityRisk = dict.GetValueOrDefault("liquidity_risk", 0f);
        }
    }

    /// <summary>
    /// Risk-aware reinforcement learning position sizer using ONNX models.
    /// Loads trained CVaR-PPO policy and recommends position multipliers.
    /// </summary>
    internal sealed class RlSizer : IDisposable
    {
        private readonly ILogger<RlSizer> _logger;
        private readonly ISizerConfig _sizerConfig;
        private InferenceSession? _session;
        private readonly float[] _actions;
        private readonly bool _sampleAction;
        private string[] _inputNames;
        private readonly Random _rng = new();
        private readonly string _modelPath;
        private DateTime _modelLastWrite;
        private readonly int _maxAgeMinutes;

        public bool IsLoaded => _session != null!;
        public float[] AvailableActions => _actions.ToArray();

        public RlSizer(
            string onnxPath,
            float[] actions,
            ISizerConfig? sizerConfig = null,
            bool sampleAction = false,
            int maxAgeMinutes = -1, // -1 means use config default
            ILogger<RlSizer>? logger = null)
        {
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger<RlSizer>.Instance;
            _sizerConfig = sizerConfig ?? new DefaultSizerConfig();
            _modelPath = onnxPath;
            _actions = actions.ToArray();
            _sampleAction = sampleAction;
            _maxAgeMinutes = maxAgeMinutes > 0 ? maxAgeMinutes : _sizerConfig.GetModelRefreshIntervalMinutes();
            _inputNames = Array.Empty<string>();

            LoadModel();
        }

        private void LoadModel()
        {
            try
            {
                // Dispose existing session
                _session?.Dispose();
                _session = null!;

                if (!File.Exists(_modelPath))
                {
                    _logger.LogWarning("[RlSizer] Model file not found: {Path}", _modelPath);
                    return;
                }

                // Load ONNX model
                var sessionOptions = new SessionOptions();
                sessionOptions.EnableMemoryPattern; // Reduce memory usage
                sessionOptions.EnableCpuMemArena;

                _session = new InferenceSession(_modelPath, sessionOptions);
                _modelLastWrite = File.GetLastWriteTimeUtc(_modelPath);

                // Extract input names
                var inputMeta = _session.InputMetadata;
                _inputNames = inputMeta.Keys.ToArray();

                _logger.LogInformation("[RlSizer] Loaded model from {Path} with {Inputs} inputs",
                    _modelPath, _inputNames.Length);
                _logger.LogDebug("[RlSizer] Input features: {Features}", string.Join(", ", _inputNames));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[RlSizer] Failed to load model from {Path}", _modelPath);
                _session?.Dispose();
                _session = null!;
            }
        }

        public void CheckForModelUpdates()
        {
            if (!File.Exists(_modelPath)) return;

            var lastWrite = File.GetLastWriteTimeUtc(_modelPath);
            var ageMinutes = (DateTime.UtcNow - lastWrite).TotalMinutes;

            if (lastWrite > _modelLastWrite && ageMinutes <= _maxAgeMinutes)
            {
                _logger.LogInformation("[RlSizer] Detected updated model, reloading...");
                LoadModel();
            }
        }

        public double Recommend(FeatureSnapshot snapshot)
        {
            if (_session == null)
            {
                var defaultSize = _sizerConfig.GetPositionSizeMultiplierBaseline();
                _logger.LogWarning("[RlSizer] Model not loaded, returning configured default size {DefaultSize}", defaultSize);
                return defaultSize;
            }

            try
            {
                // Check for model updates
                CheckForModelUpdates();

                var features = snapshot.ToDict();
                var inputs = new List<NamedOnnxValue>();

                // Prepare input tensors
                foreach (var inputName in _inputNames)
                {
                    var value = features.GetValueOrDefault(inputName, 0f);
                    var tensor = new DenseTensor<float>(new[] { value }, new[] { 1, 1 });
                    inputs.Add(NamedOnnxValue.CreateFromTensor(inputName, tensor));
                }

                // Run inference
                using var results = _session.Run(inputs);
using System.Globalization;

                // Extract logits (policy output)
                DenseTensor<float>? logits = null!;
                foreach (var output in results)
                {
                    if (output.Name == "logits")
                    {
                        logits = output.Value as DenseTensor<float>;
                        break;
                    }
                }

                if (logits == null)
                {
                    var defaultSize = _sizerConfig.GetPositionSizeMultiplierBaseline();
                    _logger.LogWarning("[RlSizer] No logits output found, returning configured default size {DefaultSize}", defaultSize);
                    return defaultSize;
                }

                var logitsArray = logits.ToArray();
                int actionIndex = _sampleAction ? SampleAction(logitsArray) : GreedyAction(logitsArray);

                // Clamp action to valid range
                actionIndex = Math.Max(0, Math.Min(actionIndex, _actions.Length - 1));
                double recommendedSize = _actions[actionIndex];

                // Apply safety bounds
                recommendedSize = Math.Max(0.1, Math.Min(recommendedSize, 3.0));

                _logger.LogDebug("[RlSizer] Recommended size: {Size:F2} (action {Index}, logits: [{Logits}])",
                    recommendedSize, actionIndex, string.Join(", ", logitsArray.Select(x => x.ToString("F3", CultureInfo.InvariantCulture))));

                return recommendedSize;
            }
            catch (Exception ex)
            {
                var defaultSize = _sizerConfig.GetPositionSizeMultiplierBaseline();
                _logger.LogError(ex, "[RlSizer] Inference failed, returning configured default size {DefaultSize}", defaultSize);
                return defaultSize;
            }
        }

        private int GreedyAction(float[] logits)
        {
            int maxIndex;
            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > logits[maxIndex])
                    maxIndex = i;
            }
            return maxIndex;
        }

        private int SampleAction(float[] logits)
        {
            // Softmax sampling
            double max = double.NegativeInfinity;
            foreach (var v in logits)
            {
                if (v > max) max = v;
            }

            double sum;
            var probs = new double[logits.Length];
            for (int i; i < logits.Length; i++)
            {
                probs[i] = Math.Exp(logits[i] - max);
                sum += probs[i];
            }

            for (int i; i < probs.Length; i++)
            {
                probs[i] /= sum;
            }

            double u = _rng.NextDouble();
            double acc;
            for (int i; i < probs.Length; i++)
            {
                acc += probs[i];
                if (u <= acc) return i;
            }

            return probs.Length - 1;
        }

        /// <summary>
        /// Predict position size multiplier for a given feature snapshot
        /// </summary>
        public decimal PredictPositionMultiplier(FeatureSnapshot snapshot)
        {
            var recommendation = Recommend(snapshot);
            return (decimal)recommendation;
        }

        public void Dispose()
        {
            _session?.Dispose();
        }
    }
}
