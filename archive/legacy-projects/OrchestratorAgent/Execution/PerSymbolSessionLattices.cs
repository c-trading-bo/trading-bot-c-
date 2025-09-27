using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using System.Threading;
using TradingBot.Abstractions;
using BotCore.Services;

namespace OrchestratorAgent.Execution
{
    /// <summary>
    /// Per-symbol, per-session configuration lattices.
    /// ES-RTH, ES-ETH, NQ-RTH, NQ-ETH each have different behavior patterns.
    /// Separate Bayesian priors and configs for each combination.
    /// Integrated with neutral band decision policy for dynamic threshold management.
    /// </summary>
    internal class PerSymbolSessionLattices
    {
        private readonly Dictionary<string, SymbolSessionConfig> _configs = new();
        private readonly Dictionary<string, SessionBayesianPriors> _priors = new();
        private readonly string _configPath = "state/setup/symbol-session-configs.json";
        private readonly SafeHoldDecisionPolicy? _neutralBandService;

        public PerSymbolSessionLattices(SafeHoldDecisionPolicy? neutralBandService = null)
        {
            _neutralBandService = neutralBandService;
            InitializeConfigs();
            LoadConfigs();
        }

        /// <summary>
        /// Gets configuration for specific symbol-session combination
        /// </summary>
        public SymbolSessionConfig GetConfig(string symbol, SessionType session)
        {
            var key = GetKey(symbol, session);
            return _configs.GetValueOrDefault(key, _configs["ES_RTH"]); // Default fallback
        }

        /// <summary>
        /// Gets Bayesian priors for specific symbol-session combination
        /// </summary>
        public SessionBayesianPriors GetPriors(string symbol, SessionType session)
        {
            var key = GetKey(symbol, session);
            if (!_priors.ContainsKey(key))
            {
                _priors[key] = new SessionBayesianPriors();
            }
            return _priors[key];
        }

        /// <summary>
        /// Updates configuration for a symbol-session combination
        /// </summary>
        public void UpdateConfig(string symbol, SessionType session, SymbolSessionConfig config)
        {
            var key = GetKey(symbol, session);
            _configs[key] = config;
            SaveConfigs();

            Console.WriteLine($"[LATTICE] Updated {key} config: winRate={config.ExpectedWinRate:F3}, volatility={config.VolatilityFactor:F2}");
        }

        /// <summary>
        /// Updates Bayesian priors with new outcome
        /// </summary>
        public void UpdatePriors(string symbol, SessionType session, bool isWin, double rMultiple)
        {
            var priors = GetPriors(symbol, session);
            priors.Update(isWin, rMultiple);

            var key = GetKey(symbol, session);
            Console.WriteLine($"[LATTICE] Updated {key} priors: alpha={priors.Alpha:F1}, beta={priors.Beta:F1}, winProb={priors.WinProbability:F3}");
        }

        /// <summary>
        /// Gets current session type based on time
        /// </summary>
        public SessionType GetCurrentSession()
        {
            var now = DateTime.Now.TimeOfDay;
            // RTH: 9:30 AM - 4:00 PM ET
            // ETH: Everything else
            return (now >= TimeSpan.FromHours(9.5) && now <= TimeSpan.FromHours(16.0)) ?
                SessionType.RTH : SessionType.ETH;
        }

        /// <summary>
        /// Evaluates trading decision using neutral band policy instead of static thresholds
        /// </summary>
        public async Task<TradingDecision?> EvaluateTradingDecisionAsync(
            string symbol, 
            SessionType session,
            double confidence, 
            string strategyId, 
            CancellationToken cancellationToken = default)
        {
            if (_neutralBandService == null)
            {
                Console.WriteLine($"[LATTICE] Warning: No neutral band service available, falling back to static thresholds");
                return EvaluateTradingDecisionStatic(symbol, session, confidence, strategyId);
            }

            var decision = await _neutralBandService.EvaluateDecisionAsync(confidence, symbol, strategyId, cancellationToken).ConfigureAwait(false);
            
            // Apply symbol-session specific adjustments to the decision
            var config = GetConfig(symbol, session);
            var priors = GetPriors(symbol, session);
            
            // Adjust confidence based on session-specific factors
            var adjustedConfidence = ApplySessionAdjustments(confidence, config, priors);
            decision.Confidence = adjustedConfidence;
            
            // Add lattice-specific metadata
            if (decision.Metadata == null)
                decision.Metadata = new Dictionary<string, object>();
                
            decision.Metadata["symbol"] = symbol;
            decision.Metadata["session"] = session.ToString();
            decision.Metadata["original_confidence"] = confidence;
            decision.Metadata["adjusted_confidence"] = adjustedConfidence;
            decision.Metadata["volatility_factor"] = config.VolatilityFactor;
            decision.Metadata["expected_win_rate"] = config.ExpectedWinRate;
            decision.Metadata["bayesian_win_prob"] = priors.WinProbability;
            decision.Metadata["sample_size"] = priors.SampleSize;
            
            Console.WriteLine($"[LATTICE] {GetKey(symbol, session)} decision: {decision.Action} (confidence: {confidence:F3} -> {adjustedConfidence:F3}, reason: {decision.Reason})");
            
            return decision;
        }

        /// <summary>
        /// Fallback method using static thresholds when neutral band service is not available
        /// </summary>
        private TradingDecision EvaluateTradingDecisionStatic(string symbol, SessionType session, double confidence, string strategyId)
        {
            var config = GetConfig(symbol, session);
            
            // Use session-specific static thresholds as fallback
            var bearishThreshold = 0.45 * config.VolatilityFactor; // Adjust threshold based on volatility
            var bullishThreshold = 0.55 / config.VolatilityFactor; // Adjust threshold inversely
            
            TradingAction action;
            string reason;
            
            if (confidence <= bearishThreshold)
            {
                action = TradingAction.Sell;
                reason = $"Below session bearish threshold ({bearishThreshold:F3})";
            }
            else if (confidence >= bullishThreshold)
            {
                action = TradingAction.Buy;
                reason = $"Above session bullish threshold ({bullishThreshold:F3})";
            }
            else
            {
                action = TradingAction.Hold;
                reason = $"In session neutral zone ({bearishThreshold:F3} - {bullishThreshold:F3})";
            }
            
            return new TradingDecision
            {
                Action = action,
                Confidence = confidence,
                Reason = reason,
                Symbol = symbol,
                StrategyId = strategyId,
                Timestamp = DateTime.UtcNow,
                Metadata = new Dictionary<string, object>
                {
                    ["symbol"] = symbol,
                    ["session"] = session.ToString(),
                    ["method"] = "static_fallback",
                    ["volatility_factor"] = config.VolatilityFactor
                }
            };
        }

        /// <summary>
        /// Apply session-specific adjustments to confidence
        /// </summary>
        private double ApplySessionAdjustments(double confidence, SymbolSessionConfig config, SessionBayesianPriors priors)
        {
            // Apply volatility dampening during high volatility periods
            var volatilityAdjustment = 1.0 / Math.Max(1.0, config.VolatilityFactor);
            
            // Apply Bayesian adjustment based on historical performance
            var bayesianWeight = Math.Min(1.0, priors.SampleSize / 30.0); // Full weight after 30 samples
            var expectedWinRate = config.ExpectedWinRate;
            var actualWinRate = priors.WinProbability;
            var winRateAdjustment = 1.0 + (actualWinRate - expectedWinRate) * bayesianWeight * 0.5;
            
            var adjustedConfidence = confidence * volatilityAdjustment * winRateAdjustment;
            
            // Clamp to reasonable range
            return Math.Max(0.0, Math.Min(1.0, adjustedConfidence));
        }

        /// <summary>
        /// Check if confidence is in neutral band (should hold) using integrated neutral band service
        /// </summary>
        public bool IsInNeutralBand(double confidence, string symbol, SessionType session)
        {
            if (_neutralBandService != null)
            {
                // Use the neutral band service for consistent decision making
                return _neutralBandService.IsInNeutralBand(confidence);
            }
            
            // Fallback to session-specific static thresholds
            var config = GetConfig(symbol, session);
            var bearishThreshold = 0.45 * config.VolatilityFactor;
            var bullishThreshold = 0.55 / config.VolatilityFactor;
            
            return confidence > bearishThreshold && confidence < bullishThreshold;
        }

        /// <summary>
        /// Gets neutral band statistics for a specific symbol-session combination
        /// </summary>
        public async Task<NeutralBandStats?> GetNeutralBandStatsAsync(string symbol, SessionType session)
        {
            if (_neutralBandService == null)
                return null;
                
            var baseStats = _neutralBandService.GetNeutralBandStats();
            var config = GetConfig(symbol, session);
            var priors = GetPriors(symbol, session);
            
            // Create session-specific stats
            return new NeutralBandStats
            {
                BearishThreshold = baseStats.BearishThreshold * config.VolatilityFactor,
                BullishThreshold = baseStats.BullishThreshold / config.VolatilityFactor,
                NeutralBandWidth = (baseStats.BullishThreshold / config.VolatilityFactor) - (baseStats.BearishThreshold * config.VolatilityFactor),
                EnableHysteresis = baseStats.EnableHysteresis,
                HysteresisBuffer = baseStats.HysteresisBuffer
            };
        }

        /// <summary>
        /// Gets optimal position sizing for symbol-session combination
        /// </summary>
        public double GetOptimalSize(string symbol, SessionType session, double baseSize)
        {
            var config = GetConfig(symbol, session);
            var priors = GetPriors(symbol, session);

            // Apply volatility factor
            var volAdjusted = baseSize * config.VolatilityFactor;

            // Apply confidence factor based on sample size
            var confidenceFactor = Math.Min(1.0, priors.SampleSize / 50.0);
            var confAdjusted = volAdjusted * (0.5 + 0.5 * confidenceFactor);

            // Apply win rate factor
            var winRateFactor = Math.Max(0.3, Math.Min(1.5, priors.WinProbability / 0.5));
            var finalSize = confAdjusted * winRateFactor;

            Console.WriteLine($"[LATTICE] {GetKey(symbol, session)} sizing: base={baseSize:F2} -> vol={volAdjusted:F2} -> conf={confAdjusted:F2} -> final={finalSize:F2}");

            return Math.Max(0.1, Math.Min(3.0, finalSize)); // Clamp to reasonable range
        }

        private void InitializeConfigs()
        {
            // ES Regular Trading Hours (9:30 AM - 4:00 PM ET)
            _configs["ES_RTH"] = new SymbolSessionConfig
            {
                Symbol = "ES",
                Session = SessionType.RTH,
                ExpectedWinRate = 0.58, // Historically higher during RTH
                VolatilityFactor = 1.0, // Base volatility
                MaxPositionSize = 2.0,
                MinRMultiple = 1.2,
                PreferredStrategies = new[] { "S2a", "S2b", "S3a" },
                SpreadThreshold = 0.50, // ES typical spread
                VolumeThreshold = 1000
            };

            // ES Extended Trading Hours (4:00 PM - 9:30 AM ET)
            _configs["ES_ETH"] = new SymbolSessionConfig
            {
                Symbol = "ES",
                Session = SessionType.ETH,
                ExpectedWinRate = 0.52, // Lower during ETH due to lower liquidity
                VolatilityFactor = 0.75, // Reduce size due to wider spreads
                MaxPositionSize = 1.5,
                MinRMultiple = 1.5, // Higher R required for ETH
                PreferredStrategies = new[] { "S2a" }, // More conservative strategies
                SpreadThreshold = 0.75, // Wider spreads during ETH
                VolumeThreshold = 500
            };

            // NQ Regular Trading Hours
            _configs["NQ_RTH"] = new SymbolSessionConfig
            {
                Symbol = "NQ",
                Session = SessionType.RTH,
                ExpectedWinRate = 0.55, // Slightly lower than ES due to higher volatility
                VolatilityFactor = 1.2, // Higher volatility = smaller position
                MaxPositionSize = 1.8,
                MinRMultiple = 1.3,
                PreferredStrategies = new[] { "S2a", "S2b", "S3a", "S3b" },
                SpreadThreshold = 1.25, // NQ has wider spreads
                VolumeThreshold = 800
            };

            // NQ Extended Trading Hours
            _configs["NQ_ETH"] = new SymbolSessionConfig
            {
                Symbol = "NQ",
                Session = SessionType.ETH,
                ExpectedWinRate = 0.48, // Lowest due to NQ volatility + ETH liquidity issues
                VolatilityFactor = 0.6, // Significantly reduce size
                MaxPositionSize = 1.0,
                MinRMultiple = 2.0, // Much higher R required
                PreferredStrategies = new[] { "S2a" }, // Only most conservative strategy
                SpreadThreshold = 2.0, // Much wider spreads
                VolumeThreshold = 300
            };
        }

        private void LoadConfigs()
        {
            try
            {
                if (File.Exists(_configPath))
                {
                    var json = File.ReadAllText(_configPath);
                    var loaded = JsonSerializer.Deserialize<Dictionary<string, SymbolSessionConfig>>(json);
                    if (loaded != null)
                    {
                        foreach (var kvp in loaded)
                        {
                            _configs[kvp.Key] = kvp.Value;
                        }
                        Console.WriteLine($"[LATTICE] Loaded {loaded.Count} symbol-session configs from {_configPath}");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LATTICE] Failed to load configs: {ex.Message}");
            }
        }

        private void SaveConfigs()
        {
            try
            {
                Directory.CreateDirectory(Path.GetDirectoryName(_configPath)!);
                var json = JsonSerializer.Serialize(_configs, new JsonSerializerOptions { WriteIndented = true });
                File.WriteAllText(_configPath, json);
                Console.WriteLine($"[LATTICE] Saved {_configs.Count} symbol-session configs to {_configPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[LATTICE] Failed to save configs: {ex.Message}");
            }
        }

        private string GetKey(string symbol, SessionType session) => $"{symbol}_{session}";
    }

    internal class SymbolSessionConfig
    {
        public string Symbol { get; set; } = "";
        public SessionType Session { get; set; }
        public double ExpectedWinRate { get; set; }
        public double VolatilityFactor { get; set; }
        public double MaxPositionSize { get; set; }
        public double MinRMultiple { get; set; }
        public string[] PreferredStrategies { get; set; } = Array.Empty<string>();
        public double SpreadThreshold { get; set; }
        public int VolumeThreshold { get; set; }
    }

    internal class SessionBayesianPriors
    {
        public double Alpha { get; private set; } = 1.0; // Prior wins + actual wins
        public double Beta { get; private set; } = 1.0;  // Prior losses + actual losses
        public int SampleSize => (int)(Alpha + Beta - 2); // Actual samples (excluding priors)
        public double WinProbability => Alpha / (Alpha + Beta);

        private double _totalR = 0.0;
        private int _trades;
        public double AverageR => _trades > 0 ? _totalR / _trades : 0.0;

        public void Update(bool isWin, double rMultiple)
        {
            if (isWin)
                Alpha += 1.0;
            else
                Beta += 1.0;

            _totalR += rMultiple;
            _trades++;
        }

        /// <summary>
        /// Gets confidence interval for win probability
        /// </summary>
        public (double lower, double upper) GetConfidenceInterval(double confidence = 0.95)
        {
            // Using beta distribution properties
            var z = confidence == 0.95 ? 1.96 : 1.645; // 95% or 90%
            var p = WinProbability;
            var n = Alpha + Beta;
            var se = Math.Sqrt(p * (1 - p) / n);

            return (Math.Max(0, p - z * se), Math.Min(1, p + z * se));
        }
    }
}
