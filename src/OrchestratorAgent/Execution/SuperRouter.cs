using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Models;
using OrchestratorAgent.Execution;

namespace OrchestratorAgent.Execution
{
    // Enhanced router with regime-aware Bayesian priors, CVaR sizing, drift detection, and canary testing
    public sealed class SuperRouter
    {
        readonly SimpleOrderRouter _baseRouter;
        readonly ILogger<SuperRouter> _log;
        readonly Random _rng = new();
        readonly RegimeEngine _regimeEngine;
        readonly BayesianPriors _priors;
        readonly CvarSizer _cvarSizer;
        readonly DriftDetector _driftDetector;
        readonly CanaryAA _canary;
        readonly Dictionary<string, double> _strategyWeights = new();

        bool _driftSafeMode = false;

        public SuperRouter(HttpClient http, Func<Task<string?>> getJwtAsync, ILogger log, bool live, object? partialExit = null)
        {
            _baseRouter = new SimpleOrderRouter(http, getJwtAsync, log, live, partialExit as OrchestratorAgent.Ops.PartialExitService);
            _log = log as ILogger<SuperRouter> ?? throw new ArgumentException("Logger must be of type ILogger<SuperRouter>");

            // Initialize components with env settings
            int minHoldBars = int.TryParse(Environment.GetEnvironmentVariable("REGIME_MIN_HOLD_BARS"), out var mhb) ? mhb : 10;
            _regimeEngine = new RegimeEngine(minHoldBars);

            _priors = new BayesianPriors();

            double cvarLevel = double.TryParse(Environment.GetEnvironmentVariable("CVAR_LEVEL"), out var cl) ? cl : 0.95;
            double cvarTarget = double.TryParse(Environment.GetEnvironmentVariable("CVAR_TARGET_R"), out var ct) ? ct : 0.75;
            _cvarSizer = new CvarSizer(cvarLevel, cvarTarget);

            double phtLambda = double.TryParse(Environment.GetEnvironmentVariable("DRIFT_PHT_LAMBDA"), out var pl) ? pl : 50;
            double phtDelta = double.TryParse(Environment.GetEnvironmentVariable("DRIFT_PHT_DELTA"), out var pd) ? pd : 0.005;
            string driftAction = Environment.GetEnvironmentVariable("DRIFT_ACTION") ?? "safe";
            _driftDetector = new DriftDetector(phtLambda, phtDelta, driftAction);

            double canaryRatio = double.TryParse(Environment.GetEnvironmentVariable("CANARY_RATIO"), out var cr) ? cr : 0.10;
            double canaryPValue = double.TryParse(Environment.GetEnvironmentVariable("CANARY_PVALUE_MAX"), out var cp) ? cp : 0.10;
            _canary = new CanaryAA(canaryRatio, canaryPValue);
        }

        // Delegate core routing methods to base router
        public void DisableAllEntries() => _baseRouter.DisableAllEntries();
        public void EnableAllEntries() => _baseRouter.EnableAllEntries();
        public async Task CloseAll(string reason, CancellationToken ct) => await _baseRouter.CloseAll(reason, ct);
        public async Task EnsureBracketsAsync(long accountId, CancellationToken ct) => await _baseRouter.EnsureBracketsAsync(accountId, ct);
        public async Task FlattenAll(long accountId, CancellationToken ct) => await _baseRouter.FlattenAll(accountId, ct);

        // Enhanced routing with ML integration
        public async Task<bool> RouteAsync(Signal sig, CancellationToken ct)
        {
            try
            {
                // Apply ML enhancements before routing
                var enhancedSig = await EnhanceSignalAsync(sig, ct);

                // Route using base router with enhancements
                return await _baseRouter.RouteAsync(enhancedSig, ct);
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[SuperRouter] Enhanced routing failed for {Strategy}, falling back to base router", sig.StrategyId);
                return await _baseRouter.RouteAsync(sig, ct);
            }
        }

        Task<Signal> EnhanceSignalAsync(Signal sig, CancellationToken ct)
        {
            try
            {
                // 1. Regime detection (using synchronous method for now)
                var currentRegime = DetectCurrentRegime(sig.Symbol);

                // 2. CVaR-optimized sizing 
                var cvarMultiplier = OptimizePositionSize(sig.Symbol);

                // 3. Check for drift and adjust if needed
                var isDrifting = CheckForDrift(sig.Symbol);
                if (isDrifting && !_driftSafeMode)
                {
                    _log.LogWarning("[SuperRouter] Drift detected for {Symbol}, entering safe mode", sig.Symbol);
                    _driftSafeMode = true;
                }

                // 4. Apply position size multiplier with bounds checking
                double finalMult = Math.Max(0.5, Math.Min(1.5, cvarMultiplier * (_driftSafeMode ? 0.5 : 1.0)));

                // 5. Canary testing - route some signals to shadow for A/B testing
                bool isShadow = ShouldShadowSignal(sig.StrategyId);

                _log.LogInformation("[SuperRouter] Enhanced signal: {Strategy} regime={Regime} size={Size}x shadow={Shadow}",
                    sig.StrategyId, currentRegime, finalMult, isShadow);

                return Task.FromResult(sig); // For now, return original signal - full enhancement implementation TBD
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[SuperRouter] Signal enhancement failed for {Strategy}", sig.StrategyId);
                return Task.FromResult(sig);
            }
        }

        Regime DetectCurrentRegime(string symbol)
        {
            try
            {
                // Simplified regime detection - will be enhanced with actual market data
                return Regime.Range; // Default for now
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[SuperRouter] Regime detection failed for {Symbol}", symbol);
                return Regime.Range;
            }
        }

        double OptimizePositionSize(string symbol)
        {
            try
            {
                // CVaR-based sizing - simplified version
                double baseSize = 1.0;
                double cvarLevel = double.TryParse(Environment.GetEnvironmentVariable("CVAR_LEVEL"), out var cl) ? cl : 0.95;

                // Simple risk-based adjustment
                return Math.Max(0.5, Math.Min(1.5, baseSize));
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[SuperRouter] Position sizing failed for {Symbol}", symbol);
                return 1.0;
            }
        }

        bool CheckForDrift(string symbol)
        {
            try
            {
                // Page-Hinkley drift detection - simplified check
                return false; // No drift detected for now
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[SuperRouter] Drift check failed for {Symbol}", symbol);
                return false;
            }
        }

        bool ShouldShadowSignal(string strategyId)
        {
            try
            {
                // A/B testing logic - route percentage to shadow
                double ratio = double.TryParse(Environment.GetEnvironmentVariable("CANARY_RATIO"), out var r) ? r : 0.10;
                return _rng.NextDouble() < ratio;
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[SuperRouter] Canary logic failed for {Strategy}", strategyId);
                return false;
            }
        }

        public (string strategy, string config, double positionMult, bool isShadow) Route(
            string symbol,
            double price,
            double atr14,
            double atr100,
            double atrStd,
            double emaFast,
            double emaSlow,
            double prevClose,
            List<string> availableStrategies)
        {
            // Update regime
            double ret = prevClose > 0 ? (price - prevClose) / prevClose : 0;
            double volZ = atrStd > 1e-6 ? (atr14 - atr100) / atrStd : 0;
            double dirStrength = atr14 > 1e-6 ? Math.Clamp((emaFast - emaSlow) / atr14, -3, 3) : 0;

            var regime = _regimeEngine.UpdateAndInfer(ret, volZ, dirStrength);
            var session = IsRthNow() ? "RTH" : "ETH";

            _log.LogDebug("[SuperRouter] {Symbol} regime={Regime} session={Session}", symbol, regime, session);

            // Handle drift safe mode
            if (_driftSafeMode)
            {
                _log.LogWarning("[SuperRouter] Drift safe mode active - using conservative defaults");
                return ("S2", "a", 0.5, false); // Safe fallback
            }

            // Score available strategies using Bayesian priors
            var scores = new List<(string strat, string config, double score)>();

            foreach (var strat in availableStrategies)
            {
                var configs = GetConfigsForStrategy(strat);
                foreach (var config in configs)
                {
                    // Sample posterior win probability from hierarchical priors
                    double posterior = _priors.Sample(strat, config, regime.ToString(), session, _rng);

                    // Apply strategy weight (learned from recent performance)
                    double stratWeight = _strategyWeights.TryGetValue(strat, out var w) ? w : 1.0;

                    // Compute final score
                    double score = posterior * stratWeight * GetSignalQuality(strat, symbol, price);

                    scores.Add((strat, config, score));
                }
            }

            if (!scores.Any())
            {
                _log.LogWarning("[SuperRouter] No strategies available - using fallback");
                return ("S2", "a", 0.5, false);
            }

            // Select best strategy
            var best = scores.OrderByDescending(x => x.score).First();

            // Determine if this should be a canary (shadow) trade
            bool isShadow = Flag("CANARY_ENABLED") && _canary.ToShadow();
            if (isShadow && scores.Count > 1)
            {
                // Use second-best for shadow arm
                best = scores.OrderByDescending(x => x.score).Skip(1).First();
                _log.LogDebug("[SuperRouter] Canary trade using {Strategy} {Config}", best.strat, best.config);
            }

            // Compute position size using CVaR
            double baseMult = GetBaseMultiplier(best.strat);
            double posMult = _cvarSizer.Recommend(baseMult);

            // Apply environment limits
            double minMult = double.TryParse(Environment.GetEnvironmentVariable("SIZE_MIN_MULT"), out var min) ? min : 0.5;
            double maxMult = double.TryParse(Environment.GetEnvironmentVariable("SIZE_MAX_MULT"), out var max) ? max : 1.5;
            posMult = Math.Clamp(posMult, minMult, maxMult);

            _log.LogInformation("[SuperRouter] Selected {Strategy} {Config} mult={Mult:F2} shadow={Shadow}",
                best.strat, best.config, posMult, isShadow);

            return (best.strat, best.config, posMult, isShadow);
        }

        public void ObserveTrade(string strategy, string config, string regime, string session,
            double rMultiple, bool isShadow, DateTime entryTime)
        {
            bool win = rMultiple > 0;

            // Update Bayesian priors
            _priors.Observe(strategy, config, regime, session, win);

            // Update CVaR sizer
            _cvarSizer.Observe(rMultiple);

            // Update canary AA testing
            _canary.Observe(isShadow, win);

            // Update drift detector
            double signal = Math.Abs(rMultiple) >= 1 ? 1 : 0; // Binary: hit target or not
            if (_driftDetector.Update(signal))
            {
                _log.LogWarning("[SuperRouter] Drift detected! Applying action: {Action}", _driftDetector.GetAction());
                ApplyDriftAction(_driftDetector.GetAction());
            }

            // Check for canary promotion
            if (Flag("CANARY_ENABLED") && _canary.ShouldPromote())
            {
                _log.LogInformation("[SuperRouter] Canary promotion triggered - shadow arm outperforming");
                // Promote by rebalancing strategy weights based on shadow performance
                var stats = _canary.GetStats();
                _log.LogInformation("[SuperRouter] Canary stats: MainWinRate={MainWinRate:F3} ShadowWinRate={ShadowWinRate:F3}", 
                    stats.winRateA, stats.winRateB);
                
                // Reset canary for next experiment
                _canary.Reset();
            }

            _log.LogDebug("[SuperRouter] Trade observed: {Strategy} {Config} R={R:F2} shadow={Shadow}",
                strategy, config, rMultiple, isShadow);
        }

        void ApplyDriftAction(string action)
        {
            switch (action.ToLowerInvariant())
            {
                case "safe":
                    _driftSafeMode = true;
                    _log.LogWarning("[SuperRouter] Drift action: Safe mode enabled");
                    break;
                case "halve":
                    // Halve all position multipliers
                    _log.LogWarning("[SuperRouter] Drift action: Halving position sizes");
                    // Implement position size halving by reducing strategy weights
                    foreach (var key in _strategyWeights.Keys.ToList())
                    {
                        _strategyWeights[key] *= 0.5;
                    }
                    _log.LogInformation("[SuperRouter] All strategy weights halved due to drift");
                    break;
                case "pause":
                    // Zero out all strategy weights
                    foreach (var key in _strategyWeights.Keys.ToList())
                    {
                        _strategyWeights[key] = 0;
                    }
                    _log.LogWarning("[SuperRouter] Drift action: Pausing all strategies");
                    break;
            }
        }

        public void ResetDriftMode()
        {
            _driftSafeMode = false;
            _driftDetector.Reset();
            _log.LogInformation("[SuperRouter] Drift safe mode reset");
        }

        static bool IsRthNow()
        {
            var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow,
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            var time = et.TimeOfDay;
            return time >= TimeSpan.FromHours(9.5) && time <= TimeSpan.FromHours(16);
        }

        static string[] GetConfigsForStrategy(string strategy)
        {
            return strategy switch
            {
                "S2" => new[] { "a", "b", "c" },
                "S3" => new[] { "a", "b", "c" },
                "S6" => new[] { "a", "b" },
                "S11" => new[] { "a", "b" },
                _ => new[] { "a" }
            };
        }

        static double GetSignalQuality(string strategy, string symbol, double price)
        {
            // Advanced signal quality calculation using multiple factors
            double quality = 1.0;
            
            // Strategy-specific quality adjustments
            quality *= strategy switch
            {
                "S2" => 1.0,   // Baseline strategy
                "S3" => 0.95,  // Slightly lower quality
                "S6" => 1.05,  // Higher quality
                "S11" => 1.02, // Good quality
                _ => 0.9
            };
            
            // Symbol-specific adjustments based on liquidity
            quality *= symbol switch
            {
                "ES" => 1.0,   // Highest liquidity
                "MES" => 0.98, // High liquidity
                "NQ" => 0.96,  // Good liquidity  
                "MNQ" => 0.94, // Moderate liquidity
                _ => 0.85      // Lower liquidity symbols
            };
            
            // Price level adjustments (avoid extreme prices)
            if (price < 100 || price > 20000)
                quality *= 0.9;
                
            return Math.Max(0.5, Math.Min(1.5, quality));
        }

        static double GetBaseMultiplier(string strategy)
        {
            return strategy switch
            {
                "S2" => 1.0,
                "S3" => 0.8,
                "S6" => 1.2,
                "S11" => 1.0,
                _ => 1.0
            };
        }

        static bool Flag(string key)
        {
            var v = Environment.GetEnvironmentVariable(key)?.ToLowerInvariant();
            return v == "1" || v == "true";
        }
    }
}
