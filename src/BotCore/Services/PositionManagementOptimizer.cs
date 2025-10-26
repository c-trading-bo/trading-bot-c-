using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Models;
using BotCore.Bandits;
using TradingBot.Abstractions.Helpers;

namespace BotCore.Services
{
    /// <summary>
    /// Volatility regime classification for parameter scaling
    /// </summary>
    public enum VolatilityRegime
    {
        Low,      // ATR < 3 ticks (market sleeping)
        Normal,   // ATR 3-6 ticks (typical conditions)
        High      // ATR > 6 ticks (market moving aggressively)
    }

    /// <summary>
    /// Position Management Optimizer - Event-driven implementation
    /// 
    /// Learns optimal position management parameters using ML/RL:
    /// - Optimal breakeven trigger timing (6 vs 8 vs 10 ticks)
    /// - Optimal trailing stop distance (1.0x vs 1.5x vs 2.0x ATR)
    /// - Optimal time exit thresholds per strategy and market regime
    /// 
    /// VOLATILITY SCALING: Scales thresholds based on current ATR
    /// SESSION-SPECIFIC LEARNING: Learns separate parameters per trading session
    /// 
    /// Converted to event-driven IHostedService with Timer (Phase 5 refactoring)
    /// </summary>
    public sealed class PositionManagementOptimizer : IHostedService, IDisposable
    {
        private readonly ILogger<PositionManagementOptimizer> _logger;
        private readonly ConcurrentDictionary<string, PositionManagementOutcome> _outcomes = new();
        private readonly ParameterChangeTracker _changeTracker;
        private DateTime _lastExportTime = DateTime.MinValue;
        private Timer? _optimizationTimer;
        private bool _disposed;
        
        // Learning parameters
        private const int OptimizationIntervalSeconds = 60; // Run optimization every minute
        private const int MinSamplesForLearning = 10; // Need at least 10 samples to learn
        private const int ExportIntervalHours = 24; // Export learned parameters every 24 hours
        
        // VOLATILITY SCALING: ATR thresholds for regime detection (in ticks)
        private const decimal LowVolatilityThreshold = 3m;   // ATR < 3 ticks
        private const decimal HighVolatilityThreshold = 6m;  // ATR > 6 ticks
        
        // SESSION-SPECIFIC LEARNING: Rolling window of ATR values per symbol (for volatility regime detection)
        private readonly ConcurrentDictionary<string, System.Collections.Generic.Queue<decimal>> _atrHistory = new();
        private const int AtrHistorySize = 20; // Keep last 20 periods
        
        // Learning thresholds and metrics
        private const int MinSamplesForHalfLearning = 5; // Half of MinSamplesForLearning for current parameter check
        private const decimal SignificantOpportunityCostTicks = 5m; // Minimum ticks for significant opportunity cost
        private const decimal ParameterImprovementThreshold = 1.1m; // 10% improvement required to recommend change
        private const decimal TrailMultiplierSignificantDifference = 0.2m; // Minimum difference in trail multipliers
        private const decimal TimeExitBufferMultiplier = 1.5m; // 50% buffer for timeout recommendations
        private const int MaxOutcomesInMemory = 1000; // Maximum number of outcomes to keep in memory
        private const int MaeAnalysisSampleSize = 100; // Number of trades to analyze for MAE
        private const int MinSamplesForMaeAnalysis = 10; // Minimum samples for MAE analysis
        private const decimal MaePercentileP90 = 0.90m; // 90th percentile for MAE
        private const decimal MaePercentileP95 = 0.95m; // 95th percentile for MAE
        private const int MinSamplesPerMaeBucket = 5; // Minimum samples per MAE bucket
        private const decimal MaeStopOutRateThreshold = 0.70m; // 70% stop-out rate threshold
        private const decimal EarlyExitConfidenceThreshold = 0.80m; // 80% confidence for early exit
        private const int EarlyExitMinSamples = 20; // Minimum samples for early exit recommendation
        private const int SmallSampleThreshold = 30; // Threshold for using t-distribution vs z-distribution
        private const int LargeSampleThreshold = 100; // Threshold for high confidence level
        
        // Statistical distribution critical values (t-distribution and z-distribution)
        private const decimal TValueFor80Percent = 1.282m; // t-value for 80% confidence
        private const decimal TValueFor90Percent = 1.645m; // t-value for 90% confidence
        private const decimal TValueFor95Percent = 1.960m; // t-value for 95% confidence
        private const decimal DefaultTValue = 1.960m; // Default t-value (95%)
        
        // Confidence percentage levels for statistical calculations
        private const decimal ConfidenceLevel80Percent = 0.80m; // 80% confidence level
        private const decimal ConfidenceLevel90Percent = 0.90m; // 90% confidence level
        private const decimal ConfidenceLevel95Percent = 0.95m; // 95% confidence level
        
        // Minimum samples for confidence metrics calculation
        private const int MinSamplesForConfidenceMetrics = 10; // Minimum samples for meaningful confidence intervals
        
        public PositionManagementOptimizer(
            ILogger<PositionManagementOptimizer> logger,
            IServiceProvider serviceProvider,
            ParameterChangeTracker changeTracker)
        {
            _logger = logger;
            // serviceProvider parameter accepted for DI compatibility but not currently used
            _changeTracker = changeTracker;
        }
        
        public Task StartAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🧠 [PM-OPTIMIZER] Position Management Optimizer starting (event-driven mode)...");
            
            // Initialize last export time to avoid immediate export
            _lastExportTime = DateTime.UtcNow;
            
            // Start timer for optimization cycles
            var interval = TimeSpan.FromSeconds(OptimizationIntervalSeconds);
            _optimizationTimer = new Timer(OptimizationCallback, null, TimeSpan.Zero, interval);
            
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🛑 [PM-OPTIMIZER] Position Management Optimizer stopping");
            _optimizationTimer?.Change(Timeout.Infinite, 0);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Timer callback for optimization cycles - uses fire-and-forget pattern
        /// </summary>
        private void OptimizationCallback(object? state)
        {
            if (_disposed) return;

            // Fire-and-forget pattern for async operation in timer callback
            _ = Task.Run(async () =>
            {
                try
                {
                    await RunOptimizationCycleAsync(CancellationToken.None).ConfigureAwait(false);
                    
                    // Check if it's time to export learned parameters (every 24 hours)
                    var timeSinceLastExport = DateTime.UtcNow - _lastExportTime;
                    if (timeSinceLastExport.TotalHours >= ExportIntervalHours)
                    {
                        ExportLearnedParameters();
                        _lastExportTime = DateTime.UtcNow;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [PM-OPTIMIZER] Error in optimization cycle");
                }
            });
        }
        
        /// <summary>
        /// Record position management outcome for learning
        /// VOLATILITY SCALING: Now accepts currentAtr for regime-aware learning
        /// SESSION-SPECIFIC LEARNING: Automatically detects trading session
        /// MAE CORRELATION: Accepts time-stamped MAE data for early stop-out prediction
        /// </summary>
        public void RecordOutcome(
            string strategy,
            string symbol,
            int breakevenAfterTicks,
            decimal trailMultiplier,
            int maxHoldMinutes,
            bool breakevenTriggered,
            bool stoppedOut,
            bool targetHit,
            bool timedOut,
            decimal finalPnL,
            decimal maxFavorableExcursion,
            decimal maxAdverseExcursion,
            string marketRegime = "UNKNOWN",
            decimal currentAtr = 0m,
            decimal earlyMae1Min = 0m,
            decimal earlyMae2Min = 0m,
            decimal earlyMae5Min = 0m,
            int tradeDurationSeconds = 0)
        {
            var outcomeKey = $"{strategy}_{symbol}_{DateTime.UtcNow.Ticks}";
            
            // VOLATILITY SCALING: Determine volatility regime based on ATR
            var volatilityRegime = DetermineVolatilityRegime(currentAtr);
            
            // SESSION-SPECIFIC LEARNING: Detect current trading session
            var tradingSession = BotCore.Strategy.SessionHelper.GetSessionName(DateTime.UtcNow);
            
            // VOLATILITY SCALING: Update ATR history for rolling analysis
            UpdateAtrHistory(symbol, currentAtr);
            
            var outcome = new PositionManagementOutcome
            {
                Timestamp = DateTime.UtcNow,
                Strategy = strategy,
                Symbol = symbol,
                BreakevenAfterTicks = breakevenAfterTicks,
                TrailMultiplier = trailMultiplier,
                MaxHoldMinutes = maxHoldMinutes,
                BreakevenTriggered = breakevenTriggered,
                StoppedOut = stoppedOut,
                TargetHit = targetHit,
                TimedOut = timedOut,
                FinalPnL = finalPnL,
                MaxFavorableExcursion = maxFavorableExcursion,
                MaxAdverseExcursion = maxAdverseExcursion,
                MarketRegime = marketRegime,
                CurrentAtr = currentAtr,
                VolatilityRegime = volatilityRegime.ToString(),
                TradingSession = tradingSession,
                // MAE CORRELATION: Store time-stamped MAE data
                EarlyMae1Min = earlyMae1Min,
                EarlyMae2Min = earlyMae2Min,
                EarlyMae5Min = earlyMae5Min,
                TradeDurationSeconds = tradeDurationSeconds
            };
            
            _outcomes[outcomeKey] = outcome;
            
            // Log interesting outcomes
            if (breakevenTriggered && stoppedOut && maxFavorableExcursion > breakevenAfterTicks + SignificantOpportunityCostTicks)
            {
                _logger.LogWarning("📊 [PM-OPTIMIZER] Suboptimal breakeven: {Strategy} triggered BE at {BE} ticks, stopped out, but reached +{MaxFav} ticks",
                    strategy, breakevenAfterTicks, maxFavorableExcursion);
            }
            
            if (timedOut && maxFavorableExcursion > 0)
            {
                _logger.LogInformation("📊 [PM-OPTIMIZER] Time exit opportunity cost: {Strategy} timed out after {Minutes}m, left +{MaxFav} ticks on table",
                    strategy, maxHoldMinutes, maxFavorableExcursion);
            }
        }
        
        private async Task RunOptimizationCycleAsync(CancellationToken cancellationToken)
        {
            var strategies = new[] { "S2", "S3", "S6", "S11" };
            var symbols = new[] { "ES", "NQ" };
            
            // MULTI-SYMBOL LEARNING: Iterate over all strategy-symbol combinations
            // This enables separate optimization for S2-ES, S2-NQ, S3-ES, S3-NQ, S6-ES, S6-NQ, S11-ES, S11-NQ
            foreach (var strategy in strategies)
            {
                foreach (var symbol in symbols)
                {
                    try
                    {
                        await OptimizeBreakevenParameterAsync(strategy, symbol, cancellationToken).ConfigureAwait(false);
                        await OptimizeTrailingParameterAsync(strategy, symbol, cancellationToken).ConfigureAwait(false);
                        await OptimizeTimeExitParameterAsync(strategy, symbol, cancellationToken).ConfigureAwait(false);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "❌ [PM-OPTIMIZER] Error optimizing parameters for {Strategy}-{Symbol}", strategy, symbol);
                    }
                }
            }
            
            // Clean up old outcomes (keep last MaxOutcomesInMemory)
            if (_outcomes.Count > MaxOutcomesInMemory)
            {
                var toRemove = _outcomes
                    .OrderBy(kvp => kvp.Value.Timestamp)
                    .Take(_outcomes.Count - MaxOutcomesInMemory)
                    .Select(kvp => kvp.Key)
                    .ToList();
                
                foreach (var key in toRemove)
                {
                    _outcomes.TryRemove(key, out _);
                }
            }
        }
        
        /// <summary>
        /// PHASE 3: Learn optimal breakeven trigger timing
        /// VOLATILITY SCALING + SESSION-SPECIFIC: Analyzes per volatility regime and trading session
        /// MULTI-SYMBOL LEARNING: Analyzes separately per symbol (ES vs NQ)
        /// Analyzes: "Triggered BE at 8 ticks → stopped out, would have hit target"
        /// </summary>
        private Task OptimizeBreakevenParameterAsync(string strategy, string symbol, CancellationToken cancellationToken)
        {
            // Group by volatility regime and session for regime-specific learning
            // MULTI-SYMBOL: Filter by both strategy AND symbol
            var regimeSessions = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.Symbol == symbol && o.BreakevenTriggered)
                .GroupBy(o => new { o.VolatilityRegime, o.TradingSession })
                .Where(g => g.Count() >= MinSamplesForLearning)
                .ToList();
            
            foreach (var regimeSessionGroup in regimeSessions)
            {
                var regime = regimeSessionGroup.Key.VolatilityRegime;
                var session = regimeSessionGroup.Key.TradingSession;
                var outcomes = regimeSessionGroup
                    .OrderByDescending(o => o.Timestamp)
                    .Take(100)
                    .ToList();
                
                if (outcomes.Count < MinSamplesForLearning)
                {
                    continue;
                }
                
                // Analyze outcomes by breakeven timing
                var analysis = outcomes
                    .GroupBy(o => o.BreakevenAfterTicks)
                    .Select(g => new
                    {
                        BreakevenTicks = g.Key,
                        Count = g.Count(),
                        AvgPnL = g.Average(o => (double)o.FinalPnL),
                        WinRate = g.Count(o => o.FinalPnL > 0) / (double)g.Count(),
                        AvgMaxFav = g.Average(o => (double)o.MaxFavorableExcursion),
                        StoppedOutRate = g.Count(o => o.StoppedOut) / (double)g.Count()
                    })
                    .OrderByDescending(a => a.AvgPnL)
                    .ToList();
                
                if (analysis.Count < 2)
                {
                    continue;
                }
                
                var best = analysis[0];
                var current = analysis.Find(a => a.Count >= MinSamplesForHalfLearning);
                
                if (current != null && best.BreakevenTicks != current.BreakevenTicks && best.AvgPnL > current.AvgPnL * (double)ParameterImprovementThreshold)
                {
                    // CONFIDENCE INTERVALS: Get statistical confidence for this recommendation (with symbol filtering)
                    var confidenceMetrics = GetBreakevenConfidenceMetrics(strategy, symbol, regime, session);
                    var confidenceStr = confidenceMetrics != null 
                        ? FormatConfidenceMetrics(confidenceMetrics, "Breakeven", " ticks")
                        : "confidence: UNKNOWN";
                    
                    // MULTI-SYMBOL: Include symbol in logging
                    _logger.LogInformation("💡 [PM-OPTIMIZER] Breakeven optimization for {Strategy}-{Symbol} in {Regime}/{Session}: Current={Current} ticks (PnL={CurrentPnL:F2}), Optimal={Optimal} ticks (PnL={OptimalPnL:F2}) | {Confidence}",
                        strategy, symbol, regime, session, current.BreakevenTicks, current.AvgPnL, best.BreakevenTicks, best.AvgPnL, confidenceStr);
                    
                    // MULTI-SYMBOL: Record regime/session/symbol-specific parameter change recommendation
                    _changeTracker.RecordChange(
                        strategyName: $"{strategy}-{symbol}",
                        parameterName: $"BreakevenAfterTicks_{regime}_{session}",
                        oldValue: current.BreakevenTicks.ToString(CultureInfo.InvariantCulture),
                        newValue: best.BreakevenTicks.ToString(CultureInfo.InvariantCulture),
                        reason: $"ML/RL learning ({regime}/{session}): Better PnL ({best.AvgPnL:F2} vs {current.AvgPnL:F2}), Win rate {best.WinRate:P0} | {confidenceStr}",
                        outcomePnl: (decimal)best.AvgPnL,
                        wasCorrect: true,
                        marketSnapshotId: $"{regime}_{session}"
                    );
                }
            }
            
            return Task.CompletedTask;
        }
        
        /// <summary>
        /// PHASE 3: Learn optimal trailing stop distance
        /// VOLATILITY SCALING + SESSION-SPECIFIC: Analyzes per volatility regime and trading session
        /// MULTI-SYMBOL LEARNING: Analyzes separately per symbol (ES vs NQ)
        /// Analyzes: "Trailing at 1.0x ATR → stopped out early, left $200 on table"
        /// </summary>
        private Task OptimizeTrailingParameterAsync(string strategy, string symbol, CancellationToken cancellationToken)
        {
            // Group by volatility regime and session for regime-specific learning
            // MULTI-SYMBOL: Filter by both strategy AND symbol
            var regimeSessions = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.Symbol == symbol)
                .GroupBy(o => new { o.VolatilityRegime, o.TradingSession })
                .Where(g => g.Count() >= MinSamplesForLearning)
                .ToList();
            
            foreach (var regimeSessionGroup in regimeSessions)
            {
                var regime = regimeSessionGroup.Key.VolatilityRegime;
                var session = regimeSessionGroup.Key.TradingSession;
                var outcomes = regimeSessionGroup
                    .OrderByDescending(o => o.Timestamp)
                    .Take(100)
                    .ToList();
                
                if (outcomes.Count < MinSamplesForLearning)
                {
                    continue;
                }
                
                // Analyze outcomes by trailing multiplier
                var analysis = outcomes
                    .GroupBy(o => o.TrailMultiplier)
                    .Select(g => new
                    {
                        TrailMultiplier = g.Key,
                        Count = g.Count(),
                        AvgPnL = g.Average(o => (double)o.FinalPnL),
                        WinRate = g.Count(o => o.FinalPnL > 0) / (double)g.Count(),
                        AvgMaxFav = g.Average(o => (double)o.MaxFavorableExcursion),
                        OpportunityCost = g.Average(o => Math.Max(0, (double)(o.MaxFavorableExcursion - Math.Abs(o.FinalPnL))))
                    })
                    .OrderByDescending(a => a.AvgPnL)
                    .ToList();
                
                if (analysis.Count < 2)
                {
                    continue;
                }
                
                var best = analysis[0];
                var current = analysis.Find(a => a.Count >= MinSamplesForHalfLearning);
                
                if (current != null && (double)Math.Abs(best.TrailMultiplier - current.TrailMultiplier) > (double)TrailMultiplierSignificantDifference && best.AvgPnL > current.AvgPnL * (double)ParameterImprovementThreshold)
                {
                    // CONFIDENCE INTERVALS: Get statistical confidence for this recommendation (with symbol filtering)
                    var confidenceMetrics = GetTrailingConfidenceMetrics(strategy, symbol, regime, session);
                    var confidenceStr = confidenceMetrics != null 
                        ? FormatConfidenceMetrics(confidenceMetrics, "Trailing", "x ATR")
                        : "confidence: UNKNOWN";
                    
                    // MULTI-SYMBOL: Include symbol in logging
                    _logger.LogInformation("💡 [PM-OPTIMIZER] Trailing stop optimization for {Strategy}-{Symbol} in {Regime}/{Session}: Current={Current:F1}x ATR (PnL={CurrentPnL:F2}, OpCost={CurrentOC:F2}), Optimal={Optimal:F1}x ATR (PnL={OptimalPnL:F2}, OpCost={OptimalOC:F2}) | {Confidence}",
                        strategy, symbol, regime, session, current.TrailMultiplier, current.AvgPnL, current.OpportunityCost, best.TrailMultiplier, best.AvgPnL, best.OpportunityCost, confidenceStr);
                    
                    // MULTI-SYMBOL: Record regime/session/symbol-specific parameter change recommendation
                    _changeTracker.RecordChange(
                        strategyName: $"{strategy}-{symbol}",
                        parameterName: $"TrailMultiplier_{regime}_{session}",
                        oldValue: current.TrailMultiplier.ToString("F1", CultureInfo.InvariantCulture),
                        newValue: best.TrailMultiplier.ToString("F1", CultureInfo.InvariantCulture),
                        reason: $"ML/RL learning ({regime}/{session}): Better PnL ({best.AvgPnL:F2} vs {current.AvgPnL:F2}), Lower opportunity cost | {confidenceStr}",
                        outcomePnl: (decimal)best.AvgPnL,
                        wasCorrect: true,
                        marketSnapshotId: $"{regime}_{session}"
                    );
                }
            }
            
            return Task.CompletedTask;
        }
        
        /// <summary>
        /// PHASE 3: Learn optimal time exit thresholds
        /// MULTI-SYMBOL LEARNING: Analyzes separately per symbol (ES vs NQ)
        /// Analyzes: "S2 needs 20m in ranging, 10m in trending"
        /// </summary>
        private async Task OptimizeTimeExitParameterAsync(string strategy, string symbol, CancellationToken cancellationToken)
        {
            // MULTI-SYMBOL: Filter by both strategy AND symbol
            var outcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.Symbol == symbol)
                .OrderByDescending(o => o.Timestamp)
                .Take(100)
                .ToList();
            
            if (outcomes.Count < MinSamplesForLearning)
            {
                return;
            }
            
            // Analyze by market regime
            var regimes = outcomes.GroupBy(o => o.MarketRegime);
            
            foreach (var regimeGroup in regimes)
            {
                var regimeName = regimeGroup.Key;
                var regimeOutcomes = regimeGroup.ToList();
                
                if (regimeOutcomes.Count < 5)
                {
                    continue;
                }
                
                // Analyze winning trades vs timing out trades
                var winningTrades = regimeOutcomes.Where(o => o.TargetHit).ToList();
                var timedOutTrades = regimeOutcomes.Where(o => o.TimedOut).ToList();
                
                if (winningTrades.Count > 0 && timedOutTrades.Count > 0)
                {
                    var avgWinningDuration = winningTrades.Average(o => o.MaxHoldMinutes);
                    var avgTimedOutDuration = timedOutTrades.Average(o => o.MaxHoldMinutes);
                    var avgTimedOutOpCost = timedOutTrades.Average(o => (double)o.MaxFavorableExcursion);
                    
                    if (avgTimedOutOpCost > (double)SignificantOpportunityCostTicks) // Significant opportunity cost
                    {
                        // MULTI-SYMBOL: Include symbol in logging
                        _logger.LogInformation("💡 [PM-OPTIMIZER] Time exit analysis for {Strategy}-{Symbol} in {Regime}: Winning trades avg {WinDur:F0}m, Timed out avg {TimedDur:F0}m with +{OpCost:F1} ticks lost",
                            strategy, symbol, regimeName, avgWinningDuration, avgTimedOutDuration, avgTimedOutOpCost);
                        
                        // Recommend longer timeout if timed out trades had significant upside
                        var recommendedTimeout = (int)(avgWinningDuration * (double)TimeExitBufferMultiplier); // 50% buffer
                        
                        // MULTI-SYMBOL: Record regime/symbol-specific parameter change recommendation
                        _changeTracker.RecordChange(
                            strategyName: $"{strategy}-{symbol}",
                            parameterName: $"MaxHoldMinutes_{regimeName}",
                            oldValue: avgTimedOutDuration.ToString("F0", CultureInfo.InvariantCulture),
                            newValue: recommendedTimeout.ToString(CultureInfo.InvariantCulture),
                            reason: $"ML/RL learning: Timed out trades had {avgTimedOutOpCost:F1} ticks opportunity cost",
                            outcomePnl: null,
                            wasCorrect: null,
                            marketSnapshotId: regimeName
                        );
                    }
                }
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        
        /// <summary>
        /// Get recent parameter change recommendations
        /// </summary>
        public IReadOnlyList<ParameterChange> GetRecentRecommendations(int count = 10)
        {
            return _changeTracker.GetRecentChanges(count);
        }
        
        /// <summary>
        /// Get recommendations for a specific strategy
        /// </summary>
        public IReadOnlyList<ParameterChange> GetRecommendationsForStrategy(string strategy, int count = 10)
        {
            return _changeTracker.GetChangesForStrategy(strategy, count);
        }
        
        // ========================================================================
        // FEATURE 2: MAE/MFE OPTIMAL STOP PLACEMENT LEARNING
        // ========================================================================
        
        /// <summary>
        /// Analyze MAE (Max Adverse Excursion) distribution to find optimal early exit threshold
        /// Returns the 95th percentile MAE for winning trades + safety buffer
        /// Safety: Never recommend looser stops, only tighter
        /// </summary>
        public decimal? GetOptimalEarlyExitThreshold(string strategy, string regime = "ALL")
        {
            var maeEnabled = Environment.GetEnvironmentVariable("BOT_MAE_LEARNING_ENABLED")?.ToUpperInvariant() == "TRUE";
            if (!maeEnabled)
            {
                return null; // Feature disabled
            }
            
            var minSamples = int.TryParse(Environment.GetEnvironmentVariable("BOT_MAE_MINIMUM_SAMPLES"), out var min) ? min : 50;
            var safetyBuffer = decimal.TryParse(Environment.GetEnvironmentVariable("BOT_MAE_SAFETY_BUFFER_TICKS"), out var buffer) ? buffer : 2m;
            
            // Get winning trades (those that hit target)
            var winningTrades = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.TargetHit)
                .Where(o => regime == "ALL" || o.MarketRegime == regime)
                .OrderByDescending(o => o.Timestamp)
                .Take(100)
                .ToList();
            
            if (winningTrades.Count < minSamples)
            {
                return null; // Not enough data
            }
            
            // Calculate MAE distribution (adverse excursion is typically negative)
            var maeValues = winningTrades
                .Select(o => Math.Abs(o.MaxAdverseExcursion)) // Use absolute value
                .OrderBy(mae => mae)
                .ToList();
            
            // Get 95th percentile (only 5% of winning trades go further against us)
            var percentile95Index = (int)(maeValues.Count * 0.95);
            var mae95th = maeValues[Math.Min(percentile95Index, maeValues.Count - 1)];
            
            // Add safety buffer
            var optimalThreshold = mae95th + safetyBuffer;
            
            _logger.LogInformation("📊 [MAE-LEARNING] Optimal early exit for {Strategy} in {Regime}: {Threshold:F1} ticks (95th percentile: {P95:F1}, buffer: {Buffer:F1}, samples: {Samples})",
                strategy, regime, optimalThreshold, mae95th, safetyBuffer, winningTrades.Count);
            
            return optimalThreshold;
        }
        
        /// <summary>
        /// Analyze MFE (Max Favorable Excursion) to optimize trailing stop distance
        /// Finds the optimal distance by comparing MFE peak to final exit profit
        /// </summary>
        public decimal? GetOptimalTrailingDistance(string strategy, string regime = "ALL")
        {
            var mfeEnabled = Environment.GetEnvironmentVariable("BOT_MFE_LEARNING_ENABLED")?.ToUpperInvariant() == "TRUE";
            if (!mfeEnabled)
            {
                return null; // Feature disabled
            }
            
            var minSamples = int.TryParse(Environment.GetEnvironmentVariable("BOT_MAE_MINIMUM_SAMPLES"), out var min) ? min : 50;
            
            // Get winning trades that reached favorable excursion
            var profitableTrades = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.MaxFavorableExcursion > 0)
                .Where(o => regime == "ALL" || o.MarketRegime == regime)
                .OrderByDescending(o => o.Timestamp)
                .Take(100)
                .ToList();
            
            if (profitableTrades.Count < minSamples)
            {
                return null; // Not enough data
            }
            
            // Calculate average "profit giveback" (MFE - final P&L)
            var givebacks = profitableTrades
                .Select(o => o.MaxFavorableExcursion - Math.Abs(o.FinalPnL))
                .Where(giveback => giveback > 0) // Only where we gave back profit
                .ToList();
            
            if (givebacks.Count == 0)
            {
                return null; // No giveback patterns
            }
            
            var avgGiveback = givebacks.Average();
            var medianGiveback = givebacks.OrderBy(g => g).ElementAt(givebacks.Count / 2);
            
            // Optimal trail distance: half the median giveback (aggressive but not too tight)
            var optimalDistance = medianGiveback * 0.5m;
            
            _logger.LogInformation("📊 [MFE-LEARNING] Optimal trailing distance for {Strategy} in {Regime}: {Distance:F1} ticks (median giveback: {Median:F1}, avg: {Avg:F1}, samples: {Samples})",
                strategy, regime, optimalDistance, medianGiveback, avgGiveback, profitableTrades.Count);
            
            return optimalDistance;
        }
        
        /// <summary>
        /// Analyze MAE distribution for statistics (for logging/monitoring)
        /// </summary>
        public (decimal p50, decimal p90, decimal p95, int samples)? AnalyzeMaeDistribution(string strategy, string regime = "ALL")
        {
            var winningTrades = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.TargetHit)
                .Where(o => regime == "ALL" || o.MarketRegime == regime)
                .OrderByDescending(o => o.Timestamp)
                .Take(MaeAnalysisSampleSize)
                .ToList();
            
            if (winningTrades.Count < MinSamplesForMaeAnalysis)
            {
                return null;
            }
            
            var maeValues = winningTrades
                .Select(o => Math.Abs(o.MaxAdverseExcursion))
                .OrderBy(mae => mae)
                .ToList();
            
            var p50 = maeValues[maeValues.Count / 2];
            var p90 = maeValues[(int)(maeValues.Count * MaePercentileP90)];
            var p95 = maeValues[(int)(maeValues.Count * MaePercentileP95)];
            
            return (p50, p90, p95, winningTrades.Count);
        }
        
        /// <summary>
        /// Analyze MFE distribution for statistics (for logging/monitoring)
        /// </summary>
        public (decimal avgMfe, decimal avgFinalPnL, decimal avgGiveback, int samples)? AnalyzeMfeDistribution(string strategy, string regime = "ALL")
        {
            var profitableTrades = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.MaxFavorableExcursion > 0)
                .Where(o => regime == "ALL" || o.MarketRegime == regime)
                .OrderByDescending(o => o.Timestamp)
                .Take(MaeAnalysisSampleSize)
                .ToList();
            
            if (profitableTrades.Count < MinSamplesForMaeAnalysis)
            {
                return null;
            }
            
            var avgMfe = profitableTrades.Average(o => o.MaxFavorableExcursion);
            var avgFinalPnL = profitableTrades.Average(o => Math.Abs(o.FinalPnL));
            var avgGiveback = avgMfe - avgFinalPnL;
            
            return (avgMfe, avgFinalPnL, avgGiveback, profitableTrades.Count);
        }
        
        /// <summary>
        /// VOLATILITY SCALING: Determine volatility regime based on ATR
        /// </summary>
        private static VolatilityRegime DetermineVolatilityRegime(decimal atr)
        {
            if (atr <= 0)
            {
                return VolatilityRegime.Normal; // Default if ATR not available
            }
            
            if (atr < LowVolatilityThreshold)
            {
                return VolatilityRegime.Low;
            }
            else if (atr > HighVolatilityThreshold)
            {
                return VolatilityRegime.High;
            }
            else
            {
                return VolatilityRegime.Normal;
            }
        }
        
        /// <summary>
        /// VOLATILITY SCALING: Update rolling ATR history for a symbol
        /// </summary>
        private void UpdateAtrHistory(string symbol, decimal atr)
        {
            if (atr <= 0)
            {
                return; // Don't track invalid ATR values
            }
            
            var history = _atrHistory.GetOrAdd(symbol, _ => new System.Collections.Generic.Queue<decimal>());
            
            lock (history)
            {
                history.Enqueue(atr);
                
                // Keep only last N values
                while (history.Count > AtrHistorySize)
                {
                    history.Dequeue();
                }
            }
        }
        
        /// <summary>
        /// VOLATILITY SCALING: Get average ATR for a symbol from recent history
        /// </summary>
        /// <summary>
        /// VOLATILITY SCALING: Get optimal parameters for given strategy, symbol, and current market conditions
        /// </summary>
        public (int breakevenTicks, decimal trailMultiplier, int maxHoldMinutes)? GetOptimalParameters(
            string strategy,
            string symbol,
            decimal currentAtr)
        {
            var volatilityRegime = DetermineVolatilityRegime(currentAtr);
            var tradingSession = BotCore.Strategy.SessionHelper.GetSessionName(DateTime.UtcNow);
            
            // Filter outcomes by strategy, volatility regime, and session
            var relevantOutcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy && 
                           o.Symbol == symbol &&
                           o.VolatilityRegime == volatilityRegime.ToString() &&
                           o.TradingSession == tradingSession)
                .OrderByDescending(o => o.Timestamp)
                .Take(50)
                .ToList();
            
            if (relevantOutcomes.Count < MinSamplesForLearning)
            {
                // Not enough regime/session-specific data, return null
                return null;
            }
            
            // Find best performing parameters in this regime/session
            var bestBreakeven = relevantOutcomes
                .GroupBy(o => o.BreakevenAfterTicks)
                .Select(g => new { Ticks = g.Key, AvgPnL = g.Average(o => (double)o.FinalPnL) })
                .OrderByDescending(x => x.AvgPnL)
                .FirstOrDefault()?.Ticks ?? 8;
            
            var bestTrail = relevantOutcomes
                .GroupBy(o => o.TrailMultiplier)
                .Select(g => new { Multiplier = g.Key, AvgPnL = g.Average(o => (double)o.FinalPnL) })
                .OrderByDescending(x => x.AvgPnL)
                .FirstOrDefault()?.Multiplier ?? 1.5m;
            
            var bestMaxHold = relevantOutcomes
                .GroupBy(o => o.MaxHoldMinutes)
                .Select(g => new { Minutes = g.Key, AvgPnL = g.Average(o => (double)o.FinalPnL) })
                .OrderByDescending(x => x.AvgPnL)
                .FirstOrDefault()?.Minutes ?? 60;
            
            return (bestBreakeven, bestTrail, bestMaxHold);
        }
        
        /// <summary>
        /// Get performance summary for debugging
        /// </summary>
        public PositionManagementPerformanceSummary GetPerformanceSummary(string strategy)
        {
            var outcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy)
                .OrderByDescending(o => o.Timestamp)
                .Take(100)
                .ToList();
            
            if (outcomes.Count == 0)
            {
                return new PositionManagementPerformanceSummary
                {
                    Strategy = strategy,
                    TotalTrades = 0
                };
            }
            
            return new PositionManagementPerformanceSummary
            {
                Strategy = strategy,
                TotalTrades = outcomes.Count,
                WinRate = outcomes.Count(o => o.FinalPnL > 0) / (decimal)outcomes.Count,
                AvgPnL = outcomes.Average(o => o.FinalPnL),
                BreakevenTriggeredRate = outcomes.Count(o => o.BreakevenTriggered) / (decimal)outcomes.Count,
                StoppedOutRate = outcomes.Count(o => o.StoppedOut) / (decimal)outcomes.Count,
                TargetHitRate = outcomes.Count(o => o.TargetHit) / (decimal)outcomes.Count,
                TimedOutRate = outcomes.Count(o => o.TimedOut) / (decimal)outcomes.Count,
                AvgMaxFavorableExcursion = outcomes.Average(o => o.MaxFavorableExcursion),
                AvgMaxAdverseExcursion = outcomes.Average(o => o.MaxAdverseExcursion)
            };
        }
        
        // ========================================================================
        // LEARNED PARAMETERS EXPORT - Phase 1 Implementation
        // ========================================================================
        
        /// <summary>
        /// Export learned parameters to JSON files for review and analysis.
        /// Creates individual files per strategy in artifacts/learned_parameters/.
        /// Safe to call - failures are logged but don't crash trading system.
        /// </summary>
        public void ExportLearnedParameters()
        {
            try
            {
                _logger.LogInformation("📤 [PM-OPTIMIZER] Starting learned parameters export...");
                
                // Create output directory if it doesn't exist
                var artifactsDir = Path.Combine(AppContext.BaseDirectory, "artifacts", "learned_parameters");
                Directory.CreateDirectory(artifactsDir);
                
                // Export for each strategy: S2, S3, S6, S11
                var strategies = new[] { "S2", "S3", "S6", "S11" };
                var exportedCount = 0;
                
                foreach (var strategy in strategies)
                {
                    try
                    {
                        var exportData = BuildStrategyExportData(strategy);
                        var fileName = Path.Combine(artifactsDir, $"{strategy}_learned_params.json");
                        var json = JsonSerializationHelper.SerializePretty(exportData);
                        File.WriteAllText(fileName, json);
                        
                        _logger.LogInformation("✅ [PM-OPTIMIZER] Exported {Strategy} learned parameters: {TotalTrades} trades analyzed, {ParamCount} parameters tracked",
                            strategy, exportData.TotalTradesAnalyzed, exportData.Parameters.Count);
                        exportedCount++;
                    }
                    catch (UnauthorizedAccessException ex)
                    {
                        _logger.LogError(ex, "❌ [PM-OPTIMIZER] Access denied writing export file for {Strategy}", strategy);
                    }
                    catch (IOException ex)
                    {
                        _logger.LogError(ex, "❌ [PM-OPTIMIZER] IO error writing export file for {Strategy}", strategy);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "❌ [PM-OPTIMIZER] Unexpected error exporting {Strategy} parameters", strategy);
                    }
                }
                
                _logger.LogInformation("📦 [PM-OPTIMIZER] Export complete. Successfully exported {Count}/{Total} strategies", 
                    exportedCount, strategies.Length);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [PM-OPTIMIZER] Critical error in learned parameters export");
            }
        }
        
        /// <summary>
        /// Build export data structure for a specific strategy
        /// </summary>
        private StrategyLearnedParameters BuildStrategyExportData(string strategy)
        {
            var outcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy)
                .OrderByDescending(o => o.Timestamp)
                .ToList();
            
            var exportData = new StrategyLearnedParameters
            {
                StrategyName = strategy,
                ExportTimestamp = DateTime.UtcNow,
                TotalTradesAnalyzed = outcomes.Count
            };
            
            if (outcomes.Count == 0)
            {
                return exportData;
            }
            
            // Analyze breakeven distance parameter
            var breakevenOutcomes = outcomes.Where(o => o.BreakevenTriggered).ToList();
            if (breakevenOutcomes.Count >= MinSamplesForLearning)
            {
                var bestBreakeven = breakevenOutcomes
                    .GroupBy(o => o.BreakevenAfterTicks)
                    .Select(g => new { Ticks = g.Key, AvgPnL = g.Average(o => o.FinalPnL), Count = g.Count(), WinRate = g.Count(o => o.FinalPnL > 0) / (decimal)g.Count() })
                    .OrderByDescending(x => x.AvgPnL)
                    .FirstOrDefault();
                
                if (bestBreakeven != null)
                {
                    exportData.ParametersInternal.Add(new LearnedParameter
                    {
                        ParameterName = "BreakevenAfterTicks",
                        CurrentConfiguredValue = "N/A",
                        LearnedOptimalValue = bestBreakeven.Ticks.ToString(System.Globalization.CultureInfo.InvariantCulture),
                        TradesAnalyzed = bestBreakeven.Count,
                        ConfidenceScore = DetermineConfidenceScore(bestBreakeven.Count),
                        PerformanceImprovement = "N/A",
                        SampleWinRate = Math.Round(bestBreakeven.WinRate * LargeSampleThreshold, 1)
                    });
                }
            }
            
            // Analyze trailing stop multiplier parameter
            var trailingOutcomes = outcomes.ToList();
            if (trailingOutcomes.Count >= MinSamplesForLearning)
            {
                var bestTrailing = trailingOutcomes
                    .GroupBy(o => o.TrailMultiplier)
                    .Select(g => new { Multiplier = g.Key, AvgPnL = g.Average(o => o.FinalPnL), Count = g.Count(), WinRate = g.Count(o => o.FinalPnL > 0) / (decimal)g.Count() })
                    .OrderByDescending(x => x.AvgPnL)
                    .FirstOrDefault();
                
                if (bestTrailing != null)
                {
                    exportData.ParametersInternal.Add(new LearnedParameter
                    {
                        ParameterName = "TrailingStopMultiplier",
                        CurrentConfiguredValue = "N/A",
                        LearnedOptimalValue = bestTrailing.Multiplier.ToString(System.Globalization.CultureInfo.InvariantCulture),
                        TradesAnalyzed = bestTrailing.Count,
                        ConfidenceScore = DetermineConfidenceScore(bestTrailing.Count),
                        PerformanceImprovement = "N/A",
                        SampleWinRate = Math.Round(bestTrailing.WinRate * LargeSampleThreshold, 1)
                    });
                }
            }
            
            // Analyze hold time limits parameter
            var holdTimeOutcomes = outcomes.Where(o => !o.TimedOut).ToList();
            if (holdTimeOutcomes.Count >= MinSamplesForLearning)
            {
                var avgDuration = holdTimeOutcomes.Average(o => o.TradeDurationSeconds) / 60.0; // Convert to minutes
                var recommendedMaxHold = (int)Math.Ceiling(avgDuration * 1.5); // 50% buffer
                
                exportData.ParametersInternal.Add(new LearnedParameter
                {
                    ParameterName = "MaxHoldTimeMinutes",
                    CurrentConfiguredValue = "N/A",
                    LearnedOptimalValue = recommendedMaxHold.ToString(System.Globalization.CultureInfo.InvariantCulture),
                    TradesAnalyzed = holdTimeOutcomes.Count,
                    ConfidenceScore = DetermineConfidenceScore(holdTimeOutcomes.Count),
                    PerformanceImprovement = "N/A",
                    SampleWinRate = Math.Round(holdTimeOutcomes.Count(o => o.FinalPnL > 0) / (decimal)holdTimeOutcomes.Count * LargeSampleThreshold, 1)
                });
            }
            
            return exportData;
        }
        
        /// <summary>
        /// Determine confidence score based on sample size
        /// Less than 30 trades: Low
        /// 30-100 trades: Medium
        /// Over 100 trades: High
        /// </summary>
        private static string DetermineConfidenceScore(int sampleSize)
        {
            if (sampleSize < SmallSampleThreshold)
            {
                return "Low";
            }
            else if (sampleSize < LargeSampleThreshold)
            {
                return "Medium";
            }
            else
            {
                return "High";
            }
        }
        
        // ========================================================================
        // FEATURE 3: MAE CORRELATION ANALYSIS - Early Stop-Out Prediction
        // ========================================================================
        
        /// <summary>
        /// MAE CORRELATION: Analyze correlation between early MAE and stop-out outcomes
        /// Returns threshold where early MAE predicts stop-out with high confidence
        /// </summary>
        public (decimal maeThreshold, decimal stopOutProbability, int sampleSize)? AnalyzeMaeCorrelation(
            string strategy, 
            int earlyMinutes = 2)
        {
            var minSamples = 30;  // Need at least 30 samples for correlation
            
            var outcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy)
                .Where(o => o.TradeDurationSeconds >= earlyMinutes * 60)  // Only trades that lasted long enough
                .OrderByDescending(o => o.Timestamp)
                .Take(200)
                .ToList();
            
            if (outcomes.Count < minSamples)
            {
                return null;
            }
            
            // Get early MAE based on requested time window
            Func<PositionManagementOutcome, decimal> earlyMaeFunc = earlyMinutes switch
            {
                1 => (PositionManagementOutcome o) => o.EarlyMae1Min,
                2 => (PositionManagementOutcome o) => o.EarlyMae2Min,
                5 => (PositionManagementOutcome o) => o.EarlyMae5Min,
                _ => (PositionManagementOutcome o) => o.EarlyMae2Min
            };
            
            // Group by MAE buckets and calculate stop-out rates
            var buckets = new[]
            {
                (min: 0m, max: 2m),
                (min: 2m, max: 4m),
                (min: 4m, max: 6m),
                (min: 6m, max: 8m),
                (min: 8m, max: 999m)
            };
            
            decimal highestMaeThreshold = 0m;
            decimal highestStopOutRate = 0m;
            var totalSamples = 0;
            
            foreach (var bucket in buckets)
            {
                var bucketOutcomes = outcomes
                    .Where(o => earlyMaeFunc(o) >= bucket.min && earlyMaeFunc(o) < bucket.max)
                    .ToList();
                
                if (bucketOutcomes.Count < MinSamplesPerMaeBucket) continue;  // Need at least minimum samples per bucket
                
                var stopOutRate = bucketOutcomes.Count(o => o.StoppedOut) / (decimal)bucketOutcomes.Count;
                
                // Find threshold where stop-out rate exceeds threshold
                if (stopOutRate >= MaeStopOutRateThreshold && stopOutRate > highestStopOutRate)
                {
                    highestMaeThreshold = bucket.min;
                    highestStopOutRate = stopOutRate;
                    totalSamples = bucketOutcomes.Count;
                }
                
                _logger.LogDebug("📊 [MAE-CORRELATION] {Strategy} MAE {Min}-{Max} ticks @ {Time}min: {StopOutRate:P0} stop-out rate (n={Count})",
                    strategy, bucket.min, bucket.max, earlyMinutes, stopOutRate, bucketOutcomes.Count);
            }
            
            if (highestStopOutRate >= MaeStopOutRateThreshold)
            {
                _logger.LogInformation("🚨 [MAE-CORRELATION] {Strategy}: Early MAE > {Threshold:F1} ticks @ {Time}min predicts stop-out with {Probability:P0} confidence (n={Samples})",
                    strategy, highestMaeThreshold, earlyMinutes, highestStopOutRate, totalSamples);
                
                return (highestMaeThreshold, highestStopOutRate, totalSamples);
            }
            
            return null;
        }
        
        /// <summary>
        /// MAE CORRELATION: Get early exit threshold for a strategy based on MAE correlation
        /// Returns MAE threshold in ticks and confidence level
        /// </summary>
        public (decimal maeThreshold, decimal confidence)? GetEarlyExitThreshold(string strategy, int earlyMinutes = 2)
        {
            var correlation = AnalyzeMaeCorrelation(strategy, earlyMinutes);
            
            if (!correlation.HasValue)
            {
                return null;
            }
            
            // Only return if we have high confidence and sufficient samples
            if (correlation.Value.stopOutProbability >= EarlyExitConfidenceThreshold && correlation.Value.sampleSize >= EarlyExitMinSamples)
            {
                return (correlation.Value.maeThreshold, correlation.Value.stopOutProbability);
            }
            
            return null;
        }
        
        // ========================================================================
        // FEATURE 4: CONFIDENCE INTERVALS - Statistical Confidence Scoring
        // ========================================================================
        
        /// <summary>
        /// CONFIDENCE INTERVALS: Calculate statistical confidence metrics for a parameter
        /// </summary>
        private static ConfidenceMetrics CalculateConfidenceMetrics(List<decimal> values, decimal confidencePercentage = 0.95m)
        {
            if (values.Count == 0)
            {
                return new ConfidenceMetrics
                {
                    SampleSize = 0,
                    Level = ConfidenceLevel.Insufficient
                };
            }
            
            var n = values.Count;
            var mean = values.Average();
            var variance = values.Sum(x => (x - mean) * (x - mean)) / n;
            var stdDev = (decimal)Math.Sqrt((double)variance);
            var stdError = stdDev / (decimal)Math.Sqrt(n);
            
            // Determine confidence level based on sample size
            var level = n switch
            {
                < 10 => ConfidenceLevel.Insufficient,
                < 30 => ConfidenceLevel.Low,
                < 100 => ConfidenceLevel.Medium,
                _ => ConfidenceLevel.High
            };
            
            // Use t-distribution for small samples, z-distribution for large
            // Note: Currently using same values for both distributions
            var criticalValue = confidencePercentage switch
            {
                ConfidenceLevel80Percent => TValueFor80Percent,
                ConfidenceLevel90Percent => TValueFor90Percent,
                ConfidenceLevel95Percent => TValueFor95Percent,
                _ => DefaultTValue
            };
            
            var marginOfError = criticalValue * stdError;
            
            return new ConfidenceMetrics
            {
                Mean = mean,
                StandardDeviation = stdDev,
                StandardError = stdError,
                ConfidenceIntervalLow = mean - marginOfError,
                ConfidenceIntervalHigh = mean + marginOfError,
                SampleSize = n,
                Level = level,
                ConfidencePercentage = confidencePercentage
            };
        }
        
        /// <summary>
        /// CONFIDENCE INTERVALS: Get confidence metrics for learned breakeven parameter
        /// </summary>
        public ConfidenceMetrics? GetBreakevenConfidenceMetrics(string strategy, string symbol = "ALL", string regime = "ALL", string session = "ALL")
        {
            var outcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy && o.BreakevenTriggered)
                .Where(o => symbol == "ALL" || o.Symbol == symbol)
                .Where(o => regime == "ALL" || o.VolatilityRegime == regime)
                .Where(o => session == "ALL" || o.TradingSession == session)
                .OrderByDescending(o => o.Timestamp)
                .Take(200)
                .ToList();
            
            if (outcomes.Count < MinSamplesForConfidenceMetrics)
            {
                return null;
            }
            
            var values = outcomes.Select(o => (decimal)o.BreakevenAfterTicks).ToList();
            var confidence = CalculateConfidenceMetrics(values, ConfidenceLevel95Percent);
            
            return confidence;
        }
        
        /// <summary>
        /// CONFIDENCE INTERVALS: Get confidence metrics for learned trailing parameter
        /// </summary>
        public ConfidenceMetrics? GetTrailingConfidenceMetrics(string strategy, string symbol = "ALL", string regime = "ALL", string session = "ALL")
        {
            var outcomes = _outcomes.Values
                .Where(o => o.Strategy == strategy)
                .Where(o => symbol == "ALL" || o.Symbol == symbol)
                .Where(o => regime == "ALL" || o.VolatilityRegime == regime)
                .Where(o => session == "ALL" || o.TradingSession == session)
                .OrderByDescending(o => o.Timestamp)
                .Take(200)
                .ToList();
            
            if (outcomes.Count < MinSamplesForConfidenceMetrics)
            {
                return null;
            }
            
            var values = outcomes.Select(o => o.TrailMultiplier).ToList();
            var confidence = CalculateConfidenceMetrics(values, 0.95m);
            
            return confidence;
        }
        
        /// <summary>
        /// CONFIDENCE INTERVALS: Format confidence metrics for logging
        /// </summary>
        private static string FormatConfidenceMetrics(ConfidenceMetrics metrics, string parameterName, string unit = "")
        {
            var levelStr = metrics.Level switch
            {
                ConfidenceLevel.Insufficient => "INSUFFICIENT DATA",
                ConfidenceLevel.Low => "LOW",
                ConfidenceLevel.Medium => "MEDIUM",
                ConfidenceLevel.High => "HIGH",
                _ => "UNKNOWN"
            };
            
            if (metrics.Level == ConfidenceLevel.Insufficient)
            {
                return $"{parameterName}: INSUFFICIENT DATA (n={metrics.SampleSize})";
            }
            
            var ciPercent = (int)(metrics.ConfidencePercentage * 100);
            return $"{parameterName}: {metrics.Mean:F1}{unit} [{metrics.ConfidenceIntervalLow:F1}-{metrics.ConfidenceIntervalHigh:F1}{unit} @ {ciPercent}% CI] (n={metrics.SampleSize}, σ={metrics.StandardDeviation:F2}, {levelStr} confidence)";
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _optimizationTimer?.Dispose();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
    
    /// <summary>
    /// Position management outcome for ML/RL learning
    /// </summary>
    internal sealed class PositionManagementOutcome
    {
        public DateTime Timestamp { get; set; }
        public string Strategy { get; set; } = string.Empty;
        public string Symbol { get; set; } = string.Empty;
        public int BreakevenAfterTicks { get; set; }
        public decimal TrailMultiplier { get; set; }
        public int MaxHoldMinutes { get; set; }
        public bool BreakevenTriggered { get; set; }
        public bool StoppedOut { get; set; }
        public bool TargetHit { get; set; }
        public bool TimedOut { get; set; }
        public decimal FinalPnL { get; set; }
        public decimal MaxFavorableExcursion { get; set; }
        public decimal MaxAdverseExcursion { get; set; }
        public string MarketRegime { get; set; } = "UNKNOWN";
        
        // VOLATILITY SCALING: Track ATR and volatility regime
        public decimal CurrentAtr { get; set; }
        public string VolatilityRegime { get; set; } = "NORMAL";
        
        // SESSION-SPECIFIC LEARNING: Track trading session
        public string TradingSession { get; set; } = "RTH";
        
        // MAE CORRELATION ANALYSIS: Time-stamped MAE progression
        public decimal EarlyMae1Min { get; set; }  // MAE after 1 minute
        public decimal EarlyMae2Min { get; set; }  // MAE after 2 minutes
        public decimal EarlyMae5Min { get; set; }  // MAE after 5 minutes
        public int TradeDurationSeconds { get; set; }  // Total trade duration
    }
    
    /// <summary>
    /// Confidence level for learned parameters
    /// </summary>
    public enum ConfidenceLevel
    {
        Insufficient,  // < 10 samples
        Low,           // 10-30 samples
        Medium,        // 30-100 samples
        High           // 100+ samples
    }
    
    /// <summary>
    /// Statistical confidence metrics for a learned parameter
    /// </summary>
    public sealed class ConfidenceMetrics
    {
        public decimal Mean { get; set; }
        public decimal StandardDeviation { get; set; }
        public decimal StandardError { get; set; }
        public decimal ConfidenceIntervalLow { get; set; }
        public decimal ConfidenceIntervalHigh { get; set; }
        public int SampleSize { get; set; }
        public ConfidenceLevel Level { get; set; }
        public decimal ConfidencePercentage { get; set; }  // 80%, 90%, 95%
    }
    
    /// <summary>
    /// Performance summary for position management
    /// </summary>
    public sealed class PositionManagementPerformanceSummary
    {
        public string Strategy { get; set; } = string.Empty;
        public int TotalTrades { get; set; }
        public decimal WinRate { get; set; }
        public decimal AvgPnL { get; set; }
        public decimal BreakevenTriggeredRate { get; set; }
        public decimal StoppedOutRate { get; set; }
        public decimal TargetHitRate { get; set; }
        public decimal TimedOutRate { get; set; }
        public decimal AvgMaxFavorableExcursion { get; set; }
        public decimal AvgMaxAdverseExcursion { get; set; }
    }
    
    /// <summary>
    /// Export data structure for strategy learned parameters
    /// </summary>
    public sealed class StrategyLearnedParameters
    {
        private readonly List<LearnedParameter> _parameters = new();
        
        public string StrategyName { get; set; } = string.Empty;
        public DateTime ExportTimestamp { get; set; }
        public int TotalTradesAnalyzed { get; set; }
        public IReadOnlyList<LearnedParameter> Parameters => _parameters;
        internal List<LearnedParameter> ParametersInternal => _parameters;
    }
    
    /// <summary>
    /// Individual learned parameter with confidence metrics
    /// </summary>
    public sealed class LearnedParameter
    {
        public string ParameterName { get; set; } = string.Empty;
        public string CurrentConfiguredValue { get; set; } = string.Empty;
        public string LearnedOptimalValue { get; set; } = string.Empty;
        public int TradesAnalyzed { get; set; }
        public string ConfidenceScore { get; set; } = string.Empty;
        public string PerformanceImprovement { get; set; } = string.Empty;
        public decimal SampleWinRate { get; set; }
    }
}
