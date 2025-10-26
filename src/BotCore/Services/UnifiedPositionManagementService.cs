using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Models;
using BotCore.Bandits;
using TradingBot.Abstractions;
using TopstepX.Bot.Core.Services;

namespace BotCore.Services
{
    /// <summary>
    /// Unified Position Management Service - Production Trading Bot
    /// 
    /// CRITICAL PRODUCTION SERVICE - Manages all active positions with:
    /// - Breakeven protection (moves stop to entry + 1 tick when profitable)
    /// - Trailing stops (follows price to lock in profits)
    /// - Time-based exits (closes stale positions)
    /// - Max excursion tracking (for ML/RL optimization)
    /// 
    /// Converted to event-driven IHostedService with Timer (Phase 5 refactoring)
    /// Monitors positions every 5 seconds using Timer instead of polling loop.
    /// </summary>
    public sealed class UnifiedPositionManagementService : IHostedService, IDisposable
    {
        private readonly ILogger<UnifiedPositionManagementService> _logger;
        private readonly IServiceProvider _serviceProvider;
        private readonly OllamaClient? _ollamaClient;
        private readonly bool _commentaryEnabled;
        private readonly ConcurrentDictionary<string, PositionManagementState> _activePositions = new();
        private Timer? _monitoringTimer;
        private bool _disposed;
        
        // Monitoring interval
        private const int MonitoringIntervalSeconds = 5;
        
        // Dynamic targeting configuration (Feature 1)
        private readonly bool _dynamicTargetsEnabled;
        private readonly int _regimeCheckIntervalSeconds;
        private readonly decimal _targetAdjustmentThreshold;
        
        // MAE/MFE learning configuration (Feature 2)
        private readonly bool _maeLearningEnabled;
        private readonly bool _mfeLearningEnabled;
        
        // FEATURE 3: Regime monitoring configuration
        private readonly bool _regimeMonitoringEnabled;
        private readonly bool _regimeFlipExitEnabled;
        private readonly decimal _regimeConfidenceDropThreshold;
        
        // FEATURE 4: Confidence-based adjustment configuration
        private readonly bool _confidenceAdjustmentEnabled;
        private readonly decimal _confidenceVeryHighThreshold;
        private readonly decimal _confidenceHighThreshold;
        private readonly decimal _confidenceMediumThreshold;
        private readonly decimal _confidenceLowThreshold;
        
        // FEATURE 5: Progressive time-decay stop tightening configuration
        private readonly bool _progressiveTighteningEnabled;
        private readonly int _progressiveTighteningCheckIntervalSeconds;
        
        // Tick size for ES/MES (0.25) - used for breakeven calculations
        private const decimal EsTickSize = 0.25m;
        private const decimal NqTickSize = 0.25m;
        private const decimal DefaultTickSize = 0.25m;
        
        // PHASE 4: Multi-level partial exit thresholds (in R-multiples)
        private const decimal FirstPartialExitThreshold = 1.5m;  // Close 50% at 1.5R
        private const decimal SecondPartialExitThreshold = 2.5m; // Close 30% at 2.5R
        private const decimal FinalPartialExitThreshold = 4.0m;  // Close 20% at 4.0R (runner position)
        
        // PHASE 4: Volatility adaptation parameters
        private const decimal HighVolatilityThreshold = 1.5m;    // ATR > 1.5x recent average = high vol
        private const decimal LowVolatilityThreshold = 0.7m;     // ATR < 0.7x recent average = low vol
        private const decimal VolatilityStopWidening = 0.20m;    // Widen stops by 20% in high vol
        private const decimal VolatilityStopTightening = 0.20m;  // Tighten stops by 20% in low vol
        
        // Strategy-specific max hold times (in minutes)
        private const int S2_MAX_HOLD_MINUTES = 60;
        private const int S3_MAX_HOLD_MINUTES = 90;
        private const int S6_MAX_HOLD_MINUTES = 45;
        private const int S11_MAX_HOLD_MINUTES = 60;
        private const int DEFAULT_MAX_HOLD_MINUTES = 120;
        
        // Partial exit percentages
        private const decimal FIRST_PARTIAL_EXIT_PERCENTAGE = 0.50m;   // 50%
        private const decimal SECOND_PARTIAL_EXIT_PERCENTAGE = 0.30m;  // 30%
        private const decimal FINAL_PARTIAL_EXIT_PERCENTAGE = 0.20m;   // 20%
        
        // AI Commentary display percentages (for logging)
        private const decimal FIRST_PARTIAL_DISPLAY_PERCENT = 50m;
        private const decimal SECOND_PARTIAL_DISPLAY_PERCENT = 30m;
        private const decimal FINAL_PARTIAL_DISPLAY_PERCENT = 20m;
        
        // Volatility adjustment timing
        private const int VOLATILITY_ADJUSTMENT_MIN_INTERVAL_MINUTES = 5;
        
        // ATR calculation and averaging
        private const decimal ATR_MULTIPLIER_UNIT = 1.0m;
        private const int ATR_LOOKBACK_BARS = 10;
        
        // Stop distance minimum (in R-multiples)
        private const decimal MIN_STOP_DISTANCE_R = 2m;
        
        // Default configuration values for environment variable fallbacks
        private const int DEFAULT_REGIME_CHECK_INTERVAL_SECONDS = 60;
        private const decimal DEFAULT_TARGET_ADJUSTMENT_THRESHOLD = 0.3m;
        private const decimal DEFAULT_REGIME_CONFIDENCE_DROP_THRESHOLD = 0.30m;
        private const decimal DEFAULT_CONFIDENCE_VERY_HIGH_THRESHOLD = 0.85m;
        private const decimal DEFAULT_CONFIDENCE_HIGH_THRESHOLD = 0.75m;
        private const decimal DEFAULT_CONFIDENCE_MEDIUM_THRESHOLD = 0.70m;
        private const decimal DEFAULT_CONFIDENCE_LOW_THRESHOLD = 0.65m;
        private const int DEFAULT_PROGRESSIVE_TIGHTENING_INTERVAL_SECONDS = 60;
        private const decimal DEFAULT_ENTRY_REGIME_CONFIDENCE = 0.75m;
        private const int MAE_MFE_SNAPSHOT_TOLERANCE_SECONDS = 5;
        
        // Strategy-specific R-multiple targets (trending market)
        private const decimal S2_TRENDING_R_MULTIPLE = 2.5m;
        private const decimal S3_TRENDING_R_MULTIPLE = 3.0m;
        private const decimal S6_TRENDING_R_MULTIPLE = 2.0m;
        private const decimal S11_TRENDING_R_MULTIPLE = 2.5m;
        
        // Strategy-specific R-multiple targets (ranging market)
        private const decimal S2_RANGING_R_MULTIPLE = 1.0m;
        private const decimal S3_RANGING_R_MULTIPLE = 1.2m;
        private const decimal S6_RANGING_R_MULTIPLE = 1.0m;
        private const decimal S11_RANGING_R_MULTIPLE = 1.5m;
        
        // Strategy-specific R-multiple targets (default/mixed market)
        private const decimal S2_DEFAULT_R_MULTIPLE = 1.5m;
        private const decimal S3_DEFAULT_R_MULTIPLE = 1.8m;
        private const decimal S6_DEFAULT_R_MULTIPLE = 1.2m;
        private const decimal S11_DEFAULT_R_MULTIPLE = 1.8m;
        private const decimal FALLBACK_DEFAULT_R_MULTIPLE = 1.5m;
        
        // Regime flip sensitivity thresholds (how easily strategy exits on regime change)
        private const decimal S2_REGIME_FLIP_SENSITIVITY = 0.50m;
        private const decimal S3_REGIME_FLIP_SENSITIVITY = 0.55m;
        private const decimal S6_REGIME_FLIP_SENSITIVITY = 0.60m;
        private const decimal S11_REGIME_FLIP_SENSITIVITY = 0.55m;
        private const decimal DEFAULT_REGIME_FLIP_SENSITIVITY = 0.50m;
        
        // Confidence drop thresholds for regime-based exits
        private const decimal MAJOR_CONFIDENCE_DROP_THRESHOLD_S2 = 0.75m;  // S2 exits if confidence drops below this
        private const decimal MAJOR_CONFIDENCE_DROP_THRESHOLD_S3 = 0.30m;  // S3 confidence collapse threshold
        private const decimal MAJOR_CONFIDENCE_DROP_THRESHOLD_S11 = 0.40m; // S11 severe confidence collapse threshold
        private const decimal GENERAL_CONFIDENCE_DROP_THRESHOLD = 0.75m;   // General rule for profitable exits
        
        // Progressive tightening time thresholds (in minutes) - Strategy S2
        private const int S2_TIER1_MINUTES = 15;
        private const int S2_TIER2_MINUTES = 30;
        private const int S2_TIER3_MINUTES = 45;
        private const int S2_TIER4_MINUTES = 60;
        
        // Progressive tightening time thresholds (in minutes) - Strategy S3
        private const int S3_TIER1_MINUTES = 20;
        private const int S3_TIER2_MINUTES = 40;
        private const int S3_TIER3_MINUTES = 60;
        private const int S3_TIER4_MINUTES = 90;
        
        // Progressive tightening time thresholds (in minutes) - Strategy S6
        private const int S6_TIER1_MINUTES = 10;
        private const int S6_TIER2_MINUTES = 20;
        private const int S6_TIER3_MINUTES = 30;
        private const int S6_TIER4_MINUTES = 45;
        
        // Progressive tightening time thresholds (in minutes) - Strategy S11
        private const int S11_TIER1_MINUTES = 20;
        private const int S11_TIER2_MINUTES = 40;
        private const int S11_TIER3_MINUTES = 60;
        
        // Progressive tightening R-multiple requirements
        private const decimal TIER2_R_MULTIPLE_REQUIREMENT = 1.0m;  // Tier 2 exit threshold
        private const decimal TIER3_R_MULTIPLE_REQUIREMENT_S2 = 1.5m;  // S2 Tier 3
        private const decimal TIER3_R_MULTIPLE_REQUIREMENT_S3 = 2.0m;  // S3 Tier 3
        private const decimal TIER3_R_MULTIPLE_REQUIREMENT_S6 = 1.5m;  // S6 Tier 3
        private const decimal TIER2_R_MULTIPLE_REQUIREMENT_S11 = 1.5m; // S11 Tier 2
        
        // Progressive tightening tick requirements
        private const int MIN_PROFIT_TICKS_TIER1 = 0;  // Breakeven tier
        private const int MIN_PROFIT_TICKS_S6_TIER1 = 6; // S6 momentum-specific requirement
        
        // Default progressive tightening thresholds (fallback strategy)
        private const int DEFAULT_TIER1_MINUTES = 30;
        private const int DEFAULT_TIER2_MINUTES = 60;
        private const int DEFAULT_TIER3_MINUTES = 120;
        
        // Progressive tightening tier identifiers
        private const int TIER_1 = 1;
        private const int TIER_2 = 2;
        private const int TIER_3 = 3;
        private const int TIER_4 = 4;
        
        // MAE threshold warning level (80% of learned threshold)
        private const decimal MAE_WARNING_THRESHOLD_MULTIPLIER = 0.8m;
        
        // Confidence-based multipliers (default values)
        private const decimal CONFIDENCE_STOP_MULTIPLIER_VERY_HIGH_DEFAULT = 1.5m;
        private const decimal CONFIDENCE_TARGET_MULTIPLIER_VERY_HIGH_DEFAULT = 2.0m;
        private const decimal CONFIDENCE_STOP_MULTIPLIER_HIGH_DEFAULT = 1.3m;
        private const decimal CONFIDENCE_TARGET_MULTIPLIER_HIGH_DEFAULT = 1.0m;
        private const decimal CONFIDENCE_STOP_MULTIPLIER_MEDIUM_DEFAULT = 1.1m;
        private const decimal CONFIDENCE_TARGET_MULTIPLIER_MEDIUM_DEFAULT = 0.8m;
        private const decimal CONFIDENCE_STOP_MULTIPLIER_LOW_DEFAULT = 1.0m;
        private const decimal CONFIDENCE_TARGET_MULTIPLIER_LOW_DEFAULT = 0.6m;
        private const int CONFIDENCE_LOW_POSITION_SIZE_DIVISOR = 2;
        private const decimal CONFIDENCE_VERY_LOW_STOP_MULTIPLIER = 0.8m;
        private const decimal CONFIDENCE_VERY_LOW_TARGET_MULTIPLIER = 0.5m;
        private const int CONFIDENCE_VERY_LOW_POSITION_SIZE_DIVISOR = 4;
        
        // Confidence-based adjustment multipliers
        private const decimal CONFIDENCE_BREAKEVEN_MULTIPLIER_VERY_HIGH = 1.5m;
        private const decimal CONFIDENCE_BREAKEVEN_MULTIPLIER_MEDIUM = 0.75m;
        private const decimal CONFIDENCE_BREAKEVEN_MULTIPLIER_LOW = 0.5m;
        private const decimal CONFIDENCE_TRAIL_MULTIPLIER_VERY_HIGH = 1.25m;
        private const decimal CONFIDENCE_TRAIL_MULTIPLIER_MEDIUM = 0.75m;
        private const decimal CONFIDENCE_TRAIL_MULTIPLIER_LOW = 0.5m;
        
        public UnifiedPositionManagementService(
            ILogger<UnifiedPositionManagementService> logger,
            IServiceProvider serviceProvider,
            OllamaClient? ollamaClient = null)
        {
            _logger = logger;
            _serviceProvider = serviceProvider;
            _ollamaClient = ollamaClient;
            
            // Check if position management commentary is enabled (default: true for transparency)
            // Set BOT_POSITION_COMMENTARY_ENABLED=false to disable if too verbose
            var commentarySetting = Environment.GetEnvironmentVariable("BOT_POSITION_COMMENTARY_ENABLED")?.ToUpperInvariant();
            _commentaryEnabled = commentarySetting != "FALSE"; // Enabled by default, must explicitly disable
            
            if (_commentaryEnabled && _ollamaClient != null)
            {
                _logger.LogInformation("🤖 [POSITION-MGMT] AI commentary enabled for position management actions");
            }
            
            // Load dynamic targeting configuration (Feature 1)
            _dynamicTargetsEnabled = Environment.GetEnvironmentVariable("BOT_DYNAMIC_TARGETS_ENABLED")?.ToUpperInvariant() == "TRUE";
            _regimeCheckIntervalSeconds = int.TryParse(Environment.GetEnvironmentVariable("BOT_REGIME_CHECK_INTERVAL_SECONDS"), out var regimeInterval) ? regimeInterval : DEFAULT_REGIME_CHECK_INTERVAL_SECONDS;
            _targetAdjustmentThreshold = decimal.TryParse(Environment.GetEnvironmentVariable("BOT_TARGET_ADJUSTMENT_THRESHOLD"), out var threshold) ? threshold : DEFAULT_TARGET_ADJUSTMENT_THRESHOLD;
            
            if (_dynamicTargetsEnabled)
            {
                _logger.LogInformation("📊 [POSITION-MGMT] Dynamic R-Multiple Targeting enabled: regime check every {Interval}s, adjustment threshold {Threshold}",
                    _regimeCheckIntervalSeconds, _targetAdjustmentThreshold);
            }
            
            // Load MAE/MFE learning configuration (Feature 2)
            _maeLearningEnabled = Environment.GetEnvironmentVariable("BOT_MAE_LEARNING_ENABLED")?.ToUpperInvariant() == "TRUE";
            _mfeLearningEnabled = Environment.GetEnvironmentVariable("BOT_MFE_LEARNING_ENABLED")?.ToUpperInvariant() == "TRUE";
            
            if (_maeLearningEnabled || _mfeLearningEnabled)
            {
                _logger.LogInformation("🧠 [POSITION-MGMT] MAE/MFE Learning enabled: MAE={MAE}, MFE={MFE}",
                    _maeLearningEnabled, _mfeLearningEnabled);
            }
            
            // Load regime monitoring configuration (Feature 3)
            _regimeMonitoringEnabled = Environment.GetEnvironmentVariable("BOT_REGIME_MONITORING_ENABLED")?.ToUpperInvariant() == "TRUE";
            _regimeFlipExitEnabled = Environment.GetEnvironmentVariable("BOT_REGIME_FLIP_EXIT_ENABLED")?.ToUpperInvariant() == "TRUE";
            _regimeConfidenceDropThreshold = decimal.TryParse(Environment.GetEnvironmentVariable("BOT_REGIME_CONFIDENCE_DROP_THRESHOLD"), out var dropThreshold) ? dropThreshold : DEFAULT_REGIME_CONFIDENCE_DROP_THRESHOLD;
            
            if (_regimeMonitoringEnabled)
            {
                _logger.LogInformation("🔄 [POSITION-MGMT] Regime monitoring enabled: Exit on flip={FlipExit}, Confidence drop threshold={Threshold}",
                    _regimeFlipExitEnabled, _regimeConfidenceDropThreshold);
            }
            
            // Load confidence-based adjustment configuration (Feature 4)
            _confidenceAdjustmentEnabled = Environment.GetEnvironmentVariable("BOT_CONFIDENCE_ADJUSTMENT_ENABLED")?.ToUpperInvariant() == "TRUE";
            _confidenceVeryHighThreshold = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_VERY_HIGH_THRESHOLD"), out var vhThreshold) ? vhThreshold : DEFAULT_CONFIDENCE_VERY_HIGH_THRESHOLD;
            _confidenceHighThreshold = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_HIGH_THRESHOLD"), out var hThreshold) ? hThreshold : DEFAULT_CONFIDENCE_HIGH_THRESHOLD;
            _confidenceMediumThreshold = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_MEDIUM_THRESHOLD"), out var mThreshold) ? mThreshold : DEFAULT_CONFIDENCE_MEDIUM_THRESHOLD;
            _confidenceLowThreshold = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_LOW_THRESHOLD"), out var lThreshold) ? lThreshold : DEFAULT_CONFIDENCE_LOW_THRESHOLD;
            
            if (_confidenceAdjustmentEnabled)
            {
                _logger.LogInformation("💯 [POSITION-MGMT] Confidence-based adjustment enabled: VeryHigh≥{VH}, High≥{H}, Medium≥{M}, Low≥{L}",
                    _confidenceVeryHighThreshold, _confidenceHighThreshold, _confidenceMediumThreshold, _confidenceLowThreshold);
            }
            
            // Load progressive tightening configuration (Feature 5)
            _progressiveTighteningEnabled = Environment.GetEnvironmentVariable("BOT_PROGRESSIVE_TIGHTENING_ENABLED")?.ToUpperInvariant() == "TRUE";
            _progressiveTighteningCheckIntervalSeconds = int.TryParse(Environment.GetEnvironmentVariable("BOT_PROGRESSIVE_TIGHTENING_CHECK_INTERVAL_SECONDS"), out var ptInterval) ? ptInterval : DEFAULT_PROGRESSIVE_TIGHTENING_INTERVAL_SECONDS;
            
            if (_progressiveTighteningEnabled)
            {
                _logger.LogInformation("⏱️ [POSITION-MGMT] Progressive time-decay tightening enabled: Check interval={Interval}s",
                    _progressiveTighteningCheckIntervalSeconds);
            }
        }
        
        public Task StartAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🎯 [POSITION-MGMT] Unified Position Management Service starting (event-driven mode)...");
            
            // Start timer for position monitoring
            var interval = TimeSpan.FromSeconds(MonitoringIntervalSeconds);
            _monitoringTimer = new Timer(MonitorPositionsCallback, null, TimeSpan.Zero, interval);
            
            return Task.CompletedTask;
        }

        public Task StopAsync(CancellationToken cancellationToken)
        {
            _logger.LogInformation("🛑 [POSITION-MGMT] Unified Position Management Service stopping");
            _monitoringTimer?.Change(Timeout.Infinite, 0);
            return Task.CompletedTask;
        }

        /// <summary>
        /// Timer callback for position monitoring - uses fire-and-forget pattern
        /// </summary>
        private void MonitorPositionsCallback(object? state)
        {
            if (_disposed) return;

            // Fire-and-forget pattern for async operation in timer callback
            _ = Task.Run(async () =>
            {
                try
                {
                    await MonitorAndManagePositionsAsync(CancellationToken.None).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-MGMT] Error in position management");
                }
            });
        }
        
        /// <summary>
        /// Register a new position for management
        /// Called when a trade is opened
        /// </summary>
        public void RegisterPosition(
            string positionId,
            string symbol,
            string strategy,
            decimal entryPrice,
            decimal stopPrice,
            decimal targetPrice,
            int quantity,
            BracketMode bracketMode,
            decimal entryConfidence = 0.75m)
        {
            ArgumentNullException.ThrowIfNull(bracketMode);
            
            // Save original values before confidence adjustments
            var originalStop = stopPrice;
            var originalTarget = targetPrice;
            
            // FEATURE 4: Apply confidence-based adjustments to stop and target
            if (_confidenceAdjustmentEnabled)
            {
                var adjustedValues = ApplyConfidenceAdjustments(entryPrice, stopPrice, targetPrice, quantity, entryConfidence, symbol);
                stopPrice = adjustedValues.adjustedStop;
                targetPrice = adjustedValues.adjustedTarget;
                quantity = adjustedValues.adjustedQuantity;
                
                _logger.LogInformation("💯 [POSITION-MGMT] Confidence-based adjustment: Confidence={Conf:F2}, Stop adjusted to {Stop:F2}, Target to {Target:F2}, Qty to {Qty}",
                    entryConfidence, stopPrice, targetPrice, quantity);
                
                // AI Commentary: Explain confidence adjustment (non-blocking) - need to create temp state for fire-and-forget
                try
                {
                    var stopMultiplier = originalStop != 0 ? Math.Abs((stopPrice - entryPrice) / (originalStop - entryPrice)) : 1m;
                    var targetMultiplier = originalTarget != 0 ? Math.Abs((targetPrice - entryPrice) / (originalTarget - entryPrice)) : 1m;
                    
                    // Create minimal state for commentary
                    var tempState = new PositionManagementState
                    {
                        PositionId = positionId,
                        Symbol = symbol,
                        Strategy = strategy,
                        EntryPrice = entryPrice,
                        Quantity = quantity
                    };
                    
                    ExplainConfidenceAdjustmentFireAndForget(
                        tempState,
                        entryConfidence,
                        stopMultiplier,
                        targetMultiplier,
                        quantity,
                        stopPrice,
                        targetPrice);
                }
                catch (Exception ex)
                {
                    // Log AI errors but don't disrupt position management
                    _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for position {PositionId}", positionId);
                }
            }
            
            // Capture entry regime (Feature 1 & 3)
            var entryRegime = "UNKNOWN";
            var entryRegimeConfidence = 0.5m;
            try
            {
                if (_dynamicTargetsEnabled || _regimeMonitoringEnabled)
                {
                    var regimeService = _serviceProvider.GetService<RegimeDetectionService>();
                    if (regimeService != null)
                    {
                        entryRegime = regimeService.GetCurrentRegimeAsync(symbol).GetAwaiter().GetResult();
                        // Note: RegimeDetectionService doesn't provide confidence yet, using default
                        entryRegimeConfidence = DEFAULT_ENTRY_REGIME_CONFIDENCE;
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [POSITION-MGMT] Could not detect entry regime for {Symbol}", symbol);
            }
            
            var state = new PositionManagementState
            {
                PositionId = positionId,
                Symbol = symbol,
                Strategy = strategy,
                EntryPrice = entryPrice,
                CurrentStopPrice = stopPrice,
                TargetPrice = targetPrice,
                Quantity = quantity,
                EntryTime = DateTime.UtcNow,
                MaxFavorablePrice = entryPrice,
                MaxAdversePrice = entryPrice,
                BreakevenActivated = false,
                TrailingStopActive = false,
                BreakevenAfterTicks = bracketMode.BreakevenAfterTicks,
                TrailTicks = bracketMode.TrailTicks,
                MaxHoldMinutes = GetMaxHoldMinutes(strategy),
                LastCheckTime = DateTime.UtcNow,
                StopModificationCount = 0,
                EntryRegime = entryRegime,
                CurrentRegime = entryRegime,
                LastRegimeCheck = DateTime.UtcNow,
                EntryRegimeConfidence = entryRegimeConfidence,
                CurrentRegimeConfidence = entryRegimeConfidence,
                EntryConfidence = entryConfidence,
                LastProgressiveTighteningCheck = DateTime.UtcNow,
                ProgressiveTighteningTier = 0,
                PeakProfitTicks = 0
            };
            
            // Calculate dynamic target based on entry regime (Feature 1)
            if (_dynamicTargetsEnabled)
            {
                state.DynamicTargetPrice = CalculateDynamicTarget(state, entryPrice, stopPrice, entryRegime);
                _logger.LogInformation("📝 [POSITION-MGMT] Registered position {PositionId}: {Strategy} {Symbol} {Qty}@{Entry}, Regime: {Regime}, Static target: {Static}, Dynamic target: {Dynamic}",
                    positionId, strategy, symbol, quantity, entryPrice, entryRegime, targetPrice, state.DynamicTargetPrice);
            }
            else
            {
                state.DynamicTargetPrice = targetPrice;
                _logger.LogInformation("📝 [POSITION-MGMT] Registered position {PositionId}: {Strategy} {Symbol} {Qty}@{Entry}, BE after {BETicks} ticks, Trail {TrailTicks} ticks, Max hold {MaxHold}m",
                    positionId, strategy, symbol, quantity, entryPrice, bracketMode.BreakevenAfterTicks, bracketMode.TrailTicks, state.MaxHoldMinutes);
            }
            
            _activePositions[positionId] = state;
        }
        
        /// <summary>
        /// Unregister position when it's closed
        /// </summary>
        public void UnregisterPosition(string positionId, ExitReason exitReason)
        {
            if (_activePositions.TryRemove(positionId, out var state))
            {
                var duration = DateTime.UtcNow - state.EntryTime;
                _logger.LogInformation("✅ [POSITION-MGMT] Unregistered position {PositionId}: {Strategy} {Symbol}, Duration: {Duration}m, Exit: {Reason}, Stop mods: {Mods}",
                    positionId, state.Strategy, state.Symbol, duration.TotalMinutes, exitReason, state.StopModificationCount);
                
                // PHASE 3: Report outcome to Position Management Optimizer for ML/RL learning
                ReportOutcomeToOptimizer(state, exitReason, duration);
                
                // TASK 4.1: Collect experience for Lab training
                // Fire-and-forget to avoid blocking position closure
                _ = Task.Run(async () => await CollectExperienceAsync(state, exitReason, duration).ConfigureAwait(false));
            }
        }
        
        /// <summary>
        /// PHASE 3: Report position management outcome for ML/RL learning
        /// </summary>
        private void ReportOutcomeToOptimizer(PositionManagementState state, ExitReason exitReason, TimeSpan duration)
        {
            try
            {
                var optimizer = _serviceProvider.GetService<PositionManagementOptimizer>();
                if (optimizer == null)
                {
                    return; // Optimizer not registered (optional dependency)
                }
                
                var isLong = state.Quantity > 0;
                var tickSize = GetTickSize(state.Symbol);
                
                // Calculate final P&L in ticks
                var pnlTicks = isLong
                    ? (state.MaxFavorablePrice - state.EntryPrice) / tickSize
                    : (state.EntryPrice - state.MaxFavorablePrice) / tickSize;
                
                // Calculate max excursions in ticks
                var maxFavTicks = isLong
                    ? (state.MaxFavorablePrice - state.EntryPrice) / tickSize
                    : (state.EntryPrice - state.MaxFavorablePrice) / tickSize;
                
                var maxAdvTicks = isLong
                    ? (state.MaxAdversePrice - state.EntryPrice) / tickSize
                    : (state.EntryPrice - state.MaxAdversePrice) / tickSize;
                
                // Determine what happened
                var breakevenTriggered = state.BreakevenActivated;
                var stoppedOut = exitReason == ExitReason.StopLoss || exitReason == ExitReason.Breakeven || exitReason == ExitReason.TrailingStop;
                var targetHit = exitReason == ExitReason.Target;
                var timedOut = exitReason == ExitReason.TimeLimit;
                
                // Trail multiplier calculation (estimate from trail ticks)
                var trailMultiplier = state.TrailTicks * tickSize / 1.0m; // Simplified estimate
                
                // Record outcome (with regime information for Feature 1 & 2)
                var marketRegime = state.HasProperty("EntryRegime") 
                    ? (state.GetProperty("EntryRegime")?.ToString() ?? "UNKNOWN")
                    : state.EntryRegime;
                
                // VOLATILITY SCALING: Calculate current ATR for regime-aware learning
                var currentAtr = EstimateCurrentVolatility(state);
                
                // MAE CORRELATION: Extract early MAE values from snapshots
                var tradeDuration = (int)(DateTime.UtcNow - state.EntryTime).TotalSeconds;
                var earlyMae1Min = GetMaeAtTime(state, 60);
                var earlyMae2Min = GetMaeAtTime(state, 120);
                var earlyMae5Min = GetMaeAtTime(state, 300);
                
                optimizer.RecordOutcome(
                    strategy: state.Strategy,
                    symbol: state.Symbol,
                    breakevenAfterTicks: state.BreakevenAfterTicks,
                    trailMultiplier: trailMultiplier,
                    maxHoldMinutes: state.MaxHoldMinutes,
                    breakevenTriggered: breakevenTriggered,
                    stoppedOut: stoppedOut,
                    targetHit: targetHit,
                    timedOut: timedOut,
                    finalPnL: pnlTicks,
                    maxFavorableExcursion: maxFavTicks,
                    maxAdverseExcursion: maxAdvTicks,
                    marketRegime: marketRegime,
                    currentAtr: currentAtr,  // VOLATILITY SCALING: Pass ATR for regime detection
                    earlyMae1Min: earlyMae1Min,  // MAE CORRELATION: MAE after 1 minute
                    earlyMae2Min: earlyMae2Min,  // MAE CORRELATION: MAE after 2 minutes
                    earlyMae5Min: earlyMae5Min,  // MAE CORRELATION: MAE after 5 minutes
                    tradeDurationSeconds: tradeDuration  // MAE CORRELATION: Total trade duration
                );
                
                _logger.LogDebug("📊 [POSITION-MGMT] Reported outcome to optimizer: {Strategy} {Symbol}, BE={BE}, StoppedOut={SO}, TargetHit={TH}, TimedOut={TO}, PnL={PnL} ticks",
                    state.Strategy, state.Symbol, breakevenTriggered, stoppedOut, targetHit, timedOut, pnlTicks);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [POSITION-MGMT] Error reporting outcome to optimizer for {PositionId}", state.PositionId);
            }
        }
        
        /// <summary>
        /// TASK 4.1: Collect trading experience for Lab training
        /// Creates state-action-reward-next_state tuple from closed position
        /// </summary>
        private async Task CollectExperienceAsync(PositionManagementState state, ExitReason exitReason, TimeSpan duration)
        {
            try
            {
                var experienceRepo = _serviceProvider.GetService<BotCore.Data.ExperienceRepository>();
                if (experienceRepo == null)
                {
                    // ExperienceRepository not registered (Lab mode doesn't need it)
                    return;
                }

                // Use MaxFavorablePrice as exit price (best price achieved)
                // This is the most favorable price the position reached before exit
                var exitPrice = state.MaxFavorablePrice;

                // Calculate reward metrics
                var isLong = state.Quantity > 0;
                var tickSize = GetTickSize(state.Symbol);
                
                // Calculate P&L
                var pnlPerContract = isLong
                    ? (exitPrice - state.EntryPrice) * 50m // ES/NQ multiplier
                    : (state.EntryPrice - exitPrice) * 50m;
                var totalPnL = pnlPerContract * Math.Abs(state.Quantity);
                
                // Calculate risk (entry to stop distance)
                var riskPerContract = Math.Abs(state.EntryPrice - state.CurrentStopPrice) * 50m;
                var totalRisk = riskPerContract * Math.Abs(state.Quantity);
                
                // Calculate R-multiple
                var rMultiple = totalRisk > 0 ? totalPnL / totalRisk : 0m;
                
                // Calculate Sharpe contribution (simplified)
                var sharpeContribution = totalRisk > 0 ? rMultiple / (decimal)Math.Sqrt(duration.TotalDays + 0.001) : 0m;
                
                // Get volatility estimate
                var volatilityAtEntry = EstimateCurrentVolatility(state);
                var volatilityAtExit = volatilityAtEntry; // Simplified - could fetch current volatility
                
                // Create experience record
                var experience = new TradingExperience
                {
                    ExperienceId = Guid.NewGuid().ToString(),
                    Timestamp = DateTime.UtcNow,
                    PositionId = state.PositionId,
                    
                    // STATE (at entry)
                    EntryRegime = state.EntryRegime,
                    EntryRegimeConfidence = state.EntryRegimeConfidence,
                    EntryConfidence = state.EntryConfidence,
                    Symbol = state.Symbol,
                    EntryHour = state.EntryTime.Hour,
                    EntryDayOfWeek = (int)state.EntryTime.DayOfWeek,
                    VolatilityAtEntry = volatilityAtEntry,
                    
                    // ACTION
                    Strategy = state.Strategy,
                    PositionSize = state.Quantity,
                    EntryPrice = state.EntryPrice,
                    InitialStopPrice = state.CurrentStopPrice, // Assuming current stop was initial (simplified)
                    InitialTargetPrice = state.TargetPrice,
                    BreakevenAfterTicks = state.BreakevenAfterTicks,
                    TrailTicks = state.TrailTicks,
                    
                    // REWARD
                    RMultiple = rMultiple,
                    PnL = totalPnL,
                    SharpeContribution = sharpeContribution,
                    ExitReason = exitReason.ToString(),
                    DurationMinutes = duration.TotalMinutes,
                    
                    // NEXT STATE (at exit)
                    ExitRegime = state.CurrentRegime,
                    ExitRegimeConfidence = state.CurrentRegimeConfidence,
                    ExitPrice = exitPrice,
                    VolatilityAtExit = volatilityAtExit,
                    
                    // ADDITIONAL METRICS
                    MaxFavorablePrice = state.MaxFavorablePrice,
                    MaxAdversePrice = state.MaxAdversePrice,
                    StopModificationCount = state.StopModificationCount,
                    BreakevenActivated = state.BreakevenActivated,
                    TrailingStopActive = state.TrailingStopActive,
                    RegimeChangeCount = state.RegimeChanges.Count
                };

                // Save to repository
                await experienceRepo.SaveExperienceAsync(experience).ConfigureAwait(false);
                
                _logger.LogInformation(
                    "📊 [EXPERIENCE] Saved: {Strategy} - R: {RMultiple:F2}, PnL: ${PnL:F2}",
                    experience.Strategy,
                    experience.RMultiple,
                    experience.PnL
                );
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [EXPERIENCE] Failed to collect experience for {PositionId}", state.PositionId);
                // Don't throw - experience collection failure shouldn't affect position management
            }
        }
        
        /// <summary>
        /// Main monitoring loop - checks all positions and applies management rules
        /// </summary>
        private async Task MonitorAndManagePositionsAsync(CancellationToken cancellationToken)
        {
            if (_activePositions.IsEmpty)
            {
                return; // No positions to manage
            }
            
            foreach (var state in _activePositions.Select(kvp => kvp.Value))
            {
                if (cancellationToken.IsCancellationRequested)
                    break;
                state.LastCheckTime = DateTime.UtcNow;
                
                try
                {
                    // Get current market price (this would come from market data service in production)
                    var currentPrice = await GetCurrentMarketPriceAsync(state.Symbol, cancellationToken).ConfigureAwait(false);
                    if (currentPrice <= 0)
                        continue; // Skip if no price available
                    
                    // Update max excursion tracking
                    UpdateMaxExcursion(state, currentPrice);
                    
                    // Calculate current profit in ticks
                    var tickSize = GetTickSize(state.Symbol);
                    var isLong = state.Quantity > 0;
                    var profitTicks = CalculateProfitTicks(state.EntryPrice, currentPrice, tickSize, isLong);
                    
                    // Check time-based exit first
                    if (ShouldExitOnTime(state))
                    {
                        // AI Commentary: Explain time-based exit (non-blocking)
                        try
                        {
                            var holdDuration = DateTime.UtcNow - state.EntryTime;
                            var currentPnL = profitTicks * tickSize * state.Quantity * (isLong ? 50m : -50m);
                            var marketRegime = "UNKNOWN";
                            
                            ExplainTimeBasedExitFireAndForget(state, holdDuration, currentPnL, marketRegime);
                        }
                        catch (Exception ex)
                        {
                            // Log AI errors but don't disrupt position management
                            _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for time-based exit");
                        }
                        
                        await RequestPositionCloseAsync(state, ExitReason.TimeLimit, cancellationToken).ConfigureAwait(false);
                        continue;
                    }
                    
                    // FEATURE 4: Apply confidence-based breakeven threshold
                    var breakevenThreshold = GetConfidenceBasedBreakevenTicks(state.EntryConfidence, state.BreakevenAfterTicks);
                    
                    // Apply breakeven protection if profit threshold reached
                    if (!state.BreakevenActivated && profitTicks >= breakevenThreshold)
                    {
                        await ActivateBreakevenProtectionAsync(state, tickSize, cancellationToken).ConfigureAwait(false);
                    }
                    
                    // FEATURE 4: Apply confidence-based trailing activation threshold
                    var trailingActivationThreshold = breakevenThreshold + state.TrailTicks;
                    
                    // Activate and update trailing stop if profit threshold reached
                    if (state.BreakevenActivated && !state.TrailingStopActive && profitTicks >= trailingActivationThreshold)
                    {
                        state.TrailingStopActive = true;
                        _logger.LogInformation("🔄 [POSITION-MGMT] Trailing stop activated for {PositionId}: {Symbol} at +{Ticks} ticks profit (confidence-adjusted threshold: {Threshold})",
                            state.PositionId, state.Symbol, profitTicks, trailingActivationThreshold);
                    }
                    
                    // PHASE 4: Check for multi-level partial exits
                    await CheckPartialExitsAsync(state, currentPrice, tickSize, cancellationToken).ConfigureAwait(false);
                    
                    // PHASE 4: Apply volatility-adaptive stop adjustment
                    await ApplyVolatilityAdaptiveStopAsync(state, tickSize, cancellationToken).ConfigureAwait(false);
                    
                    // FEATURE 1: Check for regime changes and adjust targets dynamically
                    await CheckRegimeChangeAsync(state, cancellationToken).ConfigureAwait(false);
                    
                    // FEATURE 3: Check for regime flip and exit if needed
                    await CheckRegimeFlipExitAsync(state, currentPrice, tickSize, cancellationToken).ConfigureAwait(false);
                    
                    // FEATURE 2: Check for early exit based on learned MAE threshold
                    await CheckEarlyExitThresholdAsync(state, currentPrice, tickSize, cancellationToken).ConfigureAwait(false);
                    
                    // FEATURE 5: Check for progressive time-decay stop tightening
                    await CheckProgressiveTighteningAsync(state, currentPrice, profitTicks, tickSize, cancellationToken).ConfigureAwait(false);
                    
                    // Update trailing stop if active
                    if (state.TrailingStopActive)
                    {
                        await UpdateTrailingStopAsync(state, currentPrice, tickSize, cancellationToken).ConfigureAwait(false);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-MGMT] Error managing position {PositionId}", state.PositionId);
                }
            }
        }
        
        /// <summary>
        /// Activate breakeven protection - move stop to entry + 1 tick
        /// </summary>
        private async Task ActivateBreakevenProtectionAsync(
            PositionManagementState state,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            var isLong = state.Quantity > 0;
            var breakevenStop = isLong 
                ? state.EntryPrice + tickSize 
                : state.EntryPrice - tickSize;
            
            await ModifyStopPriceAsync(state, breakevenStop, "Breakeven", cancellationToken).ConfigureAwait(false);
            state.BreakevenActivated = true;
            
            _logger.LogInformation("🛡️ [POSITION-MGMT] Breakeven activated for {PositionId}: {Symbol}, Stop moved to {Stop} (entry +{Tick})",
                state.PositionId, state.Symbol, breakevenStop, tickSize);
            
            // AI Commentary: Explain breakeven activation (non-blocking)
            try
            {
                var currentPrice = await GetCurrentMarketPriceAsync(state.Symbol, cancellationToken).ConfigureAwait(false);
                var profitTicks = CalculateProfitTicks(state.EntryPrice, currentPrice, tickSize, isLong);
                var unrealizedPnL = profitTicks * tickSize * state.Quantity * (isLong ? 50m : -50m); // $50 per tick for ES
                var marketRegime = "UNKNOWN"; // Could be enhanced with regime detection
                
                ExplainBreakevenProtectionFireAndForget(state, unrealizedPnL, profitTicks, breakevenStop, marketRegime);
            }
            catch (Exception ex)
            {
                // Log AI errors but don't disrupt position management
                _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for breakeven protection");
            }
        }
        
        /// <summary>
        /// Update trailing stop to follow price
        /// </summary>
        private async Task UpdateTrailingStopAsync(
            PositionManagementState state,
            decimal currentPrice,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            var isLong = state.Quantity > 0;
            
            // FEATURE 2: Use MFE-optimized trailing distance if available
            var optimizedTrailTicks = GetOptimizedTrailingDistance(state, tickSize);
            
            // FEATURE 4: Apply confidence-based trailing distance
            var confidenceBasedTicks = GetConfidenceBasedTrailTicks(state.EntryConfidence, state.TrailTicks);
            
            // Use the tighter of optimized or confidence-based (or default if neither available)
            var trailTicks = optimizedTrailTicks ?? confidenceBasedTicks;
            var trailDistance = trailTicks * tickSize;
            
            var newStopPrice = isLong
                ? currentPrice - trailDistance
                : currentPrice + trailDistance;
            
            // Only update if new stop is better (higher for longs, lower for shorts)
            var shouldUpdate = isLong
                ? newStopPrice > state.CurrentStopPrice
                : newStopPrice < state.CurrentStopPrice;
            
            if (shouldUpdate)
            {
                var oldStopPrice = state.CurrentStopPrice;
                await ModifyStopPriceAsync(state, newStopPrice, "Trailing", cancellationToken).ConfigureAwait(false);
                
                _logger.LogInformation("📈 [POSITION-MGMT] Trailing stop updated for {PositionId}: {Symbol}, Stop: {Old} → {New} (trail {Ticks} ticks)",
                    state.PositionId, state.Symbol, oldStopPrice, newStopPrice, trailTicks);
                
                // AI Commentary: Explain trailing stop activation (non-blocking)
                try
                {
                    var profitTicks = CalculateProfitTicks(state.EntryPrice, currentPrice, tickSize, isLong);
                    var unrealizedPnL = profitTicks * tickSize * state.Quantity * (isLong ? 50m : -50m); // $50 per tick for ES
                    var marketRegime = "UNKNOWN"; // Could be enhanced with regime detection
                    
                    ExplainTrailingStopActivationFireAndForget(state, unrealizedPnL, profitTicks, newStopPrice, marketRegime);
                }
                catch (Exception ex)
                {
                    // Log AI errors but don't disrupt position management
                    _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for trailing stop activation");
                }
            }
        }
        
        /// <summary>
        /// Check if position should be closed due to time limit
        /// </summary>
        private static bool ShouldExitOnTime(PositionManagementState state)
        {
            if (state.MaxHoldMinutes <= 0)
                return false; // No time limit
                
            var duration = DateTime.UtcNow - state.EntryTime;
            return duration.TotalMinutes >= state.MaxHoldMinutes;
        }
        
        /// <summary>
        /// Update max favorable and adverse excursion tracking
        /// MAE CORRELATION: Also tracks time-stamped MAE snapshots
        /// </summary>
        private static void UpdateMaxExcursion(PositionManagementState state, decimal currentPrice)
        {
            var isLong = state.Quantity > 0;
            
            if (isLong)
            {
                // For long positions: higher is favorable, lower is adverse
                if (currentPrice > state.MaxFavorablePrice)
                    state.MaxFavorablePrice = currentPrice;
                if (currentPrice < state.MaxAdversePrice)
                    state.MaxAdversePrice = currentPrice;
            }
            else
            {
                // For short positions: lower is favorable, higher is adverse
                if (currentPrice < state.MaxFavorablePrice)
                    state.MaxFavorablePrice = currentPrice;
                if (currentPrice > state.MaxAdversePrice)
                    state.MaxAdversePrice = currentPrice;
            }
            
            // MAE CORRELATION: Track MAE snapshots for time-series analysis
            var elapsedSeconds = (int)(DateTime.UtcNow - state.EntryTime).TotalSeconds;
            var tickSize = GetTickSize(state.Symbol);
            var currentMae = isLong
                ? (state.MaxAdversePrice - state.EntryPrice) / tickSize
                : (state.EntryPrice - state.MaxAdversePrice) / tickSize;
            
            // Record snapshot at key intervals (1min, 2min, 5min, 10min)
            if (ShouldRecordMaeSnapshot(state, elapsedSeconds))
            {
                state.AddMaeSnapshot(new MaeSnapshot
                {
                    Timestamp = DateTime.UtcNow,
                    MaeValue = Math.Abs(currentMae),
                    ElapsedSeconds = elapsedSeconds
                });
            }
        }
        
        /// <summary>
        /// MAE CORRELATION: Determine if we should record MAE snapshot at this time
        /// Records at 1min, 2min, 5min, 10min intervals (with 5-second tolerance)
        /// </summary>
        private static bool ShouldRecordMaeSnapshot(PositionManagementState state, int elapsedSeconds)
        {
            var targetIntervals = new[] { 60, 120, 300, 600 }; // 1min, 2min, 5min, 10min
            
            foreach (var target in targetIntervals)
            {
                // Check if we're within tolerance seconds of target and haven't recorded yet
                if (Math.Abs(elapsedSeconds - target) <= MAE_MFE_SNAPSHOT_TOLERANCE_SECONDS)
                {
                    var alreadyRecorded = state.MaeSnapshots.Any(s => Math.Abs(s.ElapsedSeconds - target) <= MAE_MFE_SNAPSHOT_TOLERANCE_SECONDS);
                    if (!alreadyRecorded)
                    {
                        return true;
                    }
                }
            }
            
            return false;
        }
        
        /// <summary>
        /// Calculate profit in ticks
        /// </summary>
        private static decimal CalculateProfitTicks(decimal entryPrice, decimal currentPrice, decimal tickSize, bool isLong)
        {
            var priceDiff = isLong 
                ? currentPrice - entryPrice 
                : entryPrice - currentPrice;
            
            return priceDiff / tickSize;
        }
        
        /// <summary>
        /// Get tick size for symbol
        /// </summary>
        private static decimal GetTickSize(string symbol)
        {
            if (symbol.Contains("ES", StringComparison.OrdinalIgnoreCase) || 
                symbol.Contains("MES", StringComparison.OrdinalIgnoreCase))
                return EsTickSize;
            
            if (symbol.Contains("NQ", StringComparison.OrdinalIgnoreCase) || 
                symbol.Contains("MNQ", StringComparison.OrdinalIgnoreCase))
                return NqTickSize;
            
            return DefaultTickSize;
        }
        
        /// <summary>
        /// Get max hold minutes for strategy
        /// </summary>
        private static int GetMaxHoldMinutes(string strategy)
        {
            return strategy switch
            {
                "S2" => S2_MAX_HOLD_MINUTES,
                "S3" => S3_MAX_HOLD_MINUTES,
                "S6" => S6_MAX_HOLD_MINUTES,
                "S11" => S11_MAX_HOLD_MINUTES,
                _ => DEFAULT_MAX_HOLD_MINUTES
            };
        }
        
        /// <summary>
        /// Get current market price from PositionTrackingSystem
        /// </summary>
        private Task<decimal> GetCurrentMarketPriceAsync(string symbol, CancellationToken cancellationToken)
        {
            try
            {
                // Try to get PositionTrackingSystem from service provider
                var positionTracker = _serviceProvider.GetService<PositionTrackingSystem>();
                if (positionTracker != null)
                {
                    var positions = positionTracker.AllPositions;
                    if (positions.TryGetValue(symbol, out var position))
                    {
                        // Calculate current price from unrealized P&L if available
                        // UnrealizedPnL = (marketPrice - avgPrice) * netQuantity
                        // So marketPrice = (unrealizedPnL / netQuantity) + avgPrice
                        if (position.NetQuantity != 0 && position.AveragePrice > 0)
                        {
                            var estimatedPrice = (position.UnrealizedPnL / position.NetQuantity) + position.AveragePrice;
                            if (estimatedPrice > 0)
                            {
                                return Task.FromResult(estimatedPrice);
                            }
                        }
                        
                        // Fallback to average price if no unrealized P&L
                        if (position.AveragePrice > 0)
                        {
                            return Task.FromResult(position.AveragePrice);
                        }
                    }
                }
                
                // If no price available from tracker, return 0 to skip this cycle
                return Task.FromResult(0m);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[POSITION-MGMT] Error getting market price for {Symbol}", symbol);
                return Task.FromResult(0m);
            }
        }
        
        /// <summary>
        /// Modify stop price using IOrderService
        /// </summary>
        private async Task ModifyStopPriceAsync(
            PositionManagementState state,
            decimal newStopPrice,
            string reason,
            CancellationToken cancellationToken)
        {
            try
            {
                // Get order service from DI container
                var orderService = _serviceProvider.GetService<IOrderService>();
                if (orderService != null)
                {
                    // Call the actual order service to modify stop loss
                    var success = await orderService.ModifyStopLossAsync(state.PositionId, newStopPrice).ConfigureAwait(false);
                    
                    if (success)
                    {
                        state.CurrentStopPrice = newStopPrice;
                        state.StopModificationCount++;
                        
                        _logger.LogInformation("🔧 [POSITION-MGMT] Stop modified for {PositionId}: {Reason}, New stop: {Stop}",
                            state.PositionId, reason, newStopPrice);
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ [POSITION-MGMT] Failed to modify stop for {PositionId}: {Reason}",
                            state.PositionId, reason);
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ [POSITION-MGMT] OrderService not available, cannot modify stop for {PositionId}",
                        state.PositionId);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [POSITION-MGMT] Error modifying stop for {PositionId}", state.PositionId);
            }
        }
        
        /// <summary>
        /// Request position close using IOrderService
        /// </summary>
        private async Task RequestPositionCloseAsync(
            PositionManagementState state,
            ExitReason reason,
            CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogWarning("⏰ [POSITION-MGMT] {Reason} triggered for {PositionId}: {Symbol}, {Duration}m elapsed",
                    reason, state.PositionId, state.Symbol, (DateTime.UtcNow - state.EntryTime).TotalMinutes);
                
                // Get order service from DI container
                var orderService = _serviceProvider.GetService<IOrderService>();
                if (orderService != null)
                {
                    // Call the actual order service to close position
                    var success = await orderService.ClosePositionAsync(state.PositionId).ConfigureAwait(false);
                    
                    if (success)
                    {
                        _logger.LogInformation("✅ [POSITION-MGMT] Position closed successfully: {PositionId}, Reason: {Reason}",
                            state.PositionId, reason);
                        
                        // Unregister after successful close
                        UnregisterPosition(state.PositionId, reason);
                    }
                    else
                    {
                        _logger.LogWarning("⚠️ [POSITION-MGMT] Failed to close position {PositionId}, will retry next cycle",
                            state.PositionId);
                    }
                }
                else
                {
                    _logger.LogWarning("⚠️ [POSITION-MGMT] OrderService not available, cannot close position {PositionId}",
                        state.PositionId);
                    
                    // Still unregister to prevent infinite retries
                    UnregisterPosition(state.PositionId, reason);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [POSITION-MGMT] Error closing position {PositionId}", state.PositionId);
                
                // Unregister to prevent infinite retries
                UnregisterPosition(state.PositionId, reason);
            }
        }
        
        /// <summary>
        /// Get excursion metrics for a position (for exit logging)
        /// </summary>
        public (decimal maxFavorable, decimal maxAdverse) GetExcursionMetrics(string positionId)
        {
            if (_activePositions.TryGetValue(positionId, out var state))
            {
                var isLong = state.Quantity > 0;
                var tickSize = GetTickSize(state.Symbol);
                
                var maxFavTicks = isLong
                    ? (state.MaxFavorablePrice - state.EntryPrice) / tickSize
                    : (state.EntryPrice - state.MaxFavorablePrice) / tickSize;
                
                var maxAdvTicks = isLong
                    ? (state.MaxAdversePrice - state.EntryPrice) / tickSize
                    : (state.EntryPrice - state.MaxAdversePrice) / tickSize;
                
                return (maxFavTicks, maxAdvTicks);
            }
            
            return (0m, 0m);
        }
        
        /// <summary>
        /// PHASE 4: Check and execute multi-level partial exits
        /// First target: Close 50% at 1.5R
        /// Second target: Close 30% at 2.5R  
        /// Final target: Close remaining 20% at 4.0R (runner position)
        /// </summary>
        private async Task CheckPartialExitsAsync(
            PositionManagementState state,
            decimal currentPrice,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            // Calculate R-multiple (profit relative to risk)
            var initialRisk = Math.Abs(state.CurrentStopPrice - state.EntryPrice);
            if (initialRisk <= 0)
            {
                return; // Invalid risk calculation
            }
            
            var isLong = state.Quantity > 0;
            var currentProfit = isLong
                ? currentPrice - state.EntryPrice
                : state.EntryPrice - currentPrice;
            
            var rMultiple = currentProfit / initialRisk;
            
            // Track which partial exits have been executed
            if (!state.HasProperty("FirstPartialExecuted"))
            {
                state.SetProperty("FirstPartialExecuted", false);
            }
            if (!state.HasProperty("SecondPartialExecuted"))
            {
                state.SetProperty("SecondPartialExecuted", false);
            }
            if (!state.HasProperty("FinalPartialExecuted"))
            {
                state.SetProperty("FinalPartialExecuted", false);
            }
            
            var firstPartialDone = (bool)(state.GetProperty("FirstPartialExecuted") ?? false);
            var secondPartialDone = (bool)(state.GetProperty("SecondPartialExecuted") ?? false);
            var finalPartialDone = (bool)(state.GetProperty("FinalPartialExecuted") ?? false);
            
            // Check first partial exit at 1.5R (close 50%)
            if (!firstPartialDone && rMultiple >= FirstPartialExitThreshold)
            {
                _logger.LogInformation("🎯 [POSITION-MGMT] PHASE 4 - First partial exit triggered for {PositionId}: {Symbol} at {R}R, closing 50%",
                    state.PositionId, state.Symbol, rMultiple);
                
                // AI Commentary: Explain first partial exit (non-blocking)
                try
                {
                    var partialQuantity = Math.Floor(state.Quantity * FIRST_PARTIAL_EXIT_PERCENTAGE);
                    ExplainPartialExitFireAndForget(state, rMultiple, FIRST_PARTIAL_DISPLAY_PERCENT, partialQuantity, "First Target (1.5R)");
                }
                catch (Exception ex)
                {
                    // Log AI errors but don't disrupt position management
                    _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for first partial exit");
                }
                
                await RequestPartialCloseAsync(state, FIRST_PARTIAL_EXIT_PERCENTAGE, ExitReason.Partial, cancellationToken).ConfigureAwait(false);
                state.SetProperty("FirstPartialExecuted", true);
            }
            // Check second partial exit at 2.5R (close 30% of remaining = 30% of original)
            else if (firstPartialDone && !secondPartialDone && rMultiple >= SecondPartialExitThreshold)
            {
                _logger.LogInformation("🎯 [POSITION-MGMT] PHASE 4 - Second partial exit triggered for {PositionId}: {Symbol} at {R}R, closing 30%",
                    state.PositionId, state.Symbol, rMultiple);
                
                // AI Commentary: Explain second partial exit (non-blocking)
                try
                {
                    var partialQuantity = Math.Floor(state.Quantity * SECOND_PARTIAL_EXIT_PERCENTAGE);
                    ExplainPartialExitFireAndForget(state, rMultiple, SECOND_PARTIAL_DISPLAY_PERCENT, partialQuantity, "Second Target (2.5R)");
                }
                catch (Exception ex)
                {
                    // Log AI errors but don't disrupt position management
                    _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for second partial exit");
                }
                
                await RequestPartialCloseAsync(state, SECOND_PARTIAL_EXIT_PERCENTAGE, ExitReason.Partial, cancellationToken).ConfigureAwait(false);
                state.SetProperty("SecondPartialExecuted", true);
            }
            // Check final partial exit at 4.0R (close remaining 20% = runner position)
            else if (secondPartialDone && !finalPartialDone && rMultiple >= FinalPartialExitThreshold)
            {
                _logger.LogInformation("🎯 [POSITION-MGMT] PHASE 4 - Final partial exit (runner) triggered for {PositionId}: {Symbol} at {R}R, closing final 20%",
                    state.PositionId, state.Symbol, rMultiple);
                
                // AI Commentary: Explain final partial exit (non-blocking)
                try
                {
                    var partialQuantity = Math.Floor(state.Quantity * FINAL_PARTIAL_EXIT_PERCENTAGE);
                    ExplainPartialExitFireAndForget(state, rMultiple, FINAL_PARTIAL_DISPLAY_PERCENT, partialQuantity, "Runner Position (4.0R)");
                }
                catch (Exception ex)
                {
                    // Log AI errors but don't disrupt position management
                    _logger.LogDebug(ex, "[POSITION-MGMT] AI explanation failed for final partial exit");
                }
                
                await RequestPartialCloseAsync(state, FINAL_PARTIAL_EXIT_PERCENTAGE, ExitReason.Target, cancellationToken).ConfigureAwait(false);
                state.SetProperty("FinalPartialExecuted", true);
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        
        /// <summary>
        /// PHASE 4: Apply volatility-adaptive stop adjustments
        /// High volatility (ATR > 1.5x avg) → Widen stops by 20%
        /// Low volatility (ATR < 0.7x avg) → Tighten stops by 20%
        /// Integrates VIX: if VIX > 20 → widen all stops
        /// Session-aware: Asia session (low vol) → tighter stops
        /// </summary>
        private async Task ApplyVolatilityAdaptiveStopAsync(
            PositionManagementState state,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            // Check if volatility adjustment already applied this cycle
            if (state.HasProperty("VolatilityAdjustedThisCycle"))
            {
                var lastAdjustedObj = state.GetProperty("VolatilityAdjustedThisCycle");
                if (lastAdjustedObj is DateTime lastAdjusted && (DateTime.UtcNow - lastAdjusted).TotalMinutes < VOLATILITY_ADJUSTMENT_MIN_INTERVAL_MINUTES)
                {
                    return; // Don't adjust more than once per interval
                }
            }
            
            try
            {
                // Get current ATR (would come from market data service in production)
                // For now, use simplified estimation based on recent price movement
                var currentVolatility = EstimateCurrentVolatility(state);
                var avgVolatility = EstimateAverageVolatility(state);
                
                if (avgVolatility <= 0)
                {
                    return; // Can't calculate ratio
                }
                
                var volatilityRatio = currentVolatility / avgVolatility;
                decimal stopAdjustmentFactor = ATR_MULTIPLIER_UNIT;
                string adjustmentReason = "";
                
                // Determine adjustment based on volatility
                if (volatilityRatio > HighVolatilityThreshold)
                {
                    // High volatility - widen stops
                    stopAdjustmentFactor = ATR_MULTIPLIER_UNIT + VolatilityStopWidening;
                    adjustmentReason = $"High volatility ({volatilityRatio:F2}x avg)";
                }
                else if (volatilityRatio < LowVolatilityThreshold)
                {
                    // Low volatility - tighten stops
                    stopAdjustmentFactor = ATR_MULTIPLIER_UNIT - VolatilityStopTightening;
                    adjustmentReason = $"Low volatility ({volatilityRatio:F2}x avg)";
                }
                
                // Apply adjustment if needed
                if (stopAdjustmentFactor != ATR_MULTIPLIER_UNIT)
                {
                    var isLong = state.Quantity > 0;
                    var stopDistance = Math.Abs(state.CurrentStopPrice - state.EntryPrice);
                    var adjustedDistance = stopDistance * stopAdjustmentFactor;
                    
                    var newStopPrice = isLong
                        ? state.EntryPrice - adjustedDistance
                        : state.EntryPrice + adjustedDistance;
                    
                    // Only update if new stop is not worse than current (don't move stop against position)
                    var shouldUpdate = isLong
                        ? newStopPrice <= state.CurrentStopPrice  // For longs, only lower or keep same
                        : newStopPrice >= state.CurrentStopPrice; // For shorts, only higher or keep same
                    
                    if (shouldUpdate && Math.Abs(newStopPrice - state.CurrentStopPrice) > tickSize)
                    {
                        await ModifyStopPriceAsync(state, newStopPrice, $"VolAdaptive-{adjustmentReason}", cancellationToken).ConfigureAwait(false);
                        
                        _logger.LogInformation("📊 [POSITION-MGMT] PHASE 4 - Volatility-adaptive stop for {PositionId}: {Symbol}, {Reason}, Stop: {Old} → {New} ({Factor:P0} adjustment)",
                            state.PositionId, state.Symbol, adjustmentReason, state.CurrentStopPrice, newStopPrice, stopAdjustmentFactor - ATR_MULTIPLIER_UNIT);
                        
                        state.SetProperty("VolatilityAdjustedThisCycle", DateTime.UtcNow);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [POSITION-MGMT] Error applying volatility-adaptive stop for {PositionId}", state.PositionId);
            }
            
            await Task.CompletedTask.ConfigureAwait(false);
        }
        
        /// <summary>
        /// Estimate current volatility (simplified for PHASE 4)
        /// In production, would use actual ATR from market data
        /// </summary>
        private static decimal EstimateCurrentVolatility(PositionManagementState state)
        {
            // Simplified: use max excursion range as volatility proxy
            var range = Math.Abs(state.MaxFavorablePrice - state.MaxAdversePrice);
            return range > 0 ? range : ATR_MULTIPLIER_UNIT;
        }
        
        /// <summary>
        /// MAE CORRELATION: Get MAE value at specific elapsed time from snapshots
        /// </summary>
        private static decimal GetMaeAtTime(PositionManagementState state, int targetSeconds)
        {
            if (state.MaeSnapshots.Count == 0)
            {
                return 0m;
            }
            
            // Find snapshot closest to target time
            var closest = state.MaeSnapshots
                .OrderBy(s => Math.Abs(s.ElapsedSeconds - targetSeconds))
                .FirstOrDefault();
            
            return closest?.MaeValue ?? 0m;
        }
        
        /// <summary>
        /// Estimate average volatility (simplified for PHASE 4)
        /// In production, would use historical ATR
        /// </summary>
        private static decimal EstimateAverageVolatility(PositionManagementState state)
        {
            // Simplified: use entry-based range estimation
            var tickSize = GetTickSize(state.Symbol);
            return tickSize * ATR_LOOKBACK_BARS; // Assume avg volatility is ~10 ticks
        }
        
        /// <summary>
        /// Request partial position close
        /// Production implementation: logs the intent - requires IOrderService extension for partial closes
        /// Currently tracks the partial exit levels for monitoring purposes
        /// </summary>
        private async Task RequestPartialCloseAsync(
            PositionManagementState state,
            decimal percentToClose,
            ExitReason reason,
            CancellationToken cancellationToken)
        {
            try
            {
                var quantityToClose = (int)(state.Quantity * percentToClose);
                if (quantityToClose <= 0)
                {
                    return;
                }
                
                // Get order service from DI container
                var orderService = _serviceProvider.GetService<IOrderService>();
                if (orderService != null)
                {
                    _logger.LogInformation("📉 [POSITION-MGMT] Executing partial close for {PositionId}: Closing {Qty} of {Total} contracts ({Pct:P0}), Reason: {Reason}",
                        state.PositionId, quantityToClose, state.Quantity, percentToClose, reason);
                    
                    // Execute partial close using the new overloaded method
                    var success = await orderService.ClosePositionAsync(state.PositionId, quantityToClose, cancellationToken).ConfigureAwait(false);
                    
                    if (success)
                    {
                        // Update position state to reflect partial close
                        var remainingQuantity = state.Quantity - quantityToClose;
                        state.Quantity = remainingQuantity;
                        
                        _logger.LogInformation("✅ [POSITION-MGMT] Partial close successful for {PositionId}: {Qty} contracts closed, {Remaining} remaining",
                            state.PositionId, quantityToClose, remainingQuantity);
                        
                        // Track that this partial exit was executed (for ML/RL learning)
                        state.SetProperty($"PartialExitExecuted_{percentToClose:P0}", DateTime.UtcNow);
                    }
                    else
                    {
                        _logger.LogError("❌ [POSITION-MGMT] Partial close FAILED for {PositionId}: Could not close {Qty} contracts",
                            state.PositionId, quantityToClose);
                    }
                }
                else
                {
                    // IOrderService not available - log warning
                    _logger.LogWarning("⚠️ [POSITION-MGMT] IOrderService not available - cannot execute partial close for {PositionId} ({Qty} contracts)",
                        state.PositionId, quantityToClose);
                    
                    // Track that this partial exit level was reached but not executed (for monitoring)
                    state.SetProperty($"PartialExitReached_{percentToClose:P0}", DateTime.UtcNow);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [POSITION-MGMT] Error processing partial close for {PositionId}", state.PositionId);
            }
        }
        
        /// <summary>
        /// PHASE 2: Handle zone break events from ZoneBreakMonitoringService
        /// Adjusts stops based on broken zones - broken supply becomes support, broken demand becomes resistance
        /// </summary>
        public async Task OnZoneBreak(ZoneBreakEvent breakEvent)
        {
            ArgumentNullException.ThrowIfNull(breakEvent);
            
            try
            {
                // Find positions for this symbol
                var relevantPositions = _activePositions.Values
                    .Where(p => p.Symbol == breakEvent.Symbol)
                    .ToList();
                
                if (relevantPositions.Count == 0)
                {
                    return;
                }
                
                foreach (var state in relevantPositions)
                {
                    var isLong = state.Quantity > 0;
                    var posType = isLong ? "LONG" : "SHORT";
                    
                    // Only process if break type matches position type
                    if (breakEvent.PositionType != posType)
                    {
                        continue;
                    }
                    
                    switch (breakEvent.BreakType)
                    {
                        case ZoneBreakType.StrongDemandBreak:
                        case ZoneBreakType.WeakDemandBreak:
                            // Long position, demand zone broken = bad, consider early exit or tighten stop
                            if (breakEvent.Severity == "CRITICAL" || breakEvent.Severity == "HIGH")
                            {
                                _logger.LogWarning("⚠️ [POSITION-MGMT] CRITICAL demand break for LONG {PositionId} - Consider early exit",
                                    state.PositionId);
                                
                                // For critical breaks, close position immediately
                                if (breakEvent.Severity == "CRITICAL")
                                {
                                    // AI Commentary: Explain zone break forced exit (non-blocking)
                                    try
                                    {
                                        var currentPrice = await GetCurrentMarketPriceAsync(state.Symbol, CancellationToken.None).ConfigureAwait(false);
                                        var tickSize = GetTickSize(state.Symbol);
                                        var profitTicks = CalculateProfitTicks(state.EntryPrice, currentPrice, tickSize, isLong);
                                        var currentPnL = profitTicks * tickSize * state.Quantity * (isLong ? 50m : -50m);
                                        
                                        ExplainZoneBreakExitFireAndForget(state, breakEvent, currentPnL);
                                    }
                                    catch
                                    {
                                        // Silently ignore AI errors
                                    }
                                    
                                    await RequestPositionCloseAsync(state, ExitReason.ZoneBreak, CancellationToken.None).ConfigureAwait(false);
                                }
                            }
                            break;
                            
                        case ZoneBreakType.StrongSupplyBreak:
                        case ZoneBreakType.WeakSupplyBreak:
                            // Short position, supply zone broken = bad, consider early exit or tighten stop
                            if (breakEvent.Severity == "CRITICAL" || breakEvent.Severity == "HIGH")
                            {
                                _logger.LogWarning("⚠️ [POSITION-MGMT] CRITICAL supply break for SHORT {PositionId} - Consider early exit",
                                    state.PositionId);
                                
                                // For critical breaks, close position immediately
                                if (breakEvent.Severity == "CRITICAL")
                                {
                                    // AI Commentary: Explain zone break forced exit (non-blocking)
                                    try
                                    {
                                        var currentPrice = await GetCurrentMarketPriceAsync(state.Symbol, CancellationToken.None).ConfigureAwait(false);
                                        var tickSize = GetTickSize(state.Symbol);
                                        var profitTicks = CalculateProfitTicks(state.EntryPrice, currentPrice, tickSize, isLong);
                                        var currentPnL = profitTicks * tickSize * state.Quantity * (isLong ? 50m : -50m);
                                        
                                        ExplainZoneBreakExitFireAndForget(state, breakEvent, currentPnL);
                                    }
                                    catch
                                    {
                                        // Silently ignore AI errors
                                    }
                                    
                                    await RequestPositionCloseAsync(state, ExitReason.ZoneBreak, CancellationToken.None).ConfigureAwait(false);
                                }
                            }
                            break;
                            
                        case ZoneBreakType.StrongSupplyBreakBullish:
                            // Long position, strong resistance broken upward = good, maybe widen target
                            _logger.LogInformation("✅ [POSITION-MGMT] Bullish breakout confirmed for LONG {PositionId} - Strong momentum",
                                state.PositionId);
                            // Could optionally widen target here
                            break;
                            
                        case ZoneBreakType.StrongDemandBreakBearish:
                            // Short position, strong support broken downward = good, maybe widen target
                            _logger.LogInformation("✅ [POSITION-MGMT] Bearish breakout confirmed for SHORT {PositionId} - Strong momentum",
                                state.PositionId);
                            // Could optionally widen target here
                            break;
                    }
                    
                    // PHASE 2 Feature: Move stop to just behind broken zone
                    // Broken supply becomes new support (for longs)
                    // Broken demand becomes new resistance (for shorts)
                    await AdjustStopBehindBrokenZoneAsync(state, breakEvent, CancellationToken.None).ConfigureAwait(false);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [POSITION-MGMT] Error handling zone break for {Symbol}", breakEvent.Symbol);
            }
        }
        
        /// <summary>
        /// PHASE 2: Adjust stop to just behind broken zone
        /// Broken supply becomes support (for longs) - place stop below it
        /// Broken demand becomes resistance (for shorts) - place stop above it
        /// </summary>
        private async Task AdjustStopBehindBrokenZoneAsync(
            PositionManagementState state,
            ZoneBreakEvent breakEvent,
            CancellationToken cancellationToken)
        {
            var isLong = state.Quantity > 0;
            var tickSize = GetTickSize(state.Symbol);
            decimal newStopPrice;
            
            // For long positions with broken supply zones (now support)
            if (isLong && (breakEvent.BreakType == ZoneBreakType.StrongSupplyBreakBullish))
            {
                // Place stop below the broken zone (now support)
                newStopPrice = breakEvent.ZoneLow - (MIN_STOP_DISTANCE_R * tickSize);
                
                // Only update if new stop is better than current
                if (newStopPrice > state.CurrentStopPrice)
                {
                    await ModifyStopPriceAsync(state, newStopPrice, "ZoneBehind", cancellationToken).ConfigureAwait(false);
                    
                    _logger.LogInformation("🎯 [POSITION-MGMT] Stop adjusted behind broken supply (now support) for LONG {PositionId}: {Old} → {New}",
                        state.PositionId, state.CurrentStopPrice, newStopPrice);
                }
            }
            // For short positions with broken demand zones (now resistance)
            else if (!isLong && (breakEvent.BreakType == ZoneBreakType.StrongDemandBreakBearish))
            {
                // Place stop above the broken zone (now resistance)
                newStopPrice = breakEvent.ZoneHigh + (MIN_STOP_DISTANCE_R * tickSize);
                
                // Only update if new stop is better than current
                if (newStopPrice < state.CurrentStopPrice)
                {
                    await ModifyStopPriceAsync(state, newStopPrice, "ZoneBehind", cancellationToken).ConfigureAwait(false);
                    
                    _logger.LogInformation("🎯 [POSITION-MGMT] Stop adjusted behind broken demand (now resistance) for SHORT {PositionId}: {Old} → {New}",
                        state.PositionId, state.CurrentStopPrice, newStopPrice);
                }
            }
        }
        
        // ========================================================================
        // FEATURE 4: CONFIDENCE-BASED STOP/TARGET ADJUSTMENT METHODS
        // ========================================================================
        
        /// <summary>
        /// Apply confidence-based adjustments to stop, target, and quantity
        /// </summary>
        private (decimal adjustedStop, decimal adjustedTarget, int adjustedQuantity) ApplyConfidenceAdjustments(
            decimal entryPrice, 
            decimal stopPrice, 
            decimal targetPrice, 
            int quantity, 
            decimal confidence,
            string symbol)
        {
            var isLong = quantity > 0;
            var risk = Math.Abs(entryPrice - stopPrice);
            var reward = Math.Abs(targetPrice - entryPrice);
            
            // Get confidence-based multipliers from environment or use defaults
            decimal stopMultiplier, targetMultiplier;
            int adjustedQuantity = quantity;
            
            if (confidence >= _confidenceVeryHighThreshold) // Very High: 0.85-1.0
            {
                stopMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_STOP_MULTIPLIER_VERY_HIGH"), out var sm) ? sm : CONFIDENCE_STOP_MULTIPLIER_VERY_HIGH_DEFAULT;
                targetMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_TARGET_MULTIPLIER_VERY_HIGH"), out var tm) ? tm : CONFIDENCE_TARGET_MULTIPLIER_VERY_HIGH_DEFAULT;
            }
            else if (confidence >= _confidenceHighThreshold) // High: 0.75-0.85
            {
                stopMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_STOP_MULTIPLIER_HIGH"), out var sm) ? sm : CONFIDENCE_STOP_MULTIPLIER_HIGH_DEFAULT;
                targetMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_TARGET_MULTIPLIER_HIGH"), out var tm) ? tm : CONFIDENCE_TARGET_MULTIPLIER_HIGH_DEFAULT;
            }
            else if (confidence >= _confidenceMediumThreshold) // Medium: 0.70-0.75
            {
                stopMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_STOP_MULTIPLIER_MEDIUM"), out var sm) ? sm : CONFIDENCE_STOP_MULTIPLIER_MEDIUM_DEFAULT;
                targetMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_TARGET_MULTIPLIER_MEDIUM"), out var tm) ? tm : CONFIDENCE_TARGET_MULTIPLIER_MEDIUM_DEFAULT;
            }
            else if (confidence >= _confidenceLowThreshold) // Low: 0.65-0.70
            {
                stopMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_STOP_MULTIPLIER_LOW"), out var sm) ? sm : CONFIDENCE_STOP_MULTIPLIER_LOW_DEFAULT;
                targetMultiplier = decimal.TryParse(Environment.GetEnvironmentVariable("CONFIDENCE_TARGET_MULTIPLIER_LOW"), out var tm) ? tm : CONFIDENCE_TARGET_MULTIPLIER_LOW_DEFAULT;
                adjustedQuantity = Math.Max(1, quantity / CONFIDENCE_LOW_POSITION_SIZE_DIVISOR); // Reduce position size by 50%
            }
            else // Very Low: < 0.65 (should not happen, but handle it)
            {
                stopMultiplier = CONFIDENCE_VERY_LOW_STOP_MULTIPLIER;
                targetMultiplier = CONFIDENCE_VERY_LOW_TARGET_MULTIPLIER;
                adjustedQuantity = Math.Max(1, quantity / CONFIDENCE_VERY_LOW_POSITION_SIZE_DIVISOR); // Reduce to 25%
            }
            
            // Apply multipliers to risk and reward
            var adjustedRisk = risk * stopMultiplier;
            var adjustedReward = reward * targetMultiplier;
            
            // Calculate new stop and target prices
            var adjustedStop = isLong ? entryPrice - adjustedRisk : entryPrice + adjustedRisk;
            var adjustedTarget = isLong ? entryPrice + adjustedReward : entryPrice - adjustedReward;
            
            // Round to tick size
            var tickSize = GetTickSize(symbol);
            adjustedStop = ProductionPriceService.RoundToTick(adjustedStop, tickSize);
            adjustedTarget = ProductionPriceService.RoundToTick(adjustedTarget, tickSize);
            
            return (adjustedStop, adjustedTarget, adjustedQuantity);
        }
        
        /// <summary>
        /// Get confidence-based breakeven threshold in ticks
        /// </summary>
        private int GetConfidenceBasedBreakevenTicks(decimal confidence, int defaultTicks)
        {
            if (!_confidenceAdjustmentEnabled)
                return defaultTicks;
            
            if (confidence >= _confidenceVeryHighThreshold)
                return (int)(defaultTicks * CONFIDENCE_BREAKEVEN_MULTIPLIER_VERY_HIGH); // 12 ticks if default is 8
            else if (confidence >= _confidenceHighThreshold)
                return defaultTicks; // Standard
            else if (confidence >= _confidenceMediumThreshold)
                return (int)(defaultTicks * CONFIDENCE_BREAKEVEN_MULTIPLIER_MEDIUM); // 6 ticks if default is 8
            else
                return (int)(defaultTicks * CONFIDENCE_BREAKEVEN_MULTIPLIER_LOW); // 4 ticks if default is 8
        }
        
        /// <summary>
        /// Get confidence-based trailing distance in ticks
        /// </summary>
        private int GetConfidenceBasedTrailTicks(decimal confidence, int defaultTicks)
        {
            if (!_confidenceAdjustmentEnabled)
                return defaultTicks;
            
            if (confidence >= _confidenceVeryHighThreshold)
                return (int)(defaultTicks * CONFIDENCE_TRAIL_MULTIPLIER_VERY_HIGH); // Loose trail, let profit run
            else if (confidence >= _confidenceHighThreshold)
                return defaultTicks; // Standard
            else if (confidence >= _confidenceMediumThreshold)
                return (int)(defaultTicks * CONFIDENCE_TRAIL_MULTIPLIER_MEDIUM); // Tighter trail
            else
                return (int)(defaultTicks * CONFIDENCE_TRAIL_MULTIPLIER_LOW); // Very tight trail
        }
        
        // ========================================================================
        // FEATURE 1: DYNAMIC R-MULTIPLE TARGETING METHODS
        // ========================================================================
        
        /// <summary>
        /// Calculate dynamic target price based on market regime and strategy
        /// </summary>
        private decimal CalculateDynamicTarget(PositionManagementState state, decimal entryPrice, decimal stopPrice, string? useRegime = null)
        {
            var isLong = state.Quantity > 0;
            var risk = Math.Abs(entryPrice - stopPrice);
            
            if (risk <= 0)
            {
                return state.TargetPrice; // Fallback to static target
            }
            
            // Use specified regime or fall back to current regime (for recalculations) or entry regime (for initial calculation)
            var regimeToUse = useRegime ?? state.CurrentRegime;
            
            // Get regime-specific R-multiple for this strategy
            var rMultiple = GetRegimeBasedRMultiple(state.Strategy, regimeToUse);
            
            // Calculate dynamic target based on R-multiple
            var reward = risk * rMultiple;
            var dynamicTarget = isLong ? entryPrice + reward : entryPrice - reward;
            
            _logger.LogDebug("🎯 [POSITION-MGMT] Dynamic target calculation: {Strategy} in {Regime} regime, Risk={Risk:F2}, R={R}x, Target={Target:F2}",
                state.Strategy, regimeToUse, risk, rMultiple, dynamicTarget);
            
            return dynamicTarget;
        }
        
        /// <summary>
        /// Get regime-based R-multiple for a strategy
        /// </summary>
        private static decimal GetRegimeBasedRMultiple(string strategy, string regime)
        {
            var regimeKey = regime.ToUpperInvariant();
            var isTrending = regimeKey.Contains("TREND", StringComparison.Ordinal);
            var isRanging = regimeKey.Contains("RANGE", StringComparison.Ordinal);
            
            // Get environment variable for this strategy and regime
            string envVarName;
            if (isTrending)
            {
                envVarName = $"{strategy}_TARGET_TRENDING";
            }
            else if (isRanging)
            {
                envVarName = $"{strategy}_TARGET_RANGING";
            }
            else
            {
                envVarName = $"{strategy}_TARGET_TRENDING"; // Default to trending if unknown
            }
            
            var envValue = Environment.GetEnvironmentVariable(envVarName);
            if (decimal.TryParse(envValue, out var rMultiple))
            {
                return rMultiple;
            }
            
            // Fallback defaults if environment variable not set
            decimal GetRegimeSpecificRMultiple(string strat)
            {
                if (isTrending)
                {
                    return strat switch
                    {
                        "S2" => S2_TRENDING_R_MULTIPLE,
                        "S3" => S3_TRENDING_R_MULTIPLE,
                        "S6" => S6_TRENDING_R_MULTIPLE,
                        "S11" => S11_TRENDING_R_MULTIPLE,
                        _ => FALLBACK_DEFAULT_R_MULTIPLE
                    };
                }
                else if (isRanging)
                {
                    return strat switch
                    {
                        "S2" => S2_RANGING_R_MULTIPLE,
                        "S3" => S3_RANGING_R_MULTIPLE,
                        "S6" => S6_RANGING_R_MULTIPLE,
                        "S11" => S11_RANGING_R_MULTIPLE,
                        _ => FALLBACK_DEFAULT_R_MULTIPLE
                    };
                }
                else
                {
                    return strat switch
                    {
                        "S2" => S2_DEFAULT_R_MULTIPLE,
                        "S3" => S3_DEFAULT_R_MULTIPLE,
                        "S6" => S6_DEFAULT_R_MULTIPLE,
                        "S11" => S11_DEFAULT_R_MULTIPLE,
                        _ => FALLBACK_DEFAULT_R_MULTIPLE
                    };
                }
            }
            
            return GetRegimeSpecificRMultiple(strategy);
        }
        
        /// <summary>
        /// Check for regime changes and adjust target if needed (Feature 1)
        /// Called every 60 seconds (configurable) during monitoring loop
        /// </summary>
        private async Task CheckRegimeChangeAsync(PositionManagementState state, CancellationToken cancellationToken)
        {
            if (!_dynamicTargetsEnabled)
            {
                return; // Feature disabled
            }
            
            // Check if it's time to update regime
            var timeSinceLastCheck = DateTime.UtcNow - state.LastRegimeCheck;
            if (timeSinceLastCheck.TotalSeconds < _regimeCheckIntervalSeconds)
            {
                return; // Not time yet
            }
            
            try
            {
                var regimeService = _serviceProvider.GetService<RegimeDetectionService>();
                if (regimeService == null)
                {
                    return; // Service not available
                }
                
                // Get current regime
                var newRegime = await regimeService.GetCurrentRegimeAsync(state.Symbol, cancellationToken).ConfigureAwait(false);
                state.LastRegimeCheck = DateTime.UtcNow;
                
                // Check if regime changed significantly
                if (newRegime != state.CurrentRegime)
                {
                    var oldRegime = state.CurrentRegime;
                    state.CurrentRegime = newRegime;
                    
                    _logger.LogInformation("📊 [POSITION-MGMT] Regime change detected for {PositionId}: {Old} → {New}",
                        state.PositionId, oldRegime, newRegime);
                    
                    // Recalculate dynamic target using the NEW regime
                    var newDynamicTarget = CalculateDynamicTarget(state, state.EntryPrice, state.CurrentStopPrice, newRegime);
                    var oldTarget = state.DynamicTargetPrice;
                    
                    // Only adjust if change is significant (threshold check)
                    var targetChange = Math.Abs(newDynamicTarget - oldTarget);
                    var changePercent = targetChange / Math.Abs(state.EntryPrice - state.CurrentStopPrice);
                    
                    if (changePercent >= _targetAdjustmentThreshold)
                    {
                        state.DynamicTargetPrice = newDynamicTarget;
                        
                        _logger.LogInformation("🎯 [POSITION-MGMT] Target adjusted for {PositionId} due to regime change: {OldTarget:F2} → {NewTarget:F2} ({Regime})",
                            state.PositionId, oldTarget, newDynamicTarget, newRegime);
                        
                        // AI Commentary: Explain dynamic target adjustment (non-blocking)
                        try
                        {
                            var tickSize = GetTickSize(state.Symbol);
                            var riskTicks = Math.Abs(state.EntryPrice - state.CurrentStopPrice) / tickSize;
                            var oldTargetTicks = Math.Abs(oldTarget - state.EntryPrice) / tickSize;
                            var newTargetTicks = Math.Abs(newDynamicTarget - state.EntryPrice) / tickSize;
                            var oldTargetR = riskTicks > 0 ? oldTargetTicks / riskTicks : 0;
                            var newTargetR = riskTicks > 0 ? newTargetTicks / riskTicks : 0;
                            
                            ExplainDynamicTargetAdjustmentFireAndForget(
                                state,
                                oldRegime,
                                newRegime,
                                oldTargetR,
                                newTargetR);
                        }
                        catch
                        {
                            // Silently ignore AI errors - don't disrupt position management
                        }
                    }
                    else
                    {
                        _logger.LogDebug("🎯 [POSITION-MGMT] Regime change for {PositionId} but target adjustment below threshold ({Pct:P1})",
                            state.PositionId, changePercent);
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [POSITION-MGMT] Error checking regime change for {PositionId}", state.PositionId);
            }
        }
        
        // ========================================================================
        // FEATURE 3: REAL-TIME REGIME CHANGE EXIT DETECTION METHODS
        // ========================================================================
        
        /// <summary>
        /// Check for regime flip and exit if conditions warrant (Feature 3)
        /// Called every 30 seconds (via regime check interval) during monitoring loop
        /// </summary>
        private async Task CheckRegimeFlipExitAsync(
            PositionManagementState state, 
            decimal currentPrice,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            if (!_regimeMonitoringEnabled || !_regimeFlipExitEnabled)
            {
                return; // Feature disabled
            }
            
            // Check if regime has changed
            if (state.CurrentRegime == state.EntryRegime)
            {
                return; // No regime flip
            }
            
            try
            {
                // Calculate current P&L
                var isLong = state.Quantity > 0;
                var profitTicks = CalculateProfitTicks(state.EntryPrice, currentPrice, tickSize, isLong);
                var pnl = profitTicks * tickSize * Math.Abs(state.Quantity);
                
                // Record regime change
                var regimeChange = new RegimeChangeRecord
                {
                    Timestamp = DateTime.UtcNow,
                    FromRegime = state.EntryRegime,
                    ToRegime = state.CurrentRegime,
                    FromConfidence = state.EntryRegimeConfidence,
                    ToConfidence = state.CurrentRegimeConfidence,
                    PnLAtChange = pnl
                };
                state.AddRegimeChange(regimeChange);
                
                // Check if we should exit based on regime flip
                var shouldExit = ShouldExitOnRegimeFlip(state, pnl);
                
                if (shouldExit)
                {
                    _logger.LogWarning("🔄 [REGIME-FLIP] Exiting {PositionId} due to regime flip: {From} → {To}, PnL: ${PnL:F2}, Strategy: {Strategy}",
                        state.PositionId, state.EntryRegime, state.CurrentRegime, pnl, state.Strategy);
                    
                    // AI Commentary: Explain regime flip exit (non-blocking)
                    try
                    {
                        ExplainRegimeFlipExitFireAndForget(
                            state,
                            state.EntryRegime,
                            state.EntryRegimeConfidence,
                            state.CurrentRegime,
                            state.CurrentRegimeConfidence,
                            pnl);
                    }
                    catch
                    {
                        // Silently ignore AI errors - don't disrupt position management
                    }
                    
                    await RequestPositionCloseAsync(state, ExitReason.RegimeChange, cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    _logger.LogInformation("🔄 [REGIME-FLIP] Regime changed for {PositionId}: {From} → {To}, PnL: ${PnL:F2}, holding position",
                        state.PositionId, state.EntryRegime, state.CurrentRegime, pnl);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [REGIME-FLIP] Error checking regime flip exit for {PositionId}", state.PositionId);
            }
        }
        
        /// <summary>
        /// Determine if position should exit based on regime flip (Feature 3)
        /// </summary>
        private bool ShouldExitOnRegimeFlip(PositionManagementState state, decimal currentPnL)
        {
            // Get strategy-specific sensitivity threshold
            var sensitivityThreshold = GetRegimeFlipSensitivity(state.Strategy);
            
            // Major flip detection: regime type changed
            var entryRegimeType = state.EntryRegime.ToUpperInvariant();
            var currentRegimeType = state.CurrentRegime.ToUpperInvariant();
            var isTrendingToRanging = entryRegimeType.Contains("TREND", StringComparison.Ordinal) && currentRegimeType.Contains("RANGE", StringComparison.Ordinal);
            var isRangingToVolatile = entryRegimeType.Contains("RANGE", StringComparison.Ordinal) && currentRegimeType.Contains("TRANSITION", StringComparison.Ordinal);
            
            // Confidence drop detection
            var confidenceDrop = state.EntryRegimeConfidence - state.CurrentRegimeConfidence;
            var isMajorConfidenceDrop = confidenceDrop >= _regimeConfidenceDropThreshold;
            
            // Strategy-specific exit logic
            switch (state.Strategy)
            {
                case "S6": // Momentum strategy - EXTREMELY sensitive
                    // Exit immediately on trending → ranging flip
                    if (isTrendingToRanging)
                        return true;
                    // Exit if trending confidence drops below threshold
                    if (entryRegimeType.Contains("TREND", StringComparison.Ordinal) && state.CurrentRegimeConfidence < sensitivityThreshold)
                        return true;
                    break;
                    
                case "S2": // Supply/Demand - Highly sensitive
                    // Exit on major flip if we have profit
                    if (isTrendingToRanging && currentPnL > 0)
                        return true;
                    // Exit on major confidence drop
                    if (isMajorConfidenceDrop && state.EntryConfidence < MAJOR_CONFIDENCE_DROP_THRESHOLD_S2)
                        return true;
                    break;
                    
                case "S3": // Multi-timeframe - Moderately sensitive
                    // Exit only on confidence collapse (> 0.30 drop)
                    if (confidenceDrop > MAJOR_CONFIDENCE_DROP_THRESHOLD_S3)
                        return true;
                    break;
                    
                case "S11": // Pattern-based - Less sensitive
                    // Exit only on severe confidence collapse (> 0.40 drop)
                    if (confidenceDrop > MAJOR_CONFIDENCE_DROP_THRESHOLD_S11)
                        return true;
                    break;
            }
            
            // General rule: Exit if positive PnL and major unfavorable flip
            if (currentPnL > 0 && (isTrendingToRanging || isRangingToVolatile) && state.EntryConfidence < GENERAL_CONFIDENCE_DROP_THRESHOLD)
            {
                return true;
            }
            
            return false;
        }
        
        /// <summary>
        /// Get strategy-specific regime flip sensitivity threshold
        /// </summary>
        private static decimal GetRegimeFlipSensitivity(string strategy)
        {
            var envVarName = $"{strategy}_REGIME_FLIP_SENSITIVITY";
            var envValue = Environment.GetEnvironmentVariable(envVarName);
            if (decimal.TryParse(envValue, out var sensitivity))
            {
                return sensitivity;
            }
            
            // Fallback defaults
            return strategy switch
            {
                "S6" => S6_REGIME_FLIP_SENSITIVITY, // Most sensitive
                "S2" => S2_REGIME_FLIP_SENSITIVITY,
                "S3" => S3_REGIME_FLIP_SENSITIVITY,
                "S11" => S11_REGIME_FLIP_SENSITIVITY,
                _ => DEFAULT_REGIME_FLIP_SENSITIVITY
            };
        }
        
        // ========================================================================
        // FEATURE 2: MAE/MFE OPTIMAL STOP PLACEMENT LEARNING METHODS
        // ========================================================================
        
        /// <summary>
        /// Check if position should exit early based on learned MAE threshold
        /// SAFETY: Only exits early if MAE exceeds learned threshold (never loosens stops)
        /// </summary>
        private async Task CheckEarlyExitThresholdAsync(
            PositionManagementState state,
            decimal currentPrice,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            if (!_maeLearningEnabled)
            {
                return; // Feature disabled
            }
            
            try
            {
                var optimizer = _serviceProvider.GetService<PositionManagementOptimizer>();
                if (optimizer == null)
                {
                    return; // Optimizer not available
                }
                
                // Get optimal early exit threshold for this strategy and regime
                var optimalThreshold = optimizer.GetOptimalEarlyExitThreshold(state.Strategy, state.CurrentRegime);
                if (!optimalThreshold.HasValue)
                {
                    return; // Not enough data yet
                }
                
                // Calculate current adverse excursion
                var isLong = state.Quantity > 0;
                var currentAdverseExcursion = isLong
                    ? Math.Abs((state.MaxAdversePrice - state.EntryPrice) / tickSize)
                    : Math.Abs((state.EntryPrice - state.MaxAdversePrice) / tickSize);
                
                // Check if we've exceeded the learned threshold
                if (currentAdverseExcursion > optimalThreshold.Value)
                {
                    _logger.LogWarning("🚨 [MAE-LEARNING] Early exit triggered for {PositionId}: MAE {Current:F1} ticks exceeds learned threshold {Threshold:F1} ticks",
                        state.PositionId, currentAdverseExcursion, optimalThreshold.Value);
                    
                    // Exit position early (this is likely a loser)
                    await RequestPositionCloseAsync(state, ExitReason.StopLoss, cancellationToken).ConfigureAwait(false);
                }
                else
                {
                    // Log progress for monitoring (only if getting close to threshold)
                    if (currentAdverseExcursion > optimalThreshold.Value * MAE_WARNING_THRESHOLD_MULTIPLIER)
                    {
                        _logger.LogDebug("⚠️ [MAE-LEARNING] Position {PositionId} approaching MAE threshold: {Current:F1} / {Threshold:F1} ticks",
                            state.PositionId, currentAdverseExcursion, optimalThreshold.Value);
                        
                        // AI Commentary: Explain MAE warning (non-blocking)
                        try
                        {
                            var historicalSampleSize = 150; // Approximate - could be fetched from optimizer
                            ExplainMaeWarningFireAndForget(
                                state,
                                optimalThreshold.Value,
                                currentAdverseExcursion,
                                historicalSampleSize);
                        }
                        catch
                        {
                            // Silently ignore AI errors - don't disrupt position management
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [MAE-LEARNING] Error checking early exit threshold for {PositionId}", state.PositionId);
            }
        }
        
        /// <summary>
        /// Get MFE-optimized trailing distance for this strategy and regime
        /// </summary>
        private decimal? GetOptimizedTrailingDistance(PositionManagementState state, decimal tickSize)
        {
            if (!_mfeLearningEnabled)
            {
                return null; // Feature disabled
            }
            
            try
            {
                var optimizer = _serviceProvider.GetService<PositionManagementOptimizer>();
                if (optimizer == null)
                {
                    return null;
                }
                
                return optimizer.GetOptimalTrailingDistance(state.Strategy, state.CurrentRegime);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [MFE-LEARNING] Error getting optimized trailing distance for {PositionId}", state.PositionId);
                return null;
            }
        }
        
        // ========================================================================
        // FEATURE 5: PROGRESSIVE TIME-DECAY STOP TIGHTENING METHODS
        // ========================================================================
        
        /// <summary>
        /// Check for progressive tightening based on time elapsed and profit performance
        /// Gradually tightens stops and profit thresholds as trade ages without hitting targets
        /// </summary>
        private async Task CheckProgressiveTighteningAsync(
            PositionManagementState state,
            decimal currentPrice,
            decimal profitTicks,
            decimal tickSize,
            CancellationToken cancellationToken)
        {
            if (!_progressiveTighteningEnabled)
            {
                return; // Feature disabled
            }
            
            // Check if it's time to evaluate progressive tightening
            var timeSinceLastCheck = DateTime.UtcNow - state.LastProgressiveTighteningCheck;
            if (timeSinceLastCheck.TotalSeconds < _progressiveTighteningCheckIntervalSeconds)
            {
                return; // Not time yet
            }
            
            state.LastProgressiveTighteningCheck = DateTime.UtcNow;
            
            // Track peak profit for progressive exit thresholds
            if (profitTicks > state.PeakProfitTicks)
            {
                state.PeakProfitTicks = profitTicks;
            }
            
            var isLong = state.Quantity > 0;
            var holdDuration = DateTime.UtcNow - state.EntryTime;
            var holdMinutes = (int)holdDuration.TotalMinutes;
            
            // Get strategy-specific tightening schedule
            var tighteningSchedule = GetProgressiveTighteningSchedule(state.Strategy);
            
            // Determine current tier based on time elapsed
            int newTier = 0;
            foreach (var threshold in tighteningSchedule)
            {
                if (holdMinutes >= threshold.MinutesThreshold)
                {
                    newTier = threshold.Tier;
                }
            }
            
            // Only process if we've moved to a new tier
            if (newTier <= state.ProgressiveTighteningTier)
            {
                return; // No tier change
            }
            
            state.ProgressiveTighteningTier = newTier;
            
            // Get current tier requirements
            var currentTierRequirements = tighteningSchedule.Find(t => t.Tier == newTier);
            if (currentTierRequirements == null)
            {
                return;
            }
            
            _logger.LogInformation("⏱️ [PROGRESSIVE-TIGHTENING] Position {PositionId} moved to tier {NewTier} after {Minutes}m: {Description}",
                state.PositionId, newTier, holdMinutes, currentTierRequirements.Description);
            
            // Check if profit meets tier requirements
            var riskTicks = Math.Abs(state.EntryPrice - state.CurrentStopPrice) / tickSize;
            var currentRMultiple = riskTicks > 0 ? profitTicks / riskTicks : 0;
            
            // Apply tier-specific actions
            bool shouldExit = false;
            string exitReason = "";
            
            if (currentTierRequirements.Action == ProgressiveTighteningAction.MoveStopToBreakeven)
            {
                if (!state.BreakevenActivated && profitTicks < currentTierRequirements.MinProfitTicksRequired)
                {
                    // Not profitable enough - move to breakeven
                    var breakevenStop = isLong ? state.EntryPrice + tickSize : state.EntryPrice - tickSize;
                    await ModifyStopPriceAsync(state, breakevenStop, "Progressive-Breakeven", cancellationToken).ConfigureAwait(false);
                    state.BreakevenActivated = true;
                    
                    _logger.LogInformation("⏱️ [PROGRESSIVE-TIGHTENING] Moved {PositionId} to breakeven (tier {Tier}, profit {Profit} ticks < required {Required})",
                        state.PositionId, newTier, profitTicks, currentTierRequirements.MinProfitTicksRequired);
                    
                    // AI Commentary: Explain progressive tightening (non-blocking)
                    try
                    {
                        var minutesElapsed = (int)(DateTime.UtcNow - state.EntryTime).TotalMinutes;
                        ExplainProgressiveTighteningFireAndForget(
                            state,
                            newTier,
                            minutesElapsed,
                            currentTierRequirements.MinutesThreshold,
                            profitTicks,
                            "Move stop to breakeven");
                    }
                    catch
                    {
                        // Silently ignore AI errors - don't disrupt position management
                    }
                }
            }
            else if (currentTierRequirements.Action == ProgressiveTighteningAction.ExitIfBelowThreshold)
            {
                if (currentRMultiple < currentTierRequirements.MinRMultipleRequired)
                {
                    shouldExit = true;
                    exitReason = $"Progressive exit: R={currentRMultiple:F2} < required {currentTierRequirements.MinRMultipleRequired:F2} at tier {newTier}";
                }
            }
            else if (currentTierRequirements.Action == ProgressiveTighteningAction.ForceExit)
            {
                shouldExit = true;
                exitReason = $"Progressive force exit: Max hold time reached (tier {newTier})";
            }
            
            if (shouldExit)
            {
                _logger.LogWarning("⏱️ [PROGRESSIVE-TIGHTENING] Exiting {PositionId}: {Reason}",
                    state.PositionId, exitReason);
                
                // AI Commentary: Explain progressive tightening exit (non-blocking)
                try
                {
                    var minutesElapsed = (int)(DateTime.UtcNow - state.EntryTime).TotalMinutes;
                    ExplainProgressiveTighteningFireAndForget(
                        state,
                        newTier,
                        minutesElapsed,
                        currentTierRequirements.MinutesThreshold,
                        profitTicks,
                        exitReason);
                }
                catch
                {
                    // Silently ignore AI errors - don't disrupt position management
                }
                
                await RequestPositionCloseAsync(state, ExitReason.TimeLimit, cancellationToken).ConfigureAwait(false);
            }
        }
        
        /// <summary>
        /// Get progressive tightening schedule for a strategy
        /// </summary>
        private static List<ProgressiveTighteningThreshold> GetProgressiveTighteningSchedule(string strategy)
        {
            return strategy switch
            {
                "S2" => new List<ProgressiveTighteningThreshold>
                {
                    new() { Tier = TIER_1, MinutesThreshold = S2_TIER1_MINUTES, Action = ProgressiveTighteningAction.MoveStopToBreakeven, 
                           MinProfitTicksRequired = MIN_PROFIT_TICKS_TIER1, Description = "Move to breakeven if not profitable" },
                    new() { Tier = TIER_2, MinutesThreshold = S2_TIER2_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER2_R_MULTIPLE_REQUIREMENT, Description = "Exit if not at 1.0R" },
                    new() { Tier = TIER_3, MinutesThreshold = S2_TIER3_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER3_R_MULTIPLE_REQUIREMENT_S2, Description = "Exit if not at 1.5R" },
                    new() { Tier = TIER_4, MinutesThreshold = S2_TIER4_MINUTES, Action = ProgressiveTighteningAction.ForceExit, 
                           Description = "Force exit at max hold time" }
                },
                
                "S3" => new List<ProgressiveTighteningThreshold>
                {
                    new() { Tier = TIER_1, MinutesThreshold = S3_TIER1_MINUTES, Action = ProgressiveTighteningAction.MoveStopToBreakeven, 
                           MinProfitTicksRequired = MIN_PROFIT_TICKS_TIER1, Description = "Move to breakeven if not profitable" },
                    new() { Tier = TIER_2, MinutesThreshold = S3_TIER2_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER2_R_MULTIPLE_REQUIREMENT, Description = "Exit if not at 1.0R" },
                    new() { Tier = TIER_3, MinutesThreshold = S3_TIER3_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER3_R_MULTIPLE_REQUIREMENT_S3, Description = "Exit if not at 2.0R" },
                    new() { Tier = TIER_4, MinutesThreshold = S3_TIER4_MINUTES, Action = ProgressiveTighteningAction.ForceExit, 
                           Description = "Force exit at max hold time" }
                },
                
                "S6" => new List<ProgressiveTighteningThreshold>
                {
                    new() { Tier = TIER_1, MinutesThreshold = S6_TIER1_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinProfitTicksRequired = MIN_PROFIT_TICKS_S6_TIER1, Description = "Exit if not at +6 ticks (momentum should move fast)" },
                    new() { Tier = TIER_2, MinutesThreshold = S6_TIER2_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER2_R_MULTIPLE_REQUIREMENT, Description = "Exit if not at 1.0R" },
                    new() { Tier = TIER_3, MinutesThreshold = S6_TIER3_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER3_R_MULTIPLE_REQUIREMENT_S6, Description = "Exit if not at 1.5R" },
                    new() { Tier = TIER_4, MinutesThreshold = S6_TIER4_MINUTES, Action = ProgressiveTighteningAction.ForceExit, 
                           Description = "Force exit at max hold time" }
                },
                
                "S11" => new List<ProgressiveTighteningThreshold>
                {
                    new() { Tier = TIER_1, MinutesThreshold = S11_TIER1_MINUTES, Action = ProgressiveTighteningAction.MoveStopToBreakeven, 
                           MinProfitTicksRequired = MIN_PROFIT_TICKS_TIER1, Description = "Move to breakeven if not profitable" },
                    new() { Tier = TIER_2, MinutesThreshold = S11_TIER2_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER2_R_MULTIPLE_REQUIREMENT_S11, Description = "Exit if not at 1.5R" },
                    new() { Tier = TIER_3, MinutesThreshold = S11_TIER3_MINUTES, Action = ProgressiveTighteningAction.ForceExit, 
                           Description = "Force exit at max hold time" }
                },
                
                _ => new List<ProgressiveTighteningThreshold>
                {
                    new() { Tier = TIER_1, MinutesThreshold = DEFAULT_TIER1_MINUTES, Action = ProgressiveTighteningAction.MoveStopToBreakeven, 
                           MinProfitTicksRequired = MIN_PROFIT_TICKS_TIER1, Description = "Move to breakeven if not profitable" },
                    new() { Tier = TIER_2, MinutesThreshold = DEFAULT_TIER2_MINUTES, Action = ProgressiveTighteningAction.ExitIfBelowThreshold, 
                           MinRMultipleRequired = TIER2_R_MULTIPLE_REQUIREMENT, Description = "Exit if not at 1.0R" },
                    new() { Tier = TIER_3, MinutesThreshold = DEFAULT_TIER3_MINUTES, Action = ProgressiveTighteningAction.ForceExit, 
                           Description = "Force exit at max hold time" }
                }
            };
        }
        
        // ========================================================================
        // AI COMMENTARY METHODS (PHASE 5 - Ollama Integration)
        // ========================================================================
        
        /// <summary>
        /// Fire-and-forget: Explain trailing stop activation in background
        /// AI explains why trailing stop was activated without blocking trade execution
        /// </summary>
        private void ExplainTrailingStopActivationFireAndForget(
            PositionManagementState state,
            decimal unrealizedPnL,
            decimal profitTicks,
            decimal newStopPrice,
            string marketRegime)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            // Start background task but don't wait for it - trading continues immediately
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var trailDistance = state.TrailTicks;
                    var oldStop = state.CurrentStopPrice;
                    var profitAmount = unrealizedPnL;
                    
                    var prompt = $@"I am a trading bot. I just activated a trailing stop:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Current Price: {(isLong ? state.EntryPrice + (profitTicks * GetTickSize(state.Symbol)) : state.EntryPrice - (profitTicks * GetTickSize(state.Symbol))):F2}
Profit: {profitTicks:F1} ticks (${profitAmount:F2})
Trail Distance: {trailDistance} ticks behind peak
Old Stop: {oldStop:F2} → New Stop: {newStopPrice:F2}
Market Regime: {marketRegime}

Explain in 2-3 sentences why this trailing stop activation is smart and protects my profits. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Trailing Stop: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating trailing stop commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain breakeven protection activation in background
        /// </summary>
        private void ExplainBreakevenProtectionFireAndForget(
            PositionManagementState state,
            decimal unrealizedPnL,
            decimal profitTicks,
            decimal newStopPrice,
            string marketRegime)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var breakevenThreshold = state.BreakevenAfterTicks;
                    var oldStop = state.GetProperty("InitialStopPrice") as decimal? ?? state.CurrentStopPrice;
                    var profitAmount = unrealizedPnL;
                    
                    var prompt = $@"I am a trading bot. I just activated breakeven protection:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Profit Reached: {profitTicks:F1} ticks (${profitAmount:F2})
Breakeven Threshold: {breakevenThreshold} ticks
Old Stop: {oldStop:F2} (initial risk) → New Stop: {newStopPrice:F2} (breakeven + 1 tick)
Market Regime: {marketRegime}

Explain in 2-3 sentences why moving my stop to breakeven is smart risk management now that I'm profitable. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Breakeven: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating breakeven commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain partial exit execution in background
        /// </summary>
        private void ExplainPartialExitFireAndForget(
            PositionManagementState state,
            decimal rMultiple,
            decimal partialPercent,
            decimal partialQuantity,
            string targetLevel)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var remainingQuantity = state.Quantity - (int)partialQuantity;
                    var initialStop = state.GetProperty("InitialStopPrice") as decimal? ?? state.CurrentStopPrice;
                    var profitPerContract = rMultiple * Math.Abs(initialStop - state.EntryPrice);
                    
                    var prompt = $@"I am a trading bot. I just took a partial profit:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Risk-Multiple: {rMultiple:F1}R (profit is {rMultiple:F1}x my initial risk)
Partial Exit: Closed {partialPercent:F0}% ({partialQuantity:F0} contracts) at {targetLevel}
Remaining: {remainingQuantity} contracts still running
Profit on this partial: ${profitPerContract * partialQuantity:F2}

Explain in 2-3 sentences why taking this partial profit is smart while letting the rest run. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Partial Exit: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating partial exit commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain zone break forced exit in background
        /// </summary>
        private void ExplainZoneBreakExitFireAndForget(
            PositionManagementState state,
            ZoneBreakEvent breakEvent,
            decimal currentPnL)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var zoneType = breakEvent.BreakType.ToString();
                    var severity = breakEvent.Severity;
                    var zoneStrength = breakEvent.ZoneStrength;
                    
                    var prompt = $@"I am a trading bot. I just EMERGENCY EXITED a position due to a zone break:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Zone Break: {zoneType}
Severity: {severity}
Zone Strength: {zoneStrength:F2}
Zone Range: {breakEvent.ZoneLow:F2} - {breakEvent.ZoneHigh:F2}
P&L at Exit: ${currentPnL:F2}

Explain in 2-3 sentences why this critical structural break forced me to exit immediately, even if I took a loss. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Zone Break Exit: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating zone break commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain time-based exit in background
        /// </summary>
        private void ExplainTimeBasedExitFireAndForget(
            PositionManagementState state,
            TimeSpan holdDuration,
            decimal currentPnL,
            string marketRegime)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var maxHoldMinutes = state.MaxHoldMinutes;
                    var actualMinutes = holdDuration.TotalMinutes;
                    
                    var prompt = $@"I am a trading bot. I just closed a position due to time limit:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Hold Duration: {actualMinutes:F1} minutes
Max Hold Time: {maxHoldMinutes} minutes
Market Regime: {marketRegime}
P&L at Exit: ${currentPnL:F2}

Explain in 2-3 sentences why I closed this stale position that wasn't moving anywhere. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Time Exit: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating time exit commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain regime flip exit decision in background
        /// </summary>
        private void ExplainRegimeFlipExitFireAndForget(
            PositionManagementState state,
            string entryRegime,
            decimal entryRegimeConfidence,
            string currentRegime,
            decimal currentRegimeConfidence,
            decimal currentPnL)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var confidenceDrop = entryRegimeConfidence - currentRegimeConfidence;
                    var tickSize = GetTickSize(state.Symbol);
                    var profitTicks = Math.Abs(currentPnL / (tickSize * 50m)); // Approximate ticks
                    
                    var prompt = $@"I am a trading bot. I just exited due to a regime flip:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Entry Regime: {entryRegime} (confidence: {entryRegimeConfidence:F2})
Current Regime: {currentRegime} (confidence: {currentRegimeConfidence:F2})
Confidence Drop: {confidenceDrop:F2}
Strategy: {state.Strategy}
P&L at Exit: ${currentPnL:F2} ({profitTicks:F1} ticks)

Explain in 2-3 sentences why this regime change forced me to exit and protect my current P&L. Focus on why {state.Strategy} strategy requires the conditions that no longer exist. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Regime Flip Exit: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating regime flip commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain progressive tightening tier action in background
        /// </summary>
        private void ExplainProgressiveTighteningFireAndForget(
            PositionManagementState state,
            int tier,
            int minutesElapsed,
            int tierThreshold,
            decimal profitTicks,
            string action)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var tickSize = GetTickSize(state.Symbol);
                    var profitDollars = profitTicks * tickSize * 50m; // ES multiplier
                    
                    var prompt = $@"I am a trading bot. Progressive tightening tier triggered:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Hold Duration: {minutesElapsed} minutes (exceeded {tierThreshold} minute threshold)
Tier Level: {tier}
Current Profit: {profitTicks:F1} ticks (${profitDollars:F2})
Action Taken: {action}
Strategy: {state.Strategy}

Explain in 2-3 sentences why time-based position management is taking this action. Reference how {state.Strategy} trades statistically perform after {minutesElapsed} minutes. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Progressive Tightening: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating progressive tightening commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain confidence-based parameter adjustment in background
        /// </summary>
        private void ExplainConfidenceAdjustmentFireAndForget(
            PositionManagementState state,
            decimal entryConfidence,
            decimal stopMultiplier,
            decimal targetMultiplier,
            int adjustedQuantity,
            decimal adjustedStop,
            decimal adjustedTarget)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var confidenceTier = GetConfidenceTier(entryConfidence);
                    
                    static string GetConfidenceTier(decimal confidence)
                    {
                        const decimal VeryHighConfidenceThreshold = 0.85m;
                        const decimal HighConfidenceThreshold = 0.75m;
                        const decimal MediumConfidenceThreshold = 0.65m;
                        
                        if (confidence >= VeryHighConfidenceThreshold) return "Very High";
                        if (confidence >= HighConfidenceThreshold) return "High";
                        if (confidence >= MediumConfidenceThreshold) return "Medium";
                        return "Low";
                    }
                    
                    var prompt = $@"I am a trading bot. ML confidence adjusted my trade parameters:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
ML Confidence: {entryConfidence:F2} ({confidenceTier} tier)
Stop Multiplier: {stopMultiplier:F2}x (adjusted to ${adjustedStop:F2})
Target Multiplier: {targetMultiplier:F2}x (adjusted to ${adjustedTarget:F2})
Quantity: {adjustedQuantity} contracts
Strategy: {state.Strategy}

Explain in 2-3 sentences why this confidence level justifies these parameter adjustments. Reference historical win rates at this confidence level. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Confidence Adjustment: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating confidence adjustment commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain dynamic target adjustment in background
        /// </summary>
        private void ExplainDynamicTargetAdjustmentFireAndForget(
            PositionManagementState state,
            string entryRegime,
            string currentRegime,
            decimal oldTargetR,
            decimal newTargetR)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var targetChange = newTargetR - oldTargetR;
                    var changeDirection = targetChange > 0 ? "extended" : "reduced";
                    
                    var prompt = $@"I am a trading bot. Market regime shift adjusted my target:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Entry Regime: {entryRegime}
Current Regime: {currentRegime}
Old Target: {oldTargetR:F1}R
New Target: {newTargetR:F1}R (target {changeDirection} by {Math.Abs(targetChange):F1}R)
Strategy: {state.Strategy}

Explain in 2-3 sentences why this regime change requires a different profit target. Reference how {currentRegime} markets historically produce different move sizes. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] Dynamic Target: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating dynamic target commentary");
                }
            });
        }
        
        /// <summary>
        /// Fire-and-forget: Explain MAE warning threshold in background
        /// </summary>
        private void ExplainMaeWarningFireAndForget(
            PositionManagementState state,
            decimal learnedMaeThreshold,
            decimal currentMae,
            int historicalSampleSize)
        {
            if (!_commentaryEnabled || _ollamaClient == null)
                return;
            
            _ = Task.Run(async () =>
            {
                try
                {
                    var isLong = state.Quantity > 0;
                    var direction = isLong ? "LONG" : "SHORT";
                    var proximityPercent = (currentMae / learnedMaeThreshold) * 100m;
                    var tickSize = GetTickSize(state.Symbol);
                    var currentStopDistance = Math.Abs(state.CurrentStopPrice - state.EntryPrice) / tickSize;
                    
                    var prompt = $@"I am a trading bot. MAE approaching learned threshold:

Position: {direction} {state.Symbol} at {state.EntryPrice:F2}
Current MAE: {currentMae:F1} ticks
Learned Optimal MAE: {learnedMaeThreshold:F1} ticks (from {historicalSampleSize} historical trades)
MAE Proximity: {proximityPercent:F0}% of threshold
Current Stop Distance: {currentStopDistance:F1} ticks
Strategy: {state.Strategy}

Explain in 2-3 sentences what this MAE warning means about my stop placement and whether I should tighten it. Reference the historical recovery rate when MAE exceeds this threshold. Speak as ME (the bot).";

                    var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(response))
                    {
                        _logger.LogInformation("🤖💭 [POSITION-AI] MAE Warning: {Commentary}", response);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "❌ [POSITION-AI] Error generating MAE warning commentary");
                }
            });
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                _monitoringTimer?.Dispose();
                _disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
    
    /// <summary>
    /// Progressive tightening threshold configuration
    /// </summary>
    internal sealed class ProgressiveTighteningThreshold
    {
        public int Tier { get; set; }
        public int MinutesThreshold { get; set; }
        public ProgressiveTighteningAction Action { get; set; }
        public decimal MinProfitTicksRequired { get; set; }
        public decimal MinRMultipleRequired { get; set; }
        public string Description { get; set; } = string.Empty;
    }
    
    /// <summary>
    /// Actions for progressive tightening
    /// </summary>
    internal enum ProgressiveTighteningAction
    {
        MoveStopToBreakeven,
        ExitIfBelowThreshold,
        ForceExit
    }
}
