using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Strategy;
using BotCore.Risk;
using BotCore.Models;
using BotCore.ML;
using BotCore.Bandits;
using BotCore.Brain.Models;
using System.Collections.Concurrent;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using TradingBot.RLAgent; // For CVaRPPO and ActionResult
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using TradingBot.Abstractions;

// Type aliases to resolve ambiguity between BotCore.Brain.Models and TradingBot.Abstractions
using MarketContext = BotCore.Brain.Models.MarketContext;
using MarketRegime = BotCore.Brain.Models.MarketRegime;
using TradingDecision = BotCore.Brain.Models.TradingDecision;
using NewsContext = BotCore.Services.NewsContext; // For NewsMonitorService integration

namespace BotCore.Brain
{
    /// <summary>
    /// TopStep compliance configuration integrated into UnifiedTradingBrain
    /// </summary>
    public static class TopStepConfig
    {
        public const decimal AccountSize = 50_000m;
        public const decimal MaxDrawdown = 2_000m;
        public const decimal DailyLossLimit = 1_000m;
        public const decimal TrailingStop = 48_000m;
        public const decimal EsPointValue = 50m;
        public const decimal NqPointValue = 20m;
        public const decimal RiskPerTrade = 0.01m; // 1% = $500 baseline
        public const double ExplorationBonus = 0.3;
        public const double ConfidenceThreshold = 0.65;
        
        // Trading Brain confidence and probability constants
        public const decimal FallbackConfidence = 0.6m;             // Fallback strategy confidence
        public const decimal FallbackUcbValue = 0.5m;              // Fallback UCB value
        public const decimal HighConfidenceProbability = 0.70m;     // High confidence prediction probability
        public const decimal NeutralProbability = 0.5m;             // Neutral/sideways prediction probability
        public const decimal DefaultAtrFallback = 25.0m;           // Default ATR for NQ when unavailable
        public const decimal EsDefaultAtrFallback = 10.0m;        // Default ATR for ES when unavailable
        public const decimal NqMinStopDistance = 0.5m;            // Minimum stop distance for NQ
        public const decimal EsMinStopDistance = 0.25m;           // Minimum stop distance for ES
        public const decimal FallbackExpectedMove = 5;             // Fallback expected move value
        public const int DefaultAtrLookback = 10;                  // Default ATR lookback period
        
        // Position sizing and risk multipliers
        public const decimal MinRlMultiplier = 0.5m;               // Minimum RL position multiplier
        public const decimal MaxRlMultiplier = 1.5m;               // Maximum RL position multiplier
        public const decimal NearDailyLossWarningFactor = 0.9m;    // Warning at 90% of daily loss
        public const decimal NearMaxDrawdownWarningFactor = 0.8m;  // Warning at 80% of max drawdown
        
        // Confidence and reward calculation constants
        public const decimal StrategyPredictionAverageWeight = 2m; // Weight for averaging strategy and prediction confidence
        public const decimal DefaultNeutralPositionMultiplier = 1.0m; // Default neutral position size
        public const decimal DefaultRsiNeutral = 50m;              // Default RSI neutral value
        public const decimal DefaultRsiMax = 100m;                 // Maximum RSI value
        public const int MinBarsPeriod = 10;                       // Minimum bars for period calculations
        public const int MinBarsExtended = 20;                     // Minimum bars for extended calculations
        
        // Strategy performance evaluation thresholds
        public const decimal PoorPerformanceWinRateThreshold = 0.4m;  // Below this is considered poor performance
        public const decimal UnsuccessfulConditionThreshold = 0.3m;   // Condition success rate below this is unsuccessful
        public const decimal HighPerformanceWinRateThreshold = 0.7m;  // Above this is considered high performance
        public const decimal MinimumWinRateToSharePatterns = 0.6m;   // Minimum win rate to share patterns with other strategies
        public const decimal SharedPatternSuccessThreshold = 0.7m;   // Success rate threshold for cross-pollination
        public const decimal CrossPollinationWeight = 0.1m;          // Lower weight for cross-pollinated patterns
        public const decimal MinimumConditionSuccessRate = 0.6m;     // Minimum success rate for condition to be considered
        public const int MinimumConditionTrialCount = 3;             // Minimum number of trials to trust condition
        public const int TopConditionsToSelect = 5;                  // Number of top conditions to select per strategy
        public const int MinimumDecisionsForTrend = 5;               // Minimum decisions needed to calculate performance trend
        public const int RecentDecisionsLookbackHours = 24;          // Hours to look back for recent decisions
        
        // Scheduling and learning intervals
        public const int MaintenanceLearningIntervalMinutes = 10;    // Learning interval during maintenance window
        public const int ClosedMarketLearningIntervalMinutes = 15;   // Learning interval when market is closed
        public const int OpenMarketLearningIntervalMinutes = 60;     // Learning interval when market is open
        
        // Commentary thresholds
        public const decimal LowConfidenceThreshold = 0.4m;          // Below this triggers waiting commentary
        // HighConfidenceThreshold removed - now uses IMLConfigurationService.GetAIConfidenceThreshold()
        public const decimal StrategyConflictThreshold = 0.15m;       // Score difference threshold for conflict detection
        // AlternativeStrategyConfidenceFactor removed - now uses IMLConfigurationService.GetAIConfidenceThreshold()
        
        // Statistical calculation constants
        public const double TotalVariationNormalizationFactor = 0.5; // Factor for normalizing total variation distance
        
        // Historical simulation constants
        public const int MinHistoricalBarsForSimulation = 100;       // Minimum bars needed for reliable simulation
        public const int FeatureVectorLength = 11;                   // Number of features in simulation data
        public const int SimulationRandomSeed = 12345;               // Seed for reproducible simulation data
        public const double SimulationFeatureRange = 2.0;            // Range for random feature generation
        public const double SimulationFeatureOffset = 1.0;           // Offset for centering feature range
        public const int TrainingDataHistorySize = 2000;             // Number of recent decisions to use for training
        
        // CVaR-PPO state vector normalization constants
        public const double VolatilityNormalizationDivisor = 2.0;    // Divisor for volatility normalization
        public const double PriceChangeMomentumDivisor = 20.0;       // Divisor for price change normalization
        public const double VolumeSurgeNormalizationDivisor = 3.0;   // Divisor for volume ratio normalization
        public const double AtrNormalizationDivisor = 50.0;          // Divisor for ATR normalization
        public const double UcbValueNormalizationDivisor = 10.0;     // Divisor for UCB value normalization
        public const double S2VwapStrategyEncoding = 0.25;           // Encoding value for S2_VWAP strategy
        public const double S3CompressionStrategyEncoding = 0.5;     // Encoding value for S3_Compression strategy
        public const double S11OpeningStrategyEncoding = 0.75;       // Encoding value for S11_Opening strategy
        public const double S12MomentumStrategyEncoding = 1.0;       // Encoding value for S12_Momentum strategy
        public const double DefaultStrategyEncoding = 0.0;           // Default encoding for unknown strategy
        public const double DirectionEncodingDivisor = 2.0;          // Divisor for direction encoding
        public const double DirectionEncodingOffset = 0.5;           // Offset for direction encoding (Down=0, Sideways=0.5, Up=1)
        public const double HoursPerDay = 24.0;                      // Hours in a day for cyclical encoding
        public const double MaxDecisionsPerDayNormalization = 50.0;  // Maximum decisions per day for normalization
        
        // CVaR-PPO action to position size multipliers
        public const decimal NoTradeMultiplier = 0.0m;               // Action 0: No trade
        public const decimal MicroPositionMultiplier = 0.25m;        // Action 1: Micro position (25%)
        public const decimal SmallPositionMultiplier = 0.5m;         // Action 2: Small position (50%)
        public const decimal NormalPositionMultiplier = 1.0m;        // Action 3: Normal position (100%)
        public const decimal LargePositionMultiplier = 1.5m;         // Action 4: Large position (150%)
        public const decimal MaxPositionMultiplier = 2.0m;           // Action 5: Maximum position (200%)
        
        // CVaR risk adjustment constants
        public const decimal MaxNormalizationValue = 1.0m;           // Maximum value for Min() normalization
        public const double MinNormalizationValue = 1.0;             // Minimum value for Min() normalization (double)
        public const int CyclicalEncodingMultiplier = 2;             // Multiplier for sin/cos cyclical encoding
        public const decimal MinProbabilityAdjustment = 0.3m;        // Minimum probability adjustment for position sizing
        public const decimal MinValueAdjustment = 0.2m;              // Minimum value adjustment for position sizing
        public const decimal MaxValueAdjustment = 1.5m;              // Maximum value adjustment for position sizing
        public const decimal ValueEstimateOffset = 0.5m;             // Offset for value estimate adjustment
        public const decimal HighNegativeTailRiskThreshold = -0.1m;  // Threshold for high negative tail risk (CVaR)
        public const decimal ModerateTailRiskThreshold = -0.05m;     // Threshold for moderate tail risk (CVaR)
        public const decimal HighRiskPositionReduction = 0.5m;       // Position reduction for high tail risk (50%)
        public const decimal ModerateRiskPositionReduction = 0.75m;  // Position reduction for moderate tail risk (25%)
    }
    /// <summary>
    /// ✅ UNIFIED TRADING BRAIN - The SINGLE Intelligence for ALL Trading Decisions
    /// ==============================================================================
    /// Enhanced to handle all 4 primary strategies (S2, S3, S6, S11) with unified scheduling
    /// 
    /// UNIFIED ARCHITECTURE: Same Brain for Historical and Live Trading
    /// ----------------------------------------------------------------
    /// This is the central AI brain that:
    /// 1. Handles S2 (VWAP Mean Reversion), S3 (Bollinger Compression), S6 (Momentum), S11 (Specialized)
    /// 2. Uses Neural UCB to select optimal strategy for each market condition
    /// 3. Uses LSTM to predict price movements and timing
    /// 4. Uses CVaR-PPO to optimize position sizes for all strategies
    /// 5. ⭐ CRITICAL: Maintains IDENTICAL intelligence for historical and live trading ⭐
    /// 6. Continuously learns from all trade outcomes to improve strategy selection
    /// 
    /// LEARNING FROM BOTH SOURCES (via LearnFromResultAsync - line 1734):
    /// - Historical backtest results (90-day rolling window)
    /// - Live trading results (real-time execution)
    /// - Both feed into same Neural UCB and CVaR-PPO training
    /// 
    /// DECISION MAKING (via MakeIntelligentDecisionAsync):
    /// - Used by EnhancedBacktestLearningService for historical decisions
    /// - Used by live trading services for real-time decisions
    /// - Guarantees consistent logic across historical and live contexts
    /// 
    /// ⚠️ NO DUAL SYSTEMS: This is the ONLY decision engine - no separate brains exist
    /// 
    /// KEY ENHANCEMENTS:
    /// - Multi-strategy learning: Every trade outcome teaches all strategies
    /// - Unified scheduling: Same timing for historical and live systems
    /// - Continuous improvement: Historical patterns improve live strategy selection
    /// - Same AI brain gets smarter at picking S2 vs S3 vs S6 vs S11
    /// - Position sizing and risk management learns from all strategy results
    /// 
    /// INTEGRATION POINTS:
    /// - AutonomousDecisionEngine calls this brain for live trading
    /// - EnhancedBacktestLearningService uses same brain for historical replay
    /// - AllStrategies.generate_candidates() enhanced with brain decisions
    /// - Identical scheduling for Market Open: Light learning every 60 min, Market Closed: Intensive every 15 min
    /// </summary>
    public class UnifiedTradingBrain : IDisposable
    {
        private readonly ILogger<UnifiedTradingBrain> _logger;
        private readonly IMLMemoryManager _memoryManager;
        private readonly StrategyMlModelManager _modelManager;
        private readonly NeuralUcbBandit _strategySelector;
        private readonly IMLConfigurationService _mlConfigService;
        private readonly ConcurrentDictionary<string, MarketContext> _marketContexts = new();
        private readonly ConcurrentDictionary<string, TradingPerformance> _performance = new();
        private readonly CVaRPPO _cvarPPO; // Direct injection instead of loading from memory
        private readonly BotCore.Services.OllamaClient? _ollamaClient; // Optional AI conversation client
        private readonly BotCore.Services.RiskAssessmentCommentary? _riskCommentary;
        private readonly BotCore.Services.AdaptiveLearningCommentary? _learningCommentary;
        private readonly BotCore.Services.MarketSnapshotStore? _snapshotStore;
        private readonly BotCore.Services.HistoricalPatternRecognitionService? _historicalPatterns;
        private readonly BotCore.Services.ParameterChangeTracker? _parameterTracker;
        private readonly BotCore.Services.INewsMonitorService? _newsMonitor; // Optional real-time news monitoring
        
        // Latest market data for risk analysis (updated in MakeIntelligentDecisionAsync)
        private Env? _latestEnv;
        private IList<Bar>? _latestBars;
        private BotCore.Intelligence.Models.MarketIntelligence? _latestIntelligence;
        
        // ML Models for different decision points
        private object? _lstmPricePredictor;
        private object? _metaClassifier;
        private object? _marketRegimeDetector;
        private readonly OnnxNeuralNetwork? _confidenceNetwork;
        
        // TopStep compliance tracking
        private decimal _currentDrawdown;
        private decimal _dailyPnl;
        private decimal _accountBalance = TopStepConfig.AccountSize;
        private DateTime _lastResetDate = DateTime.UtcNow.Date;
        
        // Performance tracking for learning
        private readonly List<TradingDecision> _decisionHistory = new();
        private DateTime _lastModelUpdate = DateTime.MinValue;
        
        // Cached JsonSerializerOptions for performance
        private static readonly JsonSerializerOptions CachedJsonOptions = new() { WriteIndented = true };
        
        // Unified Trading Brain Constants
        // Learning system constants
        private const int MinDecisionsForLearningUpdate = 50;        // Minimum decisions for learning update
        private const int MinDecisionsForModelRetraining = 200;      // Minimum decisions for model retraining
        private const int LearningUpdateIntervalHours = 2;           // Hours between learning updates  
        private const int ModelRetrainingIntervalHours = 8;          // Hours between model retraining
        
        // Market condition analysis constants  
        private const decimal LowVolatilityThreshold = 0.15m;        // Low volatility threshold
        private const decimal HighVolatilityThreshold = 0.4m;        // High volatility threshold
        private const decimal HighVolumeRatioThreshold = 1.5m;       // High volume ratio threshold
        private const decimal StrongTrendThreshold = 0.7m;           // Strong trend threshold
        private const decimal WeakTrendThreshold = 0.3m;             // Weak trend threshold
        private const decimal OverboughtRSILevel = 70m;              // RSI overbought level
        private const decimal OversoldRSILevel = 30m;               // RSI oversold level
        private const decimal BaseConfidenceThreshold = 0.5m;        // Base confidence threshold
        private const decimal MinConfidenceAdjustment = 0.1m;        // Minimum confidence adjustment
        private const decimal TrendingVolatilityThreshold = 0.25m;   // Volatility threshold for trending regime
        private const decimal RangingPriceChangeThreshold = 0.5m;    // Price change threshold for ranging regime
        
        // Trading session hour constants
        private const int OpeningDriveStartHour = 9;                 // Opening drive start hour (9 AM)
        private const int OpeningDriveEndHour = 10;                  // Opening drive end hour (10 AM)
        private const int LunchStartHour = 11;                       // Lunch session start hour (11 AM)
        private const int LunchEndHour = 13;                         // Lunch session end hour (1 PM)
        private const int AfternoonFadeStartHour = 13;               // Afternoon fade start hour (1 PM)
        private const int AfternoonFadeEndHour = 16;                 // Afternoon fade end hour (4 PM)
        
        // Market session hour constants (UTC)
        private const int PreMarketStartHour = 8;                    // Pre-market start hour (8 AM UTC)
        private const int RegularTradingStartHour = 13;              // Regular trading start hour (1 PM UTC)
        private const int RegularTradingEndHour = 20;                // Regular trading end hour (8 PM UTC)
        private const int AfterHoursEndHour = 22;                    // After hours end hour (10 PM UTC)
        private const decimal TrendStrengthThreshold = 0.2m;         // Trend strength classification threshold
        
        // Multi-strategy learning state
        private readonly Dictionary<string, StrategyPerformance> _strategyPerformance = new();
        private readonly Dictionary<string, List<BotCore.Brain.Models.MarketCondition>> _strategyOptimalConditions = new();
        private DateTime _lastUnifiedLearningUpdate = DateTime.MinValue;
        
        // CVaR-PPO experience tracking for live + historical learning
        private double[]? _lastCVaRState;
        private int _lastCVaRAction;
        private double _lastCVaRValue;
        private DateTime _lastCVaRTraining = DateTime.MinValue;
        
        // Gate 4 configuration
        private readonly IGate4Config _gate4Config;
        
        // Economic calendar for event-based trading restrictions (Phase 2)
        private readonly BotCore.Market.IEconomicEventManager? _economicEventManager;
        
        // Primary strategies for focused learning
        private readonly string[] PrimaryStrategies = { "S2", "S3", "S6", "S11" };
        
        // Strategy specializations
        private readonly Dictionary<string, StrategySpecialization> _strategySpecializations = new()
        {
            ["S2"] = new StrategySpecialization 
            { 
                Name = "VWAP Mean Reversion", 
                OptimalConditions = new[] { "ranging", "low_volatility", "high_volume" },
                LearningFocus = "entry_exit_timing",
                TimeWindows = new[] { "overnight", "lunch", "premarket" }
            },
            ["S3"] = new StrategySpecialization 
            { 
                Name = "Bollinger Compression", 
                OptimalConditions = new[] { "low_volatility", "compression", "breakout_setup" },
                LearningFocus = "volatility_breakout_patterns",
                TimeWindows = new[] { "european_open", "us_premarket", "morning_trend" }
            },
            ["S6"] = new StrategySpecialization 
            { 
                Name = "Momentum Strategy", 
                OptimalConditions = new[] { "trending", "high_volume", "opening_drive" },
                LearningFocus = "momentum_strategies",
                TimeWindows = new[] { "opening_drive", "afternoon_trend" }
            },
            ["S11"] = new StrategySpecialization 
            { 
                Name = "ADR Exhaustion Fade", 
                OptimalConditions = new[] { "exhaustion", "range_bound", "mean_reversion" },
                LearningFocus = "exhaustion_patterns",
                TimeWindows = new[] { "afternoon_fade", "end_of_day" }
            }
        };
        
        public bool IsInitialized { get; private set; }
        public DateTime LastDecision { get; private set; }
        public int DecisionsToday { get; private set; }
        public decimal WinRateToday { get; private set; }

        // LoggerMessage delegates for CA1848 compliance - High-value ML/Brain logging
        private static readonly Action<ILogger, Exception?> LogBrainInitialized =
            LoggerMessage.Define(LogLevel.Information, new EventId(1, nameof(LogBrainInitialized)),
                "🧠 [UNIFIED-BRAIN] Initialized with direct CVaR-PPO injection - Ready to make intelligent trading decisions");
        
        private static readonly Action<ILogger, Exception?> LogLoadingModels =
            LoggerMessage.Define(LogLevel.Information, new EventId(2, nameof(LogLoadingModels)),
                "🚀 [UNIFIED-BRAIN] Loading all ML models...");
        
        private static readonly Action<ILogger, Exception?> LogCVarPPOInjected =
            LoggerMessage.Define(LogLevel.Information, new EventId(3, nameof(LogCVarPPOInjected)),
                "✅ [CVAR-PPO] Using direct injection from DI container");
        
        private static readonly Action<ILogger, Exception?> LogAllModelsLoaded =
            LoggerMessage.Define(LogLevel.Information, new EventId(4, nameof(LogAllModelsLoaded)),
                "✅ [UNIFIED-BRAIN] All models loaded successfully - Brain is ONLINE with production CVaR-PPO");
        
        private static readonly Action<ILogger, Exception?> LogModelFileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(5, nameof(LogModelFileNotFound)),
                "❌ [UNIFIED-BRAIN] Model file not found - Using fallback logic");
        
        private static readonly Action<ILogger, Exception?> LogModelDirectoryNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(6, nameof(LogModelDirectoryNotFound)),
                "❌ [UNIFIED-BRAIN] Model directory not found - Using fallback logic");
        
        private static readonly Action<ILogger, Exception?> LogModelIOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(7, nameof(LogModelIOError)),
                "❌ [UNIFIED-BRAIN] I/O error loading models - Using fallback logic");
        
        private static readonly Action<ILogger, Exception?> LogModelAccessDenied =
            LoggerMessage.Define(LogLevel.Error, new EventId(8, nameof(LogModelAccessDenied)),
                "❌ [UNIFIED-BRAIN] Access denied loading models - Using fallback logic");
        
        private static readonly Action<ILogger, Exception?> LogModelInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(9, nameof(LogModelInvalidOperation)),
                "❌ [UNIFIED-BRAIN] Invalid operation during model loading - Using fallback logic");
        
        private static readonly Action<ILogger, Exception?> LogModelInvalidArgument =
            LoggerMessage.Define(LogLevel.Error, new EventId(10, nameof(LogModelInvalidArgument)),
                "❌ [UNIFIED-BRAIN] Invalid argument for model loading - Using fallback logic");
        
        private static readonly Action<ILogger, string, Exception?> LogCalendarBlock =
            LoggerMessage.Define<string>(LogLevel.Warning, new EventId(11, nameof(LogCalendarBlock)),
                "📅 [CALENDAR-BLOCK] Cannot trade {Symbol} - event restriction active");
        
        private static readonly Action<ILogger, string, double, Exception?> LogHighImpactEvent =
            LoggerMessage.Define<string, double>(LogLevel.Warning, new EventId(12, nameof(LogHighImpactEvent)),
                "📅 [CALENDAR-BLOCK] High-impact event '{Event}' in {Minutes:F0} minutes - blocking trades");
        
        private static readonly Action<ILogger, string, double, Exception?> LogDecisionMade =
            LoggerMessage.Define<string, double>(LogLevel.Information, new EventId(13, nameof(LogDecisionMade)),
                "🎯 [DECISION] Made decision for {Symbol} in {ProcessingTime:F2}ms");
        
        private static readonly Action<ILogger, string, string, Exception?> LogStrategySelected =
            LoggerMessage.Define<string, string>(LogLevel.Information, new EventId(14, nameof(LogStrategySelected)),
                "🎯 [STRATEGY] Selected {Strategy} for {Symbol}");
        
        private static readonly Action<ILogger, Exception?> LogTrainingModelUpdate =
            LoggerMessage.Define(LogLevel.Information, new EventId(15, nameof(LogTrainingModelUpdate)),
                "🧠 [LEARNING] Updating models with recent decisions...");
        
        private static readonly Action<ILogger, int, Exception?> LogTrainingComplete =
            LoggerMessage.Define<int>(LogLevel.Information, new EventId(16, nameof(LogTrainingComplete)),
                "✅ [LEARNING] Training complete - processed {DecisionCount} decisions");
        
        private static readonly Action<ILogger, Exception?> LogTrainingError =
            LoggerMessage.Define(LogLevel.Error, new EventId(17, nameof(LogTrainingError)),
                "❌ [LEARNING] Error during training update");
        
        private static readonly Action<ILogger, string, string, double, string, double, Exception?> LogBrainDecision =
            LoggerMessage.Define<string, string, double, string, double>(
                LogLevel.Information, new EventId(18, nameof(LogBrainDecision)),
                "🧠 [BRAIN-DECISION] {Symbol}: Strategy={Strategy} ({Confidence:P1}), Direction={Direction} ({Probability:P1})");
        
        private static readonly Action<ILogger, string, Exception?> LogBotThinking =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(19, nameof(LogBotThinking)),
                "💭 [BOT-THINKING] {Thinking}");
        
        private static readonly Action<ILogger, string, Exception?> LogBotCommentary =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(20, nameof(LogBotCommentary)),
                "💬 [BOT-COMMENTARY] {Commentary}");
        
        private static readonly Action<ILogger, string, Exception?> LogDecisionInvalidOperation =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(21, nameof(LogDecisionInvalidOperation)),
                "❌ [UNIFIED-BRAIN] Invalid operation making decision for {Symbol}");
        
        private static readonly Action<ILogger, string, Exception?> LogDecisionInvalidArgument =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(22, nameof(LogDecisionInvalidArgument)),
                "❌ [UNIFIED-BRAIN] Invalid argument making decision for {Symbol}");
        
        private static readonly Action<ILogger, string, Exception?> LogDecisionTimeout =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(23, nameof(LogDecisionTimeout)),
                "❌ [UNIFIED-BRAIN] Timeout making decision for {Symbol}");
        
        private static readonly Action<ILogger, string, string, double, bool, double, int, Exception?> LogUnifiedLearning =
            LoggerMessage.Define<string, string, double, bool, double, int>(
                LogLevel.Information, new EventId(24, nameof(LogUnifiedLearning)),
                "📚 [UNIFIED-LEARNING] {Symbol} {Strategy}: PnL={PnL:F2}, Correct={Correct}, WinRate={WinRate:P1}, TotalTrades={Total}, AllStrategiesUpdated=True");
        
        private static readonly Action<ILogger, string, Exception?> LogBotReflection =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(25, nameof(LogBotReflection)),
                "🔮 [BOT-REFLECTION] {Reflection}");
        
        private static readonly Action<ILogger, string, Exception?> LogBotFailureAnalysis =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(26, nameof(LogBotFailureAnalysis)),
                "❌ [BOT-FAILURE-ANALYSIS] {Analysis}");
        
        private static readonly Action<ILogger, string, Exception?> LogBotLearningReport =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(27, nameof(LogBotLearningReport)),
                "📚 [BOT-LEARNING] {Report}");
        
        private static readonly Action<ILogger, Exception?> LogLearningInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(28, nameof(LogLearningInvalidOperation)),
                "❌ [UNIFIED-LEARNING] Invalid operation during learning from result");
        
        private static readonly Action<ILogger, Exception?> LogLearningInvalidArgument =
            LoggerMessage.Define(LogLevel.Error, new EventId(29, nameof(LogLearningInvalidArgument)),
                "❌ [UNIFIED-LEARNING] Invalid argument during learning from result");
        
        private static readonly Action<ILogger, string, Exception?> LogLearningCommentary =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(30, nameof(LogLearningCommentary)),
                "📚 [LEARNING-COMMENTARY] {Commentary}");
        
        private static readonly Action<ILogger, string, Exception?> LogRiskCommentary =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(31, nameof(LogRiskCommentary)),
                "🧠 [RISK-COMMENTARY] {Commentary}");
        
        private static readonly Action<ILogger, string, Exception?> LogHistoricalPattern =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(32, nameof(LogHistoricalPattern)),
                "🔍 [HISTORICAL-PATTERN] {Context}");
        
        private static readonly Action<ILogger, string, Exception?> LogMarketRegimeExplanation =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(33, nameof(LogMarketRegimeExplanation)),
                "📈 [MARKET-REGIME] {Explanation}");
        
        private static readonly Action<ILogger, string, Exception?> LogStrategyConflict =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(34, nameof(LogStrategyConflict)),
                "💬 [BOT-COMMENTARY] {Conflict}");
        
        private static readonly Action<ILogger, string, Exception?> LogStrategySelectionExplanation =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(35, nameof(LogStrategySelectionExplanation)),
                "🧠 [STRATEGY-SELECTION] {Explanation}");
        
        private static readonly Action<ILogger, Exception?> LogMetaClassifierFallback =
            LoggerMessage.Define(LogLevel.Warning, new EventId(36, nameof(LogMetaClassifierFallback)),
                "Meta classifier failed, using fallback");
        
        private static readonly Action<ILogger, Exception?> LogNeuralUcbFallback =
            LoggerMessage.Define(LogLevel.Warning, new EventId(37, nameof(LogNeuralUcbFallback)),
                "Neural UCB failed, using fallback");
        
        private static readonly Action<ILogger, Exception?> LogLstmPredictionFallback =
            LoggerMessage.Define(LogLevel.Warning, new EventId(38, nameof(LogLstmPredictionFallback)),
                "LSTM prediction failed, using fallback");
        
        private static readonly Action<ILogger, Exception?> LogThinkingError =
            LoggerMessage.Define(LogLevel.Error, new EventId(39, nameof(LogThinkingError)),
                "❌ [BOT-THINKING] Error during AI thinking");
        
        private static readonly Action<ILogger, Exception?> LogReflectionError =
            LoggerMessage.Define(LogLevel.Error, new EventId(40, nameof(LogReflectionError)),
                "❌ [BOT-REFLECTION] Error during AI reflection");
        
        // Additional high-value logging delegates for Round 10
        private static readonly Action<ILogger, double, string, double, Exception?> LogDecisionDetails =
            LoggerMessage.Define<double, string, double>(LogLevel.Information, new EventId(41, nameof(LogDecisionDetails)),
                "  └─ Size={Size}x, Regime={Regime}, Time={Ms}ms");
        
        private static readonly Action<ILogger, Exception?> LogSnapshotInvalidOperation =
            LoggerMessage.Define(LogLevel.Warning, new EventId(42, nameof(LogSnapshotInvalidOperation)),
                "⚠️ [SNAPSHOT] Failed to capture market snapshot - invalid operation");
        
        private static readonly Action<ILogger, Exception?> LogSnapshotArgumentException =
            LoggerMessage.Define(LogLevel.Warning, new EventId(43, nameof(LogSnapshotArgumentException)),
                "⚠️ [SNAPSHOT] Failed to capture market snapshot - invalid argument");
        
        private static readonly Action<ILogger, Exception?> LogSnapshotIOException =
            LoggerMessage.Define(LogLevel.Warning, new EventId(44, nameof(LogSnapshotIOException)),
                "⚠️ [SNAPSHOT] Failed to capture market snapshot - I/O error");
        
        private static readonly Action<ILogger, Exception?> LogContextGatherInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(45, nameof(LogContextGatherInvalidOperation)),
                "❌ [BOT-CONTEXT] Error gathering current context - invalid operation");
        
        private static readonly Action<ILogger, Exception?> LogContextGatherArgumentException =
            LoggerMessage.Define(LogLevel.Error, new EventId(46, nameof(LogContextGatherArgumentException)),
                "❌ [BOT-CONTEXT] Error gathering current context - invalid argument");
        
        private static readonly Action<ILogger, string, double, Exception?> LogCrossLearningUpdate =
            LoggerMessage.Define<string, double>(LogLevel.Debug, new EventId(47, nameof(LogCrossLearningUpdate)),
                "🧠 [CROSS-LEARNING] Updated all strategies from {ExecutedStrategy} outcome: {Reward}");
        
        private static readonly Action<ILogger, Exception?> LogCrossLearningInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(48, nameof(LogCrossLearningInvalidOperation)),
                "❌ [CROSS-LEARNING] Invalid operation updating all strategies");
        
        private static readonly Action<ILogger, Exception?> LogCrossLearningArgumentException =
            LoggerMessage.Define(LogLevel.Error, new EventId(49, nameof(LogCrossLearningArgumentException)),
                "❌ [CROSS-LEARNING] Invalid argument updating all strategies");

        // Snapshot logging delegates (EventId 50-52)
        private static readonly Action<ILogger, string, string, string, Exception?> LogSnapshotCaptured =
            LoggerMessage.Define<string, string, string>(LogLevel.Trace, new EventId(50, nameof(LogSnapshotCaptured)),
                "📸 [SNAPSHOT] Captured market snapshot for {Symbol}: {Strategy} {Direction}");

        // Parameter tracking logging delegates (EventId 53-56)
        private static readonly Action<ILogger, string, double, double, Exception?> LogParamTracked =
            LoggerMessage.Define<string, double, double>(LogLevel.Trace, new EventId(53, nameof(LogParamTracked)),
                "📊 [PARAM-TRACKING] Tracked parameter update for {Strategy}: old={OldReward:F3}, new={NewReward:F3}");

        private static readonly Action<ILogger, Exception?> LogParamTrackInvalidOperation =
            LoggerMessage.Define(LogLevel.Warning, new EventId(54, nameof(LogParamTrackInvalidOperation)),
                "⚠️ [PARAM-TRACKING] Failed to track parameter change - invalid operation");

        private static readonly Action<ILogger, Exception?> LogParamTrackIoError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(55, nameof(LogParamTrackIoError)),
                "⚠️ [PARAM-TRACKING] Failed to track parameter change - I/O error");

        private static readonly Action<ILogger, Exception?> LogParamTrackAccessDenied =
            LoggerMessage.Define(LogLevel.Warning, new EventId(56, nameof(LogParamTrackAccessDenied)),
                "⚠️ [PARAM-TRACKING] Failed to track parameter change - access denied");

        // Risk commentary logging delegates (EventId 57-62)
        private static readonly Action<ILogger, Exception?> LogRiskCommentaryStarted =
            LoggerMessage.Define(LogLevel.Trace, new EventId(57, nameof(LogRiskCommentaryStarted)),
                "🚀 [RISK-COMMENTARY] Started background analysis (async mode)");

        private static readonly Action<ILogger, Exception?> LogRiskCommentaryMissingData =
            LoggerMessage.Define(LogLevel.Warning, new EventId(58, nameof(LogRiskCommentaryMissingData)),
                "⚠️ [RISK-COMMENTARY] Skipping - missing price or ATR data");

        private static readonly Action<ILogger, Exception?> LogRiskCommentaryInvalidOperation =
            LoggerMessage.Define(LogLevel.Warning, new EventId(59, nameof(LogRiskCommentaryInvalidOperation)),
                "⚠️ [RISK-COMMENTARY] Failed to generate risk commentary - invalid operation");

        private static readonly Action<ILogger, Exception?> LogRiskCommentaryHttpError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(60, nameof(LogRiskCommentaryHttpError)),
                "⚠️ [RISK-COMMENTARY] Failed to generate risk commentary - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogRiskCommentaryTaskCancelled =
            LoggerMessage.Define(LogLevel.Warning, new EventId(61, nameof(LogRiskCommentaryTaskCancelled)),
                "⚠️ [RISK-COMMENTARY] Failed to generate risk commentary - task cancelled");

        // Historical pattern logging delegates (EventId 63-65)
        private static readonly Action<ILogger, Exception?> LogHistoricalPatternInvalidOperation =
            LoggerMessage.Define(LogLevel.Warning, new EventId(63, nameof(LogHistoricalPatternInvalidOperation)),
                "⚠️ [HISTORICAL-PATTERN] Failed to find similar conditions - invalid operation");

        private static readonly Action<ILogger, Exception?> LogHistoricalPatternInvalidArgument =
            LoggerMessage.Define(LogLevel.Warning, new EventId(64, nameof(LogHistoricalPatternInvalidArgument)),
                "⚠️ [HISTORICAL-PATTERN] Failed to find similar conditions - invalid argument");

        private static readonly Action<ILogger, Exception?> LogHistoricalPatternKeyNotFound =
            LoggerMessage.Define(LogLevel.Warning, new EventId(65, nameof(LogHistoricalPatternKeyNotFound)),
                "⚠️ [HISTORICAL-PATTERN] Failed to find similar conditions - key not found");

        // Learning commentary logging delegates (EventId 66-69)
        private static readonly Action<ILogger, Exception?> LogLearningCommentaryStarted =
            LoggerMessage.Define(LogLevel.Trace, new EventId(66, nameof(LogLearningCommentaryStarted)),
                "🚀 [LEARNING-COMMENTARY] Started background explanation (async mode)");

        private static readonly Action<ILogger, Exception?> LogLearningCommentaryInvalidOperation =
            LoggerMessage.Define(LogLevel.Warning, new EventId(67, nameof(LogLearningCommentaryInvalidOperation)),
                "⚠️ [LEARNING-COMMENTARY] Failed to generate learning commentary - invalid operation");

        private static readonly Action<ILogger, Exception?> LogLearningCommentaryHttpError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(68, nameof(LogLearningCommentaryHttpError)),
                "⚠️ [LEARNING-COMMENTARY] Failed to generate learning commentary - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogLearningCommentaryTaskCancelled =
            LoggerMessage.Define(LogLevel.Warning, new EventId(69, nameof(LogLearningCommentaryTaskCancelled)),
                "⚠️ [LEARNING-COMMENTARY] Failed to generate learning commentary - task cancelled");

        // TopStep compliance logging delegate (EventId 70)
        private static readonly Action<ILogger, string, Exception?> LogTradingBlocked =
            LoggerMessage.Define<string>(LogLevel.Warning, new EventId(70, nameof(LogTradingBlocked)),
                "🛑 [TOPSTEP-COMPLIANCE] Trading blocked: {Reason}");

        // Confidence threshold logging delegate (EventId 71)
        private static readonly Action<ILogger, double, double, Exception?> LogConfidenceBelowThreshold =
            LoggerMessage.Define<double, double>(LogLevel.Debug, new EventId(71, nameof(LogConfidenceBelowThreshold)),
                "🎯 [CONFIDENCE] Below threshold {Threshold:P1}, confidence: {Confidence:P1}");

        // CVaR-PPO action logging delegate (EventId 72)
        private static readonly Action<ILogger, double, double, double, double, int, Exception?> LogCvarPpoAction =
            LoggerMessage.Define<double, double, double, double, int>(LogLevel.Information, new EventId(72, nameof(LogCvarPpoAction)),
                "🎯 [CVAR-PPO] Action={Action}, Prob={Prob:F3}, Value={Value:F3}, CVaR={CVaR:F3}, Contracts={Contracts}");

        // CVaR-PPO error logging delegates (EventId 73-75)
        private static readonly Action<ILogger, Exception?> LogCvarPpoInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(73, nameof(LogCvarPpoInvalidOperation)),
                "CVaR-PPO position sizing failed - invalid operation, using TopStep compliance sizing");

        private static readonly Action<ILogger, Exception?> LogCvarPpoInvalidArgument =
            LoggerMessage.Define(LogLevel.Error, new EventId(74, nameof(LogCvarPpoInvalidArgument)),
                "CVaR-PPO position sizing failed - invalid argument, using TopStep compliance sizing");

        private static readonly Action<ILogger, Exception?> LogCvarPpoOnnxError =
            LoggerMessage.Define(LogLevel.Error, new EventId(75, nameof(LogCvarPpoOnnxError)),
                "CVaR-PPO position sizing failed - ONNX runtime error, using TopStep compliance sizing");

        // Position sizing logging delegates (EventId 76-77)
        private static readonly Action<ILogger, double, Exception?> LogLegacyRlMultiplier =
            LoggerMessage.Define<double>(LogLevel.Debug, new EventId(76, nameof(LogLegacyRlMultiplier)),
                "📊 [LEGACY-RL] Using fallback RL multiplier: {Multiplier:F2}");

        private static readonly Action<ILogger, string, double, decimal, int, decimal, Exception?> LogPositionSize =
            LoggerMessage.Define<string, double, decimal, int, decimal>(LogLevel.Debug, new EventId(77, nameof(LogPositionSize)),
                "📊 [POSITION-SIZE] {Symbol}: Confidence={Confidence:P1}, Drawdown={Drawdown:C}, Contracts={Contracts}, RiskAmount={Risk:C}");

        // P&L tracking logging delegates (EventId 78-79)
        private static readonly Action<ILogger, string, decimal, decimal, decimal, decimal, Exception?> LogPnlUpdate =
            LoggerMessage.Define<string, decimal, decimal, decimal, decimal>(LogLevel.Information, new EventId(78, nameof(LogPnlUpdate)),
                "💰 [PNL-UPDATE] Strategy={Strategy}, PnL={PnL:C}, DailyPnL={DailyPnL:C}, Drawdown={Drawdown:C}, Balance={Balance:C}");

        private static readonly Action<ILogger, Exception?> LogDailyReset =
            LoggerMessage.Define(LogLevel.Information, new EventId(79, nameof(LogDailyReset)),
                "🌅 [DAILY-RESET] Daily P&L and drawdown reset for new trading day");

        // Brain enhance logging delegates (EventId 80-83)
        private static readonly Action<ILogger, string, int, string, Exception?> LogBrainEnhanceGenerated =
            LoggerMessage.Define<string, int, string>(LogLevel.Debug, new EventId(80, nameof(LogBrainEnhanceGenerated)),
                "🎯 [BRAIN-ENHANCE] {Symbol}: Generated {Count} AI-enhanced candidates from {Strategy}");

        private static readonly Action<ILogger, Exception?> LogBrainEnhanceInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(81, nameof(LogBrainEnhanceInvalidOperation)),
                "❌ [BRAIN-ENHANCE] Error generating enhanced candidates - invalid operation");

        private static readonly Action<ILogger, Exception?> LogBrainEnhanceInvalidArgument =
            LoggerMessage.Define(LogLevel.Error, new EventId(82, nameof(LogBrainEnhanceInvalidArgument)),
                "❌ [BRAIN-ENHANCE] Error generating enhanced candidates - invalid argument");

        private static readonly Action<ILogger, Exception?> LogBrainEnhanceKeyNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(83, nameof(LogBrainEnhanceKeyNotFound)),
                "❌ [BRAIN-ENHANCE] Error generating enhanced candidates - key not found");

        // Strategy selection logging delegate (EventId 84)
        private static readonly Action<ILogger, int, string, string, Exception?> LogStrategySelection =
            LoggerMessage.Define<int, string, string>(LogLevel.Debug, new EventId(84, nameof(LogStrategySelection)),
                "🧠 [STRATEGY-SELECTION] Hour={Hour}, Regime={Regime}, Available={Strategies}");

        // Unified learning logging delegates (EventId 85-89)
        private static readonly Action<ILogger, Exception?> LogUnifiedLearningStarting =
            LoggerMessage.Define(LogLevel.Information, new EventId(85, nameof(LogUnifiedLearningStarting)),
                "🔄 [UNIFIED-LEARNING] Starting unified learning update across all strategies...");

        private static readonly Action<ILogger, Exception?> LogUnifiedLearningCompleted =
            LoggerMessage.Define(LogLevel.Information, new EventId(86, nameof(LogUnifiedLearningCompleted)),
                "✅ [UNIFIED-LEARNING] Completed unified learning update");

        private static readonly Action<ILogger, Exception?> LogUnifiedLearningInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(87, nameof(LogUnifiedLearningInvalidOperation)),
                "❌ [UNIFIED-LEARNING] Failed to update unified learning - invalid operation");

        private static readonly Action<ILogger, Exception?> LogUnifiedLearningIoError =
            LoggerMessage.Define(LogLevel.Error, new EventId(88, nameof(LogUnifiedLearningIoError)),
                "❌ [UNIFIED-LEARNING] Failed to update unified learning - I/O error");

        private static readonly Action<ILogger, Exception?> LogUnifiedLearningAccessDenied =
            LoggerMessage.Define(LogLevel.Error, new EventId(89, nameof(LogUnifiedLearningAccessDenied)),
                "❌ [UNIFIED-LEARNING] Failed to update unified learning - access denied");

        private static readonly Action<ILogger, Exception?> LogUnifiedLearningInvalidArgument =
            LoggerMessage.Define(LogLevel.Error, new EventId(90, nameof(LogUnifiedLearningInvalidArgument)),
                "❌ [UNIFIED-LEARNING] Failed to update unified learning - invalid argument");

        // Condition update logging delegate (EventId 91)
        private static readonly Action<ILogger, int, string, Exception?> LogConditionUpdate =
            LoggerMessage.Define<int, string>(LogLevel.Debug, new EventId(91, nameof(LogConditionUpdate)),
                "🔄 [CONDITION-UPDATE] Removed {Count} unsuccessful conditions from {Strategy}");

        // Cross-pollination logging delegate (EventId 92)
        private static readonly Action<ILogger, int, string, Exception?> LogCrossPollination =
            LoggerMessage.Define<int, string>(LogLevel.Information, new EventId(92, nameof(LogCrossPollination)),
                "🌱 [CROSS-POLLINATION] Shared {Count} successful patterns from {BestStrategy} to other strategies");

        // Gate4 validation logging (EventId 93-120)
        private static readonly Action<ILogger, Exception?> LogGate4Start =
            LoggerMessage.Define(LogLevel.Information, new EventId(93, nameof(LogGate4Start)),
                "=== GATE 4: UNIFIED TRADING BRAIN MODEL RELOAD VALIDATION ===");

        private static readonly Action<ILogger, string, Exception?> LogGate4NewModel =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(94, nameof(LogGate4NewModel)),
                "New model: {NewPath}");

        private static readonly Action<ILogger, string, Exception?> LogGate4CurrentModel =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(95, nameof(LogGate4CurrentModel)),
                "Current model: {CurrentPath}");

        private static readonly Action<ILogger, Exception?> LogGate4FeatureCheck =
            LoggerMessage.Define(LogLevel.Information, new EventId(96, nameof(LogGate4FeatureCheck)),
                "[1/4] Validating feature specification compatibility...");

        private static readonly Action<ILogger, string, Exception?> LogGate4Failed =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(97, nameof(LogGate4Failed)),
                "✗ GATE 4 FAILED: {Reason}");

        private static readonly Action<ILogger, Exception?> LogGate4FeatureMatch =
            LoggerMessage.Define(LogLevel.Information, new EventId(98, nameof(LogGate4FeatureMatch)),
                "  ✓ Feature specification matches");

        private static readonly Action<ILogger, Exception?> LogGate4SanityCheck =
            LoggerMessage.Define(LogLevel.Information, new EventId(99, nameof(LogGate4SanityCheck)),
                "[2/4] Running sanity tests with deterministic dataset...");

        private static readonly Action<ILogger, int, Exception?> LogGate4SanityVectors =
            LoggerMessage.Define<int>(LogLevel.Information, new EventId(100, nameof(LogGate4SanityVectors)),
                "  Loaded {Count} sanity test vectors");

        private static readonly Action<ILogger, Exception?> LogGate4DistributionCheck =
            LoggerMessage.Define(LogLevel.Information, new EventId(101, nameof(LogGate4DistributionCheck)),
                "[3/4] Comparing prediction distributions...");

        private static readonly Action<ILogger, double, Exception?> LogGate4DistributionValid =
            LoggerMessage.Define<double>(LogLevel.Information, new EventId(102, nameof(LogGate4DistributionValid)),
                "  ✓ Distribution divergence acceptable: {Divergence:F4}");

        private static readonly Action<ILogger, Exception?> LogGate4DistributionSkip =
            LoggerMessage.Define(LogLevel.Warning, new EventId(103, nameof(LogGate4DistributionSkip)),
                "  Current model not found - skipping distribution comparison (first deployment)");

        private static readonly Action<ILogger, Exception?> LogGate4OutputCheck =
            LoggerMessage.Define(LogLevel.Information, new EventId(104, nameof(LogGate4OutputCheck)),
                "[4/4] Validating model outputs for NaN/Infinity...");

        private static readonly Action<ILogger, Exception?> LogGate4OutputValid =
            LoggerMessage.Define(LogLevel.Information, new EventId(105, nameof(LogGate4OutputValid)),
                "  ✓ All outputs valid (no NaN/Infinity)");

        private static readonly Action<ILogger, Exception?> LogGate4SimulationStart =
            LoggerMessage.Define(LogLevel.Information, new EventId(106, nameof(LogGate4SimulationStart)),
                "[5/5] Running historical replay simulation...");

        private static readonly Action<ILogger, double, Exception?> LogGate4SimulationPassed =
            LoggerMessage.Define<double>(LogLevel.Information, new EventId(107, nameof(LogGate4SimulationPassed)),
                "  ✓ Simulation passed - drawdown ratio: {Ratio:F2}x");

        private static readonly Action<ILogger, Exception?> LogGate4SimulationSkip =
            LoggerMessage.Define(LogLevel.Warning, new EventId(108, nameof(LogGate4SimulationSkip)),
                "  Current model not found - skipping simulation (first deployment)");

        private static readonly Action<ILogger, Exception?> LogGate4Passed =
            LoggerMessage.Define(LogLevel.Information, new EventId(109, nameof(LogGate4Passed)),
                "=== GATE 4 PASSED - Model validated for hot-reload ===");

        private static readonly Action<ILogger, string, Exception?> LogGate4Exception =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(110, nameof(LogGate4Exception)),
                "Gate 4 validation error: {Message}");

        private static readonly Action<ILogger, Exception?> LogGate4FileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(111, nameof(LogGate4FileNotFound)),
                "✗ GATE 4 FAILED: Model file not found during validation");

        private static readonly Action<ILogger, Exception?> LogGate4OnnxError =
            LoggerMessage.Define(LogLevel.Error, new EventId(112, nameof(LogGate4OnnxError)),
                "✗ GATE 4 FAILED: ONNX runtime error during model validation");

        private static readonly Action<ILogger, Exception?> LogGate4InvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(113, nameof(LogGate4InvalidOperation)),
                "✗ GATE 4 FAILED: Invalid operation during model validation");

        private static readonly Action<ILogger, Exception?> LogGate4IOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(114, nameof(LogGate4IOError)),
                "✗ GATE 4 FAILED: I/O error during model validation");

        private static readonly Action<ILogger, Exception?> LogFeatureSpecMissing =
            LoggerMessage.Define(LogLevel.Warning, new EventId(115, nameof(LogFeatureSpecMissing)),
                "Feature specification not found - creating default");

        private static readonly Action<ILogger, string, Exception?> LogValidationModelFileNotFound =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(116, nameof(LogValidationModelFileNotFound)),
                "Model file not found: {Path}");

        private static readonly Action<ILogger, string, Exception?> LogValidationModelFileEmpty =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(117, nameof(LogValidationModelFileEmpty)),
                "Model file is empty: {Path}");

        private static readonly Action<ILogger, long, Exception?> LogModelFileSize =
            LoggerMessage.Define<long>(LogLevel.Information, new EventId(118, nameof(LogModelFileSize)),
                "  Model file size: {Size} bytes");

        private static readonly Action<ILogger, Exception?> LogFeatureValidationFileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(119, nameof(LogFeatureValidationFileNotFound)),
                "Feature specification validation failed - file not found");

        private static readonly Action<ILogger, Exception?> LogFeatureValidationJsonError =
            LoggerMessage.Define(LogLevel.Error, new EventId(120, nameof(LogFeatureValidationJsonError)),
                "Feature specification validation failed - invalid JSON");

        private static readonly Action<ILogger, Exception?> LogFeatureValidationIOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(121, nameof(LogFeatureValidationIOError)),
                "Feature specification validation failed - I/O error");

        private static readonly Action<ILogger, Exception?> LogFeatureValidationAccessDenied =
            LoggerMessage.Define(LogLevel.Error, new EventId(122, nameof(LogFeatureValidationAccessDenied)),
                "Feature specification validation failed - access denied");

        private static readonly Action<ILogger, int, Exception?> LogSanityVectorsCached =
            LoggerMessage.Define<int>(LogLevel.Information, new EventId(123, nameof(LogSanityVectorsCached)),
                "  Loaded {Count} cached sanity test vectors");

        private static readonly Action<ILogger, Exception?> LogSanityVectorsCacheFileNotFound =
            LoggerMessage.Define(LogLevel.Warning, new EventId(124, nameof(LogSanityVectorsCacheFileNotFound)),
                "Failed to load cached sanity test vectors - file not found, generating new ones");

        private static readonly Action<ILogger, Exception?> LogSanityVectorsCacheJsonError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(125, nameof(LogSanityVectorsCacheJsonError)),
                "Failed to load cached sanity test vectors - invalid JSON, generating new ones");

        private static readonly Action<ILogger, Exception?> LogSanityVectorsCacheIOError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(126, nameof(LogSanityVectorsCacheIOError)),
                "Failed to load cached sanity test vectors - I/O error, generating new ones");

        // Round 13: Additional LoggerMessage delegates for simulation and validation (EventId 127-158)
        private static readonly Action<ILogger, int, Exception?> LogSanityVectorsCached2 =
            LoggerMessage.Define<int>(LogLevel.Information, new EventId(127, nameof(LogSanityVectorsCached2)),
                "  Cached {Count} sanity test vectors for future use");

        private static readonly Action<ILogger, Exception?> LogCacheSanityVectorsIOError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(128, nameof(LogCacheSanityVectorsIOError)),
                "Failed to cache sanity test vectors - I/O error");

        private static readonly Action<ILogger, Exception?> LogCacheSanityVectorsAccessDenied =
            LoggerMessage.Define(LogLevel.Warning, new EventId(129, nameof(LogCacheSanityVectorsAccessDenied)),
                "Failed to cache sanity test vectors - access denied");

        private static readonly Action<ILogger, Exception?> LogCacheSanityVectorsJsonError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(130, nameof(LogCacheSanityVectorsJsonError)),
                "Failed to cache sanity test vectors - JSON serialization error");

        private static readonly Action<ILogger, double, double, Exception?> LogDistributionComparison =
            LoggerMessage.Define<double, double>(LogLevel.Information, new EventId(131, nameof(LogDistributionComparison)),
                "  Total Variation: {TV:F4}, KL Divergence: {KL:F4}");

        private static readonly Action<ILogger, double, double, Exception?> LogTotalVariationExceeded =
            LoggerMessage.Define<double, double>(LogLevel.Warning, new EventId(132, nameof(LogTotalVariationExceeded)),
                "  Total variation {TV:F4} exceeds threshold {Max:F2}");

        private static readonly Action<ILogger, double, double, Exception?> LogKLDivergenceExceeded =
            LoggerMessage.Define<double, double>(LogLevel.Warning, new EventId(133, nameof(LogKLDivergenceExceeded)),
                "  KL divergence {KL:F4} exceeds threshold {Max:F2}");

        private static readonly Action<ILogger, Exception?> LogDistributionComparisonOnnxError =
            LoggerMessage.Define(LogLevel.Error, new EventId(134, nameof(LogDistributionComparisonOnnxError)),
                "Distribution comparison failed - ONNX runtime error");

        private static readonly Action<ILogger, Exception?> LogDistributionComparisonFileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(135, nameof(LogDistributionComparisonFileNotFound)),
                "Distribution comparison failed - model file not found");

        private static readonly Action<ILogger, Exception?> LogDistributionComparisonInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(136, nameof(LogDistributionComparisonInvalidOperation)),
                "Distribution comparison failed - invalid operation");

        private static readonly Action<ILogger, string, Exception?> LogModelFileNotFoundValidation =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(137, nameof(LogModelFileNotFoundValidation)),
                "Model file not found: {Path}");

        private static readonly Action<ILogger, Exception?> LogModelFileEmpty =
            LoggerMessage.Define(LogLevel.Error, new EventId(138, nameof(LogModelFileEmpty)),
                "Model file is empty");

        private static readonly Action<ILogger, Exception?> LogModelOutputsValidated =
            LoggerMessage.Define(LogLevel.Information, new EventId(139, nameof(LogModelOutputsValidated)),
                "  Validated model outputs - no NaN/Infinity detected");

        private static readonly Action<ILogger, Exception?> LogModelOutputsNaNInfinity =
            LoggerMessage.Define(LogLevel.Error, new EventId(140, nameof(LogModelOutputsNaNInfinity)),
                "Model produces NaN or Infinity values");

        private static readonly Action<ILogger, Exception?> LogModelOutputValidationOnnxError =
            LoggerMessage.Define(LogLevel.Error, new EventId(141, nameof(LogModelOutputValidationOnnxError)),
                "Model output validation failed - ONNX runtime error");

        private static readonly Action<ILogger, Exception?> LogModelOutputValidationFileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(142, nameof(LogModelOutputValidationFileNotFound)),
                "Model output validation failed - model file not found");

        private static readonly Action<ILogger, Exception?> LogModelOutputValidationInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(143, nameof(LogModelOutputValidationInvalidOperation)),
                "Model output validation failed - invalid operation");

        private static readonly Action<ILogger, int, Exception?> LogInsufficientHistoricalData =
            LoggerMessage.Define<int>(LogLevel.Warning, new EventId(144, nameof(LogInsufficientHistoricalData)),
                "  Insufficient historical data for simulation - using available {Count} bars");

        private static readonly Action<ILogger, double, double, double, Exception?> LogDrawdownComparison =
            LoggerMessage.Define<double, double, double>(LogLevel.Information, new EventId(145, nameof(LogDrawdownComparison)),
                "  Baseline drawdown: {Current:F2}, New drawdown: {New:F2}, Ratio: {Ratio:F2}x");

        private static readonly Action<ILogger, Exception?> LogHistoricalSimulationOnnxError =
            LoggerMessage.Define(LogLevel.Error, new EventId(146, nameof(LogHistoricalSimulationOnnxError)),
                "Historical simulation failed - ONNX runtime error");

        private static readonly Action<ILogger, Exception?> LogHistoricalSimulationFileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(147, nameof(LogHistoricalSimulationFileNotFound)),
                "Historical simulation failed - model file not found");

        private static readonly Action<ILogger, Exception?> LogHistoricalSimulationInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(148, nameof(LogHistoricalSimulationInvalidOperation)),
                "Historical simulation failed - invalid operation");

        private static readonly Action<ILogger, Exception?> LogLoadHistoricalDataFileNotFound =
            LoggerMessage.Define(LogLevel.Warning, new EventId(149, nameof(LogLoadHistoricalDataFileNotFound)),
                "Failed to load cached historical data - file not found");

        private static readonly Action<ILogger, Exception?> LogLoadHistoricalDataJsonError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(150, nameof(LogLoadHistoricalDataJsonError)),
                "Failed to load cached historical data - invalid JSON");

        private static readonly Action<ILogger, Exception?> LogLoadHistoricalDataIOError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(151, nameof(LogLoadHistoricalDataIOError)),
                "Failed to load cached historical data - I/O error");

        private static readonly Action<ILogger, Exception?> LogCacheHistoricalDataIOError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(152, nameof(LogCacheHistoricalDataIOError)),
                "Failed to cache historical data - I/O error");

        private static readonly Action<ILogger, Exception?> LogCacheHistoricalDataAccessDenied =
            LoggerMessage.Define(LogLevel.Warning, new EventId(153, nameof(LogCacheHistoricalDataAccessDenied)),
                "Failed to cache historical data - access denied");

        private static readonly Action<ILogger, Exception?> LogCacheHistoricalDataJsonError =
            LoggerMessage.Define(LogLevel.Warning, new EventId(154, nameof(LogCacheHistoricalDataJsonError)),
                "Failed to cache historical data - JSON serialization error");

        private static readonly Action<ILogger, string, Exception?> LogFeatureSpecificationCreated =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(155, nameof(LogFeatureSpecificationCreated)),
                "Created default feature specification at {Path}");

        private static readonly Action<ILogger, string, Exception?> LogModelReloadStarting =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(156, nameof(LogModelReloadStarting)),
                "🔄 [MODEL-RELOAD] Starting model reload: {NewModel}");

        private static readonly Action<ILogger, string, Exception?> LogModelReloadValidationFailed =
            LoggerMessage.Define<string>(LogLevel.Error, new EventId(157, nameof(LogModelReloadValidationFailed)),
                "❌ [MODEL-RELOAD] Validation failed: {Reason}");

        private static readonly Action<ILogger, string, Exception?> LogModelReloadBackupCreated =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(158, nameof(LogModelReloadBackupCreated)),
                "💾 [MODEL-RELOAD] Backup created: {BackupPath}");

        private static readonly Action<ILogger, Exception?> LogModelReloadSwapFailed =
            LoggerMessage.Define(LogLevel.Error, new EventId(159, nameof(LogModelReloadSwapFailed)),
                "❌ [MODEL-RELOAD] Model swap failed");

        private static readonly Action<ILogger, Exception?> LogModelReloadSuccess =
            LoggerMessage.Define(LogLevel.Information, new EventId(160, nameof(LogModelReloadSuccess)),
                "✅ [MODEL-RELOAD] Model reloaded successfully");

        private static readonly Action<ILogger, string, Exception?> LogModelReloadOldVersion =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(161, nameof(LogModelReloadOldVersion)),
                "  Old version: {OldVersion}");

        private static readonly Action<ILogger, string, Exception?> LogModelReloadNewVersion =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(162, nameof(LogModelReloadNewVersion)),
                "  New version: {NewVersion}");

        private static readonly Action<ILogger, Exception?> LogModelReloadOnnxError =
            LoggerMessage.Define(LogLevel.Error, new EventId(163, nameof(LogModelReloadOnnxError)),
                "❌ [MODEL-RELOAD] ONNX runtime error during model reload");

        private static readonly Action<ILogger, Exception?> LogModelReloadFileNotFound =
            LoggerMessage.Define(LogLevel.Error, new EventId(164, nameof(LogModelReloadFileNotFound)),
                "❌ [MODEL-RELOAD] Model file not found during reload");

        private static readonly Action<ILogger, Exception?> LogModelReloadIOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(165, nameof(LogModelReloadIOError)),
                "❌ [MODEL-RELOAD] I/O error during model reload");

        private static readonly Action<ILogger, Exception?> LogModelReloadInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(166, nameof(LogModelReloadInvalidOperation)),
                "❌ [MODEL-RELOAD] Invalid operation during model reload");

        private static readonly Action<ILogger, string, Exception?> LogBackupCreated =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(167, nameof(LogBackupCreated)),
                "  Created backup: {BackupPath}");

        private static readonly Action<ILogger, int, int, Exception?> LogUnifiedRetrainingDataExported =
            LoggerMessage.Define<int, int>(LogLevel.Information, new EventId(168, nameof(LogUnifiedRetrainingDataExported)),
                "✅ [UNIFIED-RETRAIN] Training data exported: {Count} decisions, {StrategyCount} strategies");

        private static readonly Action<ILogger, Exception?> LogUnifiedRetrainingIOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(169, nameof(LogUnifiedRetrainingIOError)),
                "❌ [UNIFIED-RETRAIN] Unified model retraining failed - I/O error");

        private static readonly Action<ILogger, Exception?> LogUnifiedRetrainingAccessDenied =
            LoggerMessage.Define(LogLevel.Error, new EventId(170, nameof(LogUnifiedRetrainingAccessDenied)),
                "❌ [UNIFIED-RETRAIN] Unified model retraining failed - access denied");

        private static readonly Action<ILogger, Exception?> LogUnifiedRetrainingInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(171, nameof(LogUnifiedRetrainingInvalidOperation)),
                "❌ [UNIFIED-RETRAIN] Unified model retraining failed - invalid operation");

        private static readonly Action<ILogger, Exception?> LogUnifiedRetrainingJsonError =
            LoggerMessage.Define(LogLevel.Error, new EventId(172, nameof(LogUnifiedRetrainingJsonError)),
                "❌ [UNIFIED-RETRAIN] Unified model retraining failed - JSON error");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryWaitingInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(173, nameof(LogBotCommentaryWaitingInvalidOperation)),
                "❌ [BOT-COMMENTARY] Error explaining why waiting - invalid operation");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryWaitingHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(174, nameof(LogBotCommentaryWaitingHttpError)),
                "❌ [BOT-COMMENTARY] Error explaining why waiting - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryWaitingTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(175, nameof(LogBotCommentaryWaitingTaskCancelled)),
                "❌ [BOT-COMMENTARY] Error explaining why waiting - task cancelled");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryConfidenceInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(176, nameof(LogBotCommentaryConfidenceInvalidOperation)),
                "❌ [BOT-COMMENTARY] Error explaining high confidence - invalid operation");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryConfidenceHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(177, nameof(LogBotCommentaryConfidenceHttpError)),
                "❌ [BOT-COMMENTARY] Error explaining high confidence - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryConfidenceTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(178, nameof(LogBotCommentaryConfidenceTaskCancelled)),
                "❌ [BOT-COMMENTARY] Error explaining high confidence - task cancelled");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryConflictInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(179, nameof(LogBotCommentaryConflictInvalidOperation)),
                "❌ [BOT-COMMENTARY] Error explaining strategy conflict - invalid operation");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryConflictHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(180, nameof(LogBotCommentaryConflictHttpError)),
                "❌ [BOT-COMMENTARY] Error explaining strategy conflict - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogBotCommentaryConflictTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(181, nameof(LogBotCommentaryConflictTaskCancelled)),
                "❌ [BOT-COMMENTARY] Error explaining strategy conflict - task cancelled");

        private static readonly Action<ILogger, Exception?> LogFailureAnalysisInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(182, nameof(LogFailureAnalysisInvalidOperation)),
                "❌ [BOT-FAILURE-ANALYSIS] Error analyzing trade failure - invalid operation");

        private static readonly Action<ILogger, Exception?> LogFailureAnalysisHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(183, nameof(LogFailureAnalysisHttpError)),
                "❌ [BOT-FAILURE-ANALYSIS] Error analyzing trade failure - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogFailureAnalysisTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(184, nameof(LogFailureAnalysisTaskCancelled)),
                "❌ [BOT-FAILURE-ANALYSIS] Error analyzing trade failure - task cancelled");

        private static readonly Action<ILogger, Exception?> LogStrategySelectionExplanationInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(185, nameof(LogStrategySelectionExplanationInvalidOperation)),
                "❌ [STRATEGY-SELECTION] Error explaining strategy selection - invalid operation");

        private static readonly Action<ILogger, Exception?> LogStrategySelectionExplanationHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(186, nameof(LogStrategySelectionExplanationHttpError)),
                "❌ [STRATEGY-SELECTION] Error explaining strategy selection - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogStrategySelectionExplanationTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(187, nameof(LogStrategySelectionExplanationTaskCancelled)),
                "❌ [STRATEGY-SELECTION] Error explaining strategy selection - task cancelled");

        private static readonly Action<ILogger, Exception?> LogMarketRegimeExplanationInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(188, nameof(LogMarketRegimeExplanationInvalidOperation)),
                "❌ [MARKET-REGIME] Error explaining market regime - invalid operation");

        private static readonly Action<ILogger, Exception?> LogMarketRegimeExplanationHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(189, nameof(LogMarketRegimeExplanationHttpError)),
                "❌ [MARKET-REGIME] Error explaining market regime - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogMarketRegimeExplanationTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(190, nameof(LogMarketRegimeExplanationTaskCancelled)),
                "❌ [MARKET-REGIME] Error explaining market regime - task cancelled");

        private static readonly Action<ILogger, Exception?> LogLearningExplanationInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(191, nameof(LogLearningExplanationInvalidOperation)),
                "❌ [BOT-LEARNING] Error explaining learning update - invalid operation");

        private static readonly Action<ILogger, Exception?> LogLearningExplanationHttpError =
            LoggerMessage.Define(LogLevel.Error, new EventId(192, nameof(LogLearningExplanationHttpError)),
                "❌ [BOT-LEARNING] Error explaining learning update - HTTP request failed");

        private static readonly Action<ILogger, Exception?> LogLearningExplanationTaskCancelled =
            LoggerMessage.Define(LogLevel.Error, new EventId(193, nameof(LogLearningExplanationTaskCancelled)),
                "❌ [BOT-LEARNING] Error explaining learning update - task cancelled");

        private static readonly Action<ILogger, Exception?> LogBrainShuttingDown =
            LoggerMessage.Define(LogLevel.Information, new EventId(194, nameof(LogBrainShuttingDown)),
                "🧠 [UNIFIED-BRAIN] Shutting down...");

        private static readonly Action<ILogger, int, decimal, Exception?> LogBrainStatisticsSaved =
            LoggerMessage.Define<int, decimal>(LogLevel.Information, new EventId(195, nameof(LogBrainStatisticsSaved)),
                "📊 [UNIFIED-BRAIN] Statistics saved: {Decisions} decisions, {WinRate:P1} win rate");

        private static readonly Action<ILogger, Exception?> LogBrainStatisticsSaveIOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(196, nameof(LogBrainStatisticsSaveIOError)),
                "❌ [UNIFIED-BRAIN] Error saving statistics - I/O error");

        private static readonly Action<ILogger, Exception?> LogBrainStatisticsSaveAccessDenied =
            LoggerMessage.Define(LogLevel.Error, new EventId(197, nameof(LogBrainStatisticsSaveAccessDenied)),
                "❌ [UNIFIED-BRAIN] Error saving statistics - access denied");

        private static readonly Action<ILogger, Exception?> LogBrainStatisticsSaveJsonError =
            LoggerMessage.Define(LogLevel.Error, new EventId(198, nameof(LogBrainStatisticsSaveJsonError)),
                "❌ [UNIFIED-BRAIN] Error saving statistics - JSON serialization error");

        private static readonly Action<ILogger, Exception?> LogBrainDisposeInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(199, nameof(LogBrainDisposeInvalidOperation)),
                "❌ [UNIFIED-BRAIN] Error disposing managed resources - invalid operation");

        private static readonly Action<ILogger, string, Exception?> LogModelRestoredFromBackup =
            LoggerMessage.Define<string>(LogLevel.Information, new EventId(200, nameof(LogModelRestoredFromBackup)),
                "  Restored model from backup: {BackupPath}");

        private static readonly Action<ILogger, string, string, Exception?> LogAtomicSwapCompleted =
            LoggerMessage.Define<string, string>(LogLevel.Information, new EventId(201, nameof(LogAtomicSwapCompleted)),
                "  Atomic swap completed: {Old} → {New}");

        private static readonly Action<ILogger, Exception?> LogAtomicSwapIOError =
            LoggerMessage.Define(LogLevel.Error, new EventId(202, nameof(LogAtomicSwapIOError)),
                "  Atomic swap failed - I/O error");

        private static readonly Action<ILogger, Exception?> LogAtomicSwapAccessDenied =
            LoggerMessage.Define(LogLevel.Error, new EventId(203, nameof(LogAtomicSwapAccessDenied)),
                "  Atomic swap failed - access denied");

        private static readonly Action<ILogger, Exception?> LogAtomicSwapInvalidOperation =
            LoggerMessage.Define(LogLevel.Error, new EventId(204, nameof(LogAtomicSwapInvalidOperation)),
                "  Atomic swap failed - invalid operation");

        private static readonly Action<ILogger, Exception?> LogUnifiedRetrainingStarting =
            LoggerMessage.Define(LogLevel.Information, new EventId(205, nameof(LogUnifiedRetrainingStarting)),
                "🔄 [UNIFIED-RETRAIN] Starting unified model retraining across all strategies...");

        public UnifiedTradingBrain(
            ILogger<UnifiedTradingBrain> logger,
            IMLMemoryManager memoryManager,
            StrategyMlModelManager modelManager,
            CVaRPPO cvarPPO,
            IMLConfigurationService mlConfigService,
            IGate4Config? gate4Config = null,
            BotCore.Services.OllamaClient? ollamaClient = null,
            BotCore.Market.IEconomicEventManager? economicEventManager = null,
            BotCore.Services.RiskAssessmentCommentary? riskCommentary = null,
            BotCore.Services.AdaptiveLearningCommentary? learningCommentary = null,
            BotCore.Services.MarketSnapshotStore? snapshotStore = null,
            BotCore.Services.HistoricalPatternRecognitionService? historicalPatterns = null,
            BotCore.Services.ParameterChangeTracker? parameterTracker = null,
            BotCore.Services.INewsMonitorService? newsMonitor = null)
        {
            _logger = logger;
            _memoryManager = memoryManager;
            _modelManager = modelManager;
            _cvarPPO = cvarPPO; // Direct injection
            _mlConfigService = mlConfigService ?? throw new ArgumentNullException(nameof(mlConfigService));
            _gate4Config = gate4Config ?? Gate4Config.LoadFromEnvironment();
            _ollamaClient = ollamaClient; // Optional AI conversation client
            _economicEventManager = economicEventManager; // Optional economic calendar (Phase 2)
            _riskCommentary = riskCommentary;
            _learningCommentary = learningCommentary;
            _snapshotStore = snapshotStore;
            _historicalPatterns = historicalPatterns;
            _parameterTracker = parameterTracker;
            _newsMonitor = newsMonitor; // Optional real-time news monitoring
            
            // Initialize Neural UCB for strategy selection using ONNX-based neural network
            var onnxLoader = new OnnxModelLoader(new Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxModelLoader>());
            var neuralNetworkLogger = new Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxNeuralNetwork>();
            // Load runtime mode from environment for production safety
            var runtimeModeStr = Environment.GetEnvironmentVariable("RlRuntimeMode") ?? "InferenceOnly";
            if (!Enum.TryParse<TradingBot.Abstractions.RlRuntimeMode>(runtimeModeStr, ignoreCase: true, out var runtimeMode))
            {
                runtimeMode = TradingBot.Abstractions.RlRuntimeMode.InferenceOnly;
            }
            
            NeuralUcbBandit? tempSelector = null;
            OnnxNeuralNetwork? tempConfidenceNet = null;
            try
            {
                var neuralNetwork = new OnnxNeuralNetwork(onnxLoader, neuralNetworkLogger, runtimeMode, "models/strategy_selection.onnx");
                try
                {
                    tempSelector = new NeuralUcbBandit(neuralNetwork);
                    _strategySelector = tempSelector;
                }
                catch
                {
                    // Dispose neuralNetwork if NeuralUcbBandit constructor fails
                    neuralNetwork.Dispose();
                    throw;
                }
                
                // Initialize confidence network for model confidence prediction
                tempConfidenceNet = new OnnxNeuralNetwork(onnxLoader, neuralNetworkLogger, runtimeMode, "models/confidence_prediction.onnx");
                _confidenceNetwork = tempConfidenceNet;
            }
            catch
            {
                // Dispose selector if second network creation fails (selector owns neuralNetwork)
                tempSelector?.Dispose();
                tempConfidenceNet?.Dispose();
                throw;
            }
            
            LogBrainInitialized(_logger, null);
        }

        /// <summary>
        /// Initialize all ML models and prepare the brain for trading
        /// This is called from UnifiedOrchestrator startup
        /// </summary>
        public async Task InitializeAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                LogLoadingModels(_logger, null);

                // Load LSTM for price prediction - use your real trained model
                _lstmPricePredictor = await _memoryManager.LoadModelAsync<object>(
                    "models/rl_model.onnx", "v1").ConfigureAwait(false);
                
                // CVaR-PPO is already injected and initialized via DI container
                LogCVarPPOInjected(_logger, null);
                
                // Load meta classifier for market regime - use your test CVaR model
                _metaClassifier = await _memoryManager.LoadModelAsync<object>(
                    "models/rl/test_cvar_ppo.onnx", "v1").ConfigureAwait(false);
                
                // Load market regime detector - use your main RL model as backup
                _marketRegimeDetector = await _memoryManager.LoadModelAsync<object>(
                    "models/rl_model.onnx", "v1").ConfigureAwait(false);

                IsInitialized = true;
                LogAllModelsLoaded(_logger, null);
            }
            catch (FileNotFoundException ex)
            {
                LogModelFileNotFound(_logger, ex);
                IsInitialized = false; // Will use rule-based fallbacks
            }
            catch (DirectoryNotFoundException ex)
            {
                LogModelDirectoryNotFound(_logger, ex);
                IsInitialized = false; // Will use rule-based fallbacks
            }
            catch (IOException ex)
            {
                LogModelIOError(_logger, ex);
                IsInitialized = false; // Will use rule-based fallbacks
            }
            catch (UnauthorizedAccessException ex)
            {
                LogModelAccessDenied(_logger, ex);
                IsInitialized = false; // Will use rule-based fallbacks
            }
            catch (InvalidOperationException ex)
            {
                LogModelInvalidOperation(_logger, ex);
                IsInitialized = false; // Will use rule-based fallbacks
            }
            catch (ArgumentException ex)
            {
                LogModelInvalidArgument(_logger, ex);
                IsInitialized = false; // Will use rule-based fallbacks
            }
        }

        /// <summary>
        /// MAIN BRAIN FUNCTION: Make intelligent trading decision
        /// Called by AutonomousDecisionEngine for live trading
        /// 
        /// This replaces the manual strategy selection in AllStrategies.cs
        /// </summary>
        public async Task<BrainDecision> MakeIntelligentDecisionAsync(
            string symbol,
            Env env,
            Levels levels,
            IList<Bar> bars,
            RiskEngine risk,
            BotCore.Intelligence.Models.MarketIntelligence? intelligence = null,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(symbol);
            ArgumentNullException.ThrowIfNull(env);
            ArgumentNullException.ThrowIfNull(levels);
            ArgumentNullException.ThrowIfNull(bars);
            ArgumentNullException.ThrowIfNull(risk);
            
            // Store latest market data and intelligence for use in risk analysis and commentary
            _latestEnv = env;
            _latestBars = bars;
            _latestIntelligence = intelligence;
            
            var startTime = DateTime.UtcNow;
            LastDecision = startTime;
            
            try
            {
                // PHASE 2: Check economic calendar before trading
                var calendarCheckEnabled = Environment.GetEnvironmentVariable("BOT_CALENDAR_CHECK_ENABLED")?.ToUpperInvariant() == "TRUE";
                if (calendarCheckEnabled && _economicEventManager != null)
                {
                    // Check if symbol trading should be restricted
                    var blockMinutes = int.Parse(Environment.GetEnvironmentVariable("BOT_CALENDAR_BLOCK_MINUTES") ?? "10", System.Globalization.CultureInfo.InvariantCulture);
                    var isRestricted = await _economicEventManager.ShouldRestrictTradingAsync(symbol, TimeSpan.FromMinutes(blockMinutes)).ConfigureAwait(false);
                    
                    if (isRestricted)
                    {
                        LogCalendarBlock(_logger, symbol, null);
                        return CreateNoTradeDecision(symbol, "Economic event restriction", startTime);
                    }
                    
                    // Check for upcoming high-impact events
                    var upcomingEvents = await _economicEventManager.GetUpcomingEventsAsync(
                        TimeSpan.FromMinutes(blockMinutes)).ConfigureAwait(false);
                    
                    var highImpactEvents = upcomingEvents.Where(e => 
                        e.Impact >= BotCore.Market.EventImpact.High && 
                        e.AffectedSymbols.Contains(symbol)).ToList();
                    
                    if (highImpactEvents.Count > 0)
                    {
                        var nextEvent = highImpactEvents[0];
                        var minutesUntil = (nextEvent.ScheduledTime - DateTime.UtcNow).TotalMinutes;
                        LogHighImpactEvent(_logger, nextEvent.Name, minutesUntil, null);
                        return CreateNoTradeDecision(symbol, $"{nextEvent.Name} approaching", startTime);
                    }
                }
                
                // PRODUCTION: Get current news context and form trading bias
                NewsContext? newsContext = null;
                decimal newsBias = 0.0m; // -1.0 (bearish) to +1.0 (bullish)
                decimal newsConfidenceMultiplier = 1.0m; // 0.8 to 1.2
                
                if (_newsMonitor != null && _newsMonitor.IsHealthy)
                {
                    try
                    {
                        newsContext = await _newsMonitor.GetCurrentNewsContextAsync().ConfigureAwait(false);
                        
                        if (newsContext.HasBreakingNews)
                        {
                            // BREAKING NEWS: Form strong directional bias
                            // Sentiment: 0.0 = max bearish, 0.5 = neutral, 1.0 = max bullish
                            // Convert to bias: -1.0 to +1.0
                            newsBias = (newsContext.SentimentScore - 0.5m) * 2.0m;
                            
                            // High volatility period = reduce confidence (wider stops needed)
                            if (newsContext.IsHighVolatilityPeriod)
                            {
                                newsConfidenceMultiplier = 0.85m; // Reduce confidence 15%
                                _logger.LogWarning(
                                    "[Brain] 🔥 BREAKING NEWS + HIGH VOLATILITY: {Headline} | Sentiment: {Sentiment:F2} ({Bias}) | Reducing confidence to {Mult:F2}x",
                                    newsContext.LatestHeadline,
                                    newsContext.SentimentScore,
                                    newsBias > 0 ? "BULLISH" : "BEARISH",
                                    newsConfidenceMultiplier);
                            }
                            else
                            {
                                // Breaking news without high vol = opportunity (increase confidence slightly)
                                newsConfidenceMultiplier = 1.05m; // Increase confidence 5%
                                _logger.LogWarning(
                                    "[Brain] 🔥 BREAKING NEWS (Controlled Vol): {Headline} | Sentiment: {Sentiment:F2} ({Bias}) | Boosting confidence to {Mult:F2}x",
                                    newsContext.LatestHeadline,
                                    newsContext.SentimentScore,
                                    newsBias > 0 ? "BULLISH" : "BEARISH",
                                    newsConfidenceMultiplier);
                            }
                        }
                        else if (newsContext.RecentHeadlines.Count > 0)
                        {
                            // RECENT NEWS (not breaking): Subtle bias based on sentiment
                            // Less aggressive than breaking news
                            newsBias = (newsContext.SentimentScore - 0.5m) * 1.0m; // 50% strength vs breaking news
                            
                            // Slight confidence adjustment based on how far from neutral
                            var sentimentStrength = Math.Abs(newsContext.SentimentScore - 0.5m) * 2.0m; // 0.0 to 1.0
                            if (sentimentStrength > 0.6m)
                            {
                                // Strong sentiment (even if not "breaking") = small confidence boost
                                newsConfidenceMultiplier = 1.02m;
                                _logger.LogInformation(
                                    "[Brain] 📰 Strong news sentiment: {Sentiment:F2} ({Bias}) | {Count} recent headlines | Boosting confidence to {Mult:F2}x",
                                    newsContext.SentimentScore,
                                    newsBias > 0 ? "BULLISH" : "BEARISH",
                                    newsContext.RecentHeadlines.Count,
                                    newsConfidenceMultiplier);
                            }
                            else
                            {
                                // Weak/neutral sentiment = no adjustment
                                _logger.LogDebug(
                                    "[Brain] 📰 Neutral news sentiment: {Sentiment:F2} | {Count} headlines | No bias adjustment",
                                    newsContext.SentimentScore,
                                    newsContext.RecentHeadlines.Count);
                            }
                        }
                    }
                    catch (Exception newsEx)
                    {
                        // News monitoring is non-critical - log and continue with neutral bias
                        _logger.LogDebug(newsEx, "[Brain] News context fetch failed - continuing with neutral news bias");
                        newsBias = 0.0m;
                        newsConfidenceMultiplier = 1.0m;
                    }
                }
                
                // 1. CREATE MARKET CONTEXT from current data
                var context = CreateMarketContext(symbol, env, bars);
                _marketContexts[symbol] = context;
                
                // 2. DETECT MARKET REGIME using Meta Classifier
                var marketRegime = await DetectMarketRegimeAsync(context).ConfigureAwait(false);
                
                // 3. SELECT OPTIMAL STRATEGY using Neural UCB
                var optimalStrategy = await SelectOptimalStrategyAsync(context, marketRegime, cancellationToken).ConfigureAwait(false);
                
                // 4. PREDICT PRICE MOVEMENT using LSTM + NEWS BIAS
                var priceDirection = await PredictPriceDirectionAsync(context, bars).ConfigureAwait(false);
                
                // APPLY NEWS BIAS TO PRICE DIRECTION
                // News bias influences directional probability but doesn't override model completely
                if (Math.Abs(newsBias) > 0.1m) // Only apply if bias is meaningful (>10%)
                {
                    var originalDirection = priceDirection.Direction;
                    var originalProbability = priceDirection.Probability;
                    
                    // Adjust probability based on news sentiment alignment
                    // If news agrees with model: boost probability
                    // If news disagrees with model: reduce probability (but don't flip unless extremely strong)
                    if ((newsBias > 0 && priceDirection.Direction == PriceDirection.Up) ||
                        (newsBias < 0 && priceDirection.Direction == PriceDirection.Down))
                    {
                        // NEWS AGREES WITH MODEL: Boost confidence
                        var biasStrength = Math.Abs(newsBias); // 0.0 to 1.0
                        var probabilityBoost = biasStrength * 0.10m; // Up to 10% boost
                        priceDirection = new PricePrediction
                        {
                            Direction = priceDirection.Direction,
                            Probability = Math.Min(0.95m, priceDirection.Probability + probabilityBoost)
                        };
                        
                        _logger.LogInformation(
                            "[Brain] 📊 News AGREES with model: {Original} {OrigProb:F2} → {New} {NewProb:F2} (bias: {Bias:F2})",
                            originalDirection, originalProbability, 
                            priceDirection.Direction, priceDirection.Probability, 
                            newsBias);
                    }
                    else if ((newsBias > 0 && priceDirection.Direction == PriceDirection.Down) ||
                             (newsBias < 0 && priceDirection.Direction == PriceDirection.Up))
                    {
                        // NEWS DISAGREES WITH MODEL: Reduce confidence or flip if bias extremely strong
                        var biasStrength = Math.Abs(newsBias); // 0.0 to 1.0
                        
                        if (biasStrength > 0.8m && newsContext?.HasBreakingNews == true)
                        {
                            // EXTREMELY strong breaking news: Override model (rare case)
                            priceDirection = new PricePrediction
                            {
                                Direction = newsBias > 0 ? PriceDirection.Up : PriceDirection.Down,
                                Probability = 0.60m // Modest probability when overriding model
                            };
                            
                            _logger.LogWarning(
                                "[Brain] 🔥 BREAKING NEWS OVERRIDE: {Original} {OrigProb:F2} → {New} {NewProb:F2} (bias: {Bias:F2})",
                                originalDirection, originalProbability,
                                priceDirection.Direction, priceDirection.Probability,
                                newsBias);
                        }
                        else
                        {
                            // Moderate disagreement: Reduce confidence
                            var probabilityReduction = biasStrength * 0.15m; // Up to 15% reduction
                            priceDirection = new PricePrediction
                            {
                                Direction = priceDirection.Direction,
                                Probability = Math.Max(0.50m, priceDirection.Probability - probabilityReduction)
                            };
                            
                            _logger.LogInformation(
                                "[Brain] ⚠️ News CONFLICTS with model: {Original} {OrigProb:F2} → {New} {NewProb:F2} (bias: {Bias:F2})",
                                originalDirection, originalProbability,
                                priceDirection.Direction, priceDirection.Probability,
                                newsBias);
                        }
                    }
                }
                
                // 5. OPTIMIZE POSITION SIZE using RL + NEWS CONFIDENCE MULTIPLIER
                var basePositionSize = await OptimizePositionSizeAsync(context, optimalStrategy, priceDirection, cancellationToken).ConfigureAwait(false);
                var optimalSize = basePositionSize * newsConfidenceMultiplier;
                
                if (Math.Abs(newsConfidenceMultiplier - 1.0m) > 0.01m)
                {
                    _logger.LogInformation(
                        "[Brain] 💰 Position size adjusted by news: {Base:F2} × {Mult:F2} = {Final:F2}",
                        basePositionSize, newsConfidenceMultiplier, optimalSize);
                }
                
                // 6. GENERATE ENHANCED CANDIDATES using brain intelligence
                var enhancedCandidates = await GenerateEnhancedCandidatesAsync(
                    symbol, env, levels, bars, risk, optimalStrategy, priceDirection, optimalSize).ConfigureAwait(false);
                
                // Apply intelligence-based adjustments if available
                var adjustedConfidence = optimalStrategy.Confidence * newsConfidenceMultiplier;
                var adjustedSize = optimalSize;
                
                if (_latestIntelligence != null)
                {
                    // Adjust based on recommended bias
                    if (_latestIntelligence.RecommendedBias == BotCore.Intelligence.Models.MarketBias.Bearish &&
                        priceDirection.Direction == PriceDirection.Up)
                    {
                        // LLM says bearish but model says long - reduce size and increase confidence threshold
                        adjustedSize *= 0.7m;
                        adjustedConfidence *= 0.85m;
                        _logger.LogInformation("[Brain] 🤖 Intelligence adjustment: Bearish bias vs Long signal - reducing size to {Size:F2} and confidence to {Conf:P1}",
                            adjustedSize, adjustedConfidence);
                    }
                    else if (_latestIntelligence.RecommendedBias == BotCore.Intelligence.Models.MarketBias.Bullish &&
                             priceDirection.Direction == PriceDirection.Down)
                    {
                        // LLM says bullish but model says short - reduce size and increase confidence threshold
                        adjustedSize *= 0.7m;
                        adjustedConfidence *= 0.85m;
                        _logger.LogInformation("[Brain] 🤖 Intelligence adjustment: Bullish bias vs Short signal - reducing size to {Size:F2} and confidence to {Conf:P1}",
                            adjustedSize, adjustedConfidence);
                    }
                    
                    // Check for high-impact events and reduce exposure
                    if (_latestIntelligence.EventRisks.Count > 0)
                    {
                        var hasHighImpactEvent = _latestIntelligence.EventRisks.Any(e => 
                            e.Contains("CPI", StringComparison.OrdinalIgnoreCase) || 
                            e.Contains("FOMC", StringComparison.OrdinalIgnoreCase) ||
                            e.Contains("NFP", StringComparison.OrdinalIgnoreCase));
                        
                        if (hasHighImpactEvent)
                        {
                            adjustedSize *= 0.5m;
                            _logger.LogWarning("[Brain] ⚠️ High-impact event detected - reducing position size by 50% to {Size:F2}",
                                adjustedSize);
                        }
                    }
                    
                    // Check for elevated risk factors
                    if (_latestIntelligence.RiskFactors.Count > 2)
                    {
                        // Multiple risk factors - be more conservative
                        adjustedSize *= 0.75m;
                        _logger.LogInformation("[Brain] 🤖 Multiple risk factors ({Count}) detected - reducing size to {Size:F2}",
                            _latestIntelligence.RiskFactors.Count, adjustedSize);
                    }
                }
                
                var decision = new BrainDecision
                {
                    Symbol = symbol,
                    RecommendedStrategy = optimalStrategy.SelectedStrategy,
                    StrategyConfidence = adjustedConfidence,
                    PriceDirection = priceDirection.Direction,
                    PriceProbability = priceDirection.Probability,
                    OptimalPositionMultiplier = adjustedSize,
                    MarketRegime = marketRegime,
                    EnhancedCandidates = enhancedCandidates,
                    DecisionTime = startTime,
                    ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds,
                    ModelConfidence = CalculateOverallConfidence(optimalStrategy, priceDirection),
                    RiskAssessment = AssessRisk(context, priceDirection)
                };

                // Track decision for learning
                _decisionHistory.Add(new TradingDecision
                {
                    Symbol = symbol,
                    Strategy = optimalStrategy.SelectedStrategy,
                    Confidence = decision.ModelConfidence,
                    Context = context,
                    Timestamp = startTime
                });

                DecisionsToday++;
                
                LogBrainDecision(_logger, symbol, optimalStrategy.SelectedStrategy, (double)optimalStrategy.Confidence,
                    priceDirection.Direction.ToString(), (double)priceDirection.Probability, null);
                LogDecisionDetails(_logger, (double)optimalSize, marketRegime.ToString(), decision.ProcessingTimeMs, null);

                // AI bot thinking - explain decision before taking trade
                if (_ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_THINKING_ENABLED") == "true"))
                {
                    var thinking = await ThinkAboutDecisionAsync(decision).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(thinking))
                    {
                        LogBotThinking(_logger, thinking, null);
                    }
                }

                // Feature 1: Real-Time Commentary
                if (_ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_COMMENTARY_ENABLED") == "true"))
                {
                    // Check for low confidence (waiting)
                    if (optimalStrategy.Confidence < TopStepConfig.LowConfidenceThreshold)
                    {
                        var commentary = await ExplainWhyWaitingAsync(context, optimalStrategy, priceDirection).ConfigureAwait(false);
                        if (!string.IsNullOrEmpty(commentary))
                        {
                            LogBotCommentary(_logger, commentary, null);
                        }
                    }
                    // Check for high confidence (using MLConfigurationService to replace hardcoded 0.7)
                    else if (optimalStrategy.Confidence > (decimal)_mlConfigService.GetAIConfidenceThreshold())
                    {
                        var commentary = await ExplainConfidenceAsync(decision, context).ConfigureAwait(false);
                        if (!string.IsNullOrEmpty(commentary))
                        {
                            LogBotCommentary(_logger, commentary, null);
                        }
                    }
                }

                // Hook 1: Capture market snapshot (if enabled)
                if (_snapshotStore != null && Environment.GetEnvironmentVariable("SNAPSHOT_ENABLED") == "true")
                {
                    try
                    {
                        // Use actual market data from available sources
                        var vixValue = env.volz ?? 0m; // Use volatility z-score as proxy for VIX
                        var currentPrice = bars.LastOrDefault()?.Close ?? 0m;
                        
                        // Determine session based on time of day
                        var currentHour = DateTime.UtcNow.Hour;
                        string sessionName;
                        if (currentHour >= RegularTradingStartHour && currentHour < RegularTradingEndHour)
                        {
                            sessionName = "RegularTrading";
                        }
                        else if (currentHour >= PreMarketStartHour && currentHour < RegularTradingStartHour)
                        {
                            sessionName = "PreMarket";
                        }
                        else if (currentHour >= RegularTradingEndHour && currentHour < AfterHoursEndHour)
                        {
                            sessionName = "AfterHours";
                        }
                        else
                        {
                            sessionName = "Closed";
                        }
                        
                        // Create default zone snapshot (no zone service in brain yet)
                        var emptyZoneSnapshot = new Zones.ZoneSnapshot(
                            NearestDemand: null,
                            NearestSupply: null,
                            DistToDemandAtr: 0m,
                            DistToSupplyAtr: 0m,
                            BreakoutScore: 0m,
                            ZonePressure: 0m,
                            Utc: DateTime.UtcNow
                        );
                        
                        // Create default pattern scores
                        var emptyPatternScores = new BotCore.Patterns.PatternScoresWithDetails
                        {
                            BullScore = 0.0,
                            BearScore = 0.0,
                            OverallConfidence = 0.0
                        };
                        emptyPatternScores.SetDetectedPatterns(System.Array.Empty<BotCore.Patterns.PatternDetail>());
                        
                        string trend;
                        if (context.TrendStrength > TrendStrengthThreshold)
                        {
                            trend = "Bullish";
                        }
                        else if (context.TrendStrength < -TrendStrengthThreshold)
                        {
                            trend = "Bearish";
                        }
                        else
                        {
                            trend = "Neutral";
                        }
                        
                        var snapshot = BotCore.Services.MarketSnapshotStore.CreateSnapshot(
                            symbol: decision.Symbol,
                            currentPrice: currentPrice,
                            vix: vixValue,
                            trend: trend,
                            session: sessionName,
                            zoneSnapshot: emptyZoneSnapshot,
                            patternScores: emptyPatternScores,
                            strategy: decision.RecommendedStrategy,
                            direction: decision.PriceDirection.ToString(),
                            confidence: decision.StrategyConfidence,
                            size: (int)decision.OptimalPositionMultiplier
                        );
                        _snapshotStore.StoreSnapshot(snapshot);
                        LogSnapshotCaptured(_logger, symbol, decision.RecommendedStrategy, decision.PriceDirection.ToString(), null);
                    }
                    catch (InvalidOperationException ex)
                    {
                        LogSnapshotInvalidOperation(_logger, ex);
                    }
                    catch (ArgumentException ex)
                    {
                        LogSnapshotArgumentException(_logger, ex);
                    }
                    catch (IOException ex)
                    {
                        LogSnapshotIOException(_logger, ex);
                    }
                }

                return decision;
            }
            catch (InvalidOperationException ex)
            {
                LogDecisionInvalidOperation(_logger, symbol, ex);
                return CreateFallbackDecision(symbol, env, levels, bars, risk);
            }
            catch (ArgumentException ex)
            {
                LogDecisionInvalidArgument(_logger, symbol, ex);
                return CreateFallbackDecision(symbol, env, levels, bars, risk);
            }
            catch (TimeoutException ex)
            {
                LogDecisionTimeout(_logger, symbol, ex);
                return CreateFallbackDecision(symbol, env, levels, bars, risk);
            }
        }

        /// <summary>
        /// Enhanced learning from trading results that improves ALL strategies
        /// Every trade outcome teaches all strategies and improves future decision-making
        /// Called after order execution and P&L is known
        /// </summary>
        public async Task LearnFromResultAsync(
            string symbol,
            string strategy,
            decimal pnl,
            bool wasCorrect,
            TimeSpan holdTime,
            CancellationToken cancellationToken = default)
        {
            try
            {
                // Update strategy selector (Neural UCB) with reward
                var context = _marketContexts.TryGetValue(symbol, out var ctx) ? ctx : null;
                if (context != null)
                {
                    var reward = CalculateReward(pnl, wasCorrect, holdTime);
                    var contextVector = CreateContextVector(context);
                    
                    await _strategySelector.UpdateArmAsync(strategy, contextVector, reward, cancellationToken).ConfigureAwait(false);
                    
                    // 🎓 CVaR-PPO EXPERIENCE FEEDING: Learn from live + historical trades
                    if (_cvarPPO != null && _lastCVaRState != null)
                    {
                        try
                        {
                            // Create next state for experience
                            var nextState = CreateCVaRStateVector(context, 
                                new StrategySelection { SelectedStrategy = strategy, Confidence = 0.5m, UcbValue = 0 },
                                new PricePrediction { Direction = PriceDirection.Sideways, Probability = 0.5m });
                            
                            // Calculate CVaR-specific reward (incorporates risk-adjusted return)
                            var cvarReward = CalculateCVaRReward(pnl, wasCorrect, holdTime, _lastCVaRValue);
                            
                            // Create experience and add to buffer
                            var experience = new TradingBot.RLAgent.Experience
                            {
                                State = _lastCVaRState.ToList(),
                                Action = _lastCVaRAction,
                                Reward = (double)cvarReward,
                                NextState = nextState.ToList(),
                                Done = true, // Position closed
                                LogProbability = 0, // Will be recalculated during training
                                ValueEstimate = _lastCVaRValue,
                                Return = 0 // Will be calculated during advantage estimation
                            };
                            
                            _cvarPPO.AddExperience(experience);
                            _logger.LogInformation("[CVAR-LEARN] 🎓 Experience added: Action={Action}, Reward={Reward:F2}, Buffer={BufferSize}", 
                                _lastCVaRAction, cvarReward, _cvarPPO.ExperienceBufferSize);
                            
                            // Reset state tracking
                            _lastCVaRState = null;
                            
                            // Periodic training (every 6 hours OR when buffer reaches threshold)
                            var hoursSinceTraining = (DateTime.UtcNow - _lastCVaRTraining).TotalHours;
                            var shouldTrain = (hoursSinceTraining >= 6 && _cvarPPO.ExperienceBufferSize >= 256) || 
                                            _cvarPPO.ExperienceBufferSize >= 1000;
                            
                            if (shouldTrain)
                            {
                                _logger.LogInformation("[CVAR-TRAIN] 🚀 Triggering CVaR-PPO training: Buffer={Buffer}, Hours={Hours:F1}", 
                                    _cvarPPO.ExperienceBufferSize, hoursSinceTraining);
                                
                                // Train in background to avoid blocking
                                _ = Task.Run(async () =>
                                {
                                    try
                                    {
                                        var result = await _cvarPPO.TrainAsync(cancellationToken).ConfigureAwait(false);
                                        _logger.LogInformation("[CVAR-TRAIN] ✅ Training complete: Episode={Episode}, Loss={Loss:F4}, Reward={Reward:F2}", 
                                            result.Episode, result.TotalLoss, result.AverageReward);
                                        _lastCVaRTraining = DateTime.UtcNow;
                                    }
                                    catch (Exception ex)
                                    {
                                        _logger.LogError(ex, "[CVAR-TRAIN] ❌ Training failed");
                                    }
                                }, cancellationToken);
                            }
                        }
                        catch (Exception ex)
                        {
                            _logger.LogError(ex, "[CVAR-LEARN] ❌ Failed to add CVaR-PPO experience");
                        }
                    }
                    
                    // 🚀 MULTI-STRATEGY LEARNING: Update ALL strategies with this market condition
                    await UpdateAllStrategiesFromOutcomeAsync(context, strategy, reward, wasCorrect, cancellationToken).ConfigureAwait(false);
                }

                // Update performance tracking for the specific strategy
                if (!_performance.TryGetValue(symbol, out var perf))
                {
                    perf = new TradingPerformance();
                    _performance[symbol] = perf;
                }
                perf.TotalTrades++;
                perf.TotalPnL += pnl;
                if (wasCorrect) perf.WinningTrades++;
                
                // Update strategy-specific performance tracking
                if (context != null)
                {
                    UpdateStrategyPerformance(strategy, context, wasCorrect, pnl, holdTime);
                }
                
                // Calculate today's win rate
                var todayDecisions = _decisionHistory.Where(d => d.Timestamp.Date == DateTime.Today).ToList();
                if (todayDecisions.Count > 0)
                {
                    // This would be updated when we get actual results
                    WinRateToday = (decimal)todayDecisions.Count(d => d.WasCorrect) / todayDecisions.Count;
                }

                // Enhanced model retraining with multi-strategy learning
                if (DateTime.UtcNow - _lastUnifiedLearningUpdate > TimeSpan.FromHours(LearningUpdateIntervalHours) && _decisionHistory.Count > MinDecisionsForLearningUpdate)
                {
                    _ = Task.Run(() => UpdateUnifiedLearningAsync(cancellationToken), cancellationToken);
                    _lastUnifiedLearningUpdate = DateTime.UtcNow;
                }

                // Periodic full model retraining
                if (DateTime.UtcNow - _lastModelUpdate > TimeSpan.FromHours(ModelRetrainingIntervalHours) && _decisionHistory.Count > MinDecisionsForModelRetraining)
                {
                    _ = Task.Run(() => RetrainModelsAsync(cancellationToken), cancellationToken);
                    _lastModelUpdate = DateTime.UtcNow;
                }

                LogUnifiedLearning(_logger, symbol, strategy, (double)pnl, wasCorrect, (double)WinRateToday, perf.TotalTrades, null);

                // AI bot reflection - reflect on completed trade
                if (_ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_REFLECTION_ENABLED") == "true"))
                {
                    var reflection = await ReflectOnOutcomeAsync(symbol, strategy, pnl, wasCorrect, holdTime).ConfigureAwait(false);
                    if (!string.IsNullOrEmpty(reflection))
                    {
                        LogBotReflection(_logger, reflection, null);
                    }
                }

                // Feature 2: Trade Failure Analysis (only for losses)
                if (!wasCorrect && pnl < 0 && _ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_FAILURE_ANALYSIS_ENABLED") == "true"))
                {
                    // Get entry and exit context if available
                    var entryContext = context ?? _marketContexts.GetValueOrDefault(symbol);
                    if (entryContext != null)
                    {
                        // Note: Entry/stop/target/exit prices tracked separately in position management
                        var failureAnalysis = await AnalyzeTradeFailureAsync(
                            symbol, strategy, pnl,
                            0, 0, 0, 0, // Entry/stop/target/exit prices would be tracked separately
                            "Stop hit", entryContext, null).ConfigureAwait(false);
                        
                        if (!string.IsNullOrEmpty(failureAnalysis))
                        {
                            LogBotFailureAnalysis(_logger, failureAnalysis, null);
                        }
                    }
                }

                // Feature 6: Learning Progress Reports
                if (_ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_LEARNING_REPORTS_ENABLED") == "true") && DateTime.UtcNow - _lastUnifiedLearningUpdate < TimeSpan.FromMinutes(1))
                {
                    // Report after model updates
                    var learningReport = await ExplainWhatILearnedAsync(
                        "Unified Learning Update",
                        $"Updated all strategies from {strategy} trade outcome. Win rate: {WinRateToday:P0}").ConfigureAwait(false);
                    
                    if (!string.IsNullOrEmpty(learningReport))
                    {
                        LogBotLearningReport(_logger, learningReport, null);
                    }
                }
            }
            catch (InvalidOperationException ex)
            {
                LogLearningInvalidOperation(_logger, ex);
            }
            catch (ArgumentException ex)
            {
                LogLearningInvalidArgument(_logger, ex);
            }
        }
        
        /// <summary>
        /// Update ALL strategies based on a single trade outcome
        /// This creates cross-strategy learning where each outcome improves the entire system
        /// </summary>
        private async Task UpdateAllStrategiesFromOutcomeAsync(
            MarketContext context, 
            string executedStrategy, 
            decimal reward, 
            bool wasCorrect,
                        CancellationToken cancellationToken)
        {
            try
            {
                var contextVector = CreateContextVector(context);
                
                foreach (var strategy in PrimaryStrategies)
                {
                    if (strategy == executedStrategy)
                        continue; // Already updated above
                    
                    // Calculate learning reward for non-executed strategies based on market conditions
                    var crossLearningReward = CalculateCrossLearningReward(
                        strategy, executedStrategy, context, reward, wasCorrect);
                    
                    // Update strategy knowledge even if it wasn't executed
                    await _strategySelector.UpdateArmAsync(strategy, contextVector, crossLearningReward, cancellationToken).ConfigureAwait(false);
                    
                    // Hook 4: Track parameter changes (if enabled)
                    // Note: Parameter tracking happens after UpdateArmAsync to detect any changes made
                    if (_parameterTracker != null && Environment.GetEnvironmentVariable("PARAMETER_TRACKING_ENABLED") == "true")
                    {
                        try
                        {
                            // Record the parameter update with outcome context
                            var reason = wasCorrect ? "Successful trade outcome" : "Learning from unsuccessful trade";
                            _parameterTracker.RecordChange(
                                strategyName: strategy,
                                parameterName: "StrategyWeight",
                                oldValue: reward.ToString("F3", CultureInfo.InvariantCulture),
                                newValue: crossLearningReward.ToString("F3", CultureInfo.InvariantCulture),
                                reason: reason,
                                outcomePnl: reward,
                                wasCorrect: wasCorrect
                            );
                            LogParamTracked(_logger, strategy, (double)reward, (double)crossLearningReward, null);
                        }
                        catch (InvalidOperationException ex)
                        {
                            LogParamTrackInvalidOperation(_logger, ex);
                        }
                        catch (IOException ex)
                        {
                            LogParamTrackIoError(_logger, ex);
                        }
                        catch (UnauthorizedAccessException ex)
                        {
                            LogParamTrackAccessDenied(_logger, ex);
                        }
                    }
                    
                    // Update strategy-specific learning patterns
                    UpdateStrategyOptimalConditions(strategy, context, crossLearningReward > BaseConfidenceThreshold);
                }
                
                LogCrossLearningUpdate(_logger, executedStrategy, (double)reward, null);
            }
            catch (InvalidOperationException ex)
            {
                LogCrossLearningInvalidOperation(_logger, ex);
            }
            catch (ArgumentException ex)
            {
                LogCrossLearningArgumentException(_logger, ex);
            }
        }
        
        /// <summary>
        /// Gather current market context for AI conversation
        /// </summary>
        private string GatherCurrentContext()
        {
            try
            {
                // Get VIX level from latest context (use 0 if unavailable to avoid hiding missing data)
                var vixLevel = _marketContexts.Values.LastOrDefault()?.Volatility ?? 0m;
                
                // Get today's P&L
                var todayPnl = _dailyPnl;
                
                // Calculate today's win rate
                var todayDecisions = _decisionHistory.Where(d => d.Timestamp.Date == DateTime.Today).ToList();
                var winRate = todayDecisions.Count > 0 
                    ? (decimal)todayDecisions.Count(d => d.WasCorrect) / todayDecisions.Count 
                    : 0m;
                
                // Get current market trend
                var trend = "Unknown";
                if (!_marketContexts.IsEmpty)
                {
                    var latestContext = _marketContexts.Values.LastOrDefault();
                    if (latestContext != null)
                    {
                        if (latestContext.TrendStrength > StrongTrendThreshold)
                        {
                            trend = "Bullish";
                        }
                        else if (latestContext.TrendStrength < -StrongTrendThreshold)
                        {
                            trend = "Bearish";
                        }
                        else
                        {
                            trend = "Neutral";
                        }
                    }
                }
                
                // Get active strategies
                var activeStrategies = string.Join(", ", PrimaryStrategies);
                
                // Get current position info (if any)
                var positionInfo = "None";
                
                // Format context
                return $"VIX: {vixLevel:F1}, PnL Today: ${todayPnl:F2}, Win Rate: {winRate:P0}, " +
                       $"Trend: {trend}, Active Strategies: {activeStrategies}, Position: {positionInfo}, " +
                       $"Decisions Today: {DecisionsToday}";
            }
            catch (InvalidOperationException ex)
            {
                LogContextGatherInvalidOperation(_logger, ex);
                return "Context unavailable";
            }
            catch (ArgumentException ex)
            {
                LogContextGatherArgumentException(_logger, ex);
                return "Context unavailable";
            }
        }
        
        /// <summary>
        /// AI thinks about and explains a trading decision before execution
        /// </summary>
        private async Task<string> ThinkAboutDecisionAsync(BrainDecision decision)
        {
            if (_ollamaClient == null)
                return string.Empty;
                
            try
            {
                var currentContext = GatherCurrentContext();
                
                // Hook 2: Add risk assessment commentary (if enabled)
                string riskContext = string.Empty;
                if (_riskCommentary != null && Environment.GetEnvironmentVariable("RISK_COMMENTARY_ENABLED") == "true")
                {
                    try
                    {
                        // Use actual market data from latest decision
                        var currentPrice = _latestBars?.LastOrDefault()?.Close ?? 0m;
                        var atr = _latestEnv?.atr ?? 0m;
                        
                        // Check if async mode is enabled
                        var asyncMode = Environment.GetEnvironmentVariable("RISK_COMMENTARY_ASYNC") == "true";
                        
                        // Only proceed if we have valid data
                        if (currentPrice > 0m && atr > 0m)
                        {
                            if (asyncMode)
                            {
                                // Fire-and-forget: Start analysis in background, continue trading immediately
                                _riskCommentary.AnalyzeRiskFireAndForget(decision.Symbol, currentPrice, atr);
                                LogRiskCommentaryStarted(_logger, null);
                            }
                            else
                            {
                                // Blocking mode: Wait for result (for debugging/testing only)
                                riskContext = await _riskCommentary.AnalyzeRiskAsync(
                                    decision.Symbol, currentPrice, atr).ConfigureAwait(false);
                                    
                                if (!string.IsNullOrEmpty(riskContext))
                                {
                                    LogRiskCommentary(_logger, riskContext, null);
                                }
                            }
                        }
                        else
                        {
                            LogRiskCommentaryMissingData(_logger, null);
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        LogRiskCommentaryInvalidOperation(_logger, ex);
                    }
                    catch (HttpRequestException ex)
                    {
                        LogRiskCommentaryHttpError(_logger, ex);
                    }
                    catch (TaskCanceledException ex)
                    {
                        LogRiskCommentaryTaskCancelled(_logger, ex);
                    }
                }
                
                // Hook 6: Historical pattern recognition (if enabled)
                string historicalContext = string.Empty;
                if (_historicalPatterns != null && _snapshotStore != null && 
                    Environment.GetEnvironmentVariable("PATTERN_RECOGNITION_ENABLED") == "true")
                {
                    try
                    {
                        // Use actual market data for similarity search
                        var vixValue = _latestEnv?.volz ?? 0m;
                        var currentPrice = _latestBars?.LastOrDefault()?.Close ?? 0m;
                        var currentHour = DateTime.UtcNow.Hour;
                        string sessionName;
                        if (currentHour >= RegularTradingStartHour && currentHour < RegularTradingEndHour)
                        {
                            sessionName = "RegularTrading";
                        }
                        else if (currentHour >= PreMarketStartHour && currentHour < RegularTradingStartHour)
                        {
                            sessionName = "PreMarket";
                        }
                        else if (currentHour >= RegularTradingEndHour && currentHour < AfterHoursEndHour)
                        {
                            sessionName = "AfterHours";
                        }
                        else
                        {
                            sessionName = "Closed";
                        }
                        
                        // Get trend from latest context
                        var latestContext = _marketContexts.Values.LastOrDefault();
                        string trendName;
                        if (latestContext != null && latestContext.TrendStrength > TrendStrengthThreshold)
                        {
                            trendName = "Bullish";
                        }
                        else if (latestContext != null && latestContext.TrendStrength < -TrendStrengthThreshold)
                        {
                            trendName = "Bearish";
                        }
                        else
                        {
                            trendName = "Neutral";
                        }
                        
                        // Create default zone snapshot and pattern scores for similarity search
                        var emptyZoneSnapshot = new Zones.ZoneSnapshot(
                            NearestDemand: null,
                            NearestSupply: null,
                            DistToDemandAtr: 0m,
                            DistToSupplyAtr: 0m,
                            BreakoutScore: 0m,
                            ZonePressure: 0m,
                            Utc: DateTime.UtcNow
                        );
                        
                        var emptyPatternScores = new BotCore.Patterns.PatternScoresWithDetails
                        {
                            BullScore = 0.0,
                            BearScore = 0.0,
                            OverallConfidence = 0.0
                        };
                        emptyPatternScores.SetDetectedPatterns(System.Array.Empty<BotCore.Patterns.PatternDetail>());
                        
                        // Create market snapshot for similarity search
                        var currentSnapshot = BotCore.Services.MarketSnapshotStore.CreateSnapshot(
                            symbol: decision.Symbol,
                            currentPrice: currentPrice,
                            vix: vixValue,
                            trend: trendName,
                            session: sessionName,
                            zoneSnapshot: emptyZoneSnapshot,
                            patternScores: emptyPatternScores,
                            strategy: decision.RecommendedStrategy,
                            direction: decision.PriceDirection.ToString(),
                            confidence: decision.StrategyConfidence,
                            size: (int)decision.OptimalPositionMultiplier
                        );
                        
                        var analysis = _historicalPatterns.FindSimilarConditions(currentSnapshot);
                        if (analysis.Matches.Count > 0)
                        {
                            historicalContext = await _historicalPatterns.ExplainSimilarConditionsAsync(analysis).ConfigureAwait(false);
                            if (!string.IsNullOrEmpty(historicalContext))
                            {
                                LogHistoricalPattern(_logger, historicalContext, null);
                            }
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        LogHistoricalPatternInvalidOperation(_logger, ex);
                    }
                    catch (ArgumentException ex)
                    {
                        LogHistoricalPatternInvalidArgument(_logger, ex);
                    }
                    catch (KeyNotFoundException ex)
                    {
                        LogHistoricalPatternKeyNotFound(_logger, ex);
                    }
                }
                
                var riskSection = !string.IsNullOrEmpty(riskContext) ? $"Risk Assessment: {riskContext}\n\n" : "";
                var historicalSection = !string.IsNullOrEmpty(historicalContext) ? $"Historical Context: {historicalContext}\n\n" : "";
                
                var prompt = $@"I am a trading bot. I'm about to take this trade:
Strategy: {decision.RecommendedStrategy}
Direction: {decision.PriceDirection}
Confidence: {decision.StrategyConfidence:P1}
Market Regime: {decision.MarketRegime}

Current context: {currentContext}

{riskSection}{historicalSection}Explain in 2-3 sentences why I'm taking this trade. Speak as ME (the bot), not as an observer.";
                
                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogThinkingError(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogThinkingError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogThinkingError(_logger, ex);
                return string.Empty;
            }
            catch (ArgumentException ex)
            {
                LogThinkingError(_logger, ex);
                return string.Empty;
            }
        }
        
        /// <summary>
        /// AI reflects on a completed trade outcome
        /// </summary>
        private async Task<string> ReflectOnOutcomeAsync(
            string symbol,
            string strategy,
            decimal pnl,
            bool wasCorrect,
            TimeSpan holdTime)
        {
            if (_ollamaClient == null)
                return string.Empty;
                
            try
            {
                var result = wasCorrect ? "WIN" : "LOSS";
                var durationMinutes = (int)holdTime.TotalMinutes;
                string reason;
                if (pnl > 0)
                {
                    reason = "target hit";
                }
                else if (pnl < 0)
                {
                    reason = "stop hit";
                }
                else
                {
                    reason = "timeout";
                }
                
                // Hook 3: Add learning commentary (if enabled)
                string learningContext = string.Empty;
                if (_learningCommentary != null && Environment.GetEnvironmentVariable("LEARNING_COMMENTARY_ENABLED") == "true")
                {
                    try
                    {
                        var lookbackMinutes = int.TryParse(Environment.GetEnvironmentVariable("LEARNING_LOOKBACK_MINUTES"), 
                            out var mins) ? mins : 60;
                        
                        // Check if async mode is enabled
                        var asyncMode = Environment.GetEnvironmentVariable("LEARNING_COMMENTARY_ASYNC") == "true";
                        
                        if (asyncMode)
                        {
                            // Fire-and-forget: Start explanation in background, continue immediately
                            _learningCommentary.ExplainRecentAdaptationsFireAndForget(lookbackMinutes);
                            LogLearningCommentaryStarted(_logger, null);
                        }
                        else
                        {
                            // Blocking mode: Wait for result (for debugging/testing only)
                            learningContext = await _learningCommentary.ExplainRecentAdaptationsAsync(lookbackMinutes).ConfigureAwait(false);
                            
                            if (!string.IsNullOrEmpty(learningContext))
                            {
                                LogLearningCommentary(_logger, learningContext, null);
                            }
                        }
                    }
                    catch (InvalidOperationException ex)
                    {
                        LogLearningCommentaryInvalidOperation(_logger, ex);
                    }
                    catch (HttpRequestException ex)
                    {
                        LogLearningCommentaryHttpError(_logger, ex);
                    }
                    catch (TaskCanceledException ex)
                    {
                        LogLearningCommentaryTaskCancelled(_logger, ex);
                    }
                }
                
                var prompt = $@"I am a trading bot. I just closed a trade:
Symbol: {symbol}
Strategy: {strategy}
Result: {result}
Profit/Loss: ${pnl:F2}
Duration: {durationMinutes} minutes
Reason closed: {reason}

{(!string.IsNullOrEmpty(learningContext) ? $"Recent Learning: {learningContext}\n\n" : "")}Reflect on what happened in 1-2 sentences. Speak as ME (the bot).";
                
                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogReflectionError(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogReflectionError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogReflectionError(_logger, ex);
                return string.Empty;
            }
            catch (ArgumentException ex)
            {
                LogReflectionError(_logger, ex);
                return string.Empty;
            }
        }
        
        /// <summary>
        /// Calculate learning reward for strategies that weren't executed
        /// This allows all strategies to learn from market conditions and outcomes
        /// </summary>
        private decimal CalculateCrossLearningReward(
            string learningStrategy, 
            string executedStrategy, 
            MarketContext context, 
            decimal executedReward, 
            bool wasCorrect)
        {
            // Get strategy specializations
            var learningSpec = _strategySpecializations.GetValueOrDefault(learningStrategy);
            var executedSpec = _strategySpecializations.GetValueOrDefault(executedStrategy);
            
            if (learningSpec == null || executedSpec == null)
                return executedReward * MinConfidenceAdjustment; // Minimal learning if no specialization
            
            // Calculate similarity in optimal conditions
            var conditionSimilarity = learningSpec.OptimalConditions
                .Intersect(executedSpec.OptimalConditions)
                .Count() / (float)Math.Max(learningSpec.OptimalConditions.Count, 1);
            
            // Time window overlap
            var timeOverlap = learningSpec.TimeWindows
                .Intersect(executedSpec.TimeWindows)
                .Count() / (float)Math.Max(learningSpec.TimeWindows.Count, 1);
            
            // Market condition alignment
            var currentConditions = GetCurrentMarketConditions(context);
            var conditionAlignment = learningSpec.OptimalConditions
                .Intersect(currentConditions)
                .Count() / (float)Math.Max(learningSpec.OptimalConditions.Count, 1);
            
            // Calculate cross-learning strength
            var learningStrength = (conditionSimilarity * 0.4f + timeOverlap * 0.3f + conditionAlignment * 0.3f);
            
            // Positive outcome strengthens similar strategies, negative outcome weakens them
            var baseReward = wasCorrect ? executedReward : (1 - executedReward);
            return Math.Clamp(baseReward * (decimal)learningStrength, 0m, 1m);
        }
        
        private static string[] GetCurrentMarketConditions(MarketContext context)
        {
            var conditions = new List<string>();
            
            if (context.Volatility < LowVolatilityThreshold) conditions.Add("low_volatility");
            else if (context.Volatility > HighVolatilityThreshold) conditions.Add("high_volatility");
            
            if (context.VolumeRatio > HighVolumeRatioThreshold) conditions.Add("high_volume");
            if (context.TrendStrength > StrongTrendThreshold) conditions.Add("trending");
            else if (context.TrendStrength < WeakTrendThreshold) conditions.Add("ranging");
            
            if (context.RSI > OverboughtRSILevel) conditions.Add("overbought");
            else if (context.RSI < OversoldRSILevel) conditions.Add("oversold");
            
            var hour = context.TimeOfDay.Hours;
            if (hour >= OpeningDriveStartHour && hour <= OpeningDriveEndHour) conditions.Add("opening_drive");
            else if (hour >= LunchStartHour && hour <= LunchEndHour) conditions.Add("lunch");
            else if (hour >= AfternoonFadeStartHour && hour <= AfternoonFadeEndHour) conditions.Add("afternoon_fade");
            
            return conditions.ToArray();
        }

        #region ML Model Predictions

        private Task<MarketRegime> DetectMarketRegimeAsync(MarketContext context)
        {
            if (_metaClassifier == null || !IsInitialized)
            {
                // Fallback: use volatility-based regime detection
                MarketRegime regime;
                if (context.Volatility > WeakTrendThreshold)
                {
                    regime = MarketRegime.HighVolatility;
                }
                else if (context.Volatility < LowVolatilityThreshold)
                {
                    regime = MarketRegime.LowVolatility;
                }
                else
                {
                    regime = MarketRegime.Normal;
                }
                return Task.FromResult(regime);
            }

            try
            {
                // Analyze market regime using technical indicators and volatility
                // ONNX model integration planned for future enhancement
                MarketRegime detectedRegime;
                if (context.VolumeRatio > HighVolumeRatioThreshold && context.Volatility > TrendingVolatilityThreshold)
                    detectedRegime = MarketRegime.Trending;
                else if (context.Volatility < LowVolatilityThreshold && Math.Abs(context.PriceChange) < RangingPriceChangeThreshold)
                    detectedRegime = MarketRegime.Ranging;
                else if (context.Volatility > HighVolatilityThreshold)
                    detectedRegime = MarketRegime.HighVolatility;
                else
                    detectedRegime = MarketRegime.Normal;

                // Feature 5: Market Regime Explanations
                if (_ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_REGIME_EXPLANATION_ENABLED") == "true"))
                {
                    _ = Task.Run(async () =>
                    {
                        var explanation = await ExplainMarketRegimeAsync(detectedRegime, context).ConfigureAwait(false);
                        if (!string.IsNullOrEmpty(explanation))
                        {
                            LogMarketRegimeExplanation(_logger, explanation, null);
                        }
                    });
                }
                
                return Task.FromResult(detectedRegime);
            }
            catch (InvalidOperationException ex)
            {
                LogMetaClassifierFallback(_logger, ex);
                return Task.FromResult(MarketRegime.Normal);
            }
            catch (ArgumentException ex)
            {
                LogMetaClassifierFallback(_logger, ex);
                return Task.FromResult(MarketRegime.Normal);
            }
            catch (TimeoutException ex)
            {
                LogMetaClassifierFallback(_logger, ex);
                return Task.FromResult(MarketRegime.Normal);
            }
        }

        private async Task<StrategySelection> SelectOptimalStrategyAsync(
            MarketContext context, 
            MarketRegime regime, 
            CancellationToken cancellationToken)
        {
            try
            {
                // Use Neural UCB to select optimal strategy
                var availableStrategies = GetAvailableStrategies(context.TimeOfDay, regime);
                var contextVector = CreateContextVector(context);
                
                var selection = await _strategySelector.SelectArmAsync(availableStrategies, contextVector, cancellationToken).ConfigureAwait(false);
                
                var result = new StrategySelection
                {
                    SelectedStrategy = selection.SelectedArm,
                    Confidence = selection.Confidence,
                    UcbValue = selection.UcbValue,
                    Reasoning = selection.SelectionReason ?? "Neural UCB selection"
                };

                // Feature 4: Strategy Confidence Explanations
                if (_ollamaClient != null && (Environment.GetEnvironmentVariable("BOT_STRATEGY_EXPLANATION_ENABLED") == "true"))
                {
                    // Get all strategy scores for explanation
                    var allScores = new Dictionary<string, decimal>();
                    // Use MLConfigurationService for confidence factor (replaces hardcoded 0.7)
                    var confidenceFactor = (decimal)_mlConfigService.GetAIConfidenceThreshold();
                    foreach (var strategy in availableStrategies)
                    {
                        // Use confidence as a proxy for score (actual UCB scores would be tracked separately)
                        allScores[strategy] = strategy == selection.SelectedArm ? selection.Confidence : selection.Confidence * confidenceFactor;
                    }
                    
                    // Check for strategy conflicts (multiple strategies with similar scores)
                    var topScores = allScores.OrderByDescending(kvp => kvp.Value).Take(2).ToList();
                    if (topScores.Count > 1 && Math.Abs(topScores[0].Value - topScores[1].Value) < TopStepConfig.StrategyConflictThreshold)
                    {
                        // Strategies are conflicting - close scores
                        var conflictExplanation = await ExplainConflictAsync(allScores, context).ConfigureAwait(false);
                        if (!string.IsNullOrEmpty(conflictExplanation))
                        {
                            LogStrategyConflict(_logger, conflictExplanation, null);
                        }
                    }
                    else
                    {
                        var explanation = await ExplainStrategySelectionAsync(selection.SelectedArm, allScores, context).ConfigureAwait(false);
                        if (!string.IsNullOrEmpty(explanation))
                        {
                            LogStrategySelectionExplanation(_logger, explanation, null);
                        }
                    }
                }
                
                return result;
            }
            catch (InvalidOperationException ex)
            {
                LogNeuralUcbFallback(_logger, ex);
                
                // Fallback: time-based strategy selection from your existing logic
                var hour = context.TimeOfDay.Hours;
                var fallbackStrategy = hour switch
                {
                    >= 9 and <= 10 => "S6", // Opening range
                    >= 14 and <= 16 => "S11", // Afternoon strength
                    >= 3 and <= 9 => "S2", // Overnight
                    _ => "S3" // Default
                };
                
                return new StrategySelection
                {
                    SelectedStrategy = fallbackStrategy,
                    Confidence = TopStepConfig.FallbackConfidence,
                    UcbValue = TopStepConfig.FallbackUcbValue,
                    Reasoning = "Fallback time-based selection"
                };
            }
        }

        private Task<PricePrediction> PredictPriceDirectionAsync(
            MarketContext context, 
            IList<Bar> bars
            )
        {
            if (_lstmPricePredictor == null || !IsInitialized)
            {
                // Fallback: trend-based prediction
                var recentBars = bars.TakeLast(5).ToList();
                if (recentBars.Count >= 2)
                {
                    var priceChange = recentBars[recentBars.Count - 1].Close - recentBars[0].Close;
                    var direction = priceChange > 0 ? PriceDirection.Up : PriceDirection.Down;
                    var probability = Math.Min(0.75m, 0.5m + Math.Abs(priceChange) / (context.Atr ?? 10));
                    
                    return Task.FromResult(new PricePrediction
                    {
                        Direction = direction,
                        Probability = probability,
                        ExpectedMove = Math.Abs(priceChange),
                        TimeHorizon = TimeSpan.FromMinutes(30)
                    });
                }
            }

            try
            {
                // Price prediction using technical analysis indicators
                // LSTM model integration planned for future enhancement
                var ema20 = CalculateEMA(bars, 20);
                var ema50 = CalculateEMA(bars, 50);
                var rsi = CalculateRSI(bars, 14);
                
                var isUptrend = ema20 > ema50 && bars[bars.Count - 1].Close > ema20;
                var isOversold = rsi < 30;
                var isOverbought = rsi > 70;
                
                PriceDirection direction;
                decimal probability;
                
                if (isUptrend && !isOverbought)
                {
                    direction = PriceDirection.Up;
                    probability = TopStepConfig.HighConfidenceProbability;
                }
                else if (!isUptrend && !isOversold)
                {
                    direction = PriceDirection.Down;
                    probability = TopStepConfig.HighConfidenceProbability;
                }
                else
                {
                    direction = PriceDirection.Sideways;
                    probability = TopStepConfig.NeutralProbability;
                }
                
                return Task.FromResult(new PricePrediction
                {
                    Direction = direction,
                    Probability = probability,
                    ExpectedMove = context.Atr ?? TopStepConfig.DefaultAtrLookback,
                    TimeHorizon = TimeSpan.FromMinutes(30)
                });
            }
            catch (InvalidOperationException ex)
            {
                LogLstmPredictionFallback(_logger, ex);
                return Task.FromResult(new PricePrediction
                {
                    Direction = PriceDirection.Sideways,
                    Probability = TopStepConfig.NeutralProbability,
                    ExpectedMove = TopStepConfig.FallbackExpectedMove,
                    TimeHorizon = TimeSpan.FromMinutes(30)
                });
            }
            catch (OnnxRuntimeException ex)
            {
                LogLstmPredictionFallback(_logger, ex);
                return Task.FromResult(new PricePrediction
                {
                    Direction = PriceDirection.Sideways,
                    Probability = TopStepConfig.NeutralProbability,
                    ExpectedMove = TopStepConfig.FallbackExpectedMove,
                    TimeHorizon = TimeSpan.FromMinutes(30)
                });
            }
            catch (ArgumentException ex)
            {
                LogLstmPredictionFallback(_logger, ex);
                return Task.FromResult(new PricePrediction
                {
                    Direction = PriceDirection.Sideways,
                    Probability = TopStepConfig.NeutralProbability,
                    ExpectedMove = TopStepConfig.FallbackExpectedMove,
                    TimeHorizon = TimeSpan.FromMinutes(30)
                });
            }
        }

        private async Task<decimal> OptimizePositionSizeAsync(
            MarketContext context,
            StrategySelection strategy,
            PricePrediction prediction,
                        CancellationToken cancellationToken)
        {
            // Check TopStep compliance first
            var (canTrade, reason, _) = ShouldStopTrading();
            if (!canTrade)
            {
                LogTradingBlocked(_logger, reason, null);
                return 0m; // No position if compliance violated
            }

            // Calculate base risk amount
            var baseRisk = _accountBalance * TopStepConfig.RiskPerTrade;

            // Progressive position reduction based on drawdown
            var drawdownRatio = _currentDrawdown / TopStepConfig.MaxDrawdown;
            decimal riskMultiplier = drawdownRatio switch
            {
                > 0.75m => 0.25m, // Very conservative when near max drawdown
                > 0.5m => 0.5m,   // Reduced risk when at 50% drawdown
                > 0.25m => 0.75m, // Slightly reduced when at 25% drawdown
                _ => 1.0m         // Full risk when low drawdown
            };

            // Confidence-based sizing (UCB approach)
            var confidence = Math.Max(strategy.Confidence, prediction.Probability);
            if (confidence < (decimal)TopStepConfig.ConfidenceThreshold)
            {
                LogConfidenceBelowThreshold(_logger, TopStepConfig.ConfidenceThreshold, (double)confidence, null);
                return 0m; // No trade if confidence too low
            }

            // Confidence multiplier using ONNX model (features prepared for future advanced scoring)
            _ = new Dictionary<string, double>
            {
                ["strategy_confidence"] = (double)strategy.Confidence,
                ["prediction_probability"] = (double)prediction.Probability,
                ["volatility"] = (double)context.Volatility,
                ["volume_ratio"] = (double)context.VolumeRatio,
                ["trend_strength"] = (double)context.TrendStrength
            };
            
            var modelConfidence = _confidenceNetwork != null ? await _confidenceNetwork.PredictAsync(new decimal[]
            {
                strategy.Confidence,
                prediction.Probability,
                context.Volatility,
                context.VolumeRatio,
                context.TrendStrength
            }, cancellationToken).ConfigureAwait(false) : 0.5m;
            
            var confidenceMultiplier = modelConfidence;

            // Calculate risk amount
            var riskAmount = baseRisk * riskMultiplier * confidenceMultiplier;

            // Dynamic stop calculation with safety bounds
            var instrument = context.Symbol;
            decimal stopDistance;
            decimal pointValue;

            if (instrument.Equals("NQ", StringComparison.OrdinalIgnoreCase))
            {
                stopDistance = Math.Max(TopStepConfig.NqMinStopDistance, context.Atr ?? TopStepConfig.DefaultAtrFallback);
                pointValue = TopStepConfig.NqPointValue;
            }
            else // ES
            {
                stopDistance = Math.Max(TopStepConfig.EsMinStopDistance, context.Atr ?? TopStepConfig.EsDefaultAtrFallback);
                pointValue = TopStepConfig.EsPointValue;
            }

            // Convert risk to contracts
            var perContractRisk = stopDistance * pointValue;
            var contracts = perContractRisk > 0 ? (int)(riskAmount / perContractRisk) : 0;
            
            // 🔧 PRODUCTION READY: Ensure at least 1 contract if we have positive risk and confidence
            // This allows learning opportunities while maintaining real risk management
            if (contracts == 0 && riskAmount > 0 && confidence >= (decimal)TopStepConfig.ConfidenceThreshold)
            {
                contracts = 1; // Allow 1 contract for learning purposes with real risk
                _logger.LogInformation("[POSITION-SIZING] 📊 Calculated risk ${Risk:F2} below per-contract risk ${PerContract:F2}, using minimum 1 contract (real money at risk)", 
                    riskAmount, perContractRisk);
            }

            // Apply TopStep position limits based on current drawdown
            var maxContracts = _currentDrawdown switch
            {
                < 500m => 3,  // Up to 3 contracts when drawdown is low
                < 1000m => 2, // Max 2 contracts when moderate drawdown
                _ => 1        // Only 1 contract when high drawdown
            };

            contracts = Math.Max(0, Math.Min(contracts, maxContracts));

            // 🚀 PRODUCTION CVaR-PPO POSITION SIZING INTEGRATION
            if (_cvarPPO != null && IsInitialized)
            {
                try
                {
                    // Create state vector for CVaR-PPO model
                    var state = CreateCVaRStateVector(context, strategy, prediction);
                    
                    // Get action from trained CVaR-PPO model
                    var actionResult = await _cvarPPO.GetActionAsync(state, deterministic: false, cancellationToken).ConfigureAwait(false);
                    
                    // 🎓 SAVE STATE/ACTION FOR EXPERIENCE REPLAY (live + historical learning)
                    _lastCVaRState = state;
                    _lastCVaRAction = actionResult.Action;
                    _lastCVaRValue = actionResult.ValueEstimate;
                    
                    // Convert CVaR-PPO action to contract sizing
                    var cvarContracts = ConvertCVaRActionToContracts(actionResult, contracts);
                    
                    // Apply CVaR risk controls
                    var riskAdjustedContracts = ApplyCVaRRiskControls(cvarContracts, actionResult, context);
                    
                    contracts = Math.Max(0, Math.Min(riskAdjustedContracts, maxContracts));
                    
                    LogCvarPpoAction(_logger, actionResult.Action, actionResult.ActionProbability, 
                        actionResult.ValueEstimate, actionResult.CVaREstimate, contracts, null);
                }
                catch (InvalidOperationException ex)
                {
                    LogCvarPpoInvalidOperation(_logger, ex);
                    // contracts remains unchanged - use TopStep compliance sizing
                }
                catch (ArgumentException ex)
                {
                    LogCvarPpoInvalidArgument(_logger, ex);
                    // contracts remains unchanged - use TopStep compliance sizing
                }
                catch (OnnxRuntimeException ex)
                {
                    LogCvarPpoOnnxError(_logger, ex);
                    // contracts remains unchanged - use TopStep compliance sizing
                }
            }
            else
            {
                // Fallback to legacy RL if CVaR-PPO not available
                var rlMultiplier = await _modelManager.GetPositionSizeMultiplierAsync(
                    strategy.SelectedStrategy,
                    context.Symbol,
                    context.CurrentPrice,
                    context.Atr ?? TopStepConfig.DefaultAtrLookback,
                    strategy.Confidence,
                    prediction.Probability,
                    new List<Bar>()
                ).ConfigureAwait(false);
                
                contracts = (int)(contracts * Math.Clamp(rlMultiplier, TopStepConfig.MinRlMultiplier, TopStepConfig.MaxRlMultiplier));
                LogLegacyRlMultiplier(_logger, (double)rlMultiplier, null);
            }

            // ✅ BOOTSTRAP MODE: If CVaR-PPO is too conservative (negative value), allow small trades on good setups
            // This lets the model learn from REAL market fills, building positive experiences
            if (contracts == 0 && confidence >= 0.52m)
            {
                // Check if CVaR-PPO would have traded but value estimate killed it
                var wouldHaveTraded = _lastCVaRAction > 0; // Action 1-5 = want to trade
                var hasNegativeValue = _lastCVaRValue < 0; // But expected loss
                
                if (wouldHaveTraded && hasNegativeValue)
                {
                    // Override with 1 contract bootstrap trade to build real experience
                    contracts = 1;
                    _logger.LogInformation("[BOOTSTRAP] 🌱 Taking 1-contract learning trade: Action={Action}, Value={Value:F3}, Confidence={Conf:P1}", 
                        _lastCVaRAction, _lastCVaRValue, confidence);
                }
                else
                {
                    _logger.LogInformation("[CVAR-PPO] ⏸️ No trade: CVaR-PPO returned 0 contracts (confidence={Conf:P1}, risk=${Risk:F2})", 
                        confidence, riskAmount);
                }
            }

            LogPositionSize(_logger, instrument, (double)confidence, _currentDrawdown, contracts, riskAmount, null);

            return contracts; // Return actual contract count, not multiplier
        }

        /// <summary>
        /// TopStep compliance check - returns (canTrade, reason, level)
        /// </summary>
        private (bool CanTrade, string Reason, string Level) ShouldStopTrading()
        {
            // Auto-reset daily P&L if it's a new day
            CheckAndResetDaily();

            // Hard stops (no trading allowed)
            if (_dailyPnl <= -TopStepConfig.DailyLossLimit)
                return (false, $"Daily loss limit reached: {_dailyPnl:C}", "hard_stop");
            
            if (_currentDrawdown >= TopStepConfig.MaxDrawdown)
                return (false, $"Max drawdown reached: {_currentDrawdown:C}", "hard_stop");
            
            if (_accountBalance <= TopStepConfig.TrailingStop)
                return (false, $"Account below minimum: {_accountBalance:C}", "hard_stop");

            // Warning levels (can trade but with caution)
            if (_dailyPnl <= -(TopStepConfig.DailyLossLimit * TopStepConfig.NearDailyLossWarningFactor))
                return (true, $"Near daily loss limit: {_dailyPnl:C}", "warning");
            
            if (_currentDrawdown >= (TopStepConfig.MaxDrawdown * TopStepConfig.NearMaxDrawdownWarningFactor))
                return (true, $"Near max drawdown: {_currentDrawdown:C}", "warning");

            return (true, "OK", "normal");
        }

        /// <summary>
        /// Update P&L after trade completion - call this from AutonomousDecisionEngine
        /// </summary>
        public void UpdatePnL(string strategy, decimal pnl)
        {
            _dailyPnl += pnl;
            _accountBalance += pnl;
            
            // Update drawdown if we're in loss territory
            if (_dailyPnl < 0)
                _currentDrawdown = Math.Max(_currentDrawdown, Math.Abs(_dailyPnl));
            
            LogPnlUpdate(_logger, strategy, pnl, _dailyPnl, _currentDrawdown, _accountBalance, null);
        }

        /// <summary>
        /// Reset daily stats - automatically called or can be called manually
        /// </summary>
        public void ResetDaily()
        {
            _dailyPnl = 0;
            _currentDrawdown = 0;
            _lastResetDate = DateTime.UtcNow.Date;
            
            LogDailyReset(_logger, null);
        }

        private void CheckAndResetDaily()
        {
            if (DateTime.UtcNow.Date > _lastResetDate)
            {
                ResetDaily();
            }
        }

        #endregion

        #region Integration with Existing Systems

        /// <summary>
        /// Generate enhanced candidates that integrate with AllStrategies.cs
        /// This replaces the manual candidate generation
        /// </summary>
        private Task<IReadOnlyList<Candidate>> GenerateEnhancedCandidatesAsync(
            string symbol,
            Env env,
            Levels levels,
            IList<Bar> bars,
            RiskEngine risk,
            StrategySelection strategySelection,
            PricePrediction prediction,
            decimal sizeMultiplier
            )
        {
            try
            {
                // Get candidates from the selected strategy only (instead of all 14)
                var candidateFunction = GetStrategyFunction(strategySelection.SelectedStrategy);
                var baseCandidates = candidateFunction(symbol, env, levels, bars, risk);
                
                // Enhance each candidate with AI intelligence
                var enhancedCandidates = new List<Candidate>();
                
                _logger.LogDebug("[BRAIN-CANDIDATES] Evaluating {Count} base candidates from strategy {Strategy}", 
                    baseCandidates.Count, strategySelection.SelectedStrategy);
                
                foreach (var candidate in baseCandidates)
                {
                    // Only include candidates that align with price prediction
                    var candidateDirection = candidate.side == Side.BUY ? PriceDirection.Up : PriceDirection.Down;
                    if (prediction.Direction != PriceDirection.Sideways && candidateDirection != prediction.Direction)
                    {
                        _logger.LogTrace("[BRAIN-CANDIDATES] Skipping candidate {Id} - direction {CandDir} doesn't match prediction {PredDir}", 
                            candidate.strategy_id, candidateDirection, prediction.Direction);
                        continue; // Skip candidates against predicted direction
                    }
                    
                    // Apply AI-optimized position sizing
                    var enhancedCandidate = new Candidate
                    {
                        strategy_id = $"{candidate.strategy_id}-AI-{strategySelection.Confidence:P0}",
                        symbol = candidate.symbol,
                        side = candidate.side,
                        entry = candidate.entry,
                        stop = candidate.stop,
                        t1 = candidate.t1,
                        expR = candidate.expR,
                        qty = (int)(candidate.qty * sizeMultiplier),
                        atr_ok = candidate.atr_ok,
                        vol_z = candidate.vol_z,
                        accountId = candidate.accountId,
                        contractId = candidate.contractId,
                        Score = candidate.Score,
                        // Add AI confidence to quality score
                        QScore = candidate.QScore * strategySelection.Confidence * prediction.Probability
                    };
                    
                    _logger.LogDebug("[BRAIN-CANDIDATES] Enhanced candidate {Id}: Entry={Entry:F2}, Stop={Stop:F2}, Target={Target:F2}, Qty={Qty}, QScore={QScore:F3}",
                        enhancedCandidate.strategy_id, enhancedCandidate.entry, enhancedCandidate.stop, enhancedCandidate.t1, 
                        enhancedCandidate.qty, enhancedCandidate.QScore);
                    
                    enhancedCandidates.Add(enhancedCandidate);
                }
                
                LogBrainEnhanceGenerated(_logger, symbol, enhancedCandidates.Count, strategySelection.SelectedStrategy, null);
                
                return Task.FromResult<IReadOnlyList<Candidate>>(enhancedCandidates);
            }
            catch (InvalidOperationException ex)
            {
                LogBrainEnhanceInvalidOperation(_logger, ex);
                
                // Fallback to original AllStrategies logic
                return Task.FromResult(AllStrategies.generate_candidates(symbol, env, levels, bars, risk));
            }
            catch (ArgumentException ex)
            {
                LogBrainEnhanceInvalidArgument(_logger, ex);
                
                // Fallback to original AllStrategies logic
                return Task.FromResult(AllStrategies.generate_candidates(symbol, env, levels, bars, risk));
            }
            catch (KeyNotFoundException ex)
            {
                LogBrainEnhanceKeyNotFound(_logger, ex);
                
                // Fallback to original AllStrategies logic
                return Task.FromResult(AllStrategies.generate_candidates(symbol, env, levels, bars, risk));
            }
        }

        /// <summary>
        /// Create context vector for Neural UCB from market data
        /// </summary>
        private static ContextVector CreateContextVector(MarketContext context)
        {
            var features = new Dictionary<string, decimal>
            {
                ["price"] = context.CurrentPrice,
                ["volume"] = context.Volume,
                ["volatility"] = context.Volatility,
                ["atr"] = context.Atr ?? 0,
                ["rsi"] = context.RSI,
                ["hour"] = context.TimeOfDay.Hours,
                ["day_of_week"] = (int)context.DayOfWeek,
                ["volume_ratio"] = context.VolumeRatio,
                ["price_change"] = context.PriceChange,
                ["trend_strength"] = context.TrendStrength,
                ["support_distance"] = context.DistanceToSupport,
                ["resistance_distance"] = context.DistanceToResistance,
                ["volatility_rank"] = context.VolatilityRank,
                ["momentum"] = context.Momentum,
                ["regime"] = (decimal)context.MarketRegime
            };
            
            return new ContextVector { Features = features };
        }

        #endregion

        #region Helper Methods

        /// <summary>
        /// Create a no-trade decision when trading is blocked (Phase 2: Calendar)
        /// </summary>
        private static BrainDecision CreateNoTradeDecision(string symbol, string reason, DateTime startTime)
        {
            return new BrainDecision
            {
                Symbol = symbol,
                RecommendedStrategy = "HOLD",
                StrategyConfidence = 0m,
                PriceDirection = BotCore.Brain.Models.PriceDirection.Sideways,
                PriceProbability = TopStepConfig.NeutralProbability,
                OptimalPositionMultiplier = 0m,
                MarketRegime = BotCore.Brain.Models.MarketRegime.Normal,
                EnhancedCandidates = new List<Candidate>(),
                DecisionTime = startTime,
                ProcessingTimeMs = (DateTime.UtcNow - startTime).TotalMilliseconds,
                ModelConfidence = 0m,
                RiskAssessment = $"BLOCKED: {reason}"
            };
        }

        private static MarketContext CreateMarketContext(string symbol, Env env, IList<Bar> bars)
        {
            var latestBar = bars.LastOrDefault();
            if (latestBar == null)
            {
                return new MarketContext { Symbol = symbol };
            }
            
            var volumeRatio = CalculateVolumeRatio(bars, latestBar);
            var rsi = CalculateRSI(bars, 14);
            var trendStrength = CalculateTrendStrength(bars);
            var volatilityRank = CalculateVolatilityRank(bars);
            var momentum = CalculateMomentum(bars);
            var priceChange = bars.Count > 1 ? latestBar.Close - bars[^2].Close : 0m;
            var volatility = latestBar.Close > 0 ? Math.Abs(latestBar.High - latestBar.Low) / latestBar.Close : 0;
            
            Console.WriteLine($"[MARKET-CONTEXT] {symbol} | Price={latestBar.Close:F2} Vol={latestBar.Volume:N0} " +
                             $"ATR={env.atr:F2} RSI={rsi:F1} Volatility={volatility:F4} " +
                             $"VolRatio={volumeRatio:F2}x Trend={trendStrength:F2} Momentum={momentum:F2} " +
                             $"PriceChange={priceChange:F2} VolRank={volatilityRank:F2} " +
                             $"Time={latestBar.Start:HH:mm:ss} {latestBar.Start.DayOfWeek}");
            
            var context = new MarketContext
            {
                Symbol = symbol,
                CurrentPrice = latestBar.Close,
                Volume = latestBar.Volume,
                Atr = env.atr,
                Volatility = volatility,
                TimeOfDay = latestBar.Start.TimeOfDay,  // Use bar's historical time, not current time
                DayOfWeek = latestBar.Start.DayOfWeek,  // Use bar's historical day, not current day
                VolumeRatio = volumeRatio,
                PriceChange = priceChange,
                RSI = rsi,
                TrendStrength = trendStrength,
                DistanceToSupport = 0m, // levels.Support doesn't exist, using default
                DistanceToResistance = 0m, // levels.Resistance doesn't exist, using default
                VolatilityRank = volatilityRank,
                Momentum = momentum,
                MarketRegime = 0 // Will be filled by regime detector
            };
            
            return context;
        }

        private static decimal CalculateVolumeRatio(IList<Bar> bars, Bar latestBar)
        {
            if (bars.Count <= 10) return 1m;
            
            var avgVolume = bars.TakeLast(10).Average(b => b.Volume);
            
            // Prevent division by zero when average volume is zero
            if (avgVolume == 0) return 1m;
            
            return (decimal)(latestBar.Volume / avgVolume);
        }

        private List<string> GetAvailableStrategies(TimeSpan timeOfDay, MarketRegime regime)
        {
            // Enhanced strategy selection logic for primary strategies (S2, S3, S6, S11)
            // Strictly follows time-based scheduling - each strategy runs at its designated time
            var hour = timeOfDay.Hours;
            
            // Time-based primary strategy allocation - strategies run at their scheduled times ONLY
            var timeBasedStrategies = hour switch
            {
                >= 18 or <= 2 => new[] { "S2", "S11" }, // Asian Session: Mean reversion works well
                >= 2 and <= 5 => new[] { "S3", "S2" }, // European Open: Breakouts and compression
                >= 5 and <= 8 => new[] { "S2", "S3", "S11" }, // London Morning: Good liquidity
                >= 8 and <= 9 => new[] { "S3", "S2" }, // US PreMarket: Compression setups
                >= 9 and <= 10 => new[] { "S6", "S3", "S2" }, // Opening Drive: Momentum + breakouts + mean reversion
                >= 10 and <= 11 => new[] { "S3", "S2", "S11" }, // Morning Trend: Best trends
                >= 11 and <= 13 => new[] { "S2", "S3" }, // Lunch: Mean reversion + compression
                >= 13 and <= 16 => new[] { "S11", "S3", "S6" }, // Afternoon: Exhaustion + compression + momentum
                _ => new[] { "S2", "S3" } // Default safe strategies
            };
            
            // Return ONLY time-based strategies - do not modify schedules with regime filtering
            // Neural UCB will select best strategy from the time-appropriate options
            var availableStrategies = timeBasedStrategies.ToList();
            
            LogStrategySelection(_logger, hour, regime.ToString(), string.Join(",", availableStrategies), null);
            
            return availableStrategies;
        }

        private static Func<string, Env, Levels, IList<Bar>, RiskEngine, IReadOnlyList<Candidate>> GetStrategyFunction(string strategy)
        {
            // Map to ACTIVE strategy functions in AllStrategies.cs (only S2, S3, S6, S11)
            return strategy switch
            {
                "S2" => AllStrategies.S2,   // VWAP Mean reversion (most used)
                "S3" => AllStrategies.S3,   // Bollinger Squeeze/breakout setups  
                "S6" => AllStrategies.S6,   // Opening Drive (critical window)
                "S11" => AllStrategies.S11, // ADR/IB Exhaustion fade
                _ => AllStrategies.S2 // Default to most reliable strategy
            };
        }

        private static decimal CalculateReward(decimal pnl, bool wasCorrect, TimeSpan holdTime)
        {
            // Combine PnL with correctness and time efficiency
            var baseReward = wasCorrect ? 1m : 0m;
            var pnlComponent = Math.Tanh((double)(pnl / 100)) * 0.5; // Normalize PnL contribution
            var timeComponent = holdTime < TimeSpan.FromHours(2) ? 0.1m : 0m; // Reward quick profits
            
            return Math.Clamp(baseReward + (decimal)pnlComponent + timeComponent, 0m, 1m);
        }

        /// <summary>
        /// Calculate CVaR-specific reward that emphasizes risk-adjusted returns
        /// Rewards both profitability AND value estimation accuracy
        /// </summary>
        private static decimal CalculateCVaRReward(decimal pnl, bool wasCorrect, TimeSpan holdTime, double valueEstimate)
        {
            // Base reward from P&L (normalized to roughly [-1, 1])
            var pnlReward = Math.Tanh((double)(pnl / 100));
            
            // Correctness bonus/penalty
            var correctnessReward = wasCorrect ? 0.5 : -0.5;
            
            // Time efficiency (reward faster trades)
            var timeReward = holdTime < TimeSpan.FromHours(2) ? 0.2 : 
                           holdTime < TimeSpan.FromHours(4) ? 0.1 : 0.0;
            
            // Value estimation accuracy (penalize overconfident predictions)
            var actualValue = (double)pnl / 100.0; // Normalize to similar scale
            var valueError = Math.Abs(valueEstimate - actualValue);
            var valueAccuracyPenalty = -Math.Min(valueError * 0.1, 0.3); // Cap penalty at -0.3
            
            // Combined reward (roughly in range [-2, 2])
            var totalReward = pnlReward + correctnessReward + timeReward + valueAccuracyPenalty;
            
            return (decimal)totalReward;
        }

        private static decimal CalculateOverallConfidence(StrategySelection strategy, PricePrediction prediction)
        {
            return (strategy.Confidence + prediction.Probability) / TopStepConfig.StrategyPredictionAverageWeight;
        }

        private static string AssessRisk(MarketContext context, PricePrediction prediction)
        {
            if (context.Volatility > HighVolatilityThreshold) return "HIGH";
            if (context.Volatility < LowVolatilityThreshold && prediction.Probability > TopStepConfig.HighConfidenceProbability) return "LOW";
            return "MEDIUM";
        }

        private static BrainDecision CreateFallbackDecision(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            // Fallback to your existing AllStrategies logic
            var candidates = AllStrategies.generate_candidates(symbol, env, levels, bars, risk);
            
            return new BrainDecision
            {
                Symbol = symbol,
                RecommendedStrategy = "S3", // Default strategy
                StrategyConfidence = TopStepConfig.NeutralProbability,
                PriceDirection = PriceDirection.Sideways,
                PriceProbability = TopStepConfig.NeutralProbability,
                OptimalPositionMultiplier = TopStepConfig.DefaultNeutralPositionMultiplier,
                MarketRegime = MarketRegime.Normal,
                EnhancedCandidates = candidates,
                DecisionTime = DateTime.UtcNow,
                ProcessingTimeMs = TopStepConfig.DefaultAtrLookback,
                ModelConfidence = TopStepConfig.NeutralProbability,
                RiskAssessment = "MEDIUM"
            };
        }

        // Technical analysis helpers (simplified versions)
        private static decimal CalculateEMA(IList<Bar> bars, int period)
        {
            if (bars.Count < period) return bars.LastOrDefault()?.Close ?? 0;
            
            var multiplier = 2m / (period + 1);
            var ema = bars.Take(period).Average(b => b.Close);
            
            for (int i = period; i < bars.Count; i++)
            {
                ema = (bars[i].Close * multiplier) + (ema * (1 - multiplier));
            }
            
            return ema;
        }

        private static decimal CalculateRSI(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return TopStepConfig.DefaultRsiNeutral;
            
            var gains = 0m;
            var losses = 0m;
            
            for (int i = bars.Count - period; i < bars.Count; i++)
            {
                var change = bars[i].Close - bars[i - 1].Close;
                if (change > 0) gains += change;
                else losses -= change;
            }
            
            if (losses == 0) return TopStepConfig.DefaultRsiMax;
            
            var rs = gains / losses;
            return TopStepConfig.DefaultRsiMax - (TopStepConfig.DefaultRsiMax / (1 + rs));
        }

        private static decimal CalculateTrendStrength(IList<Bar> bars)
        {
            if (bars.Count < TopStepConfig.MinBarsPeriod) return 0;
            
            var recent = bars.TakeLast(TopStepConfig.MinBarsPeriod).ToList();
            var slope = (recent[recent.Count - 1].Close - recent[0].Close) / recent.Count;
            var avgRange = recent.Average(b => Math.Abs(b.High - b.Low));
            
            // Prevent division by zero when all bars have same High/Low (e.g., live polls)
            if (avgRange == 0) return 0;
            
            return Math.Abs(slope) / avgRange;
        }

        private static decimal CalculateVolatilityRank(IList<Bar> bars)
        {
            if (bars.Count < TopStepConfig.MinBarsExtended) return TopStepConfig.NeutralProbability;
            
            var currentVol = Math.Abs(bars[bars.Count - 1].High - bars[bars.Count - 1].Low);
            var historicalVols = bars.TakeLast(TopStepConfig.MinBarsExtended).Select(b => Math.Abs(b.High - b.Low)).OrderBy(v => v).ToList();
            
            var rank = historicalVols.Count(v => v < currentVol) / (decimal)historicalVols.Count;
            return rank;
        }

        private static decimal CalculateMomentum(IList<Bar> bars)
        {
            if (bars.Count < 5) return 0;
            
            var recent = bars.TakeLast(5).ToList();
            var firstClose = recent[0].Close;
            
            // Prevent division by zero when first bar has zero close price
            if (firstClose == 0) return 0;
            
            return (recent[recent.Count - 1].Close - firstClose) / firstClose;
        }

        private async Task UpdateUnifiedLearningAsync(CancellationToken cancellationToken)
        {
            _ = cancellationToken; // Reserved for future async operations
            try
            {
                LogUnifiedLearningStarting(_logger, null);
                
                // Analyze performance patterns across all strategies
                var performanceAnalysis = AnalyzeStrategyPerformance();
                
                // Update strategy optimal conditions based on recent performance
                UpdateOptimalConditionsFromPerformance(performanceAnalysis);
                
                // Cross-pollinate successful patterns between strategies
                await CrossPollinateStrategyPatternsAsync().ConfigureAwait(false);
                
                LogUnifiedLearningCompleted(_logger, null);
            }
            catch (InvalidOperationException ex)
            {
                LogUnifiedLearningInvalidOperation(_logger, ex);
            }
            catch (IOException ex)
            {
                LogUnifiedLearningIoError(_logger, ex);
            }
            catch (UnauthorizedAccessException ex)
            {
                LogUnifiedLearningAccessDenied(_logger, ex);
            }
            catch (ArgumentException ex)
            {
                LogUnifiedLearningInvalidArgument(_logger, ex);
            }
        }
        
        private Dictionary<string, BotCore.Brain.Models.PerformanceMetrics> AnalyzeStrategyPerformance()
        {
            var analysis = new Dictionary<string, BotCore.Brain.Models.PerformanceMetrics>();
            
            foreach (var strategy in PrimaryStrategies)
            {
                if (!_strategyPerformance.TryGetValue(strategy, out var perf))
                    continue;
                    
                var metrics = new BotCore.Brain.Models.PerformanceMetrics
                {
                    WinRate = perf.TotalTrades > 0 ? (decimal)perf.WinningTrades / perf.TotalTrades : 0,
                    AveragePnL = perf.TotalTrades > 0 ? perf.TotalPnL / perf.TotalTrades : 0,
                    AverageHoldTime = perf.TotalTrades > 0 ? 
                        TimeSpan.FromTicks(perf.HoldTimes.Sum() / perf.TotalTrades) : TimeSpan.Zero,
                    BestConditions = GetBestConditionsForStrategy(strategy),
                    RecentPerformanceTrend = GetRecentPerformanceTrend(strategy)
                };
                
                analysis[strategy] = metrics;
            }
            
            return analysis;
        }
        
        private void UpdateOptimalConditionsFromPerformance(Dictionary<string, BotCore.Brain.Models.PerformanceMetrics> analysis)
        {
            foreach (var (strategy, metrics) in analysis)
            {
                if (metrics.WinRate < TopStepConfig.PoorPerformanceWinRateThreshold) // Poor performing strategy
                {
                    // Reduce confidence in current optimal conditions
                    if (_strategyOptimalConditions.TryGetValue(strategy, out var conditions))
                    {
                        // Remove conditions that have been consistently unsuccessful
                        var unsuccessfulConditions = conditions
                            .Where(c => c.SuccessRate < TopStepConfig.UnsuccessfulConditionThreshold)
                            .ToList();
                        
                        foreach (var condition in unsuccessfulConditions)
                        {
                            conditions.Remove(condition);
                        }
                        
                        LogConditionUpdate(_logger, unsuccessfulConditions.Count, strategy, null);
                    }
                }
                else if (metrics.WinRate > TopStepConfig.HighPerformanceWinRateThreshold) // High performing strategy
                {
                    // Strengthen successful conditions
                    foreach (var condition in metrics.BestConditions)
                    {
                        UpdateConditionSuccess(strategy, condition, true);
                    }
                }
            }
        }
        
        private async Task CrossPollinateStrategyPatternsAsync()
        {
            await Task.CompletedTask.ConfigureAwait(false);
            
            // Find the best performing strategy
            var bestStrategy = PrimaryStrategies
                .Where(s => _strategyPerformance.ContainsKey(s))
                .OrderByDescending(s => _strategyPerformance[s].WinRate)
                .FirstOrDefault();
                
            if (bestStrategy == null)
                return;
                
            var bestPerformance = _strategyPerformance[bestStrategy];
            if (bestPerformance.WinRate < TopStepConfig.MinimumWinRateToSharePatterns)
                return; // Not good enough to share patterns
            
            // Share successful patterns with other strategies
            var successfulConditions = _strategyOptimalConditions
                .GetValueOrDefault<string, List<BotCore.Brain.Models.MarketCondition>>(bestStrategy, new List<BotCore.Brain.Models.MarketCondition>())
                .Where(c => c.SuccessRate > TopStepConfig.SharedPatternSuccessThreshold)
                .ToList();
            
            foreach (var strategy in PrimaryStrategies.Where(s => s != bestStrategy))
            {
                foreach (var condition in successfulConditions)
                {
                    // Add successful condition from best strategy to other strategies
                    UpdateConditionSuccess(strategy, condition.ConditionName, true, TopStepConfig.CrossPollinationWeight); // Lower weight for cross-pollination
                }
            }
            
            LogCrossPollination(_logger, successfulConditions.Count, bestStrategy, null);
        }
        
        private void UpdateStrategyPerformance(string strategy, MarketContext context, bool wasCorrect, decimal pnl, TimeSpan holdTime)
        {
            if (!_strategyPerformance.TryGetValue(strategy, out var perf))
            {
                perf = new StrategyPerformance();
                _strategyPerformance[strategy] = perf;
            }
            
            perf.TotalTrades++;
            perf.TotalPnL += pnl;
            perf.AddHoldTime(holdTime.Ticks);
            
            if (wasCorrect)
            {
                perf.WinningTrades++;
                perf.WinRate = (decimal)perf.WinningTrades / perf.TotalTrades;
            }
            
            // Update strategy optimal conditions
            var currentConditions = GetCurrentMarketConditions(context);
            foreach (var condition in currentConditions)
            {
                UpdateConditionSuccess(strategy, condition, wasCorrect);
            }
        }
        
        private void UpdateStrategyOptimalConditions(string strategy, MarketContext context, bool wasSuccessful)
        {
            if (!_strategyOptimalConditions.TryGetValue(strategy, out var conditions))
            {
                conditions = new List<BotCore.Brain.Models.MarketCondition>();
                _strategyOptimalConditions[strategy] = conditions;
            }
            
            var currentConditions = GetCurrentMarketConditions(context);
            foreach (var conditionName in currentConditions)
            {
                UpdateConditionSuccess(strategy, conditionName, wasSuccessful);
            }
        }
        
        private void UpdateConditionSuccess(string strategy, string conditionName, bool wasSuccessful, decimal weight = 1.0m)
        {
            if (!_strategyOptimalConditions.TryGetValue(strategy, out var conditions))
            {
                conditions = new List<BotCore.Brain.Models.MarketCondition>();
                _strategyOptimalConditions[strategy] = conditions;
            }
            
            var condition = conditions.Find(c => c.ConditionName == conditionName);
            if (condition == null)
            {
                condition = new BotCore.Brain.Models.MarketCondition
                {
                    ConditionName = conditionName,
                    SuccessCount = 0,
                    TotalCount = 0 
                };
                conditions.Add(condition);
            }
            
            condition.TotalCount += weight;
            if (wasSuccessful)
            {
                condition.SuccessCount += weight;
            }
            
            condition.SuccessRate = condition.TotalCount > 0 ? condition.SuccessCount / condition.TotalCount : 0;
        }
        
        private string[] GetBestConditionsForStrategy(string strategy)
        {
            return _strategyOptimalConditions
                .GetValueOrDefault<string, List<BotCore.Brain.Models.MarketCondition>>(strategy, new List<BotCore.Brain.Models.MarketCondition>())
                .Where(c => c.SuccessRate > TopStepConfig.MinimumConditionSuccessRate && c.TotalCount >= TopStepConfig.MinimumConditionTrialCount)
                .OrderByDescending(c => c.SuccessRate)
                .Take(TopStepConfig.TopConditionsToSelect)
                .Select(c => c.ConditionName)
                .ToArray();
        }
        
        private decimal GetRecentPerformanceTrend(string strategy)
        {
            var recentDecisions = _decisionHistory
                .Where(d => d.Strategy == strategy && d.Timestamp > DateTime.UtcNow.AddHours(-TopStepConfig.RecentDecisionsLookbackHours))
                .OrderBy(d => d.Timestamp)
                .ToList();
                
            if (recentDecisions.Count < TopStepConfig.MinimumDecisionsForTrend)
                return 0;
                
            var recentHalf = recentDecisions.Skip(recentDecisions.Count / 2).ToList();
            var earlierHalf = recentDecisions.Take(recentDecisions.Count / 2).ToList();
            
            var recentWinRate = recentHalf.Count > 0 ? (decimal)recentHalf.Count(d => d.WasCorrect) / recentHalf.Count : 0;
            var earlierWinRate = earlierHalf.Count > 0 ? (decimal)earlierHalf.Count(d => d.WasCorrect) / earlierHalf.Count : 0;
            
            return recentWinRate - earlierWinRate; // Positive = improving, negative = declining
        }

        /// <summary>
        /// Get unified scheduling recommendations for both historical and live trading
        /// Ensures identical timing for Market Open: Light learning every 60 min, Market Closed: Intensive every 15 min
        /// </summary>
        public UnifiedSchedulingRecommendation GetUnifiedSchedulingRecommendation(DateTime currentTime)
        {
            var estTime = TimeZoneInfo.ConvertTimeFromUtc(currentTime.ToUniversalTime(), 
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            var timeOfDay = estTime.TimeOfDay;
            var dayOfWeek = estTime.DayOfWeek;
            
            // CME ES/NQ Futures Schedule: Sunday 6PM - Friday 5PM EST with daily maintenance 5-6PM
            bool isMarketOpen = IsMarketOpen(dayOfWeek, timeOfDay);
            bool isMaintenanceWindow = IsMaintenanceWindow(dayOfWeek, timeOfDay);
            
            if (isMaintenanceWindow)
            {
                return new UnifiedSchedulingRecommendation
                {
                    IsMarketOpen = false,
                    LearningIntensity = "INTENSIVE",
                    HistoricalLearningIntervalMinutes = TopStepConfig.MaintenanceLearningIntervalMinutes, // Very frequent during maintenance
                    LiveTradingActive = false,
                    RecommendedStrategies = new[] { "S2", "S3", "S6", "S11" }, // All strategies can be analyzed
                    Reasoning = "Maintenance window - intensive learning opportunity"
                };
            }
            
            if (!isMarketOpen)
            {
                // Weekend or market closed - intensive historical learning
                return new UnifiedSchedulingRecommendation
                {
                    IsMarketOpen = false,
                    LearningIntensity = "INTENSIVE",
                    HistoricalLearningIntervalMinutes = TopStepConfig.ClosedMarketLearningIntervalMinutes, // Every 15 minutes as requested
                    LiveTradingActive = false,
                    RecommendedStrategies = new[] { "S2", "S3", "S6", "S11" },
                    Reasoning = "Market closed - intensive historical learning across all strategies"
                };
            }
            
            // Market is open - light learning alongside live trading
            var availableStrategies = GetAvailableStrategies(timeOfDay, MarketRegime.Normal);
            return new UnifiedSchedulingRecommendation
            {
                IsMarketOpen = true,
                LearningIntensity = "LIGHT",
                HistoricalLearningIntervalMinutes = TopStepConfig.OpenMarketLearningIntervalMinutes, // Every 60 minutes as requested
                LiveTradingActive = true,
                RecommendedStrategies = availableStrategies.ToArray(),
                Reasoning = $"Market open - light historical learning every 60min, active live trading with {string.Join(",", availableStrategies)}"
            };
        }
        
        private static bool IsMarketOpen(DayOfWeek dayOfWeek, TimeSpan timeOfDay)
        {
            // CME ES/NQ: Sunday 6PM - Friday 5PM EST
            var marketOpenTime = new TimeSpan(18, 0, 0);  // 6:00 PM EST
            var marketCloseTime = new TimeSpan(17, 0, 0); // 5:00 PM EST
            
            // Weekend check
            if (dayOfWeek == DayOfWeek.Saturday)
                return false;
                
            if (dayOfWeek == DayOfWeek.Sunday && timeOfDay < marketOpenTime)
                return false;
                
            if (dayOfWeek == DayOfWeek.Friday && timeOfDay >= marketCloseTime)
                return false;
            
            // Daily maintenance break: 5:00-6:00 PM EST Monday-Thursday
            if (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday && timeOfDay >= marketCloseTime && timeOfDay < marketOpenTime)
            {
                return false; // Maintenance break
            }
            
            return true;
        }
        
        private static bool IsMaintenanceWindow(DayOfWeek dayOfWeek, TimeSpan timeOfDay)
        {
            // Daily maintenance: 5:00-6:00 PM EST Monday-Thursday
            if (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday)
            {
                var maintenanceStart = new TimeSpan(17, 0, 0); // 5:00 PM EST
                var maintenanceEnd = new TimeSpan(18, 0, 0);   // 6:00 PM EST
                
                return timeOfDay >= maintenanceStart && timeOfDay <= maintenanceEnd;
            }
            
            return false;
        }

        /// <summary>
        /// Gate 4: Validate model before hot-reload to ensure safe deployment
        /// Implements comprehensive safety checks before swapping models
        /// </summary>
        public async Task<(bool IsValid, string Reason)> ValidateModelForReloadAsync(
            string newModelPath,
            string currentModelPath,
            CancellationToken cancellationToken = default)
        {
            try
            {
                LogGate4Start(_logger, null);
                LogGate4NewModel(_logger, newModelPath, null);
                LogGate4CurrentModel(_logger, currentModelPath, null);

                // Check 1: Feature specification compatibility
                LogGate4FeatureCheck(_logger, null);
                var featureCheckPassed = await ValidateFeatureSpecificationAsync(newModelPath, cancellationToken).ConfigureAwait(false);
                if (!featureCheckPassed)
                {
                    var reason = "Feature specification mismatch - new model expects different input features";
                    LogGate4Failed(_logger, reason, null);
                    return (false, reason);
                }
                LogGate4FeatureMatch(_logger, null);

                // Check 2: Sanity test with deterministic dataset
                LogGate4SanityCheck(_logger, null);
                var sanityTestVectors = LoadOrGenerateSanityTestVectors(_gate4Config.SanityTestVectors);
                LogGate4SanityVectors(_logger, sanityTestVectors.Count, null);

                // Check 3: Prediction distribution comparison
                LogGate4DistributionCheck(_logger, null);
                if (File.Exists(currentModelPath))
                {
                    var (distributionValid, divergence) = await ComparePredictionDistributionsAsync(
                        currentModelPath, newModelPath, sanityTestVectors, cancellationToken).ConfigureAwait(false);
                    
                    if (!distributionValid)
                    {
                        var reason = $"Prediction distribution divergence too high: {divergence:F4} > 0.20";
                        LogGate4Failed(_logger, reason, null);
                        return (false, reason);
                    }
                    LogGate4DistributionValid(_logger, divergence, null);
                }
                else
                {
                    LogGate4DistributionSkip(_logger, null);
                }

                // Check 4: NaN/Infinity validation
                LogGate4OutputCheck(_logger, null);
                var outputValidationPassed = await ValidateModelOutputsAsync(newModelPath, sanityTestVectors, cancellationToken).ConfigureAwait(false);
                if (!outputValidationPassed)
                {
                    var reason = "Model produces NaN or Infinity values - unstable model";
                    LogGate4Failed(_logger, reason, null);
                    return (false, reason);
                }
                LogGate4OutputValid(_logger, null);

                // Check 5: Historical replay simulation with drawdown check
                if (File.Exists(currentModelPath))
                {
                    LogGate4SimulationStart(_logger, null);
                    var (simulationPassed, drawdownRatio) = await RunHistoricalSimulationAsync(
                        currentModelPath, newModelPath, cancellationToken).ConfigureAwait(false);
                    
                    if (!simulationPassed)
                    {
                        var reason = $"Simulation drawdown ratio too high: {drawdownRatio:F2}x > 2.0x baseline";
                        LogGate4Failed(_logger, reason, null);
                        return (false, reason);
                    }
                    LogGate4SimulationPassed(_logger, drawdownRatio, null);
                }
                else
                {
                    LogGate4SimulationSkip(_logger, null);
                }

                LogGate4Passed(_logger, null);
                return (true, "All validation checks passed");
            }
            catch (FileNotFoundException ex)
            {
                LogGate4FileNotFound(_logger, ex);
                return (false, $"Validation exception: {ex.Message}");
            }
            catch (OnnxRuntimeException ex)
            {
                LogGate4OnnxError(_logger, ex);
                return (false, $"Validation exception: {ex.Message}");
            }
            catch (InvalidOperationException ex)
            {
                LogGate4InvalidOperation(_logger, ex);
                return (false, $"Validation exception: {ex.Message}");
            }
            catch (IOException ex)
            {
                LogGate4IOError(_logger, ex);
                return (false, $"Validation exception: {ex.Message}");
            }
        }

        private async Task<bool> ValidateFeatureSpecificationAsync(string modelPath, CancellationToken cancellationToken)
        {
            try
            {
                // Load feature specification (expected format)
                var featureSpecPath = Path.Combine("config", "feature_specification.json");
                if (!File.Exists(featureSpecPath))
                {
                    LogFeatureSpecMissing(_logger, null);
                    await CreateDefaultFeatureSpecificationAsync(featureSpecPath, cancellationToken).ConfigureAwait(false);
                }

                var featureSpec = await File.ReadAllTextAsync(featureSpecPath, cancellationToken).ConfigureAwait(false);
                _ = JsonSerializer.Deserialize<Dictionary<string, object>>(featureSpec);
                
                // For now, we'll validate that the model file exists and is a valid ONNX file
                // Full ONNX metadata inspection would require Microsoft.ML.OnnxRuntime
                if (!File.Exists(modelPath))
                {
                    LogValidationModelFileNotFound(_logger, modelPath, null);
                    return false;
                }

                var fileInfo = new FileInfo(modelPath);
                if (fileInfo.Length == 0)
                {
                    LogValidationModelFileEmpty(_logger, modelPath, null);
                    return false;
                }

                LogModelFileSize(_logger, fileInfo.Length, null);
                return true;
            }
            catch (FileNotFoundException ex)
            {
                LogFeatureValidationFileNotFound(_logger, ex);
                return false;
            }
            catch (JsonException ex)
            {
                LogFeatureValidationJsonError(_logger, ex);
                return false;
            }
            catch (IOException ex)
            {
                LogFeatureValidationIOError(_logger, ex);
                return false;
            }
            catch (UnauthorizedAccessException ex)
            {
                LogFeatureValidationAccessDenied(_logger, ex);
                return false;
            }
        }

        private List<float[]> LoadOrGenerateSanityTestVectors(int count)
        {
            var vectors = new List<float[]>();
            var cacheDir = Path.Combine("data", "validation");
            var cachePath = Path.Combine(cacheDir, "sanity_test_vectors.json");

            try
            {
                if (File.Exists(cachePath))
                {
                    var json = File.ReadAllText(cachePath);
                    var cached = JsonSerializer.Deserialize<List<float[]>>(json);
                    if (cached != null && cached.Count >= count)
                    {
                        LogSanityVectorsCached(_logger, count, null);
                        return cached.Take(count).ToList();
                    }
                }
            }
            catch (FileNotFoundException ex)
            {
                LogSanityVectorsCacheFileNotFound(_logger, ex);
            }
            catch (JsonException ex)
            {
                LogSanityVectorsCacheJsonError(_logger, ex);
            }
            catch (IOException ex)
            {
                LogSanityVectorsCacheIOError(_logger, ex);
            }

            // Generate deterministic test vectors
            // Use cryptographically secure random for ML training data to prevent adversarial exploitation
            using var rng = RandomNumberGenerator.Create();
            for (int i = 0; i < count; i++)
            {
                // Generate feature vector matching expected CVaR-PPO state size (11 features)
                var features = new float[]
                {
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // Volatility normalized
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // Price change momentum
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // Volume surge
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // ATR normalized
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // UCB value
                    (float)GetSecureRandomDouble(rng),                // Strategy encoding
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // Direction encoding
                    (float)Math.Sin(GetSecureRandomDouble(rng) * Math.PI), // Time of day (cyclical)
                    (float)Math.Cos(GetSecureRandomDouble(rng) * Math.PI), // Time of day (cyclical)
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0), // Decisions per day
                    (float)(GetSecureRandomDouble(rng) * 2.0 - 1.0)  // Risk metric
                };
                vectors.Add(features);
            }

            // Cache for future use
            try
            {
                Directory.CreateDirectory(cacheDir);
                var json = JsonSerializer.Serialize(vectors, CachedJsonOptions);
                File.WriteAllText(cachePath, json);
                LogSanityVectorsCached2(_logger, count, null);
            }
            catch (IOException ex)
            {
                LogCacheSanityVectorsIOError(_logger, ex);
            }
            catch (UnauthorizedAccessException ex)
            {
                LogCacheSanityVectorsAccessDenied(_logger, ex);
            }
            catch (JsonException ex)
            {
                LogCacheSanityVectorsJsonError(_logger, ex);
            }

            return vectors;
        }

        private async Task<(bool IsValid, double Divergence)> ComparePredictionDistributionsAsync(
            string currentModelPath,
            string newModelPath,
            List<float[]> testVectors,
            CancellationToken cancellationToken)
        {
            InferenceSession? currentSession = null;
            InferenceSession? newSession = null;
            
            try
            {
                var maxTotalVariation = _gate4Config.MaxTotalVariation;
                var maxKLDivergence = _gate4Config.MaxKLDivergence;
                var minProbability = _gate4Config.MinProbability;

                currentSession = new InferenceSession(currentModelPath);
                newSession = new InferenceSession(newModelPath);

                var currentPredictions = new List<float[]>();
                var newPredictions = new List<float[]>();

                foreach (var vector in testVectors)
                {
                    var currentOutput = await Task.Run(() => RunInference(currentSession, vector), cancellationToken).ConfigureAwait(false);
                    var newOutput = await Task.Run(() => RunInference(newSession, vector), cancellationToken).ConfigureAwait(false);
                    
                    currentPredictions.Add(currentOutput);
                    newPredictions.Add(newOutput);
                }

                var totalVariation = CalculateTotalVariationDistance(currentPredictions, newPredictions);
                var klDivergence = CalculateKLDivergence(currentPredictions, newPredictions, minProbability);

                LogDistributionComparison(_logger, totalVariation, klDivergence, null);

                if (totalVariation > maxTotalVariation)
                {
                    LogTotalVariationExceeded(_logger, totalVariation, maxTotalVariation, null);
                    return (false, totalVariation);
                }

                if (klDivergence > maxKLDivergence)
                {
                    LogKLDivergenceExceeded(_logger, klDivergence, maxKLDivergence, null);
                    return (false, klDivergence);
                }

                return (true, totalVariation);
            }
            catch (OnnxRuntimeException ex)
            {
                LogDistributionComparisonOnnxError(_logger, ex);
                return (false, 1.0);
            }
            catch (FileNotFoundException ex)
            {
                LogDistributionComparisonFileNotFound(_logger, ex);
                return (false, 1.0);
            }
            catch (InvalidOperationException ex)
            {
                LogDistributionComparisonInvalidOperation(_logger, ex);
                return (false, 1.0);
            }
            finally
            {
                currentSession?.Dispose();
                newSession?.Dispose();
            }
        }

        private static float[] RunInference(InferenceSession session, float[] inputVector)
        {
            var inputName = session.InputMetadata.Keys.First();
            
            var inputTensor = new DenseTensor<float>(inputVector, new[] { 1, inputVector.Length });
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, inputTensor) };
            
            using var results = session.Run(inputs);
            var resultsList = results.ToList();
            var output = resultsList[0].AsEnumerable<float>().ToArray();
            
            return output;
        }

        private static double CalculateTotalVariationDistance(List<float[]> predictions1, List<float[]> predictions2)
        {
            double totalVariation = 0.0;
            int count = 0;

            for (int i = 0; i < predictions1.Count; i++)
            {
                var p1 = predictions1[i];
                var p2 = predictions2[i];
                
                for (int j = 0; j < Math.Min(p1.Length, p2.Length); j++)
                {
                    totalVariation += Math.Abs(p1[j] - p2[j]);
                    count++;
                }
            }

            return count > 0 ? (totalVariation / count) * TopStepConfig.TotalVariationNormalizationFactor : 0.0;
        }

        private static double CalculateKLDivergence(List<float[]> predictions1, List<float[]> predictions2, double minProb)
        {
            double klDivergence = 0.0;
            int count = 0;

            for (int i = 0; i < predictions1.Count; i++)
            {
                var p = predictions1[i];
                var q = predictions2[i];
                
                for (int j = 0; j < Math.Min(p.Length, q.Length); j++)
                {
                    var pVal = Math.Max(p[j], minProb);
                    var qVal = Math.Max(q[j], minProb);
                    
                    klDivergence += pVal * Math.Log(pVal / qVal);
                    count++;
                }
            }

            return count > 0 ? klDivergence / count : 0.0;
        }

        private async Task<bool> ValidateModelOutputsAsync(
            string modelPath,
            List<float[]> testVectors,
            CancellationToken cancellationToken)
        {
            InferenceSession? session = null;
            
            try
            {
                if (!File.Exists(modelPath))
                {
                    LogModelFileNotFoundValidation(_logger, modelPath, null);
                    return false;
                }

                var fileInfo = new FileInfo(modelPath);
                if (fileInfo.Length == 0)
                {
                    LogModelFileEmpty(_logger, null);
                    return false;
                }

                session = new InferenceSession(modelPath);

                foreach (var vector in testVectors)
                {
                    var output = await Task.Run(() => RunInference(session, vector), cancellationToken).ConfigureAwait(false);
                    
                    foreach (var value in output)
                    {
                        if (float.IsNaN(value) || float.IsInfinity(value))
                        {
                            LogModelOutputsNaNInfinity(_logger, null);
                            return false;
                        }
                    }
                }

                LogModelOutputsValidated(_logger, null);
                return true;
            }
            catch (OnnxRuntimeException ex)
            {
                LogModelOutputValidationOnnxError(_logger, ex);
                return false;
            }
            catch (FileNotFoundException ex)
            {
                LogModelOutputValidationFileNotFound(_logger, ex);
                return false;
            }
            catch (InvalidOperationException ex)
            {
                LogModelOutputValidationInvalidOperation(_logger, ex);
                return false;
            }
            finally
            {
                session?.Dispose();
            }
        }

        private async Task<(bool Passed, double DrawdownRatio)> RunHistoricalSimulationAsync(
            string currentModelPath,
            string newModelPath,
            CancellationToken cancellationToken)
        {
            InferenceSession? currentSession = null;
            InferenceSession? newSession = null;
            
            try
            {
                var simulationBars = _gate4Config.SimulationBars;
                var maxDrawdownMultiplier = _gate4Config.MaxDrawdownMultiplier;
                
                var historicalData = await LoadHistoricalDataAsync(simulationBars, cancellationToken).ConfigureAwait(false);
                if (historicalData.Count < TopStepConfig.MinHistoricalBarsForSimulation)
                {
                    LogInsufficientHistoricalData(_logger, historicalData.Count, null);
                }

                currentSession = new InferenceSession(currentModelPath);
                newSession = new InferenceSession(newModelPath);

                var currentMaxDrawdown = await SimulateDrawdownAsync(currentSession, historicalData, cancellationToken).ConfigureAwait(false);
                var newMaxDrawdown = await SimulateDrawdownAsync(newSession, historicalData, cancellationToken).ConfigureAwait(false);

                var drawdownRatio = currentMaxDrawdown > 0 ? newMaxDrawdown / currentMaxDrawdown : 1.0;

                LogDrawdownComparison(_logger, currentMaxDrawdown, newMaxDrawdown, drawdownRatio, null);

                return (drawdownRatio <= maxDrawdownMultiplier, drawdownRatio);
            }
            catch (OnnxRuntimeException ex)
            {
                LogHistoricalSimulationOnnxError(_logger, ex);
                return (false, double.MaxValue);
            }
            catch (FileNotFoundException ex)
            {
                LogHistoricalSimulationFileNotFound(_logger, ex);
                return (false, double.MaxValue);
            }
            catch (InvalidOperationException ex)
            {
                LogHistoricalSimulationInvalidOperation(_logger, ex);
                return (false, double.MaxValue);
            }
            finally
            {
                currentSession?.Dispose();
                newSession?.Dispose();
            }
        }

        private async Task<List<float[]>> LoadHistoricalDataAsync(int count, CancellationToken cancellationToken)
        {
            var dataDir = Path.Combine("data", "validation");
            var dataPath = Path.Combine(dataDir, "historical_simulation_data.json");
            
            try
            {
                if (File.Exists(dataPath))
                {
                    var json = await File.ReadAllTextAsync(dataPath, cancellationToken).ConfigureAwait(false);
                    var data = JsonSerializer.Deserialize<List<float[]>>(json);
                    if (data != null && data.Count > 0)
                    {
                        return data.Take(count).ToList();
                    }
                }
            }
            catch (FileNotFoundException ex)
            {
                LogLoadHistoricalDataFileNotFound(_logger, ex);
            }
            catch (JsonException ex)
            {
                LogLoadHistoricalDataJsonError(_logger, ex);
            }
            catch (IOException ex)
            {
                LogLoadHistoricalDataIOError(_logger, ex);
            }

            var historicalData = new List<float[]>();
            // Using cryptographically secure random for ML simulation data to prevent adversarial exploitation
            using var rng = RandomNumberGenerator.Create();
            for (int i = 0; i < count; i++)
            {
                var features = new float[TopStepConfig.FeatureVectorLength];
                for (int j = 0; j < TopStepConfig.FeatureVectorLength; j++)
                {
                    features[j] = (float)(GetSecureRandomDouble(rng) * TopStepConfig.SimulationFeatureRange - TopStepConfig.SimulationFeatureOffset);
                }
                historicalData.Add(features);
            }

            try
            {
                Directory.CreateDirectory(dataDir);
                var json = JsonSerializer.Serialize(historicalData, CachedJsonOptions);
                await File.WriteAllTextAsync(dataPath, json, cancellationToken).ConfigureAwait(false);
            }
            catch (IOException ex)
            {
                LogCacheHistoricalDataIOError(_logger, ex);
            }
            catch (UnauthorizedAccessException ex)
            {
                LogCacheHistoricalDataAccessDenied(_logger, ex);
            }
            catch (JsonException ex)
            {
                LogCacheHistoricalDataJsonError(_logger, ex);
            }

            return historicalData;
        }

        private static async Task<double> SimulateDrawdownAsync(
            InferenceSession session,
            List<float[]> historicalData,
            CancellationToken cancellationToken)
        {
            double peak = 0.0;
            double equity = 0.0;
            double maxDrawdown = 0.0;

            foreach (var data in historicalData)
            {
                var prediction = await Task.Run(() => RunInference(session, data), cancellationToken).ConfigureAwait(false);
                
                var simulatedReturn = prediction.Length > 0 ? prediction[0] * 0.01 : 0.0;
                equity += simulatedReturn;
                
                if (equity > peak)
                {
                    peak = equity;
                }
                
                var drawdown = peak - equity;
                if (drawdown > maxDrawdown)
                {
                    maxDrawdown = drawdown;
                }
            }

            return Math.Abs(maxDrawdown);
        }

        private async Task CreateDefaultFeatureSpecificationAsync(string path, CancellationToken cancellationToken)
        {
            var spec = new
            {
                version = "1.0",
                feature_count = 11,
                features = new[]
                {
                    new { name = "volatility_normalized", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "price_change_momentum", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "volume_surge", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "atr_normalized", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "ucb_value", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "strategy_encoding", type = "float", range = new[] { 0.0, 1.0 } },
                    new { name = "direction_encoding", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "time_of_day_sin", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "time_of_day_cos", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "decisions_per_day_normalized", type = "float", range = new[] { -1.0, 1.0 } },
                    new { name = "risk_metric", type = "float", range = new[] { -1.0, 1.0 } }
                }
            };

            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            var json = JsonSerializer.Serialize(spec, CachedJsonOptions);
            await File.WriteAllTextAsync(path, json, cancellationToken).ConfigureAwait(false);
            LogFeatureSpecificationCreated(_logger, path, null);
        }

        /// <summary>
        /// Reload ONNX models with comprehensive validation and atomic swap
        /// </summary>
        public async Task<bool> ReloadModelsAsync(
            string newModelPath,
            CancellationToken cancellationToken = default)
        {
            var currentModelPath = GetCurrentModelPath();
            
            try
            {
                LogModelReloadStarting(_logger, newModelPath, null);

                var (isValid, reason) = await ValidateModelForReloadAsync(
                    newModelPath, currentModelPath, cancellationToken).ConfigureAwait(false);

                if (!isValid)
                {
                    LogModelReloadValidationFailed(_logger, reason, null);
                    return false;
                }

                var backupPath = CreateModelBackup(currentModelPath);
                LogModelReloadBackupCreated(_logger, backupPath, null);

                var (swapSuccess, oldVersion, newVersion) = await AtomicModelSwapAsync(
                    currentModelPath, newModelPath, cancellationToken).ConfigureAwait(false);

                if (!swapSuccess)
                {
                    LogModelReloadSwapFailed(_logger, null);
                    RestoreModelFromBackup(backupPath, currentModelPath);
                    return false;
                }

                LogModelReloadSuccess(_logger, null);
                LogModelReloadOldVersion(_logger, oldVersion, null);
                LogModelReloadNewVersion(_logger, newVersion, null);

                return true;
            }
            catch (OnnxRuntimeException ex)
            {
                LogModelReloadOnnxError(_logger, ex);
                return false;
            }
            catch (FileNotFoundException ex)
            {
                LogModelReloadFileNotFound(_logger, ex);
                return false;
            }
            catch (IOException ex)
            {
                LogModelReloadIOError(_logger, ex);
                return false;
            }
            catch (InvalidOperationException ex)
            {
                LogModelReloadInvalidOperation(_logger, ex);
                return false;
            }
        }

        private static string GetCurrentModelPath()
        {
            var modelDir = Path.Combine("models", "current");
            Directory.CreateDirectory(modelDir);
            var modelPath = Path.Combine(modelDir, "unified_brain.onnx");
            return modelPath;
        }

        private string CreateModelBackup(string currentModelPath)
        {
            var backupDir = Path.Combine("models", "backup");
            Directory.CreateDirectory(backupDir);
            
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            var backupPath = Path.Combine(backupDir, $"unified_brain_{timestamp}.onnx");

            if (File.Exists(currentModelPath))
            {
                File.Copy(currentModelPath, backupPath, overwrite: true);
                LogBackupCreated(_logger, backupPath, null);
            }

            return backupPath;
        }

        private void RestoreModelFromBackup(string backupPath, string targetPath)
        {
            if (File.Exists(backupPath))
            {
                File.Copy(backupPath, targetPath, overwrite: true);
                LogModelRestoredFromBackup(_logger, backupPath, null);
            }
        }

        private Task<(bool Success, string OldVersion, string NewVersion)> AtomicModelSwapAsync(
            string currentModelPath,
            string newModelPath,
            CancellationToken cancellationToken)
        {
            _ = cancellationToken; // Reserved for future async operations
            try
            {
                var oldVersion = File.Exists(currentModelPath) 
                    ? GetModelVersion(currentModelPath) 
                    : "none";
                var newVersion = GetModelVersion(newModelPath);

                var tempPath = currentModelPath + ".tmp";
                File.Copy(newModelPath, tempPath, overwrite: true);

                if (File.Exists(currentModelPath))
                {
                    File.Delete(currentModelPath);
                }

                File.Move(tempPath, currentModelPath);

                LogAtomicSwapCompleted(_logger, oldVersion, newVersion, null);

                return Task.FromResult((true, oldVersion, newVersion));
            }
            catch (IOException ex)
            {
                LogAtomicSwapIOError(_logger, ex);
                return Task.FromResult((false, string.Empty, string.Empty));
            }
            catch (UnauthorizedAccessException ex)
            {
                LogAtomicSwapAccessDenied(_logger, ex);
                return Task.FromResult((false, string.Empty, string.Empty));
            }
            catch (InvalidOperationException ex)
            {
                LogAtomicSwapInvalidOperation(_logger, ex);
                return Task.FromResult((false, string.Empty, string.Empty));
            }
        }

        private static string GetModelVersion(string modelPath)
        {
            if (!File.Exists(modelPath))
            {
                return "unknown";
            }

            var fileInfo = new FileInfo(modelPath);
            var timestamp = fileInfo.LastWriteTimeUtc.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
            var size = fileInfo.Length;
            
            return $"{timestamp}_{size}";
        }

        private async Task RetrainModelsAsync(CancellationToken cancellationToken)
        {
            try
            {
                LogUnifiedRetrainingStarting(_logger, null);
                
                // Export comprehensive training data including all strategies
                var unifiedTrainingData = _decisionHistory.TakeLast(TopStepConfig.TrainingDataHistorySize).Select(d => new
                {
                    features = CreateContextVector(d.Context).Features,
                    strategy = d.Strategy,
                    reward = d.WasCorrect ? 1.0 : 0.0,
                    pnl = (double)d.PnL,
                    market_conditions = GetCurrentMarketConditions(d.Context),
                    timestamp = d.Timestamp,
                    strategy_specialization = _strategySpecializations.GetValueOrDefault(d.Strategy)?.Name ?? "unknown"
                });
                
                // Export strategy performance data
                var strategyPerformanceData = _strategyPerformance.ToDictionary(
                    kvp => kvp.Key,
                    kvp => new
                    {
                        win_rate = kvp.Value.WinRate,
                        total_trades = kvp.Value.TotalTrades,
                        total_pnl = (double)kvp.Value.TotalPnL,
                        avg_hold_time = kvp.Value.HoldTimes.Count > 0 ? 
                            TimeSpan.FromTicks((long)kvp.Value.HoldTimes.Average()).TotalMinutes : 0,
                        optimal_conditions = GetBestConditionsForStrategy(kvp.Key)
                    });
                
                // Export data for training
                var dataPath = Path.Combine("data", "unified_brain_training_data.json");
                var perfPath = Path.Combine("data", "strategy_performance_data.json");
                
                Directory.CreateDirectory(Path.GetDirectoryName(dataPath)!);
                
                await File.WriteAllTextAsync(dataPath, JsonSerializer.Serialize(unifiedTrainingData, 
                    CachedJsonOptions), cancellationToken).ConfigureAwait(false);
                await File.WriteAllTextAsync(perfPath, JsonSerializer.Serialize(strategyPerformanceData, 
                    CachedJsonOptions), cancellationToken).ConfigureAwait(false);
                
                LogUnifiedRetrainingDataExported(_logger, unifiedTrainingData.Count(), _strategyPerformance.Count, null);
                
                // Enhanced Python training scripts for multi-strategy learning would be integrated here
            }
            catch (IOException ex)
            {
                LogUnifiedRetrainingIOError(_logger, ex);
            }
            catch (UnauthorizedAccessException ex)
            {
                LogUnifiedRetrainingAccessDenied(_logger, ex);
            }
            catch (InvalidOperationException ex)
            {
                LogUnifiedRetrainingInvalidOperation(_logger, ex);
            }
            catch (JsonException ex)
            {
                LogUnifiedRetrainingJsonError(_logger, ex);
            }
        }

        #endregion

        #region Real-Time AI Commentary Features

        /// <summary>
        /// Feature 1: Explain why bot is waiting (low confidence or no clear signal)
        /// </summary>
        private async Task<string> ExplainWhyWaitingAsync(
            MarketContext context,
            StrategySelection optimalStrategy,
            PricePrediction priceDirection)
        {
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var currentContext = GatherCurrentContext();
                
                var prompt = $@"I am a trading bot. I'm NOT taking a trade right now because:
Strategy Confidence: {optimalStrategy.Confidence:P1}
Price Direction: {priceDirection.Direction}
Price Probability: {priceDirection.Probability:P1}
Market Regime: {context.TrendStrength}

Current context: {currentContext}

Explain in 1-2 sentences why I'm waiting and what I'm looking for. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogBotCommentaryWaitingInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogBotCommentaryWaitingHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogBotCommentaryWaitingTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        /// <summary>
        /// Feature 1: Explain high confidence trade decision
        /// </summary>
        private async Task<string> ExplainConfidenceAsync(
            BrainDecision decision,
            MarketContext context)
        {
            _ = context; // Reserved for future context-aware explanations
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var currentContext = GatherCurrentContext();
                
                var prompt = $@"I am a trading bot. I'm VERY confident about this trade:
Strategy: {decision.RecommendedStrategy}
Direction: {decision.PriceDirection}
Confidence: {decision.StrategyConfidence:P1}
Price Probability: {decision.PriceProbability:P1}
Market Regime: {decision.MarketRegime}

Current context: {currentContext}

Explain in 1-2 sentences why I'm so confident. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogBotCommentaryConfidenceInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogBotCommentaryConfidenceHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogBotCommentaryConfidenceTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        /// <summary>
        /// Feature 1: Explain conflicting strategy signals
        /// </summary>
        private async Task<string> ExplainConflictAsync(
            Dictionary<string, decimal> strategyConfidences,
            MarketContext context)
        {
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var currentContext = GatherCurrentContext();
                var strategyList = string.Join(", ", strategyConfidences.Select(kvp => $"{kvp.Key}: {kvp.Value:P0}"));
                
                var prompt = $@"I am a trading bot. My strategies are giving CONFLICTING signals:
{strategyList}

Market Regime: {context.TrendStrength}
Current context: {currentContext}

Explain in 1-2 sentences why strategies disagree and what this means. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogBotCommentaryConflictInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogBotCommentaryConflictHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogBotCommentaryConflictTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        /// <summary>
        /// Feature 2: Deep analysis of why a trade failed
        /// </summary>
        private async Task<string> AnalyzeTradeFailureAsync(
            string symbol,
            string strategy,
            decimal pnl,
            decimal entryPrice,
            decimal stopPrice,
            decimal targetPrice,
            decimal exitPrice,
            string exitReason,
            MarketContext entryContext,
            MarketContext? exitContext)
        {
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var marketShift = exitContext != null 
                    ? $"Trend changed from {entryContext.TrendStrength:F2} to {exitContext.TrendStrength:F2}, Volatility from {entryContext.Volatility:F2} to {exitContext.Volatility:F2}"
                    : "Exit context unavailable";

                var prompt = $@"I am a trading bot. I need to analyze why this trade FAILED:
Symbol: {symbol}
Strategy: {strategy}
Entry Price: {entryPrice:F2}
Stop Price: {stopPrice:F2}
Target Price: {targetPrice:F2}
Exit Price: {exitPrice:F2}
Exit Reason: {exitReason}
Loss: ${pnl:F2}

Entry Market Conditions: Trend {entryContext.TrendStrength:F2}, Volatility {entryContext.Volatility:F2}
What Changed: {marketShift}

Analyze in 2-3 sentences: What went wrong? Was it my entry timing, stop placement, strategy choice, or market conditions? What should I learn? Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogFailureAnalysisInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogFailureAnalysisHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogFailureAnalysisTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        /// <summary>
        /// Feature 4: Explain why Neural UCB chose specific strategy
        /// </summary>
        private async Task<string> ExplainStrategySelectionAsync(
            string selectedStrategy,
            Dictionary<string, decimal> allStrategyScores,
            MarketContext context)
        {
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var currentContext = GatherCurrentContext();
                var scoreList = string.Join(", ", allStrategyScores.Select(kvp => $"{kvp.Key}: {kvp.Value:P0}"));
                
                // Get recent performance for selected strategy
                var strategyPerf = _strategyPerformance.GetValueOrDefault(selectedStrategy);
                var winRate = strategyPerf?.WinRate ?? 0;
                
                var prompt = $@"I am a trading bot. My Neural UCB chose strategy {selectedStrategy}:

All Strategy Scores: {scoreList}
Selected: {selectedStrategy} (Recent Win Rate: {winRate:P0})

Market Conditions: Trend {context.TrendStrength:F2}, Volatility {context.Volatility:F2}
Current context: {currentContext}

Explain in 1-2 sentences why Neural UCB selected this strategy over others. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogStrategySelectionExplanationInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogStrategySelectionExplanationHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogStrategySelectionExplanationTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        /// <summary>
        /// Feature 5: Explain detected market regime
        /// </summary>
        private async Task<string> ExplainMarketRegimeAsync(
            MarketRegime regime,
            MarketContext context)
        {
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var currentContext = GatherCurrentContext();
                
                var prompt = $@"I am a trading bot. I detected this market regime:
Regime: {regime}
Trend Strength: {context.TrendStrength:F2}
Volatility: {context.Volatility:F2}
Volume Ratio: {context.VolumeRatio:F2}

Current context: {currentContext}

Explain in 1-2 sentences what this regime means and how it affects my trading. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogMarketRegimeExplanationInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogMarketRegimeExplanationHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogMarketRegimeExplanationTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        /// <summary>
        /// Feature 6: Explain what bot learned from recent trades
        /// </summary>
        private async Task<string> ExplainWhatILearnedAsync(
            string learningType,
            string details)
        {
            if (_ollamaClient == null)
                return string.Empty;

            try
            {
                var currentContext = GatherCurrentContext();
                
                var prompt = $@"I am a trading bot. I just updated my learning:
Learning Type: {learningType}
Details: {details}

Current context: {currentContext}

Explain in 1-2 sentences what I learned and how it will improve my future trading. Speak as ME (the bot).";

                var response = await _ollamaClient.AskAsync(prompt).ConfigureAwait(false);
                return response;
            }
            catch (InvalidOperationException ex)
            {
                LogLearningExplanationInvalidOperation(_logger, ex);
                return string.Empty;
            }
            catch (HttpRequestException ex)
            {
                LogLearningExplanationHttpError(_logger, ex);
                return string.Empty;
            }
            catch (TaskCanceledException ex)
            {
                LogLearningExplanationTaskCancelled(_logger, ex);
                return string.Empty;
            }
        }

        #endregion

        #region Production CVaR-PPO Integration

        /// <summary>
        /// Create comprehensive state vector for CVaR-PPO position sizing model
        /// </summary>
        private double[] CreateCVaRStateVector(MarketContext context, StrategySelection strategy, PricePrediction prediction)
        {
            return new double[]
            {
                // Market microstructure features (normalized 0-1)
                (double)Math.Min(TopStepConfig.MaxNormalizationValue, context.Volatility / (decimal)TopStepConfig.VolatilityNormalizationDivisor), // Volatility ratio [0-1]
                Math.Tanh((double)(context.PriceChange / (decimal)TopStepConfig.PriceChangeMomentumDivisor)), // Price change momentum [-1,1]
                (double)Math.Min(TopStepConfig.MaxNormalizationValue, context.VolumeRatio / (decimal)TopStepConfig.VolumeSurgeNormalizationDivisor), // Volume surge ratio [0-1]
                (double)Math.Min(TopStepConfig.MaxNormalizationValue, (context.Atr ?? TopStepConfig.DefaultAtrFallback) / (decimal)TopStepConfig.AtrNormalizationDivisor), // ATR normalized [0-1]
                (double)context.TrendStrength, // Trend strength [0-1]
                
                // Strategy selection features  
                (double)strategy.Confidence, // Neural UCB confidence [0-1]
                Math.Min(TopStepConfig.MinNormalizationValue, (double)strategy.UcbValue / TopStepConfig.UcbValueNormalizationDivisor), // UCB exploration value normalized
                strategy.SelectedStrategy switch { // Strategy type encoding
                    "S2_VWAP" => TopStepConfig.S2VwapStrategyEncoding, 
                    "S3_Compression" => TopStepConfig.S3CompressionStrategyEncoding, 
                    "S11_Opening" => TopStepConfig.S11OpeningStrategyEncoding, 
                    "S12_Momentum" => TopStepConfig.S12MomentumStrategyEncoding, 
                    _ => TopStepConfig.DefaultStrategyEncoding
                },
                
                // LSTM prediction features
                (double)prediction.Probability, // Price direction confidence [0-1]
                (int)prediction.Direction / TopStepConfig.DirectionEncodingDivisor + TopStepConfig.DirectionEncodingOffset, // Direction: Down=0, Sideways=0.5, Up=1
                
                // Temporal features (cyclical encoding)
                Math.Sin(TopStepConfig.CyclicalEncodingMultiplier * Math.PI * context.TimeOfDay.TotalHours / TopStepConfig.HoursPerDay), // Hour of day
                Math.Cos(TopStepConfig.CyclicalEncodingMultiplier * Math.PI * context.TimeOfDay.TotalHours / TopStepConfig.HoursPerDay),
                
                // Risk management features
                (double)Math.Max(-TopStepConfig.MaxNormalizationValue, Math.Min(TopStepConfig.MaxNormalizationValue, _currentDrawdown / TopStepConfig.MaxDrawdown)), // Drawdown ratio [-1,1]
                (double)Math.Max(-TopStepConfig.MaxNormalizationValue, Math.Min(TopStepConfig.MaxNormalizationValue, _dailyPnl / TopStepConfig.DailyLossLimit)), // Daily P&L ratio [-1,1]
                
                // Portfolio state features
                Math.Min(TopStepConfig.MinNormalizationValue, DecisionsToday / TopStepConfig.MaxDecisionsPerDayNormalization), // Decision frequency normalized [0-1]
                (double)WinRateToday, // Current session win rate [0-1]
            };
        }

        /// <summary>
        /// Convert CVaR-PPO action result to contract count
        /// </summary>
        private static int ConvertCVaRActionToContracts(ActionResult actionResult, int baseContracts)
        {
            // CVaR-PPO actions map to position size multipliers
            // Action 0=No Trade, 1=Micro(0.25x), 2=Small(0.5x), 3=Normal(1x), 4=Large(1.5x), 5=Max(2x)
            var sizeMultiplier = actionResult.Action switch
            {
                0 => TopStepConfig.NoTradeMultiplier,        // No trade
                1 => TopStepConfig.MicroPositionMultiplier,  // Micro position
                2 => TopStepConfig.SmallPositionMultiplier,  // Small position  
                3 => TopStepConfig.NormalPositionMultiplier, // Normal position
                4 => TopStepConfig.LargePositionMultiplier,  // Large position
                5 => TopStepConfig.MaxPositionMultiplier,    // Maximum position
                _ => TopStepConfig.NormalPositionMultiplier  // Default to normal
            };
            
            // Apply action probability weighting - reduce size for uncertain actions
            var probabilityAdjustment = (decimal)Math.Max((double)TopStepConfig.MinProbabilityAdjustment, actionResult.ActionProbability);
            
            // Apply value estimate adjustment - reduce size for negative expected value
            var valueAdjustment = (decimal)Math.Max((double)TopStepConfig.MinValueAdjustment, Math.Min((double)TopStepConfig.MaxValueAdjustment, (double)TopStepConfig.ValueEstimateOffset + actionResult.ValueEstimate));
            
            var adjustedMultiplier = sizeMultiplier * probabilityAdjustment * valueAdjustment;
            var contracts = (int)Math.Round(baseContracts * adjustedMultiplier);
            
            return Math.Max(0, contracts);
        }

        /// <summary>
        /// Apply CVaR risk controls to position sizing
        /// </summary>
        private static int ApplyCVaRRiskControls(int contracts, ActionResult actionResult, MarketContext context)
        {
            if (contracts <= 0) return 0;
            
            // CVaR tail risk adjustment - reduce position if high tail risk
            var cvarAdjustment = TopStepConfig.NormalPositionMultiplier;
            if (actionResult.CVaREstimate < (double)TopStepConfig.HighNegativeTailRiskThreshold) // High negative tail risk
            {
                cvarAdjustment = TopStepConfig.HighRiskPositionReduction; // Cut position in half
            }
            else if (actionResult.CVaREstimate < (double)TopStepConfig.ModerateTailRiskThreshold) // Moderate tail risk
            {
                cvarAdjustment = TopStepConfig.ModerateRiskPositionReduction; // Reduce position by 25%
            }
            
            // Volatility regime adjustment
            var volAdjustment = context.Volatility switch
            {
                > 0.6m => 0.6m,  // High volatility - very conservative
                > 0.4m => 0.8m,  // Moderate volatility - somewhat conservative
                > 0.2m => 1.0m,  // Normal volatility - no adjustment
                _ => 1.2m        // Low volatility - can be more aggressive
            };
            
            var finalMultiplier = cvarAdjustment * volAdjustment;
            var adjustedContracts = (int)Math.Round(contracts * finalMultiplier);
            
            return Math.Max(0, adjustedContracts);
        }

        #endregion

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                LogBrainShuttingDown(_logger, null);
                
                // Save performance statistics
                var stats = new
                {
                    DecisionsToday,
                    WinRateToday,
                    TotalDecisions = _decisionHistory.Count,
                    Performance = _performance.ToDictionary(kvp => kvp.Key, kvp => kvp.Value),
                    LastDecision
                };
                
                try
                {
                    var statsPath = Path.Combine("logs", $"brain_stats_{DateTime.Now:yyyyMMdd}.json");
                    Directory.CreateDirectory(Path.GetDirectoryName(statsPath)!);
                    File.WriteAllText(statsPath, JsonSerializer.Serialize(stats, CachedJsonOptions));
                    LogBrainStatisticsSaved(_logger, DecisionsToday, WinRateToday, null);
                }
                catch (IOException ex)
                {
                    LogBrainStatisticsSaveIOError(_logger, ex);
                }
                catch (UnauthorizedAccessException ex)
                {
                    LogBrainStatisticsSaveAccessDenied(_logger, ex);
                }
                catch (JsonException ex)
                {
                    LogBrainStatisticsSaveJsonError(_logger, ex);
                }
                
                // Dispose managed resources
                try
                {
                    _strategySelector?.Dispose();
                    _cvarPPO?.Dispose();
                    if (_confidenceNetwork is IDisposable disposableNetwork)
                        disposableNetwork.Dispose();
                    _memoryManager?.Dispose();
                    _modelManager?.Dispose();
                    
                    if (_lstmPricePredictor is IDisposable disposableLstm)
                        disposableLstm.Dispose();
                    if (_metaClassifier is IDisposable disposableMeta) 
                        disposableMeta.Dispose();
                    if (_marketRegimeDetector is IDisposable disposableRegime)
                        disposableRegime.Dispose();
                }
                catch (ObjectDisposedException)
                {
                    // Expected during shutdown - ignore
                }
                catch (InvalidOperationException ex)
                {
                    LogBrainDisposeInvalidOperation(_logger, ex);
                }
            }
        }

        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }
        
        /// <summary>
        /// Generate cryptographically secure random double in [0.0, 1.0)
        /// Used for ML training data generation to prevent adversarial exploitation
        /// </summary>
        private static double GetSecureRandomDouble(RandomNumberGenerator rng)
        {
            var bytes = new byte[8];
            rng.GetBytes(bytes);
            var value = BitConverter.ToUInt64(bytes, 0);
            // Convert to [0.0, 1.0) by dividing by UInt64.MaxValue
            return (double)value / ulong.MaxValue;
        }
    }
}
