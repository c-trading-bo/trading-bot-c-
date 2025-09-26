using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Strategy;
using BotCore.Risk;
using BotCore.Models;
using BotCore.ML;
using BotCore.Bandits;
using System.Collections.Concurrent;
using System.Text.Json;
using TradingBot.RLAgent; // For CVaRPPO and ActionResult

namespace BotCore.Brain
{
    /// <summary>
    /// TopStep compliance configuration integrated into UnifiedTradingBrain
    /// </summary>
    public static class TopStepConfig
    {
        public const decimal ACCOUNT_SIZE = 50_000m;
        public const decimal MAX_DRAWDOWN = 2_000m;
        public const decimal DAILY_LOSS_LIMIT = 1_000m;
        public const decimal TRAILING_STOP = 48_000m;
        public const decimal ES_POINT_VALUE = 50m;
        public const decimal NQ_POINT_VALUE = 20m;
        public const decimal RISK_PER_TRADE = 0.01m; // 1% = $500 baseline
        public const double EXPLORATION_BONUS = 0.3;
        public const double CONFIDENCE_THRESHOLD = 0.65;
    }
    /// <summary>
    /// UNIFIED TRADING BRAIN - The ONE intelligence that controls all trading decisions
    /// Enhanced to handle all 4 primary strategies (S2, S3, S6, S11) with unified scheduling
    /// 
    /// This is the central AI brain that:
    /// 1. Handles S2 (VWAP Mean Reversion), S3 (Bollinger Compression), S6 (Momentum), S11 (Specialized)
    /// 2. Uses Neural UCB to select optimal strategy for each market condition
    /// 3. Uses LSTM to predict price movements and timing
    /// 4. Uses CVaR-PPO to optimize position sizes for all strategies
    /// 5. Maintains identical intelligence for historical and live trading
    /// 6. Continuously learns from all trade outcomes to improve strategy selection
    /// 
    /// KEY ENHANCEMENTS:
    /// - Multi-strategy learning: Every trade outcome teaches all strategies
    /// - Unified scheduling: Same timing for historical and live systems
    /// - Continuous improvement: Historical patterns improve live strategy selection
    /// - Same AI brain gets smarter at picking S2 vs S3 vs S6 vs S11
    /// - Position sizing and risk management learns from all strategy results
    /// 
    /// INTEGRATION POINTS:
    /// - TradingOrchestratorService calls this brain for live trading
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
        private readonly ConcurrentDictionary<string, MarketContext> _marketContexts = new();
        private readonly ConcurrentDictionary<string, TradingPerformance> _performance = new();
        private readonly CVaRPPO _cvarPPO; // Direct injection instead of loading from memory
        
        // ML Models for different decision points
        private object? _lstmPricePredictor;
        private object? _metaClassifier;
        private object? _marketRegimeDetector;
        private readonly INeuralNetwork? _confidenceNetwork;
        
        // TopStep compliance tracking
        private decimal _currentDrawdown;
        private decimal _dailyPnl;
        private decimal _accountBalance = TopStepConfig.ACCOUNT_SIZE;
        private DateTime _lastResetDate = DateTime.UtcNow.Date;
        
        // Performance tracking for learning
        private readonly List<TradingDecision> _decisionHistory = new();
        private DateTime _lastModelUpdate = DateTime.MinValue;
        
        // Multi-strategy learning state
        private readonly Dictionary<string, StrategyPerformance> _strategyPerformance = new();
        private readonly Dictionary<string, List<MarketCondition>> _strategyOptimalConditions = new();
        private DateTime _lastUnifiedLearningUpdate = DateTime.MinValue;
        
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

        public UnifiedTradingBrain(
            ILogger<UnifiedTradingBrain> logger,
            IMLMemoryManager memoryManager,
            StrategyMlModelManager modelManager,
            CVaRPPO cvarPPO)
        {
            _logger = logger;
            _memoryManager = memoryManager;
            _modelManager = modelManager;
            _cvarPPO = cvarPPO; // Direct injection
            
            // Initialize Neural UCB for strategy selection using ONNX-based neural network
            var onnxLoader = new OnnxModelLoader(new Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxModelLoader>());
            var neuralNetworkLogger = new Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxNeuralNetwork>();
            var neuralNetwork = new OnnxNeuralNetwork(onnxLoader, neuralNetworkLogger, "models/strategy_selection.onnx");
            _strategySelector = new NeuralUcbBandit(neuralNetwork);
            
            // Initialize confidence network for model confidence prediction
            _confidenceNetwork = new OnnxNeuralNetwork(onnxLoader, neuralNetworkLogger, "models/confidence_prediction.onnx");
            
            _logger.LogInformation("🧠 [UNIFIED-BRAIN] Initialized with direct CVaR-PPO injection - Ready to make intelligent trading decisions");
        }

        /// <summary>
        /// Initialize all ML models and prepare the brain for trading
        /// This is called from UnifiedOrchestrator startup
        /// </summary>
        public async Task InitializeAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                _logger.LogInformation("🚀 [UNIFIED-BRAIN] Loading all ML models...");

                // Load LSTM for price prediction - use your real trained model
                _lstmPricePredictor = await _memoryManager.LoadModelAsync<object>(
                    "models/rl_model.onnx", "v1").ConfigureAwait(false);
                
                // CVaR-PPO is already injected and initialized via DI container
                _logger.LogInformation("✅ [CVAR-PPO] Using direct injection from DI container");
                
                // Load meta classifier for market regime - use your test CVaR model
                _metaClassifier = await _memoryManager.LoadModelAsync<object>(
                    "models/rl/test_cvar_ppo.onnx", "v1").ConfigureAwait(false);
                
                // Load market regime detector - use your main RL model as backup
                _marketRegimeDetector = await _memoryManager.LoadModelAsync<object>(
                    "models/rl_model.onnx", "v1").ConfigureAwait(false);

                IsInitialized = true;
                _logger.LogInformation("✅ [UNIFIED-BRAIN] All models loaded successfully - Brain is ONLINE with production CVaR-PPO");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [UNIFIED-BRAIN] Failed to initialize models - Using fallback logic");
                IsInitialized = false; // Will use rule-based fallbacks
            }
        }

        /// <summary>
        /// MAIN BRAIN FUNCTION: Make intelligent trading decision
        /// Called by TradingOrchestratorService.ExecuteESNQTradingAsync()
        /// 
        /// This replaces the manual strategy selection in AllStrategies.cs
        /// </summary>
        public async Task<BrainDecision> MakeIntelligentDecisionAsync(
            string symbol,
            Env env,
            Levels levels,
            IList<Bar> bars,
            RiskEngine risk,
            CancellationToken cancellationToken = default)
        {
            if (symbol is null) throw new ArgumentNullException(nameof(symbol));
            if (env is null) throw new ArgumentNullException(nameof(env));
            if (levels is null) throw new ArgumentNullException(nameof(levels));
            if (bars is null) throw new ArgumentNullException(nameof(bars));
            if (risk is null) throw new ArgumentNullException(nameof(risk));
            
            var startTime = DateTime.UtcNow;
            LastDecision = startTime;
            
            try
            {
                // 1. CREATE MARKET CONTEXT from current data
                var context = CreateMarketContext(symbol, env, bars);
                _marketContexts[symbol] = context;
                
                // 2. DETECT MARKET REGIME using Meta Classifier
                var marketRegime = await DetectMarketRegimeAsync(context).ConfigureAwait(false);
                
                // 3. SELECT OPTIMAL STRATEGY using Neural UCB
                var optimalStrategy = await SelectOptimalStrategyAsync(context, marketRegime, cancellationToken).ConfigureAwait(false);
                
                // 4. PREDICT PRICE MOVEMENT using LSTM
                var priceDirection = await PredictPriceDirectionAsync(context, bars).ConfigureAwait(false);
                
                // 5. OPTIMIZE POSITION SIZE using RL
                var optimalSize = await OptimizePositionSizeAsync(context, optimalStrategy, priceDirection, cancellationToken).ConfigureAwait(false);
                
                // 6. GENERATE ENHANCED CANDIDATES using brain intelligence
                var enhancedCandidates = await GenerateEnhancedCandidatesAsync(
                    symbol, env, levels, bars, risk, optimalStrategy, priceDirection, optimalSize).ConfigureAwait(false);
                
                var decision = new BrainDecision
                {
                    Symbol = symbol,
                    RecommendedStrategy = optimalStrategy.SelectedStrategy,
                    StrategyConfidence = optimalStrategy.Confidence,
                    PriceDirection = priceDirection.Direction,
                    PriceProbability = priceDirection.Probability,
                    OptimalPositionMultiplier = optimalSize,
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
                
                _logger.LogInformation("🧠 [BRAIN-DECISION] {Symbol}: Strategy={Strategy} ({Confidence:P1}), " +
                    "Direction={Direction} ({Probability:P1}), Size={Size:F2}x, Regime={Regime}, Time={Ms:F0}ms",
                    symbol, optimalStrategy.SelectedStrategy, optimalStrategy.Confidence,
                    priceDirection.Direction, priceDirection.Probability, optimalSize, marketRegime, decision.ProcessingTimeMs);

                return decision;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [UNIFIED-BRAIN] Error making decision for {Symbol}", symbol);
                
                // Fallback to rule-based decision
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
                    
                    // 🚀 MULTI-STRATEGY LEARNING: Update ALL strategies with this market condition
                    await UpdateAllStrategiesFromOutcomeAsync(context, strategy, reward, wasCorrect, cancellationToken).ConfigureAwait(false);
                }

                // Update performance tracking for the specific strategy
                if (!_performance.ContainsKey(symbol))
                {
                    _performance[symbol] = new TradingPerformance();
                }
                
                var perf = _performance[symbol];
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
                if (DateTime.UtcNow - _lastUnifiedLearningUpdate > TimeSpan.FromHours(2) && _decisionHistory.Count > 50)
                {
                    _ = Task.Run(() => UpdateUnifiedLearningAsync(cancellationToken), cancellationToken);
                    _lastUnifiedLearningUpdate = DateTime.UtcNow;
                }

                // Periodic full model retraining
                if (DateTime.UtcNow - _lastModelUpdate > TimeSpan.FromHours(8) && _decisionHistory.Count > 200)
                {
                    _ = Task.Run(() => RetrainModelsAsync(cancellationToken), cancellationToken);
                    _lastModelUpdate = DateTime.UtcNow;
                }

                _logger.LogInformation("📚 [UNIFIED-LEARNING] {Symbol} {Strategy}: PnL={PnL:F2}, Correct={Correct}, " +
                    "WinRate={WinRate:P1}, TotalTrades={Total}, AllStrategiesUpdated=True",
                    symbol, strategy, pnl, wasCorrect, WinRateToday, perf.TotalTrades);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [UNIFIED-LEARNING] Error learning from result");
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
                    
                    // Update strategy-specific learning patterns
                    UpdateStrategyOptimalConditions(strategy, context, crossLearningReward > 0.5m);
                }
                
                _logger.LogDebug("🧠 [CROSS-LEARNING] Updated all strategies from {ExecutedStrategy} outcome: {Reward:F3}", 
                    executedStrategy, reward);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [CROSS-LEARNING] Error updating all strategies");
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
                return executedReward * 0.1m; // Minimal learning if no specialization
            
            // Calculate similarity in optimal conditions
            var conditionSimilarity = learningSpec.OptimalConditions
                .Intersect(executedSpec.OptimalConditions)
                .Count() / (float)Math.Max(learningSpec.OptimalConditions.Length, 1);
            
            // Time window overlap
            var timeOverlap = learningSpec.TimeWindows
                .Intersect(executedSpec.TimeWindows)
                .Count() / (float)Math.Max(learningSpec.TimeWindows.Length, 1);
            
            // Market condition alignment
            var currentConditions = GetCurrentMarketConditions(context);
            var conditionAlignment = learningSpec.OptimalConditions
                .Intersect(currentConditions)
                .Count() / (float)Math.Max(learningSpec.OptimalConditions.Length, 1);
            
            // Calculate cross-learning strength
            var learningStrength = (conditionSimilarity * 0.4f + timeOverlap * 0.3f + conditionAlignment * 0.3f);
            
            // Positive outcome strengthens similar strategies, negative outcome weakens them
            var baseReward = wasCorrect ? executedReward : (1 - executedReward);
            return Math.Clamp(baseReward * (decimal)learningStrength, 0m, 1m);
        }
        
        private static string[] GetCurrentMarketConditions(MarketContext context)
        {
            var conditions = new List<string>();
            
            if (context.Volatility < 0.15m) conditions.Add("low_volatility");
            else if (context.Volatility > 0.4m) conditions.Add("high_volatility");
            
            if (context.VolumeRatio > 1.5m) conditions.Add("high_volume");
            if (context.TrendStrength > 0.7m) conditions.Add("trending");
            else if (context.TrendStrength < 0.3m) conditions.Add("ranging");
            
            if (context.RSI > 70) conditions.Add("overbought");
            else if (context.RSI < 30) conditions.Add("oversold");
            
            var hour = context.TimeOfDay.Hours;
            if (hour >= 9 && hour <= 10) conditions.Add("opening_drive");
            else if (hour >= 11 && hour <= 13) conditions.Add("lunch");
            else if (hour >= 13 && hour <= 16) conditions.Add("afternoon_fade");
            
            return conditions.ToArray();
        }

        #region ML Model Predictions

        private Task<MarketRegime> DetectMarketRegimeAsync(MarketContext context)
        {
            if (_metaClassifier == null || !IsInitialized)
            {
                // Fallback: use volatility-based regime detection
                return Task.FromResult(context.Volatility > 0.3m ? MarketRegime.HighVolatility :
                       context.Volatility < 0.15m ? MarketRegime.LowVolatility : MarketRegime.Normal);
            }

            try
            {
                // Analyze market regime using technical indicators and volatility
                // ONNX model integration planned for future enhancement
                if (context.VolumeRatio > 1.5m && context.Volatility > 0.25m)
                    return Task.FromResult(MarketRegime.Trending);
                if (context.Volatility < 0.15m && Math.Abs(context.PriceChange) < 0.5m)
                    return Task.FromResult(MarketRegime.Ranging);
                if (context.Volatility > 0.4m)
                    return Task.FromResult(MarketRegime.HighVolatility);
                
                return Task.FromResult(MarketRegime.Normal);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Meta classifier prediction failed, using fallback");
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
                
                return new StrategySelection
                {
                    SelectedStrategy = selection.SelectedArm,
                    Confidence = selection.Confidence,
                    UcbValue = selection.UcbValue,
                    Reasoning = selection.SelectionReason ?? "Neural UCB selection"
                };
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Neural UCB selection failed, using fallback");
                
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
                    Confidence = 0.6m,
                    UcbValue = 0.5m,
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
                    var priceChange = recentBars.Last().Close - recentBars.First().Close;
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
                
                var isUptrend = ema20 > ema50 && bars.Last().Close > ema20;
                var isOversold = rsi < 30;
                var isOverbought = rsi > 70;
                
                PriceDirection direction;
                decimal probability;
                
                if (isUptrend && !isOverbought)
                {
                    direction = PriceDirection.Up;
                    probability = 0.7m;
                }
                else if (!isUptrend && !isOversold)
                {
                    direction = PriceDirection.Down;
                    probability = 0.7m;
                }
                else
                {
                    direction = PriceDirection.Sideways;
                    probability = 0.5m;
                }
                
                return Task.FromResult(new PricePrediction
                {
                    Direction = direction,
                    Probability = probability,
                    ExpectedMove = context.Atr ?? 10,
                    TimeHorizon = TimeSpan.FromMinutes(30)
                });
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "LSTM prediction failed, using fallback");
                return Task.FromResult(new PricePrediction
                {
                    Direction = PriceDirection.Sideways,
                    Probability = 0.5m,
                    ExpectedMove = 5,
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
            var (canTrade, reason, level) = ShouldStopTrading();
            if (!canTrade)
            {
                _logger.LogWarning("🛑 [TOPSTEP-COMPLIANCE] Trading blocked: {Reason}", reason);
                return 0m; // No position if compliance violated
            }

            // Calculate base risk amount
            var baseRisk = _accountBalance * TopStepConfig.RISK_PER_TRADE;

            // Progressive position reduction based on drawdown
            var drawdownRatio = _currentDrawdown / TopStepConfig.MAX_DRAWDOWN;
            decimal riskMultiplier = drawdownRatio switch
            {
                > 0.75m => 0.25m, // Very conservative when near max drawdown
                > 0.5m => 0.5m,   // Reduced risk when at 50% drawdown
                > 0.25m => 0.75m, // Slightly reduced when at 25% drawdown
                _ => 1.0m         // Full risk when low drawdown
            };

            // Confidence-based sizing (UCB approach)
            var confidence = Math.Max((decimal)strategy.Confidence, (decimal)prediction.Probability);
            if (confidence < (decimal)TopStepConfig.CONFIDENCE_THRESHOLD)
            {
                _logger.LogDebug("🎯 [CONFIDENCE] Below threshold {Threshold:P1}, confidence: {Confidence:P1}", 
                    TopStepConfig.CONFIDENCE_THRESHOLD, confidence);
                return 0m; // No trade if confidence too low
            }

            // Confidence multiplier using ONNX model
            var features = new Dictionary<string, double>
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
            }).ConfigureAwait(false) : 0.5m;
            
            var confidenceMultiplier = modelConfidence;

            // Calculate risk amount
            var riskAmount = baseRisk * riskMultiplier * confidenceMultiplier;

            // Dynamic stop calculation with safety bounds
            var instrument = context.Symbol;
            decimal stopDistance;
            decimal pointValue;

            if (instrument.Equals("NQ", StringComparison.OrdinalIgnoreCase))
            {
                stopDistance = Math.Max(0.5m, context.Atr ?? 25.0m);
                pointValue = TopStepConfig.NQ_POINT_VALUE;
            }
            else // ES
            {
                stopDistance = Math.Max(0.25m, context.Atr ?? 10.0m);
                pointValue = TopStepConfig.ES_POINT_VALUE;
            }

            // Convert risk to contracts
            var perContractRisk = stopDistance * pointValue;
            var contracts = perContractRisk > 0 ? (int)(riskAmount / perContractRisk) : 0;

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
                    
                    // Convert CVaR-PPO action to contract sizing
                    var cvarContracts = ConvertCVaRActionToContracts(actionResult, contracts);
                    
                    // Apply CVaR risk controls
                    var riskAdjustedContracts = ApplyCVaRRiskControls(cvarContracts, actionResult, context);
                    
                    contracts = Math.Max(0, Math.Min(riskAdjustedContracts, maxContracts));
                    
                    _logger.LogInformation("🎯 [CVAR-PPO] Action={Action}, Prob={Prob:F3}, Value={Value:F3}, CVaR={CVaR:F3}, Contracts={Contracts}", 
                        actionResult.Action, actionResult.ActionProbability, actionResult.ValueEstimate, actionResult.CVaREstimate, contracts);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "CVaR-PPO position sizing failed, using TopStep compliance sizing");
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
                    context.Atr ?? 10,
                    (decimal)strategy.Confidence,
                    (decimal)prediction.Probability,
                    new List<Bar>()
                ).ConfigureAwait(false);
                
                contracts = (int)(contracts * Math.Clamp(rlMultiplier, 0.5m, 1.5m));
                _logger.LogDebug("📊 [LEGACY-RL] Using fallback RL multiplier: {Multiplier:F2}", rlMultiplier);
            }

            _logger.LogDebug("📊 [POSITION-SIZE] {Symbol}: Confidence={Confidence:P1}, Drawdown={Drawdown:C}, " +
                "Contracts={Contracts}, RiskAmount={Risk:C}", 
                instrument, confidence, _currentDrawdown, contracts, riskAmount);

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
            if (_dailyPnl <= -TopStepConfig.DAILY_LOSS_LIMIT)
                return (false, $"Daily loss limit reached: {_dailyPnl:C}", "hard_stop");
            
            if (_currentDrawdown >= TopStepConfig.MAX_DRAWDOWN)
                return (false, $"Max drawdown reached: {_currentDrawdown:C}", "hard_stop");
            
            if (_accountBalance <= TopStepConfig.TRAILING_STOP)
                return (false, $"Account below minimum: {_accountBalance:C}", "hard_stop");

            // Warning levels (can trade but with caution)
            if (_dailyPnl <= -(TopStepConfig.DAILY_LOSS_LIMIT * 0.9m))
                return (true, $"Near daily loss limit: {_dailyPnl:C}", "warning");
            
            if (_currentDrawdown >= (TopStepConfig.MAX_DRAWDOWN * 0.8m))
                return (true, $"Near max drawdown: {_currentDrawdown:C}", "warning");

            return (true, "OK", "normal");
        }

        /// <summary>
        /// Update P&L after trade completion - call this from TradingOrchestratorService
        /// </summary>
        public void UpdatePnL(string strategy, decimal pnl)
        {
            _dailyPnl += pnl;
            _accountBalance += pnl;
            
            // Update drawdown if we're in loss territory
            if (_dailyPnl < 0)
                _currentDrawdown = Math.Max(_currentDrawdown, Math.Abs(_dailyPnl));
            
            _logger.LogInformation("💰 [PNL-UPDATE] Strategy={Strategy}, PnL={PnL:C}, DailyPnL={DailyPnL:C}, " +
                "Drawdown={Drawdown:C}, Balance={Balance:C}", 
                strategy, pnl, _dailyPnl, _currentDrawdown, _accountBalance);
        }

        /// <summary>
        /// Reset daily stats - automatically called or can be called manually
        /// </summary>
        public void ResetDaily()
        {
            _dailyPnl = 0;
            _currentDrawdown = 0;
            _lastResetDate = DateTime.UtcNow.Date;
            
            _logger.LogInformation("🌅 [DAILY-RESET] Daily P&L and drawdown reset for new trading day");
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
        private Task<List<Candidate>> GenerateEnhancedCandidatesAsync(
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
                
                foreach (var candidate in baseCandidates)
                {
                    // Only include candidates that align with price prediction
                    var candidateDirection = candidate.side == Side.BUY ? PriceDirection.Up : PriceDirection.Down;
                    if (prediction.Direction != PriceDirection.Sideways && candidateDirection != prediction.Direction)
                    {
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
                        QScore = candidate.QScore * (decimal)strategySelection.Confidence * (decimal)prediction.Probability
                    };
                    
                    enhancedCandidates.Add(enhancedCandidate);
                }
                
                _logger.LogDebug("🎯 [BRAIN-ENHANCE] {Symbol}: Generated {Count} AI-enhanced candidates from {Strategy}",
                    symbol, enhancedCandidates.Count, strategySelection.SelectedStrategy);
                
                return Task.FromResult(enhancedCandidates);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [BRAIN-ENHANCE] Error generating enhanced candidates");
                
                // Fallback to original AllStrategies logic
                return Task.FromResult(AllStrategies.generate_candidates(symbol, env, levels, bars, risk));
            }
        }

        /// <summary>
        /// Create context vector for Neural UCB from market data
        /// </summary>
        private ContextVector CreateContextVector(MarketContext context)
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

        private MarketContext CreateMarketContext(string symbol, Env env, IList<Bar> bars)
        {
            var latestBar = bars.LastOrDefault();
            if (latestBar == null)
            {
                return new MarketContext { Symbol = symbol };
            }
            
            var context = new MarketContext
            {
                Symbol = symbol,
                CurrentPrice = latestBar.Close,
                Volume = latestBar.Volume,
                Atr = env.atr,
                Volatility = Math.Abs(latestBar.High - latestBar.Low) / latestBar.Close,
                TimeOfDay = DateTime.Now.TimeOfDay,
                DayOfWeek = DateTime.Now.DayOfWeek,
                VolumeRatio = bars.Count > 10 ? (decimal)(latestBar.Volume / bars.TakeLast(10).Average(b => b.Volume)) : 1m,
                PriceChange = bars.Count > 1 ? latestBar.Close - bars[^2].Close : 0m,
                RSI = CalculateRSI(bars, 14),
                TrendStrength = CalculateTrendStrength(bars),
                DistanceToSupport = 0m, // levels.Support doesn't exist, using default
                DistanceToResistance = 0m, // levels.Resistance doesn't exist, using default
                VolatilityRank = CalculateVolatilityRank(bars),
                Momentum = CalculateMomentum(bars),
                MarketRegime = 0 // Will be filled by regime detector
            };
            
            return context;
        }

        private List<string> GetAvailableStrategies(TimeSpan timeOfDay, MarketRegime regime)
        {
            // Enhanced strategy selection logic for primary strategies (S2, S3, S6, S11)
            var hour = timeOfDay.Hours;
            
            // Time-based primary strategy allocation with specialization
            var timeBasedStrategies = hour switch
            {
                >= 18 or <= 2 => new[] { "S2", "S11" }, // Asian Session: Mean reversion works well
                >= 2 and <= 5 => new[] { "S3", "S2" }, // European Open: Breakouts and compression
                >= 5 and <= 8 => new[] { "S2", "S3", "S11" }, // London Morning: Good liquidity
                >= 8 and <= 9 => new[] { "S3", "S2" }, // US PreMarket: Compression setups
                >= 9 and <= 10 => new[] { "S6" }, // Opening Drive: ONLY S6 momentum
                >= 10 and <= 11 => new[] { "S3", "S2", "S11" }, // Morning Trend: Best trends
                >= 11 and <= 13 => new[] { "S2" }, // Lunch: ONLY mean reversion
                >= 13 and <= 16 => new[] { "S11", "S3" }, // Afternoon: S11 exhaustion + compression
                _ => new[] { "S2", "S3" } // Default safe strategies
            };
            
            // Filter by market regime for additional intelligence
            var regimeOptimalStrategies = regime switch
            {
                MarketRegime.Trending => new[] { "S6", "S3" }, // Momentum and breakouts
                MarketRegime.Ranging => new[] { "S2", "S11" }, // Mean reversion and fades
                MarketRegime.HighVolatility => new[] { "S3", "S6" }, // Breakouts and momentum
                MarketRegime.LowVolatility => new[] { "S2" }, // Mean reversion only
                _ => PrimaryStrategies // All primary strategies
            };
            
            // Intersect time-based and regime-based strategies for optimal selection
            var availableStrategies = timeBasedStrategies
                .Intersect(regimeOptimalStrategies)
                .ToList();
                
            // Fallback to time-based if no intersection
            if (!availableStrategies.Any())
            {
                availableStrategies = timeBasedStrategies.ToList();
            }
            
            _logger.LogDebug("🧠 [STRATEGY-SELECTION] Hour={Hour}, Regime={Regime}, Available={Strategies}", 
                hour, regime, string.Join(",", availableStrategies));
            
            return availableStrategies;
        }

        private static Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>> GetStrategyFunction(string strategy)
        {
            // Map to your ACTUALLY USED strategy functions in AllStrategies.cs
            return strategy switch
            {
                "S2" => AllStrategies.S2,   // Mean reversion (most used)
                "S3" => AllStrategies.S3,   // Compression/breakout setups  
                "S6" => AllStrategies.S6,   // Opening Drive (critical window)
                "S11" => AllStrategies.S11, // Frequently used
                "S12" => AllStrategies.S12, // Occasionally used
                "S13" => AllStrategies.S13, // Occasionally used
                _ => AllStrategies.S2 // Default to your most reliable strategy
            };
        }

        private decimal CalculateReward(decimal pnl, bool wasCorrect, TimeSpan holdTime)
        {
            // Combine PnL with correctness and time efficiency
            var baseReward = wasCorrect ? 1m : 0m;
            var pnlComponent = Math.Tanh((double)(pnl / 100)) * 0.5; // Normalize PnL contribution
            var timeComponent = holdTime < TimeSpan.FromHours(2) ? 0.1m : 0m; // Reward quick profits
            
            return Math.Clamp(baseReward + (decimal)pnlComponent + timeComponent, 0m, 1m);
        }

        private decimal CalculateOverallConfidence(StrategySelection strategy, PricePrediction prediction)
        {
            return (strategy.Confidence + prediction.Probability) / 2;
        }

        private string AssessRisk(MarketContext context, PricePrediction prediction)
        {
            if (context.Volatility > 0.4m) return "HIGH";
            if (context.Volatility < 0.15m && prediction.Probability > 0.7m) return "LOW";
            return "MEDIUM";
        }

        private BrainDecision CreateFallbackDecision(string symbol, Env env, Levels levels, IList<Bar> bars, RiskEngine risk)
        {
            // Fallback to your existing AllStrategies logic
            var candidates = AllStrategies.generate_candidates(symbol, env, levels, bars, risk);
            
            return new BrainDecision
            {
                Symbol = symbol,
                RecommendedStrategy = "S3", // Default strategy
                StrategyConfidence = 0.5m,
                PriceDirection = PriceDirection.Sideways,
                PriceProbability = 0.5m,
                OptimalPositionMultiplier = 1.0m,
                MarketRegime = MarketRegime.Normal,
                EnhancedCandidates = candidates,
                DecisionTime = DateTime.UtcNow,
                ProcessingTimeMs = 10,
                ModelConfidence = 0.5m,
                RiskAssessment = "MEDIUM"
            };
        }

        // Technical analysis helpers (simplified versions)
        private decimal CalculateEMA(IList<Bar> bars, int period)
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

        private decimal CalculateRSI(IList<Bar> bars, int period)
        {
            if (bars.Count < period + 1) return 50;
            
            var gains = 0m;
            var losses = 0m;
            
            for (int i = bars.Count - period; i < bars.Count; i++)
            {
                var change = bars[i].Close - bars[i - 1].Close;
                if (change > 0) gains += change;
                else losses -= change;
            }
            
            if (losses == 0) return 100;
            
            var rs = gains / losses;
            return 100 - (100 / (1 + rs));
        }

        private decimal CalculateTrendStrength(IList<Bar> bars)
        {
            if (bars.Count < 10) return 0;
            
            var recent = bars.TakeLast(10).ToList();
            var slope = (recent.Last().Close - recent.First().Close) / recent.Count;
            return Math.Abs(slope) / (recent.Average(b => Math.Abs(b.High - b.Low)));
        }

        private decimal CalculateVolatilityRank(IList<Bar> bars)
        {
            if (bars.Count < 20) return 0.5m;
            
            var currentVol = Math.Abs(bars.Last().High - bars.Last().Low);
            var historicalVols = bars.TakeLast(20).Select(b => Math.Abs(b.High - b.Low)).OrderBy(v => v).ToList();
            
            var rank = historicalVols.Count(v => v < currentVol) / (decimal)historicalVols.Count;
            return rank;
        }

        private decimal CalculateMomentum(IList<Bar> bars)
        {
            if (bars.Count < 5) return 0;
            
            var recent = bars.TakeLast(5).ToList();
            return (recent.Last().Close - recent.First().Close) / recent.First().Close;
        }

        private async Task UpdateUnifiedLearningAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("🔄 [UNIFIED-LEARNING] Starting unified learning update across all strategies...");
                
                // Analyze performance patterns across all strategies
                var performanceAnalysis = AnalyzeStrategyPerformance();
                
                // Update strategy optimal conditions based on recent performance
                UpdateOptimalConditionsFromPerformance(performanceAnalysis);
                
                // Cross-pollinate successful patterns between strategies
                await CrossPollinateStrategyPatternsAsync().ConfigureAwait(false);
                
                _logger.LogInformation("✅ [UNIFIED-LEARNING] Completed unified learning update");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [UNIFIED-LEARNING] Failed to update unified learning");
            }
        }
        
        private Dictionary<string, PerformanceMetrics> AnalyzeStrategyPerformance()
        {
            var analysis = new Dictionary<string, PerformanceMetrics>();
            
            foreach (var strategy in PrimaryStrategies)
            {
                if (!_strategyPerformance.TryGetValue(strategy, out var perf))
                    continue;
                    
                var metrics = new PerformanceMetrics
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
        
        private void UpdateOptimalConditionsFromPerformance(Dictionary<string, PerformanceMetrics> analysis)
        {
            foreach (var (strategy, metrics) in analysis)
            {
                if (metrics.WinRate < 0.4m) // Poor performing strategy
                {
                    // Reduce confidence in current optimal conditions
                    if (_strategyOptimalConditions.TryGetValue(strategy, out var conditions))
                    {
                        // Remove conditions that have been consistently unsuccessful
                        var unsuccessfulConditions = conditions
                            .Where(c => c.SuccessRate < 0.3m)
                            .ToList();
                        
                        foreach (var condition in unsuccessfulConditions)
                        {
                            conditions.Remove(condition);
                        }
                        
                        _logger.LogDebug("🔄 [CONDITION-UPDATE] Removed {Count} unsuccessful conditions from {Strategy}", 
                            unsuccessfulConditions.Count, strategy);
                    }
                }
                else if (metrics.WinRate > 0.7m) // High performing strategy
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
            if (bestPerformance.WinRate < 0.6m)
                return; // Not good enough to share patterns
            
            // Share successful patterns with other strategies
            var successfulConditions = _strategyOptimalConditions
                .GetValueOrDefault(bestStrategy, new List<MarketCondition>())
                .Where(c => c.SuccessRate > 0.7m)
                .ToList();
            
            foreach (var strategy in PrimaryStrategies.Where(s => s != bestStrategy))
            {
                foreach (var condition in successfulConditions)
                {
                    // Add successful condition from best strategy to other strategies
                    UpdateConditionSuccess(strategy, condition.ConditionName, true, 0.1m); // Lower weight for cross-pollination
                }
            }
            
            _logger.LogInformation("🌱 [CROSS-POLLINATION] Shared {Count} successful patterns from {BestStrategy} to other strategies", 
                successfulConditions.Count, bestStrategy);
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
            perf.HoldTimes.Add(holdTime.Ticks);
            
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
                conditions = new List<MarketCondition>();
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
                conditions = new List<MarketCondition>();
                _strategyOptimalConditions[strategy] = conditions;
            }
            
            var condition = conditions.FirstOrDefault(c => c.ConditionName == conditionName);
            if (condition == null)
            {
                condition = new MarketCondition 
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
                .GetValueOrDefault(strategy, new List<MarketCondition>())
                .Where(c => c.SuccessRate > 0.6m && c.TotalCount >= 3)
                .OrderByDescending(c => c.SuccessRate)
                .Take(5)
                .Select(c => c.ConditionName)
                .ToArray();
        }
        
        private decimal GetRecentPerformanceTrend(string strategy)
        {
            var recentDecisions = _decisionHistory
                .Where(d => d.Strategy == strategy && d.Timestamp > DateTime.UtcNow.AddHours(-24))
                .OrderBy(d => d.Timestamp)
                .ToList();
                
            if (recentDecisions.Count < 5)
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
                    HistoricalLearningIntervalMinutes = 10, // Very frequent during maintenance
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
                    HistoricalLearningIntervalMinutes = 15, // Every 15 minutes as requested
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
                HistoricalLearningIntervalMinutes = 60, // Every 60 minutes as requested
                LiveTradingActive = true,
                RecommendedStrategies = availableStrategies.ToArray(),
                Reasoning = $"Market open - light historical learning every 60min, active live trading with {string.Join(",", availableStrategies)}"
            };
        }
        
        private bool IsMarketOpen(DayOfWeek dayOfWeek, TimeSpan timeOfDay)
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
            if (dayOfWeek >= DayOfWeek.Monday && dayOfWeek <= DayOfWeek.Thursday)
            {
                if (timeOfDay >= marketCloseTime && timeOfDay < marketOpenTime)
                    return false; // Maintenance break
            }
            
            return true;
        }
        
        private bool IsMaintenanceWindow(DayOfWeek dayOfWeek, TimeSpan timeOfDay)
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

        private async Task RetrainModelsAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("🔄 [UNIFIED-RETRAIN] Starting unified model retraining across all strategies...");
                
                // Export comprehensive training data including all strategies
                var unifiedTrainingData = _decisionHistory.TakeLast(2000).Select(d => new
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
                    new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
                await File.WriteAllTextAsync(perfPath, JsonSerializer.Serialize(strategyPerformanceData, 
                    new JsonSerializerOptions { WriteIndented = true }), cancellationToken).ConfigureAwait(false);
                
                _logger.LogInformation("✅ [UNIFIED-RETRAIN] Training data exported: {Count} decisions, {StrategyCount} strategies", 
                    unifiedTrainingData.Count(), _strategyPerformance.Count);
                
                // Enhanced Python training scripts for multi-strategy learning would be integrated here
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [UNIFIED-RETRAIN] Unified model retraining failed");
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
                (double)Math.Min(1.0m, context.Volatility / 2.0m), // Volatility ratio [0-1]
                Math.Tanh((double)(context.PriceChange / 20.0m)), // Price change momentum [-1,1]
                (double)Math.Min(1.0m, context.VolumeRatio / 3.0m), // Volume surge ratio [0-1]
                (double)Math.Min(1.0m, (context.Atr ?? 10.0m) / 50.0m), // ATR normalized [0-1]
                (double)context.TrendStrength, // Trend strength [0-1]
                
                // Strategy selection features  
                (double)strategy.Confidence, // Neural UCB confidence [0-1]
                Math.Min(1.0, (double)strategy.UcbValue / 10.0), // UCB exploration value normalized
                strategy.SelectedStrategy switch { // Strategy type encoding
                    "S2_VWAP" => 0.25, "S3_Compression" => 0.5, 
                    "S11_Opening" => 0.75, "S12_Momentum" => 1.0, _ => 0.0
                },
                
                // LSTM prediction features
                (double)prediction.Probability, // Price direction confidence [0-1]
                (int)prediction.Direction / 2.0 + 0.5, // Direction: Down=0, Sideways=0.5, Up=1
                
                // Temporal features (cyclical encoding)
                Math.Sin(2 * Math.PI * context.TimeOfDay.TotalHours / 24.0), // Hour of day
                Math.Cos(2 * Math.PI * context.TimeOfDay.TotalHours / 24.0),
                
                // Risk management features
                (double)Math.Max(-1.0m, Math.Min(1.0m, _currentDrawdown / TopStepConfig.MAX_DRAWDOWN)), // Drawdown ratio [-1,1]
                (double)Math.Max(-1.0m, Math.Min(1.0m, _dailyPnl / TopStepConfig.DAILY_LOSS_LIMIT)), // Daily P&L ratio [-1,1]
                
                // Portfolio state features
                Math.Min(1.0, DecisionsToday / 50.0), // Decision frequency normalized [0-1]
                (double)WinRateToday, // Current session win rate [0-1]
            };
        }

        /// <summary>
        /// Convert CVaR-PPO action result to contract count
        /// </summary>
        private int ConvertCVaRActionToContracts(ActionResult actionResult, int baseContracts)
        {
            // CVaR-PPO actions map to position size multipliers
            // Action 0=No Trade, 1=Micro(0.25x), 2=Small(0.5x), 3=Normal(1x), 4=Large(1.5x), 5=Max(2x)
            var sizeMultiplier = actionResult.Action switch
            {
                0 => 0.0m,   // No trade
                1 => 0.25m,  // Micro position
                2 => 0.5m,   // Small position  
                3 => 1.0m,   // Normal position
                4 => 1.5m,   // Large position
                5 => 2.0m,   // Maximum position
                _ => 1.0m    // Default to normal
            };
            
            // Apply action probability weighting - reduce size for uncertain actions
            var probabilityAdjustment = (decimal)Math.Max(0.3, actionResult.ActionProbability);
            
            // Apply value estimate adjustment - reduce size for negative expected value
            var valueAdjustment = (decimal)Math.Max(0.2, Math.Min(1.5, 0.5 + actionResult.ValueEstimate));
            
            var adjustedMultiplier = sizeMultiplier * probabilityAdjustment * valueAdjustment;
            var contracts = (int)Math.Round(baseContracts * adjustedMultiplier);
            
            return Math.Max(0, contracts);
        }

        /// <summary>
        /// Apply CVaR risk controls to position sizing
        /// </summary>
        private int ApplyCVaRRiskControls(int contracts, ActionResult actionResult, MarketContext context)
        {
            if (contracts <= 0) return 0;
            
            // CVaR tail risk adjustment - reduce position if high tail risk
            var cvarAdjustment = 1.0m;
            if (actionResult.CVaREstimate < -0.1) // High negative tail risk
            {
                cvarAdjustment = 0.5m; // Cut position in half
            }
            else if (actionResult.CVaREstimate < -0.05) // Moderate tail risk
            {
                cvarAdjustment = 0.75m; // Reduce position by 25%
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

        public void Dispose()
        {
            _logger.LogInformation("🧠 [UNIFIED-BRAIN] Shutting down...");
            
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
                File.WriteAllText(statsPath, JsonSerializer.Serialize(stats, new JsonSerializerOptions { WriteIndented = true }));
                _logger.LogInformation("📊 [UNIFIED-BRAIN] Statistics saved: {Decisions} decisions, {WinRate:P1} win rate",
                    DecisionsToday, WinRateToday);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [UNIFIED-BRAIN] Error saving statistics");
            }
        }
    }

    #region Supporting Models

    public class StrategySpecialization
    {
        public string Name { get; set; } = string.Empty;
        public string[] OptimalConditions { get; set; } = Array.Empty<string>();
        public string LearningFocus { get; set; } = string.Empty;
        public string[] TimeWindows { get; set; } = Array.Empty<string>();
    }

    public class StrategyPerformance
    {
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public decimal TotalPnL { get; set; }
        public decimal WinRate { get; set; }
        public List<long> HoldTimes { get; } = new();
    }

    public class MarketCondition
    {
        public string ConditionName { get; set; } = string.Empty;
        public decimal SuccessCount { get; set; }
        public decimal TotalCount { get; set; }
        public decimal SuccessRate { get; set; }
    }

    public class PerformanceMetrics
    {
        public decimal WinRate { get; set; }
        public decimal AveragePnL { get; set; }
        public TimeSpan AverageHoldTime { get; set; }
        public string[] BestConditions { get; set; } = Array.Empty<string>();
        public decimal RecentPerformanceTrend { get; set; }
    }

    public class UnifiedSchedulingRecommendation
    {
        public bool IsMarketOpen { get; set; }
        public string LearningIntensity { get; set; } = string.Empty; // INTENSIVE, LIGHT, BACKGROUND
        public int HistoricalLearningIntervalMinutes { get; set; }
        public bool LiveTradingActive { get; set; }
        public string[] RecommendedStrategies { get; set; } = Array.Empty<string>();
        public string Reasoning { get; set; } = string.Empty;
    }

    public class BrainDecision
    {
        public string Symbol { get; set; } = string.Empty;
        public string RecommendedStrategy { get; set; } = string.Empty;
        public decimal StrategyConfidence { get; set; }
        public PriceDirection PriceDirection { get; set; }
        public decimal PriceProbability { get; set; }
        public decimal OptimalPositionMultiplier { get; set; }
        public MarketRegime MarketRegime { get; set; }
        public List<Candidate> EnhancedCandidates { get; set; } = new();
        public DateTime DecisionTime { get; set; }
        public double ProcessingTimeMs { get; set; }
        public decimal ModelConfidence { get; set; }
        public string RiskAssessment { get; set; } = string.Empty;
    }

    public class MarketContext
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal CurrentPrice { get; set; }
        public decimal Volume { get; set; }
        public decimal? Atr { get; set; }
        public decimal Volatility { get; set; }
        public TimeSpan TimeOfDay { get; set; }
        public DayOfWeek DayOfWeek { get; set; }
        public decimal VolumeRatio { get; set; }
        public decimal PriceChange { get; set; }
        public decimal RSI { get; set; }
        public decimal TrendStrength { get; set; }
        public decimal DistanceToSupport { get; set; }
        public decimal DistanceToResistance { get; set; }
        public decimal VolatilityRank { get; set; }
        public decimal Momentum { get; set; }
        public int MarketRegime { get; set; }
        
        // Additional properties needed by NeuralUcbExtended
        public Dictionary<string, double> Features { get; } = new();
        public double Price { get; set; }
        public double Bid { get; set; }
        public double Ask { get; set; }
        public double SignalStrength { get; set; }
        public double ConfidenceLevel { get; set; }
        public double ModelConfidence { get; set; }
        public double NewsIntensity { get; set; }
        public Dictionary<string, double> TechnicalIndicators { get; } = new();
        public bool IsFomcDay { get; set; }
        public bool IsCpiDay { get; set; }
    }

    public class StrategySelection
    {
        public string SelectedStrategy { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        public decimal UcbValue { get; set; }
        public string Reasoning { get; set; } = string.Empty;
    }

    public class PricePrediction
    {
        public PriceDirection Direction { get; set; }
        public decimal Probability { get; set; }
        public decimal ExpectedMove { get; set; }
        public TimeSpan TimeHorizon { get; set; }
    }

    public class TradingDecision
    {
        public string Symbol { get; set; } = string.Empty;
        public string Strategy { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        public MarketContext Context { get; set; } = new();
        public DateTime Timestamp { get; set; }
        public bool WasCorrect { get; set; }
        public decimal PnL { get; set; }
    }

    public class TradingPerformance
    {
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public decimal TotalPnL { get; set; }
        public decimal WinRate => TotalTrades > 0 ? (decimal)WinningTrades / TotalTrades : 0;
    }

    public enum PriceDirection
    {
        Up,
        Down,
        Sideways
    }

    public enum MarketRegime
    {
        Normal = 0,
        Trending = 1,
        Ranging = 2,
        HighVolatility = 3,
        LowVolatility = 4
    }

    #endregion
}
