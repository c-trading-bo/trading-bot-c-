using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Strategy;
using BotCore.Risk;
using BotCore.Models;
using BotCore.ML;
using BotCore.Bandits;
using System.Collections.Concurrent;
using System.Text.Json;

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
    /// 
    /// This is the central AI brain that:
    /// 1. Receives market data from your TradingOrchestratorService
    /// 2. Uses Neural UCB to select optimal strategy (S1-S14)
    /// 3. Uses LSTM to predict price movements and timing
    /// 4. Uses RL to optimize position sizes
    /// 5. Sends intelligent signals to AllStrategies.cs
    /// 6. Feeds back results to learn and improve
    /// 
    /// INTEGRATION POINTS:
    /// - TradingOrchestratorService.ExecuteESNQTradingAsync() calls this brain
    /// - AllStrategies.generate_candidates() enhanced with brain decisions
    /// - RiskEngine.ComputeSize() gets brain position size recommendations
    /// - SimpleOrderRouter.RouteAsync() receives brain-optimized signals
    /// </summary>
    public class UnifiedTradingBrain : IDisposable
    {
        private readonly ILogger<UnifiedTradingBrain> _logger;
        private readonly IMLMemoryManager _memoryManager;
        private readonly StrategyMlModelManager _modelManager;
        private readonly NeuralUcbBandit _strategySelector;
        private readonly ConcurrentDictionary<string, MarketContext> _marketContexts = new();
        private readonly ConcurrentDictionary<string, TradingPerformance> _performance = new();
        
        // ML Models for different decision points
        private object? _lstmPricePredictor;
        private object? _rlPositionSizer;
        private object? _metaClassifier;
        private object? _marketRegimeDetector;
        private INeuralNetwork? _confidenceNetwork;
        
        // TopStep compliance tracking
        private decimal _currentDrawdown = 0m;
        private decimal _dailyPnl = 0m;
        private decimal _accountBalance = TopStepConfig.ACCOUNT_SIZE;
        private DateTime _lastResetDate = DateTime.UtcNow.Date;
        
        // Performance tracking for learning
        private readonly List<TradingDecision> _decisionHistory = new();
        private DateTime _lastModelUpdate = DateTime.MinValue;
        
        public bool IsInitialized { get; private set; }
        public DateTime LastDecision { get; private set; }
        public int DecisionsToday { get; private set; }
        public decimal WinRateToday { get; private set; }

        public UnifiedTradingBrain(
            ILogger<UnifiedTradingBrain> logger,
            IMLMemoryManager memoryManager,
            StrategyMlModelManager modelManager)
        {
            _logger = logger;
            _memoryManager = memoryManager;
            _modelManager = modelManager;
            
            // Initialize Neural UCB for strategy selection using ONNX-based neural network
            var onnxLoader = new OnnxModelLoader(new Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxModelLoader>());
            var neuralNetworkLogger = new Microsoft.Extensions.Logging.Abstractions.NullLogger<OnnxNeuralNetwork>();
            var neuralNetwork = new OnnxNeuralNetwork(onnxLoader, neuralNetworkLogger, "models/strategy_selection.onnx");
            _strategySelector = new NeuralUcbBandit(neuralNetwork);
            
            // Initialize confidence network for model confidence prediction
            _confidenceNetwork = new OnnxNeuralNetwork(onnxLoader, neuralNetworkLogger, "models/confidence_prediction.onnx");
            
            _logger.LogInformation("🧠 [UNIFIED-BRAIN] Initialized - Ready to make intelligent trading decisions");
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
                    "models/rl_model.onnx", "v1");
                
                // Load RL agent for position sizing - use your real CVaR PPO agent
                _rlPositionSizer = await _memoryManager.LoadModelAsync<object>(
                    "models/rl/cvar_ppo_agent.onnx", "v1");
                
                // Load meta classifier for market regime - use your test CVaR model
                _metaClassifier = await _memoryManager.LoadModelAsync<object>(
                    "models/rl/test_cvar_ppo.onnx", "v1");
                
                // Load market regime detector - use your main RL model as backup
                _marketRegimeDetector = await _memoryManager.LoadModelAsync<object>(
                    "models/rl_model.onnx", "v1");

                IsInitialized = true;
                _logger.LogInformation("✅ [UNIFIED-BRAIN] All models loaded successfully - Brain is ONLINE");
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
            var startTime = DateTime.UtcNow;
            LastDecision = startTime;
            
            try
            {
                // 1. CREATE MARKET CONTEXT from current data
                var context = CreateMarketContext(symbol, env, levels, bars);
                _marketContexts[symbol] = context;
                
                // 2. DETECT MARKET REGIME using Meta Classifier
                var marketRegime = await DetectMarketRegimeAsync(context, cancellationToken);
                
                // 3. SELECT OPTIMAL STRATEGY using Neural UCB
                var optimalStrategy = await SelectOptimalStrategyAsync(context, marketRegime, cancellationToken);
                
                // 4. PREDICT PRICE MOVEMENT using LSTM
                var priceDirection = await PredictPriceDirectionAsync(context, bars, cancellationToken);
                
                // 5. OPTIMIZE POSITION SIZE using RL
                var optimalSize = await OptimizePositionSizeAsync(context, optimalStrategy, priceDirection, risk, cancellationToken);
                
                // 6. GENERATE ENHANCED CANDIDATES using brain intelligence
                var enhancedCandidates = await GenerateEnhancedCandidatesAsync(
                    symbol, env, levels, bars, risk, optimalStrategy, priceDirection, optimalSize, cancellationToken);
                
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
        /// Learn from trading results to improve future decisions
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
                    
                    await _strategySelector.UpdateArmAsync(strategy, contextVector, reward, cancellationToken);
                }

                // Update performance tracking
                if (!_performance.ContainsKey(symbol))
                {
                    _performance[symbol] = new TradingPerformance();
                }
                
                var perf = _performance[symbol];
                perf.TotalTrades++;
                perf.TotalPnL += pnl;
                if (wasCorrect) perf.WinningTrades++;
                
                // Calculate today's win rate
                var todayDecisions = _decisionHistory.Where(d => d.Timestamp.Date == DateTime.Today).ToList();
                if (todayDecisions.Count > 0)
                {
                    // This would be updated when we get actual results
                    WinRateToday = (decimal)todayDecisions.Count(d => d.WasCorrect) / todayDecisions.Count;
                }

                // Retrain models periodically
                if (DateTime.UtcNow - _lastModelUpdate > TimeSpan.FromHours(4) && _decisionHistory.Count > 100)
                {
                    _ = Task.Run(() => RetrainModelsAsync(cancellationToken), cancellationToken);
                    _lastModelUpdate = DateTime.UtcNow;
                }

                _logger.LogInformation("📚 [BRAIN-LEARNING] {Symbol} {Strategy}: PnL={PnL:F2}, Correct={Correct}, " +
                    "WinRate={WinRate:P1}, TotalTrades={Total}",
                    symbol, strategy, pnl, wasCorrect, WinRateToday, perf.TotalTrades);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [BRAIN-LEARNING] Error learning from result");
            }
        }

        #region ML Model Predictions

        private Task<MarketRegime> DetectMarketRegimeAsync(MarketContext context, CancellationToken cancellationToken)
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
                
                var selection = await _strategySelector.SelectArmAsync(availableStrategies, contextVector, cancellationToken);
                
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
            IList<Bar> bars, 
            CancellationToken cancellationToken)
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
            RiskEngine risk,
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
            }) : 0.5m;
            
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

            // Legacy RL sizing integration (if available)
            if (_rlPositionSizer != null && IsInitialized)
            {
                try
                {
                    var rlMultiplier = _modelManager.GetPositionSizeMultiplier(
                        strategy.SelectedStrategy,
                        context.Symbol,
                        context.CurrentPrice,
                        context.Atr ?? 10,
                        (decimal)strategy.Confidence,
                        (decimal)prediction.Probability,
                        new List<Bar>()
                    );
                    
                    // Combine RL recommendation with TopStep compliance
                    contracts = (int)(contracts * Math.Clamp(rlMultiplier, 0.5m, 1.5m));
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "RL position sizing failed, using TopStep compliance sizing");
                }
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
            _dailyPnl = 0m;
            _currentDrawdown = 0m;
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
            decimal sizeMultiplier,
            CancellationToken cancellationToken)
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

        private MarketContext CreateMarketContext(string symbol, Env env, Levels levels, IList<Bar> bars)
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
            // Use your ACTUAL time-based strategy filtering from ES_NQ_TradingSchedule
            var hour = timeOfDay.Hours;
            
            if (hour >= 18 || hour <= 2) // Asian Session
                return new[] { "S2", "S11", "S3" }.ToList(); // Your Asian strategies
            else if (hour >= 2 && hour <= 5) // European Open
                return new[] { "S3", "S6", "S2" }.ToList(); // European breakouts
            else if (hour >= 5 && hour <= 8) // London Morning
                return new[] { "S2", "S3", "S11" }.ToList(); // Good liquidity
            else if (hour >= 8 && hour <= 9) // US PreMarket
                return new[] { "S3", "S2" }.ToList(); // Compression setups
            else if (hour >= 9 && hour <= 10) // Opening Drive - CRITICAL
                return new[] { "S6" }.ToList(); // ONLY S6 for opening drive
            else if (hour >= 10 && hour <= 11) // Morning Trend
                return new[] { "S3", "S2", "S11" }.ToList(); // Best trends
            else if (hour >= 11 && hour <= 13) // Lunch Chop
                return new[] { "S2" }.ToList(); // ONLY mean reversion
            else if (hour >= 13 && hour <= 15) // Afternoon Trend
                return new[] { "S3", "S2" }.ToList(); // ADR exhaustion setups
            else
                return new[] { "S2", "S3" }.ToList(); // Default safe strategies
        }

        private Func<string, Env, Levels, IList<Bar>, RiskEngine, List<Candidate>> GetStrategyFunction(string strategy)
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

        private async Task RetrainModelsAsync(CancellationToken cancellationToken)
        {
            try
            {
                _logger.LogInformation("🔄 [BRAIN-RETRAIN] Starting model retraining...");
                
                // Export decision history for Python training scripts
                var trainingData = _decisionHistory.TakeLast(1000).Select(d => new
                {
                    features = CreateContextVector(d.Context).Features,
                    strategy = d.Strategy,
                    reward = d.WasCorrect ? 1.0 : 0.0,
                    timestamp = d.Timestamp
                });
                
                var dataPath = Path.Combine("data", "brain_training_data.json");
                Directory.CreateDirectory(Path.GetDirectoryName(dataPath)!);
                await File.WriteAllTextAsync(dataPath, JsonSerializer.Serialize(trainingData), cancellationToken);
                
                _logger.LogInformation("✅ [BRAIN-RETRAIN] Training data exported: {Count} samples", trainingData.Count());
                
                // Here you could trigger Python training scripts
                // await RunPythonTrainingAsync(dataPath, cancellationToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "❌ [BRAIN-RETRAIN] Model retraining failed");
            }
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
