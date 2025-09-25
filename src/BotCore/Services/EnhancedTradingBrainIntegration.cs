using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.ML;
using TradingBot.RLAgent;
using BotCore.Services;
using BotCore.Brain;
using BotCore.Models;
using BotCore.Risk;
using System.Text.Json;
using TradingBot.Abstractions;
using static BotCore.Brain.UnifiedTradingBrain;

namespace BotCore.Services;

/// <summary>
/// Enhanced integration service that coordinates ML/RL/Cloud services with UnifiedTradingBrain
/// This enhances existing trading logic by adding automated model management and feedback loops
/// </summary>
public class EnhancedTradingBrainIntegration
{
    private readonly ILogger<EnhancedTradingBrainIntegration> _logger;
    private readonly UnifiedTradingBrain _tradingBrain;
    private readonly ModelEnsembleService _ensembleService;
    private readonly TradingFeedbackService _feedbackService;
    private readonly CloudModelSynchronizationService _cloudSync;
    private readonly IServiceProvider _serviceProvider;
    
    // Integration state
    private readonly Dictionary<string, DateTime> _lastPredictions = new();
    private readonly Dictionary<string, double> _predictionAccuracies = new();
    private readonly bool _isEnhancementActive = true;
    
    public EnhancedTradingBrainIntegration(
        ILogger<EnhancedTradingBrainIntegration> logger,
        UnifiedTradingBrain tradingBrain,
        ModelEnsembleService ensembleService,
        TradingFeedbackService feedbackService,
        CloudModelSynchronizationService cloudSync,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _tradingBrain = tradingBrain;
        _ensembleService = ensembleService;
        _feedbackService = feedbackService;
        _cloudSync = cloudSync;
        _serviceProvider = serviceProvider;
        
        _logger.LogInformation("🧠 [ENHANCED-BRAIN] Integration service initialized - enhancing existing trading logic");
    }

    /// <summary>
    /// Enhanced decision making that augments UnifiedTradingBrain with ensemble predictions
    /// This ENHANCES existing logic rather than replacing it
    /// </summary>
    public async Task<EnhancedTradingDecision> MakeEnhancedDecisionAsync(
        string symbol,
        Dictionary<string, object> marketContext,
        List<string> availableStrategies,
        CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogDebug("🧠 [ENHANCED-BRAIN] Making enhanced decision for {Symbol}", symbol);
            
            // Step 1: Get original UnifiedTradingBrain decision
            var env = CreateSampleEnv();
            var levels = CreateSampleLevels();
            var bars = CreateSampleBars();
            var risk = CreateSampleRisk();
            
            var originalBrainDecision = await _tradingBrain.MakeIntelligentDecisionAsync(
                symbol, env, levels, bars, risk, cancellationToken).ConfigureAwait(false);
            
            if (!_isEnhancementActive)
            {
                // Return original decision if enhancement is disabled
                return new EnhancedTradingDecision
                {
                    OriginalDecision = ConvertBrainToTradingDecision(originalBrainDecision),
                    EnhancedStrategy = originalBrainDecision.RecommendedStrategy,
                    EnhancedConfidence = originalBrainDecision.StrategyConfidence,
                    EnhancedPositionSize = 1.0m, // Default size
                    EnhancementApplied = false,
                    Timestamp = DateTime.UtcNow
                };
            }
            
            // Step 2: Get ensemble predictions to enhance the decision
            // Convert dictionary marketContext to proper MarketContext object
            var brainMarketContext = ConvertToMarketContext(marketContext);
            var contextVector = ExtractContextVector(brainMarketContext);
            var marketFeatures = ExtractMarketFeatures(brainMarketContext);
            
            // Get ensemble strategy prediction
            var strategyPrediction = await _ensembleService.GetStrategySelectionPredictionAsync(
                contextVector, availableStrategies, cancellationToken).ConfigureAwait(false);
            
            // Get ensemble price direction prediction
            var pricePrediction = await _ensembleService.GetPriceDirectionPredictionAsync(
                marketFeatures, cancellationToken).ConfigureAwait(false);
            
            // Get ensemble CVaR action
            var convertedDecision = ConvertBrainToTradingDecision(originalBrainDecision);
            var state = CreateStateVector(convertedDecision, brainMarketContext);
            var ensembleAction = await _ensembleService.GetEnsembleActionAsync(
                state, true, cancellationToken).ConfigureAwait(false);
            
            // Step 3: Enhance the original decision using ensemble insights
            var enhancedDecision = EnhanceDecision(
                convertedDecision, 
                strategyPrediction, 
                pricePrediction, 
                ensembleAction,
                symbol,
                brainMarketContext);
            
            // Step 4: Log the enhancement
            LogDecisionEnhancement(convertedDecision, enhancedDecision, symbol);
            
            // Step 5: Track prediction for feedback
            TrackPredictionForFeedback(enhancedDecision, symbol, brainMarketContext);
            
            return enhancedDecision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🧠 [ENHANCED-BRAIN] Error in enhanced decision making for {Symbol}", symbol);
            
            // Fallback to original decision on error
            try
            {
                var env = CreateSampleEnv();
                var levels = CreateSampleLevels();
                var bars = CreateSampleBars();
                var risk = CreateSampleRisk();
                
                var originalBrainDecision = await _tradingBrain.MakeIntelligentDecisionAsync(
                    symbol, env, levels, bars, risk, cancellationToken).ConfigureAwait(false);
                
                return new EnhancedTradingDecision
                {
                    OriginalDecision = ConvertBrainToTradingDecision(originalBrainDecision),
                    EnhancedStrategy = originalBrainDecision.RecommendedStrategy,
                    EnhancedConfidence = originalBrainDecision.StrategyConfidence,
                    EnhancedPositionSize = 1.0m,
                    EnhancementApplied = false,
                    ErrorMessage = ex.Message,
                    Timestamp = DateTime.UtcNow
                };
            }
            catch
            {
                // Ultimate fallback
                return new EnhancedTradingDecision
                {
                    OriginalDecision = CreateFallbackTradingDecision(),
                    EnhancedStrategy = "S3", // Safe default
                    EnhancedConfidence = 0.5m,
                    EnhancedPositionSize = 1.0m,
                    EnhancementApplied = false,
                    ErrorMessage = ex.Message,
                    Timestamp = DateTime.UtcNow
                };
            }
        }
    }

    /// <summary>
    /// Enhance the original decision using ensemble predictions
    /// This preserves the original logic while adding intelligent enhancements
    /// </summary>
    private EnhancedTradingDecision EnhanceDecision(
        BotCore.Brain.TradingDecision originalDecision,
        EnsemblePrediction strategyPrediction,
        EnsemblePrediction pricePrediction,
        EnsembleActionResult ensembleAction,
        string symbol,
        BotCore.Brain.MarketContext marketContext)
    {
        var enhancedDecision = new EnhancedTradingDecision
        {
            OriginalDecision = originalDecision,
            StrategyPrediction = strategyPrediction,
            PricePrediction = pricePrediction,
            EnsembleAction = ensembleAction,
            Timestamp = DateTime.UtcNow
        };
        
        // Enhancement 1: Strategy Selection
        enhancedDecision.EnhancedStrategy = EnhanceStrategySelection(
            originalDecision.Strategy, // Use Strategy instead of SelectedStrategy
            strategyPrediction,
            originalDecision.Confidence);
        
        // Enhancement 2: Confidence Adjustment
        enhancedDecision.EnhancedConfidence = EnhanceConfidence(
            originalDecision.Confidence,
            strategyPrediction.Confidence,
            pricePrediction.Confidence);
        
        // Enhancement 3: Position Sizing
        enhancedDecision.EnhancedPositionSize = EnhancePositionSizing(
            1.0m, // Default position size since BotCore.Brain.TradingDecision doesn't have PositionSize
            ensembleAction,
            enhancedDecision.EnhancedConfidence,
            pricePrediction);
        
        // Enhancement 4: Risk Adjustment
        enhancedDecision.EnhancedRiskLevel = EnhanceRiskLevel(
            0.5m, // Default risk level since BotCore.Brain.TradingDecision doesn't have RiskLevel
            ensembleAction.CVaREstimate,
            pricePrediction);
        
        // Enhancement 5: Market Timing
        enhancedDecision.MarketTimingSignal = CalculateMarketTiming(
            pricePrediction,
            ensembleAction,
            marketContext);
        
        enhancedDecision.EnhancementApplied = true;
        enhancedDecision.EnhancementReason = GenerateEnhancementReason(
            originalDecision, strategyPrediction, pricePrediction, ensembleAction);
        
        return enhancedDecision;
    }

    /// <summary>
    /// Enhance strategy selection by combining original with ensemble prediction
    /// </summary>
    private string EnhanceStrategySelection(string originalStrategy, EnsemblePrediction strategyPrediction, decimal originalConfidence)
    {
        // If ensemble prediction is very confident and different, consider switching
        if (strategyPrediction.Confidence > 0.8m && 
            strategyPrediction.Result is StrategyPrediction stratPred &&
            stratPred.SelectedStrategy != originalStrategy &&
            originalConfidence < 0.7m)
        {
            _logger.LogInformation("🧠 [ENHANCED-BRAIN] Strategy enhanced: {Original} → {Enhanced} (ensemble confidence: {Confidence:P1})",
                originalStrategy, stratPred.SelectedStrategy, strategyPrediction.Confidence);
            return stratPred.SelectedStrategy;
        }
        
        return originalStrategy; // Keep original strategy
    }

    /// <summary>
    /// Enhance confidence by blending original with ensemble predictions
    /// </summary>
    private decimal EnhanceConfidence(decimal originalConfidence, decimal strategyConfidence, decimal priceConfidence)
    {
        // Weighted average: 50% original, 30% strategy ensemble, 20% price ensemble
        var enhancedConfidence = (originalConfidence * 0.5m) + 
                                (strategyConfidence * 0.3m) + 
                                (priceConfidence * 0.2m);
        
        return Math.Max(0.1m, Math.Min(1.0m, enhancedConfidence));
    }

    /// <summary>
    /// Enhance position sizing using CVaR and ensemble insights
    /// </summary>
    private decimal EnhancePositionSizing(decimal originalSize, EnsembleActionResult ensembleAction, decimal confidence, EnsemblePrediction pricePrediction)
    {
        var sizeMultiplier = 1.0m;
        
        // Adjust based on confidence
        if (confidence > 0.8m)
        {
            sizeMultiplier *= 1.2m; // Increase size for high confidence
        }
        else if (confidence < 0.5m)
        {
            sizeMultiplier *= 0.8m; // Decrease size for low confidence
        }
        
        // Adjust based on CVaR estimate (risk management)
        if (ensembleAction.CVaREstimate < -0.1)
        {
            sizeMultiplier *= 0.7m; // Reduce size for high risk
        }
        
        // Adjust based on price prediction strength
        if (pricePrediction.Result is PriceDirectionPrediction pricePred)
        {
            if (pricePred.Direction != "Sideways" && pricePred.Probability > 0.7)
            {
                sizeMultiplier *= 1.1m; // Slightly increase for strong directional bias
            }
        }
        
        var enhancedSize = originalSize * sizeMultiplier;
        
        // Safety bounds
        return Math.Max(0.1m, Math.Min(2.0m * originalSize, enhancedSize));
    }

    /// <summary>
    /// Enhance risk level based on CVaR and market conditions
    /// </summary>
    private decimal EnhanceRiskLevel(decimal originalRisk, double cvarEstimate, EnsemblePrediction pricePrediction)
    {
        var riskAdjustment = 0.0m; // Initialize the variable
        
        // Adjust based on CVaR
        if (cvarEstimate < -0.2)
        {
            riskAdjustment -= 0.1m; // Reduce risk for high CVaR
        }
        else if (cvarEstimate > 0.1)
        {
            riskAdjustment += 0.05m; // Slightly increase risk for positive CVaR
        }
        
        // Adjust based on price prediction uncertainty
        if (pricePrediction.Confidence < 0.6m)
        {
            riskAdjustment -= 0.05m; // Reduce risk for uncertain predictions
        }
        
        return Math.Max(0.1m, Math.Min(1.0m, originalRisk + riskAdjustment));
    }

    /// <summary>
    /// Calculate market timing signal
    /// </summary>
    private string CalculateMarketTiming(EnsemblePrediction pricePrediction, EnsembleActionResult ensembleAction, BotCore.Brain.MarketContext marketContext)
    {
        if (pricePrediction.Result is PriceDirectionPrediction pricePred)
        {
            // Strong directional signal with high action probability
            if (pricePred.Probability > 0.75 && ensembleAction.ActionProbability > 0.7)
            {
                return pricePred.Direction == "Up" ? "STRONG_BUY" : 
                       pricePred.Direction == "Down" ? "STRONG_SELL" : "HOLD";
            }
            
            // Moderate signal
            if (pricePred.Probability > 0.6)
            {
                return pricePred.Direction == "Up" ? "BUY" : 
                       pricePred.Direction == "Down" ? "SELL" : "NEUTRAL";
            }
        }
        
        return "NEUTRAL";
    }

    /// <summary>
    /// Generate human-readable enhancement reason
    /// </summary>
    private string GenerateEnhancementReason(
        BotCore.Brain.TradingDecision originalDecision,
        EnsemblePrediction strategyPred, 
        EnsemblePrediction pricePred, 
        EnsembleActionResult action)
    {
        var reasons = new List<string>();
        
        if (strategyPred.Confidence > 0.7m)
        {
            reasons.Add($"Strategy ensemble confidence: {strategyPred.Confidence:P0}");
        }
        
        if (pricePred.Confidence > 0.7m && pricePred.Result is PriceDirectionPrediction pricePredResult)
        {
            reasons.Add($"Price direction: {pricePredResult.Direction} ({pricePred.Confidence:P0})");
        }
        
        if (Math.Abs(action.CVaREstimate) > 0.1)
        {
            reasons.Add($"CVaR adjustment: {action.CVaREstimate:F2}");
        }
        
        return reasons.Any() ? string.Join(", ", reasons) : "Ensemble enhancement applied";
    }

    /// <summary>
    /// Submit trading outcome for feedback learning
    /// </summary>
    public void SubmitTradingOutcome(
        string symbol,
        string strategy,
        string action,
        decimal realizedPnL,
        Dictionary<string, object> context)
    {
        try
        {
            // Calculate prediction accuracy if we have tracked predictions
            var predictionKey = $"{symbol}_{strategy}_{DateTime.UtcNow:yyyyMMdd}";
            var accuracy = _predictionAccuracies.TryGetValue(predictionKey, out var acc) ? acc : 0.5;
            
            var outcome = new TradingOutcome
            {
                Timestamp = DateTime.UtcNow,
                Strategy = strategy,
                Action = action,
                Symbol = symbol,
                PredictionAccuracy = accuracy,
                RealizedPnL = realizedPnL,
                MarketConditions = JsonSerializer.Serialize(context)
            };
            outcome.ReplaceTradingContext(context);
            
            _feedbackService.SubmitTradingOutcome(outcome);
            
            _logger.LogDebug("🧠 [ENHANCED-BRAIN] Trading outcome submitted: {Strategy} {Action} P&L: {PnL:C2}", 
                strategy, action, realizedPnL);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🧠 [ENHANCED-BRAIN] Error submitting trading outcome");
        }
    }

    /// <summary>
    /// Submit prediction feedback for model improvement
    /// </summary>
    public void SubmitPredictionFeedback(
        string modelName,
        string symbol,
        string predictedAction,
        string actualOutcome,
        double accuracy,
        decimal pnlImpact)
    {
        try
        {
            var feedback = new PredictionFeedback
            {
                Timestamp = DateTime.UtcNow,
                ModelName = modelName,
                Symbol = symbol,
                PredictedAction = predictedAction,
                ActualOutcome = actualOutcome,
                ActualAccuracy = accuracy,
                ImpactOnPnL = pnlImpact
            };
            
            _feedbackService.SubmitPredictionFeedback(feedback);
            
            _logger.LogDebug("🧠 [ENHANCED-BRAIN] Prediction feedback submitted: {Model} accuracy: {Accuracy:P1}", 
                modelName, accuracy);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🧠 [ENHANCED-BRAIN] Error submitting prediction feedback");
        }
    }

    /// <summary>
    /// Initialize and load models from cloud
    /// </summary>
    public async Task InitializeAsync(CancellationToken cancellationToken = default)
    {
        try
        {
            _logger.LogInformation("🧠 [ENHANCED-BRAIN] Initializing enhanced trading brain integration");
            
            // Trigger initial cloud model synchronization
            await _cloudSync.SynchronizeModelsAsync(cancellationToken).ConfigureAwait(false);
            
            // Load default models into ensemble
            await LoadDefaultModels(cancellationToken).ConfigureAwait(false);
            
            _logger.LogInformation("🧠 [ENHANCED-BRAIN] Enhanced trading brain integration initialized successfully");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🧠 [ENHANCED-BRAIN] Error initializing enhanced trading brain");
            
            // Disable enhancement on initialization failure
            _isEnhancementActive = false;
            _logger.LogWarning("🧠 [ENHANCED-BRAIN] Enhancement disabled due to initialization failure");
        }
    }

    /// <summary>
    /// Load default models into ensemble service
    /// </summary>
    private async Task LoadDefaultModels(CancellationToken cancellationToken)
    {
        try
        {
            // Load CVaR-PPO from DI
            var cvarPPO = _serviceProvider.GetService<CVaRPPO>();
            if (cvarPPO != null)
            {
                await _ensembleService.LoadModelAsync("cvar_ppo_default", "", ModelSource.Local, 1.0, cancellationToken).ConfigureAwait(false);
                _logger.LogInformation("🧠 [ENHANCED-BRAIN] Loaded CVaR-PPO model into ensemble");
            }
            
            // Load other models from data directory
            var modelsPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data", "ml");
            if (Directory.Exists(modelsPath))
            {
                var onnxFiles = Directory.GetFiles(modelsPath, "*.onnx");
                foreach (var onnxFile in onnxFiles)
                {
                    var modelName = Path.GetFileNameWithoutExtension(onnxFile);
                    await _ensembleService.LoadModelAsync(modelName, onnxFile, ModelSource.Local, 0.8, cancellationToken).ConfigureAwait(false);
                    _logger.LogDebug("🧠 [ENHANCED-BRAIN] Loaded local model: {ModelName}", modelName);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🧠 [ENHANCED-BRAIN] Error loading default models");
        }
    }

    /// <summary>
    /// Track prediction for feedback analysis
    /// </summary>
    private void TrackPredictionForFeedback(EnhancedTradingDecision decision, string symbol, BotCore.Brain.MarketContext marketContext)
    {
        try
        {
            var predictionKey = $"{symbol}_{decision.EnhancedStrategy}_{DateTime.UtcNow:yyyyMMdd}";
            _lastPredictions[predictionKey] = DateTime.UtcNow;
            
            // Store prediction details for later accuracy calculation
            // This would be enhanced with actual prediction tracking logic
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "🧠 [ENHANCED-BRAIN] Error tracking prediction");
        }
    }

    /// <summary>
    /// Log decision enhancement details
    /// </summary>
    private void LogDecisionEnhancement(BotCore.Brain.TradingDecision original, EnhancedTradingDecision enhanced, string symbol)
    {
        _logger.LogInformation("🧠 [ENHANCED-BRAIN] Decision enhanced for {Symbol}: {OriginalStrategy} → {EnhancedStrategy} " +
                             "(confidence: {OriginalConf:P1} → {EnhancedConf:P1})",
            symbol,
            original.Strategy, // Use Strategy instead of SelectedStrategy
            enhanced.EnhancedStrategy,
            original.Confidence,
            enhanced.EnhancedConfidence);
    }

    #region Helper Methods

    private double[] ExtractContextVector(BotCore.Brain.MarketContext marketContext)
    {
        // Extract and normalize market context into feature vector
        return new double[] { 
            (double)marketContext.CurrentPrice / 5000.0, // Normalized price
            (double)marketContext.Volume / 100000.0, // Normalized volume
            (double)marketContext.Volatility, // Volatility
            marketContext.TimeOfDay.TotalHours / 24.0, // Time of day
            (double)marketContext.VolumeRatio // Volume ratio
        };
    }

    private double[] ExtractMarketFeatures(BotCore.Brain.MarketContext marketContext)
    {
        // Extract market features for price prediction
        return new double[] { 
            (double)marketContext.CurrentPrice, // Current price
            (double)marketContext.Volume, // Volume
            (double)marketContext.Volatility, // Volatility
            (double)marketContext.PriceChange, // Price change
            (double)marketContext.VolumeRatio, // Volume ratio
            marketContext.TimeOfDay.TotalHours // Time factor
        };
    }

    private double[] CreateStateVector(BotCore.Brain.TradingDecision decision, BotCore.Brain.MarketContext marketContext)
    {
        // Create state vector for CVaR-PPO using available properties
        return new double[] { 
            (double)decision.Confidence, // Decision confidence
            (double)marketContext.CurrentPrice, // Current price
            (double)marketContext.Volume, // Market volume
            (double)marketContext.Volatility, // Volatility
            // Use strategy as action encoding instead of Action property (which doesn't exist)
            decision.Strategy.Contains("S3", StringComparison.OrdinalIgnoreCase) ? 1.0 : 
            decision.Strategy.Contains("S6", StringComparison.OrdinalIgnoreCase) ? -1.0 : 0.0 // Strategy-based encoding
        };
    }

    /// <summary>
    /// Convert Dictionary marketContext to BotCore.Brain.MarketContext
    /// </summary>
    private BotCore.Brain.MarketContext ConvertToMarketContext(Dictionary<string, object> marketContext)
    {
        var brainContext = new BotCore.Brain.MarketContext();
        
        if (marketContext.TryGetValue("Symbol", out var symbol) && symbol is string symbolStr)
            brainContext.Symbol = symbolStr;
        
        if (marketContext.TryGetValue("CurrentPrice", out var price) && price is decimal priceDecimal)
            brainContext.CurrentPrice = priceDecimal;
        else if (price is double priceDouble)
            brainContext.CurrentPrice = (decimal)priceDouble;
        
        if (marketContext.TryGetValue("Volume", out var volume) && volume is decimal volumeDecimal)
            brainContext.Volume = volumeDecimal;
        else if (volume is double volumeDouble)
            brainContext.Volume = (decimal)volumeDouble;
        
        if (marketContext.TryGetValue("Volatility", out var volatility) && volatility is decimal volatilityDecimal)
            brainContext.Volatility = volatilityDecimal;
        else if (volatility is double volatilityDouble)
            brainContext.Volatility = (decimal)volatilityDouble;
        else
            brainContext.Volatility = 0.15m; // Default volatility
        
        if (marketContext.TryGetValue("TimeOfDay", out var timeOfDay) && timeOfDay is TimeSpan timeSpan)
            brainContext.TimeOfDay = timeSpan;
        else
            brainContext.TimeOfDay = DateTime.UtcNow.TimeOfDay;
        
        brainContext.DayOfWeek = DateTime.UtcNow.DayOfWeek;
        brainContext.VolumeRatio = 1.0m; // Default
        brainContext.PriceChange = 0.0m; // Default
        
        return brainContext;
    }

    private Env CreateSampleEnv()
    {
        return new Env
        {
            Symbol = "ES",
            atr = 5.0m + (decimal)(Random.Shared.NextDouble() * 2), // ATR around 5-7
            volz = 0.15m + (decimal)(Random.Shared.NextDouble() * 0.1) // Volume Z-score
        };
    }

    private Levels CreateSampleLevels()
    {
        var basePrice = 4500.0m;
        return new Levels
        {
            Support1 = basePrice - 10,
            Support2 = basePrice - 20,
            Support3 = basePrice - 30,
            Resistance1 = basePrice + 10,
            Resistance2 = basePrice + 20,
            Resistance3 = basePrice + 30,
            VWAP = basePrice,
            DailyPivot = basePrice,
            WeeklyPivot = basePrice + 5,
            MonthlyPivot = basePrice - 5
        };
    }

    private IList<Bar> CreateSampleBars()
    {
        var bars = new List<Bar>();
        var basePrice = 4500.0m;
        var currentTime = DateTime.UtcNow;
        
        for (int i = 0; i < 10; i++)
        {
            var variation = (decimal)(Random.Shared.NextDouble() - 0.5) * 5;
            var openPrice = basePrice + variation;
            var closePrice = openPrice + (decimal)(Random.Shared.NextDouble() - 0.5) * 2;
            
            bars.Add(new Bar
            {
                Symbol = "ES",
                Start = currentTime.AddMinutes(-i),
                Ts = ((DateTimeOffset)currentTime.AddMinutes(-i)).ToUnixTimeMilliseconds(),
                Open = openPrice,
                High = Math.Max(openPrice, closePrice) + (decimal)Random.Shared.NextDouble(),
                Low = Math.Min(openPrice, closePrice) - (decimal)Random.Shared.NextDouble(),
                Close = closePrice,
                Volume = 100 + Random.Shared.Next(200)
            });
        }
        
        return bars;
    }

    private RiskEngine CreateSampleRisk()
    {
        var riskEngine = new RiskEngine();
        riskEngine.cfg.risk_per_trade = 100m; // $100 risk per trade
        riskEngine.cfg.max_daily_drawdown = 1000m;
        riskEngine.cfg.max_open_positions = 1;
        return riskEngine;
    }

    private BotCore.Brain.TradingDecision ConvertBrainToTradingDecision(object brainDecision)
    {
        // Convert UnifiedTradingBrain decision format to BotCore TradingDecision
        if (brainDecision == null)
        {
            return new BotCore.Brain.TradingDecision
            {
                Symbol = "UNKNOWN",
                Strategy = "none",
                Confidence = 0.0m,
                Timestamp = DateTime.UtcNow
            };
        }

        // If it's a BrainDecision, convert it to BotCore.Brain.TradingDecision
        if (brainDecision is BrainDecision brain)
        {
            return new BotCore.Brain.TradingDecision
            {
                Symbol = brain.Symbol,
                Strategy = brain.RecommendedStrategy,
                Confidence = brain.StrategyConfidence,
                Timestamp = brain.DecisionTime
            };
        }

        // If it's already a TradingDecision, return as-is
        if (brainDecision is BotCore.Brain.TradingDecision decision)
        {
            return decision;
        }

        // Fallback for unknown types
        return new BotCore.Brain.TradingDecision
        {
            Symbol = "FALLBACK",
            Strategy = "unknown",
            Confidence = 0.0m,
            Timestamp = DateTime.UtcNow
        };
    }

    private BotCore.Brain.TradingDecision CreateFallbackTradingDecision()
    {
        return new BotCore.Brain.TradingDecision
        {
            Symbol = "FALLBACK",
            Strategy = "fallback",
            Confidence = 0.0m,
            Timestamp = DateTime.UtcNow
        };
    }

    #endregion
}

#region Data Models

public class EnhancedTradingDecision
{
    public BotCore.Brain.TradingDecision OriginalDecision { get; set; } = null!;
    public string EnhancedStrategy { get; set; } = string.Empty;
    public decimal EnhancedConfidence { get; set; }
    public decimal EnhancedPositionSize { get; set; }
    public decimal EnhancedRiskLevel { get; set; }
    public string MarketTimingSignal { get; set; } = string.Empty;
    public bool EnhancementApplied { get; set; }
    public string EnhancementReason { get; set; } = string.Empty;
    public string? ErrorMessage { get; set; }
    
    // Ensemble prediction details
    public EnsemblePrediction? StrategyPrediction { get; set; }
    public EnsemblePrediction? PricePrediction { get; set; }
    public EnsembleActionResult? EnsembleAction { get; set; }
    
    public DateTime Timestamp { get; set; }
}

#endregion