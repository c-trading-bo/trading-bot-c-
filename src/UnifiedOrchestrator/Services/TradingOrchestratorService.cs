using Microsoft.Extensions.Logging;
using Microsoft.AspNetCore.SignalR.Client;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;
using BotCore;
using BotCore.Models;
using BotCore.Risk;
using BotCore.Strategy;
using BotCore.Brain;
using BotCore.ML;
using System.Text.Json;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Unified trading orchestrator that consolidates all TopstepX trading functionality
/// Replaces multiple trading orchestrators with one unified system
/// </summary>
public class TradingOrchestratorService : ITradingOrchestrator, IDisposable
{
    private readonly ILogger<TradingOrchestratorService> _logger;
    private readonly HttpClient _httpClient;
    private readonly TopstepAuthAgent _authAgent;
    private readonly ICentralMessageBus _messageBus;
    private readonly UnifiedTradingBrain _tradingBrain; // 🧠 THE AI BRAIN
    private readonly UCBManager? _ucbManager; // 🎯 OPTIONAL UCB SERVICE
    
    // TopstepX Connections
    private HubConnection? _userHub;
    private HubConnection? _marketHub;
    private string? _jwtToken;
    private long _accountId;
    private bool _isConnected = false;
    private bool _isDemo = false; // Flag for demo/fallback mode

    // Trading Components (unified from all orchestrators)
    private readonly RiskEngine _riskEngine;
    private readonly Dictionary<string, IStrategy> _strategies = new();
    private readonly Dictionary<string, string> _contractIds = new(); // symbol -> contractId mapping
    
    // Supported actions
    public IReadOnlyList<string> SupportedActions { get; } = new[]
    {
        "analyzeESNQ", "checkSignals", "executeTrades",
        "calculateRisk", "checkThresholds", "adjustPositions",
        "analyzeOrderFlow", "readTape", "trackMMs",
        "scanOptionsFlow", "detectDarkPools", "trackSmartMoney"
    };

    public TradingOrchestratorService(
        ILogger<TradingOrchestratorService> logger,
        HttpClient httpClient,
        TopstepAuthAgent authAgent,
        ICentralMessageBus messageBus,
        UnifiedTradingBrain tradingBrain,
        UCBManager? ucbManager = null) // Optional UCB service
    {
        _logger = logger;
        _httpClient = httpClient;
        _authAgent = authAgent;
        _messageBus = messageBus;
        _tradingBrain = tradingBrain;
        _ucbManager = ucbManager;
        _riskEngine = new RiskEngine();
        
        // Set TopstepX base URL
        _httpClient.BaseAddress ??= new Uri("https://api.topstepx.com");
        
        // Initialize the AI brain
        _ = Task.Run(async () => await _tradingBrain.InitializeAsync());
        
        var ucbStatus = _ucbManager != null ? "with UCB service" : "without UCB service";
        _logger.LogInformation("🧠 [ORCHESTRATOR] Unified Trading Brain integrated {UcbStatus} and initializing...", ucbStatus);
    }

    #region ITradingOrchestrator Implementation

    public async Task ConnectAsync(CancellationToken cancellationToken = default)
    {
        if (_isConnected) return;

        // Check for paper trading mode
        var paperMode = Environment.GetEnvironmentVariable("PAPER_MODE") == "1" || 
                       Environment.GetEnvironmentVariable("AUTO_PAPER_TRADING") == "1";
        var tradingMode = Environment.GetEnvironmentVariable("TRADING_MODE") ?? "DEMO";

        if (paperMode)
        {
            _logger.LogInformation("🎯 Connecting to TopstepX in PAPER TRADING mode...");
            _logger.LogInformation("📋 Trading Mode: {TradingMode}", tradingMode);
            _logger.LogInformation("💰 Risk Level: SIMULATION ONLY - No real money involved");
        }
        else
        {
            _logger.LogInformation("🔌 Connecting to TopstepX API and hubs...");
        }

        try
        {
            // Get authentication (simulate in paper mode)
            await AuthenticateAsync(cancellationToken);
            
            if (_isDemo)
            {
                _logger.LogWarning("🎭 Running in DEMO MODE - TopstepX authentication unavailable");
                _logger.LogInformation("✅ Demo mode initialized - Simulated trading only");
                _isConnected = true;
                return;
            }
            
            // Connect to SignalR hubs (simulate in paper mode)
            await ConnectToHubsAsync(cancellationToken);
            
            // Initialize contract mappings
            await InitializeContractsAsync(cancellationToken);
            
            _isConnected = true;
            
            if (paperMode)
            {
                _logger.LogInformation("✅ Successfully connected to TopstepX - PAPER TRADING MODE ACTIVE");
                _logger.LogInformation("🎭 All trades will be simulated - No real money at risk");
            }
            else
            {
                _logger.LogInformation("✅ Successfully connected to TopstepX");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Failed to connect to TopstepX");
            throw;
        }
    }

    public async Task DisconnectAsync()
    {
        if (!_isConnected) return;

        _logger.LogInformation("🔌 Disconnecting from TopstepX...");

        try
        {
            if (_userHub != null)
            {
                await _userHub.DisposeAsync();
                _userHub = null;
            }

            if (_marketHub != null)
            {
                await _marketHub.DisposeAsync();
                _marketHub = null;
            }

            _isConnected = false;
            _logger.LogInformation("✅ Disconnected from TopstepX");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "⚠️ Error during disconnect");
        }
    }

    /// <summary>
    /// Start trading day - resets brain's daily P&L tracking for TopStep compliance
    /// Call this at market open or when starting a new trading session
    /// </summary>
    public async Task StartTradingDayAsync(CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🌅 [TRADING-DAY] Starting new trading day - resetting brain daily stats");
        
        // Reset brain's daily P&L tracking for TopStep compliance
        _tradingBrain.ResetDaily();
        
        // Reset UCB service if available
        if (_ucbManager != null)
        {
            try
            {
                await _ucbManager.ResetDailyAsync(cancellationToken);
                _logger.LogInformation("🎯 [UCB] Daily reset completed");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [UCB] Failed to reset daily stats - continuing without UCB");
            }
        }
        
        // Ensure brain is initialized
        if (!_tradingBrain.IsInitialized)
        {
            await _tradingBrain.InitializeAsync(cancellationToken);
        }
        
        _logger.LogInformation("✅ [TRADING-DAY] Brain reset complete - ready for TopStep compliant trading");
    }

    public async Task ExecuteESNQTradingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        if (!_isConnected)
        {
            throw new InvalidOperationException("Not connected to TopstepX");
        }

        _logger.LogInformation("📊 Executing ES/NQ trading analysis with cloud intelligence...");

        try
        {
            // Get current market data for ES and NQ
            var esData = await GetMarketDataAsync("ES", cancellationToken);
            var nqData = await GetMarketDataAsync("NQ", cancellationToken);

            context.Logs.Add($"ES Price: {esData?.LastPrice}, NQ Price: {nqData?.LastPrice}");

            // 🌐 GET CLOUD INTELLIGENCE - This is where the 27 GitHub workflows influence trading!
            var esCloudRecommendation = _messageBus.GetSharedState<CloudTradingRecommendation>("cloud.trading_recommendation.ES");
            var nqCloudRecommendation = _messageBus.GetSharedState<CloudTradingRecommendation>("cloud.trading_recommendation.NQ");
            
            if (esCloudRecommendation != null)
            {
                _logger.LogInformation("🧠 ES Cloud Intelligence: {Signal} (confidence: {Confidence:P1}) - {Reasoning}", 
                    esCloudRecommendation.Signal, esCloudRecommendation.Confidence, esCloudRecommendation.Reasoning);
                context.Logs.Add($"ES Cloud Signal: {esCloudRecommendation.Signal} ({esCloudRecommendation.Confidence:P1})");
            }
            
            if (nqCloudRecommendation != null)
            {
                _logger.LogInformation("🧠 NQ Cloud Intelligence: {Signal} (confidence: {Confidence:P1}) - {Reasoning}", 
                    nqCloudRecommendation.Signal, nqCloudRecommendation.Confidence, nqCloudRecommendation.Reasoning);
                context.Logs.Add($"NQ Cloud Signal: {nqCloudRecommendation.Signal} ({nqCloudRecommendation.Confidence:P1})");
            }

            // 🧠 NEW: USE UNIFIED TRADING BRAIN FOR INTELLIGENT DECISIONS
            try
            {
                // Get market data for brain analysis
                var esEnv = new Env { atr = 12.5m, volz = 1.2m }; // Sample data - replace with real market data
                var nqEnv = new Env { atr = 15.0m, volz = 1.1m };
                var levels = new Levels(); // Empty levels since it has no properties
                var sampleBars = new List<Bar>(); // Replace with real bar data

                // Let the AI brain make intelligent decisions for ES
                var esBrainDecision = await _tradingBrain.MakeIntelligentDecisionAsync(
                    "ES", esEnv, levels, sampleBars, _riskEngine, cancellationToken);
                
                // Let the AI brain make intelligent decisions for NQ  
                var nqBrainDecision = await _tradingBrain.MakeIntelligentDecisionAsync(
                    "NQ", nqEnv, levels, sampleBars, _riskEngine, cancellationToken);

                // 🎯 UCB INTEGRATION: Get multi-armed bandit recommendations if UCB service is available
                string? ucbStrategy = null;
                
                if (_ucbManager != null)
                {
                    try
                    {
                        // Create MarketData for UCB recommendation with sample/default data
                        // In a real implementation, this would come from actual market data feeds
                        var marketData = new BotCore.ML.MarketData
                        {
                            ESPrice = 4800m,      // Default ES price - should come from real data
                            NQPrice = 16000m,     // Default NQ price - should come from real data
                            ESVolume = 100000,    // Sample volume
                            NQVolume = 50000,     // Sample volume
                            ES_ATR = esEnv.atr ?? 20m,    // Use ATR from environment if available
                            NQ_ATR = nqEnv.atr ?? 80m,    // Use ATR from environment if available
                            VIX = 20m,            // Default VIX
                            TICK = 0,             // Neutral TICK
                            ADD = 0,              // Neutral ADD
                            Correlation = 0.8m,   // Default ES/NQ correlation
                            RSI_ES = 50m,         // Neutral RSI
                            RSI_NQ = 50m,         // Neutral RSI
                            PrimaryInstrument = "ES"
                        };
                        
                        var ucbResult = await _ucbManager.GetRecommendationAsync(marketData, cancellationToken);
                        ucbStrategy = ucbResult?.Strategy;
                        
                        _logger.LogInformation("🎯 [UCB] Multi-armed bandit recommendation: {Strategy} (confidence: {Confidence:F2}, trade: {ShouldTrade})", 
                            ucbStrategy ?? "None", ucbResult?.Confidence ?? 0.0m, ucbResult?.Trade ?? false);
                    }
                    catch (Exception ucbEx)
                    {
                        _logger.LogWarning(ucbEx, "⚠️ [UCB] Failed to get recommendations - using Brain only");
                    }
                }

                _logger.LogInformation("🧠 [AI-DECISIONS] ES: {Strategy} ({Confidence:P1}), NQ: {NQStrategy} ({NQConfidence:P1}) | UCB: {UCBStrategy} | Active Strategies: S2,S3,S6,S11",
                    esBrainDecision.RecommendedStrategy, esBrainDecision.ModelConfidence,
                    nqBrainDecision.RecommendedStrategy, nqBrainDecision.ModelConfidence,
                    ucbStrategy ?? "None");

                // 🎯 ENHANCED: Process AI-enhanced candidates with UCB filtering
                foreach (var candidate in esBrainDecision.EnhancedCandidates)
                {
                    // Apply UCB strategy filtering if available
                    bool shouldProcessCandidate = true;
                    if (!string.IsNullOrEmpty(ucbStrategy))
                    {
                        // Only process candidates that match UCB recommendation or are high confidence
                        shouldProcessCandidate = candidate.strategy_id.Contains(ucbStrategy, StringComparison.OrdinalIgnoreCase) ||
                                               esBrainDecision.ModelConfidence > 0.8m;
                        
                        if (!shouldProcessCandidate)
                        {
                            _logger.LogInformation("🎯 [UCB-FILTER] Skipping ES candidate {Strategy} - UCB recommends {UCBStrategy}", 
                                candidate.strategy_id, ucbStrategy);
                            continue;
                        }
                    }

                    var aiSignal = ConvertCandidateToTradingSignal(candidate, "ES", esBrainDecision);
                    await ProcessAITradingSignalAsync(aiSignal, esBrainDecision, context, cancellationToken);
                    context.Logs.Add($"🧠 AI ES Signal: {candidate.strategy_id} {candidate.side} @ {candidate.entry:F2} (Confidence: {esBrainDecision.ModelConfidence:P1})");
                }

                foreach (var candidate in nqBrainDecision.EnhancedCandidates)
                {
                    // Apply UCB strategy filtering if available
                    bool shouldProcessCandidate = true;
                    if (!string.IsNullOrEmpty(ucbStrategy))
                    {
                        // Only process candidates that match UCB recommendation or are high confidence
                        shouldProcessCandidate = candidate.strategy_id.Contains(ucbStrategy, StringComparison.OrdinalIgnoreCase) ||
                                               nqBrainDecision.ModelConfidence > 0.8m;
                        
                        if (!shouldProcessCandidate)
                        {
                            _logger.LogInformation("🎯 [UCB-FILTER] Skipping NQ candidate {Strategy} - UCB recommends {UCBStrategy}", 
                                candidate.strategy_id, ucbStrategy);
                            continue;
                        }
                    }

                    var aiSignal = ConvertCandidateToTradingSignal(candidate, "NQ", nqBrainDecision);
                    await ProcessAITradingSignalAsync(aiSignal, nqBrainDecision, context, cancellationToken);
                    context.Logs.Add($"🧠 AI NQ Signal: {candidate.strategy_id} {candidate.side} @ {candidate.entry:F2} (Confidence: {nqBrainDecision.ModelConfidence:P1})");
                }
            }
            catch (Exception brainEx)
            {
                _logger.LogError(brainEx, "❌ AI Brain analysis failed, falling back to traditional methods");
                
                // FALLBACK: Use traditional strategy analysis if brain fails
                await ExecuteTraditionalTradingAsync(context, cancellationToken);
            }

            context.Logs.Add($"ES/NQ trading analysis completed with AI brain integration - {DateTime.UtcNow:HH:mm:ss}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error executing ES/NQ trading");
            throw;
        }
    }

    /// <summary>
    /// Fallback method using traditional strategy analysis
    /// </summary>
    private async Task ExecuteTraditionalTradingAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        var esData = new MarketData { Symbol = "ES" };
        var nqData = new MarketData { Symbol = "NQ" };
        
        // Get cloud intelligence
        var esCloudRecommendation = _messageBus.GetSharedState<CloudTradingRecommendation>("cloud.trading_recommendation.ES");
        var nqCloudRecommendation = _messageBus.GetSharedState<CloudTradingRecommendation>("cloud.trading_recommendation.NQ");

        // Run strategy analysis with cloud intelligence integration
        foreach (var strategy in _strategies.Values)
        {
            if (strategy is IESNQStrategy esNqStrategy)
            {
                var signals = await esNqStrategy.AnalyzeAsync(esData, nqData, cancellationToken);
                
                foreach (var originalSignal in signals)
                {
                    // 🎯 ENHANCE SIGNAL WITH CLOUD INTELLIGENCE
                    var enhancedSignal = EnhanceSignalWithCloudIntelligence(originalSignal, esCloudRecommendation, nqCloudRecommendation);
                    
                    await ProcessTradingSignalAsync(enhancedSignal, context, cancellationToken);
                }
            }
        }
    }

    /// <summary>
    /// Convert AI candidate to trading signal format
    /// </summary>
    private TradingSignal ConvertCandidateToTradingSignal(Candidate candidate, string symbol, BrainDecision brainDecision)
    {
        return new TradingSignal
        {
            Symbol = symbol,
            Direction = candidate.side.ToString(),
            Price = candidate.entry,
            PositionSize = candidate.qty,
            Confidence = (double)brainDecision.ModelConfidence,
            Timestamp = DateTime.UtcNow,
            Reasoning = $"AI Enhanced - {brainDecision.RecommendedStrategy} (R: {candidate.expR:F2})"
        };
    }

    /// <summary>
    /// Process AI-enhanced trading signal with brain feedback
    /// </summary>
    private async Task ProcessAITradingSignalAsync(TradingSignal signal, BrainDecision brainDecision, 
        WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        try
        {
            _logger.LogInformation("🧠 Processing AI signal: {Direction} {Symbol} @ {Price:F2}", 
                signal.Direction, signal.Symbol, signal.Price);

            // Apply AI-enhanced risk management
            var enhancedSignal = ApplyAIRiskManagement(signal, brainDecision);
            
            // Route through standard signal processing
            await ProcessTradingSignalAsync(enhancedSignal, context, cancellationToken);
            
            // 🎯 CRITICAL: Feed execution result back to brain for learning
            var executionResult = new ExecutionResult
            {
                SignalId = signal.Reasoning, // Use reasoning as signal ID
                Symbol = enhancedSignal.Symbol,
                ExecutedPrice = enhancedSignal.Price, // This should come from actual execution
                Success = true, // This should reflect actual execution success
                Timestamp = DateTime.UtcNow,
                PnL = 0m // This will be updated when position closes
            };
            
            await _tradingBrain.LearnFromResultAsync(
                enhancedSignal.Symbol, 
                signal.Reasoning, 
                executionResult.PnL, 
                executionResult.Success, 
                TimeSpan.Zero, 
                cancellationToken);

            // 💰 TOPSTEP COMPLIANCE: Update daily P&L tracking in brain
            if (executionResult.Success && executionResult.PnL != 0m)
            {
                _tradingBrain.UpdatePnL(signal.Reasoning, executionResult.PnL);
                _logger.LogInformation("💰 [BRAIN-PNL] Updated strategy {Strategy} with P&L: {PnL:C}", 
                    signal.Reasoning, executionResult.PnL);

                // Also update UCB service if available
                if (_ucbManager != null)
                {
                    try
                    {
                        await _ucbManager.UpdatePnLAsync(signal.Reasoning, executionResult.PnL);
                        _logger.LogInformation("🎯 [UCB-PNL] Updated strategy {Strategy} with P&L: {PnL:C}", 
                            signal.Reasoning, executionResult.PnL);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "⚠️ [UCB] Failed to update P&L - continuing");
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error processing AI trading signal: {Symbol}", signal.Symbol);
            
            // Feed failure back to brain for learning
            var failureResult = new ExecutionResult
            {
                SignalId = signal.Reasoning,
                Symbol = signal.Symbol,
                Success = false,
                ErrorMessage = ex.Message,
                Timestamp = DateTime.UtcNow
            };
            
            await _tradingBrain.LearnFromResultAsync(
                signal.Symbol, 
                signal.Reasoning, 
                0m, // No PnL for failed execution
                false, // Failed execution
                TimeSpan.Zero, 
                cancellationToken);
        }
    }

    /// <summary>
    /// Apply AI-enhanced risk management to signal
    /// </summary>
    private TradingSignal ApplyAIRiskManagement(TradingSignal signal, BrainDecision brainDecision)
    {
        // Use brain's recommended position multiplier instead of default
        if (brainDecision.OptimalPositionMultiplier > 0)
        {
            signal.PositionSize = signal.PositionSize * brainDecision.OptimalPositionMultiplier;
            _logger.LogInformation("🧠 AI adjusted position size: {Size} contracts (multiplier: {Mult:F2})", 
                signal.PositionSize, brainDecision.OptimalPositionMultiplier);
        }

        // Apply AI confidence-based adjustments
        if (brainDecision.ModelConfidence < 0.6m)
        {
            signal.PositionSize = Math.Max(1, signal.PositionSize / 2); // Reduce size for low confidence
            _logger.LogInformation("🧠 Low confidence - reduced position size to {Size}", signal.PositionSize);
        }

        return signal;
    }

    public async Task ManagePortfolioRiskAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("⚖️ Managing portfolio risk and heat...");

        try
        {
            // Get current positions
            var positions = await GetPositionsAsync(cancellationToken);
            
            // Calculate portfolio metrics
            var totalPnL = positions.Sum(p => p.UnrealizedPnL + p.RealizedPnL);
            var totalHeat = positions.Sum(p => Math.Abs(p.Quantity * p.AveragePrice * 0.01m)); // 1% risk per position
            
            context.Logs.Add($"Portfolio PnL: {totalPnL:C}, Heat: {totalHeat:C}");

            // Check risk thresholds
            var maxDailyLoss = -850m; // TopstepX eval account limit
            var maxHeat = 5000m; // Maximum portfolio heat
            
            if (totalPnL < maxDailyLoss)
            {
                _logger.LogWarning("🔥 Daily loss limit approaching: {PnL}", totalPnL);
                await FlattenAllPositionsAsync("Daily loss limit", cancellationToken);
                context.Logs.Add("Flattened all positions due to daily loss limit");
            }
            
            if (totalHeat > maxHeat)
            {
                _logger.LogWarning("🔥 Portfolio heat too high: {Heat}", totalHeat);
                await ReducePositionsAsync(0.5m, "Portfolio heat reduction", cancellationToken);
                context.Logs.Add("Reduced positions due to high portfolio heat");
            }

            // Update risk metrics
            context.Parameters["totalPnL"] = totalPnL;
            context.Parameters["totalHeat"] = totalHeat;
            
            _logger.LogInformation("✅ Portfolio risk management completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in portfolio risk management");
            throw;
        }
    }

    public async Task AnalyzeMicrostructureAsync(WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        _logger.LogInformation("🔬 Analyzing market microstructure...");

        try
        {
            // Get order book data
            var esOrderBook = await GetOrderBookAsync("ES", cancellationToken);
            var nqOrderBook = await GetOrderBookAsync("NQ", cancellationToken);

            // Analyze order flow
            var esFlow = AnalyzeOrderFlow(esOrderBook);
            var nqFlow = AnalyzeOrderFlow(nqOrderBook);

            context.Logs.Add($"ES Order Flow: {esFlow.Direction} ({esFlow.Strength})");
            context.Logs.Add($"NQ Order Flow: {nqFlow.Direction} ({nqFlow.Strength})");

            // Detect market maker activity
            var esMMActivity = DetectMarketMakerActivity(esOrderBook);
            var nqMMActivity = DetectMarketMakerActivity(nqOrderBook);

            context.Parameters["ES_OrderFlow"] = esFlow;
            context.Parameters["NQ_OrderFlow"] = nqFlow;
            context.Parameters["ES_MMActivity"] = esMMActivity;
            context.Parameters["NQ_MMActivity"] = nqMMActivity;

            _logger.LogInformation("✅ Microstructure analysis completed");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "❌ Error in microstructure analysis");
            throw;
        }
    }

    #endregion

    #region IWorkflowActionExecutor Implementation

    public async Task<WorkflowExecutionResult> ExecuteActionAsync(string action, WorkflowExecutionContext context, CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        
        try
        {
            switch (action)
            {
                case "analyzeESNQ":
                case "checkSignals":
                case "executeTrades":
                    await ExecuteESNQTradingAsync(context, cancellationToken);
                    break;
                    
                case "calculateRisk":
                case "checkThresholds":
                case "adjustPositions":
                    await ManagePortfolioRiskAsync(context, cancellationToken);
                    break;
                    
                case "analyzeOrderFlow":
                case "readTape":
                case "trackMMs":
                    await AnalyzeMicrostructureAsync(context, cancellationToken);
                    break;
                    
                case "scanOptionsFlow":
                case "detectDarkPools":
                case "trackSmartMoney":
                    await AnalyzeOptionsFlowAsync(context, cancellationToken);
                    break;
                    
                default:
                    throw new NotSupportedException($"Action '{action}' is not supported by TradingOrchestrator");
            }

            return new WorkflowExecutionResult
            {
                Success = true,
                Duration = DateTime.UtcNow - startTime
            };
        }
        catch (Exception ex)
        {
            return new WorkflowExecutionResult
            {
                Success = false,
                ErrorMessage = ex.Message,
                Duration = DateTime.UtcNow - startTime
            };
        }
    }

    public bool CanExecute(string action)
    {
        return SupportedActions.Contains(action);
    }

    #endregion

    #region Private Methods

    private async Task AuthenticateAsync(CancellationToken cancellationToken)
    {
        var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
        var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
        _jwtToken = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");

        _logger.LogInformation("🔍 Auth Debug - Username: {Username}", username ?? "NULL");
        _logger.LogInformation("🔍 Auth Debug - API Key: {ApiKey}", apiKey?.Substring(0, Math.Min(10, apiKey.Length)) + "..." ?? "NULL");
        _logger.LogInformation("🔍 Auth Debug - JWT Token: {JwtToken}", _jwtToken?.Substring(0, Math.Min(10, _jwtToken.Length)) + "..." ?? "NULL");

        if (string.IsNullOrEmpty(_jwtToken) && (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey)))
        {
            _logger.LogInformation("🔐 Attempting to get JWT token using username/API key...");
            try
            {
                _jwtToken = await _authAgent.GetJwtAsync(username, apiKey, cancellationToken);
                _logger.LogInformation("🔐 JWT token received: {JwtToken}", _jwtToken?.Substring(0, Math.Min(10, _jwtToken.Length)) + "..." ?? "NULL");
            }
            catch (Exception ex)
            {
                _logger.LogError("❌ TopstepX authentication failed: {Error}", ex.Message);
                _logger.LogWarning("⚠️  This could be due to:");
                _logger.LogWarning("   • Expired API key - Check your TopstepX account to regenerate");
                _logger.LogWarning("   • Invalid credentials - Verify username and API key");
                _logger.LogWarning("   • Account not authorized for API access");
                _logger.LogWarning("   • TopstepX API service temporarily unavailable");
                _logger.LogInformation("🔄 For now, the bot will continue with limited functionality");
                
                // Set a flag to indicate we're in fallback mode
                _isDemo = true;
                return;
            }
        }

        if (string.IsNullOrEmpty(_jwtToken))
        {
            _logger.LogWarning("⚠️  No TopstepX authentication available - running in demo mode");
            _isDemo = true;
            return;
        }

        _httpClient.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _jwtToken);
        
        // Get account ID
        _accountId = await GetAccountIdAsync(cancellationToken);
        
        _logger.LogInformation("✅ TopstepX authentication successful for account {AccountId}", _accountId);
    }

    private async Task ConnectToHubsAsync(CancellationToken cancellationToken)
    {
        // Connect to User Hub
        _userHub = new HubConnectionBuilder()
            .WithUrl("https://rtc.topstepx.com/hubs/user", options =>
            {
                options.AccessTokenProvider = () => Task.FromResult(_jwtToken);
            })
            .Build();

        await _userHub.StartAsync(cancellationToken);
        await _userHub.InvokeAsync("SubscribeOrders", _accountId, cancellationToken);
        await _userHub.InvokeAsync("SubscribeTrades", _accountId, cancellationToken);

        // Connect to Market Hub
        _marketHub = new HubConnectionBuilder()
            .WithUrl("https://rtc.topstepx.com/hubs/market", options =>
            {
                options.AccessTokenProvider = () => Task.FromResult(_jwtToken);
            })
            .Build();

        await _marketHub.StartAsync(cancellationToken);

        _logger.LogInformation("✅ Connected to TopstepX SignalR hubs");
    }

    private async Task InitializeContractsAsync(CancellationToken cancellationToken)
    {
        // Get ES and NQ contract IDs
        var esContract = await GetContractIdAsync("ES", cancellationToken);
        var nqContract = await GetContractIdAsync("NQ", cancellationToken);

        if (!string.IsNullOrEmpty(esContract))
            _contractIds["ES"] = esContract;
        
        if (!string.IsNullOrEmpty(nqContract))
            _contractIds["NQ"] = nqContract;

        _logger.LogInformation("✅ Contract mappings initialized: ES={EsContract}, NQ={NqContract}", esContract, nqContract);
    }

    private async Task<long> GetAccountIdAsync(CancellationToken cancellationToken)
    {
        var response = await _httpClient.GetAsync("/api/Account", cancellationToken);
        response.EnsureSuccessStatusCode();
        
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        using var doc = JsonDocument.Parse(json);
        
        return doc.RootElement.GetProperty("accountId").GetInt64();
    }

    private async Task<string?> GetContractIdAsync(string symbol, CancellationToken cancellationToken)
    {
        var response = await _httpClient.GetAsync($"/api/Contract/available?symbol={symbol}&live=false", cancellationToken);
        if (!response.IsSuccessStatusCode) return null;
        
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        using var doc = JsonDocument.Parse(json);
        
        if (doc.RootElement.ValueKind == JsonValueKind.Array && doc.RootElement.GetArrayLength() > 0)
        {
            return doc.RootElement[0].GetProperty("contractId").GetString();
        }
        
        return null;
    }

    private async Task<MarketData?> GetMarketDataAsync(string symbol, CancellationToken cancellationToken)
    {
        if (!_contractIds.TryGetValue(symbol, out var contractId))
            return null;

        // Subscribe to market data if not already subscribed
        if (_marketHub != null)
        {
            await _marketHub.InvokeAsync("Subscribe", contractId, cancellationToken);
        }

        // For now, return a placeholder - in real implementation this would come from the market hub
        return new MarketData
        {
            Symbol = symbol,
            LastPrice = 5000m, // Placeholder
            BidPrice = 4999.75m,
            AskPrice = 5000.25m,
            Timestamp = DateTime.UtcNow
        };
    }

    private async Task<List<Position>> GetPositionsAsync(CancellationToken cancellationToken)
    {
        var response = await _httpClient.GetAsync($"/api/Position/{_accountId}", cancellationToken);
        if (!response.IsSuccessStatusCode) return new List<Position>();
        
        var json = await response.Content.ReadAsStringAsync(cancellationToken);
        // Parse positions from JSON - implementation depends on TopstepX API structure
        
        return new List<Position>(); // Placeholder
    }

    private async Task ProcessTradingSignalAsync(TradingSignal signal, WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        context.Logs.Add($"Processing signal: {signal.Symbol} {signal.Direction} @ {signal.Price}");
        
        // Risk check using available RiskEngine methods
        var risk = RiskEngine.ComputeRisk(signal.Price, signal.Price * 0.98m, signal.Price * 1.02m, signal.Direction == "BUY");
        if (risk <= 0)
        {
            context.Logs.Add($"Signal rejected by risk engine: {signal.Symbol} - Invalid risk calculation");
            return;
        }

        // Place order (placeholder - would use actual order placement logic)
        context.Logs.Add($"Order placed for {signal.Symbol}");
    }

    private async Task FlattenAllPositionsAsync(string reason, CancellationToken cancellationToken)
    {
        _logger.LogWarning("🔥 Flattening all positions: {Reason}", reason);
        // Implementation would close all open positions
    }

    private async Task ReducePositionsAsync(decimal reductionFactor, string reason, CancellationToken cancellationToken)
    {
        _logger.LogWarning("📉 Reducing positions by {Factor}: {Reason}", reductionFactor, reason);
        // Implementation would reduce position sizes
    }

    private async Task<OrderBook?> GetOrderBookAsync(string symbol, CancellationToken cancellationToken)
    {
        // Placeholder - would get real order book data
        return new OrderBook { Symbol = symbol };
    }

    private OrderFlowAnalysis AnalyzeOrderFlow(OrderBook? orderBook)
    {
        // Placeholder order flow analysis
        return new OrderFlowAnalysis 
        { 
            Direction = "Bullish", 
            Strength = "Medium" 
        };
    }

    private MarketMakerActivity DetectMarketMakerActivity(OrderBook? orderBook)
    {
        // Placeholder market maker detection
        return new MarketMakerActivity 
        { 
            IsActive = true, 
            Side = "Both" 
        };
    }

    private async Task AnalyzeOptionsFlowAsync(WorkflowExecutionContext context, CancellationToken cancellationToken)
    {
        _logger.LogInformation("📈 Analyzing options flow for smart money detection...");
        
        // Placeholder implementation
        context.Logs.Add("Options flow analysis completed");
        context.Parameters["OptionsFlow"] = new { SmartMoney = "Bullish", Volume = "High" };
    }

    #endregion

    #region Supporting Classes

    public class MarketData
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal LastPrice { get; set; }
        public decimal BidPrice { get; set; }
        public decimal AskPrice { get; set; }
        public DateTime Timestamp { get; set; }
    }

    public class Position
    {
        public string Symbol { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public decimal AveragePrice { get; set; }
        public decimal UnrealizedPnL { get; set; }
        public decimal RealizedPnL { get; set; }
    }

    public class TradingSignal
    {
        public string Symbol { get; set; } = string.Empty;
        public string Direction { get; set; } = string.Empty;
        public string Side => Direction; // Alias for compatibility
        public decimal Price { get; set; }
        public DateTime Timestamp { get; set; }
        public decimal PositionSize { get; set; } = 1;
        public double Confidence { get; set; } = 0.5;
        public string Reasoning { get; set; } = string.Empty;
    }

    public class OrderBook
    {
        public string Symbol { get; set; } = string.Empty;
    }

    public class OrderFlowAnalysis
    {
        public string Direction { get; set; } = string.Empty;
        public string Strength { get; set; } = string.Empty;
    }

    public class MarketMakerActivity
    {
        public bool IsActive { get; set; }
        public string Side { get; set; } = string.Empty;
    }

    #endregion

    #region IDisposable

    public void Dispose()
    {
        DisconnectAsync().GetAwaiter().GetResult();
    }
    
    /// <summary>
    /// Enhance trading signal with cloud intelligence from 27 GitHub workflows
    /// This is where cloud data actually influences trading decisions!
    /// </summary>
    private TradingSignal EnhanceSignalWithCloudIntelligence(
        TradingSignal originalSignal, 
        CloudTradingRecommendation? esCloudRec, 
        CloudTradingRecommendation? nqCloudRec)
    {
        // Get relevant cloud recommendation based on symbol
        var cloudRec = originalSignal.Symbol == "ES" ? esCloudRec : nqCloudRec;
        
        if (cloudRec == null || cloudRec.Signal == "ERROR")
        {
            _logger.LogInformation("⚠️ No cloud intelligence available for {Symbol}", originalSignal.Symbol);
            return originalSignal; // Return original signal if no cloud data
        }

        var enhancedSignal = new TradingSignal
        {
            Symbol = originalSignal.Symbol,
            Direction = originalSignal.Direction,
            Price = originalSignal.Price,
            Timestamp = originalSignal.Timestamp,
            PositionSize = originalSignal.PositionSize,
            Confidence = originalSignal.Confidence,
            Reasoning = originalSignal.Reasoning
        };
        
        // 🧠 CLOUD INTELLIGENCE INTEGRATION
        
        // 1. Confidence Adjustment - Cloud adds or reduces confidence
        var cloudConfidenceMultiplier = cloudRec.Confidence;
        enhancedSignal.Confidence *= cloudConfidenceMultiplier;
        
        // 2. Signal Direction Validation - Cloud can override or confirm
        if (cloudRec.Signal == "BUY" && originalSignal.Side == "SELL")
        {
            _logger.LogWarning("🔄 Cloud intelligence conflicts: Original={OriginalSide}, Cloud={CloudSignal} - reducing confidence", 
                originalSignal.Side, cloudRec.Signal);
            enhancedSignal.Confidence *= 0.5; // Reduce confidence on conflict
        }
        else if (cloudRec.Signal == "SELL" && originalSignal.Side == "BUY")
        {
            _logger.LogWarning("🔄 Cloud intelligence conflicts: Original={OriginalSide}, Cloud={CloudSignal} - reducing confidence", 
                originalSignal.Side, cloudRec.Signal);
            enhancedSignal.Confidence *= 0.5; // Reduce confidence on conflict
        }
        else if (cloudRec.Signal == originalSignal.Side)
        {
            _logger.LogInformation("✅ Cloud intelligence confirms: {Side} - boosting confidence", originalSignal.Side);
            enhancedSignal.Confidence *= 1.2; // Boost confidence on confirmation
        }
        
        // 3. Position Size Adjustment - Cloud influences position sizing
        if (cloudRec.Confidence > 0.7)
        {
            enhancedSignal.PositionSize = Math.Min(enhancedSignal.PositionSize * 1.1m, 5); // Max 10% increase, cap at 5 contracts
            _logger.LogInformation("📈 High cloud confidence - increasing position size to {PositionSize}", enhancedSignal.PositionSize);
        }
        else if (cloudRec.Confidence < 0.3)
        {
            enhancedSignal.PositionSize = Math.Max(enhancedSignal.PositionSize * 0.8m, 1); // Max 20% decrease, min 1 contract
            _logger.LogInformation("📉 Low cloud confidence - reducing position size to {PositionSize}", enhancedSignal.PositionSize);
        }

        // 4. Add cloud reasoning to signal
        enhancedSignal.Reasoning += $" | Cloud: {cloudRec.Signal} ({cloudRec.Confidence:P1}) - {cloudRec.Reasoning}";
        
        _logger.LogInformation("🧠 Signal enhanced with cloud intelligence: {Symbol} {Side} - Original confidence: {OriginalConf:P1}, Enhanced: {EnhancedConf:P1}", 
            enhancedSignal.Symbol, enhancedSignal.Side, originalSignal.Confidence, enhancedSignal.Confidence);
            
        return enhancedSignal;
    }

    #endregion
}

// Interface for ES/NQ specific strategies
public interface IESNQStrategy : IStrategy
{
    Task<List<TradingOrchestratorService.TradingSignal>> AnalyzeAsync(
        TradingOrchestratorService.MarketData? esData, 
        TradingOrchestratorService.MarketData? nqData, 
        CancellationToken cancellationToken);
}

// Execution result for brain learning
public class ExecutionResult
{
    public string SignalId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public decimal ExecutedPrice { get; set; }
    public bool Success { get; set; }
    public DateTime Timestamp { get; set; }
    public decimal PnL { get; set; }
    public string? ErrorMessage { get; set; }
}