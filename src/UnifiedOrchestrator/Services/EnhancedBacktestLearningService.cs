using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using TradingBot.UnifiedOrchestrator.Interfaces;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Enhanced BacktestLearningService → UnifiedTradingBrain Integration
/// 
/// CRITICAL REQUIREMENT: Uses SAME UnifiedTradingBrain for historical replay as live trading
/// This ensures historical data pipeline uses identical intelligence as live trading:
/// - Same data formatting and feature engineering
/// - Same decision-making logic (UnifiedTradingBrain.MakeIntelligentDecisionAsync)
/// - Same risk management and position sizing
/// - Identical context and outputs for reproducible results
/// - Same scheduling: Market Open: Light learning every 60 min, Market Closed: Intensive every 15 min
/// </summary>
public class EnhancedBacktestLearningService : BackgroundService
{
    private readonly ILogger<EnhancedBacktestLearningService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IMarketHoursService _marketHours;
    
    // CRITICAL: Direct injection of UnifiedTradingBrain for identical intelligence
    private readonly BotCore.Brain.UnifiedTradingBrain _unifiedBrain;
    
    // Historical replay state
    private readonly Dictionary<string, HistoricalReplayContext> _replayContexts = new();
    private readonly List<BacktestResult> _recentBacktests = new();

    public EnhancedBacktestLearningService(
        ILogger<EnhancedBacktestLearningService> logger,
        IServiceProvider serviceProvider,
        IMarketHoursService marketHours,
        BotCore.Brain.UnifiedTradingBrain unifiedBrain)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _marketHours = marketHours;
        _unifiedBrain = unifiedBrain; // Same brain as live trading
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        _logger.LogInformation("[ENHANCED-BACKTEST] Starting enhanced backtest learning service with UnifiedTradingBrain");
        
        // Wait for system initialization
        await Task.Delay(TimeSpan.FromSeconds(30), stoppingToken);
        
        while (!stoppingToken.IsCancellationRequested)
        {
            try
            {
                // Use UnifiedTradingBrain's unified scheduling for identical timing
                var currentTime = DateTime.UtcNow;
                var schedulingRecommendation = _unifiedBrain.GetUnifiedSchedulingRecommendation(currentTime);
                
                _logger.LogInformation("[UNIFIED-SCHEDULING] {Reasoning} - Learning every {Minutes} minutes", 
                    schedulingRecommendation.Reasoning, schedulingRecommendation.HistoricalLearningIntervalMinutes);
                
                if (schedulingRecommendation.LearningIntensity == "INTENSIVE" || 
                    schedulingRecommendation.LearningIntensity == "LIGHT")
                {
                    await RunUnifiedBacktestLearningAsync(schedulingRecommendation, stoppingToken);
                }
                
                // Use the exact interval recommended by UnifiedTradingBrain
                var delayMinutes = schedulingRecommendation.HistoricalLearningIntervalMinutes;
                await Task.Delay(TimeSpan.FromMinutes(delayMinutes), stoppingToken);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[ENHANCED-BACKTEST] Error in backtest learning service");
                await Task.Delay(TimeSpan.FromMinutes(30), stoppingToken);
            }
        }
    }

    /// <summary>
    /// Run unified backtest learning session using same UnifiedTradingBrain as live trading
    /// Focuses on 4 primary strategies: S2 (Mean Reversion), S3 (Compression), S6 (Momentum), S11 (Exhaustion)
    /// </summary>
    private async Task RunUnifiedBacktestLearningAsync(
        BotCore.Brain.UnifiedSchedulingRecommendation scheduling, 
        CancellationToken cancellationToken)
    {
        _logger.LogInformation("[UNIFIED-BACKTEST] Starting unified backtest learning session with {Intensity} intensity on strategies: {Strategies}", 
            scheduling.LearningIntensity, string.Join(",", scheduling.RecommendedStrategies));
        
        try
        {
            // Generate backtest configurations for recommended strategies
            var backtestConfigs = GenerateUnifiedBacktestConfigs(scheduling);
            
            var parallelJobs = scheduling.LearningIntensity switch
            {
                "INTENSIVE" => 4, // Process all 4 strategies in parallel
                "LIGHT" => 2,     // Process 2 strategies in parallel during market hours
                _ => 1            // Single strategy processing
            };
            
            var semaphore = new SemaphoreSlim(parallelJobs, parallelJobs);
            
            var tasks = backtestConfigs.Select(async config =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    return await RunUnifiedHistoricalBacktestAsync(config, cancellationToken);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var results = await Task.WhenAll(tasks);
            
            // Feed results back to UnifiedTradingBrain for continuous learning
            await FeedResultsToUnifiedBrainAsync(results, cancellationToken);
            
            _logger.LogInformation("[UNIFIED-BACKTEST] Completed unified backtest learning session - processed {Count} backtests across {Strategies} strategies", 
                results.Length, string.Join(",", scheduling.RecommendedStrategies));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[UNIFIED-BACKTEST] Failed unified backtest learning session");
        }
    }
    
    private List<UnifiedBacktestConfig> GenerateUnifiedBacktestConfigs(BotCore.Brain.UnifiedSchedulingRecommendation scheduling)
    {
        var configs = new List<UnifiedBacktestConfig>();
        var endDate = DateTime.UtcNow.Date;
        
        // Configuration based on learning intensity
        var (lookbackDays, symbols) = scheduling.LearningIntensity switch
        {
            "INTENSIVE" => (30, new[] { "ES", "NQ" }), // 30 days lookback, both symbols
            "LIGHT" => (7, new[] { "ES" }),            // 7 days lookback, ES only during market hours
            _ => (14, new[] { "ES" })                  // 14 days default
        };
        
        var startDate = endDate.AddDays(-lookbackDays);
        
        foreach (var strategy in scheduling.RecommendedStrategies)
        {
            foreach (var symbol in symbols)
            {
                configs.Add(new UnifiedBacktestConfig
                {
                    Strategy = strategy,
                    Symbol = symbol,
                    StartDate = startDate,
                    EndDate = endDate,
                    InitialCapital = 50000m, // TopStep account size
                    UseUnifiedBrain = true,
                    LearningMode = true,
                    ConfigId = $"{strategy}_{symbol}_{startDate:yyyyMMdd}_{endDate:yyyyMMdd}"
                });
            }
        }
        
        return configs;
    }

    /// <summary>
    /// Run historical backtest using SAME UnifiedTradingBrain as live trading
    /// This ensures identical intelligence is used for both historical and live contexts
    /// </summary>
    public async Task<UnifiedBacktestResult> RunUnifiedHistoricalBacktestAsync(
        UnifiedBacktestConfig config, 
        CancellationToken cancellationToken = default)
    {
        var backtestId = GenerateBacktestId();
        
        try
        {
            _logger.LogInformation("[UNIFIED-BACKTEST] Starting historical backtest {BacktestId} using UnifiedTradingBrain for strategy {Strategy}", 
                backtestId, config.Strategy);
            
            // Initialize unified replay context
            var replayContext = new UnifiedHistoricalReplayContext
            {
                BacktestId = backtestId,
                Config = config,
                StartTime = config.StartDate,
                EndTime = config.EndDate,
                CurrentTime = config.StartDate,
                Strategy = config.Strategy,
                Symbol = config.Symbol,
                TotalBars = 0,
                ProcessedBars = 0,
                IsActive = true
            };
            
            _replayContexts[backtestId] = replayContext;

            // Load historical data with identical formatting as live trading
            var historicalBars = await LoadHistoricalBarsAsync(config, cancellationToken);
            if (!historicalBars.Any())
            {
                throw new InvalidOperationException($"No historical data found for {config.Symbol} in period {config.StartDate} to {config.EndDate}");
            }

            _logger.LogInformation("[UNIFIED-BACKTEST] Loaded {DataPoints} historical bars for {Symbol} {Strategy} from {Start} to {End}",
                historicalBars.Count, config.Symbol, config.Strategy, config.StartDate, config.EndDate);

            // Initialize backtest state
            var backtestState = new UnifiedBacktestState
            {
                StartingCapital = config.InitialCapital,
                CurrentCapital = config.InitialCapital,
                Position = 0,
                UnrealizedPnL = 0,
                RealizedPnL = 0,
                TotalTrades = 0,
                WinningTrades = 0,
                LosingTrades = 0,
                UnifiedDecisions = new List<UnifiedHistoricalDecision>(),
                Strategy = config.Strategy,
                Symbol = config.Symbol
            };

            // Process historical data using SAME UnifiedTradingBrain as live trading
            var barGroups = historicalBars.GroupBy(b => b.Start.Date).OrderBy(g => g.Key);
            
            foreach (var dayGroup in barGroups)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;
                    
                var dailyBars = dayGroup.OrderBy(b => b.Start).ToList();
                await ProcessDailyBarsWithUnifiedBrainAsync(dailyBars, backtestState, replayContext, cancellationToken);
                
                replayContext.ProcessedBars += dailyBars.Count;
                
                // Report progress
                var progressPct = (double)replayContext.ProcessedBars / replayContext.TotalBars * 100;
                if (replayContext.ProcessedBars % 100 == 0)
                {
                    _logger.LogDebug("[UNIFIED-BACKTEST] Progress: {Progress:F1}% - {Trades} trades, PnL: {PnL:F2}", 
                        progressPct, backtestState.TotalTrades, backtestState.RealizedPnL);
                }
            }

            // Calculate final metrics
            var result = CreateUnifiedBacktestResult(backtestState, replayContext, config);
            
            _recentBacktests.Add(result);
            if (_recentBacktests.Count > 50) // Keep only recent backtests
            {
                _recentBacktests.RemoveAt(0);
            }

            _logger.LogInformation("[UNIFIED-BACKTEST] Completed backtest {BacktestId}: {Trades} trades, {WinRate:P1} win rate, {PnL:F2} PnL, {Sharpe:F2} Sharpe", 
                backtestId, result.TotalTrades, result.WinRate, result.NetPnL, result.SharpeRatio);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[UNIFIED-BACKTEST] Failed historical backtest {BacktestId}", backtestId);
            throw;
        }
        finally
        {
            _replayContexts.TryRemove(backtestId, out _);
        }
    }

    /// <summary>
    /// Process daily bars using the SAME UnifiedTradingBrain logic as live trading
    /// This ensures identical decision-making process
    /// </summary>
    private async Task ProcessDailyBarsWithUnifiedBrainAsync(
        List<BotCore.Models.Bar> dailyBars,
        UnifiedBacktestState backtestState,
        UnifiedHistoricalReplayContext replayContext,
        CancellationToken cancellationToken)
    {
        for (int i = 0; i < dailyBars.Count; i++)
        {
            if (cancellationToken.IsCancellationRequested)
                break;
                
            var currentBar = dailyBars[i];
            var historicalBars = dailyBars.Take(i + 1).ToList();
            
            // Skip if we don't have enough bars for decision making
            if (historicalBars.Count < 20)
                continue;
            
            try
            {
                // Create market environment identical to live trading
                var env = new BotCore.Models.Env
                {
                    atr = CalculateATR(historicalBars, 14),
                    volz = CalculateVolZ(historicalBars)
                };
                
                var levels = new BotCore.Models.Levels(); // Initialize as needed
                var riskEngine = new BotCore.Risk.RiskEngine();
                
                // 🚀 CRITICAL: Use SAME UnifiedTradingBrain as live trading
                var brainDecision = await _unifiedBrain.MakeIntelligentDecisionAsync(
                    replayContext.Symbol, env, levels, historicalBars, riskEngine, cancellationToken);
                
                // Record the decision for learning
                var historicalDecision = new UnifiedHistoricalDecision
                {
                    Timestamp = currentBar.Start,
                    Symbol = replayContext.Symbol,
                    Strategy = brainDecision.RecommendedStrategy,
                    Price = currentBar.Close,
                    Decision = brainDecision,
                    MarketContext = CreateMarketContextFromBar(currentBar, historicalBars)
                };
                
                backtestState.UnifiedDecisions.Add(historicalDecision);
                
                // Execute trades if brain recommends them
                if (brainDecision.EnhancedCandidates.Any() && brainDecision.ModelConfidence > 0.7m)
                {
                    var bestCandidate = brainDecision.EnhancedCandidates
                        .OrderByDescending(c => c.QScore)
                        .FirstOrDefault();
                    
                    if (bestCandidate != null)
                    {
                        await ExecuteHistoricalTradeAsync(bestCandidate, backtestState, currentBar, cancellationToken);
                    }
                }
                
                // Feed result back to brain for continuous learning (simulate trade outcome)
                if (historicalDecision.Strategy != null && i + 10 < dailyBars.Count) // Look ahead 10 bars
                {
                    var futureBar = dailyBars[i + 10];
                    var priceMove = futureBar.Close - currentBar.Close;
                    var wasCorrect = (brainDecision.PriceDirection == BotCore.Brain.PriceDirection.Up && priceMove > 0) ||
                                   (brainDecision.PriceDirection == BotCore.Brain.PriceDirection.Down && priceMove < 0);
                    
                    // Feed learning back to UnifiedTradingBrain
                    await _unifiedBrain.LearnFromResultAsync(
                        replayContext.Symbol,
                        historicalDecision.Strategy,
                        priceMove,
                        wasCorrect,
                        TimeSpan.FromMinutes(10),
                        cancellationToken);
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[UNIFIED-BACKTEST] Error processing bar at {Time}", currentBar.Start);
            }
        }
    }

    /// <summary>
    /// Process a single historical data point using UnifiedTradingBrain
    /// This is the core method that ensures identical logic to live trading
    /// </summary>
    private async Task<HistoricalDecision> ProcessHistoricalDataPointAsync(
        HistoricalDataPoint dataPoint,
        BacktestState state,
        HistoricalReplayContext context,
        CancellationToken cancellationToken)
    {
        try
        {
            // Create TradingContext identical to live trading
            var tradingContext = new TradingContext
            {
                Symbol = dataPoint.Symbol,
                Timestamp = dataPoint.Timestamp,
                Price = dataPoint.Close,
                Volume = dataPoint.Volume,
                High = dataPoint.High,
                Low = dataPoint.Low,
                Open = dataPoint.Open,
                
                // Position and risk context
                CurrentPosition = state.Position,
                UnrealizedPnL = state.UnrealizedPnL,
                RealizedPnL = state.RealizedPnL,
                DailyPnL = state.RealizedPnL,
                
                // Risk limits (identical to live trading)
                DailyLossLimit = -1000m,
                MaxDrawdown = -2000m,
                MaxPositionSize = 5m,
                
                // Context flags
                IsBacktest = true,
                IsEmergencyStop = false
            };

            // Use UnifiedTradingBrain adapter for decision (IDENTICAL to live trading)
            var brainDecision = await _unifiedBrain.MakeIntelligentDecisionAsync(tradingContext, cancellationToken);
            
            var historicalDecision = new HistoricalDecision
            {
                Timestamp = dataPoint.Timestamp,
                Symbol = dataPoint.Symbol,
                Price = dataPoint.Close,
                
                // Copy brain decision (should be identical to live trading logic)
                Action = brainDecision?.Action ?? "HOLD",
                Size = brainDecision?.Size ?? 0,
                Confidence = brainDecision?.Confidence ?? 0,
                Strategy = brainDecision?.Strategy ?? "UNKNOWN",
                
                // Include brain attribution for validation
                AlgorithmVersions = brainDecision?.AlgorithmVersions ?? new Dictionary<string, string>(),
                ProcessingTimeMs = brainDecision?.ProcessingTimeMs ?? 0,
                PassedRiskChecks = brainDecision?.PassedRiskChecks ?? false,
                RiskWarnings = brainDecision?.RiskWarnings ?? new List<string>(),
                
                // Backtest-specific metadata
                BacktestId = context.BacktestId,
                BarNumber = context.ProcessedBars,
                PreviousPosition = state.Position,
                PreviousCapital = state.CurrentCapital
            };
            
            return historicalDecision;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-BACKTEST] Failed to process historical data point at {Timestamp}", dataPoint.Timestamp);
            
            // Return safe fallback decision
            return new HistoricalDecision
            {
                Timestamp = dataPoint.Timestamp,
                Symbol = dataPoint.Symbol,
                Price = dataPoint.Close,
                Action = "HOLD",
                Size = 0,
                Confidence = 0,
                Strategy = "ERROR",
                BacktestId = context.BacktestId,
                BarNumber = context.ProcessedBars,
                RiskWarnings = new List<string> { ex.Message }
            };
        }
    }

    /// <summary>
    /// Load historical data with identical formatting to live data
    /// </summary>
    private async Task<List<HistoricalDataPoint>> LoadHistoricalDataAsync(BacktestConfig config, CancellationToken cancellationToken)
    {
        await Task.CompletedTask; // Placeholder
        
        // In production, this would:
        // 1. Load data from same sources as live trading
        // 2. Apply identical preprocessing and feature engineering
        // 3. Ensure timestamp alignment and data quality
        // 4. Format data exactly as live market data
        
        var data = new List<HistoricalDataPoint>();
        var currentDate = config.StartDate;
        var currentPrice = 4500m; // ES starting price
        
        while (currentDate <= config.EndDate)
        {
            if (cancellationToken.IsCancellationRequested) break;
            
            // Generate sample OHLCV data (in production, load from historical database)
            var random = new Random(currentDate.GetHashCode());
            var change = (decimal)(random.NextDouble() - 0.5) * 0.02m; // ±1% daily change
            var newPrice = currentPrice * (1 + change);
            
            var high = Math.Max(currentPrice, newPrice) * (1 + (decimal)random.NextDouble() * 0.005m);
            var low = Math.Min(currentPrice, newPrice) * (1 - (decimal)random.NextDouble() * 0.005m);
            var volume = random.Next(50000, 200000);
            
            data.Add(new HistoricalDataPoint
            {
                Symbol = config.Symbol,
                Timestamp = currentDate,
                Open = currentPrice,
                High = high,
                Low = low,
                Close = newPrice,
                Volume = volume
            });
            
            currentPrice = newPrice;
            currentDate = currentDate.AddDays(1);
        }
        
        return data;
    }

    /// <summary>
    /// Update backtest state based on trading decision
    /// </summary>
    private async Task UpdateBacktestStateAsync(
        BacktestState state,
        HistoricalDecision decision,
        HistoricalDataPoint dataPoint,
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        var previousPosition = state.Position;
        var positionChange = decision.Size - previousPosition;
        
        // Process position change
        if (Math.Abs(positionChange) > 0.01m)
        {
            var fillPrice = dataPoint.Close; // Assume market order execution
            var tradeValue = positionChange * fillPrice;
            
            // Update position
            state.Position = decision.Size;
            
            // Update cash (assuming ES contract value of $50 per point)
            var contractMultiplier = 50m;
            state.CurrentCapital -= Math.Abs(positionChange) * fillPrice * contractMultiplier * 0.001m; // Transaction costs
            
            // Track trade
            state.TotalTrades++;
            
            // Calculate PnL if closing position
            if (Math.Sign(previousPosition) != Math.Sign(decision.Size) && previousPosition != 0)
            {
                var pnl = Math.Sign(previousPosition) * (fillPrice - state.AverageEntryPrice) * Math.Abs(previousPosition) * contractMultiplier;
                state.RealizedPnL += pnl;
                
                if (pnl > 0)
                    state.WinningTrades++;
                else
                    state.LosingTrades++;
            }
            
            // Update average entry price
            if (decision.Size != 0)
            {
                state.AverageEntryPrice = fillPrice;
            }
        }
        
        // Update unrealized PnL
        if (state.Position != 0)
        {
            var contractMultiplier = 50m;
            state.UnrealizedPnL = Math.Sign(state.Position) * (dataPoint.Close - state.AverageEntryPrice) * Math.Abs(state.Position) * contractMultiplier;
        }
        else
        {
            state.UnrealizedPnL = 0;
        }
    }

    /// <summary>
    /// Calculate final backtest metrics
    /// </summary>
    private async Task<BacktestResult> CalculateBacktestMetricsAsync(
        BacktestState state,
        HistoricalReplayContext context,
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        var totalPnL = state.RealizedPnL + state.UnrealizedPnL;
        var totalReturn = totalPnL / state.StartingCapital;
        
        // Calculate performance metrics from decisions
        var returns = state.Decisions
            .Where(d => d.Action != "HOLD")
            .Select(d => (decimal)(Random.Shared.NextDouble() - 0.5) * 0.02m) // Mock returns
            .ToList();
        
        var avgReturn = returns.Any() ? returns.Average() : 0;
        var volatility = returns.Any() ? CalculateStandardDeviation(returns) : 0;
        var sharpeRatio = volatility > 0 ? avgReturn / volatility * (decimal)Math.Sqrt(252) : 0;
        
        // Calculate max drawdown (simplified)
        var maxDrawdown = Math.Min(totalReturn, -0.05m); // Mock calculation
        
        return new BacktestResult
        {
            BacktestId = context.BacktestId,
            StartDate = context.StartTime,
            EndDate = context.EndTime,
            InitialCapital = state.StartingCapital,
            FinalCapital = state.CurrentCapital + totalPnL,
            TotalReturn = totalReturn,
            SharpeRatio = sharpeRatio,
            SortinoRatio = sharpeRatio * 1.2m, // Mock calculation
            MaxDrawdown = maxDrawdown,
            TotalTrades = state.TotalTrades,
            WinningTrades = state.WinningTrades,
            LosingTrades = state.LosingTrades,
            CompletedAt = DateTime.UtcNow,
            
            // UnifiedTradingBrain specific metrics
            BrainDecisionCount = state.Decisions.Count,
            AverageProcessingTimeMs = state.Decisions.Any() ? (double)state.Decisions.Average(d => d.ProcessingTimeMs) : 0,
            RiskCheckFailures = state.Decisions.Count(d => !d.PassedRiskChecks),
            AlgorithmUsage = CalculateAlgorithmUsage(state.Decisions).ToDictionary(kvp => kvp.Key, kvp => (object)kvp.Value)
        };
    }

    #region Helper Methods

    private List<BacktestConfig> GenerateBacktestConfigs(TrainingIntensity intensity)
    {
        var configs = new List<BacktestConfig>();
        
        // Generate configs based on intensity
        var configCount = intensity.Level switch
        {
            TrainingIntensityLevel.Intensive => 8,
            TrainingIntensityLevel.High => 4,
            TrainingIntensityLevel.Medium => 2,
            _ => 1
        };
        
        var endDate = DateTime.UtcNow.Date.AddDays(-1); // Yesterday
        
        for (int i = 0; i < configCount; i++)
        {
            var daysBack = 30 + (i * 30); // 30, 60, 90, etc. days
            var startDate = endDate.AddDays(-daysBack);
            
            configs.Add(new BacktestConfig
            {
                Symbol = "ES",
                StartDate = startDate,
                EndDate = endDate,
                InitialCapital = 50000m,
                MaxDrawdown = -2000m,
                MaxPositionSize = 5m
            });
        }
        
        return configs;
    }

    private async Task AnalyzeBacktestResultsAsync(BacktestResult[] results, TrainingIntensity intensity, CancellationToken cancellationToken)
    {
        await Task.CompletedTask;
        
        // Analyze backtest results for potential challenger training
        var bestResult = results.OrderByDescending(r => r.SharpeRatio).FirstOrDefault();
        
        if (bestResult != null && bestResult.SharpeRatio > 1.5m)
        {
            _logger.LogInformation("[ENHANCED-BACKTEST] Found promising backtest result with Sharpe ratio {Sharpe:F2} - considering challenger training", bestResult.SharpeRatio);
            
            // TODO: Trigger challenger training based on promising results
            // This would use the patterns found in the backtest to train a new challenger
        }
    }

    private Dictionary<string, int> CalculateAlgorithmUsage(List<HistoricalDecision> decisions)
    {
        var usage = new Dictionary<string, int>();
        
        foreach (var decision in decisions)
        {
            foreach (var algorithm in decision.AlgorithmVersions.Keys)
            {
                usage[algorithm] = usage.GetValueOrDefault(algorithm, 0) + 1;
            }
        }
        
        return usage;
    }

    private decimal CalculateStandardDeviation(List<decimal> values)
    {
        if (!values.Any()) return 0;
        
        var mean = values.Average();
        var sumOfSquares = values.Sum(v => (v - mean) * (v - mean));
        return (decimal)Math.Sqrt((double)(sumOfSquares / values.Count));
    }

    private string GenerateBacktestId()
    {
        return $"BT_{DateTime.UtcNow:yyyyMMdd_HHmmss}_{Guid.NewGuid().ToString("N")[..8]}";
    }

    /// <summary>
    /// Calculate ATR using Welles Wilder formula
    /// </summary>
    private decimal CalculateATR(IList<BotCore.Models.Bar> bars, int period = 14)
    {
        if (bars.Count < period + 1) return 0;

        var trueRanges = new List<decimal>();
        
        for (int i = 1; i < bars.Count; i++)
        {
            var current = bars[i];
            var previous = bars[i - 1];
            
            var tr1 = current.High - current.Low;
            var tr2 = Math.Abs(current.High - previous.Close);
            var tr3 = Math.Abs(current.Low - previous.Close);
            
            var trueRange = Math.Max(tr1, Math.Max(tr2, tr3));
            trueRanges.Add(trueRange);
        }

        if (trueRanges.Count < period) return 0;

        // Welles Wilder's smoothing (modified EMA with alpha = 1/period)
        var atr = trueRanges.Take(period).Average();
        
        for (int i = period; i < trueRanges.Count; i++)
        {
            atr = ((atr * (period - 1)) + trueRanges[i]) / period;
        }

        return atr;
    }

    /// <summary>
    /// Calculate VolZ (volatility z-score) using rolling mean and standard deviation
    /// </summary>
    private decimal CalculateVolZ(IList<BotCore.Models.Bar> bars, int period = 20)
    {
        if (bars.Count < period) return 0;

        var returns = new List<decimal>();
        
        // Calculate returns
        for (int i = 1; i < bars.Count; i++)
        {
            if (bars[i - 1].Close != 0)
            {
                var ret = (bars[i].Close - bars[i - 1].Close) / bars[i - 1].Close;
                returns.Add(ret);
            }
        }

        if (returns.Count < period) return 0;

        // Get the last period for calculation
        var lastReturns = returns.TakeLast(period).ToList();
        var mean = lastReturns.Average();
        var variance = lastReturns.Sum(r => (r - mean) * (r - mean)) / (period - 1);
        var stdDev = (decimal)Math.Sqrt((double)variance);

        if (stdDev == 0) return 0;

        // Current return
        var currentReturn = returns.LastOrDefault();
        
        // Z-score: (current - mean) / stddev
        return (currentReturn - mean) / stdDev;
    }

    /// <summary>
    /// Create market context from bar data
    /// </summary>
    private TradingContext CreateMarketContextFromBar(BotCore.Models.Bar bar)
    {
        return new TradingContext
        {
            Symbol = bar.Symbol,
            Timestamp = bar.Start, // Use Start property instead of Timestamp
            High = bar.High,
            Low = bar.Low,
            Open = bar.Open,
            Close = bar.Close,
            CurrentPrice = bar.Close,
            Price = bar.Close,
            Volume = (long)bar.Volume,
            IsBacktest = true,
            TechnicalIndicators = new Dictionary<string, decimal>(),
            Metadata = new Dictionary<string, object>() // Use Metadata instead of AdditionalData
        };
    }

    /// <summary>
    /// Execute historical trade with slippage, fees, and limits
    /// </summary>
    private async Task<TradeResult> ExecuteHistoricalTradeAsync(
        UnifiedHistoricalDecision decision, 
        decimal currentPrice, 
        UnifiedBacktestState state,
        CancellationToken cancellationToken)
    {
        await Task.CompletedTask; // Placeholder for async operations
        
        var slippage = 0.25m; // ES tick size
        var commission = 2.50m; // Per contract
        
        var executionPrice = decision.Action switch
        {
            "BUY" => currentPrice + slippage,
            "SELL" => currentPrice - slippage,
            _ => currentPrice
        };

        var tradeSize = decision.Size;
        var tradeValue = tradeSize * executionPrice;
        
        // Calculate PnL for position changes
        var pnl = 0m;
        if (decision.Action == "BUY")
        {
            pnl = (state.Position < 0) ? Math.Abs(state.Position) * (state.AverageEntryPrice - executionPrice) : 0;
            state.Position += tradeSize;
        }
        else if (decision.Action == "SELL")
        {
            pnl = (state.Position > 0) ? state.Position * (executionPrice - state.AverageEntryPrice) : 0;
            state.Position -= tradeSize;
        }

        // Update position tracking
        if (state.Position != 0)
        {
            state.AverageEntryPrice = executionPrice;
        }

        state.RealizedPnL += pnl - commission;
        state.CurrentCapital = state.StartingCapital + state.RealizedPnL;

        return new TradeResult
        {
            Success = true,
            ExecutionPrice = executionPrice,
            ExecutedSize = tradeSize,
            Commission = commission,
            Slippage = slippage,
            RealizedPnL = pnl,
            Timestamp = decision.Timestamp
        };
    }

    #endregion
}

#region Supporting Models

/// <summary>
/// Historical replay context for tracking backtest progress
/// </summary>
public class HistoricalReplayContext
{
    public string BacktestId { get; set; } = string.Empty;
    public BacktestConfig Config { get; set; } = new();
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public DateTime CurrentTime { get; set; }
    public int TotalBars { get; set; }
    public int ProcessedBars { get; set; }
    public bool IsActive { get; set; }
}

/// <summary>
/// Historical data point with identical structure to live data
/// </summary>
public class HistoricalDataPoint
{
    public string Symbol { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public decimal Open { get; set; }
    public decimal High { get; set; }
    public decimal Low { get; set; }
    public decimal Close { get; set; }
    public long Volume { get; set; }
}

/// <summary>
/// Backtest state tracking
/// </summary>
public class BacktestState
{
    public decimal StartingCapital { get; set; }
    public decimal CurrentCapital { get; set; }
    public decimal Position { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public decimal RealizedPnL { get; set; }
    public decimal AverageEntryPrice { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public List<HistoricalDecision> Decisions { get; set; } = new();
}

/// <summary>
/// Historical decision from UnifiedTradingBrain
/// </summary>
public class HistoricalDecision
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public string Action { get; set; } = string.Empty;
    public decimal Size { get; set; }
    public decimal Confidence { get; set; }
    public string Strategy { get; set; } = string.Empty;
    
    // Brain attribution
    public Dictionary<string, string> AlgorithmVersions { get; set; } = new();
    public decimal ProcessingTimeMs { get; set; }
    public bool PassedRiskChecks { get; set; }
    public List<string> RiskWarnings { get; set; } = new();
    
    // Backtest context
    public string BacktestId { get; set; } = string.Empty;
    public int BarNumber { get; set; }
    public decimal PreviousPosition { get; set; }
    public decimal PreviousCapital { get; set; }
}

/// <summary>
/// Result of executing a historical trade
/// </summary>
public class TradeResult
{
    public bool Success { get; set; }
    public decimal ExecutionPrice { get; set; }
    public decimal ExecutedSize { get; set; }
    public decimal Commission { get; set; }
    public decimal Slippage { get; set; }
    public decimal RealizedPnL { get; set; }
    public DateTime Timestamp { get; set; }
    public string? ErrorMessage { get; set; }
}

#endregion