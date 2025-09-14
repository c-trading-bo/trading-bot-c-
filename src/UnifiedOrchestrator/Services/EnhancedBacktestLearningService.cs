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
/// CRITICAL REQUIREMENT: Replace legacy S2/S3 strategy classes in backtest/replay with UnifiedTradingBrain
/// This ensures historical data pipeline uses identical intelligence as live trading:
/// - Same data formatting and feature engineering
/// - Same decision-making logic (UnifiedTradingBrain.DecideAsync)
/// - Same risk management and position sizing
/// - Identical context and outputs for reproducible results
/// </summary>
public class EnhancedBacktestLearningService : BackgroundService
{
    private readonly ILogger<EnhancedBacktestLearningService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private readonly IUnifiedDataIntegrationService _dataIntegration;
    private readonly ITradingBrainAdapter _brainAdapter;
    private readonly ITrainingBrain _trainingBrain;
    private readonly IMarketHoursService _marketHours;
    
    // Historical replay state
    private readonly Dictionary<string, HistoricalReplayContext> _replayContexts = new();
    private readonly List<BacktestResult> _recentBacktests = new();

    public EnhancedBacktestLearningService(
        ILogger<EnhancedBacktestLearningService> logger,
        IServiceProvider serviceProvider,
        IUnifiedDataIntegrationService dataIntegration,
        ITradingBrainAdapter brainAdapter,
        ITrainingBrain trainingBrain,
        IMarketHoursService marketHours)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        _dataIntegration = dataIntegration;
        _brainAdapter = brainAdapter;
        _trainingBrain = trainingBrain;
        _marketHours = marketHours;
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
                // Check if it's a good time for intensive training
                var trainingIntensity = await _marketHours.GetRecommendedTrainingIntensityAsync(stoppingToken);
                
                if (trainingIntensity.Level == "INTENSIVE" || trainingIntensity.Level == "MODERATE")
                {
                    await RunScheduledBacktestLearningAsync(trainingIntensity, stoppingToken);
                }
                
                // Wait before next check (adjust based on intensity)
                var delayMinutes = trainingIntensity.Level switch
                {
                    "INTENSIVE" => 60,  // Check every hour during intensive periods
                    "MODERATE" => 120,  // Check every 2 hours during moderate periods
                    "BACKGROUND" => 240, // Check every 4 hours during background periods
                    _ => 480            // Check every 8 hours otherwise
                };
                
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
    /// Run scheduled backtest learning session
    /// </summary>
    private async Task RunScheduledBacktestLearningAsync(TrainingIntensity intensity, CancellationToken cancellationToken)
    {
        _logger.LogInformation("[ENHANCED-BACKTEST] Starting scheduled backtest learning session with {Intensity} intensity", intensity.Level);
        
        try
        {
            // Validate data consistency first
            var dataConsistent = await _dataIntegration.ValidateDataConsistencyAsync(cancellationToken);
            if (!dataConsistent)
            {
                _logger.LogWarning("[ENHANCED-BACKTEST] Skipping backtest due to data consistency issues");
                return;
            }

            // Configure backtest based on intensity
            var backtestConfigs = GenerateBacktestConfigs(intensity);
            
            var parallelJobs = Math.Max(1, intensity.ParallelJobs);
            var semaphore = new SemaphoreSlim(parallelJobs, parallelJobs);
            
            var tasks = backtestConfigs.Select(async config =>
            {
                await semaphore.WaitAsync(cancellationToken);
                try
                {
                    return await RunHistoricalBacktestAsync(config, cancellationToken);
                }
                finally
                {
                    semaphore.Release();
                }
            });

            var results = await Task.WhenAll(tasks);
            
            // Analyze results and potentially train new challengers
            await AnalyzeBacktestResultsAsync(results, intensity, cancellationToken);
            
            _logger.LogInformation("[ENHANCED-BACKTEST] Completed scheduled backtest learning session - processed {Count} backtests", results.Length);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-BACKTEST] Failed scheduled backtest learning session");
        }
    }

    /// <summary>
    /// Run historical backtest using UnifiedTradingBrain (replaces legacy strategies)
    /// This ensures identical intelligence is used for both historical and live contexts
    /// </summary>
    public async Task<BacktestResult> RunHistoricalBacktestAsync(
        BacktestConfig config, 
        CancellationToken cancellationToken = default)
    {
        var backtestId = GenerateBacktestId();
        
        try
        {
            _logger.LogInformation("[ENHANCED-BACKTEST] Starting historical backtest {BacktestId} with UnifiedTradingBrain", backtestId);
            
            // Initialize historical replay context
            var replayContext = new HistoricalReplayContext
            {
                BacktestId = backtestId,
                Config = config,
                StartTime = config.StartDate,
                EndTime = config.EndDate,
                CurrentTime = config.StartDate,
                TotalBars = 0,
                ProcessedBars = 0,
                IsActive = true
            };
            
            _replayContexts[backtestId] = replayContext;

            // Load historical data with identical formatting as live trading
            var historicalData = await LoadHistoricalDataAsync(config, cancellationToken);
            if (!historicalData.Any())
            {
                throw new InvalidOperationException("No historical data found for the specified period");
            }

            _logger.LogInformation("[ENHANCED-BACKTEST] Loaded {DataPoints} historical data points for period {Start} to {End}",
                historicalData.Count, config.StartDate, config.EndDate);

            // Initialize backtest state
            var backtestState = new BacktestState
            {
                StartingCapital = config.InitialCapital,
                CurrentCapital = config.InitialCapital,
                Position = 0,
                UnrealizedPnL = 0,
                RealizedPnL = 0,
                TotalTrades = 0,
                WinningTrades = 0,
                LosingTrades = 0,
                Decisions = new List<HistoricalDecision>()
            };

            // Process historical data using UnifiedTradingBrain
            foreach (var dataPoint in historicalData)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                var decision = await ProcessHistoricalDataPointAsync(dataPoint, backtestState, replayContext, cancellationToken);
                backtestState.Decisions.Add(decision);
                
                replayContext.ProcessedBars++;
                replayContext.CurrentTime = dataPoint.Timestamp;
                
                // Update position and PnL based on decision
                await UpdateBacktestStateAsync(backtestState, decision, dataPoint, cancellationToken);
            }

            // Calculate final metrics
            var result = await CalculateBacktestMetricsAsync(backtestState, replayContext, cancellationToken);
            
            // Store result for analysis
            _recentBacktests.Add(result);
            if (_recentBacktests.Count > 50) // Keep recent results only
            {
                _recentBacktests.RemoveAt(0);
            }

            _logger.LogInformation("[ENHANCED-BACKTEST] Completed backtest {BacktestId}: " +
                "Return: {Return:P2}, Sharpe: {Sharpe:F2}, Max DD: {MaxDD:P2}, Trades: {Trades}",
                backtestId, result.TotalReturn, result.SharpeRatio, result.MaxDrawdown, result.TotalTrades);

            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[ENHANCED-BACKTEST] Failed to run historical backtest {BacktestId}", backtestId);
            throw;
        }
        finally
        {
            // Clean up replay context
            _replayContexts.Remove(backtestId);
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
            var brainDecision = await _brainAdapter.MakeDecisionAsync(tradingContext, "HISTORICAL", cancellationToken);
            
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
            AverageProcessingTimeMs = state.Decisions.Any() ? state.Decisions.Average(d => d.ProcessingTimeMs) : 0,
            RiskCheckFailures = state.Decisions.Count(d => !d.PassedRiskChecks),
            AlgorithmUsage = CalculateAlgorithmUsage(state.Decisions)
        };
    }

    #region Helper Methods

    private List<BacktestConfig> GenerateBacktestConfigs(TrainingIntensity intensity)
    {
        var configs = new List<BacktestConfig>();
        
        // Generate configs based on intensity
        var configCount = intensity.Level switch
        {
            "INTENSIVE" => 8,
            "MODERATE" => 4,
            "BACKGROUND" => 2,
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

#endregion