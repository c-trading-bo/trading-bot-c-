using Microsoft.Extensions.Logging;
using BotCore.Services;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Services;

/// <summary>
/// Execution approval service for tick-level trade validation.
/// Evaluates execution candidates using tick microstructure features.
/// 
/// Phase 5: Live Inference Services (Week 5-6) - Advanced Feature
/// - Train tiny execution model on tick data
/// - Evaluate candidates using tick features
/// - Return approve/reject + slippage estimate
/// 
/// Design principles:
/// - Fast approval: &lt;10ms decision time
/// - Risk-aware: Considers spread, imbalance, volatility
/// - Production-ready: Comprehensive error handling
/// </summary>
public class ExecutionApprovalService
{
    private readonly ILogger<ExecutionApprovalService> _logger;
    private readonly TickBufferService _tickBuffer;
    
    // Approval thresholds
    private const double MaxSpreadBps = 10.0;           // Max 10 basis points spread
    private const double MaxTickVolatility = 2.0;       // Max tick volatility
    private const double MinTickIntensity = 1.0;        // Min 1 tick per second
    private const double MaxOrderFlowImbalance = 0.8;   // Max 80% imbalance
    private const double SlippageBpsPerBpsSpread = 0.5; // Slippage = 50% of spread
    
    // Approval scoring weights
    private const double SpreadWeight = 0.3;
    private const double ImbalanceWeight = 0.25;
    private const double VolatilityWeight = 0.25;
    private const double IntensityWeight = 0.2;
    
    // Approval threshold
    private const double MinApprovalScore = 0.6; // 60% score required
    
    public ExecutionApprovalService(
        ILogger<ExecutionApprovalService> logger,
        TickBufferService tickBuffer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _tickBuffer = tickBuffer ?? throw new ArgumentNullException(nameof(tickBuffer));
    }
    
    /// <summary>
    /// Evaluate execution candidate and return approval decision.
    /// </summary>
    /// <param name="symbol">Symbol to trade</param>
    /// <param name="direction">Trade direction (1=buy, -1=sell)</param>
    /// <param name="quantity">Quantity to trade</param>
    /// <param name="price">Expected execution price</param>
    /// <returns>Execution approval result</returns>
    public ExecutionApproval EvaluateExecution(
        string symbol,
        int direction,
        double quantity,
        double price)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            _logger.LogWarning("[EXEC_APPROVAL] Invalid symbol");
            return new ExecutionApproval
            {
                Approved = false,
                Reason = "Invalid symbol",
                Score = 0.0
            };
        }
        
        try
        {
            var startTime = DateTimeOffset.UtcNow;
            
            // Get tick features
            var tickFeatures = _tickBuffer.ComputeTickFeatures(symbol);
            
            if (tickFeatures.Count == 0)
            {
                _logger.LogWarning(
                    "[EXEC_APPROVAL] No tick data available for {Symbol}",
                    symbol);
                return new ExecutionApproval
                {
                    Approved = false,
                    Reason = "Insufficient tick data",
                    Score = 0.0
                };
            }
            
            // Extract features
            var spread = tickFeatures.GetValueOrDefault("spread_bps", 0.0);
            var imbalance = Math.Abs(tickFeatures.GetValueOrDefault("order_flow_imbalance", 0.0));
            var volatility = tickFeatures.GetValueOrDefault("tick_volatility", 0.0);
            var intensity = tickFeatures.GetValueOrDefault("tick_intensity", 0.0);
            
            // Check hard limits
            var violations = new List<string>();
            
            if (spread > MaxSpreadBps)
            {
                violations.Add($"Spread too wide: {spread:F2} bps > {MaxSpreadBps} bps");
            }
            
            if (volatility > MaxTickVolatility)
            {
                violations.Add($"Volatility too high: {volatility:F3} > {MaxTickVolatility}");
            }
            
            if (intensity < MinTickIntensity)
            {
                violations.Add($"Low tick intensity: {intensity:F2} < {MinTickIntensity}");
            }
            
            if (imbalance > MaxOrderFlowImbalance)
            {
                violations.Add($"High order flow imbalance: {imbalance:F2} > {MaxOrderFlowImbalance}");
            }
            
            // If any hard limits violated, reject
            if (violations.Count > 0)
            {
                _logger.LogWarning(
                    "[EXEC_APPROVAL] Execution rejected for {Symbol}: {Violations}",
                    symbol, string.Join("; ", violations));
                
                return new ExecutionApproval
                {
                    Approved = false,
                    Reason = string.Join("; ", violations),
                    Score = 0.0,
                    EstimatedSlippageBps = spread * SlippageBpsPerBpsSpread
                };
            }
            
            // Compute approval score (0-1)
            var spreadScore = 1.0 - Math.Min(spread / MaxSpreadBps, 1.0);
            var imbalanceScore = 1.0 - Math.Min(imbalance / MaxOrderFlowImbalance, 1.0);
            var volatilityScore = 1.0 - Math.Min(volatility / MaxTickVolatility, 1.0);
            var intensityScore = Math.Min(intensity / (MinTickIntensity * 5.0), 1.0); // Good if 5x min
            
            var overallScore = (spreadScore * SpreadWeight) +
                             (imbalanceScore * ImbalanceWeight) +
                             (volatilityScore * VolatilityWeight) +
                             (intensityScore * IntensityWeight);
            
            var approved = overallScore >= MinApprovalScore;
            
            // Estimate slippage based on spread and imbalance
            var slippageBps = spread * SlippageBpsPerBpsSpread;
            if (imbalance > 0.5)
            {
                // High imbalance increases slippage
                slippageBps *= (1.0 + imbalance);
            }
            
            var elapsedMs = (DateTimeOffset.UtcNow - startTime).TotalMilliseconds;
            
            _logger.LogInformation(
                "[EXEC_APPROVAL] {Result} execution for {Symbol} {Direction} {Quantity}@{Price}: " +
                "Score={Score:F2}, Slippage={Slippage:F2}bps, Time={Time:F2}ms",
                approved ? "APPROVED" : "REJECTED",
                symbol, direction > 0 ? "BUY" : "SELL", quantity, price,
                overallScore, slippageBps, elapsedMs);
            
            return new ExecutionApproval
            {
                Approved = approved,
                Reason = approved ? "Execution approved" : $"Low score: {overallScore:F2} < {MinApprovalScore}",
                Score = overallScore,
                EstimatedSlippageBps = slippageBps,
                TickFeatures = tickFeatures
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[EXEC_APPROVAL] Error evaluating execution for {Symbol}", symbol);
            
            return new ExecutionApproval
            {
                Approved = false,
                Reason = $"Error: {ex.Message}",
                Score = 0.0
            };
        }
    }
    
    /// <summary>
    /// Get execution statistics for a symbol.
    /// </summary>
    public ExecutionStatistics GetStatistics(string symbol)
    {
        var tickCount = _tickBuffer.GetTickCount(symbol);
        var features = _tickBuffer.ComputeTickFeatures(symbol);
        
        return new ExecutionStatistics
        {
            Symbol = symbol,
            TickCount = tickCount,
            CurrentSpreadBps = features.GetValueOrDefault("spread_bps", 0.0),
            CurrentImbalance = features.GetValueOrDefault("order_flow_imbalance", 0.0),
            CurrentVolatility = features.GetValueOrDefault("tick_volatility", 0.0),
            CurrentIntensity = features.GetValueOrDefault("tick_intensity", 0.0)
        };
    }
}

/// <summary>
/// Execution approval result.
/// </summary>
public class ExecutionApproval
{
    public bool Approved { get; set; }
    public string Reason { get; set; } = string.Empty;
    public double Score { get; set; }
    public double EstimatedSlippageBps { get; set; }
    public Dictionary<string, double> TickFeatures { get; set; } = new();
}

/// <summary>
/// Execution statistics for a symbol.
/// </summary>
public class ExecutionStatistics
{
    public string Symbol { get; set; } = string.Empty;
    public int TickCount { get; set; }
    public double CurrentSpreadBps { get; set; }
    public double CurrentImbalance { get; set; }
    public double CurrentVolatility { get; set; }
    public double CurrentIntensity { get; set; }
}
