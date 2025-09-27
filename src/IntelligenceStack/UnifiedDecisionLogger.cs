using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using TradingBot.Abstractions.Helpers;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Unified Decision Logger for requirement 10: Observability & Lineage
/// Logs complete decision lineage with symbol, strategy, regime, predictions, sizes, scores
/// </summary>
public class UnifiedDecisionLogger
{
    private readonly ILogger<UnifiedDecisionLogger> _logger;
    
    // LoggerMessage delegates for CA1848 compliance
    private static readonly Action<ILogger, string, Exception?> LoggerInitialized =
        LoggerMessage.Define<string>(LogLevel.Information, new EventId(1001, "LoggerInitialized"),
            "[UNIFIED_LOGGER] Decision logger initialized: {LogFile}");
            
    private static readonly Action<ILogger, string, string, string, double, int, string, Exception?> DecisionLogged =
        LoggerMessage.Define<string, string, string, double, int, string>(LogLevel.Information, new EventId(1002, "DecisionLogged"),
            "[UNIFIED_DECISION] {Symbol} {Strategy} {Regime}: P_final={PFinal:F3} size={Size} action={Action}");
            
    private static readonly Action<ILogger, Exception?> DecisionLogFailed =
        LoggerMessage.Define(LogLevel.Error, new EventId(1003, "DecisionLogFailed"),
            "[UNIFIED_LOGGER] Failed to log decision");
    private readonly string _logFilePath;
    
    public UnifiedDecisionLogger(ILogger<UnifiedDecisionLogger> logger, string logDirectory = "logs")
    {
        _logger = logger;
        
        // Create logs directory
        Directory.CreateDirectory(logDirectory);
        
        // Create daily log file
        var logFileName = $"unified_decisions_{DateTime.UtcNow:yyyyMMdd}.jsonl";
        _logFilePath = Path.Combine(logDirectory, logFileName);
        
        LoggerInitialized(_logger, _logFilePath, null);
    }
    
    /// <summary>
    /// Log a complete trading decision with full lineage
    /// </summary>
    public async Task LogDecisionAsync(UnifiedDecisionRecord decision)
    {
        ArgumentNullException.ThrowIfNull(decision);

        try
        {
            // Add timestamp if not set
            if (decision.Timestamp == default)
            {
                decision.Timestamp = DateTime.UtcNow;
            }
            
            // Serialize to JSON
            var jsonLine = JsonSerializationHelper.SerializeCompact(decision);
            
            // Write to file (append mode)
            await File.AppendAllTextAsync(_logFilePath, jsonLine + Environment.NewLine).ConfigureAwait(false);
            
            // Also log to console for immediate visibility
            DecisionLogged(_logger, decision.Symbol, decision.Strategy, decision.Regime, decision.PFinal, decision.FinalSize, decision.Action, null);
                
        }
        catch (JsonException ex)
        {
            DecisionLogFailed(_logger, ex);
        }
        catch (IOException ex)
        {
            DecisionLogFailed(_logger, ex);
        }
        catch (UnauthorizedAccessException ex)
        {
            DecisionLogFailed(_logger, ex);
        }
    }
    
    /// <summary>
    /// Create a decision record from trading components
    /// </summary>
    public static UnifiedDecisionRecord CreateDecisionRecord(
        string symbol,
        string strategy,
        string regime,
        double pCloud,
        double pOnline,
        double pFinal,
        int finalSize,
        string action,
        Dictionary<string, object>? additionalData = null)
    {
        var record = new UnifiedDecisionRecord
        {
            Symbol = symbol,
            Strategy = strategy,
            Regime = regime,
            PCloud = pCloud,
            POnline = pOnline,
            PFinal = pFinal,
            FinalSize = finalSize,
            Action = action,
            Timestamp = DateTime.UtcNow
        };

        // Add additional data if provided
        if (additionalData != null)
        {
            foreach (var kvp in additionalData)
            {
                record.AdditionalData[kvp.Key] = kvp.Value;
            }
        }

        return record;
    }
}

/// <summary>
/// Complete decision record with full lineage for requirement 10
/// </summary>
public class UnifiedDecisionRecord
{
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Regime { get; set; } = string.Empty;
    
    // ML Predictions
    public double PCloud { get; set; }
    public double POnline { get; set; }
    public double PFinal { get; set; }
    public double ConfidenceThreshold { get; set; }
    public bool ConfidenceGatePassed { get; set; }
    
    // Position Sizing
    public int SacSize { get; set; }
    public int FinalSize { get; set; }
    public double UcbScore { get; set; }
    
    // Risk Management
    public Dictionary<string, int> RiskCaps { get; } = new();
    public bool RiskGatePassed { get; set; }
    
    // Decision Output
    public string Action { get; set; } = string.Empty;
    public string Reason { get; set; } = string.Empty;
    
    // Drift & Quality Metrics
    public bool DriftFlag { get; set; }
    public double ModelHealth { get; set; }
    public string ModelVersion { get; set; } = string.Empty;
    
    // Market Context
    public double Price { get; set; }
    public double Volume { get; set; }
    public double Volatility { get; set; }
    
    // Execution Details
    public string OrderId { get; set; } = string.Empty;
    public double? FillPrice { get; set; }
    public DateTime? FillTime { get; set; }
    
    // Additional data for extensibility
    public Dictionary<string, object> AdditionalData { get; } = new();
}

/// <summary>
/// Extension methods for easy decision logging integration
/// </summary>
public static class DecisionLoggingExtensions
{
    /// <summary>
    /// Log a trading decision from BotSupervisor
    /// </summary>
    public static Task LogTradingDecisionAsync(
        this UnifiedDecisionLogger logger,
        string symbol,
        string strategy,
        double pCloud,
        double pOnline,
        double pFinal,
        bool gatesPassed,
        int finalSize,
        string action,
        string? orderId = null)
    {
        if (logger == null) return Task.CompletedTask;
        
        var decision = UnifiedDecisionLogger.CreateDecisionRecord(
            symbol, strategy, "UNKNOWN", pCloud, pOnline, pFinal, finalSize, action,
            new Dictionary<string, object>
            {
                ["confidence_gates_passed"] = gatesPassed,
                ["order_id"] = orderId ?? "",
                ["source"] = "BotSupervisor"
            });
            
        return logger.LogDecisionAsync(decision);
    }
    
    /// <summary>
    /// Log a regime detection decision
    /// </summary>
    public static Task LogRegimeDecisionAsync(
        this UnifiedDecisionLogger logger,
        string symbol,
        string previousRegime,
        string newRegime,
        double confidence,
        Dictionary<string, double> indicators)
    {
        if (logger == null) return Task.CompletedTask;
        
        var decision = UnifiedDecisionLogger.CreateDecisionRecord(
            symbol, "REGIME_DETECTION", newRegime, 0, 0, confidence, 0, 
            previousRegime != newRegime ? "REGIME_CHANGE" : "REGIME_STABLE",
            new Dictionary<string, object>
            {
                ["previous_regime"] = previousRegime,
                ["regime_indicators"] = indicators,
                ["source"] = "RegimeDetector"
            });
            
        return logger.LogDecisionAsync(decision);
    }
    
    /// <summary>
    /// Log an order fill with complete context
    /// </summary>
    public static Task LogOrderFillAsync(
        this UnifiedDecisionLogger logger,
        string orderId,
        string symbol,
        string strategy,
        int quantity,
        double fillPrice,
        string side)
    {
        if (logger == null) return Task.CompletedTask;
        
        var decision = UnifiedDecisionLogger.CreateDecisionRecord(
            symbol, strategy, "UNKNOWN", 0, 0, 0, quantity, "FILL",
            new Dictionary<string, object>
            {
                ["order_id"] = orderId,
                ["fill_price"] = fillPrice,
                ["side"] = side,
                ["fill_time"] = DateTime.UtcNow,
                ["source"] = "OrderFill"
            });
            
        decision.OrderId = orderId;
        decision.FillPrice = fillPrice;
        decision.FillTime = DateTime.UtcNow;
        
        return logger.LogDecisionAsync(decision);
    }
}