using System;
using System.Collections.Generic;

namespace TradingBot.IntelligenceStack;

#region Cloud Flow Classes (merged from CloudFlowService)

/// <summary>
/// Configuration options for cloud flow service (merged from CloudFlowService)
/// </summary>
public class CloudFlowOptions
{
    public bool Enabled { get; set; } = true;
    public string CloudEndpoint { get; set; } = string.Empty;
    public string InstanceId { get; set; } = Environment.MachineName;
    public int TimeoutSeconds { get; set; } = 30;
}

/// <summary>
/// Trade record for cloud push (merged from CloudFlowService)
/// </summary>
public class CloudTradeRecord
{
    public string TradeId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal EntryPrice { get; set; }
    public decimal ExitPrice { get; set; }
    public decimal PnL { get; set; }
    public DateTime EntryTime { get; set; }
    public DateTime ExitTime { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; } = new();
}

/// <summary>
/// Service metrics for cloud push (merged from CloudFlowService)
/// </summary>
public class CloudServiceMetrics
{
    public double InferenceLatencyMs { get; set; }
    public double PredictionAccuracy { get; set; }
    public double FeatureDrift { get; set; }
    public int ActiveModels { get; set; }
    public long MemoryUsageMB { get; set; }
    public Dictionary<string, double> CustomMetrics { get; } = new();
}

/// <summary>
/// ML prediction result for requirement 2: Use ML Predictions in Trading Decisions
/// </summary>
public class MLPrediction
{
    public string Symbol { get; set; } = string.Empty;
    public double Confidence { get; set; }
    public string Direction { get; set; } = string.Empty; // BUY, SELL, HOLD
    public string ModelId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public bool IsValid { get; set; }
    public Dictionary<string, object> Metadata { get; } = new();
}

#endregion