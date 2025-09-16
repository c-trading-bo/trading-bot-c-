using System;

namespace TradingBot.IntelligenceStack;

/// <summary>
/// Configuration for SLO (Service Level Objective) monitoring
/// </summary>
public class SLOConfig
{
    /// <summary>
    /// Maximum allowed decision latency in milliseconds
    /// </summary>
    public double MaxDecisionLatencyMs { get; set; } = 100.0;
    
    /// <summary>
    /// Maximum allowed order execution latency in milliseconds
    /// </summary>
    public double MaxOrderLatencyMs { get; set; } = 500.0;
    
    /// <summary>
    /// P99 decision latency threshold in milliseconds
    /// </summary>
    public double DecisionLatencyP99Ms { get; set; } = 200.0;
    
    /// <summary>
    /// P99 end-to-end order latency threshold in milliseconds
    /// </summary>
    public double E2eOrderP99Ms { get; set; } = 1000.0;
    
    /// <summary>
    /// Daily error budget percentage (e.g., 1.0 = 1% allowed error rate per day)
    /// </summary>
    public double DailyErrorBudgetPct { get; set; } = 1.0;
    
    /// <summary>
    /// Error budget percentage (e.g., 0.01 = 1% allowed error rate)
    /// </summary>
    public double ErrorBudgetPercentage { get; set; } = 0.01;
    
    /// <summary>
    /// Time window for SLO calculations in minutes
    /// </summary>
    public int TimeWindowMinutes { get; set; } = 60;
    
    /// <summary>
    /// Minimum number of samples before SLO violations are reported
    /// </summary>
    public int MinSamplesThreshold { get; set; } = 10;
    
    /// <summary>
    /// Alert threshold for SLO violations (percentage of budget consumed)
    /// </summary>
    public double AlertThreshold { get; set; } = 0.8; // Alert when 80% of error budget is consumed
    
    /// <summary>
    /// Whether to enable automated remediation on SLO violations
    /// </summary>
    public bool EnableAutomatedRemediation { get; set; } = true;
    
    /// <summary>
    /// Maximum number of consecutive SLO violations before escalation
    /// </summary>
    public int MaxConsecutiveViolations { get; set; } = 3;
}