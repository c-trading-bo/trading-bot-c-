using System;
using System.Collections.Generic;

namespace Dashboard
{
    /// <summary>
    /// Captures a snapshot of system metrics for dashboard display
    /// </summary>
    public class MetricsSnapshot
    {
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        // Trading Metrics
        public int ActiveSignals { get; set; }
        public decimal TotalPnL { get; set; }
        public decimal DailyPnL { get; set; }
        public int TotalTrades { get; set; }
        public decimal WinRate { get; set; }
        public decimal AverageR { get; set; }
        
        // System Metrics
        public double CpuUsage { get; set; }
        public double MemoryUsage { get; set; }
        public int ActiveConnections { get; set; }
        public TimeSpan Uptime { get; set; }
        
        // Strategy Metrics
        public Dictionary<string, StrategyMetrics> StrategyPerformance { get; set; } = new();
        
        // Health Metrics
        public bool IsHealthy { get; set; } = true;
        public List<string> Warnings { get; set; } = new();
        public List<string> Errors { get; set; } = new();
    }

    public class StrategyMetrics
    {
        public string StrategyName { get; set; } = string.Empty;
        public decimal PnL { get; set; }
        public int TradeCount { get; set; }
        public decimal WinRate { get; set; }
        public decimal AverageR { get; set; }
        public bool IsActive { get; set; }
        public DateTime LastSignal { get; set; }
    }
}
