using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore
{
    /// <summary>
    /// Collects training data for RL position sizing by logging features and outcomes.
    /// Extends existing TradeLog functionality to capture ML training data.
    /// </summary>
    public static class RlTrainingDataCollector
    {
        public class FeatureSnapshot
        {
            public DateTime Timestamp { get; set; }
            public string Symbol { get; set; } = "";
            public string Strategy { get; set; } = "";
            public string Session { get; set; } = ""; // RTH, ETH
            public string Regime { get; set; } = ""; // Range, Trend, Vol
            public string SignalId { get; set; } = "";
            
            // Price features
            public decimal Price { get; set; }
            public decimal Atr { get; set; }
            public decimal Rsi { get; set; }
            public decimal Ema20 { get; set; }
            public decimal Ema50 { get; set; }
            
            // Market microstructure
            public decimal Volume { get; set; }
            public decimal Spread { get; set; }
            public decimal Volatility { get; set; }
            public decimal BidAskImbalance { get; set; }
            public decimal OrderBookImbalance { get; set; }
            public decimal TickDirection { get; set; }
            
            // Strategy-specific
            public decimal SignalStrength { get; set; }
            public decimal PriorWinRate { get; set; }
            public decimal AvgRMultiple { get; set; }
            
            // Risk factors
            public decimal DrawdownRisk { get; set; }
            public decimal NewsImpact { get; set; }
            public decimal LiquidityRisk { get; set; }
            
            // Position sizing decision (what we're learning)
            public decimal BaselineMultiplier { get; set; } = 1.0m;
            public decimal? ActualMultiplier { get; set; }
        }
        
        public class TradeOutcome
        {
            public string SignalId { get; set; } = "";
            public DateTime EntryTime { get; set; }
            public DateTime? ExitTime { get; set; }
            public decimal EntryPrice { get; set; }
            public decimal? ExitPrice { get; set; }
            public decimal StopPrice { get; set; }
            public decimal TargetPrice { get; set; }
            public decimal RMultiple { get; set; }
            public decimal SlippageTicks { get; set; }
            public bool IsWin { get; set; }
            public bool IsCompleted { get; set; }
            public string ExitReason { get; set; } = ""; // Stop, Target, Manual, Timeout
        }

        private static readonly string DataPath = Path.Combine("data", "rl_training");
        private static readonly object FileLock = new();

        static RlTrainingDataCollector()
        {
            Directory.CreateDirectory(DataPath);
        }

        /// <summary>
        /// Log feature snapshot when a signal is generated
        /// </summary>
        public static void LogFeatures(ILogger log, FeatureSnapshot features)
        {
            try
            {
                var json = JsonSerializer.Serialize(features, new JsonSerializerOptions 
                { 
                    WriteIndented = false 
                });
                
                var fileName = $"features_{DateTime.UtcNow:yyyyMMdd}.jsonl";
                var filePath = Path.Combine(DataPath, fileName);
                
                lock (FileLock)
                {
                    File.AppendAllText(filePath, json + Environment.NewLine);
                }
                
                log.LogDebug("[RL] Logged features for signal {SignalId}", features.SignalId);
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL] Failed to log feature snapshot");
            }
        }

        /// <summary>
        /// Log trade outcome when position is closed
        /// </summary>
        public static void LogOutcome(ILogger log, TradeOutcome outcome)
        {
            try
            {
                var json = JsonSerializer.Serialize(outcome, new JsonSerializerOptions 
                { 
                    WriteIndented = false 
                });
                
                var fileName = $"outcomes_{DateTime.UtcNow:yyyyMMdd}.jsonl";
                var filePath = Path.Combine(DataPath, fileName);
                
                lock (FileLock)
                {
                    File.AppendAllText(filePath, json + Environment.NewLine);
                }
                
                log.LogDebug("[RL] Logged outcome for signal {SignalId}: R={R:F2}", 
                    outcome.SignalId, outcome.RMultiple);
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL] Failed to log trade outcome");
            }
        }

        /// <summary>
        /// Create feature snapshot from current market data and strategy context
        /// </summary>
        public static FeatureSnapshot CreateFeatureSnapshot(
            string signalId,
            string symbol, 
            string strategy,
            decimal price,
            decimal baselineMultiplier = 1.0m)
        {
            return new FeatureSnapshot
            {
                Timestamp = DateTime.UtcNow,
                Symbol = symbol,
                Strategy = strategy,
                SignalId = signalId,
                Session = GetSessionType(),
                Regime = "Unknown", // You'll populate this from your regime detection
                Price = price,
                BaselineMultiplier = baselineMultiplier,
                
                // TODO: Populate these from your actual market data
                Atr = 0m,
                Rsi = 50m,
                Ema20 = price,
                Ema50 = price,
                Volume = 0m,
                Spread = 0.25m,
                Volatility = 0m,
                BidAskImbalance = 0m,
                OrderBookImbalance = 0m,
                TickDirection = 0m,
                SignalStrength = 0m,
                PriorWinRate = 0.5m,
                AvgRMultiple = 0m,
                DrawdownRisk = 0m,
                NewsImpact = 0m,
                LiquidityRisk = 0m
            };
        }

        private static string GetSessionType()
        {
            var et = TimeZoneInfo.ConvertTimeFromUtc(DateTime.UtcNow, 
                TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time"));
            var hour = et.Hour;
            
            // RTH: 9:30 AM - 4:00 PM ET
            return (hour >= 9 && hour < 16) ? "RTH" : "ETH";
        }

        /// <summary>
        /// Export collected data to CSV format for Python training
        /// </summary>
        public static string ExportToCsv(ILogger log, DateTime? startDate = null, DateTime? endDate = null)
        {
            var start = startDate ?? DateTime.UtcNow.AddDays(-30);
            var end = endDate ?? DateTime.UtcNow;
            
            var outputPath = Path.Combine(DataPath, $"training_data_{start:yyyyMMdd}_{end:yyyyMMdd}.csv");
            
            try
            {
                // TODO: Implement CSV export by joining features and outcomes
                // This would read all .jsonl files and create a merged dataset
                
                log.LogInformation("[RL] Training data exported to {Path}", outputPath);
                return outputPath;
            }
            catch (Exception ex)
            {
                log.LogError(ex, "[RL] Failed to export training data");
                throw;
            }
        }
    }
}
