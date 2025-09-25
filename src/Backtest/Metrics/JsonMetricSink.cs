using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.Backtest.Metrics
{
    /// <summary>
    /// JSON-based metric storage implementation
    /// Captures all decisions, fills, and position closures to files
    /// Enables detailed post-backtest analysis and debugging
    /// </summary>
    public class JsonMetricSink : IMetricSink
    {
        private readonly ILogger<JsonMetricSink> _logger;
        private readonly string _storageDirectory;
        private readonly List<DecisionLog> _decisions;
        private readonly List<FillLog> _fills;
        private readonly List<PositionClosureLog> _closures;
        private readonly object _lockObject = new object();

        public JsonMetricSink(string storageDirectory, ILogger<JsonMetricSink> logger)
        {
            _logger = logger;
            _storageDirectory = storageDirectory;
            _decisions = new List<DecisionLog>();
            _fills = new List<FillLog>();
            _closures = new List<PositionClosureLog>();

            // Ensure storage directory exists
            Directory.CreateDirectory(_storageDirectory);
        }

        /// <summary>
        /// Record a trading decision made by the strategy
        /// </summary>
        public async Task RecordDecisionAsync(DecisionLog decision, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask; // Satisfy async requirement

            lock (_lockObject)
            {
                _decisions.Add(decision);
            }

            _logger.LogDebug("Recorded decision: {Decision} for {Symbol} at {Timestamp} (Confidence: {Confidence:P2})",
                decision.Decision, decision.Symbol, decision.Timestamp, decision.Confidence);
        }

        /// <summary>
        /// Record an order fill execution
        /// </summary>
        public async Task RecordFillAsync(FillLog fill, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask; // Satisfy async requirement

            lock (_lockObject)
            {
                _fills.Add(fill);
            }

            _logger.LogDebug("Recorded fill: {Side} {Quantity} {Symbol} at {Price:F4} (PnL: {PnL:F2})",
                fill.Side, fill.Quantity, fill.Symbol, fill.FillPrice, fill.RealizedPnL + fill.UnrealizedPnL);
        }

        /// <summary>
        /// Record a complete position closure (round-trip trade)
        /// </summary>
        public async Task RecordPositionClosureAsync(PositionClosureLog closure, CancellationToken cancellationToken = default)
        {
            await Task.CompletedTask; // Satisfy async requirement

            lock (_lockObject)
            {
                _closures.Add(closure);
            }

            _logger.LogDebug("Recorded position closure: {Side} {Quantity} {Symbol} - Net PnL: {NetPnL:F2} ({HoldTime})",
                closure.Side, closure.Quantity, closure.Symbol, closure.NetPnL, closure.HoldTime);
        }

        /// <summary>
        /// Flush all pending metrics to persistent storage
        /// </summary>
        public async Task FlushAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");

                // Save decisions
                await SaveToFileAsync($"decisions_{timestamp}.json", _decisions, cancellationToken);

                // Save fills  
                await SaveToFileAsync($"fills_{timestamp}.json", _fills, cancellationToken);

                // Save closures
                await SaveToFileAsync($"closures_{timestamp}.json", _closures, cancellationToken);

                // Create summary metrics
                var summary = CreateSummary();
                await SaveToFileAsync($"summary_{timestamp}.json", summary, cancellationToken);

                _logger.LogInformation("Flushed metrics to storage: {Decisions} decisions, {Fills} fills, {Closures} closures",
                    _decisions.Count, _fills.Count, _closures.Count);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to flush metrics to storage");
                throw;
            }
        }

        /// <summary>
        /// Get the storage location for generated metrics
        /// </summary>
        public string GetStoragePath()
        {
            return _storageDirectory;
        }

        private async Task SaveToFileAsync<T>(string filename, T data, CancellationToken cancellationToken)
        {
            var filePath = Path.Combine(_storageDirectory, filename);
            var options = new JsonSerializerOptions
            {
                WriteIndented = true,
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase
            };

            using var fileStream = File.Create(filePath);
            await JsonSerializer.SerializeAsync(fileStream, data, options, cancellationToken);

            _logger.LogDebug("Saved metrics to {FilePath}", filePath);
        }

        private object CreateSummary()
        {
            lock (_lockObject)
            {
                var totalPnL = 0m;
                var winningTrades = 0;
                var losingTrades = 0;
                var totalCommissions = 0m;

                foreach (var closure in _closures)
                {
                    totalPnL += closure.NetPnL;
                    totalCommissions += closure.Commission;

                    if (closure.NetPnL > 0)
                        winningTrades++;
                    else if (closure.NetPnL < 0)
                        losingTrades++;
                }

                var totalTrades = winningTrades + losingTrades;
                var winRate = totalTrades > 0 ? (decimal)winningTrades / totalTrades : 0m;

                return new
                {
                    GeneratedAt = DateTime.UtcNow,
                    TotalDecisions = _decisions.Count,
                    TotalFills = _fills.Count,
                    TotalRoundTripTrades = _closures.Count,
                    TotalPnL = totalPnL,
                    TotalCommissions = totalCommissions,
                    NetPnL = totalPnL - totalCommissions,
                    WinningTrades = winningTrades,
                    LosingTrades = losingTrades,
                    WinRate = winRate,
                    AverageTrade = totalTrades > 0 ? totalPnL / totalTrades : 0m,
                    ActionBreakdown = CalculateActionBreakdown(),
                    HourlyBreakdown = CalculateHourlyBreakdown(),
                    DailyBreakdown = CalculateDailyBreakdown()
                };
            }
        }

        private Dictionary<string, int> CalculateActionBreakdown()
        {
            var breakdown = new Dictionary<string, int>();
            
            foreach (var decision in _decisions)
            {
                var action = decision.Decision.ToString();
                breakdown[action] = breakdown.GetValueOrDefault(action, 0) + 1;
            }

            return breakdown;
        }

        private Dictionary<int, int> CalculateHourlyBreakdown()
        {
            var breakdown = new Dictionary<int, int>();
            
            foreach (var decision in _decisions)
            {
                var hour = decision.Timestamp.Hour;
                breakdown[hour] = breakdown.GetValueOrDefault(hour, 0) + 1;
            }

            return breakdown;
        }

        private Dictionary<string, int> CalculateDailyBreakdown()
        {
            var breakdown = new Dictionary<string, int>();
            
            foreach (var decision in _decisions)
            {
                var day = decision.Timestamp.ToString("yyyy-MM-dd");
                breakdown[day] = breakdown.GetValueOrDefault(day, 0) + 1;
            }

            return breakdown;
        }
    }
}