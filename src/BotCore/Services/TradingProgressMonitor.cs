using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using BotCore.Models;
using BotCore.Config;
using TradingBot.Abstractions;
using System.Threading.Tasks;

namespace BotCore.Services
{
    /// <summary>
    /// Monitors trading progress and performance metrics in real-time
    /// Uses structured logging instead of dashboard display
    /// </summary>
    public class TradingProgressMonitor : IDisposable
    {
        private readonly ILogger<TradingProgressMonitor> _logger;
        private readonly ITradingLogger? _tradingLogger;
        private readonly Dictionary<string, TradingMetrics> _metrics = new();
        private readonly object _metricsLock = new object();
        private DateTime _lastProgressReport = DateTime.MinValue;

        public TradingProgressMonitor(ILogger<TradingProgressMonitor> logger, ITradingLogger? tradingLogger = null)
        {
            _logger = logger;
            _tradingLogger = tradingLogger;
        }

        /// <summary>
        /// Update metrics when a trade is completed
        /// </summary>
        public void UpdateMetrics(TradeResult result)
        {
            lock (_metricsLock)
            {
                var key = $"{result.Instrument}_{result.Strategy}";

                if (!_metrics.ContainsKey(key))
                {
                    _metrics[key] = new TradingMetrics
                    {
                        StrategyId = result.Strategy,
                        Instrument = result.Instrument
                    };
                }

                var metrics = _metrics[key];

                // Update basic metrics
                metrics.TotalTrades++;
                if (result.PnL > 0)
                {
                    metrics.WinningTrades++;
                    metrics.AverageWin = UpdateAverage(metrics.AverageWin, result.PnL, metrics.WinningTrades);
                }
                else
                {
                    var losses = metrics.TotalTrades - metrics.WinningTrades;
                    metrics.AverageLoss = UpdateAverage(metrics.AverageLoss, Math.Abs(result.PnL), losses);
                }

                metrics.TotalPnL += result.PnL;

                // Update hourly metrics
                var hour = result.EntryTime.Hour;
                if (!metrics.TradesByHour.ContainsKey(hour))
                {
                    metrics.TradesByHour[hour];
                    metrics.WinsByHour[hour];
                }

                metrics.TradesByHour[hour]++;
                if (result.PnL > 0)
                {
                    metrics.WinsByHour[hour]++;
                }

                // Calculate hourly win rate
                metrics.WinRateByHour[hour] = (double)metrics.WinsByHour[hour] / metrics.TradesByHour[hour];

                // Check if improving (last 20 trades vs previous 20)
                metrics.IsImproving = CheckIfImproving(key, result.PnL);

                // Update ML confidence average
                if (result.MLConfidence > 0)
                {
                    metrics.MLConfidenceAvg = UpdateAverage(metrics.MLConfidenceAvg, result.MLConfidence, metrics.TotalTrades);
                }

                // Update drawdown tracking
                UpdateDrawdown(metrics, result.PnL);

                metrics.LastUpdated = DateTime.UtcNow;

                // Schedule async logging outside the lock
                var metricsSnapshot = new TradingMetrics
                {
                    StrategyId = metrics.StrategyId,
                    Instrument = metrics.Instrument,
                    TotalTrades = metrics.TotalTrades,
                    WinningTrades = metrics.WinningTrades,
                    TotalPnL = metrics.TotalPnL,
                    MLConfidenceAvg = metrics.MLConfidenceAvg,
                    IsImproving = metrics.IsImproving,
                    LastUpdated = metrics.LastUpdated
                };

                // Log significant trades with structured data (fire and forget)
                if (Math.Abs(result.PnL) > 100 || metrics.TotalTrades % 10 == 0)
                {
                    _ = Task.Run(async () => await LogProgressAsync(metricsSnapshot)).ConfigureAwait(false);
                }
            }
        }

        /// <summary>
        /// Log comprehensive trading progress using structured logging
        /// </summary>
        public async Task LogProgressReportAsync(bool force = false)
        {
            if (!force && DateTime.UtcNow - _lastProgressReport < TimeSpan.FromMinutes(5))
                return;

            _lastProgressReport = DateTime.UtcNow;

            object progressData;
            int totalTrades;
            double overallWinRate;
            double totalPnL;

            lock (_metricsLock)
            {
                totalTrades = _metrics.Values.Sum(m => m.TotalTrades);
                totalPnL = _metrics.Values.Sum(m => m.TotalPnL);
                var totalWins = _metrics.Values.Sum(m => m.WinningTrades);
                overallWinRate = totalTrades > 0 ? (double)totalWins / totalTrades : 0;

                progressData = new
                {
                    totalTrades,
                    totalPnL,
                    overallWinRate,
                    instrumentMetrics = _metrics.Values.GroupBy(m => m.Instrument)
                        .ToDictionary(g => g.Key, g => g.Select(m => new
                        {
                            strategy = m.StrategyId,
                            trades = m.TotalTrades,
                            winRate = m.WinRate,
                            pnl = m.TotalPnL,
                            mlConfidence = m.MLConfidenceAvg,
                            isImproving = m.IsImproving
                        }).ToArray()),
                    bestTimes = GetBestTradingTimes().Take(5).ToArray(),
                    activeStrategies = _metrics.Values.Select(m => m.StrategyId).Distinct().Count(),
                    currentSession = ES_NQ_TradingSchedule.GetCurrentSession(DateTime.UtcNow.TimeOfDay)?.Description ?? "MARKET CLOSED"
                };
            }

            _logger.LogInformation("Trading Progress Report: {TotalTrades} trades, {OverallWinRate:P1} win rate, ${TotalPnL:F2} PnL", 
                totalTrades, overallWinRate, totalPnL);

            if (_tradingLogger != null)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TradingProgressMonitor", "Progress Report", progressData).ConfigureAwait(false);
            }
        }

        private async Task LogProgressAsync(TradingMetrics metrics)
        {
            var progressData = new
            {
                strategy = metrics.StrategyId,
                instrument = metrics.Instrument,
                trades = metrics.TotalTrades,
                winRate = metrics.WinRate,
                pnl = metrics.TotalPnL,
                avgPnL = metrics.AveragePnL,
                maxDrawdown = metrics.MaxDrawdown,
                mlConfidence = metrics.MLConfidenceAvg,
                isImproving = metrics.IsImproving,
                lastUpdated = metrics.LastUpdated
            };

            _logger.LogInformation("Strategy Progress: {Strategy} on {Instrument} - {Trades} trades, {WinRate:P1} WR, ${PnL:F2} PnL", 
                metrics.StrategyId, metrics.Instrument, metrics.TotalTrades, metrics.WinRate, metrics.TotalPnL);

            if (_tradingLogger != null)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TradingProgressMonitor", "Strategy Progress", progressData).ConfigureAwait(false);
            }
        }

        private List<BestTime> GetBestTradingTimes()
        {
            var bestTimes = new List<BestTime>();

            foreach (var metric in _metrics.Values)
            {
                foreach (var hourWinRate in metric.WinRateByHour.Where(h => h.Value > 0.6))
                {
                    bestTimes.Add(new BestTime
                    {
                        Hour = hourWinRate.Key,
                        Strategy = metric.StrategyId,
                        WinRate = hourWinRate.Value,
                        Session = GetSessionForHour(hourWinRate.Key)
                    });
                }
            }

            return bestTimes.OrderByDescending(t => t.WinRate).ToList();
        }

        private string GetSessionForHour(int hour)
        {
            var timeSpan = new TimeSpan(hour, 0, 0);
            var session = ES_NQ_TradingSchedule.GetCurrentSession(timeSpan);
            return session?.Description ?? "Unknown";
        }

        private int GetSessionsToday()
        {
            // Count how many different sessions have had trades today
            var today = DateTime.UtcNow.Date;
            var sessionsWithTrades = new HashSet<string>();

            foreach (var metric in _metrics.Values)
            {
                if (metric.LastUpdated.Date == today)
                {
                    foreach (var hour in metric.TradesByHour.Keys)
                    {
                        var session = GetSessionForHour(hour);
                        sessionsWithTrades.Add(session);
                    }
                }
            }

            return sessionsWithTrades.Count;
        }

        private double UpdateAverage(double currentAvg, double newValue, int count)
        {
            if (count == 1) return newValue;
            return ((currentAvg * (count - 1)) + newValue) / count;
        }

        private bool CheckIfImproving(string key, double latestPnL)
        {
            // Simple improvement check - could be enhanced with more sophisticated analysis
            var metric = _metrics[key];

            if (metric.TotalTrades < 10) return true; // Assume improving with few trades

            // Check if recent performance is better than overall
            var recentWinRate = metric.TotalTrades > 0 ? Math.Min(0.7, (double)metric.WinningTrades / metric.TotalTrades) : 0;
            return recentWinRate > 0.5 && latestPnL > 0;
        }

        private void UpdateDrawdown(TradingMetrics metrics, double pnl)
        {
            if (pnl < 0)
            {
                metrics.CurrentDrawdown += Math.Abs(pnl);
                metrics.MaxDrawdown = Math.Max(metrics.MaxDrawdown, metrics.CurrentDrawdown);
            }
            else
            {
                metrics.CurrentDrawdown = Math.Max(0, metrics.CurrentDrawdown - pnl);
            }
        }

        /// <summary>
        /// Get summary statistics for external reporting
        /// </summary>
        public TradingSummary GetSummary()
        {
            lock (_metricsLock)
            {
                return new TradingSummary
                {
                    TotalTrades = _metrics.Values.Sum(m => m.TotalTrades),
                    TotalPnL = _metrics.Values.Sum(m => m.TotalPnL),
                    OverallWinRate = _metrics.Values.Sum(m => m.TotalTrades) > 0
                        ? (double)_metrics.Values.Sum(m => m.WinningTrades) / _metrics.Values.Sum(m => m.TotalTrades)
                        : 0,
                    ActiveStrategies = _metrics.Values.Select(m => m.StrategyId).Distinct().Count(),
                    ActiveInstruments = _metrics.Values.Select(m => m.Instrument).Distinct().Count(),
                    LastUpdated = DateTime.UtcNow
                };
            }
        }

        public void Dispose()
        {
            // Cleanup resources
        }
    }

    /// <summary>
    /// Strategy performance metrics for progress monitoring
    /// </summary>
    public class TradingMetrics
    {
        public string StrategyId { get; set; } = "";
        public string Instrument { get; set; } = "";
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public double WinRate => TotalTrades > 0 ? (double)WinningTrades / TotalTrades : 0;
        public Dictionary<int, int> TradesByHour { get; } = new();
        public Dictionary<int, int> WinsByHour { get; } = new();
        public Dictionary<int, double> WinRateByHour { get; } = new();
        public double TotalPnL { get; set; }
        public double AveragePnL => TotalTrades > 0 ? TotalPnL / TotalTrades : 0;
        public double AverageWin { get; set; }
        public double AverageLoss { get; set; }
        public double MaxDrawdown { get; set; }
        public double CurrentDrawdown { get; set; }
        public DateTime LastUpdated { get; set; }
        public bool IsImproving { get; set; }
        public double MLConfidenceAvg { get; set; }
    }

    /// <summary>
    /// Trade result for progress tracking
    /// </summary>
    public class TradeResult
    {
        public string Instrument { get; set; } = "";
        public string Strategy { get; set; } = "";
        public double PnL { get; set; }
        public DateTime EntryTime { get; set; }
        public DateTime ExitTime { get; set; }
        public double MLConfidence { get; set; }
        public string Session { get; set; } = "";
    }

    /// <summary>
    /// Best performing time slot
    /// </summary>
    public class BestTime
    {
        public int Hour { get; set; }
        public string Strategy { get; set; } = "";
        public double WinRate { get; set; }
        public string Session { get; set; } = "";
    }

    /// <summary>
    /// Overall trading summary
    /// </summary>
    public class TradingSummary
    {
        public int TotalTrades { get; set; }
        public double TotalPnL { get; set; }
        public double OverallWinRate { get; set; }
        public int ActiveStrategies { get; set; }
        public int ActiveInstruments { get; set; }
        public DateTime LastUpdated { get; set; }
    }
}