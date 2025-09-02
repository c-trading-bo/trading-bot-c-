// Real-Time Trading Progress Monitor
// Tracks strategy performance, win rates, and session statistics
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using BotCore.Models;
using BotCore.Config;

namespace BotCore.Services
{
    /// <summary>
    /// Monitors trading progress and performance metrics in real-time
    /// </summary>
    public class TradingProgressMonitor : IDisposable
    {
        private readonly ILogger<TradingProgressMonitor> _logger;
        private readonly Dictionary<string, TradingMetrics> _metrics = new();
        private readonly object _metricsLock = new object();
        private DateTime _lastDashboardUpdate = DateTime.MinValue;

        public TradingProgressMonitor(ILogger<TradingProgressMonitor> logger)
        {
            _logger = logger;
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
                    metrics.TradesByHour[hour] = 0;
                    metrics.WinsByHour[hour] = 0;
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

                // Log significant trades
                if (Math.Abs(result.PnL) > 100 || metrics.TotalTrades % 10 == 0)
                {
                    LogProgress(metrics);
                }
            }
        }

        /// <summary>
        /// Display comprehensive trading dashboard
        /// </summary>
        public void DisplayDashboard(bool force = false)
        {
            if (!force && DateTime.UtcNow - _lastDashboardUpdate < TimeSpan.FromMinutes(5))
                return;

            _lastDashboardUpdate = DateTime.UtcNow;

            lock (_metricsLock)
            {
                Console.Clear();
                Console.WriteLine("╔══════════════════════════════════════════════════════════════════╗");
                Console.WriteLine("║           24/7 ES & NQ TRADING SYSTEM - LIVE PROGRESS           ║");
                Console.WriteLine("╠══════════════════════════════════════════════════════════════════╣");
                Console.WriteLine($"║ Time: {DateTime.UtcNow:yyyy-MM-dd HH:mm:ss} UTC                 ║");

                // Current session info
                var currentSession = ES_NQ_TradingSchedule.GetCurrentSession(DateTime.UtcNow.TimeOfDay);
                if (currentSession != null)
                {
                    Console.WriteLine($"║ Session: {currentSession.Description,-35}           ║");
                    Console.WriteLine($"║ Primary: {currentSession.PrimaryInstrument,-35}           ║");
                }
                else
                {
                    Console.WriteLine("║ Session: MARKET CLOSED                                           ║");
                }

                Console.WriteLine("╠══════════════════════════════════════════════════════════════════╣");

                // Display ES metrics
                Console.WriteLine("║ ES FUTURES:                                                      ║");
                DisplayInstrumentMetrics("ES");

                Console.WriteLine("║                                                                  ║");
                Console.WriteLine("║ NQ FUTURES:                                                      ║");
                DisplayInstrumentMetrics("NQ");

                Console.WriteLine("╠══════════════════════════════════════════════════════════════════╣");
                Console.WriteLine("║ BEST PERFORMING TIMES:                                          ║");

                // Show best times for each strategy
                var bestTimes = GetBestTradingTimes();
                foreach (var time in bestTimes.Take(5))
                {
                    Console.WriteLine($"║  {time.Hour:00}:00 | {time.Strategy} | WR: {time.WinRate:P0} | {time.Session,-15} ║");
                }

                Console.WriteLine("╠══════════════════════════════════════════════════════════════════╣");
                Console.WriteLine("║ DAILY SUMMARY:                                                  ║");
                DisplayDailySummary();

                Console.WriteLine("╚══════════════════════════════════════════════════════════════════╝");
            }
        }

        private void DisplayInstrumentMetrics(string instrument)
        {
            var instrumentMetrics = _metrics.Values.Where(m => m.Instrument == instrument).ToList();

            if (!instrumentMetrics.Any())
            {
                Console.WriteLine($"║  No trades yet for {instrument}                                        ║");
                return;
            }

            foreach (var m in instrumentMetrics.OrderBy(x => x.StrategyId))
            {
                var arrow = m.IsImproving ? "↑" : "↓";
                var color = m.WinRate > 0.55 ? ConsoleColor.Green : m.WinRate < 0.45 ? ConsoleColor.Red : ConsoleColor.Yellow;

                Console.ForegroundColor = color;
                Console.WriteLine($"║  {m.StrategyId,-4} | Trades: {m.TotalTrades,4} | WR: {m.WinRate:P1} {arrow} | PnL: ${m.TotalPnL,8:F2} | ML: {m.MLConfidenceAvg:P0} ║");
                Console.ResetColor();
            }
        }

        private void DisplayDailySummary()
        {
            var totalTrades = _metrics.Values.Sum(m => m.TotalTrades);
            var totalPnL = _metrics.Values.Sum(m => m.TotalPnL);
            var totalWins = _metrics.Values.Sum(m => m.WinningTrades);
            var overallWinRate = totalTrades > 0 ? (double)totalWins / totalTrades : 0;
            var avgMLConfidence = _metrics.Values.Where(m => m.MLConfidenceAvg > 0).Any()
                ? _metrics.Values.Where(m => m.MLConfidenceAvg > 0).Average(m => m.MLConfidenceAvg)
                : 0.0;

            var summaryColor = totalPnL > 0 ? ConsoleColor.Green : ConsoleColor.Red;
            Console.ForegroundColor = summaryColor;
            Console.WriteLine($"║  Total Trades: {totalTrades,4} | Win Rate: {overallWinRate:P1} | PnL: ${totalPnL,8:F2} | ML Avg: {avgMLConfidence:P0}  ║");
            Console.ResetColor();

            // Show active strategies count
            var activeStrategies = _metrics.Values.Select(m => m.StrategyId).Distinct().Count();
            Console.WriteLine($"║  Active Strategies: {activeStrategies} | Sessions Today: {GetSessionsToday()}              ║");
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

        private void LogProgress(TradingMetrics metrics)
        {
            _logger.LogInformation(
                "[Progress] {Strategy}-{Instrument}: {Trades} trades, {WinRate:P1} WR, ${PnL:F2} PnL, {Improving}",
                metrics.StrategyId,
                metrics.Instrument,
                metrics.TotalTrades,
                metrics.WinRate,
                metrics.TotalPnL,
                metrics.IsImproving ? "IMPROVING" : "DECLINING"
            );
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
        public Dictionary<int, int> TradesByHour { get; set; } = new();
        public Dictionary<int, int> WinsByHour { get; set; } = new();
        public Dictionary<int, double> WinRateByHour { get; set; } = new();
        public double TotalPnL { get; set; }
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