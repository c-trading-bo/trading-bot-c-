using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// 📈 AUTONOMOUS PERFORMANCE TRACKER 📈
/// 
/// Tracks and analyzes trading performance to enable continuous learning
/// and optimization of the autonomous trading engine. Provides real-time
/// performance metrics, learning insights, and optimization recommendations.
/// 
/// KEY FEATURES:
/// ✅ Real-time P&L tracking and analysis
/// ✅ Win rate and performance metric calculation
/// ✅ Strategy-specific performance analysis
/// ✅ Time-based performance patterns
/// ✅ Risk-adjusted performance metrics
/// ✅ Continuous learning from trade outcomes
/// ✅ Performance-based parameter optimization
/// 
/// This enables the autonomous engine to:
/// - Learn from every trade outcome
/// - Optimize strategy selection based on performance
/// - Adjust risk parameters dynamically
/// - Identify best trading times and conditions
/// - Generate actionable performance insights
/// </summary>
public class AutonomousPerformanceTracker
{
    private readonly ILogger<AutonomousPerformanceTracker> _logger;
    
    // Performance tracking collections
    private readonly List<TradeOutcome> _allTrades = new();
    private readonly Dictionary<string, List<TradeOutcome>> _tradesByStrategy = new();
    private readonly Dictionary<string, List<TradeOutcome>> _tradesBySymbol = new();
    private readonly Dictionary<DateTime, decimal> _dailyPnL = new();
    private readonly Queue<PerformanceSnapshot> _performanceHistory = new();
    private readonly object _trackingLock = new();
    
    // Current performance state
    private decimal _totalPnL = 0m;
    private decimal _todayPnL = 0m;
    private decimal _weekPnL = 0m;
    private decimal _monthPnL = 0m;
    private int _totalTrades = 0;
    private int _winningTrades = 0;
    private int _losingTrades = 0;
    private decimal _largestWin = 0m;
    private decimal _largestLoss = 0m;
    private decimal _avgWin = 0m;
    private decimal _avgLoss = 0m;
    private decimal _winRate = 0m;
    private decimal _profitFactor = 0m;
    private decimal _sharpeRatio = 0m;
    private decimal _maxDrawdown = 0m;
    private DateTime _lastUpdateTime = DateTime.MinValue;
    
    // Learning and optimization
    private readonly Dictionary<string, StrategyLearning> _strategyLearning = new();
    private readonly Queue<OptimizationInsight> _optimizationInsights = new();
    
    public AutonomousPerformanceTracker(ILogger<AutonomousPerformanceTracker> logger)
    {
        _logger = logger;
        
        InitializeStrategyLearning();
        
        _logger.LogInformation("📈 [PERFORMANCE-TRACKER] Initialized - Real-time performance tracking and learning ready");
    }
    
    /// <summary>
    /// Record a new trade outcome for analysis
    /// </summary>
    public async Task RecordTradeAsync(TradeOutcome trade, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        lock (_trackingLock)
        {
            // Add to collections
            _allTrades.Add(trade);
            
            // Organize by strategy
            if (!_tradesByStrategy.ContainsKey(trade.Strategy))
            {
                _tradesByStrategy[trade.Strategy] = new List<TradeOutcome>();
            }
            _tradesByStrategy[trade.Strategy].Add(trade);
            
            // Organize by symbol
            if (!_tradesBySymbol.ContainsKey(trade.Symbol))
            {
                _tradesBySymbol[trade.Symbol] = new List<TradeOutcome>();
            }
            _tradesBySymbol[trade.Symbol].Add(trade);
            
            // Update daily P&L tracking
            var tradeDate = trade.EntryTime.Date;
            if (!_dailyPnL.ContainsKey(tradeDate))
            {
                _dailyPnL[tradeDate] = 0m;
            }
            _dailyPnL[tradeDate] += trade.PnL;
            
            // Update performance metrics
            UpdatePerformanceMetrics();
            
            // Record learning insights
            await RecordLearningInsightAsync(trade, cancellationToken);
            
            _logger.LogDebug("📊 [PERFORMANCE-TRACKER] Trade recorded: {Strategy} {Symbol} ${PnL:F2} (Total: {Trades} trades, ${TotalPnL:F2})",
                trade.Strategy, trade.Symbol, trade.PnL, _totalTrades, _totalPnL);
        }
    }
    
    /// <summary>
    /// Update performance metrics from current trade data
    /// </summary>
    public async Task UpdateMetricsAsync(TradeOutcome[] recentTrades, CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        lock (_trackingLock)
        {
            foreach (var trade in recentTrades)
            {
                if (!_allTrades.Any(t => t.EntryTime == trade.EntryTime && t.Strategy == trade.Strategy))
                {
                    _allTrades.Add(trade);
                }
            }
            
            UpdatePerformanceMetrics();
            _lastUpdateTime = DateTime.UtcNow;
        }
    }
    
    /// <summary>
    /// Get recent P&L for specified time period
    /// </summary>
    public decimal GetRecentPnL(TimeSpan period)
    {
        lock (_trackingLock)
        {
            var cutoffTime = DateTime.UtcNow - period;
            return _allTrades
                .Where(t => t.EntryTime >= cutoffTime)
                .Sum(t => t.PnL);
        }
    }
    
    /// <summary>
    /// Get recent win rate for specified time period
    /// </summary>
    public decimal GetRecentWinRate(TimeSpan period)
    {
        lock (_trackingLock)
        {
            var recentTrades = _allTrades
                .Where(t => t.EntryTime >= DateTime.UtcNow - period)
                .ToArray();
            
            if (recentTrades.Length == 0) return 0.5m; // Default 50% if no trades
            
            var wins = recentTrades.Count(t => t.PnL > 0);
            return (decimal)wins / recentTrades.Length;
        }
    }
    
    /// <summary>
    /// Get performance metrics for specific strategy
    /// </summary>
    public StrategyPerformance GetStrategyPerformance(string strategy)
    {
        lock (_trackingLock)
        {
            if (!_tradesByStrategy.ContainsKey(strategy))
            {
                return new StrategyPerformance { StrategyName = strategy };
            }
            
            var trades = _tradesByStrategy[strategy];
            var winningTrades = trades.Where(t => t.PnL > 0).ToArray();
            var losingTrades = trades.Where(t => t.PnL < 0).ToArray();
            
            return new StrategyPerformance
            {
                StrategyName = strategy,
                TotalTrades = trades.Count,
                WinningTrades = winningTrades.Length,
                LosingTrades = losingTrades.Length,
                TotalPnL = trades.Sum(t => t.PnL),
                WinRate = trades.Count > 0 ? (decimal)winningTrades.Length / trades.Count : 0m,
                AvgWin = winningTrades.Length > 0 ? winningTrades.Average(t => t.PnL) : 0m,
                AvgLoss = losingTrades.Length > 0 ? losingTrades.Average(t => t.PnL) : 0m,
                LargestWin = winningTrades.Length > 0 ? winningTrades.Max(t => t.PnL) : 0m,
                LargestLoss = losingTrades.Length > 0 ? losingTrades.Min(t => t.PnL) : 0m,
                ProfitFactor = CalculateProfitFactor(winningTrades, losingTrades),
                RecentPerformance = GetRecentStrategyPerformance(strategy, TimeSpan.FromDays(7))
            };
        }
    }
    
    /// <summary>
    /// Get current performance summary
    /// </summary>
    public AutonomousPerformanceSummary GetCurrentPerformance()
    {
        lock (_trackingLock)
        {
            return new AutonomousPerformanceSummary
            {
                TotalPnL = _totalPnL,
                TodayPnL = _todayPnL,
                WeekPnL = _weekPnL,
                MonthPnL = _monthPnL,
                TotalTrades = _totalTrades,
                WinningTrades = _winningTrades,
                LosingTrades = _losingTrades,
                WinRate = _winRate,
                AvgWin = _avgWin,
                AvgLoss = _avgLoss,
                LargestWin = _largestWin,
                LargestLoss = _largestLoss,
                ProfitFactor = _profitFactor,
                SharpeRatio = _sharpeRatio,
                MaxDrawdown = _maxDrawdown,
                LastUpdateTime = _lastUpdateTime,
                BestStrategy = GetBestPerformingStrategy(),
                WorstStrategy = GetWorstPerformingStrategy()
            };
        }
    }
    
    /// <summary>
    /// Generate daily performance report
    /// </summary>
    public async Task<DailyPerformanceReport> GenerateDailyReportAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask;
        
        lock (_trackingLock)
        {
            var today = DateTime.Today;
            var todayTrades = _allTrades.Where(t => t.EntryTime.Date == today).ToArray();
            
            var report = new DailyPerformanceReport
            {
                Date = today,
                DailyPnL = todayTrades.Sum(t => t.PnL),
                TotalTrades = todayTrades.Length,
                WinningTrades = todayTrades.Count(t => t.PnL > 0),
                LosingTrades = todayTrades.Count(t => t.PnL < 0),
                WinRate = todayTrades.Length > 0 ? (decimal)todayTrades.Count(t => t.PnL > 0) / todayTrades.Length : 0m,
                LargestWin = todayTrades.Length > 0 ? todayTrades.Where(t => t.PnL > 0).DefaultIfEmpty().Max(t => t?.PnL ?? 0m) : 0m,
                LargestLoss = todayTrades.Length > 0 ? todayTrades.Where(t => t.PnL < 0).DefaultIfEmpty().Min(t => t?.PnL ?? 0m) : 0m,
                BestStrategy = GetBestPerformingStrategyForDay(today),
                WorstStrategy = GetWorstPerformingStrategyForDay(today),
                TradingInsights = await GenerateTradingInsightsAsync(todayTrades, cancellationToken),
                OptimizationRecommendations = GenerateOptimizationRecommendations()
            };
            
            return report;
        }
    }
    
    /// <summary>
    /// Get learning insights for strategy optimization
    /// </summary>
    public List<LearningInsight> GetLearningInsights(string strategy = "")
    {
        lock (_trackingLock)
        {
            var insights = new List<LearningInsight>();
            
            if (string.IsNullOrEmpty(strategy))
            {
                // Get insights for all strategies
                foreach (var strategyLearning in _strategyLearning.Values)
                {
                    insights.AddRange(strategyLearning.Insights);
                }
            }
            else if (_strategyLearning.ContainsKey(strategy))
            {
                insights.AddRange(_strategyLearning[strategy].Insights);
            }
            
            return insights.OrderByDescending(i => i.Timestamp).Take(20).ToList();
        }
    }
    
    /// <summary>
    /// Get performance-based recommendations for parameter optimization
    /// </summary>
    public List<OptimizationRecommendation> GetOptimizationRecommendations()
    {
        lock (_trackingLock)
        {
            var recommendations = new List<OptimizationRecommendation>();
            
            // Analyze strategy performance
            foreach (var strategy in _tradesByStrategy.Keys)
            {
                var performance = GetStrategyPerformance(strategy);
                var strategyRecommendations = AnalyzeStrategyForOptimization(performance);
                recommendations.AddRange(strategyRecommendations);
            }
            
            // Analyze time-based patterns
            var timeRecommendations = AnalyzeTimeBasedPatterns();
            recommendations.AddRange(timeRecommendations);
            
            // Analyze risk management
            var riskRecommendations = AnalyzeRiskManagement();
            recommendations.AddRange(riskRecommendations);
            
            return recommendations.OrderByDescending(r => r.Priority).Take(10).ToList();
        }
    }
    
    private void UpdatePerformanceMetrics()
    {
        if (_allTrades.Count == 0) return;
        
        // Basic metrics
        _totalTrades = _allTrades.Count;
        _totalPnL = _allTrades.Sum(t => t.PnL);
        
        var winningTrades = _allTrades.Where(t => t.PnL > 0).ToArray();
        var losingTrades = _allTrades.Where(t => t.PnL < 0).ToArray();
        
        _winningTrades = winningTrades.Length;
        _losingTrades = losingTrades.Length;
        _winRate = _totalTrades > 0 ? (decimal)_winningTrades / _totalTrades : 0m;
        
        _avgWin = _winningTrades > 0 ? winningTrades.Average(t => t.PnL) : 0m;
        _avgLoss = _losingTrades > 0 ? losingTrades.Average(t => t.PnL) : 0m;
        
        _largestWin = _winningTrades > 0 ? winningTrades.Max(t => t.PnL) : 0m;
        _largestLoss = _losingTrades > 0 ? losingTrades.Min(t => t.PnL) : 0m;
        
        _profitFactor = CalculateProfitFactor(winningTrades, losingTrades);
        
        // Time-based P&L
        var today = DateTime.Today;
        var weekStart = today.AddDays(-(int)today.DayOfWeek);
        var monthStart = new DateTime(today.Year, today.Month, 1);
        
        _todayPnL = _allTrades.Where(t => t.EntryTime.Date == today).Sum(t => t.PnL);
        _weekPnL = _allTrades.Where(t => t.EntryTime.Date >= weekStart).Sum(t => t.PnL);
        _monthPnL = _allTrades.Where(t => t.EntryTime.Date >= monthStart).Sum(t => t.PnL);
        
        // Advanced metrics
        _sharpeRatio = CalculateSharpeRatio();
        _maxDrawdown = CalculateMaxDrawdown();
        
        // Save performance snapshot
        _performanceHistory.Enqueue(new PerformanceSnapshot
        {
            Timestamp = DateTime.UtcNow,
            TotalPnL = _totalPnL,
            TotalTrades = _totalTrades,
            WinRate = _winRate,
            SharpeRatio = _sharpeRatio
        });
        
        // Keep limited history
        while (_performanceHistory.Count > 1000)
        {
            _performanceHistory.Dequeue();
        }
    }
    
    private decimal CalculateProfitFactor(TradeOutcome[] winningTrades, TradeOutcome[] losingTrades)
    {
        var totalWins = winningTrades.Sum(t => t.PnL);
        var totalLosses = Math.Abs(losingTrades.Sum(t => t.PnL));
        
        return totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? 99m : 0m;
    }
    
    private decimal CalculateSharpeRatio()
    {
        if (_allTrades.Count < 30) return 0m; // Need minimum trades for reliable calculation
        
        var returns = _allTrades.Select(t => t.PnL).ToArray();
        var avgReturn = returns.Average();
        var stdDev = CalculateStandardDeviation(returns);
        
        return stdDev > 0 ? avgReturn / stdDev * (decimal)Math.Sqrt(252) : 0m; // Annualized
    }
    
    private decimal CalculateMaxDrawdown()
    {
        if (_allTrades.Count == 0) return 0m;
        
        var runningPnL = 0m;
        var peak = 0m;
        var maxDrawdown = 0m;
        
        foreach (var trade in _allTrades.OrderBy(t => t.EntryTime))
        {
            runningPnL += trade.PnL;
            if (runningPnL > peak) peak = runningPnL;
            
            var drawdown = peak - runningPnL;
            if (drawdown > maxDrawdown) maxDrawdown = drawdown;
        }
        
        return maxDrawdown;
    }
    
    private decimal CalculateStandardDeviation(decimal[] values)
    {
        if (values.Length <= 1) return 0m;
        
        var average = values.Average();
        var sumOfSquares = values.Sum(val => Math.Pow((double)(val - average), 2));
        var variance = sumOfSquares / (values.Length - 1);
        
        return (decimal)Math.Sqrt(variance);
    }
    
    private string GetBestPerformingStrategy()
    {
        if (_tradesByStrategy.Count == 0) return "N/A";
        
        return _tradesByStrategy
            .OrderByDescending(kvp => kvp.Value.Sum(t => t.PnL))
            .First().Key;
    }
    
    private string GetWorstPerformingStrategy()
    {
        if (_tradesByStrategy.Count == 0) return "N/A";
        
        return _tradesByStrategy
            .OrderBy(kvp => kvp.Value.Sum(t => t.PnL))
            .First().Key;
    }
    
    private void InitializeStrategyLearning()
    {
        var strategies = new[] { "S2", "S3", "S6", "S11" };
        foreach (var strategy in strategies)
        {
            _strategyLearning[strategy] = new StrategyLearning
            {
                StrategyName = strategy,
                Insights = new List<LearningInsight>()
            };
        }
    }
    
    private async Task RecordLearningInsightAsync(TradeOutcome trade, CancellationToken cancellationToken)
    {
        // Generate learning insights from trade outcome
        var insight = new LearningInsight
        {
            Timestamp = DateTime.UtcNow,
            Strategy = trade.Strategy,
            TradeOutcome = trade,
            InsightType = trade.PnL > 0 ? "SUCCESS_PATTERN" : "LOSS_PATTERN",
            Description = $"{trade.Strategy} {trade.Direction} {trade.Symbol} resulted in ${trade.PnL:F2}",
            Confidence = trade.Confidence,
            MarketConditions = new Dictionary<string, object>
            {
                ["Regime"] = trade.MarketRegime.ToString(),
                ["EntryTime"] = trade.EntryTime.ToString("HH:mm"),
                ["Direction"] = trade.Direction
            }
        };
        
        if (_strategyLearning.ContainsKey(trade.Strategy))
        {
            _strategyLearning[trade.Strategy].Insights.Add(insight);
            
            // Keep limited insights per strategy
            while (_strategyLearning[trade.Strategy].Insights.Count > 100)
            {
                _strategyLearning[trade.Strategy].Insights.RemoveAt(0);
            }
        }
    }
    
    private decimal GetRecentStrategyPerformance(string strategy, TimeSpan period)
    {
        if (!_tradesByStrategy.ContainsKey(strategy)) return 0m;
        
        var cutoffTime = DateTime.UtcNow - period;
        return _tradesByStrategy[strategy]
            .Where(t => t.EntryTime >= cutoffTime)
            .Sum(t => t.PnL);
    }
    
    private string GetBestPerformingStrategyForDay(DateTime date)
    {
        var dayTrades = _allTrades.Where(t => t.EntryTime.Date == date).ToArray();
        if (dayTrades.Length == 0) return "N/A";
        
        var strategyPnL = dayTrades
            .GroupBy(t => t.Strategy)
            .ToDictionary(g => g.Key, g => g.Sum(t => t.PnL));
        
        if (strategyPnL.Count == 0) return "N/A";
        
        return strategyPnL.OrderByDescending(kvp => kvp.Value).First().Key;
    }
    
    private string GetWorstPerformingStrategyForDay(DateTime date)
    {
        var dayTrades = _allTrades.Where(t => t.EntryTime.Date == date).ToArray();
        if (dayTrades.Length == 0) return "N/A";
        
        var strategyPnL = dayTrades
            .GroupBy(t => t.Strategy)
            .ToDictionary(g => g.Key, g => g.Sum(t => t.PnL));
        
        if (strategyPnL.Count == 0) return "N/A";
        
        return strategyPnL.OrderBy(kvp => kvp.Value).First().Key;
    }
    
    private async Task<List<string>> GenerateTradingInsightsAsync(TradeOutcome[] trades, CancellationToken cancellationToken)
    {
        var insights = new List<string>();
        
        if (trades.Length == 0)
        {
            insights.Add("No trades executed today");
            return insights;
        }
        
        // Performance insights
        var winRate = (decimal)trades.Count(t => t.PnL > 0) / trades.Length;
        if (winRate > 0.7m)
        {
            insights.Add($"Excellent win rate today: {winRate:P1}");
        }
        else if (winRate < 0.4m)
        {
            insights.Add($"Low win rate today: {winRate:P1} - review strategy selection");
        }
        
        // Strategy insights
        var strategyPerformance = trades.GroupBy(t => t.Strategy)
            .ToDictionary(g => g.Key, g => g.Sum(t => t.PnL));
        
        if (strategyPerformance.Any())
        {
            var bestStrategy = strategyPerformance.OrderByDescending(kvp => kvp.Value).First();
            var worstStrategy = strategyPerformance.OrderBy(kvp => kvp.Value).First();
            
            insights.Add($"Best strategy: {bestStrategy.Key} (${bestStrategy.Value:F2})");
            if (worstStrategy.Value < 0)
            {
                insights.Add($"Worst strategy: {worstStrategy.Key} (${worstStrategy.Value:F2})");
            }
        }
        
        // Time-based insights
        var morningTrades = trades.Where(t => t.EntryTime.Hour >= 9 && t.EntryTime.Hour < 12).ToArray();
        var afternoonTrades = trades.Where(t => t.EntryTime.Hour >= 13 && t.EntryTime.Hour < 16).ToArray();
        
        if (morningTrades.Length > 0 && afternoonTrades.Length > 0)
        {
            var morningPnL = morningTrades.Sum(t => t.PnL);
            var afternoonPnL = afternoonTrades.Sum(t => t.PnL);
            
            if (morningPnL > afternoonPnL * 1.5m)
            {
                insights.Add("Morning session significantly outperformed afternoon");
            }
            else if (afternoonPnL > morningPnL * 1.5m)
            {
                insights.Add("Afternoon session significantly outperformed morning");
            }
        }
        
        return insights;
    }
    
    private List<OptimizationRecommendation> GenerateOptimizationRecommendations()
    {
        var recommendations = new List<OptimizationRecommendation>();
        
        // Strategy-specific recommendations
        foreach (var strategy in _tradesByStrategy.Keys)
        {
            var performance = GetStrategyPerformance(strategy);
            var strategyRecs = AnalyzeStrategyForOptimization(performance);
            recommendations.AddRange(strategyRecs);
        }
        
        return recommendations.OrderByDescending(r => r.Priority).ToList();
    }
    
    private List<OptimizationRecommendation> AnalyzeStrategyForOptimization(StrategyPerformance performance)
    {
        var recommendations = new List<OptimizationRecommendation>();
        
        if (performance.WinRate < 0.4m && performance.TotalTrades > 20)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = "STRATEGY_UNDERPERFORM",
                Priority = 9,
                Description = $"Strategy {performance.StrategyName} has low win rate: {performance.WinRate:P1}",
                Action = "Consider reducing allocation or reviewing entry criteria",
                Strategy = performance.StrategyName
            });
        }
        
        if (performance.ProfitFactor < 1.2m && performance.TotalTrades > 10)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = "PROFIT_FACTOR_LOW",
                Priority = 8,
                Description = $"Strategy {performance.StrategyName} has low profit factor: {performance.ProfitFactor:F2}",
                Action = "Review risk management and exit criteria",
                Strategy = performance.StrategyName
            });
        }
        
        if (performance.RecentPerformance > performance.TotalPnL * 0.1m)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = "RECENT_IMPROVEMENT",
                Priority = 7,
                Description = $"Strategy {performance.StrategyName} showing recent improvement",
                Action = "Consider increasing allocation",
                Strategy = performance.StrategyName
            });
        }
        
        return recommendations;
    }
    
    private List<OptimizationRecommendation> AnalyzeTimeBasedPatterns()
    {
        var recommendations = new List<OptimizationRecommendation>();
        
        // Analyze performance by hour
        var hourlyPerformance = _allTrades
            .GroupBy(t => t.EntryTime.Hour)
            .ToDictionary(g => g.Key, g => g.Sum(t => t.PnL));
        
        if (hourlyPerformance.Any())
        {
            var bestHour = hourlyPerformance.OrderByDescending(kvp => kvp.Value).First();
            var worstHour = hourlyPerformance.OrderBy(kvp => kvp.Value).First();
            
            if (bestHour.Value > worstHour.Value * 2)
            {
                recommendations.Add(new OptimizationRecommendation
                {
                    Type = "TIME_OPTIMIZATION",
                    Priority = 6,
                    Description = $"Hour {bestHour.Key} performs significantly better than hour {worstHour.Key}",
                    Action = "Focus trading activity during higher-performing hours",
                    Strategy = "ALL"
                });
            }
        }
        
        return recommendations;
    }
    
    private List<OptimizationRecommendation> AnalyzeRiskManagement()
    {
        var recommendations = new List<OptimizationRecommendation>();
        
        if (_maxDrawdown > _totalPnL * 0.3m && _totalTrades > 10)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = "RISK_MANAGEMENT",
                Priority = 10,
                Description = $"High maximum drawdown: ${_maxDrawdown:F2}",
                Action = "Reduce position sizes or improve stop loss management",
                Strategy = "ALL"
            });
        }
        
        if (_avgLoss > _avgWin * 1.5m && _totalTrades > 20)
        {
            recommendations.Add(new OptimizationRecommendation
            {
                Type = "RISK_REWARD",
                Priority = 8,
                Description = "Average loss significantly exceeds average win",
                Action = "Improve profit targets or tighten stop losses",
                Strategy = "ALL"
            });
        }
        
        return recommendations;
    }
}

/// <summary>
/// Strategy-specific performance metrics
/// </summary>
public class StrategyPerformance
{
    public string StrategyName { get; set; } = "";
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal WinRate { get; set; }
    public decimal AvgWin { get; set; }
    public decimal AvgLoss { get; set; }
    public decimal LargestWin { get; set; }
    public decimal LargestLoss { get; set; }
    public decimal ProfitFactor { get; set; }
    public decimal RecentPerformance { get; set; }
}

/// <summary>
/// Overall performance summary (Autonomous)
/// </summary>
public class AutonomousPerformanceSummary
{
    public decimal TotalPnL { get; set; }
    public decimal TodayPnL { get; set; }
    public decimal WeekPnL { get; set; }
    public decimal MonthPnL { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal AvgWin { get; set; }
    public decimal AvgLoss { get; set; }
    public decimal LargestWin { get; set; }
    public decimal LargestLoss { get; set; }
    public decimal ProfitFactor { get; set; }
    public decimal SharpeRatio { get; set; }
    public decimal MaxDrawdown { get; set; }
    public DateTime LastUpdateTime { get; set; }
    public string BestStrategy { get; set; } = "";
    public string WorstStrategy { get; set; } = "";
}

/// <summary>
/// Daily performance report
/// </summary>
public class DailyPerformanceReport
{
    public DateTime Date { get; set; }
    public decimal DailyPnL { get; set; }
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal LargestWin { get; set; }
    public decimal LargestLoss { get; set; }
    public string BestStrategy { get; set; } = "";
    public string WorstStrategy { get; set; } = "";
    public List<string> TradingInsights { get; set; } = new();
    public List<OptimizationRecommendation> OptimizationRecommendations { get; set; } = new();
}

/// <summary>
/// Learning insight from trade analysis
/// </summary>
public class LearningInsight
{
    public DateTime Timestamp { get; set; }
    public string Strategy { get; set; } = "";
    public TradeOutcome TradeOutcome { get; set; } = new();
    public string InsightType { get; set; } = "";
    public string Description { get; set; } = "";
    public decimal Confidence { get; set; }
    public Dictionary<string, object> MarketConditions { get; set; } = new();
}

/// <summary>
/// Strategy learning data
/// </summary>
public class StrategyLearning
{
    public string StrategyName { get; set; } = "";
    public List<LearningInsight> Insights { get; set; } = new();
}

/// <summary>
/// Performance snapshot for tracking
/// </summary>
public class PerformanceSnapshot
{
    public DateTime Timestamp { get; set; }
    public decimal TotalPnL { get; set; }
    public int TotalTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal SharpeRatio { get; set; }
}

/// <summary>
/// Optimization recommendation
/// </summary>
public class OptimizationRecommendation
{
    public string Type { get; set; } = "";
    public int Priority { get; set; }
    public string Description { get; set; } = "";
    public string Action { get; set; } = "";
    public string Strategy { get; set; } = "";
}

/// <summary>
/// Optimization insight
/// </summary>
public class OptimizationInsight
{
    public DateTime Timestamp { get; set; }
    public string Type { get; set; } = "";
    public string Description { get; set; } = "";
    public decimal ImpactScore { get; set; }
}