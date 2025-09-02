using System.Text.Json;
using Microsoft.Extensions.Logging;

namespace BotCore.Services;

/// <summary>
/// Service for tracking complete trade lifecycle and performance metrics.
/// Logs all trades for personalization and learning.
/// </summary>
public class PerformanceTracker
{
    private readonly ILogger<PerformanceTracker> _logger;
    private readonly string _tradesPath;
    private readonly JsonSerializerOptions _jsonOptions;

    public PerformanceTracker(ILogger<PerformanceTracker> logger, string? tradesPath = null)
    {
        _logger = logger;
        _tradesPath = tradesPath ?? "../Intelligence/data/trades";
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            WriteIndented = true
        };

        // Ensure trades directory exists
        Directory.CreateDirectory(_tradesPath);
    }

    /// <summary>
    /// Log complete trade lifecycle with all relevant data
    /// </summary>
    public async Task LogTradeAsync(TradeRecord trade)
    {
        try
        {
            // Calculate trade metrics
            var duration = trade.ExitTime - trade.EntryTime;
            var pnlPercent = trade.EntryPrice != 0 ? ((trade.ExitPrice - trade.EntryPrice) / trade.EntryPrice) * 100 : 0;
            var rMultiple = CalculateRMultiple(trade);

            var tradeLog = new
            {
                // Basic trade info
                TradeId = trade.TradeId,
                Symbol = trade.Symbol,
                Strategy = trade.Strategy,
                Side = trade.Side, // "BUY" or "SELL"

                // Timing
                EntryTime = trade.EntryTime.ToString("O"),
                ExitTime = trade.ExitTime.ToString("O"),
                DurationMinutes = (int)duration.TotalMinutes,
                DurationHours = Math.Round(duration.TotalHours, 2),

                // Prices and PnL
                EntryPrice = Math.Round(trade.EntryPrice, 4),
                ExitPrice = Math.Round(trade.ExitPrice, 4),
                StopPrice = trade.StopPrice.HasValue ? (decimal?)Math.Round(trade.StopPrice.Value, 4) : null,
                TargetPrice = trade.TargetPrice.HasValue ? (decimal?)Math.Round(trade.TargetPrice.Value, 4) : null,
                Quantity = trade.Quantity,
                PnLDollar = Math.Round(trade.PnLDollar, 2),
                PnLPercent = Math.Round(pnlPercent, 4),
                RMultiple = Math.Round(rMultiple, 2),

                // Exit reason and quality
                ExitReason = trade.ExitReason, // "TARGET", "STOP", "TIME", "MANUAL", etc.
                TradeQuality = ClassifyTradeQuality(trade, rMultiple),

                // Market context (if available)
                MarketRegime = trade.MarketRegime,
                VolatilityLevel = trade.VolatilityLevel,
                IntelligenceUsed = trade.IntelligenceUsed,
                IntelligenceConfidence = trade.IntelligenceConfidence,

                // Custom tags and notes
                Tags = trade.Tags,
                Notes = trade.Notes,

                // Performance metrics
                MaxFavorableExcursion = trade.MaxFavorableExcursion,
                MaxAdverseExcursion = trade.MaxAdverseExcursion,

                // Timestamp
                LoggedAt = DateTime.UtcNow.ToString("O")
            };

            _logger.LogInformation("[TRADE_LOG] {Symbol} {Strategy} {Side} pnl={PnLPercent:P2} R={RMultiple:F1} dur={DurationMinutes}m reason={ExitReason}",
                trade.Symbol, trade.Strategy, trade.Side, pnlPercent / 100, rMultiple, duration.TotalMinutes, trade.ExitReason);

            // Save to daily trade log
            await SaveTradeLogAsync(tradeLog);

            // Update personal metrics
            await UpdatePersonalMetricsAsync(trade, rMultiple);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error logging trade {TradeId}", trade.TradeId);
        }
    }

    /// <summary>
    /// Track personal edge patterns and metrics
    /// </summary>
    public async Task UpdatePersonalMetricsAsync(TradeRecord trade, double rMultiple)
    {
        try
        {
            var metricsFile = Path.Combine(_tradesPath, "personal_metrics.json");

            // Load existing metrics
            var metrics = await LoadPersonalMetricsAsync() ?? new PersonalMetrics();

            // Update trade counts
            metrics.TotalTrades++;
            if (trade.PnLDollar > 0) metrics.WinningTrades++;
            if (trade.PnLDollar < 0) metrics.LosingTrades++;

            // Update PnL
            metrics.TotalPnL += trade.PnLDollar;
            metrics.GrossPnL += Math.Abs(trade.PnLDollar);

            // Update R multiples
            metrics.TotalRMultiple += rMultiple;
            metrics.AverageRMultiple = metrics.TotalTrades > 0 ? metrics.TotalRMultiple / metrics.TotalTrades : 0;

            // Update win rate
            metrics.WinRate = metrics.TotalTrades > 0 ? (double)metrics.WinningTrades / metrics.TotalTrades : 0;

            // Track best performing strategies
            if (!metrics.StrategyPerformance.ContainsKey(trade.Strategy))
            {
                metrics.StrategyPerformance[trade.Strategy] = new StrategyMetrics();
            }

            var strategyMetrics = metrics.StrategyPerformance[trade.Strategy];
            strategyMetrics.TradeCount++;
            strategyMetrics.TotalPnL += trade.PnLDollar;
            strategyMetrics.TotalRMultiple += rMultiple;
            strategyMetrics.AverageRMultiple = strategyMetrics.TotalRMultiple / strategyMetrics.TradeCount;
            if (trade.PnLDollar > 0) strategyMetrics.Wins++;
            strategyMetrics.WinRate = strategyMetrics.TradeCount > 0 ? (double)strategyMetrics.Wins / strategyMetrics.TradeCount : 0;

            // Update timestamps
            metrics.LastUpdated = DateTime.UtcNow;
            metrics.LastTradeTime = trade.ExitTime;

            // Save updated metrics
            await File.WriteAllTextAsync(metricsFile, JsonSerializer.Serialize(metrics, _jsonOptions));

            _logger.LogInformation("[PERSONAL_METRICS] Total trades: {TotalTrades}, Win rate: {WinRate:P1}, Avg R: {AvgR:F2}",
                metrics.TotalTrades, metrics.WinRate, metrics.AverageRMultiple);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating personal metrics");
        }
    }

    /// <summary>
    /// Push trade data to cloud for ML training (placeholder for future implementation)
    /// </summary>
    public async Task PushToCloudAsync(TradeRecord trade)
    {
        try
        {
            // Placeholder for cloud push functionality
            // In a real implementation, this would send data to cloud ML pipeline

            _logger.LogDebug("[CLOUD_PUSH] Trade {TradeId} queued for cloud learning", trade.TradeId);

            // For now, just save to a cloud queue file
            var cloudQueueFile = Path.Combine(_tradesPath, "cloud_queue.json");
            var cloudData = new
            {
                TradeId = trade.TradeId,
                Symbol = trade.Symbol,
                Strategy = trade.Strategy,
                PnL = trade.PnLDollar,
                Success = trade.PnLDollar > 0,
                QueuedAt = DateTime.UtcNow.ToString("O")
            };

            var queue = new List<object>();
            if (File.Exists(cloudQueueFile))
            {
                var json = await File.ReadAllTextAsync(cloudQueueFile);
                var existing = JsonSerializer.Deserialize<List<object>>(json);
                if (existing != null) queue = existing;
            }

            queue.Add(cloudData);

            // Keep only last 1000 items
            if (queue.Count > 1000)
            {
                queue = queue.Skip(queue.Count - 1000).ToList();
            }

            await File.WriteAllTextAsync(cloudQueueFile, JsonSerializer.Serialize(queue, _jsonOptions));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error pushing trade to cloud queue");
        }
    }

    /// <summary>
    /// Get personal trading metrics
    /// </summary>
    public async Task<PersonalMetrics?> LoadPersonalMetricsAsync()
    {
        try
        {
            var metricsFile = Path.Combine(_tradesPath, "personal_metrics.json");

            if (!File.Exists(metricsFile))
                return null;

            var json = await File.ReadAllTextAsync(metricsFile);
            return JsonSerializer.Deserialize<PersonalMetrics>(json, _jsonOptions);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading personal metrics");
            return null;
        }
    }

    private double CalculateRMultiple(TradeRecord trade)
    {
        if (!trade.StopPrice.HasValue || trade.EntryPrice == 0)
            return 0;

        var risk = Math.Abs((double)(trade.EntryPrice - trade.StopPrice.Value));
        if (risk == 0) return 0;

        var reward = Math.Abs((double)(trade.ExitPrice - trade.EntryPrice));
        return reward / risk;
    }

    private string ClassifyTradeQuality(TradeRecord trade, double rMultiple)
    {
        if (trade.PnLDollar > 0)
        {
            if (rMultiple >= 2.0) return "excellent_win";
            if (rMultiple >= 1.0) return "good_win";
            return "small_win";
        }
        else
        {
            if (Math.Abs(rMultiple) <= 0.5) return "controlled_loss";
            if (Math.Abs(rMultiple) <= 1.0) return "acceptable_loss";
            return "large_loss";
        }
    }

    private async Task SaveTradeLogAsync(object tradeLog)
    {
        try
        {
            var today = DateTime.Today.ToString("yyyy-MM-dd");
            var logFile = Path.Combine(_tradesPath, $"trades_{today}.json");

            var trades = new List<object>();
            if (File.Exists(logFile))
            {
                var json = await File.ReadAllTextAsync(logFile);
                var existing = JsonSerializer.Deserialize<List<object>>(json);
                if (existing != null) trades = existing;
            }

            trades.Add(tradeLog);

            await File.WriteAllTextAsync(logFile, JsonSerializer.Serialize(trades, _jsonOptions));
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving trade log");
        }
    }
}

/// <summary>
/// Complete trade record for logging
/// </summary>
public class TradeRecord
{
    public string TradeId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty; // "BUY" or "SELL"

    public DateTime EntryTime { get; set; }
    public DateTime ExitTime { get; set; }

    public decimal EntryPrice { get; set; }
    public decimal ExitPrice { get; set; }
    public decimal? StopPrice { get; set; }
    public decimal? TargetPrice { get; set; }
    public int Quantity { get; set; }

    public decimal PnLDollar { get; set; }
    public string ExitReason { get; set; } = string.Empty;

    // Market context
    public string? MarketRegime { get; set; }
    public string? VolatilityLevel { get; set; }
    public bool IntelligenceUsed { get; set; }
    public decimal? IntelligenceConfidence { get; set; }

    // Additional metrics
    public decimal? MaxFavorableExcursion { get; set; }
    public decimal? MaxAdverseExcursion { get; set; }

    public List<string> Tags { get; set; } = new();
    public string? Notes { get; set; }
}

/// <summary>
/// Personal trading metrics
/// </summary>
public class PersonalMetrics
{
    public int TotalTrades { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public double WinRate { get; set; }

    public decimal TotalPnL { get; set; }
    public decimal GrossPnL { get; set; }

    public double TotalRMultiple { get; set; }
    public double AverageRMultiple { get; set; }

    public Dictionary<string, StrategyMetrics> StrategyPerformance { get; set; } = new();

    public DateTime LastUpdated { get; set; }
    public DateTime LastTradeTime { get; set; }
}

/// <summary>
/// Strategy-specific metrics
/// </summary>
public class StrategyMetrics
{
    public int TradeCount { get; set; }
    public int Wins { get; set; }
    public double WinRate { get; set; }
    public decimal TotalPnL { get; set; }
    public double TotalRMultiple { get; set; }
    public double AverageRMultiple { get; set; }
}