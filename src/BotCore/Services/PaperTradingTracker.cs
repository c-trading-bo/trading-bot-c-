using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace BotCore.Services;

/// <summary>
/// Tracks simulated trades in DRY_RUN mode using REAL market data.
/// NOT random - monitors actual price movements to determine fills and P&L.
/// Converted to IHostedService - purely reactive to trade and market data events (Phase 5 refactoring).
/// </summary>
public class PaperTradingTracker : IHostedService, IDisposable
{
    private readonly ILogger<PaperTradingTracker> _logger;
    private readonly ConcurrentDictionary<string, SimulatedPosition> _activePositions = new();
    private readonly List<SimulatedTradeResult> _completedTrades = new();
    private readonly object _statsLock = new();
    private bool _disposed;
    
    // Event fired when simulated trade closes - allows learning from paper trades
    public event EventHandler<SimulatedTradeResult>? SimulatedTradeCompleted;
    
    // Performance tracking
    private decimal _totalPnL;
    private int _wins;
    private int _losses;
    private decimal _maxDrawdown;
    private decimal _peakEquity;

    public PaperTradingTracker(ILogger<PaperTradingTracker> logger)
    {
        _logger = logger;
        _peakEquity = 0m;
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("📊 [PAPER-TRADE] Paper Trading Tracker initialized - monitoring real market data for simulated fills (event-driven mode)");
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("📊 [PAPER-TRADE] Paper Trading Tracker stopped - Total P&L: ${PnL:F2}, Wins: {Wins}, Losses: {Losses}", 
            _totalPnL, _wins, _losses);
        return Task.CompletedTask;
    }

    /// <summary>
    /// Start tracking a simulated trade with real entry price
    /// </summary>
    public void OpenSimulatedTrade(
        string tradeId,
        string symbol,
        string direction,
        int size,
        decimal entryPrice,
        decimal stopLoss,
        decimal takeProfit,
        string strategy)
    {
        var position = new SimulatedPosition
        {
            TradeId = tradeId,
            Symbol = symbol,
            Direction = direction,
            Size = size,
            EntryPrice = entryPrice,
            StopLoss = stopLoss,
            TakeProfit = takeProfit,
            Strategy = strategy,
            EntryTime = DateTime.UtcNow,
            CurrentPrice = entryPrice,
            UnrealizedPnL = 0m
        };

        if (_activePositions.TryAdd(tradeId, position))
        {
            _logger.LogInformation(
                "📊 [PAPER-TRADE] SIMULATED ENTRY: {Direction} {Size} {Symbol} @ ${Entry:F2} | Stop: ${Stop:F2} | Target: ${Target:F2} | Strategy: {Strategy}",
                direction, size, symbol, entryPrice, stopLoss, takeProfit, strategy);
            
            _logger.LogInformation(
                "💰 [PAPER-TRADE] Max Risk: ${Risk:F2} | Max Reward: ${Reward:F2} | R-Multiple: {RMultiple:F2}",
                CalculateRisk(position), CalculateReward(position), CalculateRMultiple(position));
        }
    }

    /// <summary>
    /// Update position with real market price and check for simulated fills
    /// </summary>
    public void UpdateMarketPrice(string symbol, decimal currentPrice)
    {
        var symbolPositions = _activePositions.Values.Where(p => p.Symbol == symbol).ToList();
        
        foreach (var position in symbolPositions)
        {
            var previousPrice = position.CurrentPrice;
            position.CurrentPrice = currentPrice;
            
            // Calculate real P&L based on actual price movement
            var previousPnL = position.UnrealizedPnL;
            position.UnrealizedPnL = CalculateUnrealizedPnL(position);
            
            // Log P&L changes for significant moves
            var pnlChange = position.UnrealizedPnL - previousPnL;
            if (Math.Abs(pnlChange) >= 12.50m) // Log on $12.50+ moves (0.25 ES tick = $12.50)
            {
                _logger.LogInformation(
                    "📈 [PAPER-TRADE] {Symbol} P&L Update: ${PnL:F2} ({Direction} {Size} @ ${Entry:F2}, Now: ${Current:F2})",
                    symbol, position.UnrealizedPnL, position.Direction, position.Size, position.EntryPrice, currentPrice);
            }
            
            // Check if stop loss or take profit was hit
            CheckForSimulatedExit(position, currentPrice);
        }
    }

    /// <summary>
    /// Check if real market price hit stop or target (simulated fill)
    /// </summary>
    private void CheckForSimulatedExit(SimulatedPosition position, decimal currentPrice)
    {
        bool isStopHit = false;
        bool isTargetHit = false;
        decimal exitPrice = currentPrice;
        string exitReason = "";

        if (position.Direction == "Buy")
        {
            // Long position: stop below, target above
            if (currentPrice <= position.StopLoss)
            {
                isStopHit = true;
                exitPrice = position.StopLoss;
                exitReason = "STOP_LOSS";
            }
            else if (currentPrice >= position.TakeProfit)
            {
                isTargetHit = true;
                exitPrice = position.TakeProfit;
                exitReason = "TAKE_PROFIT";
            }
        }
        else // Sell
        {
            // Short position: stop above, target below
            if (currentPrice >= position.StopLoss)
            {
                isStopHit = true;
                exitPrice = position.StopLoss;
                exitReason = "STOP_LOSS";
            }
            else if (currentPrice <= position.TakeProfit)
            {
                isTargetHit = true;
                exitPrice = position.TakeProfit;
                exitReason = "TAKE_PROFIT";
            }
        }

        if (isStopHit || isTargetHit)
        {
            CloseSimulatedTrade(position, exitPrice, exitReason);
        }
    }

    /// <summary>
    /// Close simulated trade and record actual outcome based on real price
    /// </summary>
    private void CloseSimulatedTrade(SimulatedPosition position, decimal exitPrice, string exitReason)
    {
        if (!_activePositions.TryRemove(position.TradeId, out _))
        {
            return; // Already closed
        }

        var realizedPnL = CalculateRealizedPnL(position, exitPrice);
        var holdingTime = DateTime.UtcNow - position.EntryTime;
        
        var result = new SimulatedTradeResult
        {
            TradeId = position.TradeId,
            Symbol = position.Symbol,
            Direction = position.Direction,
            Size = position.Size,
            EntryPrice = position.EntryPrice,
            ExitPrice = exitPrice,
            EntryTime = position.EntryTime,
            ExitTime = DateTime.UtcNow,
            RealizedPnL = realizedPnL,
            ExitReason = exitReason,
            Strategy = position.Strategy,
            HoldingTime = holdingTime
        };

        lock (_statsLock)
        {
            _completedTrades.Add(result);
            _totalPnL += realizedPnL;
            
            if (realizedPnL > 0)
                _wins++;
            else
                _losses++;

            // Track drawdown
            _peakEquity = Math.Max(_peakEquity, _totalPnL);
            var currentDrawdown = _peakEquity - _totalPnL;
            _maxDrawdown = Math.Max(_maxDrawdown, currentDrawdown);
        }

        var isWin = realizedPnL > 0;
        var emoji = isWin ? "✅" : "❌";
        
        _logger.LogInformation(
            "{Emoji} [PAPER-TRADE] SIMULATED EXIT: {Reason} | {Direction} {Size} {Symbol} @ ${Exit:F2} | P&L: ${PnL:F2} | Duration: {Duration}",
            emoji, exitReason, position.Direction, position.Size, position.Symbol, exitPrice, realizedPnL, FormatDuration(holdingTime));
        
        _logger.LogInformation(
            "📊 [PAPER-TRADE] Entry: ${Entry:F2} → Exit: ${Exit:F2} | Move: ${Move:F2} ({MovePercent:F2}%) | Strategy: {Strategy}",
            position.EntryPrice, exitPrice, Math.Abs(exitPrice - position.EntryPrice), 
            Math.Abs((exitPrice - position.EntryPrice) / position.EntryPrice * 100), position.Strategy);
        
        LogPerformanceSummary();
        
        // Fire event so autonomous engine can learn from simulated trade
        SimulatedTradeCompleted?.Invoke(this, result);
        
        _logger.LogDebug("📚 [PAPER-TRADE] Trade outcome sent to learning system");
    }

    /// <summary>
    /// Calculate unrealized P&L based on real current market price
    /// </summary>
    private decimal CalculateUnrealizedPnL(SimulatedPosition position)
    {
        var priceDiff = position.Direction == "Buy" 
            ? position.CurrentPrice - position.EntryPrice 
            : position.EntryPrice - position.CurrentPrice;
        
        // ES: $50 per point, NQ: $20 per point
        var multiplier = position.Symbol == "ES" ? 50m : 20m;
        
        return priceDiff * position.Size * multiplier;
    }

    /// <summary>
    /// Calculate realized P&L based on actual exit price
    /// </summary>
    private decimal CalculateRealizedPnL(SimulatedPosition position, decimal exitPrice)
    {
        var priceDiff = position.Direction == "Buy" 
            ? exitPrice - position.EntryPrice 
            : position.EntryPrice - exitPrice;
        
        var multiplier = position.Symbol == "ES" ? 50m : 20m;
        
        return priceDiff * position.Size * multiplier;
    }

    private decimal CalculateRisk(SimulatedPosition position)
    {
        var priceDiff = Math.Abs(position.EntryPrice - position.StopLoss);
        var multiplier = position.Symbol == "ES" ? 50m : 20m;
        return priceDiff * position.Size * multiplier;
    }

    private decimal CalculateReward(SimulatedPosition position)
    {
        var priceDiff = Math.Abs(position.EntryPrice - position.TakeProfit);
        var multiplier = position.Symbol == "ES" ? 50m : 20m;
        return priceDiff * position.Size * multiplier;
    }

    private decimal CalculateRMultiple(SimulatedPosition position)
    {
        var risk = CalculateRisk(position);
        var reward = CalculateReward(position);
        return risk > 0 ? reward / risk : 0m;
    }

    private void LogPerformanceSummary()
    {
        lock (_statsLock)
        {
            var totalTrades = _wins + _losses;
            var winRate = totalTrades > 0 ? (_wins / (decimal)totalTrades * 100) : 0;
            
            _logger.LogInformation(
                "📈 [PAPER-PERFORMANCE] Total P&L: ${TotalPnL:F2} | Trades: {Total} | Wins: {Wins} | Losses: {Losses} | Win Rate: {WinRate:F1}%",
                _totalPnL, totalTrades, _wins, _losses, winRate);
            
            if (_maxDrawdown > 0)
            {
                _logger.LogInformation(
                    "📉 [PAPER-PERFORMANCE] Max Drawdown: ${MaxDD:F2} | Peak Equity: ${Peak:F2}",
                    _maxDrawdown, _peakEquity);
            }
        }
    }

    public PaperTradingStats GetStats()
    {
        lock (_statsLock)
        {
            var totalTrades = _wins + _losses;
            var winRate = totalTrades > 0 ? (_wins / (decimal)totalTrades) : 0;
            
            return new PaperTradingStats
            {
                TotalPnL = _totalPnL,
                TotalTrades = totalTrades,
                Wins = _wins,
                Losses = _losses,
                WinRate = winRate,
                MaxDrawdown = _maxDrawdown,
                PeakEquity = _peakEquity,
                ActivePositions = _activePositions.Count,
                CompletedTrades = _completedTrades.ToList()
            };
        }
    }

    private string FormatDuration(TimeSpan duration)
    {
        if (duration.TotalMinutes < 1)
            return $"{duration.TotalSeconds:F0}s";
        if (duration.TotalHours < 1)
            return $"{duration.TotalMinutes:F0}m";
        return $"{duration.TotalHours:F1}h";
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
        }
    }
}

/// <summary>
/// Active simulated position being tracked with real market data
/// </summary>
public class SimulatedPosition
{
    public required string TradeId { get; set; }
    public required string Symbol { get; set; }
    public required string Direction { get; set; }
    public required int Size { get; set; }
    public required decimal EntryPrice { get; set; }
    public required decimal StopLoss { get; set; }
    public required decimal TakeProfit { get; set; }
    public required string Strategy { get; set; }
    public required DateTime EntryTime { get; set; }
    public decimal CurrentPrice { get; set; }
    public decimal UnrealizedPnL { get; set; }
}

/// <summary>
/// Completed simulated trade with actual outcome based on real prices
/// </summary>
public class SimulatedTradeResult
{
    public required string TradeId { get; set; }
    public required string Symbol { get; set; }
    public required string Direction { get; set; }
    public required int Size { get; set; }
    public required decimal EntryPrice { get; set; }
    public required decimal ExitPrice { get; set; }
    public required DateTime EntryTime { get; set; }
    public required DateTime ExitTime { get; set; }
    public required decimal RealizedPnL { get; set; }
    public required string ExitReason { get; set; }
    public required string Strategy { get; set; }
    public required TimeSpan HoldingTime { get; set; }
}

/// <summary>
/// Performance statistics for paper trading
/// </summary>
public class PaperTradingStats
{
    public decimal TotalPnL { get; set; }
    public int TotalTrades { get; set; }
    public int Wins { get; set; }
    public int Losses { get; set; }
    public decimal WinRate { get; set; }
    public decimal MaxDrawdown { get; set; }
    public decimal PeakEquity { get; set; }
    public int ActivePositions { get; set; }
    public List<SimulatedTradeResult> CompletedTrades { get; set; } = new();
}
