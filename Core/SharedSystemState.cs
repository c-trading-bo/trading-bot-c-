using System;
using System.Collections.Generic;

namespace TradingBot.Core
{
    /// <summary>
    /// SHARED STATE - Thread-safe with proper setters
    /// </summary>
    public class SharedSystemState
    {
        private readonly object _lock = new();
        
        // Use Core.MarketData for consistency
        public MarketData CurrentMarketData { get; private set; }
        public MarketAnalysis CurrentAnalysis { get; private set; }
        public Dictionary<string, StrategyPerformance> StrategyStats { get; } = new();
        public List<Position> OpenPositions { get; } = new();
        public decimal DailyPnL { get; private set; }
        public decimal Drawdown { get; private set; }
        public TradingMode TradingMode { get; private set; } = TradingMode.Normal;

        public void UpdateMarketData(MarketData data)
        {
            lock (_lock) CurrentMarketData = data;
        }

        public void UpdateAnalysis(MarketAnalysis analysis)
        {
            lock (_lock) CurrentAnalysis = analysis;
        }

        public void UpdateTradeResult(TradeResult result)
        {
            lock (_lock)
            {
                DailyPnL += result.RealizedPnL ?? 0;
                if (DailyPnL < 0)
                    Drawdown = Math.Max(Drawdown, Math.Abs(DailyPnL));
            }
        }

        public void SetTradingMode(TradingMode mode)
        {
            lock (_lock) TradingMode = mode;
        }

        // NEW METHOD FOR RISK UPDATES
        public void SetRiskSnapshot(decimal dailyPnL, decimal drawdown)
        {
            lock (_lock)
            {
                DailyPnL = dailyPnL;
                Drawdown = drawdown;
            }
        }

        public RiskLimits GetRiskLimits()
        {
            lock (_lock)
            {
                return new RiskLimits
                {
                    DailyLoss = Math.Abs(DailyPnL),
                    MaxDrawdown = Drawdown
                };
            }
        }

        public SystemState GetCurrentState()
        {
            lock (_lock)
            {
                return new SystemState
                {
                    DailyPnL = DailyPnL,
                    Drawdown = Drawdown,
                    OpenPositions = OpenPositions.Count,
                    TradingMode = TradingMode
                };
            }
        }

        public StrategyPerformance GetStrategyPerformance(string strategy)
        {
            lock (_lock)
            {
                return StrategyStats.GetValueOrDefault(strategy) ?? new StrategyPerformance();
            }
        }
    }
    
    public class StrategyPerformance
    {
        public int TotalTrades { get; set; }
        public int Wins { get; set; }
        public int Losses { get; set; }
        public decimal TotalPnL { get; set; }
        public double WinRate => TotalTrades > 0 ? (double)Wins / TotalTrades : 0;
    }

    public class Position
    {
        public string Symbol { get; set; }
        public int Quantity { get; set; }
        public decimal EntryPrice { get; set; }
        public DateTime EntryTime { get; set; }
    }

    public class TradeResult
    {
        public decimal? RealizedPnL { get; set; }
        public string Symbol { get; set; }
        public DateTime ExecutionTime { get; set; }
    }
}
