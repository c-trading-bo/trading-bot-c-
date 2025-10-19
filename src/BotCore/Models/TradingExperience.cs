using System;

namespace BotCore.Models
{
    /// <summary>
    /// Trading experience record for reinforcement learning
    /// Task 4.1: Experience Collection in Terminal
    /// 
    /// Captures state-action-reward-next_state tuple when positions close
    /// Used by Lab for training CVaR-PPO, Neural UCB, and other RL models
    /// </summary>
    public sealed class TradingExperience
    {
        /// <summary>
        /// Unique identifier for this experience
        /// </summary>
        public string ExperienceId { get; set; } = Guid.NewGuid().ToString();
        
        /// <summary>
        /// Timestamp when position was closed (experience created)
        /// </summary>
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        /// <summary>
        /// Position ID that generated this experience
        /// </summary>
        public string PositionId { get; set; } = string.Empty;
        
        // ======================================================================
        // STATE (Market conditions at entry time)
        // ======================================================================
        
        /// <summary>
        /// Market regime at entry (Trend/Range/Transition)
        /// </summary>
        public string EntryRegime { get; set; } = "UNKNOWN";
        
        /// <summary>
        /// Regime confidence at entry (0.0-1.0)
        /// </summary>
        public decimal EntryRegimeConfidence { get; set; }
        
        /// <summary>
        /// Strategy confidence at entry (0.0-1.0)
        /// </summary>
        public decimal EntryConfidence { get; set; }
        
        /// <summary>
        /// Symbol traded (ES, NQ, etc.)
        /// </summary>
        public string Symbol { get; set; } = string.Empty;
        
        /// <summary>
        /// Time of day at entry (hour, 0-23)
        /// </summary>
        public int EntryHour { get; set; }
        
        /// <summary>
        /// Day of week at entry (0=Sunday, 1=Monday, etc.)
        /// </summary>
        public int EntryDayOfWeek { get; set; }
        
        /// <summary>
        /// Volatility level at entry (normalized ATR or similar)
        /// </summary>
        public decimal VolatilityAtEntry { get; set; }
        
        // ======================================================================
        // ACTION (What the bot decided to do)
        // ======================================================================
        
        /// <summary>
        /// Strategy chosen (S2, S3, S6, S11, etc.)
        /// </summary>
        public string Strategy { get; set; } = string.Empty;
        
        /// <summary>
        /// Position size (number of contracts, signed: + for long, - for short)
        /// </summary>
        public int PositionSize { get; set; }
        
        /// <summary>
        /// Entry price
        /// </summary>
        public decimal EntryPrice { get; set; }
        
        /// <summary>
        /// Initial stop loss price
        /// </summary>
        public decimal InitialStopPrice { get; set; }
        
        /// <summary>
        /// Initial target price
        /// </summary>
        public decimal InitialTargetPrice { get; set; }
        
        /// <summary>
        /// Breakeven trigger distance (ticks)
        /// </summary>
        public int BreakevenAfterTicks { get; set; }
        
        /// <summary>
        /// Trailing stop distance (ticks)
        /// </summary>
        public int TrailTicks { get; set; }
        
        // ======================================================================
        // REWARD (Outcome metrics)
        // ======================================================================
        
        /// <summary>
        /// R-multiple (profit / risk)
        /// Positive = winning trade, Negative = losing trade
        /// Example: +2.5 = made 2.5x the risk, -1.0 = lost 1x the risk
        /// </summary>
        public decimal RMultiple { get; set; }
        
        /// <summary>
        /// Actual P&L in dollars
        /// </summary>
        public decimal PnL { get; set; }
        
        /// <summary>
        /// Sharpe contribution (risk-adjusted return)
        /// Calculated as: (return - risk_free_rate) / volatility
        /// </summary>
        public decimal SharpeContribution { get; set; }
        
        /// <summary>
        /// Exit reason (StopLoss, Target, Time, Manual, etc.)
        /// </summary>
        public string ExitReason { get; set; } = string.Empty;
        
        /// <summary>
        /// Position duration in minutes
        /// </summary>
        public double DurationMinutes { get; set; }
        
        // ======================================================================
        // NEXT STATE (Market conditions at exit time)
        // ======================================================================
        
        /// <summary>
        /// Market regime at exit
        /// </summary>
        public string ExitRegime { get; set; } = "UNKNOWN";
        
        /// <summary>
        /// Regime confidence at exit (0.0-1.0)
        /// </summary>
        public decimal ExitRegimeConfidence { get; set; }
        
        /// <summary>
        /// Exit price
        /// </summary>
        public decimal ExitPrice { get; set; }
        
        /// <summary>
        /// Volatility level at exit
        /// </summary>
        public decimal VolatilityAtExit { get; set; }
        
        // ======================================================================
        // ADDITIONAL METRICS (For advanced learning)
        // ======================================================================
        
        /// <summary>
        /// Maximum Favorable Excursion (best price reached)
        /// </summary>
        public decimal MaxFavorablePrice { get; set; }
        
        /// <summary>
        /// Maximum Adverse Excursion (worst price reached)
        /// </summary>
        public decimal MaxAdversePrice { get; set; }
        
        /// <summary>
        /// Number of stop modifications during trade
        /// </summary>
        public int StopModificationCount { get; set; }
        
        /// <summary>
        /// Whether breakeven was activated
        /// </summary>
        public bool BreakevenActivated { get; set; }
        
        /// <summary>
        /// Whether trailing stop was activated
        /// </summary>
        public bool TrailingStopActive { get; set; }
        
        /// <summary>
        /// Number of regime changes during trade
        /// </summary>
        public int RegimeChangeCount { get; set; }
    }
}
