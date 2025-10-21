using TradingBot.RLAgent; // For other types
using BotCore.Models; // For Candidate type

namespace BotCore.Brain.Models
{
    /// <summary>
    /// Decision result from the unified trading brain
    /// </summary>
    public class BrainDecision
    {
        public string Symbol { get; set; } = string.Empty;
        public string RecommendedStrategy { get; set; } = string.Empty;
        public decimal StrategyConfidence { get; set; }
        public PriceDirection PriceDirection { get; set; }
        public decimal PriceProbability { get; set; }
        public decimal OptimalPositionMultiplier { get; set; }
        public MarketRegime MarketRegime { get; set; }
        public IReadOnlyList<Candidate> EnhancedCandidates { get; set; } = new List<Candidate>();
        public DateTime DecisionTime { get; set; }
        public double ProcessingTimeMs { get; set; }
        public decimal ModelConfidence { get; set; }
        public string RiskAssessment { get; set; } = string.Empty;
    }

    /// <summary>
    /// Market context information for decision making
    /// </summary>
    public class MarketContext
    {
        public string Symbol { get; set; } = string.Empty;
        public decimal CurrentPrice { get; set; }
        public decimal Volume { get; set; }
        public decimal? Atr { get; set; }
        public decimal Volatility { get; set; }
        public TimeSpan TimeOfDay { get; set; }
        public DayOfWeek DayOfWeek { get; set; }
        public decimal VolumeRatio { get; set; }
        public decimal PriceChange { get; set; }
        public decimal RSI { get; set; }
        public decimal TrendStrength { get; set; }
        public decimal DistanceToSupport { get; set; }
        public decimal DistanceToResistance { get; set; }
        public decimal VolatilityRank { get; set; }
        public decimal Momentum { get; set; }
        public int MarketRegime { get; set; }
        
        // Additional properties needed by NeuralUcbExtended
        public Dictionary<string, double> Features { get; } = new();
        public double Price { get; set; }
        public double Bid { get; set; }
        public double Ask { get; set; }
        public double SignalStrength { get; set; }
        public double ConfidenceLevel { get; set; }
        public double ModelConfidence { get; set; }
        public double NewsIntensity { get; set; }
        public Dictionary<string, double> TechnicalIndicators { get; } = new();
        public bool IsFomcDay { get; set; }
        public bool IsCpiDay { get; set; }
    }

    /// <summary>
    /// Strategy selection result
    /// </summary>
    public class StrategySelection
    {
        public string SelectedStrategy { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        public decimal UcbValue { get; set; }
        public string Reasoning { get; set; } = string.Empty;
    }

    /// <summary>
    /// Price prediction result
    /// </summary>
    public class PricePrediction
    {
        public PriceDirection Direction { get; set; }
        public decimal Probability { get; set; }
        public decimal ExpectedMove { get; set; }
        public TimeSpan TimeHorizon { get; set; }
    }

    /// <summary>
    /// Trading decision with context
    /// </summary>
    public class TradingDecision
    {
        public string Symbol { get; set; } = string.Empty;
        public string Strategy { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        public MarketContext Context { get; set; } = new();
        public DateTime Timestamp { get; set; }
        public bool WasCorrect { get; set; }
        public decimal PnL { get; set; }
    }

    /// <summary>
    /// Trading performance summary
    /// </summary>
    public class TradingPerformance
    {
        public int TotalTrades { get; set; }
        public int WinningTrades { get; set; }
        public decimal TotalPnL { get; set; }
        public decimal WinRate => TotalTrades > 0 ? (decimal)WinningTrades / TotalTrades : 0;
    }

    /// <summary>
    /// Decision history entry for Lab Mode persistence
    /// Captures strategy selection, market conditions, and outcomes for offline analysis
    /// </summary>
    public class DecisionHistoryEntry
    {
        public DateTime Timestamp { get; set; }
        public string Symbol { get; set; } = string.Empty;
        public string StrategyChosen { get; set; } = string.Empty;
        public decimal Confidence { get; set; }
        
        // Market conditions at decision time
        public decimal VIX { get; set; }
        public string PriceDirection { get; set; } = string.Empty;
        public decimal Volatility { get; set; }
        public decimal Volume { get; set; }
        public decimal RSI { get; set; }
        public decimal TrendStrength { get; set; }
        public string TimeOfDay { get; set; } = string.Empty;
        public string DayOfWeek { get; set; } = string.Empty;
        
        // Outcome
        public decimal PnL { get; set; }
        public bool WasCorrect { get; set; }
        
        // Selection reason
        public string MarketRegime { get; set; } = string.Empty;
    }

    /// <summary>
    /// Price direction enumeration
    /// </summary>
    public enum PriceDirection
    {
        Up,
        Down,
        Sideways
    }

    /// <summary>
    /// Market regime classification
    /// </summary>
    public enum MarketRegime
    {
        Normal = 0,
        Trending = 1,
        Ranging = 2,
        HighVolatility = 3,
        LowVolatility = 4
    }
}