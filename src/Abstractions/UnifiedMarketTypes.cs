using System;

namespace TradingBot.Abstractions;

/// <summary>
/// Unified market regime classification used across all components
/// This ensures consistency between autonomous engine, performance analyzers, and safety systems
/// </summary>
public enum TradingMarketRegime
{
    Unknown,
    Trending,
    Ranging,
    Volatile,
    LowVolatility,
    Crisis,
    Recovery
}

/// <summary>
/// Market volatility levels
/// </summary>
public enum MarketVolatility
{
    VeryLow,
    Low,
    Normal,
    High,
    VeryHigh
}

/// <summary>
/// Volume regime classification
/// </summary>
public enum VolumeRegime
{
    Low,
    Normal,
    High,
    Extreme
}

/// <summary>
/// Time of day classification for trading
/// </summary>
public enum TimeOfDay
{
    PreMarket,
    MarketOpen,
    MidMorning,
    Lunch,
    Afternoon,
    MarketClose,
    AfterHours
}

/// <summary>
/// Market session classification
/// </summary>
public enum MarketSession
{
    Closed,
    PreMarket,
    RegularHours,
    PostMarket
}

/// <summary>
/// Unified trade outcome used across all performance tracking systems
/// </summary>
public class UnifiedTradeOutcome
{
    public string TradeId { get; set; } = "";
    public string Strategy { get; set; } = "";
    public string Symbol { get; set; } = "";
    public string Direction { get; set; } = "";
    public DateTime EntryTime { get; set; }
    public DateTime? ExitTime { get; set; }
    public decimal EntryPrice { get; set; }
    public decimal? ExitPrice { get; set; }
    public int Size { get; set; }
    public decimal PnL { get; set; }
    public decimal Confidence { get; set; }
    public TradingMarketRegime MarketRegime { get; set; } = TradingMarketRegime.Unknown;
    public MarketVolatility Volatility { get; set; } = MarketVolatility.Normal;
    public VolumeRegime VolumeCondition { get; set; } = VolumeRegime.Normal;
    public TimeOfDay TimeSlot { get; set; } = TimeOfDay.MarketOpen;
    public bool IsWin { get; set; }
    public decimal RMultiple { get; set; }
    public TimeSpan Duration { get; set; }
    public string Notes { get; set; } = "";
}