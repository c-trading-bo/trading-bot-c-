using System.ComponentModel.DataAnnotations;
using TradingBot.Abstractions;

namespace BotCore.Models;

// MarketContext removed - use canonical version from TradingBot.Abstractions

/// <summary>
/// Individual trade setup suggested by the Intelligence system
/// </summary>
public class TradeSetup
{
    /// <summary>
    /// Time window for the setup: "Opening30Min", "Afternoon", etc.
    /// </summary>
    [Required]
    public string TimeWindow { get; set; } = string.Empty;

    /// <summary>
    /// Trade direction: "Long", "Short"
    /// </summary>
    [Required]
    public string Direction { get; set; } = string.Empty;

    /// <summary>
    /// Setup confidence score (0-1)
    /// </summary>
    [Range(0, 1)]
    public decimal ConfidenceScore { get; set; }

    /// <summary>
    /// Suggested risk multiple for position sizing
    /// </summary>
    public decimal SuggestedRiskMultiple { get; set; }

    /// <summary>
    /// Human-readable explanation for the setup
    /// </summary>
    public string Rationale { get; set; } = string.Empty;
}