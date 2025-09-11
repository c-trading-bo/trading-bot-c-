using System;

namespace TradingBot.Abstractions;

/// <summary>
/// Represents the actively selected TopstepX account for the current session
/// </summary>
public class ActiveAccount
{
    public string Id { get; set; } = string.Empty;
    public string DisplayNumber { get; set; } = string.Empty;
    public string Alias { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public string Phase { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty;
    public bool CanTrade { get; set; }
    public decimal Balance { get; set; }
    public DateTime ResolvedAt { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = string.Empty;

    /// <summary>
    /// True if this appears to be an evaluation/practice account
    /// </summary>
    public bool IsEvaluationAccount => Phase?.Contains("Eval", StringComparison.OrdinalIgnoreCase) == true ||
                                      Alias?.Contains("PRAC", StringComparison.OrdinalIgnoreCase) == true ||
                                      Name?.Contains("Evaluation", StringComparison.OrdinalIgnoreCase) == true;

    /// <summary>
    /// True if this appears to be a live trading account
    /// </summary>
    public bool IsLiveAccount => !IsEvaluationAccount && CanTrade;

    public override string ToString()
    {
        return $"id={Id} display={DisplayNumber} alias={Alias} canTrade={CanTrade} phase={Phase} balance=${Balance:F2}";
    }
}
