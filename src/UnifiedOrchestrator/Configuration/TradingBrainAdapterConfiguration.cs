namespace TradingBot.UnifiedOrchestrator.Configuration;

/// <summary>
/// Configuration for TradingBrainAdapter thresholds and behavioral parameters
/// AUDIT-CLEAN: Configuration-driven approach eliminates hardcoded trading thresholds
/// </summary>
public class TradingBrainAdapterConfiguration
{
    /// <summary>
    /// Position multiplier threshold for full buy/sell orders (default: 0.5)
    /// </summary>
    public decimal FullPositionThreshold { get; set; } = 0.5m;
    
    /// <summary>
    /// Position multiplier threshold for small buy/sell orders (default: 0.1) 
    /// </summary>
    public decimal SmallPositionThreshold { get; set; } = 0.1m;
    
    /// <summary>
    /// Size tolerance for decision comparison equivalence (default: 0.01)
    /// </summary>
    public decimal SizeComparisonTolerance { get; set; } = 0.01m;
    
    /// <summary>
    /// Confidence tolerance for decision comparison equivalence (default: 0.1)
    /// </summary>
    public decimal ConfidenceComparisonTolerance { get; set; } = 0.1m;
    
    /// <summary>
    /// Agreement rate threshold for promotion consideration (default: 0.8 = 80%)
    /// </summary>
    public double PromotionAgreementThreshold { get; set; } = 0.8;
    
    /// <summary>
    /// Number of recent decisions to consider for promotion evaluation (default: 100)
    /// </summary>
    public int PromotionEvaluationWindow { get; set; } = 100;
}