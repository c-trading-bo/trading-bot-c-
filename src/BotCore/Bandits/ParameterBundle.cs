using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Bandits;

/// <summary>
/// Bracket mode configuration for automated order management
/// Enables learning optimal stop/target combinations for different market conditions
/// </summary>
public record BracketMode
{
    /// <summary>
    /// Stop loss distance in ticks
    /// </summary>
    public int StopTicks { get; init; }
    
    /// <summary>
    /// Target profit distance in ticks
    /// </summary>
    public int TargetTicks { get; init; }
    
    /// <summary>
    /// Move to breakeven after this many ticks of profit
    /// </summary>
    public int BreakevenAfterTicks { get; init; }
    
    /// <summary>
    /// Trailing stop distance in ticks
    /// </summary>
    public int TrailTicks { get; init; }
    
    /// <summary>
    /// Bracket mode identifier for bundle identification
    /// Example: "C12T18" = Conservative with 12 stop, 18 target
    /// </summary>
    public string ModeId => $"{ModeType[0]}{StopTicks}T{TargetTicks}";
    
    /// <summary>
    /// Human readable bracket mode type
    /// </summary>
    public string ModeType { get; init; } = "Conservative";
    
    /// <summary>
    /// Human-readable description
    /// </summary>
    public string Description => $"{ModeType} bracket ({StopTicks}S/{TargetTicks}T)";
    
    /// <summary>
    /// Risk-to-reward ratio for this bracket mode
    /// </summary>
    public decimal RiskRewardRatio => StopTicks > 0 ? (decimal)TargetTicks / StopTicks : 0m;
    
    /// <summary>
    /// Validate bracket parameters are within safe ranges
    /// </summary>
    public bool IsValid =>
        StopTicks >= 6 && StopTicks <= 20 &&         // Stop: 6-20 ticks (safe range for ES/MES)
        TargetTicks >= 8 && TargetTicks <= 30 &&     // Target: 8-30 ticks
        BreakevenAfterTicks >= 4 && BreakevenAfterTicks <= 16 && // Breakeven: 4-16 ticks
        TrailTicks >= 3 && TrailTicks <= 12 &&       // Trail: 3-12 ticks
        TargetTicks > StopTicks &&                    // Target must be greater than stop
        !string.IsNullOrEmpty(ModeType);
    
    /// <summary>
    /// Predefined bracket modes for different market conditions
    /// </summary>
    public static class Presets
    {
        /// <summary>Conservative bracket: Tight stops, moderate targets</summary>
        public static readonly BracketMode Conservative = new()
        {
            ModeType = "Conservative",
            StopTicks = 10,
            TargetTicks = 15,
            BreakevenAfterTicks = 6,
            TrailTicks = 4
        };
        
        /// <summary>Balanced bracket: Balanced risk/reward</summary>
        public static readonly BracketMode Balanced = new()
        {
            ModeType = "Balanced", 
            StopTicks = 12,
            TargetTicks = 18,
            BreakevenAfterTicks = 8,
            TrailTicks = 6
        };
        
        /// <summary>Aggressive bracket: Wider stops, larger targets</summary>
        public static readonly BracketMode Aggressive = new()
        {
            ModeType = "Aggressive",
            StopTicks = 15,
            TargetTicks = 25,
            BreakevenAfterTicks = 10,
            TrailTicks = 8
        };
        
        /// <summary>Scalping bracket: Very tight stops and targets</summary>
        public static readonly BracketMode Scalping = new()
        {
            ModeType = "Scalping",
            StopTicks = 8,
            TargetTicks = 12,
            BreakevenAfterTicks = 4,
            TrailTicks = 3
        };
        
        /// <summary>Swing bracket: Wide stops for trend following</summary>
        public static readonly BracketMode Swing = new()
        {
            ModeType = "Swing",
            StopTicks = 18,
            TargetTicks = 30,
            BreakevenAfterTicks = 12,
            TrailTicks = 10
        };
        
        /// <summary>Get all predefined bracket modes</summary>
        public static readonly BracketMode[] All = 
        {
            Conservative, Balanced, Aggressive, Scalping, Swing
        };
    }
}

/// <summary>
/// Parameter bundle for strategy-parameter combinations in Neural UCB
/// Replaces hardcoded values with learned parameter selections
/// 
/// Example: S2-1.3x-0.65-C12T18 means Strategy S2 with 1.3x position multiplier, 0.65 confidence threshold,
/// Conservative bracket mode with 12 stop ticks and 18 target ticks
/// </summary>
public record ParameterBundle
{
    /// <summary>
    /// Strategy identifier (S2, S3, S6, S11)
    /// </summary>
    public string Strategy { get; init; } = string.Empty;
    
    /// <summary>
    /// Position size multiplier (replaces hardcoded 2.5)
    /// Range: 1.0x to 1.6x for conservative to aggressive sizing
    /// </summary>
    public decimal Mult { get; init; }
    
    /// <summary>
    /// Confidence threshold (replaces hardcoded 0.7)
    /// Range: 0.60 to 0.70 for different confidence levels
    /// </summary>
    public decimal Thr { get; init; }
    
    /// <summary>
    /// Bracket mode for order management (replaces hardcoded bracket settings)
    /// Enables learning optimal stop/target combinations
    /// </summary>
    public BracketMode BracketMode { get; init; } = BracketMode.Presets.Conservative;
    
    /// <summary>
    /// Unique bundle identifier: Strategy-Multiplier-Threshold-BracketMode
    /// Example: "S2-1.3x-0.65-C12T18"
    /// </summary>
    public string BundleId => $"{Strategy}-{Mult:F1}x-{Thr:F2}-{BracketMode.ModeId}";
    
    /// <summary>
    /// Human-readable description
    /// </summary>
    public string Description => $"Strategy {Strategy} with {Mult:F1}x sizing, {Thr:P0} confidence, and {BracketMode.Description}";
    
    /// <summary>
    /// Validate bundle parameters are within safe ranges
    /// </summary>
    public bool IsValid =>
        !string.IsNullOrEmpty(Strategy) &&
        Mult >= 1.0m && Mult <= 1.6m &&
        Thr >= 0.60m && Thr <= 0.70m &&
        BracketMode.IsValid;
    
    /// <summary>
    /// Create a safe default parameter bundle
    /// </summary>
    public static ParameterBundle CreateSafeDefault()
    {
        return new ParameterBundle
        {
            Strategy = "S2",
            Mult = 1.0m,
            Thr = 0.65m,
            BracketMode = BracketMode.Presets.Conservative
        };
    }
}

/// <summary>
/// Factory for creating predefined parameter bundles
/// Creates all valid strategy-parameter combinations for Neural UCB including bracket modes
/// </summary>
public static class ParameterBundleFactory
{
    /// <summary>
    /// Available strategies from existing system
    /// </summary>
    public static readonly string[] Strategies = { "S2", "S3", "S6", "S11" };
    
    /// <summary>
    /// Position multiplier options: Conservative, Balanced, Aggressive
    /// </summary>
    public static readonly decimal[] Multipliers = { 1.0m, 1.3m, 1.6m };
    
    /// <summary>
    /// Confidence threshold options: Lower, Medium, Higher
    /// </summary>
    public static readonly decimal[] Thresholds = { 0.60m, 0.65m, 0.70m };
    
    /// <summary>
    /// Available bracket modes for learning
    /// </summary>
    public static readonly BracketMode[] BracketModes = BracketMode.Presets.All;
    
    /// <summary>
    /// Create all valid parameter bundles (180 total combinations)
    /// 4 strategies × 3 multipliers × 3 thresholds × 5 bracket modes = 180 bundles
    /// </summary>
    public static List<ParameterBundle> CreateAllBundles()
    {
        var bundles = new List<ParameterBundle>();
        
        foreach (var strategy in Strategies)
        {
            foreach (var multiplier in Multipliers)
            {
                foreach (var threshold in Thresholds)
                {
                    foreach (var bracketMode in BracketModes)
                    {
                        var bundle = new ParameterBundle
                        {
                            Strategy = strategy,
                            Mult = multiplier,
                            Thr = threshold,
                            BracketMode = bracketMode
                        };
                        
                        if (bundle.IsValid)
                        {
                            bundles.Add(bundle);
                        }
                    }
                }
            }
        }
        
        return bundles;
    }
    
    /// <summary>
    /// Create bundles for a specific strategy
    /// </summary>
    public static List<ParameterBundle> CreateBundlesForStrategy(string strategy)
    {
        return CreateAllBundles()
            .Where(b => b.Strategy == strategy)
            .ToList();
    }
    
    /// <summary>
    /// Get default conservative bundle for emergency fallback
    /// </summary>
    public static ParameterBundle GetDefaultBundle()
    {
        return new ParameterBundle
        {
            Strategy = "S2", // Most conservative strategy
            Mult = 1.0m,     // Conservative sizing
            Thr = 0.70m,     // High confidence requirement
            BracketMode = BracketMode.Presets.Conservative // Conservative bracket
        };
    }
    
    /// <summary>
    /// Parse bundle from string identifier
    /// Example: "S2-1.3x-0.65-C12T18" -> ParameterBundle
    /// Legacy format "S2-1.3x-0.65" defaults to Conservative bracket mode
    /// </summary>
    public static ParameterBundle? ParseBundle(string bundleId)
    {
        if (string.IsNullOrEmpty(bundleId))
            return null;
            
        var parts = bundleId.Split('-');
        if (parts.Length < 3 || parts.Length > 4)
            return null;
            
        var strategy = parts[0];
        
        // Parse multiplier (remove 'x' suffix)
        if (!decimal.TryParse(parts[1].TrimEnd('x'), out var multiplier))
            return null;
            
        // Parse threshold
        if (!decimal.TryParse(parts[2], out var threshold))
            return null;
        
        // Parse bracket mode (optional for backward compatibility)
        var bracketMode = BracketMode.Presets.Conservative; // Default fallback
        if (parts.Length == 4)
        {
            var modeId = parts[3];
            bracketMode = BracketModes.FirstOrDefault(bm => bm.ModeId == modeId) 
                         ?? BracketMode.Presets.Conservative;
        }
            
        var bundle = new ParameterBundle
        {
            Strategy = strategy,
            Mult = multiplier,
            Thr = threshold,
            BracketMode = bracketMode
        };
        
        return bundle.IsValid ? bundle : null;
    }
    
    /// <summary>
    /// Get bundle recommendations based on market conditions
    /// Includes bracket mode optimization for different market environments
    /// </summary>
    public static List<ParameterBundle> GetRecommendedBundles(MarketCondition condition)
    {
        return condition switch
        {
            MarketCondition.Volatile => CreateAllBundles()
                .Where(b => b.Mult <= 1.3m && b.Thr >= 0.65m && 
                           (b.BracketMode.ModeType == "Conservative" || b.BracketMode.ModeType == "Scalping")) // Tighter brackets for volatility
                .ToList(),
                
            MarketCondition.Trending => CreateAllBundles()
                .Where(b => b.Mult >= 1.3m && b.Thr >= 0.60m && 
                           (b.BracketMode.ModeType == "Aggressive" || b.BracketMode.ModeType == "Swing")) // Wider brackets for trends
                .ToList(),
                
            MarketCondition.Ranging => CreateAllBundles()
                .Where(b => b.Mult <= 1.3m && b.Thr <= 0.65m && 
                           (b.BracketMode.ModeType == "Balanced" || b.BracketMode.ModeType == "Scalping")) // Balanced brackets for ranges
                .ToList(),
                
            MarketCondition.LowVolume => CreateAllBundles()
                .Where(b => b.Mult <= 1.0m && 
                           (b.BracketMode.ModeType == "Conservative" || b.BracketMode.ModeType == "Scalping")) // Tight brackets for low volume
                .ToList(),
                
            MarketCondition.HighVolume => CreateAllBundles()
                .Where(b => b.Thr >= 0.65m && 
                           (b.BracketMode.ModeType == "Aggressive" || b.BracketMode.ModeType == "Swing")) // Wider brackets for high volume
                .ToList(),
                
            _ => CreateAllBundles() // Unknown conditions: use all bundles
        };
    }
}

/// <summary>
/// Market condition enumeration for bundle recommendations
/// </summary>
public enum MarketCondition
{
    Unknown,
    Volatile,
    Trending,
    Ranging,
    LowVolume,
    HighVolume
}

/// <summary>
/// Bundle selection result from Neural UCB
/// </summary>
public record BundleSelection
{
    /// <summary>
    /// Selected parameter bundle
    /// </summary>
    public ParameterBundle Bundle { get; init; } = ParameterBundleFactory.GetDefaultBundle();
    
    /// <summary>
    /// UCB confidence value for this selection
    /// </summary>
    public decimal UcbValue { get; init; }
    
    /// <summary>
    /// Neural network prediction value
    /// </summary>
    public decimal Prediction { get; init; }
    
    /// <summary>
    /// Uncertainty estimate
    /// </summary>
    public decimal Uncertainty { get; init; }
    
    /// <summary>
    /// Selection reason for debugging
    /// </summary>
    public string SelectionReason { get; init; } = string.Empty;
    
    /// <summary>
    /// Context features used for selection
    /// </summary>
    public Dictionary<string, decimal> ContextFeatures { get; init; } = new();
    
    /// <summary>
    /// Timestamp of selection
    /// </summary>
    public DateTime Timestamp { get; init; } = DateTime.UtcNow;
}