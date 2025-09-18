using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Bandits;

/// <summary>
/// Parameter bundle for strategy-parameter combinations in Neural UCB
/// Replaces hardcoded values with learned parameter selections
/// 
/// Example: S2-1.3x-0.65 means Strategy S2 with 1.3x position multiplier and 0.65 confidence threshold
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
    /// Unique bundle identifier: Strategy-Multiplier-Threshold
    /// Example: "S2-1.3x-0.65"
    /// </summary>
    public string BundleId => $"{Strategy}-{Mult:F1}x-{Thr:F2}";
    
    /// <summary>
    /// Human-readable description
    /// </summary>
    public string Description => $"Strategy {Strategy} with {Mult:F1}x sizing and {Thr:P0} confidence";
    
    /// <summary>
    /// Validate bundle parameters are within safe ranges
    /// </summary>
    public bool IsValid =>
        !string.IsNullOrEmpty(Strategy) &&
        Mult >= 1.0m && Mult <= 1.6m &&
        Thr >= 0.60m && Thr <= 0.70m;
    
    /// <summary>
    /// Create a safe default parameter bundle
    /// </summary>
    public static ParameterBundle CreateSafeDefault()
    {
        return new ParameterBundle
        {
            Strategy = "S2",
            Mult = 1.0m,
            Thr = 0.65m
        };
    }
}

/// <summary>
/// Factory for creating predefined parameter bundles
/// Creates all valid strategy-parameter combinations for Neural UCB
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
    /// Create all valid parameter bundles (36 total combinations)
    /// 4 strategies × 3 multipliers × 3 thresholds = 36 bundles
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
                    var bundle = new ParameterBundle
                    {
                        Strategy = strategy,
                        Mult = multiplier,
                        Thr = threshold
                    };
                    
                    if (bundle.IsValid)
                    {
                        bundles.Add(bundle);
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
            Thr = 0.70m      // High confidence requirement
        };
    }
    
    /// <summary>
    /// Parse bundle from string identifier
    /// Example: "S2-1.3x-0.65" -> ParameterBundle
    /// </summary>
    public static ParameterBundle? ParseBundle(string bundleId)
    {
        if (string.IsNullOrEmpty(bundleId))
            return null;
            
        var parts = bundleId.Split('-');
        if (parts.Length != 3)
            return null;
            
        var strategy = parts[0];
        
        // Parse multiplier (remove 'x' suffix)
        if (!decimal.TryParse(parts[1].TrimEnd('x'), out var multiplier))
            return null;
            
        // Parse threshold
        if (!decimal.TryParse(parts[2], out var threshold))
            return null;
            
        var bundle = new ParameterBundle
        {
            Strategy = strategy,
            Mult = multiplier,
            Thr = threshold
        };
        
        return bundle.IsValid ? bundle : null;
    }
    
    /// <summary>
    /// Get bundle recommendations based on market conditions
    /// </summary>
    public static List<ParameterBundle> GetRecommendedBundles(MarketCondition condition)
    {
        return condition switch
        {
            MarketCondition.Volatile => CreateAllBundles()
                .Where(b => b.Mult <= 1.3m && b.Thr >= 0.65m) // Conservative sizing, higher confidence
                .ToList(),
                
            MarketCondition.Trending => CreateAllBundles()
                .Where(b => b.Mult >= 1.3m && b.Thr >= 0.60m) // Aggressive sizing, flexible confidence
                .ToList(),
                
            MarketCondition.Ranging => CreateAllBundles()
                .Where(b => b.Mult <= 1.3m && b.Thr <= 0.65m) // Moderate sizing, lower confidence OK
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