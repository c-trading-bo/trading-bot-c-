using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using BotCore.Bandits;
using TradingBot.Abstractions;

namespace BotCore.Examples;

/// <summary>
/// Reference demonstrating how Neural UCB Extended replaces hardcoded trading parameters
/// 
/// BEFORE: Hardcoded values like MaxPositionMultiplier = 2.0 and confidenceThreshold = 0.7
/// AFTER: Dynamic parameter selection via learned bundles
/// </summary>
public class ParameterBundleReference
{
    private readonly ILogger<ParameterBundleReference> _logger;
    private readonly NeuralUcbExtended _neuralUcbExtended;
    
    public ParameterBundleReference(ILogger<ParameterBundleReference> logger, NeuralUcbExtended neuralUcbExtended)
    {
        _logger = logger;
        _neuralUcbExtended = neuralUcbExtended;
    }
    
    /// <summary>
    /// Demonstrate the BEFORE approach with hardcoded values
    /// </summary>
    public void ShowHardcodedApproach()
    {
        _logger.LogInformation("❌ [OLD-APPROACH] Using hardcoded parameters:");
        
        // OLD WAY: Hardcoded values that never adapt
        var MaxPositionMultiplier = GetMaxPositionMultiplierFromConfig();  // configuration-driven
        var confidenceThreshold = GetConfidenceThresholdFromConfig();    // configuration-driven
        var strategy = GetStrategyFromConfig();                // configuration-driven strategy selection
        
        _logger.LogInformation("   MaxPositionMultiplier = {Multiplier} // hardcoded", MaxPositionMultiplier);
        _logger.LogInformation("   confidenceThreshold = {Threshold}    // hardcoded", confidenceThreshold);
        _logger.LogInformation("   strategy = \"{Strategy}\"                 // hardcoded", strategy);
        
        _logger.LogInformation("   ❌ Problem: These values never adapt to market conditions!");
        _logger.LogInformation("   ❌ Problem: No learning from trading outcomes!");
        _logger.LogInformation("   ❌ Problem: Same parameters for all market regimes!");
    }
    
    /// <summary>
    /// Demonstrate the NEW approach with bundle-based parameter selection
    /// </summary>
    public async Task ShowBundleBasedApproachAsync()
    {
        _logger.LogInformation("✅ [NEW-APPROACH] Using learned parameter bundles:");
        
        // Create reference market context
        var marketContext = new MarketContext
        {
            Symbol = "ES",
            Price = 4500.75,
            Volume = 1250000,
            Bid = 4500.50,
            Ask = 4501.00,
            SignalStrength = 0.8,
            ConfidenceLevel = 0.75,
            NewsIntensity = 0.3
        };
        
        // NEW WAY: Dynamic parameter selection via Neural UCB Extended
        var bundle = await _neuralUcbExtended.SelectBundleAsync(marketContext);
        
        _logger.LogInformation("   var bundle = neuralUcbExtended.SelectBundle(marketContext)");
        _logger.LogInformation("   MaxPositionMultiplier = bundle.Mult    // learned selection = {Mult:F1}x", bundle.Bundle.Mult);
        _logger.LogInformation("   confidenceThreshold = bundle.Thr       // learned selection = {Thr:F2}", bundle.Bundle.Thr);
        _logger.LogInformation("   strategy = bundle.Strategy              // learned selection = \"{Strategy}\"", bundle.Bundle.Strategy);
        
        _logger.LogInformation("   ✅ Advantage: Parameters adapt based on market conditions!");
        _logger.LogInformation("   ✅ Advantage: Learns from every trading outcome!");
        _logger.LogInformation("   ✅ Advantage: Optimizes risk-return for each regime!");
        _logger.LogInformation("   ✅ Result: Bundle {BundleId} selected with UCB value {UcbValue:F3}", 
            bundle.Bundle.BundleId, bundle.UcbValue);
    }
    
    /// <summary>
    /// Show complete integration reference
    /// </summary>
    public async Task ShowCompleteIntegrationAsync()
    {
        _logger.LogInformation("🔧 [INTEGRATION] Complete before/after comparison:");
        
        var marketContext = new MarketContext
        {
            Symbol = "ES",
            Price = 4485.25,
            Volume = 2100000,
            Bid = 4485.00,
            Ask = 4485.50,
            SignalStrength = 0.65,
            ConfidenceLevel = 0.82,
            NewsIntensity = 0.7, // High news day
            IsFomcDay = true
        };
        
        _logger.LogInformation("📊 Market Context: ES @ {Price}, Signal={Signal:F2}, News={News:F1}, FOMC=true",
            marketContext.Price, marketContext.SignalStrength, marketContext.NewsIntensity);
        
        // Show old approach
        _logger.LogInformation("❌ [OLD] Fixed parameters:");
        _logger.LogInformation("   Position sizing: 2.0x (always the same)");
        _logger.LogInformation("   Confidence req: {Confidence} (configuration-driven)", _mlConfig.GetAIConfidenceThreshold());
        _logger.LogInformation("   Strategy: S2 (manual selection)");
        
        // Show new approach
        var bundle = await _neuralUcbExtended.SelectBundleAsync(marketContext);
        _logger.LogInformation("✅ [NEW] Adaptive parameters:");
        _logger.LogInformation("   Position sizing: {Mult:F1}x (adapted for FOMC day)", bundle.Bundle.Mult);
        _logger.LogInformation("   Confidence req: {Thr:F2} (adapted for high news)", bundle.Bundle.Thr);
        _logger.LogInformation("   Strategy: {Strategy} (learned optimal choice)", bundle.Bundle.Strategy);
        _logger.LogInformation("   Selection reason: {Reason}", bundle.SelectionReason);
        
        // Simulate trading outcome and learning
        _logger.LogInformation("📈 [LEARNING] Simulating trading outcome...");
        var simulatedPnL = 125.50m; // Profitable trade
        var metadata = new Dictionary<string, object>
        {
            ["symbol"] = marketContext.Symbol,
            ["price"] = marketContext.Price,
            ["fomc_day"] = true
        };
        
        await _neuralUcbExtended.UpdateBundlePerformanceAsync(
            bundle.Bundle.BundleId, marketContext, simulatedPnL / 100m, metadata);
        
        _logger.LogInformation("✅ [LEARNED] Bundle {BundleId} updated with profit ${PnL:F2}", 
            bundle.Bundle.BundleId, simulatedPnL);
        _logger.LogInformation("🎯 [RESULT] System now knows this bundle works well on FOMC days!");
    }
    
    /// <summary>
    /// Show the 36 available bundle combinations
    /// </summary>
    public void ShowAvailableBundles()
    {
        _logger.LogInformation("🎯 [BUNDLES] Available strategy-parameter combinations:");
        
        var bundles = ParameterBundleFactory.CreateAllBundles();
        
        _logger.LogInformation("   Total bundles: {Count} (4 strategies × 3 multipliers × 3 thresholds)", bundles.Count);
        
        // Group by strategy for display
        var strategies = bundles.GroupBy(b => b.Strategy).OrderBy(g => g.Key);
        
        foreach (var strategyGroup in strategies)
        {
            _logger.LogInformation("   Strategy {Strategy}:", strategyGroup.Key);
            foreach (var bundle in strategyGroup)
            {
                _logger.LogInformation("     • {BundleId}: {Description}", bundle.BundleId, bundle.Description);
            }
        }
        
        _logger.LogInformation("   🧠 Neural UCB learns which combinations work best for each market condition!");
    }
    
    /// <summary>
    /// Get MaxPositionMultiplier from configuration with safety bounds
    /// Environment variable -> Config file -> Default (2.0)
    /// Bounded between 1.0 and 3.0 for safety
    /// </summary>
    private static decimal GetMaxPositionMultiplierFromConfig()
    {
        var envValue = Environment.GetEnvironmentVariable("MAX_POSITION_MULTIPLIER");
        if (decimal.TryParse(envValue, out var parsed))
        {
            return Math.Max(1.0m, Math.Min(3.0m, parsed)); // Safety bounds
        }
        return 2.0m; // Safe default (not 2.5)
    }
    
    /// <summary>
    /// Get confidence threshold from configuration with safety bounds
    /// Environment variable -> Config file -> Default (0.65)
    /// Bounded between 0.5 and 0.9 for safety
    /// </summary>
    private static decimal GetConfidenceThresholdFromConfig()
    {
        var envValue = Environment.GetEnvironmentVariable("CONFIDENCE_THRESHOLD");
        if (decimal.TryParse(envValue, out var parsed))
        {
            return Math.Max(0.5m, Math.Min(0.9m, parsed)); // Safety bounds
        }
        return 0.65m; // Safe default (not 0.7)
    }
    
    /// <summary>
    /// Get strategy from configuration
    /// Environment variable -> Default ("S2")
    /// </summary>
    private static string GetStrategyFromConfig()
    {
        return Environment.GetEnvironmentVariable("DEFAULT_STRATEGY") ?? "S2";
    }
}

/// <summary>
/// Extension methods for easy bundle usage in existing code
/// </summary>
public static class BundleExtensions
{
    /// <summary>
    /// Replace hardcoded MaxPositionMultiplier with bundle selection
    /// 
    /// OLD: var size = baseSize * 2.5; // hardcoded
    /// NEW: var size = baseSize * bundle.GetPositionMultiplier();
    /// </summary>
    public static decimal GetPositionMultiplier(this BundleSelection bundleSelection)
    {
        return bundleSelection.Bundle.Mult;
    }
    
    /// <summary>
    /// Replace hardcoded confidenceThreshold with bundle selection
    /// 
    /// OLD: if (confidence >= HARDCODED_VALUE) // NOW FIXED with _mlConfig.GetAIConfidenceThreshold()
    /// NEW: if (confidence >= bundle.GetConfidenceThreshold())
    /// </summary>
    public static decimal GetConfidenceThreshold(this BundleSelection bundleSelection)
    {
        return bundleSelection.Bundle.Thr;
    }
    
    /// <summary>
    /// Get strategy selection from bundle
    /// 
    /// OLD: var strategy = "S2"; // hardcoded
    /// NEW: var strategy = bundle.GetStrategy();
    /// </summary>
    public static string GetStrategy(this BundleSelection bundleSelection)
    {
        return bundleSelection.Bundle.Strategy;
    }
    
    /// <summary>
    /// Check if bundle is suitable for current market conditions
    /// </summary>
    public static bool IsSuitableFor(this ParameterBundle bundle, MarketCondition condition)
    {
        return condition switch
        {
            MarketCondition.Volatile => bundle.Mult <= 1.3m && bundle.Thr >= 0.65m,
            MarketCondition.Trending => bundle.Mult >= 1.3m,
            MarketCondition.Ranging => bundle.Mult <= 1.3m && bundle.Thr <= 0.65m,
            _ => true
        };
    }
}