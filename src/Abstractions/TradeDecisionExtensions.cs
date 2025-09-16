using System;
using System.Collections.Generic;

namespace TradingBot.Abstractions;

/// <summary>
/// Extension methods for TradingDecision to implement critical trading logic fixes
/// Addresses: Size Floor Mechanism to prevent smart direction + zero size = HOLD
/// </summary>
public static class TradeDecisionExtensions
{
    /// <summary>
    /// Applies minimum size floor when confidence is good but size is zero
    /// Critical Fix: Prevents "smart direction, zero size" from becoming HOLD
    /// </summary>
    /// <param name="decision">The trading decision to process</param>
    /// <param name="minimumConfidenceThreshold">Minimum confidence to apply size floor (default: 55%)</param>
    /// <param name="minimumSize">Minimum size to apply when confidence is good (default: 1 contract)</param>
    /// <returns>Modified decision with size floor applied if conditions are met</returns>
    public static TradingDecision ApplySizeFloor(
        this TradingDecision decision, 
        decimal minimumConfidenceThreshold = 0.55m, 
        decimal minimumSize = 1m)
    {
        // Only apply size floor if:
        // 1. Action is not already HOLD
        // 2. Confidence meets threshold 
        // 3. Current quantity/size is zero or negative
        if (decision.Action != TradingAction.Hold && 
            decision.Confidence >= minimumConfidenceThreshold && 
            decision.Quantity <= 0)
        {
            var originalQuantity = decision.Quantity;
            decision.Quantity = minimumSize;
            
            // Add reasoning for audit trail
            if (decision.Reasoning == null)
                decision.Reasoning = new Dictionary<string, object>();
                
            decision.Reasoning["SizeFloorApplied"] = true;
            decision.Reasoning["OriginalQuantity"] = originalQuantity;
            decision.Reasoning["AppliedMinimumSize"] = minimumSize;
            decision.Reasoning["ConfidenceAtApplication"] = decision.Confidence;
            decision.Reasoning["SizeFloorReason"] = "Prevented smart direction with zero size from becoming HOLD";
        }
        
        return decision;
    }
    
    /// <summary>
    /// Applies size floor with enhanced confidence-based sizing
    /// Uses confidence level to determine appropriate minimum size
    /// </summary>
    /// <param name="decision">The trading decision to process</param>
    /// <param name="minimumConfidenceThreshold">Minimum confidence to apply size floor</param>
    /// <returns>Modified decision with confidence-based size floor</returns>
    public static TradingDecision ApplyConfidenceBasedSizeFloor(
        this TradingDecision decision, 
        decimal minimumConfidenceThreshold = 0.55m)
    {
        if (decision.Action != TradingAction.Hold && 
            decision.Confidence >= minimumConfidenceThreshold && 
            decision.Quantity <= 0)
        {
            var originalQuantity = decision.Quantity;
            
            // Confidence-based sizing:
            // 55-65%: 1 contract
            // 65-75%: 2 contracts  
            // 75%+:   3 contracts (cap for safety)
            decimal sizeByConfidence = decision.Confidence switch
            {
                >= 0.75m => 3m,
                >= 0.65m => 2m,
                >= 0.55m => 1m,
                _ => 0m
            };
            
            decision.Quantity = sizeByConfidence;
            
            // Enhanced reasoning
            if (decision.Reasoning == null)
                decision.Reasoning = new Dictionary<string, object>();
                
            decision.Reasoning["ConfidenceBasedSizeFloorApplied"] = true;
            decision.Reasoning["OriginalQuantity"] = originalQuantity;
            decision.Reasoning["ConfidenceBasedSize"] = sizeByConfidence;
            decision.Reasoning["ConfidenceLevel"] = decision.Confidence;
            decision.Reasoning["SizeFloorReason"] = "Applied confidence-based size floor to prevent HOLD conversion";
        }
        
        return decision;
    }
    
    /// <summary>
    /// Validates that a decision is ready for execution (not zero-sized unless HOLD)
    /// </summary>
    /// <param name="decision">The trading decision to validate</param>
    /// <returns>True if decision is execution-ready, false if needs size floor treatment</returns>
    public static bool IsExecutionReady(this TradingDecision decision)
    {
        // HOLD actions can have zero size
        if (decision.Action == TradingAction.Hold)
            return true;
            
        // Buy/Sell actions must have positive size
        if (decision.Action == TradingAction.Buy || decision.Action == TradingAction.Sell)
            return decision.Quantity > 0;
            
        // Other actions also need positive size
        return decision.Quantity > 0;
    }
    
    /// <summary>
    /// Gets a summary of size floor application for logging
    /// </summary>
    /// <param name="decision">The trading decision</param>
    /// <returns>Human-readable summary of size floor logic</returns>
    public static string GetSizeFloorSummary(this TradingDecision decision)
    {
        if (decision.Reasoning?.ContainsKey("SizeFloorApplied") == true ||
            decision.Reasoning?.ContainsKey("ConfidenceBasedSizeFloorApplied") == true)
        {
            var originalQty = decision.Reasoning.GetValueOrDefault("OriginalQuantity", 0);
            var appliedSize = decision.Quantity;
            var confidence = decision.Confidence;
            
            return $"Size floor applied: {originalQty} -> {appliedSize} (confidence: {confidence:P1})";
        }
        
        return decision.IsExecutionReady() ? "Ready for execution" : "Needs size floor treatment";
    }
}