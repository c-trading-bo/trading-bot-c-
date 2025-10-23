using Microsoft.Extensions.Logging;
using BotCore.Services;
using BotCore.ML;
using System;
using System.Collections.Generic;
using System.Linq;

namespace BotCore.Brain;

/// <summary>
/// Multi-timeframe adapter for UnifiedTradingBrain.
/// Provides multi-timeframe features to trading decisions.
/// 
/// Phase 6: UnifiedTradingBrain Integration (Week 7)
/// - Subscribe to bar completion events
/// - Provide multi-timeframe features to decision engine
/// - Minimal integration point for existing UnifiedTradingBrain
/// 
/// Design principles:
/// - Non-invasive: Adapter pattern doesn't modify existing brain
/// - Production-ready: Can be added via dependency injection
/// - Thread-safe: Concurrent access support
/// </summary>
public class MultiTimeframeBrainAdapter : IDisposable
{
    private readonly ILogger<MultiTimeframeBrainAdapter> _logger;
    private readonly BarAggregationService _barAggregator;
    private readonly LiveMultiTimeframeFeatureComputer _featureComputer;
    private readonly ExecutionApprovalService? _executionApproval;
    private bool _disposed;
    
    // Feature availability tracking
    private readonly Dictionary<string, bool> _featuresReady = new();
    private readonly object _readyLock = new();
    
    public MultiTimeframeBrainAdapter(
        ILogger<MultiTimeframeBrainAdapter> logger,
        BarAggregationService barAggregator,
        LiveMultiTimeframeFeatureComputer featureComputer,
        ExecutionApprovalService? executionApproval = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _barAggregator = barAggregator ?? throw new ArgumentNullException(nameof(barAggregator));
        _featureComputer = featureComputer ?? throw new ArgumentNullException(nameof(featureComputer));
        _executionApproval = executionApproval;
        
        // Subscribe to 5m bar completion events (strategic decision points)
        _barAggregator.Bar5mCompleted += OnBar5mCompleted;
        
        _logger.LogInformation(
            "[MTF_ADAPTER] Multi-timeframe adapter initialized and subscribed to bar events");
    }
    
    /// <summary>
    /// Get multi-timeframe features for a symbol if available.
    /// Returns null if features not yet ready.
    /// </summary>
    /// <param name="symbol">Symbol (e.g., "ES", "NQ")</param>
    /// <returns>Feature dictionary or null</returns>
    public Dictionary<string, double>? GetMultiTimeframeFeatures(string symbol)
    {
        if (string.IsNullOrWhiteSpace(symbol))
        {
            return null;
        }
        
        // Check if features are ready
        lock (_readyLock)
        {
            if (!_featuresReady.GetValueOrDefault(symbol, false))
            {
                _logger.LogDebug(
                    "[MTF_ADAPTER] Multi-timeframe features not yet ready for {Symbol}",
                    symbol);
                return null;
            }
        }
        
        return _featureComputer.GetLatestFeatures(symbol);
    }
    
    /// <summary>
    /// Check if execution should be approved based on tick microstructure.
    /// Returns true if execution approval service not available (fail-open).
    /// </summary>
    /// <param name="symbol">Symbol</param>
    /// <param name="direction">Trade direction (1=buy, -1=sell)</param>
    /// <param name="quantity">Quantity</param>
    /// <param name="price">Expected price</param>
    /// <returns>True if approved or approval service not available</returns>
    public bool ShouldApproveExecution(
        string symbol,
        int direction,
        double quantity,
        double price)
    {
        if (_executionApproval == null)
        {
            // Fail-open: approve if service not available
            return true;
        }
        
        try
        {
            var approval = _executionApproval.EvaluateExecution(
                symbol, direction, quantity, price);
            
            return approval.Approved;
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "[MTF_ADAPTER] Error evaluating execution approval for {Symbol}, defaulting to approve",
                symbol);
            
            // Fail-open on error
            return true;
        }
    }
    
    /// <summary>
    /// Get execution approval details for logging/analysis.
    /// </summary>
    public ExecutionApproval? GetExecutionApprovalDetails(
        string symbol,
        int direction,
        double quantity,
        double price)
    {
        if (_executionApproval == null)
        {
            return null;
        }
        
        try
        {
            return _executionApproval.EvaluateExecution(
                symbol, direction, quantity, price);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "[MTF_ADAPTER] Error getting execution approval details for {Symbol}",
                symbol);
            return null;
        }
    }
    
    /// <summary>
    /// Check if multi-timeframe features are ready for a symbol.
    /// </summary>
    public bool AreFeaturesReady(string symbol)
    {
        lock (_readyLock)
        {
            return _featuresReady.GetValueOrDefault(symbol, false);
        }
    }
    
    /// <summary>
    /// Get feature metadata (timestamps, counts, etc).
    /// </summary>
    public FeatureMetadata? GetFeatureMetadata(string symbol)
    {
        return _featureComputer.GetFeatureMetadata(symbol);
    }
    
    /// <summary>
    /// Get average feature computation time for performance monitoring.
    /// </summary>
    public double GetAverageComputeTimeMs()
    {
        return _featureComputer.GetAverageComputeTimeMs();
    }
    
    /// <summary>
    /// Handle 5m bar completion - marks features as ready for strategic decisions.
    /// </summary>
    private void OnBar5mCompleted(object? sender, BarCompletedEventArgs e)
    {
        try
        {
            _logger.LogInformation(
                "[MTF_ADAPTER] 5m bar completed for {Symbol} at {Timestamp}, features ready for strategic decision",
                e.Symbol, e.Bar.Timestamp);
            
            lock (_readyLock)
            {
                _featuresReady[e.Symbol] = true;
            }
            
            // Features are automatically computed by LiveMultiTimeframeFeatureComputer
            // which is subscribed to the same bar completion events
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "[MTF_ADAPTER] Error handling 5m bar completion for {Symbol}",
                e.Symbol);
        }
    }
    
    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }
        
        _disposed = true;
        
        // Unsubscribe from events
        _barAggregator.Bar5mCompleted -= OnBar5mCompleted;
        
        _logger.LogInformation("[MTF_ADAPTER] Multi-timeframe adapter disposed");
    }
}
