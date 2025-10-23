using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using BotCore.Services;
using TradingBot.UnifiedOrchestrator.Services;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Integrates live TopstepX bar/tick data with multi-timeframe services.
/// Subscribes to TopstepX adapter and feeds data to BarAggregationService and TickBufferService.
/// 
/// MODE DISTINCTION (CRITICAL):
/// - SUNDAY LAB MODE: This service is DISABLED (LAB_MODE=1 detected)
///   → MultiTimeframeDataLoader used for heavy neural network training
///   → Full gradient descent on 90 days of synchronized 5m + 1m + tick data
///   → Trains multi-branch models, applies overfitting prevention
///   → Outputs frozen ONNX models for Terminal Mode
/// 
/// - TERMINAL MODE (Mon-Sat): This service is ENABLED
///   → Subscribes to live TopstepX feed
///   → Feeds real-time data to BarAggregationService and TickBufferService
///   → Uses already-trained ONNX models for inference (NOT training)
///   → Lightweight online calibration via MultiTimeframeOnlineLearning (NOT full retraining)
/// </summary>
internal sealed class MultiTimeframeDataIntegrationService : IHostedService
{
    private readonly ILogger<MultiTimeframeDataIntegrationService> _logger;
    private readonly TopstepXAdapterService? _topstepXAdapter;
    private readonly BarAggregationService _barAggregator;
    private readonly TickBufferService _tickBuffer;
    
    public MultiTimeframeDataIntegrationService(
        ILogger<MultiTimeframeDataIntegrationService> logger,
        BarAggregationService barAggregator,
        TickBufferService tickBuffer,
        TopstepXAdapterService? topstepXAdapter = null)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _barAggregator = barAggregator ?? throw new ArgumentNullException(nameof(barAggregator));
        _tickBuffer = tickBuffer ?? throw new ArgumentNullException(nameof(tickBuffer));
        _topstepXAdapter = topstepXAdapter; // Optional - not available in Lab Mode
    }
    
    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Check if in Lab Mode (SUNDAY LAB MODE - don't subscribe to live feed during heavy training)
        // In Sunday Lab Mode, MultiTimeframeDataLoader is used instead for historical data training
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        if (labMode == "1" || labMode?.ToLowerInvariant() == "true" || _topstepXAdapter == null)
        {
            _logger.LogInformation(
                "[MTF-INTEGRATION] SUNDAY LAB MODE detected - skipping live data feed integration. " +
                "MultiTimeframeDataLoader will be used for historical training instead.");
            return Task.CompletedTask;
        }
        
        // Subscribe to bar events from TopstepX adapter (TERMINAL MODE ONLY)
        _topstepXAdapter.SubscribeToBarEvents(OnBarEventReceived);
        
        _logger.LogInformation(
            "[MTF-INTEGRATION] TERMINAL MODE - Multi-timeframe data integration started. " +
            "Feeding live bars to aggregation services for real-time inference.");
        
        return Task.CompletedTask;
    }
    
    public Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[MTF-INTEGRATION] Multi-timeframe data integration stopped");
        return Task.CompletedTask;
    }
    
    private void OnBarEventReceived(BarEventData barData)
    {
        try
        {
            // Extract symbol (instrument)
            var symbol = barData.Instrument;
            
            // Convert bar data to tick data for aggregation
            // Use close price and volume as the "tick"
            var tick = new TickData
            {
                Timestamp = barData.Timestamp,
                Price = (double)barData.Close,
                Size = (double)barData.Volume
            };
            
            // Feed to bar aggregator (builds 1m and 5m bars)
            _barAggregator.OnTick(symbol, tick);
            
            // Feed to tick buffer (for execution approval)
            _tickBuffer.AddTick(symbol, tick);
            
            _logger.LogDebug(
                "[MTF-INTEGRATION] Processed bar for {Symbol}: {Price} @ {Time}",
                symbol, barData.Close, barData.Timestamp);
        }
        catch (Exception ex)
        {
            _logger.LogError(
                ex,
                "[MTF-INTEGRATION] Error processing bar event for {Instrument}",
                barData.Instrument);
        }
    }
}
