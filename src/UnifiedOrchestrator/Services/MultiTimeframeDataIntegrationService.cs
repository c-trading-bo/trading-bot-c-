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
/// </summary>
internal sealed class MultiTimeframeDataIntegrationService : IHostedService
{
    private readonly ILogger<MultiTimeframeDataIntegrationService> _logger;
    private readonly TopstepXAdapterService _topstepXAdapter;
    private readonly BarAggregationService _barAggregator;
    private readonly TickBufferService _tickBuffer;
    
    public MultiTimeframeDataIntegrationService(
        ILogger<MultiTimeframeDataIntegrationService> logger,
        TopstepXAdapterService topstepXAdapter,
        BarAggregationService barAggregator,
        TickBufferService tickBuffer)
    {
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _topstepXAdapter = topstepXAdapter ?? throw new ArgumentNullException(nameof(topstepXAdapter));
        _barAggregator = barAggregator ?? throw new ArgumentNullException(nameof(barAggregator));
        _tickBuffer = tickBuffer ?? throw new ArgumentNullException(nameof(tickBuffer));
    }
    
    public Task StartAsync(CancellationToken cancellationToken)
    {
        // Check if in Lab Mode (don't subscribe to live feed during training)
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
        if (labMode == "1" || labMode?.ToLowerInvariant() == "true")
        {
            _logger.LogInformation("[MTF-INTEGRATION] LAB_MODE detected - skipping live data feed integration");
            return Task.CompletedTask;
        }
        
        // Subscribe to bar events from TopstepX adapter
        _topstepXAdapter.SubscribeToBarEvents(OnBarEventReceived);
        
        _logger.LogInformation(
            "[MTF-INTEGRATION] Multi-timeframe data integration started - feeding live bars to aggregation services");
        
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
