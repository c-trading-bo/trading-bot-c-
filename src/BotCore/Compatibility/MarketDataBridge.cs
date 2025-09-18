using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.DependencyInjection;
using BotCore.Bandits;
using TradingBot.Abstractions;

namespace BotCore.Compatibility;

/// <summary>
/// Market data bridge that plugs directly into your existing TopstepX SDK feeds
/// 
/// Market Data Integration: The delegate-based subscription system plugs directly 
/// into your existing TopstepX SDK feeds without requiring changes to your proven 
/// market data infrastructure or SignalR systems.
/// </summary>
public class MarketDataBridge : IDisposable
{
    private readonly ILogger<MarketDataBridge> _logger;
    private readonly IServiceProvider _serviceProvider;
    
    // Delegates for market data subscription
    public event Func<MarketDataUpdate, Task>? MarketDataReceived;
    public event Func<string, MarketContext, Task>? MarketContextUpdated;
    
    // Integration with existing services
    private readonly IMarketDataService? _existingMarketDataService;
    private readonly ITopstepXService? _topstepXService;
    
    public MarketDataBridge(ILogger<MarketDataBridge> logger, IServiceProvider serviceProvider)
    {
        _logger = logger;
        _serviceProvider = serviceProvider;
        
        // Get existing services (optional dependencies)
        _existingMarketDataService = serviceProvider.GetService<IMarketDataService>();
        _topstepXService = serviceProvider.GetService<ITopstepXService>();
        
        // Subscribe to existing market data feeds
        SubscribeToExistingFeeds();
        
        _logger.LogInformation("MarketDataBridge initialized - Connected to existing TopstepX feeds");
    }
    
    private void SubscribeToExistingFeeds()
    {
        try
        {
            // Subscribe to existing market data service if available
            if (_existingMarketDataService != null)
            {
                _existingMarketDataService.MarketDataUpdated += OnExistingMarketDataUpdated;
                _logger.LogDebug("Subscribed to existing market data service");
            }
            
            // Subscribe to TopstepX service if available
            if (_topstepXService != null)
            {
                _topstepXService.PriceUpdated += OnTopstepXPriceUpdated;
                _logger.LogDebug("Subscribed to TopstepX service");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error subscribing to existing market data feeds");
        }
    }
    
    private async Task OnExistingMarketDataUpdated(MarketDataUpdate update)
    {
        try
        {
            // Forward to compatibility kit subscribers
            if (MarketDataReceived != null)
            {
                await MarketDataReceived(update);
            }
            
            // Create enhanced market context
            var marketContext = CreateMarketContext(update);
            if (MarketContextUpdated != null)
            {
                await MarketContextUpdated(update.Symbol, marketContext);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing market data update for {Symbol}", update.Symbol);
        }
    }
    
    private async Task OnTopstepXPriceUpdated(string symbol, decimal price, DateTime timestamp)
    {
        try
        {
            var update = new MarketDataUpdate
            {
                Symbol = symbol,
                Price = price,
                Timestamp = timestamp,
                Source = "TopstepX"
            };
            
            await OnExistingMarketDataUpdated(update);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error processing TopstepX price update for {Symbol}", symbol);
        }
    }
    
    private MarketContext CreateMarketContext(MarketDataUpdate update)
    {
        // Create market context based on current market data
        // This can be enhanced with more sophisticated market analysis
        
        return new MarketContext
        {
            Symbol = update.Symbol,
            CurrentPrice = update.Price,
            Timestamp = update.Timestamp,
            Volatility = CalculateVolatility(update),
            IsTrending = DetermineTrendingState(update),
            IsVolatile = DetermineVolatileState(update),
            Confidence = 0.5m, // Default confidence, can be enhanced
            MaxPositionMultiplier = 1.0m, // Default, will be overridden by bundle selection
            ConfidenceThreshold = 0.65m // Default, will be overridden by bundle selection
        };
    }
    
    private decimal CalculateVolatility(MarketDataUpdate update)
    {
        // Simple volatility calculation - can be enhanced with historical data
        return 0.02m; // 2% default volatility
    }
    
    private bool DetermineTrendingState(MarketDataUpdate update)
    {
        // Simple trending determination - can be enhanced with technical indicators
        return false; // Default to non-trending
    }
    
    private bool DetermineVolatileState(MarketDataUpdate update)
    {
        // Simple volatile state determination - can be enhanced
        return CalculateVolatility(update) > 0.03m; // 3% volatility threshold
    }
    
    /// <summary>
    /// Subscribe to market context updates for a specific symbol
    /// </summary>
    public void SubscribeToMarketContext(string symbol, Func<MarketContext, Task> handler)
    {
        MarketContextUpdated += async (sym, context) =>
        {
            if (sym == symbol)
            {
                await handler(context);
            }
        };
        
        _logger.LogDebug("Subscribed to market context updates for {Symbol}", symbol);
    }
    
    /// <summary>
    /// Get current market context for a symbol
    /// </summary>
    public async Task<MarketContext> GetCurrentMarketContextAsync(string symbol, CancellationToken cancellationToken = default)
    {
        try
        {
            // Try to get from existing services first
            if (_existingMarketDataService != null)
            {
                var currentData = await _existingMarketDataService.GetCurrentDataAsync(symbol, cancellationToken);
                if (currentData != null)
                {
                    return CreateMarketContext(currentData);
                }
            }
            
            // Fallback to basic context
            return new MarketContext
            {
                Symbol = symbol,
                CurrentPrice = 0,
                Timestamp = DateTime.UtcNow,
                Volatility = 0.02m,
                IsTrending = false,
                IsVolatile = false,
                Confidence = 0.5m,
                MaxPositionMultiplier = 1.0m,
                ConfidenceThreshold = 0.65m
            };
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error getting current market context for {Symbol}", symbol);
            throw;
        }
    }
    
    public void Dispose()
    {
        try
        {
            // Unsubscribe from existing feeds
            if (_existingMarketDataService != null)
            {
                _existingMarketDataService.MarketDataUpdated -= OnExistingMarketDataUpdated;
            }
            
            if (_topstepXService != null)
            {
                _topstepXService.PriceUpdated -= OnTopstepXPriceUpdated;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error disposing MarketDataBridge");
        }
        
        _logger.LogInformation("MarketDataBridge disposed");
    }
}

/// <summary>
/// Market data update from existing feeds
/// </summary>
public class MarketDataUpdate
{
    public string Symbol { get; set; } = string.Empty;
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
    public string Source { get; set; } = string.Empty;
    public decimal Volume { get; set; }
    public decimal High { get; set; }
    public decimal Low { get; set; }
    public decimal Open { get; set; }
}

/// <summary>
/// Enhanced market context for decision making
/// </summary>
public record MarketContext
{
    public string Symbol { get; init; } = string.Empty;
    public decimal CurrentPrice { get; init; }
    public DateTime Timestamp { get; init; }
    public decimal Volatility { get; init; }
    public bool IsTrending { get; init; }
    public bool IsVolatile { get; init; }
    public decimal Confidence { get; init; }
    public decimal MaxPositionMultiplier { get; init; }
    public decimal ConfidenceThreshold { get; init; }
    public ConfigurationSource? ConfiguredParameters { get; init; }
}

// Placeholder interfaces for existing services
public interface IMarketDataService
{
    event Func<MarketDataUpdate, Task> MarketDataUpdated;
    Task<MarketDataUpdate?> GetCurrentDataAsync(string symbol, CancellationToken cancellationToken = default);
}

public interface ITopstepXService
{
    event Func<string, decimal, DateTime, Task> PriceUpdated;
}