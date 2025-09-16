using System.Text.Json;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// MarketDataService with real /hubs/market SignalR integration
/// NO STUBS - Uses actual TopstepX market hub for live price data
/// </summary>
public interface IMarketDataService
{
    Task<bool> ConnectAsync();
    Task<decimal> GetLastPriceAsync(string symbol);
    Task<MarketDepth> GetOrderBookAsync(string symbol);
    event Action<MarketTick> OnMarketTick;
}

public record MarketTick(string Symbol, decimal Price, decimal Volume, DateTime Timestamp);
public record MarketDepth(string Symbol, decimal BidPrice, decimal AskPrice, int BidSize, int AskSize);

public class MarketDataService : IMarketDataService, IAsyncDisposable, IDisposable
{
    private readonly ILogger<MarketDataService> _logger;
    private readonly AppOptions _config;
    private readonly HttpClient _httpClient;
    private HubConnection? _hubConnection;
    private bool _isConnected = false;
    
    public event Action<MarketTick>? OnMarketTick;

    public MarketDataService(ILogger<MarketDataService> logger, IOptions<AppOptions> config, HttpClient httpClient)
    {
        _logger = logger;
        _config = config.Value;
        _httpClient = httpClient;
        _httpClient.BaseAddress = new Uri(_config.ApiBase);
    }

    public async Task<bool> ConnectAsync()
    {
        try
        {
            _logger.LogInformation("[MARKET] Connecting to TopstepX market hub at {Url}", _config.ApiBase);
            
            // Real SignalR connection to /hubs/market
            var hubUrl = $"{_config.ApiBase.TrimEnd('/')}/hubs/market";
            
            // Build the HubConnection
            _hubConnection = new HubConnectionBuilder()
                .WithUrl(hubUrl, options =>
                {
                    if (!string.IsNullOrEmpty(_config.AuthToken))
                    {
                        options.Headers.Add("Authorization", $"Bearer {_config.AuthToken}");
                    }
                })
                .WithAutomaticReconnect()
                .Build();

            // Subscribe to real market data events
            _hubConnection.On<object>("MarketData", tick =>
            {
                try
                {
                    var tickJson = JsonSerializer.Serialize(tick);
                    _logger.LogDebug("[MARKET] Market tick: {Tick}", tickJson);
                    
                    // Parse and emit market tick
                    if (tick != null)
                    {
                        OnMarketTick?.Invoke(new MarketTick(
                            Symbol: ExtractProperty(tick, "symbol") ?? "UNKNOWN",
                            Price: ParseDecimal(ExtractProperty(tick, "price")),
                            Volume: ParseDecimal(ExtractProperty(tick, "volume")),
                            Timestamp: DateTime.UtcNow
                        ));
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[MARKET] Error processing market tick");
                }
            });

            // Start the connection
            await _hubConnection.StartAsync();
            _isConnected = true;
            
            _logger.LogInformation("[MARKET] Γ£à Connected to live TopstepX market data feed");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MARKET] Failed to connect to market data");
            _isConnected = false;
            return false;
        }
    }

    public async Task<decimal> GetLastPriceAsync(string symbol)
    {
        if (!_isConnected)
        {
            throw new InvalidOperationException("Market data service not connected. Call ConnectAsync() first.");
        }
        
        try
        {
            // Real REST call to /api/Market/lastPrice/{symbol}
            // This replaces: return 5500m + (decimal)(new Random().NextDouble() * 20 - 10);
            var response = await _httpClient.GetAsync($"/api/Market/lastPrice/{symbol}");
            response.EnsureSuccessStatusCode();
            
            var json = await response.Content.ReadAsStringAsync();
            var priceData = JsonSerializer.Deserialize<JsonElement>(json);
            
            return priceData.GetProperty("price").GetDecimal();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MARKET] Failed to get last price for {Symbol}", symbol);
            throw;
        }
    }

    public async Task<MarketDepth> GetOrderBookAsync(string symbol)
    {
        try
        {
            // Real REST call to /api/Market/orderBook/{symbol}
            var response = await _httpClient.GetAsync($"/api/Market/orderBook/{symbol}");
            response.EnsureSuccessStatusCode();
            
            var json = await response.Content.ReadAsStringAsync();
            var bookData = JsonSerializer.Deserialize<JsonElement>(json);
            
            return new MarketDepth(
                symbol,
                bookData.GetProperty("bidPrice").GetDecimal(),
                bookData.GetProperty("askPrice").GetDecimal(),
                bookData.GetProperty("bidSize").GetInt32(),
                bookData.GetProperty("askSize").GetInt32()
            );
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MARKET] Failed to get order book for {Symbol}", symbol);
            throw;
        }
    }

    // Helper methods for parsing dynamic objects from SignalR
    private static string? ExtractProperty(object obj, string propertyName)
    {
        try
        {
            if (obj is JsonElement element && element.TryGetProperty(propertyName, out var prop))
            {
                return prop.GetString();
            }
            
            // Try reflection as fallback
            var type = obj.GetType();
            var property = type.GetProperty(propertyName, System.Reflection.BindingFlags.IgnoreCase | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            return property?.GetValue(obj)?.ToString();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MARKET-DATA] Failed to parse market data");
            throw new InvalidOperationException("Failed to parse market data - invalid format", ex);
        }
    }

    private static decimal ParseDecimal(string? value) => decimal.TryParse(value, out var result) ? result : 0m;

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (_hubConnection != null)
            {
                await _hubConnection.DisposeAsync();
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MARKET] Error disposing hub connection");
        }
    }

    public void Dispose()
    {
        DisposeAsync().AsTask().GetAwaiter().GetResult();
    }
}
