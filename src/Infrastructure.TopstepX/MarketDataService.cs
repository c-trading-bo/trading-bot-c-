using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.Http.Connections;
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

public class MarketDataService : IMarketDataService, IDisposable
{
    private readonly ILogger<MarketDataService> _logger;
    private readonly AppOptions _config;
    private readonly HttpClient _httpClient;
    private HubConnection? _hubConnection;
    private bool _isConnected;
    
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
            
            // Build SignalR connection with authentication
            _hubConnection = new HubConnectionBuilder()
                .WithUrl(hubUrl, options =>
                {
                    options.AccessTokenProvider = async () =>
                    {
                        // Use JWT token from environment or config
                        return Environment.GetEnvironmentVariable("TOPSTEPX_JWT") ?? _config.AuthToken;
                    };
                    
                    options.Transports = HttpTransportType.WebSockets |
                                       HttpTransportType.LongPolling |
                                       HttpTransportType.ServerSentEvents;
                })
                .WithAutomaticReconnect()
                .Build();

            // Set up market data event handler
            _hubConnection.On<JsonElement>("MarketData", (data) =>
            {
                try
                {
                    // Parse market tick data and trigger OnMarketTick
                    var symbol = data.GetProperty("symbol").GetString() ?? "";
                    var price = data.GetProperty("price").GetDecimal();
                    var volume = data.GetProperty("volume").GetDecimal();
                    var timestamp = data.GetProperty("timestamp").GetDateTime();
                    
                    var tick = new MarketTick(symbol, price, volume, timestamp);
                    OnMarketTick?.Invoke(tick);
                    
                    _logger.LogDebug("[MARKET] Tick: {Symbol} @ {Price}", symbol, price);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[MARKET] Error processing market tick");
                }
            });

            // Start the connection
            await _hubConnection.StartAsync();
            _isConnected = true;
            
            _logger.LogInformation("[MARKET] ✅ Connected to live TopstepX market data feed");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[MARKET] Failed to connect to market data");
            return false;
        }
    }

    public async Task<decimal> GetLastPriceAsync(string symbol)
    {
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

    public void Dispose()
    {
        try
        {
            _hubConnection?.StopAsync().Wait(TimeSpan.FromSeconds(5));
            _hubConnection?.DisposeAsync().AsTask().Wait(TimeSpan.FromSeconds(5));
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[MARKET] Error disposing hub connection");
        }
        
        _httpClient?.Dispose();
        _hubConnection = null;
        _isConnected = false;
    }
}