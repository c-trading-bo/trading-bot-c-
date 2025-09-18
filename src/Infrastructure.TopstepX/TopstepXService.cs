using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Net.Http;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using System.Text.Json;
using System.Text;
using TradingBot.Abstractions;

namespace BotCore.Services;

/// <summary>
/// Production constants for TopstepX service
/// </summary>
internal static class TopstepXServiceConstants
{
    public const int RECONNECTION_DELAY_MS = 1000;
    public const int MAX_RETRY_ATTEMPTS = 5;
}

/// <summary>
/// Enhanced TopstepX service with improved SignalR connection handling
/// </summary>
public interface ITopstepXService
{
    Task<bool> ConnectAsync();
    Task<bool> DisconnectAsync();
    bool IsConnected { get; }
    event Action<MarketData>? OnMarketData;
    event Action<OrderBookData>? OnLevel2Update;
    event Action<TradeConfirmation>? OnTradeConfirmed;
    event Action<GatewayUserOrder>? OnGatewayUserOrder;
    event Action<GatewayUserTrade>? OnGatewayUserTrade;
    event Action<string>? OnError;
    Task<bool> SubscribeOrdersAsync(string accountId);
    Task<bool> SubscribeTradesAsync(string accountId);
}

public class TopstepXService : ITopstepXService, IDisposable
{
    private readonly ILogger<TopstepXService> _logger;
    private readonly IConfiguration _configuration;
    private HubConnection? _hubConnection;
    private string? _jwtToken;
    private readonly HttpClient _httpClient;
    private readonly Timer _reconnectTimer;
    private volatile bool _isConnecting;
    private volatile bool _disposed;
    private volatile bool _wired = false; // Double-subscription guard

    public bool IsConnected => _hubConnection?.State == HubConnectionState.Connected;

    public event Action<MarketData>? OnMarketData;
    public event Action<OrderBookData>? OnLevel2Update;
    public event Action<TradeConfirmation>? OnTradeConfirmed;
    public event Action<GatewayUserOrder>? OnGatewayUserOrder;
    public event Action<GatewayUserTrade>? OnGatewayUserTrade;
    public event Action<string>? OnError;

    public TopstepXService(ILogger<TopstepXService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;

        // Create HttpClient with enhanced SSL handling
        var handler = new HttpClientHandler
        {
            ServerCertificateCustomValidationCallback = ValidateServerCertificate,
            SslProtocols = System.Security.Authentication.SslProtocols.Tls12 | System.Security.Authentication.SslProtocols.Tls13,
            MaxConnectionsPerServer = 10
        };

        _httpClient = new HttpClient(handler)
        {
            Timeout = TimeSpan.FromSeconds(30),
            BaseAddress = new Uri(_configuration["TopstepX:ApiUrl"] ?? 
                Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? 
                throw new InvalidOperationException("TopstepX API URL must be configured via TopstepX:ApiUrl setting or TOPSTEPX_API_BASE environment variable"))
        };

        // Set environment variable for .NET HTTP handler compatibility
        Environment.SetEnvironmentVariable("DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER", "false");

        // Auto-reconnect timer (every 30 seconds if disconnected)
        _reconnectTimer = new Timer(CheckAndReconnect, null, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
    }

    /// <summary>
    /// Enhanced SSL certificate validation
    /// </summary>
    private bool ValidateServerCertificate(HttpRequestMessage request, X509Certificate2? certificate,
        X509Chain? chain, SslPolicyErrors sslPolicyErrors)
    {
        // In production, you might want to implement proper certificate validation
        // For now, we'll accept all certificates to handle SSL issues
        var isDevelopment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") == "Development";
        var bypassSsl = Environment.GetEnvironmentVariable("BYPASS_SSL_VALIDATION") == "true";

        if (isDevelopment || bypassSsl)
        {
            if (sslPolicyErrors != SslPolicyErrors.None)
            {
                _logger.LogWarning("[TOPSTEPX] SSL validation bypassed. Errors: {Errors}", sslPolicyErrors);
            }
            return true;
        }

        // Production certificate validation
        if (sslPolicyErrors == SslPolicyErrors.None)
            return true;

        _logger.LogWarning("[TOPSTEPX] SSL certificate validation failed: {Errors}", sslPolicyErrors);
        return false;
    }

    public async Task<bool> ConnectAsync()
    {
        if (_disposed)
            return false;

        if (_isConnecting)
        {
            _logger.LogDebug("[TOPSTEPX] Connection already in progress");
            return false;
        }

        if (IsConnected)
        {
            _logger.LogDebug("[TOPSTEPX] Already connected");
            return true;
        }

        _isConnecting = true;

        try
        {
            // Get JWT token first
            _jwtToken = await GetJwtTokenAsync();
            if (string.IsNullOrEmpty(_jwtToken))
            {
                _logger.LogError("[TOPSTEPX] Failed to obtain JWT token");
                return false;
            }

            // Build SignalR connection with enhanced configuration
            await BuildSignalRConnection();

            // Start connection with retry logic
            var connected = await StartConnectionWithRetry();

            if (connected)
            {
                // Subscribe to market data after successful connection
                await SubscribeToMarketData();
                _logger.LogInformation("[TOPSTEPX] Successfully connected and subscribed to market data");
            }

            return connected;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Connection failed");
            return false;
        }
        finally
        {
            _isConnecting = false;
        }
    }

    private Task BuildSignalRConnection()
    {
        var signalRUrl = _configuration["TopstepX:SignalRUrl"] ?? 
            Environment.GetEnvironmentVariable("TOPSTEPX_SIGNALR_URL") ?? 
            throw new InvalidOperationException("TopstepX SignalR URL must be configured via TopstepX:SignalRUrl setting or TOPSTEPX_SIGNALR_URL environment variable");
        var hubUrl = $"{signalRUrl.TrimEnd('/')}/tradingHub";

        _logger.LogInformation("[TOPSTEPX] Building SignalR connection to {Url}", hubUrl);

        _hubConnection = new HubConnectionBuilder()
            .WithUrl(hubUrl, options =>
            {
                options.AccessTokenProvider = () => Task.FromResult(_jwtToken);

                // Configure HTTP message handler with SSL bypass
                options.HttpMessageHandlerFactory = (message) =>
                {
                    if (message is HttpClientHandler clientHandler)
                    {
                        clientHandler.ServerCertificateCustomValidationCallback = ValidateServerCertificate;
                        clientHandler.SslProtocols = System.Security.Authentication.SslProtocols.Tls12 |
                                                   System.Security.Authentication.SslProtocols.Tls13;
                    }
                    return message;
                };

                // Try WebSockets first, then fallback to other transports
                options.Transports = HttpTransportType.WebSockets |
                                   HttpTransportType.LongPolling |
                                   HttpTransportType.ServerSentEvents;

                // Set timeouts
                options.CloseTimeout = TimeSpan.FromSeconds(30);
                options.SkipNegotiation = false; // Allow negotiation for transport fallback
            })
            .WithAutomaticReconnect(new ExponentialBackoffRetryPolicy())
            .ConfigureLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Information);
                logging.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Warning);
                logging.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Warning);
            })
            .Build();

        // Configure connection settings
        _hubConnection.ServerTimeout = TimeSpan.FromSeconds(60);
        _hubConnection.KeepAliveInterval = TimeSpan.FromSeconds(15);
        _hubConnection.HandshakeTimeout = TimeSpan.FromSeconds(30);

        // Set up event handlers
        SetupConnectionEventHandlers();
        RegisterMessageHandlers();

        return Task.CompletedTask;
    }

    private void SetupConnectionEventHandlers()
    {
        if (_hubConnection == null) return;

        _hubConnection.Closed += async (error) =>
        {
            _logger.LogWarning("[TOPSTEPX] Connection closed");
            // Log the error details securely without exposing them
            if (error != null)
            {
                _logger.LogDebug("Connection closed with error: {ErrorType}", error.GetType().Name);
            }

            // Don't auto-reconnect immediately, let the timer handle it
            await Task.Delay(TopstepXServiceConstants.RECONNECTION_DELAY_MS);
        };

        _hubConnection.Reconnecting += (error) =>
        {
            _logger.LogInformation("[TOPSTEPX] Attempting reconnection");
            if (error != null)
            {
                _logger.LogDebug("Reconnecting due to: {ErrorType}", error.GetType().Name);
            }
            return Task.CompletedTask;
        };

        _hubConnection.Reconnected += async (connectionId) =>
        {
            _logger.LogInformation("[TOPSTEPX] Reconnected successfully");

            // Re-subscribe to market data after reconnection
            try
            {
                await SubscribeToMarketData();
                _logger.LogInformation("[TOPSTEPX] Re-subscribed to market data after reconnection");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TOPSTEPX] Failed to re-subscribe after reconnection");
            }
        };
    }

    private void RegisterMessageHandlers()
    {
        if (_hubConnection == null) return;

        // Market data updates
        _hubConnection.On<JsonElement>("MarketData", (data) =>
        {
            try
            {
                var marketData = JsonSerializer.Deserialize<MarketData>(data.GetRawText());
                if (marketData != null)
                {
                    _logger.LogDebug("[MARKET] {Symbol}: Bid={Bid} Ask={Ask} Last={Last}",
                        marketData.Symbol, marketData.Bid, marketData.Ask, marketData.Last);
                    OnMarketData?.Invoke(marketData);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TOPSTEPX] Error processing market data");
            }
        });

        // Level 2 order book updates
        _hubConnection.On<JsonElement>("Level2Update", (data) =>
        {
            try
            {
                var orderBook = JsonSerializer.Deserialize<OrderBookData>(data.GetRawText());
                if (orderBook != null)
                {
                    _logger.LogDebug("[LEVEL2] {Symbol}: BidSize={BidSize} AskSize={AskSize}",
                        orderBook.Symbol, orderBook.BidSize, orderBook.AskSize);
                    OnLevel2Update?.Invoke(orderBook);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TOPSTEPX] Error processing Level 2 data");
            }
        });

        // Trade confirmations
        _hubConnection.On<JsonElement>("TradeConfirmed", (data) =>
        {
            try
            {
                var trade = JsonSerializer.Deserialize<TradeConfirmation>(data.GetRawText());
                if (trade != null)
                {
                    _logger.LogInformation("[TRADE] Confirmed: {Symbol} {Side} {Quantity} @ {Price}",
                        trade.Symbol, trade.Side, trade.Quantity, trade.Price);
                    OnTradeConfirmed?.Invoke(trade);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TOPSTEPX] Error processing trade confirmation");
            }
        });

        // Error messages
        _hubConnection.On<string>("Error", (message) =>
        {
            _logger.LogError("[TOPSTEPX] Server error: {Message}", message);
            OnError?.Invoke(message);
        });

        // GatewayUserOrder handler - subscribe and correlate by customTag
        _hubConnection.On<JsonElement>("GatewayUserOrder", (data) =>
        {
            try
            {
                var order = JsonSerializer.Deserialize<GatewayUserOrder>(data.GetRawText());
                if (order != null)
                {
                    var orderData = new
                    {
                        timestamp = DateTime.UtcNow,
                        component = "topstepx_service",
                        operation = "gateway_user_order",
                        accountId = order.AccountId,
                        orderId = order.OrderId,
                        customTag = order.CustomTag,
                        status = order.Status,
                        reason = order.Reason
                    };

                    _logger.LogInformation("ORDER: {OrderData}", 
                        System.Text.Json.JsonSerializer.Serialize(orderData));

                    OnGatewayUserOrder?.Invoke(order);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TOPSTEPX] Error processing GatewayUserOrder");
            }
        });

        // GatewayUserTrade handler - subscribe and correlate by customTag
        _hubConnection.On<JsonElement>("GatewayUserTrade", (data) =>
        {
            try
            {
                var trade = JsonSerializer.Deserialize<GatewayUserTrade>(data.GetRawText());
                if (trade != null)
                {
                    var tradeData = new
                    {
                        timestamp = DateTime.UtcNow,
                        component = "topstepx_service",
                        operation = "gateway_user_trade",
                        accountId = trade.AccountId,
                        orderId = trade.OrderId,
                        fillPrice = TradingBot.Infrastructure.TopstepX.Px.F2(trade.FillPrice),
                        quantity = trade.Quantity,
                        time = trade.Time.ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
                    };

                    _logger.LogInformation("TRADE: {TradeData}", 
                        System.Text.Json.JsonSerializer.Serialize(tradeData));

                    OnGatewayUserTrade?.Invoke(trade);
                }
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "[TOPSTEPX] Error processing GatewayUserTrade");
            }
        });

        // Mark as wired to prevent double subscription
        if (!_wired)
        {
            _wired = true;
            _logger.LogDebug("[TOPSTEPX] Message handlers registered successfully");
        }
        else
        {
            var guardData = new
            {
                timestamp = DateTime.UtcNow,
                component = "topstepx_service",
                operation = "register_handlers",
                guard = "already_wired"
            };

            _logger.LogWarning("SUBSCRIPTION_GUARD: {GuardData}", 
                System.Text.Json.JsonSerializer.Serialize(guardData));
        }
    }

    private async Task<bool> StartConnectionWithRetry()
    {
        const int maxRetries = 3;

        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                if (_hubConnection == null)
                    throw new InvalidOperationException("Hub connection not initialized");

                _logger.LogInformation("[TOPSTEPX] Starting connection (attempt {Attempt}/{Max})", attempt, maxRetries);

                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                await _hubConnection.StartAsync(cts.Token);

                // CRITICAL FIX: Wait for connection to be fully established
                // This prevents the race condition where subscriptions are called before connection is ready
                await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.WaitForConnected(_hubConnection, TimeSpan.FromSeconds(30), cts.Token, _logger);

                _logger.LogInformation("[TOPSTEPX] Connection started and confirmed ready. State: {State}", _hubConnection.State);
                return true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[TOPSTEPX] Connection attempt {Attempt} failed", attempt);

                if (attempt < maxRetries)
                {
                    var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt)); // Exponential backoff
                    _logger.LogInformation("[TOPSTEPX] Retrying in {Delay} seconds...", delay.TotalSeconds);
                    await Task.Delay(delay);
                }
            }
        }

        return false;
    }

    private async Task SubscribeToMarketData()
    {
        if (_hubConnection == null)
        {
            _logger.LogWarning("[TOPSTEPX] Cannot subscribe - hub connection is null");
            return;
        }

        try
        {
            // Subscribe to market data for configured symbols
            var symbols = _configuration["TopstepX:Symbols"]?.Split(',') ?? new[] { "ES", "NQ" };

            foreach (var symbol in symbols)
            {
                // Use SignalRSafeInvoker to ensure connection is ready before invoking
                await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                    _hubConnection,
                    () => _hubConnection.InvokeAsync("SubscribeToMarketData", symbol.Trim()),
                    _logger,
                    CancellationToken.None);
                    
                _logger.LogInformation("[TOPSTEPX] Subscribed to market data for {Symbol}", symbol.Trim());
            }

            // Subscribe to Level 2 data if enabled
            var enableLevel2Section = _configuration.GetSection("TopstepX:EnableLevel2");
            var enableLevel2 = enableLevel2Section.Exists() && bool.Parse(enableLevel2Section.Value ?? "false");
            if (enableLevel2)
            {
                foreach (var symbol in symbols)
                {
                    await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                        _hubConnection,
                        () => _hubConnection.InvokeAsync("SubscribeToLevel2", symbol.Trim()),
                        _logger,
                        CancellationToken.None);
                        
                    _logger.LogInformation("[TOPSTEPX] Subscribed to Level 2 data for {Symbol}", symbol.Trim());
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Failed to subscribe to market data");
            throw new InvalidOperationException("Failed to subscribe to TopstepX market data events", ex);
        }
    }

    private async Task<string?> GetJwtTokenAsync()
    {
        try
        {
            // Use relative path since BaseAddress is configured
            var request = new HttpRequestMessage(HttpMethod.Post, "auth/token");

            var loginPayload = new
            {
                username = _configuration["TopstepX:Username"],
                password = _configuration["TopstepX:Password"],
                apiKey = _configuration["TopstepX:ApiKey"]
            };

            request.Content = new StringContent(
                JsonSerializer.Serialize(loginPayload),
                Encoding.UTF8,
                "application/json"
            );

            var response = await _httpClient.SendAsync(request);

            if (response.IsSuccessStatusCode)
            {
                var content = await response.Content.ReadAsStringAsync();
                var tokenResponse = JsonSerializer.Deserialize<TokenResponse>(content);

                if (tokenResponse?.Token != null)
                {
                    _logger.LogInformation("[TOPSTEPX] JWT token obtained successfully");
                    return tokenResponse.Token;
                }
            }

            _logger.LogError("[TOPSTEPX] JWT request failed with status: {Status}",
                response.StatusCode);
            
            var errorContent = await response.Content.ReadAsStringAsync();
            throw new HttpRequestException($"JWT authentication failed: HTTP {(int)response.StatusCode} - {errorContent}");
        }
        catch (Exception ex) when (!(ex is HttpRequestException))
        {
            _logger.LogError(ex, "[TOPSTEPX] JWT token request failed");
            throw new InvalidOperationException("JWT token request failed due to network or service error", ex);
        }
    }

    private void CheckAndReconnect(object? state)
    {
        if (_disposed || _isConnecting)
            return;

        if (!IsConnected)
        {
            _logger.LogInformation("[TOPSTEPX] Auto-reconnect triggered - connection not active");
            _ = Task.Run(async () =>
            {
                try
                {
                    await ConnectAsync();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] Auto-reconnect failed");
                }
            });
        }
        else
        {
            // Perform health check ping for connected state
            _ = Task.Run(async () =>
            {
                try
                {
                    await PerformHealthCheckPing();
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] Health check ping failed");
                }
            });
        }
    }

    /// <summary>
    /// Performs a health check ping to verify the connection is still responsive
    /// </summary>
    private async Task PerformHealthCheckPing()
    {
        if (_hubConnection?.State != HubConnectionState.Connected)
            return;

        try
        {
            // Try to invoke a simple ping method with timeout
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
            
            await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                _hubConnection,
                () => _hubConnection.InvokeAsync("Ping", cancellationToken: cts.Token),
                _logger,
                cts.Token,
                maxAttempts: 1,  // Single attempt for health check
                waitMs: TopstepXServiceConstants.RECONNECTION_DELAY_MS);

            _logger.LogDebug("[TOPSTEPX] Health check ping successful");
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[TOPSTEPX] Health check ping failed - connection may be unhealthy");
            
            // If ping fails, trigger reconnection
            if (!_isConnecting)
            {
                _logger.LogInformation("[TOPSTEPX] Triggering reconnection due to failed health check");
                _ = Task.Run(async () =>
                {
                    try
                    {
                        await DisconnectAsync();
                        await Task.Delay(TopstepXServiceConstants.RECONNECTION_DELAY_MS); // Brief delay before reconnecting
                        await ConnectAsync();
                    }
                    catch (Exception reconEx)
                    {
                        _logger.LogError(reconEx, "[TOPSTEPX] Health check triggered reconnection failed");
                    }
                });
            }
        }
    }

    public async Task<bool> DisconnectAsync()
    {
        try
        {
            if (_hubConnection != null)
            {
                await _hubConnection.StopAsync();
                await _hubConnection.DisposeAsync();
                _hubConnection = null;
            }

            _logger.LogInformation("[TOPSTEPX] Disconnected successfully");
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Error during disconnection");
            return false;
        }
    }

    public async Task<bool> SubscribeOrdersAsync(string accountId)
    {
        try
        {
            if (_hubConnection == null)
            {
                _logger.LogWarning("[TOPSTEPX] Cannot subscribe to orders - hub connection is null");
                return false;
            }

            // Use SignalRSafeInvoker to ensure safe invocation with connection state checking
            await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                _hubConnection,
                () => _hubConnection.InvokeAsync("SubscribeOrders", accountId),
                _logger,
                CancellationToken.None);
                
            _logger.LogInformation("[TOPSTEPX] Subscribed to orders for account {AccountId}", SecurityHelpers.MaskSpecificAccountId(accountId));
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Failed to subscribe to orders for account {AccountId}", SecurityHelpers.MaskSpecificAccountId(accountId));
            return false;
        }
    }

    public async Task<bool> SubscribeTradesAsync(string accountId)
    {
        try
        {
            if (_hubConnection == null)
            {
                _logger.LogWarning("[TOPSTEPX] Cannot subscribe to trades - hub connection is null");
                return false;
            }

            // Use SignalRSafeInvoker to ensure safe invocation with connection state checking
            await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                _hubConnection,
                () => _hubConnection.InvokeAsync("SubscribeTrades", accountId),
                _logger,
                CancellationToken.None);
                
            _logger.LogInformation("[TOPSTEPX] Subscribed to trades for account {AccountId}", SecurityHelpers.MaskSpecificAccountId(accountId));
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Failed to subscribe to trades for account {AccountId}", SecurityHelpers.MaskSpecificAccountId(accountId));
            return false;
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _reconnectTimer?.Dispose();

                try
                {
                    _hubConnection?.StopAsync().Wait(TimeSpan.FromSeconds(5));
                    _hubConnection?.DisposeAsync().AsTask().Wait(TimeSpan.FromSeconds(5));
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] Error disposing hub connection");
                }

                _httpClient?.Dispose();
            }
            _disposed = true;
        }
    }
}

/// <summary>
/// Custom retry policy with exponential backoff
/// </summary>
public class ExponentialBackoffRetryPolicy : IRetryPolicy
{
    public TimeSpan? NextRetryDelay(RetryContext retryContext)
    {
        // Max 5 retries with exponential backoff
        if (retryContext.PreviousRetryCount >= TopstepXServiceConstants.MAX_RETRY_ATTEMPTS)
            return null; // Stop retrying after max attempts

        var delay = TimeSpan.FromSeconds(Math.Pow(2, retryContext.PreviousRetryCount));
        return delay > TimeSpan.FromSeconds(30) ? TimeSpan.FromSeconds(30) : delay;
    }
}

// Data models for SignalR messages
public class MarketData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal Bid { get; set; }
    public decimal Ask { get; set; }
    public decimal Last { get; set; }
    public long Volume { get; set; }
    public DateTime Timestamp { get; set; }
    
    // Aliases for compatibility with different parts of the system
    public decimal BidPrice => Bid;
    public decimal AskPrice => Ask;
    public decimal LastPrice => Last;
}

public class OrderBookData
{
    public string Symbol { get; set; } = string.Empty;
    public decimal BidPrice { get; set; }
    public int BidSize { get; set; }
    public decimal AskPrice { get; set; }
    public int AskSize { get; set; }
    public DateTime Timestamp { get; set; }
}

public class TradeConfirmation
{
    public string OrderId { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
}

// New models for GatewayUserOrder and GatewayUserTrade handlers
public class GatewayUserOrder
{
    public string AccountId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public string Status { get; set; } = string.Empty; // "New", "Open", "Filled", "Cancelled", "Rejected"
    public string Reason { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public int Quantity { get; set; }
    public decimal Price { get; set; }
    public DateTime Timestamp { get; set; }
}

public class GatewayUserTrade
{
    public string AccountId { get; set; } = string.Empty;
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public decimal FillPrice { get; set; }
    public int Quantity { get; set; }
    public DateTime Time { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
}

public class TokenResponse
{
    public string? Token { get; set; }
    public int ExpiresIn { get; set; }
}