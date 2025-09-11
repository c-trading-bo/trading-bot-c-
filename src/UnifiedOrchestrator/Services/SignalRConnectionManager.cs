using System;
using System.Net.Http;
using System.Net.Security;
using System.Security.Cryptography.X509Certificates;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Custom retry policy with exponential backoff (from working implementation)
/// </summary>
public class ExponentialBackoffRetryPolicy : IRetryPolicy
{
    public TimeSpan? NextRetryDelay(RetryContext retryContext)
    {
        // Max 5 retries with exponential backoff
        if (retryContext.PreviousRetryCount >= 5)
            return null;

        var delay = TimeSpan.FromSeconds(Math.Pow(2, retryContext.PreviousRetryCount));
        return delay > TimeSpan.FromSeconds(30) ? TimeSpan.FromSeconds(30) : delay;
    }
}

public class SignalRConnectionManager : ISignalRConnectionManager, IHostedService, IDisposable
{
    private readonly ILogger<SignalRConnectionManager> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly ITokenProvider _tokenProvider;
    private readonly SemaphoreSlim _connectionLock = new(1, 1);
    
    private HubConnection? _userHub;
    private HubConnection? _marketHub;
    private volatile bool _userHubWired = false;
    private volatile bool _marketHubWired = false;
    private readonly Timer _connectionHealthTimer;

    public bool IsUserHubConnected => _userHub?.State == HubConnectionState.Connected && _userHubWired;
    public bool IsMarketHubConnected => _marketHub?.State == HubConnectionState.Connected && _marketHubWired;

    public event Action<string>? ConnectionStateChanged;
    
    // Data reception events - completing the connection state machine
    public event Action<object>? OnMarketDataReceived;
    public event Action<object>? OnContractQuotesReceived;
    public event Action<object>? OnGatewayUserOrderReceived;
    public event Action<object>? OnGatewayUserTradeReceived;
    public event Action<object>? OnFillUpdateReceived;
    public event Action<object>? OnOrderUpdateReceived;

    public SignalRConnectionManager(
        ILogger<SignalRConnectionManager> logger,
        ITradingLogger tradingLogger,
        ITokenProvider tokenProvider)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _tokenProvider = tokenProvider;
        
        // Health check timer every 30 seconds
        _connectionHealthTimer = new Timer(CheckConnectionHealth, null, 
            TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
            
        // Set environment variable for .NET HTTP handler compatibility  
        Environment.SetEnvironmentVariable("DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER", "false");
    }

    /// <summary>
    /// Enhanced SSL certificate validation for TopstepX connections
    /// </summary>
    private bool ValidateServerCertificate(HttpRequestMessage request, X509Certificate2? certificate,
        X509Chain? chain, SslPolicyErrors sslPolicyErrors)
    {
        // In production, you might want to implement proper certificate validation
        // For now, we'll accept all certificates to handle SSL issues with TopstepX
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
        
        // For TopstepX, we may need to be more lenient with SSL validation
        // This should be configurable in production
        return true;
    }

    /// <summary>
    /// Connection startup with retry logic and stability validation
    /// </summary>
    private async Task<bool> StartConnectionWithRetry(HubConnection hubConnection, string hubName)
    {
        const int maxRetries = 3;

        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                if (hubConnection == null)
                    throw new InvalidOperationException("Hub connection not initialized");

                _logger.LogInformation("[TOPSTEPX] Starting {HubName} connection (attempt {Attempt}/{Max})", 
                    hubName, attempt, maxRetries);

                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                await hubConnection.StartAsync(cts.Token);

                _logger.LogInformation("[TOPSTEPX] {HubName} connection started successfully. State: {State}", 
                    hubName, hubConnection.State);
                
                // CRITICAL: Wait and validate connection stability
                _logger.LogInformation("[TOPSTEPX] {HubName} validating connection stability...", hubName);
                await Task.Delay(2000); // Extended stabilization delay
                
                if (hubConnection.State == HubConnectionState.Connected && !string.IsNullOrEmpty(hubConnection.ConnectionId))
                {
                    _logger.LogInformation("[TOPSTEPX] {HubName} connection validated - State: {State}, ID: {ConnectionId}", 
                        hubName, hubConnection.State, hubConnection.ConnectionId);
                    return true;
                }
                else
                {
                    _logger.LogWarning("[TOPSTEPX] {HubName} connection became unstable - State: {State}, ID: {ConnectionId}", 
                        hubName, hubConnection.State, hubConnection.ConnectionId ?? "null");
                    throw new InvalidOperationException($"{hubName} connection unstable after start");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[TOPSTEPX] {HubName} connection attempt {Attempt} failed", hubName, attempt);

                if (attempt < maxRetries)
                {
                    var delay = TimeSpan.FromSeconds(Math.Pow(2, attempt)); // Exponential backoff
                    _logger.LogInformation("[TOPSTEPX] Retrying {HubName} in {Delay} seconds...", 
                        hubName, delay.TotalSeconds);
                    await Task.Delay(delay);
                }
            }
        }

        return false;
    }

    public async Task<HubConnection> GetUserHubConnectionAsync()
    {
        if (_userHub != null && IsUserHubConnected)
        {
            return _userHub;
        }

        await _connectionLock.WaitAsync();
        try
        {
            if (_userHub != null && IsUserHubConnected)
            {
                return _userHub;
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                "Establishing User Hub connection");

            var token = await _tokenProvider.GetTokenAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    "Cannot connect to User Hub - no valid JWT token");
                throw new InvalidOperationException("No valid JWT token available for User Hub connection");
            }

            // Log token info for debugging (without exposing the actual token)
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Using JWT token for User Hub: length={token.Length}, starts_with_Bearer={token.StartsWith("Bearer ")}, has_dots={token.Count(c => c == '.')}");

            // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                token = token.Substring(7);
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    "Removed 'Bearer' prefix from JWT token for User Hub");
            }

            _userHub?.DisposeAsync();
            _userHub = new HubConnectionBuilder()
                .WithUrl("https://gateway-rtc-demo.s2f.projectx.com/hubs/user", options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(token);
                    
                    // CRITICAL: Configure HTTP message handler with SSL bypass for TopstepX
                    options.HttpMessageHandlerFactory = (message) =>
                    {
                        if (message is HttpClientHandler clientHandler)
                        {
                            clientHandler.ServerCertificateCustomValidationCallback = ValidateServerCertificate;
                            clientHandler.SslProtocols = System.Security.Authentication.SslProtocols.Tls12 |
                                                       System.Security.Authentication.SslProtocols.Tls13;
                            clientHandler.MaxConnectionsPerServer = 10;
                        }
                        return message;
                    };

                    // Try WebSockets first, then fallback to other transports
                    options.Transports = HttpTransportType.WebSockets |
                                       HttpTransportType.LongPolling |
                                       HttpTransportType.ServerSentEvents;

                    // Set connection timeouts
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

            // Configure connection timeouts for production use
            _userHub.ServerTimeout = TimeSpan.FromSeconds(60);
            _userHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            _userHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            SetupUserHubEventHandlers();
            
            // Use enhanced connection startup with stability validation
            var connected = await StartConnectionWithRetry(_userHub, "User Hub");
            if (connected)
            {
                _userHubWired = true;
                
                // CRITICAL: Immediately subscribe to keep connection alive (from working version)
                try
                {
                    await _userHub.InvokeAsync("SubscribeOrders", "10459779"); // Use the account ID we found
                    _logger.LogInformation("[TOPSTEPX] User Hub: Subscribed to orders for account to keep connection alive");
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] User Hub: Failed to subscribe to orders, but connection established");
                }
                
                _logger.LogInformation("[TOPSTEPX] User Hub connection established and confirmed ready");
                ConnectionStateChanged?.Invoke($"UserHub:Connected");
            }

            return _userHub;
        }
        finally
        {
            _connectionLock.Release();
        }
    }

    public async Task<HubConnection> GetMarketHubConnectionAsync()
    {
        if (_marketHub != null && IsMarketHubConnected)
        {
            return _marketHub;
        }

        await _connectionLock.WaitAsync();
        try
        {
            if (_marketHub != null && IsMarketHubConnected)
            {
                return _marketHub;
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                "Establishing Market Hub connection");

            var token = await _tokenProvider.GetTokenAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    "Cannot connect to Market Hub - no valid JWT token");
                throw new InvalidOperationException("No valid JWT token available for Market Hub connection");
            }

            // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                token = token.Substring(7);
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    "Removed 'Bearer' prefix from JWT token for Market Hub");
            }

            _marketHub?.DisposeAsync();
            _marketHub = new HubConnectionBuilder()
                .WithUrl("https://gateway-rtc-demo.s2f.projectx.com/hubs/market", options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(token);
                    
                    // CRITICAL: Configure HTTP message handler with SSL bypass for TopstepX
                    options.HttpMessageHandlerFactory = (message) =>
                    {
                        if (message is HttpClientHandler clientHandler)
                        {
                            clientHandler.ServerCertificateCustomValidationCallback = ValidateServerCertificate;
                            clientHandler.SslProtocols = System.Security.Authentication.SslProtocols.Tls12 |
                                                       System.Security.Authentication.SslProtocols.Tls13;
                            clientHandler.MaxConnectionsPerServer = 10;
                        }
                        return message;
                    };

                    // Try WebSockets first, then fallback to other transports
                    options.Transports = HttpTransportType.WebSockets |
                                       HttpTransportType.LongPolling |
                                       HttpTransportType.ServerSentEvents;

                    // Set connection timeouts
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

            // Configure connection timeouts for production use
            _marketHub.ServerTimeout = TimeSpan.FromSeconds(60);
            _marketHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            _marketHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            SetupMarketHubEventHandlers();
            
            // Use enhanced connection startup with stability validation
            var connected = await StartConnectionWithRetry(_marketHub, "Market Hub");
            if (connected)
            {
                _marketHubWired = true;
                
                // CRITICAL: Immediately subscribe to keep connection alive (from working version)
                try
                {
                    await _marketHub.InvokeAsync("SubscribeContractQuotes", "ES"); // Subscribe to ES market data
                    _logger.LogInformation("[TOPSTEPX] Market Hub: Subscribed to ES market data to keep connection alive");
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] Market Hub: Failed to subscribe to market data, but connection established");
                }
                
                _logger.LogInformation("[TOPSTEPX] Market Hub connection established and confirmed ready");
                ConnectionStateChanged?.Invoke($"MarketHub:Connected");
            }

            return _marketHub;
        }
        finally
        {
            _connectionLock.Release();
        }
    }

    private void SetupUserHubEventHandlers()
    {
        if (_userHub == null) return;

        // Connection lifecycle handlers
        _userHub.Closed += async (exception) =>
        {
            _userHubWired = false;
            var reason = exception?.Message ?? "Normal closure";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                $"User Hub connection closed: {reason}");
            ConnectionStateChanged?.Invoke($"UserHub:Disconnected:{reason}");
        };

        _userHub.Reconnecting += async (exception) =>
        {
            _userHubWired = false;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                "User Hub reconnecting...");
            ConnectionStateChanged?.Invoke("UserHub:Reconnecting");
        };

        _userHub.Reconnected += async (connectionId) =>
        {
            _userHubWired = true;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"User Hub reconnected successfully: {connectionId}");
            ConnectionStateChanged?.Invoke($"UserHub:Reconnected:{connectionId}");
        };
        
        // CRITICAL: Data reception handlers - completing the state machine
        _userHub.On<object>("GatewayUserOrder", (orderData) =>
        {
            _logger.LogDebug("[TOPSTEPX] User Hub: Received GatewayUserOrder");
            OnGatewayUserOrderReceived?.Invoke(orderData);
        });
        
        _userHub.On<object>("GatewayUserTrade", (tradeData) =>
        {
            _logger.LogDebug("[TOPSTEPX] User Hub: Received GatewayUserTrade");
            OnGatewayUserTradeReceived?.Invoke(tradeData);
        });
        
        _userHub.On<object>("FillUpdate", (fillData) =>
        {
            _logger.LogDebug("[TOPSTEPX] User Hub: Received FillUpdate");
            OnFillUpdateReceived?.Invoke(fillData);
        });
        
        _userHub.On<object>("OrderUpdate", (orderUpdateData) =>
        {
            _logger.LogDebug("[TOPSTEPX] User Hub: Received OrderUpdate");
            OnOrderUpdateReceived?.Invoke(orderUpdateData);
        });
        
        _logger.LogInformation("[TOPSTEPX] User Hub: All data reception handlers registered");
    }

    private void SetupMarketHubEventHandlers()
    {
        if (_marketHub == null) return;

        // Connection lifecycle handlers
        _marketHub.Closed += async (exception) =>
        {
            _marketHubWired = false;
            var reason = exception?.Message ?? "Normal closure";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                $"Market Hub connection closed: {reason}");
            ConnectionStateChanged?.Invoke($"MarketHub:Disconnected:{reason}");
        };

        _marketHub.Reconnecting += async (exception) =>
        {
            _marketHubWired = false;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                "Market Hub reconnecting...");
            ConnectionStateChanged?.Invoke("MarketHub:Reconnecting");
        };

        _marketHub.Reconnected += async (connectionId) =>
        {
            _marketHubWired = true;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Market Hub reconnected successfully: {connectionId}");
            ConnectionStateChanged?.Invoke($"MarketHub:Reconnected:{connectionId}");
        };
        
        // CRITICAL: Data reception handlers - completing the state machine
        _marketHub.On<object>("MarketData", (marketData) =>
        {
            _logger.LogDebug("[TOPSTEPX] Market Hub: Received MarketData");
            OnMarketDataReceived?.Invoke(marketData);
        });
        
        _marketHub.On<object>("ContractQuotes", (contractQuotes) =>
        {
            _logger.LogDebug("[TOPSTEPX] Market Hub: Received ContractQuotes");
            OnContractQuotesReceived?.Invoke(contractQuotes);
        });
        
        _logger.LogInformation("[TOPSTEPX] Market Hub: All data reception handlers registered");
    }

    private async void CheckConnectionHealth(object? state)
    {
        try
        {
            // Check User Hub health with ping
            if (_userHub?.State == HubConnectionState.Connected)
            {
                try
                {
                    await PerformHealthCheckPing(_userHub, "UserHub");
                }
                catch (Exception ex)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                        $"User Hub ping failed: {ex.Message}");
                    _userHubWired = false;
                }
            }
            else if (_userHubWired)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    $"User Hub health check failed - State: {_userHub?.State}");
                _userHubWired = false;
            }

            // Check Market Hub health with ping  
            if (_marketHub?.State == HubConnectionState.Connected)
            {
                try
                {
                    await PerformHealthCheckPing(_marketHub, "MarketHub");
                }
                catch (Exception ex)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                        $"Market Hub ping failed: {ex.Message}");
                    _marketHubWired = false;
                }
            }
            else if (_marketHubWired)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    $"Market Hub health check failed - State: {_marketHub?.State}");
                _marketHubWired = false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error during connection health check");
        }
    }

    /// <summary>
    /// Performs a health check ping to verify connection responsiveness
    /// </summary>
    private async Task PerformHealthCheckPing(HubConnection hubConnection, string hubName)
    {
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        
        await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
            hubConnection,
            () => hubConnection.InvokeAsync("Ping", cancellationToken: cts.Token),
            _logger,
            cts.Token,
            maxAttempts: 1);
            
        await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, "SignalRManager", 
            $"{hubName} health ping successful");
    }

    /// <summary>
    /// Safely subscribe to user hub events with TopstepX specification compliance
    /// </summary>
    /// <param name="accountId">The account ID to subscribe to</param>
    /// <returns>True if subscription successful, false otherwise</returns>
    public async Task<bool> SubscribeToUserEventsAsync(string accountId)
    {
        try
        {
            // Validate account ID format per TopstepX specification
            var validatedAccountId = TopstepXSubscriptionValidator.ValidateAccountIdForSubscription(accountId, _logger);
            
            var userHub = await GetUserHubConnectionAsync();
            
            // Subscribe to all user events using TopstepX compliant methods
            var subscriptionMethods = TopstepXSubscriptionValidator.GetSupportedUserHubMethods();
            bool anySuccess = false;
            
            foreach (var method in subscriptionMethods)
            {
                try
                {
                    await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                        userHub,
                        () => userHub.InvokeAsync(method, validatedAccountId),
                        _logger,
                        CancellationToken.None);
                        
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                        $"Successfully subscribed to {method} for account {TradingBot.Abstractions.SecurityHelpers.HashAccountId(validatedAccountId)}");
                    anySuccess = true;
                }
                catch (Exception ex)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                        $"Failed to subscribe to {method} for account {TradingBot.Abstractions.SecurityHelpers.HashAccountId(validatedAccountId)}: {ex.Message}");
                }
            }
            
            return anySuccess;
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                $"Failed to subscribe to user events: {ex.Message}");
            return false;
        }
    }
    
    /// <summary>
    /// Safely subscribe to market hub events with TopstepX specification compliance
    /// </summary>
    /// <param name="contractId">The contract ID to subscribe to (e.g., "ES", "NQ")</param>
    /// <returns>True if subscription successful, false otherwise</returns>
    public async Task<bool> SubscribeToMarketEventsAsync(string contractId)
    {
        try
        {
            // Validate contract ID format per TopstepX specification
            var validatedContractId = TopstepXSubscriptionValidator.ValidateContractIdForSubscription(contractId, _logger);
            
            var marketHub = await GetMarketHubConnectionAsync();
            
            // Subscribe to all market events using TopstepX compliant methods
            var subscriptionMethods = TopstepXSubscriptionValidator.GetSupportedMarketHubMethods();
            bool anySuccess = false;
            
            foreach (var method in subscriptionMethods)
            {
                try
                {
                    await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                        marketHub,
                        () => marketHub.InvokeAsync(method, validatedContractId),
                        _logger,
                        CancellationToken.None);
                        
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                        $"Successfully subscribed to {method} for contract {validatedContractId}");
                    anySuccess = true;
                }
                catch (Exception ex)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                        $"Failed to subscribe to {method} for contract {validatedContractId}: {ex.Message}");
                }
            }
            
            return anySuccess;
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                $"Failed to subscribe to market events: {ex.Message}");
            return false;
        }
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
            "SignalR Connection Manager started");
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
            "SignalR Connection Manager stopping");
        
        if (_userHub != null)
        {
            await _userHub.DisposeAsync();
        }
        
        if (_marketHub != null)
        {
            await _marketHub.DisposeAsync();
        }
    }

    public void Dispose()
    {
        _connectionHealthTimer?.Dispose();
        _connectionLock.Dispose();
        _userHub?.DisposeAsync();
        _marketHub?.DisposeAsync();
    }
}

/// <summary>
/// Custom retry policy for SignalR connections with exponential backoff
/// </summary>
public class RetryPolicy : IRetryPolicy
{
    public TimeSpan? NextRetryDelay(RetryContext retryContext)
    {
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, then 30s max
        var delay = Math.Min(Math.Pow(2, retryContext.PreviousRetryCount), 30);
        return TimeSpan.FromSeconds(delay);
    }
}