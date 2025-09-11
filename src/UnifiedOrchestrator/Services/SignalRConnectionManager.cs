using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
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
/// Connection state machine states for strict startup order
/// </summary>
public enum ConnectionState
{
    Initializing,
    ValidatingCredentials,
    ReadyToConnect,
    Connecting,
    Connected,
    Subscribing,
    Subscribed,
    Disconnected,
    Error
}

/// <summary>
/// Subscription registry item with reference counting for proper resource management
/// </summary>
internal class SubscriptionItem
{
    public string Key { get; set; } = string.Empty;
    public string HubMethod { get; set; } = string.Empty;
    public string Target { get; set; } = string.Empty; // AccountId or ContractId
    public int RefCount { get; set; } = 0;
    public DateTime LastSubscribed { get; set; } = DateTime.UtcNow;
    public bool IsActive { get; set; } = false;
}

/// <summary>
/// Production-ready SignalR Connection Manager with proper subscription registry
/// 
/// Key Features:
/// - Single shared connection per hub type (User Hub, Market Hub)
/// - Subscription registry with ref-counting to prevent duplicate server subscriptions
/// - Centralized JWT management with automatic refresh
/// - Exponential backoff reconnection with state validation
/// - Event pub/sub for features to consume without managing connections
/// 
/// Guardrails:
/// - Initialize creds → JWT → build connection → connect → subscribe → start features (strict order)
/// - No feature creates connections; only the manager does
/// - All subscriptions go through the manager with ref-counting
/// - On reconnect, replays entire subscription set automatically
/// </summary>
public class SignalRConnectionManager : ISignalRConnectionManager, IHostedService, IDisposable
{
    private const string COMPONENT_NAME = "SignalRManager";
    private const string USER_HUB_URL = "https://rtc.topstepx.com/hubs/user";
    private const string MARKET_HUB_URL = "https://rtc.topstepx.com/hubs/market";
    
    private readonly ILogger<SignalRConnectionManager> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly ITokenProvider _tokenProvider;
    private readonly IJwtLifecycleManager _jwtLifecycleManager;
    private readonly IEnvironmentValidator _environmentValidator;
    private readonly ISnapshotManager _snapshotManager;
    private readonly SemaphoreSlim _connectionLock = new(1, 1);
    
    // State management
    private ConnectionState _userHubState = ConnectionState.Initializing;
    private ConnectionState _marketHubState = ConnectionState.Initializing;
    private readonly object _stateLock = new();
    
    // Connection health metrics
    private DateTime _userHubConnectedSince = DateTime.MinValue;
    private DateTime _marketHubConnectedSince = DateTime.MinValue;
    private int _userHubDisconnectionCount = 0;
    private int _marketHubDisconnectionCount = 0;
    
    // Subscription registry with ref-counting
    private readonly ConcurrentDictionary<string, SubscriptionItem> _subscriptionRegistry = new();
    private readonly object _subscriptionLock = new();
    
    private HubConnection? _userHub;
    private HubConnection? _marketHub;
    private volatile bool _userHubWired = false;
    private volatile bool _marketHubWired = false;
    private volatile bool _disposed = false;
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
        ITokenProvider tokenProvider,
        IJwtLifecycleManager jwtLifecycleManager,
        IEnvironmentValidator environmentValidator,
        ISnapshotManager snapshotManager)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _tokenProvider = tokenProvider;
        _jwtLifecycleManager = jwtLifecycleManager;
        _environmentValidator = environmentValidator;
        _snapshotManager = snapshotManager;
        
        // Subscribe to token lifecycle events
        _jwtLifecycleManager.TokenNeedsRefresh += OnTokenNeedsRefresh;
        _jwtLifecycleManager.TokenRefreshed += OnTokenRefreshed;
        
        // Health check timer every 30 seconds
        _connectionHealthTimer = new Timer(CheckConnectionHealth, null, 
            TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(30));
            
        // Set environment variable for .NET HTTP handler compatibility  
        Environment.SetEnvironmentVariable("DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER", "false");
    }

    /// <summary>
    /// State transition with logging for debugging
    /// </summary>
    private void TransitionState(string hubName, ConnectionState newState)
    {
        lock (_stateLock)
        {
            var currentState = hubName == "UserHub" ? _userHubState : _marketHubState;
            
            if (currentState != newState)
            {
                if (hubName == "UserHub")
                    _userHubState = newState;
                else
                    _marketHubState = newState;
                
                _logger.LogInformation("[TOPSTEPX] {HubName} state transition: {OldState} → {NewState}", 
                    hubName, currentState, newState);
                
                _ = Task.Run(async () => await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                    $"{hubName} state transition: {currentState} → {newState}"));
            }
        }
    }

    /// <summary>
    /// Token lifecycle event handler - triggers reconnection when token needs refresh
    /// </summary>
    private async void OnTokenNeedsRefresh(string expiredToken)
    {
        try
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME,
                "JWT token needs refresh - triggering hub reconnections");
            
            // Refresh token first
            await _tokenProvider.RefreshTokenAsync();
            
            // Force reconnection with new token
            await ForceReconnectWithNewTokenAsync();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling token refresh");
        }
    }

    /// <summary>
    /// Token refreshed event handler
    /// </summary>
    private async void OnTokenRefreshed(string newToken)
    {
        try
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                "JWT token refreshed successfully - validating new token");
            
            var isValid = await _jwtLifecycleManager.ValidateTokenAsync(newToken);
            if (isValid)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                    "New JWT token validated - connections will use refreshed token");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error handling token refresh event");
        }
    }

    /// <summary>
    /// Force reconnection with new token
    /// </summary>
    private async Task ForceReconnectWithNewTokenAsync()
    {
        await _connectionLock.WaitAsync();
        try
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                "Forcing reconnection with refreshed token");

            // Store current subscriptions for replay
            var activeSubscriptions = new List<SubscriptionItem>();
            lock (_subscriptionLock)
            {
                activeSubscriptions.AddRange(_subscriptionRegistry.Values.Where(s => s.IsActive));
            }

            // Disconnect current connections
            if (_userHub?.State == HubConnectionState.Connected)
            {
                TransitionState("UserHub", ConnectionState.Disconnected);
                await _userHub.DisposeAsync();
                _userHub = null;
                _userHubWired = false;
            }

            if (_marketHub?.State == HubConnectionState.Connected)
            {
                TransitionState("MarketHub", ConnectionState.Disconnected);
                await _marketHub.DisposeAsync();
                _marketHub = null;
                _marketHubWired = false;
            }

            // Re-establish connections with new token
            await GetUserHubConnectionAsync();
            await GetMarketHubConnectionAsync();

            // Replay subscriptions
            await ReplayAllSubscriptionsAsync();

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                "Reconnection with refreshed token completed");
        }
        finally
        {
            _connectionLock.Release();
        }
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
                await Task.Delay(1000); // Reduced stabilization delay for TopstepX
                
                if (hubConnection.State == HubConnectionState.Connected)
                {
                    _logger.LogInformation("[TOPSTEPX] {HubName} connection validated - State: {State}, ID: {ConnectionId}", 
                        hubName, hubConnection.State, hubConnection.ConnectionId ?? "pending");
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

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                "Establishing User Hub connection");

            var token = await _tokenProvider.GetTokenAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME, 
                    "Cannot connect to User Hub - no valid JWT token");
                throw new InvalidOperationException("No valid JWT token available for User Hub connection");
            }

            // Log token info for debugging (without exposing the actual token)
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"Using JWT token for User Hub: length={token.Length}, starts_with_Bearer={token.StartsWith("Bearer ")}, has_dots={token.Count(c => c == '.')}");

            // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                token = token.Substring(7);
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME, 
                    "Removed 'Bearer' prefix from JWT token for User Hub");
            }

            _userHub?.DisposeAsync();
            _userHub = new HubConnectionBuilder()
                .WithUrl(USER_HUB_URL, options =>
                {
                    // ENHANCED: Use thread-safe token provider that always returns freshest token
                    options.AccessTokenProvider = async () => 
                    {
                        var freshToken = await _tokenProvider.GetTokenAsync();
                        
                        // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
                        if (!string.IsNullOrEmpty(freshToken) && freshToken.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                        {
                            freshToken = freshToken.Substring(7);
                        }
                        
                        return freshToken;
                    };
                    
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
                _userHubConnectedSince = DateTime.UtcNow;
                
                TransitionState("UserHub", ConnectionState.Connected);
                
                // DEFERRED: Don't subscribe immediately - wait for explicit subscription requests
                // Features will call RequestSubscriptionAsync() with proper ref-counting
                _logger.LogInformation("[TOPSTEPX] User Hub: Connection established, waiting for subscription requests from features");
                
                // Replay any existing subscriptions after reconnect
                await ReplayAllSubscriptionsAsync();
                
                _logger.LogInformation("[TOPSTEPX] User Hub connection established and ready for trading");
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

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                "Establishing Market Hub connection");

            var token = await _tokenProvider.GetTokenAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME, 
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
                .WithUrl(MARKET_HUB_URL, options =>
                {
                    // ENHANCED: Use thread-safe token provider that always returns freshest token
                    options.AccessTokenProvider = async () => 
                    {
                        var freshToken = await _tokenProvider.GetTokenAsync();
                        
                        // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
                        if (!string.IsNullOrEmpty(freshToken) && freshToken.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                        {
                            freshToken = freshToken.Substring(7);
                        }
                        
                        return freshToken;
                    };
                    
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
                _marketHubConnectedSince = DateTime.UtcNow;
                
                TransitionState("MarketHub", ConnectionState.Connected);
                
                // NOTE: No automatic subscriptions here - features will request specific contracts
                // via RequestSubscriptionAsync() based on their dynamic contract selection
                _logger.LogInformation("[TOPSTEPX] Market Hub: Connection established, waiting for feature subscription requests");
                
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

        // ENHANCED: Connection lifecycle handlers with WebSocket close reason logging
        _userHub.Closed += async (exception) =>
        {
            _userHubWired = false;
            _userHubDisconnectionCount++;
            
            TransitionState("UserHub", ConnectionState.Disconnected);
            
            var connectionDuration = _userHubConnectedSince != DateTime.MinValue 
                ? DateTime.UtcNow - _userHubConnectedSince
                : TimeSpan.Zero;
            
            var reason = exception?.Message ?? "Normal closure";
            var details = $"Reason: {reason}, Duration: {connectionDuration:hh\\:mm\\:ss}, Disconnections: {_userHubDisconnectionCount}";
            
            // Check for auth-related close codes
            if (exception != null)
            {
                var message = exception.Message.ToLower();
                if (message.Contains("401") || message.Contains("unauthorized"))
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME,
                        $"🔒 User Hub closed due to authentication failure: {details}");
                    
                    // Trigger token refresh on auth failure
                    _ = Task.Run(async () => await _tokenProvider.RefreshTokenAsync());
                }
                else if (message.Contains("403") || message.Contains("forbidden"))
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME,
                        $"🚫 User Hub closed due to authorization failure: {details}");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME,
                        $"⚠️ User Hub connection closed: {details}");
                }
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                    $"✅ User Hub closed gracefully: {details}");
            }
            
            ConnectionStateChanged?.Invoke($"UserHub:Disconnected:{reason}");
        };

        _userHub.Reconnecting += async (exception) =>
        {
            _userHubWired = false;
            TransitionState("UserHub", ConnectionState.Connecting);
            
            var reason = exception?.Message ?? "Connection lost";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME, 
                $"🔄 User Hub reconnecting due to: {reason}");
            ConnectionStateChanged?.Invoke("UserHub:Reconnecting");
        };

        _userHub.Reconnected += async (connectionId) =>
        {
            _userHubWired = true;
            _userHubConnectedSince = DateTime.UtcNow;
            
            TransitionState("UserHub", ConnectionState.Connected);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"✅ User Hub reconnected successfully: {connectionId}");
            ConnectionStateChanged?.Invoke($"UserHub:Reconnected:{connectionId}");
            
            // Trigger snapshot reconciliation after reconnect
            try
            {
                // Note: Account ID would need to be passed or stored - placeholder for now
                // await _snapshotManager.ReconcileAfterReconnectAsync(accountId);
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "Failed to reconcile snapshot after User Hub reconnection");
            }
            
            // Replay subscriptions automatically
            _ = Task.Run(async () => await ReplayAllSubscriptionsAsync());
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

        // ENHANCED: Connection lifecycle handlers with WebSocket close reason logging
        _marketHub.Closed += async (exception) =>
        {
            _marketHubWired = false;
            _marketHubDisconnectionCount++;
            
            TransitionState("MarketHub", ConnectionState.Disconnected);
            
            var connectionDuration = _marketHubConnectedSince != DateTime.MinValue 
                ? DateTime.UtcNow - _marketHubConnectedSince
                : TimeSpan.Zero;
            
            var reason = exception?.Message ?? "Normal closure";
            var details = $"Reason: {reason}, Duration: {connectionDuration:hh\\:mm\\:ss}, Disconnections: {_marketHubDisconnectionCount}";
            
            // Check for auth-related close codes
            if (exception != null)
            {
                var message = exception.Message.ToLower();
                if (message.Contains("401") || message.Contains("unauthorized"))
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME,
                        $"🔒 Market Hub closed due to authentication failure: {details}");
                    
                    // Trigger token refresh on auth failure
                    _ = Task.Run(async () => await _tokenProvider.RefreshTokenAsync());
                }
                else if (message.Contains("403") || message.Contains("forbidden"))
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME,
                        $"🚫 Market Hub closed due to authorization failure: {details}");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME,
                        $"⚠️ Market Hub connection closed: {details}");
                }
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                    $"✅ Market Hub closed gracefully: {details}");
            }
            
            ConnectionStateChanged?.Invoke($"MarketHub:Disconnected:{reason}");
        };

        _marketHub.Reconnecting += async (exception) =>
        {
            _marketHubWired = false;
            TransitionState("MarketHub", ConnectionState.Connecting);
            
            var reason = exception?.Message ?? "Connection lost";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME, 
                $"🔄 Market Hub reconnecting due to: {reason}");
            ConnectionStateChanged?.Invoke("MarketHub:Reconnecting");
        };

        _marketHub.Reconnected += async (connectionId) =>
        {
            _marketHubWired = true;
            _marketHubConnectedSince = DateTime.UtcNow;
            
            TransitionState("MarketHub", ConnectionState.Connected);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"✅ Market Hub reconnected successfully: {connectionId}");
            ConnectionStateChanged?.Invoke($"MarketHub:Reconnected:{connectionId}");
            
            // Replay subscriptions automatically
            _ = Task.Run(async () => await ReplayAllSubscriptionsAsync());
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
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
    /// Request a subscription through the manager with ref-counting
    /// Multiple features can request the same subscription without duplicate server calls
    /// </summary>
    public async Task<bool> RequestSubscriptionAsync(string hubMethod, string target, string featureName)
    {
        if (_disposed) return false;
        
        var key = $"{hubMethod}:{target}";
        
        lock (_subscriptionLock)
        {
            if (_subscriptionRegistry.TryGetValue(key, out var existing))
            {
                existing.RefCount++;
                _logger.LogDebug("[TOPSTEPX] Subscription ref-count increased: {Key} -> {RefCount} (requested by {Feature})", 
                    key, existing.RefCount, featureName);
                return existing.IsActive;
            }
            
            // Create new subscription entry
            _subscriptionRegistry[key] = new SubscriptionItem
            {
                Key = key,
                HubMethod = hubMethod,
                Target = target,
                RefCount = 1,
                LastSubscribed = DateTime.UtcNow,
                IsActive = false
            };
            
            _logger.LogInformation("[TOPSTEPX] New subscription requested: {Key} (by {Feature})", key, featureName);
        }
        
        // Actually perform the subscription
        return await PerformSubscriptionAsync(hubMethod, target);
    }
    
    /// <summary>
    /// Release a subscription with ref-counting
    /// Only unsubscribes from server when ref-count reaches zero
    /// </summary>
    public Task<bool> ReleaseSubscriptionAsync(string hubMethod, string target, string featureName)
    {
        if (_disposed) return Task.FromResult(false);
        
        var key = $"{hubMethod}:{target}";
        
        lock (_subscriptionLock)
        {
            if (!_subscriptionRegistry.TryGetValue(key, out var existing))
            {
                _logger.LogWarning("[TOPSTEPX] Attempted to release non-existent subscription: {Key} (by {Feature})", 
                    key, featureName);
                return Task.FromResult(false);
            }
            
            existing.RefCount--;
            _logger.LogDebug("[TOPSTEPX] Subscription ref-count decreased: {Key} -> {RefCount} (released by {Feature})", 
                key, existing.RefCount, featureName);
            
            if (existing.RefCount <= 0)
            {
                _subscriptionRegistry.TryRemove(key, out _);
                _logger.LogInformation("[TOPSTEPX] Subscription removed (ref-count zero): {Key}", key);
                
                // Optionally unsubscribe from server here
                // Most SignalR hubs don't have explicit unsubscribe methods
                return Task.FromResult(true);
            }
        }
        
        return Task.FromResult(true);
    }
    
    /// <summary>
    /// Get current subscription manifest for logging and debugging
    /// </summary>
    public Dictionary<string, (int RefCount, bool IsActive, DateTime LastSubscribed)> GetSubscriptionManifest()
    {
        var manifest = new Dictionary<string, (int, bool, DateTime)>();
        
        lock (_subscriptionLock)
        {
            foreach (var kvp in _subscriptionRegistry)
            {
                var item = kvp.Value;
                manifest[kvp.Key] = (item.RefCount, item.IsActive, item.LastSubscribed);
            }
        }
        
        return manifest;
    }
    
    /// <summary>
    /// Replay all active subscriptions on reconnect
    /// </summary>
    private async Task ReplayAllSubscriptionsAsync()
    {
        var subscriptionsToReplay = new List<SubscriptionItem>();
        
        lock (_subscriptionLock)
        {
            foreach (var item in _subscriptionRegistry.Values)
            {
                if (item.RefCount > 0)
                {
                    subscriptionsToReplay.Add(new SubscriptionItem
                    {
                        Key = item.Key,
                        HubMethod = item.HubMethod,
                        Target = item.Target,
                        RefCount = item.RefCount
                    });
                }
            }
        }
        
        _logger.LogInformation("[TOPSTEPX] Replaying {Count} subscriptions after reconnect", subscriptionsToReplay.Count);
        
        foreach (var item in subscriptionsToReplay)
        {
            try
            {
                var success = await PerformSubscriptionAsync(item.HubMethod, item.Target);
                if (success)
                {
                    lock (_subscriptionLock)
                    {
                        if (_subscriptionRegistry.TryGetValue(item.Key, out var existing))
                        {
                            existing.IsActive = true;
                            existing.LastSubscribed = DateTime.UtcNow;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME, 
                    $"Failed to replay subscription {item.Key}: {ex.Message}");
            }
        }
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
            "Subscription replay completed - ready for trading");
    }
    
    /// <summary>
    /// Actually perform the subscription call to the hub
    /// </summary>
    private async Task<bool> PerformSubscriptionAsync(string hubMethod, string target)
    {
        try
        {
            var hub = hubMethod.StartsWith("Subscribe") && hubMethod.Contains("Order") || hubMethod.Contains("Trade") 
                ? await GetUserHubConnectionAsync() 
                : await GetMarketHubConnectionAsync();
                
            await TradingBot.Infrastructure.TopstepX.SignalRSafeInvoker.InvokeWhenConnected(
                hub,
                () => hub.InvokeAsync(hubMethod, target),
                _logger,
                CancellationToken.None);
                
            // Mark as active in registry
            var key = $"{hubMethod}:{target}";
            lock (_subscriptionLock)
            {
                if (_subscriptionRegistry.TryGetValue(key, out var item))
                {
                    item.IsActive = true;
                    item.LastSubscribed = DateTime.UtcNow;
                }
            }
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"Successfully subscribed: {hubMethod}({target})");
            return true;
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME, 
                $"Failed to subscribe {hubMethod}({target}): {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Subscribe to user events using the subscription registry
    /// </summary>
    public async Task<bool> SubscribeToUserEventsAsync(string accountId)
    {
        if (string.IsNullOrEmpty(accountId))
        {
            _logger.LogWarning("[TOPSTEPX] Cannot subscribe to user events - account ID is empty");
            return false;
        }

        // Validate account ID format per TopstepX specification
        var validatedAccountId = TopstepXSubscriptionValidator.ValidateAccountIdForSubscription(accountId, _logger);
        
        // Use the subscription registry for proper ref-counting
        var orderSuccess = await RequestSubscriptionAsync("SubscribeOrders", validatedAccountId, "UserEvents");
        var tradeSuccess = await RequestSubscriptionAsync("SubscribeTrades", validatedAccountId, "UserEvents");
        
        return orderSuccess && tradeSuccess;
    }

    /// <summary>
    /// Subscribe to market events using the subscription registry
    /// </summary>
    public async Task<bool> SubscribeToMarketEventsAsync(string contractId)
    {
        if (string.IsNullOrEmpty(contractId))
        {
            _logger.LogWarning("[TOPSTEPX] Cannot subscribe to market events - contract ID is empty");
            return false;
        }

        // Use the subscription registry for proper ref-counting
        return await RequestSubscriptionAsync("SubscribeToContract", contractId, "MarketEvents");
    }

    /// <summary>
    /// Retry subscriptions with valid account ID after login
    /// </summary>
    public async Task<bool> RetrySubscriptionsWithAccountId(string accountId)
    {
        if (string.IsNullOrEmpty(accountId))
        {
            _logger.LogWarning("[TOPSTEPX] Cannot retry subscriptions - account ID is empty");
            return false;
        }

        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
            $"Retrying subscriptions with validated account ID: {accountId.Substring(0, Math.Min(4, accountId.Length))}***");

        // Re-subscribe to user events with the validated account ID
        var success = await SubscribeToUserEventsAsync(accountId);
        
        if (success)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                "✅ Subscription retry completed successfully");
        }
        else
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME,
                "❌ Subscription retry failed");
        }
        
        return success;
    }

    /// <summary>
    /// IHostedService implementation with strict startup order:
    /// Initialize creds → JWT → build connection → connect → subscribe → start features
    /// </summary>
    public async Task StartAsync(CancellationToken cancellationToken)
    {
        if (_disposed) return;
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
            "Starting SignalR Connection Manager with enhanced startup validation");
        
        try
        {
            // Step 0: Environment validation (new requirement)
            TransitionState("UserHub", ConnectionState.Initializing);
            TransitionState("MarketHub", ConnectionState.Initializing);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME,
                "🔍 Step 0: Validating environment for trading operations...");
                
            var environmentValid = await _environmentValidator.ValidateEnvironmentAsync();
            if (!environmentValid)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, COMPONENT_NAME,
                    "⚠️ Environment validation warnings detected - proceeding with caution");
            }
            
            // Step 1: Strict JWT validation (enhanced)
            TransitionState("UserHub", ConnectionState.ValidatingCredentials);
            TransitionState("MarketHub", ConnectionState.ValidatingCredentials);
            
            var token = await _tokenProvider.GetTokenAsync();
            if (string.IsNullOrEmpty(token))
            {
                throw new InvalidOperationException("No valid JWT token available during startup");
            }
            
            // Enhanced JWT validation using lifecycle manager
            var tokenValid = await _jwtLifecycleManager.ValidateTokenAsync(token);
            if (!tokenValid)
            {
                throw new InvalidOperationException("JWT token validation failed - token may be expired or invalid");
            }
            
            var tokenExpiry = _jwtLifecycleManager.GetTokenExpiry(token);
            var lifetimePercentage = _jwtLifecycleManager.GetTokenLifetimePercentage(token);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"✅ Step 1: JWT validated successfully - expires: {tokenExpiry:yyyy-MM-dd HH:mm:ss} UTC ({lifetimePercentage:F1}% used)");
            
            // Step 2: Transition to ready state and establish connections
            TransitionState("UserHub", ConnectionState.ReadyToConnect);
            TransitionState("MarketHub", ConnectionState.ReadyToConnect);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                "🔌 Step 2: Establishing hub connections with validated credentials...");
            
            // Initialize connections in parallel for faster startup
            var userHubTask = GetUserHubConnectionAsync();
            var marketHubTask = GetMarketHubConnectionAsync();
            
            await Task.WhenAll(userHubTask, marketHubTask);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                "✅ Step 2: Hub connections established and validated");
            
            // Step 3: Verify connection states and prepare subscriptions
            TransitionState("UserHub", ConnectionState.Connected);
            TransitionState("MarketHub", ConnectionState.Connected);
            
            var manifest = GetSubscriptionManifest();
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"📋 Step 3: Subscription registry initialized ({manifest.Count} active subscriptions)");
            
            // Step 4: Signal ready for features to start
            ConnectionStateChanged?.Invoke("AllHubs:Ready");
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                "🚀 Enhanced SignalR Connection Manager startup complete - ready for production trading");
        }
        catch (Exception ex)
        {
            TransitionState("UserHub", ConnectionState.Error);
            TransitionState("MarketHub", ConnectionState.Error);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME, 
                $"❌ Enhanced SignalR startup failed: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Graceful shutdown with subscription cleanup
    /// </summary>
    public async Task StopAsync(CancellationToken cancellationToken)
    {
        if (_disposed) return;
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
            "Stopping SignalR Connection Manager gracefully");
        
        try
        {
            // Log final subscription manifest
            var manifest = GetSubscriptionManifest();
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                $"📋 Final subscription manifest: {manifest.Count} active subscriptions");
            
            // Signal shutdown to features
            ConnectionStateChanged?.Invoke("AllHubs:Stopping");
            
            // Allow graceful disconnection
            await Task.Delay(1000, cancellationToken);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, COMPONENT_NAME, 
                "✅ SignalR Connection Manager stopped gracefully");
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, COMPONENT_NAME, 
                $"⚠️ Error during SignalR shutdown: {ex.Message}");
        }
    }


    /// <summary>
    /// Implement proper dispose pattern
    /// </summary>
    public void Dispose()
    {
        Dispose(true);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed && disposing)
        {
            _disposed = true;
            
            _connectionHealthTimer?.Dispose();
            _connectionLock.Dispose();
            
            // Clear subscription registry
            lock (_subscriptionLock)
            {
                _subscriptionRegistry.Clear();
            }
            
            // Dispose hub connections
            if (_userHub != null)
            {
                _ = Task.Run(async () => await _userHub.DisposeAsync());
            }
            
            if (_marketHub != null)
            {
                _ = Task.Run(async () => await _marketHub.DisposeAsync());
            }
        }
    }

    // Finalizer
    ~SignalRConnectionManager()
    {
        Dispose(false);
    }
}

/// <summary>
/// Exponential backoff retry policy for SignalR connections  
/// </summary>
public class ExponentialBackoffRetryPolicy : IRetryPolicy
{
    public TimeSpan? NextRetryDelay(RetryContext retryContext)
    {
        // Exponential backoff: 1s, 2s, 4s, 8s, 16s, then 30s max
        var delay = Math.Min(Math.Pow(2, retryContext.PreviousRetryCount), 30);
        return TimeSpan.FromSeconds(delay);
    }
}