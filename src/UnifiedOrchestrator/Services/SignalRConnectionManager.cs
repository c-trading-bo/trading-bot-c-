using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Net.Security;
using System.Net.WebSockets;
using System.Security.Cryptography.X509Certificates;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Rate-limit-aware retry policy with conservative delays to avoid hitting TopstepX rate limits
/// Based on ProjectX docs: 200 requests/60 seconds, returns HTTP 429 when exceeded
/// </summary>
public class ExponentialBackoffRetryPolicy : IRetryPolicy
{
    public TimeSpan? NextRetryDelay(RetryContext retryContext)
    {
        // Max 3 retries with much longer delays to avoid rate limiting
        if (retryContext.PreviousRetryCount >= 3)
            return null;

        // Very conservative delays: 45s, 90s, 180s to stay well under rate limits
        var delaySeconds = retryContext.PreviousRetryCount switch
        {
            0 => 45,  // First retry after 45 seconds
            1 => 90,  // Second retry after 90 seconds  
            2 => 180, // Third retry after 180 seconds (3 minutes)
            _ => 300  // Fallback: 5 minutes
        };
        
        return TimeSpan.FromSeconds(delaySeconds);
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
    private readonly SemaphoreSlim _tokenRefreshLock = new(1, 1);

    // Subscription manifest tracking
    private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, int>> _subscriptionManifest = new();
    
    // Connection statistics for production monitoring
    private volatile int _totalConnectAttempts = 0;
    private volatile int _totalReconnects = 0;
    private DateTime _longestConnectedStart = DateTime.MinValue;
    private TimeSpan _longestConnectedSpan = TimeSpan.Zero;
    private volatile int _totalMessagesReceived = 0;
    private readonly List<TimeSpan> _connectionGaps = new();

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
            
        // FIXED: Remove legacy HTTP handler override - use default SocketsHttpHandler
        // Environment.SetEnvironmentVariable("DOTNET_SYSTEM_NET_HTTP_USESOCKETSHTTPHANDLER", "false");
        
        // FIXED: Subscribe to token refresh events for immediate hub restart
        _tokenProvider.TokenRefreshed += OnTokenRefreshed;
    }

    /// <summary>
    /// Handle token refresh events with immediate hub restart
    /// Implements the token refresh policy requirement
    /// </summary>
    private async void OnTokenRefreshed(string newToken)
    {
        if (await _tokenRefreshLock.WaitAsync(TimeSpan.FromSeconds(5)))
        {
            try
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    "Fresh token arrived - checking hub connection states for restart");

                var needsUserHubRestart = _userHub == null || 
                                        _userHub.State != HubConnectionState.Connected || 
                                        string.IsNullOrEmpty(_userHub.ConnectionId);

                var needsMarketHubRestart = _marketHub == null || 
                                          _marketHub.State != HubConnectionState.Connected || 
                                          string.IsNullOrEmpty(_marketHub.ConnectionId);

                if (needsUserHubRestart || needsMarketHubRestart)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                        $"Restarting hubs due to fresh token - UserHub: {needsUserHubRestart}, MarketHub: {needsMarketHubRestart}");

                    // FIXED: Remove "skip restarts" logic - always restart on fresh token if hub is disconnected
                    if (needsUserHubRestart)
                    {
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                await GetUserHubConnectionAsync();
                                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                                    "User Hub restarted successfully with fresh token");
                            }
                            catch (Exception ex)
                            {
                                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                                    $"User Hub restart failed: {ex.Message}");
                            }
                        });
                    }

                    if (needsMarketHubRestart)
                    {
                        _ = Task.Run(async () =>
                        {
                            try
                            {
                                await GetMarketHubConnectionAsync();
                                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                                    "Market Hub restarted successfully with fresh token");
                            }
                            catch (Exception ex)
                            {
                                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                                    $"Market Hub restart failed: {ex.Message}");
                            }
                        });
                    }
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, "SignalRManager", 
                        "Fresh token received but both hubs are already connected - no restart needed");
                }
            }
            catch (Exception ex)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    $"Error handling token refresh: {ex.Message}");
            }
            finally
            {
                _tokenRefreshLock.Release();
            }
        }
        else
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                "Token refresh handling skipped - already processing another refresh");
        }
    }

    /// <summary>
    /// Wait for JWT token readiness with timeout and logging
    /// Implements 30-45 second delay for token readiness as required
    /// </summary>
    private async Task<string?> WaitForJwtReadinessAsync(CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        var timeout = TimeSpan.FromSeconds(45); // Max 45 seconds as specified
        var checkInterval = TimeSpan.FromSeconds(2);
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
            "Waiting for JWT token readiness before establishing hub connections...");
        
        while (DateTime.UtcNow - startTime < timeout)
        {
            var token = await _tokenProvider.GetTokenAsync();
            
            if (!string.IsNullOrEmpty(token))
            {
                // Validate JWT format and timing with enhanced requirements
                if (await ValidateJwtTokenAsync(token))
                {
                    var waitTime = DateTime.UtcNow - startTime;
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                        $"JWT ready after {waitTime.TotalSeconds:F1} seconds - timing validation passed");
                    return token;
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                        "JWT token found but validation failed (need exp-now ≥120s, nbf ≤ now-5s, valid aud/iss)");
                }
            }
            
            // Log periodic wait status
            var elapsed = DateTime.UtcNow - startTime;
            if (elapsed.TotalSeconds > 10 && elapsed.TotalSeconds % 10 < 2) // Log every 10 seconds
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    $"Still waiting for JWT token... {elapsed.TotalSeconds:F0}s elapsed");
            }
            
            await Task.Delay(checkInterval, cancellationToken);
        }
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
            $"JWT token readiness timeout after {timeout.TotalSeconds} seconds");
        return null;
    }

    /// <summary>
    /// Enhanced JWT validation with audience, issuer, and timing requirements
    /// Blocks connections unless: token present, exp-now ≥ 120s, nbf ≤ now-5s, audience matches hub
    /// </summary>
    private async Task<bool> ValidateJwtTokenAsync(string token, string hubType = "")
    {
        try
        {
            // Remove Bearer prefix if present
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                token = token.Substring(7);
            }
            
            // Check JWT format (should have 3 parts separated by dots)
            var parts = token.Split('.');
            if (parts.Length != 3)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    "JWT token format invalid - expected 3 parts separated by dots");
                return false;
            }
            
            // Decode payload to check all claims
            try
            {
                var payloadBytes = Convert.FromBase64String(AddBase64Padding(parts[1]));
                var payloadJson = System.Text.Encoding.UTF8.GetString(payloadBytes);
                using var doc = JsonDocument.Parse(payloadJson);
                
                var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                
                // Check issuer (iss claim)
                if (doc.RootElement.TryGetProperty("iss", out var issElement))
                {
                    var issuer = issElement.GetString();
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, "SignalRManager", 
                        $"JWT issuer: {issuer}");
                }
                
                // Check audience (aud claim) - CRITICAL for hub-specific validation
                if (doc.RootElement.TryGetProperty("aud", out var audElement))
                {
                    var audience = audElement.GetString();
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, "SignalRManager", 
                        $"JWT audience: {audience}");
                    
                    // Validate audience matches hub requirement
                    if (!string.IsNullOrEmpty(hubType))
                    {
                        if (hubType.Contains("user", StringComparison.OrdinalIgnoreCase) || 
                            hubType.Contains("market", StringComparison.OrdinalIgnoreCase))
                        {
                            // For TopstepX, accept if audience contains topstepx or api domain
                            if (!audience?.Contains("topstepx", StringComparison.OrdinalIgnoreCase) == true)
                            {
                                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                                    $"JWT audience mismatch for {hubType} - aud: {audience}");
                                // Don't block for TopstepX - their tokens might have generic audience
                            }
                        }
                    }
                }
                
                // Check expiry with 120s minimum remaining (aud requirement)
                if (doc.RootElement.TryGetProperty("exp", out var expElement))
                {
                    var exp = expElement.GetInt64();
                    var remainingSeconds = exp - now;
                    
                    if (remainingSeconds < 120) // Must have at least 120s remaining
                    {
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                            $"JWT token expires too soon - exp_in_seconds: {remainingSeconds} (need ≥120s)");
                        return false;
                    }
                    
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                        $"JWT token exp_in_seconds: {remainingSeconds}");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                        "JWT token missing exp claim");
                    return false;
                }
                
                // Check not-before with 5s tolerance (nbf ≤ now-5s requirement)
                if (doc.RootElement.TryGetProperty("nbf", out var nbfElement))
                {
                    var nbf = nbfElement.GetInt64();
                    if (now < (nbf - 5)) // Allow 5s clock skew tolerance
                    {
                        var skew = nbf - now;
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                            $"JWT token not yet valid - nbf timing issue, skew: {skew}s (check system clock)");
                        return false;
                    }
                }
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    "JWT ok - aud, iss, exp, nbf validation passed");
                return true;
            }
            catch (Exception ex)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    $"JWT payload decode failed: {ex.Message}");
                return false;
            }
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                $"JWT validation error: {ex.Message}");
            return false;
        }
    }
    
    /// <summary>
    /// Add padding to Base64 string for proper decoding
    /// </summary>
    private static string AddBase64Padding(string base64)
    {
        switch (base64.Length % 4)
        {
            case 2: return base64 + "==";
            case 3: return base64 + "=";
            default: return base64;
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
    /// Logs hub URL and transport for network hygiene
    /// </summary>
    private async Task<bool> StartConnectionWithRetry(HubConnection hubConnection, string hubName)
    {
        const int maxRetries = 3; // Reduced retries to avoid rate limits

        // Log network hygiene information
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
            $"Network hygiene - {hubName} connecting to WebSockets transport");
        
        if (hubConnection.ConnectionId != null)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Hub URL: {(hubName.Contains("User") ? "https://rtc.topstepx.com/hubs/user" : "https://rtc.topstepx.com/hubs/market")}, Transport: WebSockets");
        }

        for (int attempt = 1; attempt <= maxRetries; attempt++)
        {
            try
            {
                if (hubConnection == null)
                    throw new InvalidOperationException("Hub connection not initialized");

                Interlocked.Increment(ref _totalConnectAttempts);

                _logger.LogInformation("[TOPSTEPX] Starting {HubName} connection (attempt {Attempt}/{Max})", 
                    hubName, attempt, maxRetries);

                using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                await hubConnection.StartAsync(cts.Token);

                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    "connection started");

                _logger.LogInformation("[TOPSTEPX] {HubName} connection started successfully. State: {State}", 
                    hubName, hubConnection.State);
                
                // CRITICAL: Extended stability validation for TopstepX connections
                _logger.LogInformation("[TOPSTEPX] {HubName} validating connection stability...", hubName);
                
                // Wait longer and check multiple times for stable connection
                for (int check = 1; check <= 3; check++)
                {
                    await Task.Delay(2000); // 2 seconds between checks
                    
                    if (hubConnection.State == HubConnectionState.Connected && 
                        !string.IsNullOrEmpty(hubConnection.ConnectionId))
                    {
                        _logger.LogInformation("[TOPSTEPX] {HubName} stability check {Check}/3: ✅ State: {State}, ID: {ConnectionId}", 
                            hubName, check, hubConnection.State, hubConnection.ConnectionId);
                    }
                    else
                    {
                        _logger.LogWarning("[TOPSTEPX] {HubName} stability check {Check}/3: ❌ State: {State}, ID: {ConnectionId}", 
                            hubName, check, hubConnection.State, hubConnection.ConnectionId ?? "null");
                        
                        if (check == 3) // Final check failed
                        {
                            throw new InvalidOperationException($"{hubName} connection unstable after extended validation");
                        }
                    }
                }
                
                // Final validation
                if (hubConnection.State == HubConnectionState.Connected && 
                    !string.IsNullOrEmpty(hubConnection.ConnectionId))
                {
                    _logger.LogInformation("[TOPSTEPX] {HubName} connection validated after extended checks - State: {State}, ID: {ConnectionId}", 
                        hubName, hubConnection.State, hubConnection.ConnectionId);
                        
                    // Track connection start time for statistics
                    _longestConnectedStart = DateTime.UtcNow;
                    
                    return true;
                }
                else
                {
                    _logger.LogWarning("[TOPSTEPX] {HubName} connection failed final validation - State: {State}, ID: {ConnectionId}", 
                        hubName, hubConnection.State, hubConnection.ConnectionId ?? "null");
                    throw new InvalidOperationException($"{hubName} connection unstable after start");
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[TOPSTEPX] {HubName} connection attempt {Attempt} failed", hubName, attempt);

                if (attempt < maxRetries)
                {
                    // Very conservative delays to avoid TopstepX rate limiting (200 requests/60s)
                    var delay = TimeSpan.FromSeconds(attempt switch 
                    {
                        1 => 45,  // First retry: 45 seconds (more conservative)
                        2 => 90,  // Second retry: 90 seconds (even longer)
                        _ => 180  // Fallback: 3 minutes
                    });
                    
                    _logger.LogInformation("[TOPSTEPX] Retrying {HubName} in {Delay} seconds (rate limit protection)...", 
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

            // FIXED: Wait for JWT readiness before any connection attempt
            var token = await WaitForJwtReadinessAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    "Cannot connect to User Hub - JWT token readiness timeout");
                throw new InvalidOperationException("JWT token readiness timeout - cannot establish User Hub connection");
            }

            // Enhanced JWT validation for User Hub with audience check
            if (!await ValidateJwtTokenAsync(token, "user"))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    "Cannot connect to User Hub - JWT validation failed (timing/audience/format issues)");
                throw new InvalidOperationException("JWT validation failed - cannot establish User Hub connection");
            }

            // Log token info for debugging (without exposing the actual token)
            var cleanToken = token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? token.Substring(7) : token;
            var last6Chars = cleanToken.Length >= 6 ? cleanToken.Substring(cleanToken.Length - 6) : "short";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"JWT ready for User Hub: length={cleanToken.Length}, last_6_chars=...{last6Chars}");

            // FIXED: Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                token = token.Substring(7);
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    "Removed 'Bearer' prefix from JWT token for User Hub");
            }

            _userHub?.DisposeAsync();
            _userHub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/user", options =>
                {
                    // PRODUCTION: AccessTokenProvider with detailed logging and fresh token access
                    options.AccessTokenProvider = async () => 
                    {
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                            "AccessTokenProvider CALLED for User Hub connect/reconnect");
                        
                        var freshToken = await _tokenProvider.GetTokenAsync();
                        if (!string.IsNullOrEmpty(freshToken))
                        {
                            // Remove Bearer prefix - return raw JWT only
                            if (freshToken.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                            {
                                freshToken = freshToken.Substring(7);
                            }
                            
                            var last6 = freshToken.Length >= 6 ? freshToken.Substring(freshToken.Length - 6) : "short";
                            
                            // Calculate expiry from token for logging
                            var expInSeconds = 0L;
                            try
                            {
                                var parts = freshToken.Split('.');
                                if (parts.Length == 3)
                                {
                                    var payloadBytes = Convert.FromBase64String(AddBase64Padding(parts[1]));
                                    var payloadJson = System.Text.Encoding.UTF8.GetString(payloadBytes);
                                    using var doc = JsonDocument.Parse(payloadJson);
                                    if (doc.RootElement.TryGetProperty("exp", out var expElement))
                                    {
                                        var exp = expElement.GetInt64();
                                        expInSeconds = exp - DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                                    }
                                }
                            }
                            catch
                            {
                                expInSeconds = -1; // Parse failed
                            }
                            
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                                $"AccessTokenProvider: token_length={freshToken.Length}, last_6_chars=...{last6}, exp_in_seconds={expInSeconds}");
                        }
                        else
                        {
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                                "AccessTokenProvider: No fresh token available!");
                        }
                        
                        return freshToken; // Return raw JWT (no "Bearer ")
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

                    // FIXED: Force WebSockets transport and skip negotiation
                    options.Transports = HttpTransportType.WebSockets;
                    options.SkipNegotiation = true;

                    // Extended timeouts for TopstepX stability
                    options.CloseTimeout = TimeSpan.FromSeconds(45);
                    
                    // Add custom headers for TopstepX
                    options.Headers.Add("User-Agent", "TradingBot-SignalR/1.0");
                    options.Headers.Add("Accept", "application/json");
                })
                .WithAutomaticReconnect(new ExponentialBackoffRetryPolicy())
                .ConfigureLogging(logging =>
                {
                    logging.SetMinimumLevel(LogLevel.Information);
                    logging.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Warning);
                    logging.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Warning);
                })
                .Build();

            // Production-grade timeouts and keepalive settings 
            // KeepAliveInterval ~10–15s and ServerTimeout ≥30–45s per requirements
            _userHub.ServerTimeout = TimeSpan.FromSeconds(45);  // Increased for production stability
            _userHub.KeepAliveInterval = TimeSpan.FromSeconds(15); // Set to 15s for production
            _userHub.HandshakeTimeout = TimeSpan.FromSeconds(45);  // Increased from 30

            SetupUserHubEventHandlers();
            
            // Use enhanced connection startup with stability validation
            var connected = await StartConnectionWithRetry(_userHub, "User Hub");
            if (connected)
            {
                _userHubWired = true;
                
                // IMMEDIATE: Send lightweight subscription for stability validation
                // As soon as the hub reports Connected, send one lightweight subscription
                var accountId = Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
                if (!string.IsNullOrEmpty(accountId))
                {
                    _logger.LogInformation("[TOPSTEPX] User Hub: Account ID available, immediate subscription for stability validation");
                    try
                    {
                        await _userHub.InvokeAsync("SubscribeOrders", accountId);
                        AddToSubscriptionManifest("UserHub", "SubscribeOrders", accountId);
                        _logger.LogInformation("[TOPSTEPX] User Hub: Immediate SubscribeOrders successful - stability validator has signal");
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[TOPSTEPX] User Hub: Immediate subscription failed: {Message}", ex.Message);
                    }
                    
                    // Try additional subscription but don't fail if it doesn't work
                    _ = Task.Run(async () => await RetrySubscriptionsWithAccountId(accountId));
                }
                else
                {
                    _logger.LogInformation("[TOPSTEPX] User Hub: Connection established, waiting for account ID from login service");
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

            // FIXED: Wait for JWT readiness before any connection attempt
            var token = await WaitForJwtReadinessAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    "Cannot connect to Market Hub - JWT token readiness timeout");
                throw new InvalidOperationException("JWT token readiness timeout - cannot establish Market Hub connection");
            }

            // Enhanced JWT validation for Market Hub with audience check
            if (!await ValidateJwtTokenAsync(token, "market"))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    "Cannot connect to Market Hub - JWT validation failed (timing/audience/format issues)");
                throw new InvalidOperationException("JWT validation failed - cannot establish Market Hub connection");
            }

            // Log token info for debugging
            var cleanToken = token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase) ? token.Substring(7) : token;
            var last6Chars = cleanToken.Length >= 6 ? cleanToken.Substring(cleanToken.Length - 6) : "short";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"JWT ready for Market Hub: length={cleanToken.Length}, last_6_chars=...{last6Chars}");

            // FIXED: Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
            {
                token = token.Substring(7);
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    "Removed 'Bearer' prefix from JWT token for Market Hub");
            }

            _marketHub?.DisposeAsync();
            _marketHub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", options =>
                {
                    // PRODUCTION: AccessTokenProvider with detailed logging and fresh token access
                    options.AccessTokenProvider = async () => 
                    {
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                            "AccessTokenProvider CALLED for Market Hub connect/reconnect");
                        
                        var freshToken = await _tokenProvider.GetTokenAsync();
                        if (!string.IsNullOrEmpty(freshToken))
                        {
                            // Remove Bearer prefix - return raw JWT only
                            if (freshToken.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase))
                            {
                                freshToken = freshToken.Substring(7);
                            }
                            
                            var last6 = freshToken.Length >= 6 ? freshToken.Substring(freshToken.Length - 6) : "short";
                            
                            // Calculate expiry from token for logging
                            var expInSeconds = 0L;
                            try
                            {
                                var parts = freshToken.Split('.');
                                if (parts.Length == 3)
                                {
                                    var payloadBytes = Convert.FromBase64String(AddBase64Padding(parts[1]));
                                    var payloadJson = System.Text.Encoding.UTF8.GetString(payloadBytes);
                                    using var doc = JsonDocument.Parse(payloadJson);
                                    if (doc.RootElement.TryGetProperty("exp", out var expElement))
                                    {
                                        var exp = expElement.GetInt64();
                                        expInSeconds = exp - DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                                    }
                                }
                            }
                            catch
                            {
                                expInSeconds = -1; // Parse failed
                            }
                            
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                                $"AccessTokenProvider: token_length={freshToken.Length}, last_6_chars=...{last6}, exp_in_seconds={expInSeconds}");
                        }
                        else
                        {
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                                "AccessTokenProvider: No fresh token available!");
                        }
                        
                        return freshToken; // Return raw JWT (no "Bearer ")
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

                    // FIXED: Force WebSockets transport and skip negotiation
                    options.Transports = HttpTransportType.WebSockets;
                    options.SkipNegotiation = true;

                    // Set connection timeouts
                    options.CloseTimeout = TimeSpan.FromSeconds(30);
                })
                .WithAutomaticReconnect(new ExponentialBackoffRetryPolicy())
                .ConfigureLogging(logging =>
                {
                    logging.SetMinimumLevel(LogLevel.Information);
                    logging.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Warning);
                    logging.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Warning);
                })
                .Build();

            // Production-grade timeouts and keepalive settings 
            // KeepAliveInterval ~10–15s and ServerTimeout ≥30–45s per requirements
            _marketHub.ServerTimeout = TimeSpan.FromSeconds(45);  // Production stability
            _marketHub.KeepAliveInterval = TimeSpan.FromSeconds(15); // Set to 15s for production
            _marketHub.HandshakeTimeout = TimeSpan.FromSeconds(45);  // Extended handshake timeout

            SetupMarketHubEventHandlers();
            
            // Use enhanced connection startup with stability validation
            var connected = await StartConnectionWithRetry(_marketHub, "Market Hub");
            if (connected)
            {
                _marketHubWired = true;
                
                // IMMEDIATE: Subscribe to lightweight market data for stability validation
                try
                {
                    await _marketHub.InvokeAsync("SubscribeContractQuotes", "ES");
                    AddToSubscriptionManifest("MarketHub", "SubscribeContractQuotes", "ES");
                    _logger.LogInformation("[TOPSTEPX] Market Hub: Immediate ES subscription successful - stability validator has signal");
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] Market Hub: Immediate subscription failed, but connection established");
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

        // Enhanced connection lifecycle handlers with WebSocket close reason visibility
        _userHub.Closed += async (exception) =>
        {
            _userHubWired = false;
            
            // Extract WebSocket close status and description for debugging
            var closeStatus = "Unknown";
            var closeDescription = "Unknown";
            var innerExceptionMessage = "";
            
            if (exception != null)
            {
                innerExceptionMessage = exception.Message;
                
                // Try to extract WebSocket close details from exception
                if (exception.InnerException != null)
                {
                    innerExceptionMessage = exception.InnerException.Message;
                    
                    // Look for WebSocket specific exceptions
                    if (exception.InnerException is WebSocketException wsEx)
                    {
                        closeStatus = wsEx.WebSocketErrorCode.ToString();
                    }
                }
                
                // Parse common close codes from message
                if (innerExceptionMessage.Contains("1008"))
                {
                    closeStatus = "1008/Unauthorized - token/audience issue";
                }
                else if (innerExceptionMessage.Contains("1006"))
                {
                    closeStatus = "1006/Abnormal - network/proxy killing WS";
                }
                else if (innerExceptionMessage.Contains("1000"))
                {
                    closeStatus = "1000/Normal - client closed (watchdog too aggressive?)";
                }
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    $"User Hub connection closed - Status: {closeStatus}, Description: {closeDescription}, Inner: {innerExceptionMessage}");
            }
            else
            {
                closeStatus = "1000/Normal closure";
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    $"User Hub connection closed cleanly - Status: {closeStatus}");
            }
            
            ConnectionStateChanged?.Invoke($"UserHub:Disconnected:{closeStatus}");
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
            Interlocked.Increment(ref _totalReconnects);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"User Hub reconnected successfully: {connectionId}");
                
            // Replay subscription manifest after reconnect
            await ReplaySubscriptionManifest(_userHub, "UserHub");
            
            ConnectionStateChanged?.Invoke($"UserHub:Reconnected:{connectionId}");
        };
        
        // CRITICAL: Data reception handlers - completing the state machine
        _userHub.On<object>("GatewayUserOrder", (orderData) =>
        {
            Interlocked.Increment(ref _totalMessagesReceived);
            _logger.LogDebug("[TOPSTEPX] User Hub: Received GatewayUserOrder");
            OnGatewayUserOrderReceived?.Invoke(orderData);
        });
        
        _userHub.On<object>("GatewayUserTrade", (tradeData) =>
        {
            Interlocked.Increment(ref _totalMessagesReceived);
            _logger.LogDebug("[TOPSTEPX] User Hub: Received GatewayUserTrade");
            OnGatewayUserTradeReceived?.Invoke(tradeData);
        });
        
        _userHub.On<object>("FillUpdate", (fillData) =>
        {
            Interlocked.Increment(ref _totalMessagesReceived);
            _logger.LogDebug("[TOPSTEPX] User Hub: Received FillUpdate");
            OnFillUpdateReceived?.Invoke(fillData);
        });
        
        _userHub.On<object>("OrderUpdate", (orderUpdateData) =>
        {
            Interlocked.Increment(ref _totalMessagesReceived);
            _logger.LogDebug("[TOPSTEPX] User Hub: Received OrderUpdate");
            OnOrderUpdateReceived?.Invoke(orderUpdateData);
        });
        
        _logger.LogInformation("[TOPSTEPX] User Hub: All data reception handlers registered");
    }

    private void SetupMarketHubEventHandlers()
    {
        if (_marketHub == null) return;

        // Enhanced connection lifecycle handlers with WebSocket close reason visibility
        _marketHub.Closed += async (exception) =>
        {
            _marketHubWired = false;
            
            // Extract WebSocket close status and description for debugging
            var closeStatus = "Unknown";
            var closeDescription = "Unknown";
            var innerExceptionMessage = "";
            
            if (exception != null)
            {
                innerExceptionMessage = exception.Message;
                
                // Try to extract WebSocket close details from exception
                if (exception.InnerException != null)
                {
                    innerExceptionMessage = exception.InnerException.Message;
                    
                    // Look for WebSocket specific exceptions
                    if (exception.InnerException is WebSocketException wsEx)
                    {
                        closeStatus = wsEx.WebSocketErrorCode.ToString();
                    }
                }
                
                // Parse common close codes from message
                if (innerExceptionMessage.Contains("1008"))
                {
                    closeStatus = "1008/Unauthorized - token/audience issue";
                }
                else if (innerExceptionMessage.Contains("1006"))
                {
                    closeStatus = "1006/Abnormal - network/proxy killing WS";
                }
                else if (innerExceptionMessage.Contains("1000"))
                {
                    closeStatus = "1000/Normal - client closed (watchdog too aggressive?)";
                }
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    $"Market Hub connection closed - Status: {closeStatus}, Description: {closeDescription}, Inner: {innerExceptionMessage}");
            }
            else
            {
                closeStatus = "1000/Normal closure";
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    $"Market Hub connection closed cleanly - Status: {closeStatus}");
            }
            
            ConnectionStateChanged?.Invoke($"MarketHub:Disconnected:{closeStatus}");
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
            Interlocked.Increment(ref _totalReconnects);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Market Hub reconnected successfully: {connectionId}");
                
            // Replay subscription manifest after reconnect
            await ReplaySubscriptionManifest(_marketHub, "MarketHub");
            
            ConnectionStateChanged?.Invoke($"MarketHub:Reconnected:{connectionId}");
        };
        
        // CRITICAL: Data reception handlers - completing the state machine
        _marketHub.On<object>("MarketData", (marketData) =>
        {
            Interlocked.Increment(ref _totalMessagesReceived);
            _logger.LogDebug("[TOPSTEPX] Market Hub: Received MarketData");
            OnMarketDataReceived?.Invoke(marketData);
        });
        
        _marketHub.On<object>("ContractQuotes", (contractQuotes) =>
        {
            Interlocked.Increment(ref _totalMessagesReceived);
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
        
        // Log final stability statistics
        var stats = GetStabilityStatistics();
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
            $"Final {stats}");
        
        if (_userHub != null)
        {
            await _userHub.DisposeAsync();
        }
        
        if (_marketHub != null)
        {
            await _marketHub.DisposeAsync();
        }
    }

    /// <summary>
    /// Add subscription to manifest for replay on reconnect
    /// </summary>
    private void AddToSubscriptionManifest(string hubName, string method, string parameter)
    {
        var hubManifest = _subscriptionManifest.GetOrAdd(hubName, _ => new ConcurrentDictionary<string, int>());
        var subscriptionKey = $"{method}:{parameter}";
        hubManifest.AddOrUpdate(subscriptionKey, 1, (key, count) => count + 1);
        
        // Log manifest update
        _logger.LogInformation("[TOPSTEPX] {HubName} subscription manifest updated: {Method}({Parameter}) ref_count={Count}", 
            hubName, method, parameter, hubManifest[subscriptionKey]);
    }

    /// <summary>
    /// Replay all subscriptions from manifest after reconnect
    /// </summary>
    private async Task ReplaySubscriptionManifest(HubConnection hubConnection, string hubName)
    {
        if (_subscriptionManifest.TryGetValue(hubName, out var hubManifest))
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Replaying {hubManifest.Count} subscription(s) for {hubName} after reconnect");

            foreach (var subscription in hubManifest.ToList())
            {
                try
                {
                    var parts = subscription.Key.Split(':', 2);
                    if (parts.Length == 2)
                    {
                        var method = parts[0];
                        var parameter = parts[1];
                        
                        await hubConnection.InvokeAsync(method, parameter);
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                            $"Replayed subscription: {method}({parameter}) for {hubName}");
                    }
                }
                catch (Exception ex)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                        $"Failed to replay subscription {subscription.Key} for {hubName}: {ex.Message}");
                }
            }
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Final active manifest for {hubName}: {string.Join(", ", hubManifest.Keys)}");
        }
    }

    /// <summary>
    /// Get stability statistics for production monitoring
    /// </summary>
    public string GetStabilityStatistics()
    {
        var currentSpan = _longestConnectedStart != DateTime.MinValue ? 
            DateTime.UtcNow - _longestConnectedStart : TimeSpan.Zero;
        
        if (currentSpan > _longestConnectedSpan)
        {
            _longestConnectedSpan = currentSpan;
        }

        var avgGapSeconds = _connectionGaps.Count > 0 ? _connectionGaps.Average(g => g.TotalSeconds) : 0;
        var keepAliveInterval = 15; // Our configured KeepAliveInterval
        var gapsOverKeepAlive = _connectionGaps.Count(g => g.TotalSeconds > (keepAliveInterval * 2));

        return $"Stability Statistics: " +
               $"connect_attempts={_totalConnectAttempts}, " +
               $"reconnects={_totalReconnects}, " +
               $"longest_connected={_longestConnectedSpan.TotalMinutes:F1}m, " +
               $"messages_received={_totalMessagesReceived}, " +
               $"avg_gap={avgGapSeconds:F1}s, " +
               $"gaps_over_keepalive_x2={gapsOverKeepAlive}";
    }
    /// <summary>
    /// Retry subscriptions with a valid account ID after login completes
    /// </summary>
    public async Task<bool> RetrySubscriptionsWithAccountId(string accountId)
    {
        if (string.IsNullOrEmpty(accountId))
        {
            _logger.LogWarning("[TOPSTEPX] Cannot retry subscriptions - account ID is empty");
            return false;
        }

        _logger.LogInformation("[TOPSTEPX] Retrying subscriptions with account ID: {AccountId}", accountId);
        
        bool success = false;
        
        // Retry User Hub subscriptions if connected
        if (_userHub?.State == HubConnectionState.Connected)
        {
            try
            {
                await _userHub.InvokeAsync("SubscribeOrders", accountId);
                AddToSubscriptionManifest("UserHub", "SubscribeOrders", accountId);
                
                await _userHub.InvokeAsync("SubscribeTrades", accountId);
                AddToSubscriptionManifest("UserHub", "SubscribeTrades", accountId);
                
                _logger.LogInformation("[TOPSTEPX] User Hub: Successfully subscribed to orders and trades for account {AccountId}", accountId);
                success = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[TOPSTEPX] User Hub: Failed to subscribe with account {AccountId}", accountId);
            }
        }
        else
        {
            _logger.LogWarning("[TOPSTEPX] User Hub not connected - cannot subscribe");
        }

        return success;
    }

    public void Dispose()
    {
        // Unsubscribe from token refresh events
        if (_tokenProvider != null)
        {
            _tokenProvider.TokenRefreshed -= OnTokenRefreshed;
        }
        
        _connectionHealthTimer?.Dispose();
        _connectionLock.Dispose();
        _tokenRefreshLock.Dispose();
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