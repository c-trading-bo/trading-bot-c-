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

public class ExponentialBackoffRetryPolicy : IRetryPolicy
{
    public TimeSpan? NextRetryDelay(RetryContext retryContext)
    {
        // More conservative retry policy to avoid overwhelming the server
        if (retryContext.PreviousRetryCount >= 5) return null; // Increased max retries from 3 to 5
        var delaySeconds = retryContext.PreviousRetryCount switch 
        { 
            0 => 30,   // Reduced initial delay from 45s to 30s
            1 => 60,   // Reduced from 90s to 60s  
            2 => 120,  // Reduced from 180s to 120s
            3 => 240,  // Added intermediate step
            4 => 300,  // Final step at 5 minutes
            _ => 300 
        };
        return TimeSpan.FromSeconds(delaySeconds);
    }
}

public class SignalRConnectionManager : ISignalRConnectionManager, IHostedService, IDisposable
{
    private const string SignalRManagerLogSource = "SignalRManager";
    private readonly ILogger<SignalRConnectionManager> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly ITokenProvider _tokenProvider;
    private readonly SemaphoreSlim _userHubLock = new(1, 1);
    private readonly SemaphoreSlim _marketHubLock = new(1, 1);
    private readonly IHostApplicationLifetime _appLifetime;
    private readonly TradingBot.UnifiedOrchestrator.Services.ILoginCompletionState _loginCompletionState;
    private readonly TaskCompletionSource _hubsConnected = new();
    
    private HubConnection? _userHub;
    private HubConnection? _marketHub;
    private volatile bool _userHubWired = false;
    private volatile bool _marketHubWired = false;
    private readonly Timer _connectionHealthTimer;
    private readonly SemaphoreSlim _tokenRefreshLock = new(1, 1);
    private readonly ConcurrentDictionary<string, ConcurrentDictionary<string, int>> _subscriptionManifest = new();
    
    private volatile int _totalConnectAttempts = 0;
    private DateTime _longestConnectedStart = DateTime.MinValue;
    private volatile int _totalMessagesReceived = 0;
    private bool _disposed = false;

    public bool IsUserHubConnected => _userHub?.State == HubConnectionState.Connected && _userHubWired;
    public bool IsMarketHubConnected => _marketHub?.State == HubConnectionState.Connected && _marketHubWired;

    public event Action<string>? ConnectionStateChanged;
    public event Action<object>? OnMarketDataReceived;
    public event Action<object>? OnContractQuotesReceived;
    public event Action<object>? OnGatewayUserOrderReceived;
    public event Action<object>? OnGatewayUserTradeReceived;
#pragma warning disable CS0067 // The event is never used
    public event Action<object>? OnFillUpdateReceived;
    public event Action<object>? OnOrderUpdateReceived;
#pragma warning restore CS0067 // The event is never used

    public SignalRConnectionManager(
        ILogger<SignalRConnectionManager> logger,
        ITradingLogger tradingLogger,
        ITokenProvider tokenProvider,
        IHostApplicationLifetime appLifetime,
        TradingBot.UnifiedOrchestrator.Services.ILoginCompletionState loginCompletionState)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _tokenProvider = tokenProvider;
        _appLifetime = appLifetime;
        _loginCompletionState = loginCompletionState;
        _connectionHealthTimer = new Timer(CheckConnectionHealth, null, TimeSpan.FromSeconds(60), TimeSpan.FromSeconds(60)); // Increased from 30s to 60s to be less aggressive
        _tokenProvider.TokenRefreshed += (token) => _ = Task.Run(() => OnTokenRefreshed(token));
    }

    public Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[{Source}] Hosted service starting. Registering connection startup hook.", SignalRManagerLogSource);
        _appLifetime.ApplicationStarted.Register(() =>
        {
            _logger.LogInformation("[{Source}] Application started. Kicking off connection process.", SignalRManagerLogSource);
            _ = Task.Run(async () =>
            {
                try
                {
                    _logger.LogInformation("[{Source}] Background task started. Waiting for login completion.", SignalRManagerLogSource);
                    await _loginCompletionState.WaitForLoginCompletion();
                    _logger.LogInformation("[{Source}] Login completed. Proceeding with SignalR hub connections.", SignalRManagerLogSource);
                    
                    // Add detailed logging for each hub connection attempt
                    try
                    {
                        _logger.LogInformation("[{Source}] Starting User Hub connection attempt...", SignalRManagerLogSource);
                        var userHub = await GetUserHubConnectionAsync();
                        _logger.LogInformation("[{Source}] User Hub connection attempt completed. State: {State}, Connected: {Connected}", 
                            SignalRManagerLogSource, userHub?.State, IsUserHubConnected);
                    }
                    catch (Exception userEx)
                    {
                        _logger.LogError(userEx, "[{Source}] User Hub connection failed: {Message}", SignalRManagerLogSource, userEx.Message);
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"User Hub connection failed: {userEx.Message}");
                    }
                    
                    try
                    {
                        _logger.LogInformation("[{Source}] Starting Market Hub connection attempt...", SignalRManagerLogSource);
                        var marketHub = await GetMarketHubConnectionAsync();
                        _logger.LogInformation("[{Source}] Market Hub connection attempt completed. State: {State}, Connected: {Connected}", 
                            SignalRManagerLogSource, marketHub?.State, IsMarketHubConnected);
                    }
                    catch (Exception marketEx)
                    {
                        _logger.LogError(marketEx, "[{Source}] Market Hub connection failed: {Message}", SignalRManagerLogSource, marketEx.Message);
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"Market Hub connection failed: {marketEx.Message}");
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[{Source}] An error occurred during the background connection process.", SignalRManagerLogSource);
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"Background connection process failed: {ex.Message}");
                }
            }, cancellationToken);
        });
        return Task.CompletedTask;
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        _connectionHealthTimer.Change(Timeout.Infinite, 0);
        return Task.CompletedTask;
    }

    private async Task OnTokenRefreshed(string newToken)
    {
        if (await _tokenRefreshLock.WaitAsync(TimeSpan.FromSeconds(5)))
        {
            try
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Fresh token arrived - checking hub connection states for restart");
                var needsUserHubRestart = _userHub == null || _userHub.State != HubConnectionState.Connected || string.IsNullOrEmpty(_userHub.ConnectionId);
                var needsMarketHubRestart = _marketHub == null || _marketHub.State != HubConnectionState.Connected || string.IsNullOrEmpty(_marketHub.ConnectionId);

                if (needsUserHubRestart || needsMarketHubRestart)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"Restarting hubs due to fresh token - UserHub: {needsUserHubRestart}, MarketHub: {needsMarketHubRestart}");
                    if (needsUserHubRestart)
                    {
                        _ = Task.Run(async () => { try { await GetUserHubConnectionAsync(); await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "User Hub restarted successfully with fresh token"); } catch (Exception ex) { await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"User Hub restart failed: {ex.Message}"); } });
                    }
                    if (needsMarketHubRestart)
                    {
                        _ = Task.Run(async () => { try { await GetMarketHubConnectionAsync(); await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Market Hub restarted successfully with fresh token"); } catch (Exception ex) { await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"Market Hub restart failed: {ex.Message}"); } });
                    }
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, SignalRManagerLogSource, "Fresh token received but both hubs are already connected - no restart needed");
                }
            }
            catch (Exception ex)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"Error handling token refresh: {ex.Message}");
            }
            finally
            {
                _tokenRefreshLock.Release();
            }
        }
        else
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, "Token refresh handling skipped - already processing another refresh");
        }
    }

    private async Task<string?> WaitForJwtReadinessAsync(CancellationToken cancellationToken = default)
    {
        var startTime = DateTime.UtcNow;
        var timeout = TimeSpan.FromSeconds(45);
        var checkInterval = TimeSpan.FromSeconds(2);
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Waiting for JWT token readiness before establishing hub connections...");
        while (DateTime.UtcNow - startTime < timeout)
        {
            var token = await _tokenProvider.GetTokenAsync();
            if (!string.IsNullOrEmpty(token))
            {
                if (await ValidateJwtTokenAsync(token))
                {
                    var waitTime = DateTime.UtcNow - startTime;
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"JWT ready after {waitTime.TotalSeconds:F1} seconds - timing validation passed");
                    return token;
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, "JWT token found but validation failed (need exp-now ≥120s, nbf ≤ now-5s, valid aud/iss)");
                }
            }
            var elapsed = DateTime.UtcNow - startTime;
            if (elapsed.TotalSeconds > 10 && elapsed.TotalSeconds % 10 < 2)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"Still waiting for JWT token... {elapsed.TotalSeconds:F0}s elapsed");
            }
            await Task.Delay(checkInterval, cancellationToken);
        }
        await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"JWT token readiness timeout after {timeout.TotalSeconds} seconds");
        return null;
    }

    private async Task<bool> ValidateJwtTokenAsync(string token, string hubType = "")
    {
        try
        {
            if (token.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)) token = token.Substring(7);
            var parts = token.Split('.');
            if (parts.Length != 3)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, "JWT token format invalid - expected 3 parts separated by dots");
                return false;
            }
            try
            {
                var payloadBytes = Convert.FromBase64String(AddBase64Padding(parts[1]));
                var payloadJson = System.Text.Encoding.UTF8.GetString(payloadBytes);
                using var doc = JsonDocument.Parse(payloadJson);
                var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                if (doc.RootElement.TryGetProperty("iss", out var issElement)) await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, SignalRManagerLogSource, $"JWT issuer: {issElement.GetString()}");
                if (doc.RootElement.TryGetProperty("aud", out var audElement))
                {
                    var audience = audElement.GetString();
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, SignalRManagerLogSource, $"JWT audience: {audience}");
                    if (!string.IsNullOrEmpty(hubType) && (hubType.Contains("user", StringComparison.OrdinalIgnoreCase) || hubType.Contains("market", StringComparison.OrdinalIgnoreCase)) && !audience?.Contains("topstepx", StringComparison.OrdinalIgnoreCase) == true)
                    {
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, $"JWT audience mismatch for {hubType} - aud: {audience}");
                    }
                }
                if (doc.RootElement.TryGetProperty("exp", out var expElement))
                {
                    var exp = expElement.GetInt64();
                    var remainingSeconds = exp - now;
                    if (remainingSeconds < 120)
                    {
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, $"JWT token expires too soon - exp_in_seconds: {remainingSeconds} (need ≥120s)");
                        return false;
                    }
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"JWT token exp_in_seconds: {remainingSeconds}");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, "JWT token missing exp claim");
                    return false;
                }
                if (doc.RootElement.TryGetProperty("nbf", out var nbfElement))
                {
                    var nbf = nbfElement.GetInt64();
                    if (now < (nbf - 5))
                    {
                        var skew = nbf - now;
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, $"JWT token not yet valid - nbf timing issue, skew: {skew}s (check system clock)");
                        return false;
                    }
                }
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "JWT ok - aud, iss, exp, nbf validation passed");
                return true;
            }
            catch (Exception ex)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, $"JWT payload decode failed: {ex.Message}");
                return false;
            }
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"JWT validation error: {ex.Message}");
            return false;
        }
    }
    
    private static string AddBase64Padding(string base64)
    {
        switch (base64.Length % 4) { case 2: return base64 + "=="; case 3: return base64 + "="; default: return base64; }
    }

    private bool ValidateServerCertificate(HttpRequestMessage request, X509Certificate2? certificate, X509Chain? chain, SslPolicyErrors sslPolicyErrors)
    {
        var isDevelopment = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") == "Development";
        var bypassSsl = Environment.GetEnvironmentVariable("BYPASS_SSL_VALIDATION") == "true";
        if (isDevelopment || bypassSsl)
        {
            if (sslPolicyErrors != SslPolicyErrors.None) _logger.LogWarning("[TOPSTEPX] SSL validation bypassed. Errors: {Errors}", sslPolicyErrors);
            return true;
        }
        if (sslPolicyErrors == SslPolicyErrors.None) return true;
        _logger.LogWarning("[TOPSTEPX] SSL certificate validation failed: {Errors}", sslPolicyErrors);
        return true;
    }

    private async Task<bool> StartConnectionWithRetry(HubConnection hubConnection, string hubName)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"Network hygiene - {hubName} connecting to WebSockets transport");
        if (hubConnection.ConnectionId != null) await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"Hub URL: {(hubName.Contains("User") ? "https://rtc.topstepx.com/hubs/user" : "https://rtc.topstepx.com/hubs/market")}, Transport: WebSockets");

        try
        {
            if (hubConnection == null) throw new InvalidOperationException("Hub connection not initialized");
            Interlocked.Increment(ref _totalConnectAttempts);
            _logger.LogInformation("[TOPSTEPX] Starting {HubName} connection", hubName);
            
            // Check if connection is in a state that can be started
            if (hubConnection.State != HubConnectionState.Disconnected)
            {
                _logger.LogInformation("[TOPSTEPX] {HubName} connection is not disconnected (State: {State}), stopping first...", hubName, hubConnection.State);
                try
                {
                    await hubConnection.StopAsync();
                }
                catch (Exception stopEx)
                {
                    _logger.LogWarning(stopEx, "[TOPSTEPX] Error stopping {HubName} connection: {Message}", hubName, stopEx.Message);
                }
            }
            
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
            await hubConnection.StartAsync(cts.Token);
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "connection started");
            _logger.LogInformation("[TOPSTEPX] {HubName} connection started successfully. State: {State}", hubName, hubConnection.State);
            _logger.LogInformation("[TOPSTEPX] {HubName} waiting for ConnectionId assignment...", hubName);
            for (int wait = 0; wait < 10; wait++)
            {
                await Task.Delay(50);
                if (!string.IsNullOrEmpty(hubConnection.ConnectionId) && hubConnection.State == HubConnectionState.Connected)
                {
                    _logger.LogInformation("[TOPSTEPX] {HubName} ConnectionId ready: {ConnectionId}", hubName, hubConnection.ConnectionId);
                    break;
                }
                if (hubConnection.State != HubConnectionState.Connected)
                {
                    _logger.LogWarning("[TOPSTEPX] {HubName} connection lost during handshake - State: {State}", hubName, hubConnection.State);
                    throw new InvalidOperationException($"{hubName} connection lost during handshake");
                }
            }
            if (hubConnection.State == HubConnectionState.Connected)
            {
                try
                {
                    if (hubName.Contains("Market")) { await hubConnection.InvokeAsync("SubscribeContractQuotes", "ES"); _logger.LogInformation("[TOPSTEPX] {HubName} immediate ES subscription successful", hubName); }
                    else if (hubName.Contains("User")) { _logger.LogInformation("[TOPSTEPX] {HubName} ready for subscriptions when account ID available", hubName); }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] {HubName} immediate subscription failed: {Message}", hubName, ex.Message);
                }
            }
            
            // Updated validation logic: ConnectionId is null when SkipNegotiation=true with WebSockets
            // This is expected behavior, not a failure - we only need State == Connected
            if (hubConnection.State == HubConnectionState.Connected)
            {
                if (!string.IsNullOrEmpty(hubConnection.ConnectionId))
                {
                    _logger.LogInformation("[TOPSTEPX] {HubName} connection validated - State: {State}, ID: {ConnectionId}", hubName, hubConnection.State, hubConnection.ConnectionId);
                }
                else
                {
                    _logger.LogInformation("[TOPSTEPX] {HubName} connection validated - State: {State}, Transport: WebSockets, Negotiation: Skipped, ConnectionId: <n/a>", hubName, hubConnection.State);
                }
                _longestConnectedStart = DateTime.UtcNow;
                return true;
            }
            else
            {
                _logger.LogWarning("[TOPSTEPX] {HubName} connection validation failed - State: {State}, ID: {ConnectionId}", hubName, hubConnection.State, hubConnection.ConnectionId ?? "null");
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[TOPSTEPX] {HubName} connection attempt failed", hubName);
            return false;
        }
    }

    public async Task<HubConnection> GetUserHubConnectionAsync()
    {
        // Quick check without lock - if already connected, return immediately
        if (_userHub != null && IsUserHubConnected) 
        {
            _logger.LogDebug("[{Source}] User Hub already connected, returning existing connection", SignalRManagerLogSource);
            return _userHub;
        }
        
        _logger.LogInformation("[{Source}] Attempting to acquire connection lock for User Hub...", SignalRManagerLogSource);
        
        // Add timeout to prevent infinite deadlock - using dedicated User Hub lock
        var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        try
        {
            await _userHubLock.WaitAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            _logger.LogError("[{Source}] Connection lock acquisition cancelled for User Hub", SignalRManagerLogSource);
            throw new TimeoutException("Failed to acquire connection lock for User Hub within 2 minutes");
        }
        
        _logger.LogInformation("[{Source}] Connection lock acquired for User Hub", SignalRManagerLogSource);
        
        try
        {
            if (_userHub != null && IsUserHubConnected) return _userHub;
            _logger.LogInformation("[{Source}] Attempting to establish User Hub connection.", SignalRManagerLogSource);
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Attempting to establish User Hub connection.");
            
            _logger.LogInformation("[{Source}] Getting JWT token for User Hub...", SignalRManagerLogSource);
            var token = await WaitForJwtReadinessAsync();
            _logger.LogInformation("[{Source}] JWT token obtained: {TokenExists}", SignalRManagerLogSource, !string.IsNullOrEmpty(token));
            
            if (string.IsNullOrEmpty(token))
            {
                _logger.LogError("[{Source}] Cannot connect to User Hub: JWT token is not available or invalid.", SignalRManagerLogSource);
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, "Cannot connect to User Hub: JWT token is not available or invalid.");
                throw new InvalidOperationException("Cannot connect to User Hub: JWT token is not available or invalid.");
            }
            
            var userHubUrl = $"https://rtc.topstepx.com/hubs/user?access_token={token.Replace("Bearer ", "")}";
            _logger.LogInformation("[{Source}] User Hub URL created", SignalRManagerLogSource);
            await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, SignalRManagerLogSource, $"User Hub URL: {userHubUrl}");
            
            _logger.LogInformation("[{Source}] Disposing old User Hub connection...", SignalRManagerLogSource);
            await (_userHub?.DisposeAsync() ?? ValueTask.CompletedTask);
            
            _logger.LogInformation("[{Source}] Creating new User Hub connection...", SignalRManagerLogSource);
            _userHub = new HubConnectionBuilder()
                .WithUrl(userHubUrl, options =>
                {
                    options.SkipNegotiation = true;
                    options.Transports = HttpTransportType.WebSockets;
                    // NOTE: Using query string authentication for WebSocket transport per Microsoft best practices
                    // AccessTokenProvider removed to avoid conflicts with query string auth
                    options.HttpMessageHandlerFactory = (message) => { if (message is HttpClientHandler clientHandler) { clientHandler.ServerCertificateCustomValidationCallback = ValidateServerCertificate; } return message; };
                })
                .WithAutomaticReconnect(new ExponentialBackoffRetryPolicy())
                .ConfigureLogging(logging => { logging.SetMinimumLevel(LogLevel.Trace); })
                .Build();
            
            _logger.LogInformation("[{Source}] Configuring User Hub timeouts...", SignalRManagerLogSource);
            _userHub.ServerTimeout = TimeSpan.FromSeconds(120);  // Increased from 45s to 120s for stability
            _userHub.KeepAliveInterval = TimeSpan.FromSeconds(30);  // Increased from 15s to 30s to reduce network noise
            _userHub.HandshakeTimeout = TimeSpan.FromSeconds(60);  // Increased from 45s to 60s for more reliable handshakes
            
            // Add disconnection event handler for proper cleanup and state management
            _userHub.Closed += async (error) =>
            {
                _userHubWired = false;
                if (error != null)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, $"User Hub disconnected with error: {error.Message}");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "User Hub disconnected normally");
                }
                ConnectionStateChanged?.Invoke("UserHub:Disconnected");
            };
            
            // Add reconnecting event handler to track reconnection attempts
            _userHub.Reconnecting += async (error) =>
            {
                _userHubWired = false;
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"User Hub reconnecting due to: {error?.Message ?? "Unknown reason"}");
                ConnectionStateChanged?.Invoke("UserHub:Reconnecting");
            };
            
            // Add reconnected event handler to restore state
            _userHub.Reconnected += async (connectionId) =>
            {
                _userHubWired = true;
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"User Hub reconnected with ID: {connectionId}");
                ConnectionStateChanged?.Invoke("UserHub:Reconnected");
                CheckAndSetHubsConnected();
            };
            
            _logger.LogInformation("[{Source}] Setting up User Hub event handlers...", SignalRManagerLogSource);
            SetupUserHubEventHandlers();
            
            _logger.LogInformation("[{Source}] Calling StartConnectionWithRetry for User Hub...", SignalRManagerLogSource);
            var connected = await StartConnectionWithRetry(_userHub, "User Hub");
            _logger.LogInformation("[{Source}] StartConnectionWithRetry completed. Connected: {Connected}", SignalRManagerLogSource, connected);
            
            if (connected)
            {
                _userHubWired = true;
                _logger.LogInformation("[TOPSTEPX] User Hub connection established and confirmed ready");
                ConnectionStateChanged?.Invoke($"UserHub:Connected");
                CheckAndSetHubsConnected();
            }
            return _userHub;
        }
        finally { _userHubLock.Release(); }
    }

    public async Task<HubConnection> GetMarketHubConnectionAsync()
    {
        // Quick check without lock - if already connected, return immediately
        if (_marketHub != null && IsMarketHubConnected) 
        {
            _logger.LogDebug("[{Source}] Market Hub already connected, returning existing connection", SignalRManagerLogSource);
            return _marketHub;
        }
        
        _logger.LogInformation("[{Source}] Attempting to acquire connection lock for Market Hub...", SignalRManagerLogSource);
        
        // Add timeout to prevent infinite deadlock - using dedicated Market Hub lock
        var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        try
        {
            await _marketHubLock.WaitAsync(cts.Token);
        }
        catch (OperationCanceledException)
        {
            _logger.LogError("[{Source}] Connection lock acquisition cancelled for Market Hub", SignalRManagerLogSource);
            throw new TimeoutException("Failed to acquire connection lock for Market Hub within 2 minutes");
        }
        
        _logger.LogInformation("[{Source}] Connection lock acquired for Market Hub", SignalRManagerLogSource);
        
        try
        {
            if (_marketHub != null && IsMarketHubConnected) return _marketHub;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Attempting to establish Market Hub connection.");
            var token = await WaitForJwtReadinessAsync();
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, "Cannot connect to Market Hub: JWT token is not available or invalid.");
                throw new InvalidOperationException("Cannot connect to Market Hub: JWT token is not available or invalid.");
            }
            var marketHubUrl = $"https://rtc.topstepx.com/hubs/market?access_token={token.Replace("Bearer ", "")}";
            await _tradingLogger.LogSystemAsync(TradingLogLevel.DEBUG, SignalRManagerLogSource, $"Market Hub URL: {marketHubUrl}");
            await (_marketHub?.DisposeAsync() ?? ValueTask.CompletedTask);
            _marketHub = new HubConnectionBuilder()
                .WithUrl(marketHubUrl, options =>
                {
                    options.SkipNegotiation = true;
                    options.Transports = HttpTransportType.WebSockets;
                    // NOTE: Using query string authentication for WebSocket transport per Microsoft best practices
                    // AccessTokenProvider removed to avoid conflicts with query string auth
                    options.HttpMessageHandlerFactory = (message) => { if (message is HttpClientHandler clientHandler) { clientHandler.ServerCertificateCustomValidationCallback = ValidateServerCertificate; } return message; };
                })
                .WithAutomaticReconnect(new ExponentialBackoffRetryPolicy())
                .ConfigureLogging(logging => { logging.SetMinimumLevel(LogLevel.Trace); })
                .Build();
            _marketHub.ServerTimeout = TimeSpan.FromSeconds(120);  // Increased from 45s to 120s for stability
            _marketHub.KeepAliveInterval = TimeSpan.FromSeconds(30);  // Increased from 15s to 30s to reduce network noise
            _marketHub.HandshakeTimeout = TimeSpan.FromSeconds(60);  // Increased from 45s to 60s for more reliable handshakes
            
            // Add disconnection event handler for proper cleanup and state management
            _marketHub.Closed += async (error) =>
            {
                _marketHubWired = false;
                if (error != null)
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, $"Market Hub disconnected with error: {error.Message}");
                }
                else
                {
                    await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Market Hub disconnected normally");
                }
                ConnectionStateChanged?.Invoke("MarketHub:Disconnected");
            };
            
            // Add reconnecting event handler to track reconnection attempts
            _marketHub.Reconnecting += async (error) =>
            {
                _marketHubWired = false;
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"Market Hub reconnecting due to: {error?.Message ?? "Unknown reason"}");
                ConnectionStateChanged?.Invoke("MarketHub:Reconnecting");
            };
            
            // Add reconnected event handler to restore state
            _marketHub.Reconnected += async (connectionId) =>
            {
                _marketHubWired = true;
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, $"Market Hub reconnected with ID: {connectionId}");
                ConnectionStateChanged?.Invoke("MarketHub:Reconnected");
                CheckAndSetHubsConnected();
            };
            SetupMarketHubEventHandlers();
            var connected = await StartConnectionWithRetry(_marketHub, "Market Hub");
            if (connected)
            {
                _marketHubWired = true;
                _logger.LogInformation("[TOPSTEPX] Market Hub connection established and confirmed ready");
                ConnectionStateChanged?.Invoke($"MarketHub:Connected");
                CheckAndSetHubsConnected();
            }
            return _marketHub;
        }
        finally { _marketHubLock.Release(); }
    }

    private void CheckAndSetHubsConnected()
    {
        if (IsUserHubConnected && IsMarketHubConnected)
        {
            _logger.LogInformation("[SignalRManager] Both User and Market hubs are connected. Signaling completion.");
            _hubsConnected.TrySetResult();
        }
    }

    public Task WaitForHubsConnected(CancellationToken cancellationToken) => _hubsConnected.Task.WaitAsync(cancellationToken);

    public async Task<bool> SubscribeToUserEventsAsync(string accountId)
    {
        if (!IsUserHubConnected || _userHub is null)
        {
            _logger.LogWarning("[TOPSTEPX] Cannot subscribe to user account - User Hub not connected.");
            return false;
        }
        
        bool hasAnySuccess = false;
        
        try
        {
            // Try Market Hub-style naming patterns since SubscribeContractQuotes works
            var orderMethods = new[]
            {
                "SubscribeAccountOrders",
                "SubscribeUserOrders", 
                "SubscribeGatewayUserOrder",
                "SubscribeOrders",
                "SubscribeToOrders",
                "Subscribe"  // Simplified, just pass accountId
            };
            
            var tradeMethods = new[]
            {
                "SubscribeAccountTrades",
                "SubscribeUserTrades",
                "SubscribeGatewayUserTrade", 
                "SubscribeTrades",
                "SubscribeToTrades"
            };

            _logger.LogInformation("[TOPSTEPX] Testing {OrderCount} order subscription methods for account {AccountId}", orderMethods.Length, accountId);
            
            foreach (var method in orderMethods)
            {
                try
                {
                    await _userHub.InvokeAsync(method, accountId);
                    AddToSubscriptionManifest("UserHub", method, accountId);
                    _logger.LogInformation("[TOPSTEPX] ✅ Successfully subscribed to Orders using {Method} for account {AccountId}", method, accountId);
                    hasAnySuccess = true;
                    break; // Stop trying once we find a working method
                }
                catch (Exception ex)
                {
                    _logger.LogDebug("[TOPSTEPX] ❌ {Method} failed for account {AccountId}: {Error}", method, accountId, ex.Message);
                }
            }

            _logger.LogInformation("[TOPSTEPX] Testing {TradeCount} trade subscription methods for account {AccountId}", tradeMethods.Length, accountId);
            
            foreach (var method in tradeMethods)
            {
                try
                {
                    await _userHub.InvokeAsync(method, accountId);
                    AddToSubscriptionManifest("UserHub", method, accountId);
                    _logger.LogInformation("[TOPSTEPX] ✅ Successfully subscribed to Trades using {Method} for account {AccountId}", method, accountId);
                    hasAnySuccess = true;
                    break; // Stop trying once we find a working method
                }
                catch (Exception ex)
                {
                    _logger.LogDebug("[TOPSTEPX] ❌ {Method} failed for account {AccountId}: {Error}", method, accountId, ex.Message);
                }
            }

            if (hasAnySuccess)
            {
                _logger.LogInformation("[TOPSTEPX] ✅ User subscription successful - at least one method worked for account {AccountId}", accountId);
            }
            else
            {
                _logger.LogWarning("[TOPSTEPX] ⚠️ No subscription methods worked for account {AccountId}. User events may still be received automatically.", accountId);
            }
            
            _logger.LogInformation("[TOPSTEPX] User subscription process completed for account {AccountId}", accountId);
            return hasAnySuccess;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Failed to subscribe to user account {AccountId}", accountId);
            return false;
        }
    }

    public async Task<bool> SubscribeToMarketEventsAsync(string contractId)
    {
        if (!IsMarketHubConnected || _marketHub == null)
        {
            _logger.LogWarning("[TOPSTEPX] Cannot subscribe to market events - Market Hub not connected.");
            return false;
        }
        try
        {
            await _marketHub.InvokeAsync("SubscribeContractQuotes", contractId);
            AddToSubscriptionManifest("MarketHub", "SubscribeContractQuotes", contractId);
            _logger.LogInformation("[TOPSTEPX] Subscribed to Contract Quotes for {Symbol}", contractId);
            return true;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Failed to subscribe to market events for {Symbol}", contractId);
            return false;
        }
    }

    private void AddToSubscriptionManifest(string hub, string method, string parameter)
    {
        var hubSubscriptions = _subscriptionManifest.GetOrAdd(hub, new ConcurrentDictionary<string, int>());
        hubSubscriptions.AddOrUpdate(method, 1, (key, count) => count + 1);
        _logger.LogInformation("[SUB-MANIFEST] Added: {Hub} -> {Method}({Parameter})", hub, method, parameter);
    }

    private void SetupUserHubEventHandlers()
    {
        if (_userHub == null) return;
        _userHub.On<object>("GatewayUserOrder", data => { Interlocked.Increment(ref _totalMessagesReceived); OnGatewayUserOrderReceived?.Invoke(data); _logger.LogInformation("[USER-HUB] Received GatewayUserOrder"); });
        _userHub.On<object>("GatewayUserTrade", data => { Interlocked.Increment(ref _totalMessagesReceived); OnGatewayUserTradeReceived?.Invoke(data); _logger.LogInformation("[USER-HUB] Received GatewayUserTrade"); });
    }

    private void SetupMarketHubEventHandlers()
    {
        if (_marketHub == null) return;
        _marketHub.On<object>("MarketData", data => { Interlocked.Increment(ref _totalMessagesReceived); OnMarketDataReceived?.Invoke(data); });
        _marketHub.On<object>("ContractQuotes", data => { Interlocked.Increment(ref _totalMessagesReceived); OnContractQuotesReceived?.Invoke(data); });
    }

    public async Task<bool> RetrySubscriptionsWithAccountId(string accountId)
    {
        _logger.LogInformation("[TOPSTEPX] Retrying subscriptions with account ID: {AccountId}", accountId);
        if (!IsUserHubConnected)
        {
            _logger.LogWarning("[TOPSTEPX] User Hub not connected - cannot subscribe");
            return false;
        }
        return await SubscribeToUserEventsAsync(accountId);
    }

    private async void CheckConnectionHealth(object? state)
    {
        try
        {
            // Improved health check logic: only reconnect if actually disconnected for a reasonable time
            if (!IsUserHubConnected)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, "Health check detected User Hub disconnection - waiting 10s before reconnect attempt");
                
                // Wait a bit in case automatic reconnection is already in progress
                await Task.Delay(TimeSpan.FromSeconds(10));
                
                // Double-check if still disconnected after waiting
                if (!IsUserHubConnected)
                {
                    _logger.LogWarning("[HEALTH] User Hub is still disconnected after grace period. Attempting manual reconnect...");
                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await GetUserHubConnectionAsync();
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "User Hub manual reconnection successful");
                        }
                        catch (Exception ex)
                        {
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"User Hub manual reconnection failed: {ex.Message}");
                        }
                    });
                }
            }
            
            if (!IsMarketHubConnected)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, SignalRManagerLogSource, "Health check detected Market Hub disconnection - waiting 10s before reconnect attempt");
                
                // Wait a bit in case automatic reconnection is already in progress
                await Task.Delay(TimeSpan.FromSeconds(10));
                
                // Double-check if still disconnected after waiting
                if (!IsMarketHubConnected)
                {
                    _logger.LogWarning("[HEALTH] Market Hub is still disconnected after grace period. Attempting manual reconnect...");
                    _ = Task.Run(async () =>
                    {
                        try
                        {
                            await GetMarketHubConnectionAsync();
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, SignalRManagerLogSource, "Market Hub manual reconnection successful");
                        }
                        catch (Exception ex)
                        {
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"Market Hub manual reconnection failed: {ex.Message}");
                        }
                    });
                }
            }
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, SignalRManagerLogSource, $"Health check error: {ex.Message}");
        }
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (_disposed) return;
        if (disposing)
        {
            _connectionHealthTimer?.Dispose();
            _userHubLock?.Dispose();
            _marketHubLock?.Dispose();
            _tokenRefreshLock?.Dispose();
            _userHub?.DisposeAsync().AsTask().GetAwaiter().GetResult();
            _marketHub?.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
        _disposed = true;
    }
}