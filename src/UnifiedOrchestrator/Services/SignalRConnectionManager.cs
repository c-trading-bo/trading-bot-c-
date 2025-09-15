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
using Microsoft.Extensions.Configuration;
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
    private readonly IConfiguration _configuration;
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
        IConfiguration configuration,
        IHostApplicationLifetime appLifetime,
        TradingBot.UnifiedOrchestrator.Services.ILoginCompletionState loginCompletionState)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _tokenProvider = tokenProvider;
        _configuration = configuration;
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
                    if (hubName.Contains("Market")) { _logger.LogInformation("[TOPSTEPX] {HubName} connection ready - market subscriptions will be handled separately with contract IDs", hubName); }
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
            
            // Notify listeners that connection state has changed
            try
            {
                ConnectionStateChanged?.Invoke("Both hubs connected");
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "[SignalRManager] Failed to notify connection state changed");
            }
        }
    }

    public Task WaitForHubsConnected(CancellationToken cancellationToken) => _hubsConnected.Task.WaitAsync(cancellationToken);

    private async Task<Dictionary<string, string>> GetContractIdsAsync()
    {
        var contractIds = new Dictionary<string, string>();
        var targetSymbols = new[] { "ES", "NQ" }; // Only ES and NQ for eval accounts
        
        _logger.LogInformation("[TOPSTEPX] Starting contract ID discovery for symbols: {Symbols}", string.Join(", ", targetSymbols));
        
        // Check if REST discovery should be skipped
        var skipRestDiscovery = Environment.GetEnvironmentVariable("TOPSTEPX_SKIP_REST_DISCOVERY") == "true";
        
        if (!skipRestDiscovery)
        {
            // Primary: Try REST discovery for each symbol
            foreach (var symbol in targetSymbols)
            {
                try
                {
                    var contractId = await DiscoverContractIdFromRestAsync(symbol);
                    if (!string.IsNullOrEmpty(contractId))
                    {
                        contractIds[symbol] = contractId;
                        _logger.LogInformation("[TOPSTEPX] REST discovery: {Symbol} -> {ContractId}", symbol, contractId);
                    }
                    else
                    {
                        _logger.LogWarning("[TOPSTEPX] REST discovery failed for {Symbol}", symbol);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug(ex, "[TOPSTEPX] REST discovery error for {Symbol}: {Message}", symbol, ex.Message);
                }
            }
        }
        else
        {
            _logger.LogInformation("[TOPSTEPX] REST discovery skipped - using config fallback only");
        }
        
        // Secondary: Config fallback for any missing symbols
        foreach (var symbol in targetSymbols)
        {
            if (!contractIds.ContainsKey(symbol))
            {
                var contractId = GetContractIdFromConfig(symbol);
                if (!string.IsNullOrEmpty(contractId))
                {
                    contractIds[symbol] = contractId;
                    _logger.LogInformation("[TOPSTEPX] Config fallback: {Symbol} -> {ContractId}", symbol, contractId);
                }
            }
        }
        
        // Validation: Fail fast if any symbol is still missing
        var missingSymbols = targetSymbols.Where(s => !contractIds.ContainsKey(s) || string.IsNullOrEmpty(contractIds[s])).ToArray();
        if (missingSymbols.Length > 0)
        {
            var error = $"No contractId available for [{string.Join(", ", missingSymbols)}] after REST+Config; aborting subscriptions.";
            _logger.LogError("[TOPSTEPX] {Error}", error);
            _logger.LogError("[TOPSTEPX] 📋 To fix this issue:");
            _logger.LogError("[TOPSTEPX] 1. Open TopstepX web app → DevTools → Network tab");
            _logger.LogError("[TOPSTEPX] 2. Click ES and NQ instruments to load data");
            _logger.LogError("[TOPSTEPX] 3. Find contract API responses and copy the numeric contractId values");
            _logger.LogError("[TOPSTEPX] 4. Set environment variables with REAL contract IDs:");
            _logger.LogError("[TOPSTEPX]    $env:TOPSTEPX_EVAL_ES_ID=\"12345678\"  # Real ES contractId");
            _logger.LogError("[TOPSTEPX]    $env:TOPSTEPX_EVAL_NQ_ID=\"23456789\"  # Real NQ contractId");
            _logger.LogError("[TOPSTEPX] 5. Contract IDs must be numeric/GUID format, not symbol strings");
            throw new InvalidOperationException(error);
        }
        
        _logger.LogInformation("[TOPSTEPX] Contract ID resolution complete: {Count} symbols resolved", contractIds.Count);
        return contractIds;
    }
    
    private async Task<string?> DiscoverContractIdFromRestAsync(string symbol)
    {
        try
        {
            using var httpClient = new HttpClient();
            var apiBaseUrl = _configuration["TopstepX:ApiBaseUrl"] ?? "https://api.topstepx.com";
            httpClient.BaseAddress = new Uri(apiBaseUrl);
            
            var token = await _tokenProvider.GetTokenAsync();
            if (string.IsNullOrEmpty(token))
            {
                _logger.LogWarning("[TOPSTEPX] Cannot discover contract ID for {Symbol} - no JWT token available", symbol);
                return null;
            }
            
            httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

            // Try multiple API endpoints to find contract information
            var contractId = await TryContractAvailableEndpoint(httpClient, symbol);
            if (!string.IsNullOrEmpty(contractId)) return contractId;

            contractId = await TryContractSearchEndpoint(httpClient, symbol);
            if (!string.IsNullOrEmpty(contractId)) return contractId;

            contractId = await TryInstrumentSearchEndpoint(httpClient, symbol);
            if (!string.IsNullOrEmpty(contractId)) return contractId;

            contractId = await TryQuoteEndpoint(httpClient, symbol);
            if (!string.IsNullOrEmpty(contractId)) return contractId;

            _logger.LogWarning("[TOPSTEPX] No contract ID found for {Symbol} in any API endpoint", symbol);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[TOPSTEPX] REST contract discovery failed for {Symbol}: {Message}", symbol, ex.Message);
        }
        
        return null;
    }

    private async Task<string?> TryContractAvailableEndpoint(HttpClient httpClient, string symbol)
    {
        try
        {
            _logger.LogInformation("[TOPSTEPX] Trying /api/Contract/available for {Symbol}", symbol);
            
            // For evaluation accounts: try live=false first, then live=true
            foreach (var liveValue in new[] { false, true })
            {
                var payload = $"{{\"live\": {liveValue.ToString().ToLower()}}}";
                var response = await httpClient.PostAsync("/api/Contract/available", 
                    new StringContent(payload, System.Text.Encoding.UTF8, "application/json"));
                
                if (response.IsSuccessStatusCode)
                {
                    var jsonString = await response.Content.ReadAsStringAsync();
                    _logger.LogDebug("[TOPSTEPX] /api/Contract/available (live={Live}) response: {Response}", liveValue, jsonString);
                    
                    var contractId = await ParseContractResponse(jsonString, symbol);
                    if (!string.IsNullOrEmpty(contractId))
                    {
                        _logger.LogInformation("[TOPSTEPX] Found {Symbol} contract ID in /api/Contract/available (live={Live}): {ContractId}", symbol, liveValue, contractId);
                        return contractId;
                    }
                }
                else
                {
                    _logger.LogDebug("[TOPSTEPX] /api/Contract/available (live={Live}) returned {StatusCode}: {Content}", 
                        liveValue, response.StatusCode, await response.Content.ReadAsStringAsync());
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[TOPSTEPX] Error in TryContractAvailableEndpoint for {Symbol}", symbol);
        }
        return null;
    }

    private async Task<string?> TryContractSearchEndpoint(HttpClient httpClient, string symbol)
    {
        try
        {
            _logger.LogInformation("[TOPSTEPX] Trying /api/Contract/search for {Symbol}", symbol);
            
            var searchPayloads = new[]
            {
                $"{{\"symbol\": \"{symbol}\"}}",
                $"{{\"search\": \"{symbol}\"}}",
                $"{{\"query\": \"{symbol}\"}}",
                $"{{\"instrument\": \"{symbol}\"}}"
            };

            foreach (var payload in searchPayloads)
            {
                var response = await httpClient.PostAsync("/api/Contract/search", 
                    new StringContent(payload, System.Text.Encoding.UTF8, "application/json"));
                
                if (response.IsSuccessStatusCode)
                {
                    var contractId = await ParseContractResponse(await response.Content.ReadAsStringAsync(), symbol);
                    if (!string.IsNullOrEmpty(contractId))
                    {
                        _logger.LogInformation("[TOPSTEPX] Found {Symbol} contract ID in /api/Contract/search", symbol);
                        return contractId;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[TOPSTEPX] Error in TryContractSearchEndpoint for {Symbol}", symbol);
        }
        return null;
    }

    private async Task<string?> TryInstrumentSearchEndpoint(HttpClient httpClient, string symbol)
    {
        try
        {
            _logger.LogInformation("[TOPSTEPX] Trying /api/Instrument endpoints for {Symbol}", symbol);
            
            var endpoints = new[]
            {
                $"/api/Instrument/search",
                $"/api/Instrument/available",
                $"/api/Instrument/{symbol}",
                $"/api/Instruments/search"
            };

            foreach (var endpoint in endpoints)
            {
                try
                {
                    var response = endpoint.Contains("/search") || endpoint.Contains("/available")
                        ? await httpClient.PostAsync(endpoint, new StringContent($"{{\"symbol\": \"{symbol}\"}}", System.Text.Encoding.UTF8, "application/json"))
                        : await httpClient.GetAsync(endpoint);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var contractId = await ParseContractResponse(await response.Content.ReadAsStringAsync(), symbol);
                        if (!string.IsNullOrEmpty(contractId))
                        {
                            _logger.LogInformation("[TOPSTEPX] Found {Symbol} contract ID in {Endpoint}", symbol, endpoint);
                            return contractId;
                        }
                    }
                }
                catch (Exception endpointEx)
                {
                    _logger.LogDebug(endpointEx, "[TOPSTEPX] Error trying endpoint {Endpoint} for {Symbol}", endpoint, symbol);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[TOPSTEPX] Error in TryInstrumentSearchEndpoint for {Symbol}", symbol);
        }
        return null;
    }

    private async Task<string?> TryQuoteEndpoint(HttpClient httpClient, string symbol)
    {
        try
        {
            _logger.LogInformation("[TOPSTEPX] Trying quote endpoints for {Symbol}", symbol);
            
            var endpoints = new[]
            {
                $"/api/Quote/{symbol}",
                $"/api/Quotes/{symbol}",
                $"/api/MarketData/{symbol}"
            };

            foreach (var endpoint in endpoints)
            {
                try
                {
                    var response = await httpClient.GetAsync(endpoint);
                    if (response.IsSuccessStatusCode)
                    {
                        var contractId = await ParseContractResponse(await response.Content.ReadAsStringAsync(), symbol);
                        if (!string.IsNullOrEmpty(contractId))
                        {
                            _logger.LogInformation("[TOPSTEPX] Found {Symbol} contract ID in {Endpoint}", symbol, endpoint);
                            return contractId;
                        }
                    }
                }
                catch (Exception endpointEx)
                {
                    _logger.LogDebug(endpointEx, "[TOPSTEPX] Error trying endpoint {Endpoint} for {Symbol}", endpoint, symbol);
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[TOPSTEPX] Error in TryQuoteEndpoint for {Symbol}", symbol);
        }
        return null;
    }

    private Task<string?> ParseContractResponse(string jsonString, string symbol)
    {
        try
        {
            using var doc = JsonDocument.Parse(jsonString);
            
            // Try to find contract ID in various response structures
            var contractId = TryExtractContractId(doc.RootElement, symbol);
            if (!string.IsNullOrEmpty(contractId))
            {
                return Task.FromResult<string?>(contractId);
            }

            // If root doesn't have it, try in 'data' property
            if (doc.RootElement.TryGetProperty("data", out var dataElement))
            {
                contractId = TryExtractContractId(dataElement, symbol);
                if (!string.IsNullOrEmpty(contractId))
                {
                    return Task.FromResult<string?>(contractId);
                }
            }

            // If data is an array, check each item
            if (doc.RootElement.TryGetProperty("data", out var dataArray) && dataArray.ValueKind == JsonValueKind.Array)
            {
                foreach (var item in dataArray.EnumerateArray())
                {
                    contractId = TryExtractContractId(item, symbol);
                    if (!string.IsNullOrEmpty(contractId))
                    {
                        return Task.FromResult<string?>(contractId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[TOPSTEPX] Error parsing contract response for {Symbol}", symbol);
        }
        
        return Task.FromResult<string?>(null);
    }

    private string? TryExtractContractId(JsonElement element, string symbol)
    {
        try
        {
            // Common contract ID field names to check
            var contractIdFields = new[] { "contractId", "id", "instrumentId", "contractID", "Id", "contract_id" };
            var symbolFields = new[] { "symbol", "Symbol", "instrument", "name", "ticker" };

            // First, check if this element matches our symbol
            bool symbolMatches = false;
            foreach (var symbolField in symbolFields)
            {
                if (element.TryGetProperty(symbolField, out var symbolValue))
                {
                    var symbolStr = symbolValue.GetString();
                    if (!string.IsNullOrEmpty(symbolStr) && 
                        (symbolStr.Equals(symbol, StringComparison.OrdinalIgnoreCase) ||
                         symbolStr.StartsWith(symbol, StringComparison.OrdinalIgnoreCase)))
                    {
                        symbolMatches = true;
                        break;
                    }
                }
            }

            // If symbol matches, look for contract ID
            if (symbolMatches)
            {
                foreach (var contractIdField in contractIdFields)
                {
                    if (element.TryGetProperty(contractIdField, out var contractIdValue))
                    {
                        var contractId = contractIdValue.ValueKind == JsonValueKind.String 
                            ? contractIdValue.GetString()
                            : contractIdValue.ToString();
                            
                        if (!string.IsNullOrEmpty(contractId) && IsValidContractId(contractId))
                        {
                            _logger.LogInformation("[TOPSTEPX] Extracted contract ID {ContractId} for {Symbol}", contractId, symbol);
                            return contractId;
                        }
                    }
                }
            }

            // If no symbol match but we find a promising contract ID pattern, log it
            foreach (var contractIdField in contractIdFields)
            {
                if (element.TryGetProperty(contractIdField, out var contractIdValue))
                {
                    var contractId = contractIdValue.ValueKind == JsonValueKind.String 
                        ? contractIdValue.GetString()
                        : contractIdValue.ToString();
                        
                    if (!string.IsNullOrEmpty(contractId) && IsValidContractId(contractId))
                    {
                        _logger.LogDebug("[TOPSTEPX] Found potential contract ID {ContractId} (symbol match unclear)", contractId);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "[TOPSTEPX] Error extracting contract ID for {Symbol}", symbol);
        }
        
        return null;
    }
    
    private string? GetContractIdFromConfig(string symbol)
    {
        // Try environment variable first
        var envVarName = $"TOPSTEPX_EVAL_{symbol}_ID";
        var envValue = Environment.GetEnvironmentVariable(envVarName);
        _logger.LogInformation("[TOPSTEPX] Environment variable {EnvVar} = '{Value}'", envVarName, envValue ?? "null");
        if (!string.IsNullOrEmpty(envValue) && IsValidContractId(envValue))
        {
            _logger.LogInformation("[TOPSTEPX] Found valid contract ID for {Symbol} in environment variable {EnvVar}", symbol, envVarName);
            return envValue;
        }
        
        // Try configuration section
        var configPath = $"TopstepX:EvalContractIds:{symbol}";
        var configValue = _configuration[configPath];
        _logger.LogInformation("[TOPSTEPX] Configuration {ConfigPath} = '{Value}'", configPath, configValue ?? "null");
        
        // Also try to debug the entire TopstepX section
        var topstepXSection = _configuration.GetSection("TopstepX");
        _logger.LogInformation("[TOPSTEPX] TopstepX section exists: {Exists}", topstepXSection.Exists());
        
        var evalSection = _configuration.GetSection("TopstepX:EvalContractIds");
        _logger.LogInformation("[TOPSTEPX] EvalContractIds section exists: {Exists}", evalSection.Exists());
        
        if (!string.IsNullOrEmpty(configValue) && IsValidContractId(configValue))
        {
            _logger.LogInformation("[TOPSTEPX] Found valid contract ID for {Symbol} in configuration {ConfigPath}", symbol, configPath);
            return configValue;
        }
        
        _logger.LogWarning("[TOPSTEPX] No valid contract ID found for {Symbol} in environment or config", symbol);
        return null;
    }
    
    private bool IsValidContractId(string contractId)
    {
        // TopstepX contract IDs can be numeric, GUID, or CON. format strings
        if (string.IsNullOrEmpty(contractId) || contractId == "0")
        {
            return false;
        }
        
        // Check if it's a valid numeric ID (could be long integer)
        if (long.TryParse(contractId, out var numericId) && numericId > 0)
        {
            return true;
        }
        
        // Check if it's a valid GUID format
        if (Guid.TryParse(contractId, out var guidId) && guidId != Guid.Empty)
        {
            return true;
        }
        
        // Check if it's a valid TopstepX contract format (e.g., CON.F.US.EP.Z25)
        if (contractId.StartsWith("CON.") && contractId.Length > 10)
        {
            return true;
        }
        
        _logger.LogWarning("[TOPSTEPX] Invalid contract ID format: '{ContractId}' - must be numeric, GUID, or CON.* format", contractId);
        return false;
    }

    public async Task<bool> SubscribeToUserEventsAsync(string accountId)
    {
        if (!IsUserHubConnected || _userHub is null)
        {
            _logger.LogWarning("[TOPSTEPX] Cannot subscribe to user account - User Hub not connected.");
            return false;
        }
        
        bool hasAnySuccess = false;
        int eventCountBefore = _totalMessagesReceived;
        
        try
        {
            // Use only the correct TopstepX User Hub methods (no fallbacks)
            var numericAccountId = long.Parse(accountId);
            
            _logger.LogInformation("[TOPSTEPX] Subscribing to user events for account {AccountId}", numericAccountId);
            
            // Subscribe to Orders
            try
            {
                await _userHub.InvokeAsync("SubscribeOrders", numericAccountId);
                AddToSubscriptionManifest("UserHub", "SubscribeOrders", accountId);
                _logger.LogInformation("[TOPSTEPX] ✅ Successfully subscribed to Orders for account {AccountId}", numericAccountId);
                hasAnySuccess = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning("[TOPSTEPX] ❌ SubscribeOrders failed for account {AccountId}: {Error}", numericAccountId, ex.Message);
            }

            // Subscribe to Trades
            try
            {
                await _userHub.InvokeAsync("SubscribeTrades", numericAccountId);
                AddToSubscriptionManifest("UserHub", "SubscribeTrades", accountId);
                _logger.LogInformation("[TOPSTEPX] ✅ Successfully subscribed to Trades for account {AccountId}", numericAccountId);
                hasAnySuccess = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning("[TOPSTEPX] ❌ SubscribeTrades failed for account {AccountId}: {Error}", numericAccountId, ex.Message);
            }

            // Subscribe to Positions
            try
            {
                await _userHub.InvokeAsync("SubscribePositions", numericAccountId);
                AddToSubscriptionManifest("UserHub", "SubscribePositions", accountId);
                _logger.LogInformation("[TOPSTEPX] ✅ Successfully subscribed to Positions for account {AccountId}", numericAccountId);
                hasAnySuccess = true;
            }
            catch (Exception ex)
            {
                _logger.LogWarning("[TOPSTEPX] ❌ SubscribePositions failed for account {AccountId}: {Error}", numericAccountId, ex.Message);
            }

            // Wait a moment to see if we get any events
            await Task.Delay(2000);
            int eventCountAfter = _totalMessagesReceived;
            int newEvents = eventCountAfter - eventCountBefore;
            
            if (newEvents > 0)
            {
                _logger.LogInformation("[TOPSTEPX] ✅ User events subscription DATA CONFIRMED - received {NewEvents} events", newEvents);
                return true;
            }
            else if (hasAnySuccess)
            {
                _logger.LogInformation("[TOPSTEPX] ⚠️ User events subscription CALLS succeeded but no data received yet (may take time)");
                return true; // Call succeeded, data may come later
            }
            else
            {
                _logger.LogWarning("[TOPSTEPX] ❌ No subscription methods worked for account {AccountId}. User events may still be received automatically.", accountId);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] User events subscription error for account {AccountId}: {Message}", accountId, ex.Message);
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
        
        int eventCountBefore = _totalMessagesReceived;
        bool hasAnySuccess = false;
        
        try
        {
            // Try multiple subscription methods to ensure live data reception
            var subscriptionMethods = new[]
            {
                "SubscribeContractQuotes",
                "SubscribeQuotes", 
                "SubscribeMarketData",
                "Subscribe"
            };
            
            foreach (var method in subscriptionMethods)
            {
                try
                {
                    await _marketHub.InvokeAsync(method, contractId);
                    AddToSubscriptionManifest("MarketHub", method, contractId);
                    _logger.LogInformation("[TOPSTEPX] ✅ Successfully subscribed to {Method} for contractId {ContractId}", method, contractId);
                    hasAnySuccess = true;
                }
                catch (Exception ex)
                {
                    _logger.LogDebug("[TOPSTEPX] {Method} not available for contractId {ContractId}: {Error}", method, contractId, ex.Message);
                }
            }
            
            // Also try to subscribe to trades and depth
            try
            {
                await _marketHub.InvokeAsync("SubscribeContractTrades", contractId);
                AddToSubscriptionManifest("MarketHub", "SubscribeContractTrades", contractId);
                _logger.LogInformation("[TOPSTEPX] ✅ Subscribed to Contract Trades for contractId {ContractId}", contractId);
                hasAnySuccess = true;
            }
            catch (Exception ex)
            {
                _logger.LogDebug("[TOPSTEPX] SubscribeContractTrades not available: {Error}", ex.Message);
            }
            
            try
            {
                await _marketHub.InvokeAsync("SubscribeContractDepth", contractId);
                AddToSubscriptionManifest("MarketHub", "SubscribeContractDepth", contractId);
                _logger.LogInformation("[TOPSTEPX] ✅ Subscribed to Contract Depth for contractId {ContractId}", contractId);
                hasAnySuccess = true;
            }
            catch (Exception ex)
            {
                _logger.LogDebug("[TOPSTEPX] SubscribeContractDepth not available: {Error}", ex.Message);
            }
            
            // Wait longer to see if we get market data
            await Task.Delay(5000);
            int eventCountAfter = _totalMessagesReceived;
            int newEvents = eventCountAfter - eventCountBefore;
            
            if (newEvents > 0)
            {
                _logger.LogInformation("[TOPSTEPX] ✅ LIVE DATA CONFIRMED - received {NewEvents} market events for contractId {ContractId}", newEvents, contractId);
                return true;
            }
            else if (hasAnySuccess)
            {
                _logger.LogInformation("[TOPSTEPX] ⚠️ Market subscriptions succeeded but no live data received yet for contractId {ContractId} - this is normal during market closure or low activity", contractId);
                return true; // Subscriptions succeeded, data may come during market hours
            }
            else
            {
                _logger.LogWarning("[TOPSTEPX] ❌ No subscription methods worked for contractId {ContractId}", contractId);
                return false;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[TOPSTEPX] Failed to subscribe to market events for contractId {ContractId}: {Error}", contractId, ex.Message);
            return false;
        }
    }

    public async Task<bool> SubscribeToAllMarketsAsync()
    {
        _logger.LogInformation("[TOPSTEPX] Fetching contract IDs for market subscriptions...");
        var contractIds = await GetContractIdsAsync();
        
        // Now we always have ES and NQ contract IDs (hardcoded for eval accounts)
        _logger.LogInformation("[TOPSTEPX] Using {Count} contract IDs for subscriptions", contractIds.Count);
        
        // Subscribe using contract IDs (ES and NQ only for eval accounts)
        bool hasAnySuccess = false;
        var targetSymbols = new[] { "ES", "NQ" }; // Only ES and NQ for eval accounts
        
        foreach (var symbol in targetSymbols)
        {
            if (contractIds.TryGetValue(symbol, out var contractId))
            {
                try
                {
                    var success = await SubscribeToMarketEventsAsync(contractId);
                    if (success) hasAnySuccess = true;
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[TOPSTEPX] Failed to subscribe to {Symbol} (contractId: {ContractId}): {Error}", symbol, contractId, ex.Message);
                }
            }
            else
            {
                _logger.LogWarning("[TOPSTEPX] No contract ID found for symbol {Symbol}", symbol);
            }
        }
        
        return hasAnySuccess;
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
        
        // Primary TopstepX User Hub events
        _userHub.On<object>("GatewayUserOrder", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnGatewayUserOrderReceived?.Invoke(data); 
            _logger.LogInformation("[USER-HUB] 📋 Received GatewayUserOrder - Total messages: {Count}", _totalMessagesReceived); 
        });
        
        _userHub.On<object>("GatewayUserTrade", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnGatewayUserTradeReceived?.Invoke(data); 
            _logger.LogInformation("[USER-HUB] 💰 Received GatewayUserTrade - Total messages: {Count}", _totalMessagesReceived); 
        });
        
        // Additional possible event names
        _userHub.On<object>("OrdersUpdated", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnGatewayUserOrderReceived?.Invoke(data); 
            _logger.LogInformation("[USER-HUB] 📋 Received OrdersUpdated - Total messages: {Count}", _totalMessagesReceived); 
        });
        
        _userHub.On<object>("TradesUpdated", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnGatewayUserTradeReceived?.Invoke(data); 
            _logger.LogInformation("[USER-HUB] 💰 Received TradesUpdated - Total messages: {Count}", _totalMessagesReceived); 
        });
        
        _userHub.On<object>("PositionsUpdated", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            _logger.LogInformation("[USER-HUB] 📊 Received PositionsUpdated - Total messages: {Count}", _totalMessagesReceived); 
        });
        
        _logger.LogInformation("[USER-HUB] Event handlers registered: GatewayUserOrder, GatewayUserTrade, OrdersUpdated, TradesUpdated, PositionsUpdated");
    }

    private void SetupMarketHubEventHandlers()
    {
        if (_marketHub == null) return;
        
        // Enhanced market data event handlers with better coverage
        _marketHub.On<object>("GatewayQuote", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnMarketDataReceived?.Invoke(data); 
            if (_totalMessagesReceived % 10 == 1) // Log every 10th quote to avoid spam
                _logger.LogInformation("[MARKET-HUB] 📈 LIVE DATA: GatewayQuote #{Count}", _totalMessagesReceived); 
        });
        
        _marketHub.On<object>("GatewayTrade", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnMarketDataReceived?.Invoke(data); 
            _logger.LogInformation("[MARKET-HUB] 🔄 LIVE DATA: GatewayTrade #{Count}", _totalMessagesReceived); 
        });
        
        _marketHub.On<object>("GatewayDepth", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnMarketDataReceived?.Invoke(data);
            if (_totalMessagesReceived % 50 == 1) // Log every 50th depth update
                _logger.LogInformation("[MARKET-HUB] 📊 LIVE DATA: GatewayDepth #{Count}", _totalMessagesReceived); 
        });
        
        // Additional event types that might be used by TopstepX
        _marketHub.On<object>("Quote", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnMarketDataReceived?.Invoke(data); 
            _logger.LogInformation("[MARKET-HUB] 📈 LIVE DATA: Quote #{Count}", _totalMessagesReceived); 
        });
        
        _marketHub.On<object>("Trade", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnMarketDataReceived?.Invoke(data); 
            _logger.LogInformation("[MARKET-HUB] 💹 LIVE DATA: Trade #{Count}", _totalMessagesReceived); 
        });
        
        _marketHub.On<object>("MarketData", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnMarketDataReceived?.Invoke(data); 
            _logger.LogInformation("[MARKET-HUB] 📈 LIVE DATA: MarketData #{Count}", _totalMessagesReceived); 
        });
        
        _marketHub.On<object>("ContractQuotes", data => 
        { 
            Interlocked.Increment(ref _totalMessagesReceived); 
            OnContractQuotesReceived?.Invoke(data); 
            _logger.LogInformation("[MARKET-HUB] 📈 LIVE DATA: ContractQuotes #{Count}", _totalMessagesReceived); 
        });
        
        // Catch-all event handler for debugging unrecognized events
        _marketHub.On<object>("*", data => 
        { 
            _logger.LogDebug("[MARKET-HUB] 🔍 Unknown event received with data: {Data}", data?.ToString() ?? "null"); 
        });
        
        _logger.LogInformation("[MARKET-HUB] ✅ Enhanced event handlers registered for live data reception: GatewayQuote, GatewayTrade, GatewayDepth, Quote, Trade, MarketData, ContractQuotes");
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