using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Production-ready SignalR hub connection manager with proper state management
/// Fixes hub connection instability and double subscription issues
/// </summary>
public interface ISignalRConnectionManager
{
    Task<HubConnection> GetUserHubConnectionAsync();
    Task<HubConnection> GetMarketHubConnectionAsync();
    bool IsUserHubConnected { get; }
    bool IsMarketHubConnected { get; }
    event Action<string> ConnectionStateChanged;
    
    // TopstepX specification compliant subscription methods
    Task<bool> SubscribeToUserEventsAsync(string accountId);
    Task<bool> SubscribeToMarketEventsAsync(string contractId);
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

            _userHub?.DisposeAsync();
            _userHub = BuildHub("https://rtc.topstepx.com/hubs/user");

            // Configure connection timeouts for production use
            _userHub.ServerTimeout = TimeSpan.FromSeconds(60);
            _userHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            _userHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            SetupUserHubEventHandlers();
            
            // Use enhanced connection startup with retry logic
            await StartAndSubscribeWithRetryAsync(
                _userHub, 
                "UserHub", 
                async () => {
                    // Empty subscription function - subscriptions are handled separately
                    await Task.CompletedTask;
                }, 
                CancellationToken.None);
            
            _userHubWired = true;

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                "User Hub connection established and confirmed ready");

            ConnectionStateChanged?.Invoke($"UserHub:Connected");
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

            _marketHub?.DisposeAsync();
            _marketHub = BuildHub("https://rtc.topstepx.com/hubs/market");

            // Configure connection timeouts for production use
            _marketHub.ServerTimeout = TimeSpan.FromSeconds(60);
            _marketHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            _marketHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            SetupMarketHubEventHandlers();
            
            // Use enhanced connection startup with retry logic
            await StartAndSubscribeWithRetryAsync(
                _marketHub, 
                "MarketHub", 
                async () => {
                    // Empty subscription function - subscriptions are handled separately
                    await Task.CompletedTask;
                }, 
                CancellationToken.None);
            
            _marketHubWired = true;

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                "Market Hub connection established and confirmed ready");

            ConnectionStateChanged?.Invoke($"MarketHub:Connected");
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
    }

    private void SetupMarketHubEventHandlers()
    {
        if (_marketHub == null) return;

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
            
            // Use the enhanced subscription method with proper validation
            if (long.TryParse(validatedAccountId, out var accountIdLong))
            {
                await SubscribeUserAsync(userHub, accountIdLong, _logger);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                    $"Successfully subscribed to user events for account {TradingBot.Abstractions.SecurityHelpers.HashAccountId(validatedAccountId)}");
                return true;
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                    $"Failed to parse account ID as long: {TradingBot.Abstractions.SecurityHelpers.HashAccountId(validatedAccountId)}");
                return false;
            }
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
            
            // Use the enhanced subscription method with proper validation
            await SubscribeMarketAsync(marketHub, new[] { validatedContractId }, _logger);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Successfully subscribed to market events for contract {validatedContractId}");
            return true;
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                $"Failed to subscribe to market events: {ex.Message}");
            return false;
        }
    }

    /// <summary>
    /// Subscribe to multiple contracts at once
    /// </summary>
    /// <param name="contractIds">List of contract IDs to subscribe to</param>
    /// <returns>True if all subscriptions successful, false otherwise</returns>
    public async Task<bool> SubscribeToMultipleContractsAsync(IEnumerable<string> contractIds)
    {
        try
        {
            var validatedContractIds = contractIds
                .Select(cid => TopstepXSubscriptionValidator.ValidateContractIdForSubscription(cid, _logger))
                .ToList();
            
            var marketHub = await GetMarketHubConnectionAsync();
            
            // Use the enhanced subscription method for multiple contracts
            await SubscribeMarketAsync(marketHub, validatedContractIds, _logger);
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                $"Successfully subscribed to market events for {validatedContractIds.Count} contracts");
            return true;
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "SignalRManager", 
                $"Failed to subscribe to multiple contracts: {ex.Message}");
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

    /// <summary>
    /// Enhanced connection startup with robust retry logic and proper subscription handling
    /// </summary>
    private async Task StartAndSubscribeWithRetryAsync(
        HubConnection hub,
        string hubName,
        Func<Task> subscribeAsync,
        CancellationToken cancellationToken)
    {
        var attempts = 0;
        const int maxAttempts = 5;
        var delay = TimeSpan.FromSeconds(1);

        while (attempts < maxAttempts)
        {
            attempts++;
            try
            {
                _logger.LogInformation("[{HubName}] Attempt {Attempt}/{MaxAttempts}: Starting and subscribing...", hubName, attempts, maxAttempts);
                
                using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
                using var combinedCts = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeoutCts.Token);
                
                // Start the connection
                await hub.StartAsync(combinedCts.Token);
                
                // CRITICAL FIX: Wait for connection to be truly ready
                var waitAttempts = 0;
                while (hub.State != HubConnectionState.Connected && waitAttempts < 50)
                {
                    await Task.Delay(100, combinedCts.Token);
                    waitAttempts++;
                }
                
                if (hub.State != HubConnectionState.Connected)
                {
                    throw new InvalidOperationException($"Hub failed to reach Connected state after starting (current state: {hub.State})");
                }
                
                _logger.LogInformation("[{HubName}] Connection confirmed in Connected state. Invoking subscriptions...", hubName);
                
                // Add a small delay to ensure the server is ready
                await Task.Delay(500, combinedCts.Token);
                
                // Now invoke subscriptions with verification
                try
                {
                    await subscribeAsync();
                    
                    // Verify connection is still active after subscription
                    if (hub.State != HubConnectionState.Connected)
                    {
                        throw new InvalidOperationException($"Connection lost during subscription (state: {hub.State})");
                    }
                    
                    _logger.LogInformation("✅ [{HubName}] Successfully connected and subscribed on attempt {Attempt}.", hubName, attempts);
                    return; // Success
                }
                catch (InvalidOperationException invEx) when (invEx.Message.Contains("InvokeCoreAsync"))
                {
                    _logger.LogWarning("⚠️ [{HubName}] Subscription failed, connection may have closed. Will retry...", hubName);
                    throw;
                }
            }
            catch (Exception ex)
            {
                _logger.LogWarning(ex, "⚠️ [{HubName}] Attempt {Attempt}/{MaxAttempts} failed. Reason: {ErrorMessage}", hubName, attempts, maxAttempts, ex.Message);
                
                // Ensure hub is stopped before retry
                if (hub.State != HubConnectionState.Disconnected)
                {
                    try { await hub.StopAsync(CancellationToken.None); } catch { }
                }
                
                if (attempts >= maxAttempts)
                {
                    _logger.LogCritical("🚫 [{HubName}] All {MaxAttempts} connection attempts failed. Giving up.", hubName, maxAttempts);
                    throw;
                }

                await Task.Delay(delay, cancellationToken);
                delay *= 2; // Exponential backoff
            }
        }
    }

    /// <summary>
    /// Subscribe to User Hub events with proper state validation
    /// </summary>
    private static async Task SubscribeUserAsync(HubConnection hub, long accountId, ILogger logger)
    {
        logger.LogInformation("Subscribing to User Hub for account {AccountId}", accountId);
        
        // Subscribe one at a time with verification
        if (hub.State == HubConnectionState.Connected)
        {
            await hub.InvokeAsync("SubscribeOrders", accountId);
            logger.LogDebug("✓ SubscribeOrders completed");
        }
        
        if (hub.State == HubConnectionState.Connected)
        {
            await hub.InvokeAsync("SubscribeTrades", accountId);
            logger.LogDebug("✓ SubscribeTrades completed");
        }
    }

    /// <summary>
    /// Subscribe to Market Hub events with proper state validation
    /// </summary>
    private static async Task SubscribeMarketAsync(HubConnection hub, IEnumerable<string> contractIds, ILogger logger)
    {
        logger.LogInformation("Subscribing to Market Hub for {Count} contracts", contractIds.Count());
        
        foreach (var cid in contractIds)
        {
            if (hub.State == HubConnectionState.Connected)
            {
                await hub.InvokeAsync("SubscribeContractQuotes", cid);
                logger.LogDebug("✓ Subscribed to {ContractId}", cid);
            }
        }
    }

    /// <summary>
    /// Build a properly configured SignalR HubConnection with production settings
    /// </summary>
    private HubConnection BuildHub(string hubPath)
    {
        _logger.LogInformation("Building Hub for {Path}", hubPath);
        return new HubConnectionBuilder()
            .WithUrl(hubPath, options =>
            {
                options.AccessTokenProvider = () => _tokenProvider.GetTokenAsync();
                options.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                options.SkipNegotiation = true;
                options.CloseTimeout = TimeSpan.FromSeconds(30);
            })
            .ConfigureLogging(logging =>
            {
                logging.SetMinimumLevel(LogLevel.Warning);
            })
            .WithAutomaticReconnect(new[]
            {
                TimeSpan.Zero,
                TimeSpan.FromSeconds(2),
                TimeSpan.FromSeconds(5),
                TimeSpan.FromSeconds(10),
                TimeSpan.FromSeconds(30)
            })
            .Build();
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