using System;
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

            // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
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
                    options.AccessTokenProvider = () => Task.FromResult<string?>(token);
                })
                .WithAutomaticReconnect(new RetryPolicy())
                .Build();

            // Configure connection timeouts for production use
            _userHub.ServerTimeout = TimeSpan.FromSeconds(60);
            _userHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            _userHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            SetupUserHubEventHandlers();
            
            // Start connection and wait for it to be fully established
            await _userHub.StartAsync();
            
            // CRITICAL FIX: Wait for connection to be ready before marking as wired
            await BotCore.HubSafe.WaitForConnected(_userHub, TimeSpan.FromSeconds(30), CancellationToken.None, _logger);
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

            // Ensure token doesn't have "Bearer " prefix (SignalR adds this automatically)
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
                    options.AccessTokenProvider = () => Task.FromResult<string?>(token);
                })
                .WithAutomaticReconnect(new RetryPolicy())
                .Build();

            // Configure connection timeouts for production use
            _marketHub.ServerTimeout = TimeSpan.FromSeconds(60);
            _marketHub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            _marketHub.HandshakeTimeout = TimeSpan.FromSeconds(30);

            SetupMarketHubEventHandlers();
            
            // Start connection and wait for it to be fully established
            await _marketHub.StartAsync();
            
            // CRITICAL FIX: Wait for connection to be ready before marking as wired
            await BotCore.HubSafe.WaitForConnected(_marketHub, TimeSpan.FromSeconds(30), CancellationToken.None, _logger);
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