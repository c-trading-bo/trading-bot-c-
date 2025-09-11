using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;

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
            _userHub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/user", options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(token);
                })
                .WithAutomaticReconnect(new RetryPolicy())
                .Build();

            SetupUserHubEventHandlers();
            
            await _userHub.StartAsync();
            _userHubWired = true;

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                "User Hub connection established successfully");

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
            _marketHub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult<string?>(token);
                })
                .WithAutomaticReconnect(new RetryPolicy())
                .Build();

            SetupMarketHubEventHandlers();
            
            await _marketHub.StartAsync();
            _marketHubWired = true;

            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "SignalRManager", 
                "Market Hub connection established successfully");

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
            if (_userHub?.State != HubConnectionState.Connected && _userHubWired)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "SignalRManager", 
                    $"User Hub health check failed - State: {_userHub?.State}");
                _userHubWired = false;
            }

            if (_marketHub?.State != HubConnectionState.Connected && _marketHubWired)
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