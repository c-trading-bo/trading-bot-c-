using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;
using TradingBot.Infrastructure.TopstepX;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Centralized JWT token provider with automatic refresh capability
/// Fixes authentication token propagation issues across all services
/// </summary>
public interface ITokenProvider
{
    Task<string?> GetTokenAsync();
    Task RefreshTokenAsync();
    bool IsTokenValid { get; }
    event Action<string> TokenRefreshed;
}

public class CentralizedTokenProvider : ITokenProvider, IHostedService
{
    private readonly ILogger<CentralizedTokenProvider> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly IServiceProvider _serviceProvider;
    private readonly Timer _refreshTimer;
    private readonly SemaphoreSlim _refreshLock = new(1, 1);
    
    private string? _currentToken;
    private DateTime _tokenExpiry = DateTime.MinValue;
    private volatile bool _isRefreshing = false;

    public event Action<string>? TokenRefreshed;
    public bool IsTokenValid => !string.IsNullOrEmpty(_currentToken) && DateTime.UtcNow < _tokenExpiry.AddMinutes(-5);

    public CentralizedTokenProvider(
        ILogger<CentralizedTokenProvider> logger,
        ITradingLogger tradingLogger,
        IServiceProvider serviceProvider)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _serviceProvider = serviceProvider;
        
        // Refresh token every 30 minutes
        _refreshTimer = new Timer(RefreshTimerCallback, null, TimeSpan.Zero, TimeSpan.FromMinutes(30));
    }

    public async Task<string?> GetTokenAsync()
    {
        if (IsTokenValid)
        {
            return _currentToken;
        }

        if (!_isRefreshing)
        {
            _ = Task.Run(async () => await RefreshTokenAsync()); // Fire and forget refresh
        }

        // Add small async delay
        await Task.Delay(1);

        // Return current token even if expired (better than null)
        return _currentToken ?? Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
    }

    public async Task RefreshTokenAsync()
    {
        if (_isRefreshing) return;

        await _refreshLock.WaitAsync();
        try
        {
            _isRefreshing = true;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", "Starting token refresh");

            // Always check environment variable first as AutoTopstepXLoginService updates it
            var envToken = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            if (!string.IsNullOrEmpty(envToken) && envToken != _currentToken)
            {
                _currentToken = envToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", 
                    "Token refreshed from environment variable");
                
                TokenRefreshed?.Invoke(_currentToken);
                return;
            }

            // NOTE: AutoTopstepXLoginService integration temporarily disabled due to type resolution issues
            // Will be re-enabled once dependency injection is properly configured

            // If we already have a token from environment, use it even if it's the same
            if (!string.IsNullOrEmpty(envToken))
            {
                _currentToken = envToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "TokenProvider", 
                    "Using environment token as fallback");
                
                TokenRefreshed?.Invoke(_currentToken);
                return;
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "TokenProvider", 
                "Failed to refresh token - no valid source available");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error refreshing JWT token");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "TokenProvider", 
                $"Token refresh failed: {ex.Message}");
        }
        finally
        {
            _isRefreshing = false;
            _refreshLock.Release();
        }
    }

    private async void RefreshTimerCallback(object? state)
    {
        await RefreshTokenAsync();
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", "Centralized token provider started");
        await RefreshTokenAsync();
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", "Centralized token provider stopped");
        if (_refreshTimer != null)
        {
            await _refreshTimer.DisposeAsync();
        }
        _refreshLock.Dispose();
    }
}