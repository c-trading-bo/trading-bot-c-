using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using TradingBot.Abstractions;

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
    private volatile bool _isRefreshing;
    private AutoTopstepXLoginService? _autoLoginService;

    public event Action<string>? TokenRefreshed;
    public bool IsTokenValid => !string.IsNullOrEmpty(_currentToken) && DateTime.UtcNow < _tokenExpiry.AddMinutes(-5) && IsJwtTokenActuallyValid(_currentToken);

    private bool IsJwtTokenActuallyValid(string? token)
    {
        if (string.IsNullOrEmpty(token)) return false;
        
        try
        {
            var parts = token.Split('.');
            if (parts.Length != 3) return false;
            
            var payload = parts[1];
            while (payload.Length % 4 != 0) { payload += "="; }
            
            var decoded = System.Text.Encoding.UTF8.GetString(System.Convert.FromBase64String(payload));
            var json = System.Text.Json.JsonDocument.Parse(decoded);
            
            if (json.RootElement.TryGetProperty("exp", out var expElement))
            {
                var expiry = DateTimeOffset.FromUnixTimeSeconds(expElement.GetInt64());
                return expiry > DateTimeOffset.UtcNow.AddMinutes(5); // Valid if expires more than 5 minutes from now
            }
            
            return false;
        }
        catch
        {
            return false;
        }
    }

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
        // Try to get the AutoTopstepXLoginService if we haven't already
        if (_autoLoginService == null)
        {
            _autoLoginService = _serviceProvider.GetService<AutoTopstepXLoginService>();
        }

        // First priority: Get token directly from AutoTopstepXLoginService
        if (_autoLoginService?.JwtToken != null)
        {
            _logger.LogInformation("[TOKEN_PROVIDER] Got JWT from AutoTopstepXLoginService: Length={Length}", 
                _autoLoginService.JwtToken.Length);
            
            // Update our cached token if it's different
            if (_autoLoginService.JwtToken != _currentToken)
            {
                _currentToken = _autoLoginService.JwtToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                _logger.LogInformation("[TOKEN_PROVIDER] JWT token updated from AutoTopstepXLoginService");
            }
            return _currentToken;
        }

        // Fallback: Check environment variable (for compatibility)
        var envToken = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
        _logger.LogInformation("[TOKEN_PROVIDER] Environment variable TOPSTEPX_JWT: {HasToken}, Length: {Length}", 
            !string.IsNullOrEmpty(envToken), envToken?.Length ?? 0);
        
        if (!string.IsNullOrEmpty(envToken))
        {
            // Update our cached token if we found one in environment
            if (envToken != _currentToken)
            {
                _currentToken = envToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                _logger.LogInformation("[TOKEN_PROVIDER] JWT token updated from environment variable");
            }
            return _currentToken;
        }

        // If no token found and we have a cached valid token, use it
        if (IsTokenValid)
        {
            _logger.LogInformation("[TOKEN_PROVIDER] Using cached valid token");
            return _currentToken;
        }

        // If no environment token and we're not already refreshing, try to refresh
        if (!_isRefreshing)
        {
            await RefreshTokenAsync().ConfigureAwait(false);
        }

        // Return whatever we have (could be null if refresh failed)
        _logger.LogInformation("[TOKEN_PROVIDER] Returning token: {HasToken}, Length: {Length}", 
            !string.IsNullOrEmpty(_currentToken), _currentToken?.Length ?? 0);
        return _currentToken;
    }

    public async Task RefreshTokenAsync()
    {
        if (_isRefreshing) return;

        await _refreshLock.WaitAsync().ConfigureAwait(false);
        try
        {
            _isRefreshing = true;
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", "Starting token refresh").ConfigureAwait(false);

            // Always check environment variable first as AutoTopstepXLoginService updates it
            var envToken = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
            if (!string.IsNullOrEmpty(envToken) && envToken != _currentToken)
            {
                _currentToken = envToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", 
                    "Token refreshed from environment variable").ConfigureAwait(false);
                
                TokenRefreshed?.Invoke(_currentToken);
                return;
            }

            // Try to refresh via AutoTopstepXLoginService if available
            if (_autoLoginService != null && !string.IsNullOrEmpty(_autoLoginService.JwtToken))
            {
                _currentToken = _autoLoginService.JwtToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", 
                    "Token refreshed from AutoTopstepXLoginService").ConfigureAwait(false);
                
                TokenRefreshed?.Invoke(_currentToken);
                return;
            }

            // If we already have a token from environment, use it even if it's the same
            if (!string.IsNullOrEmpty(envToken))
            {
                _currentToken = envToken;
                _tokenExpiry = DateTime.UtcNow.AddHours(1);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "TokenProvider", 
                    "Using environment token as fallback").ConfigureAwait(false);
                
                TokenRefreshed?.Invoke(_currentToken);
                return;
            }

            // If we still don't have a valid token, try to trigger authentication
            if (!IsTokenValid)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "TokenProvider", 
                    "No valid token found, attempting to trigger authentication via AutoTopstepXLoginService").ConfigureAwait(false);
                
                // Force authentication attempt if AutoTopstepXLoginService is available
                if (_autoLoginService != null && !_autoLoginService.IsAuthenticated)
                {
                    // The AutoTopstepXLoginService should handle authentication in its background service
                    // Just wait a bit and check again
                    await Task.Delay(2000).ConfigureAwait(false);
                    
                    if (!string.IsNullOrEmpty(_autoLoginService.JwtToken))
                    {
                        _currentToken = _autoLoginService.JwtToken;
                        _tokenExpiry = DateTime.UtcNow.AddHours(1);
                        Environment.SetEnvironmentVariable("TOPSTEPX_JWT", _currentToken);
                        
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", 
                            "Token obtained after authentication attempt").ConfigureAwait(false);
                        
                        TokenRefreshed?.Invoke(_currentToken);
                    }
                }
            }

            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "TokenProvider", 
                "Failed to refresh token - no valid source available").ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error refreshing JWT token");
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "TokenProvider", 
                $"Token refresh failed: {ex.Message}").ConfigureAwait(false);
        }
        finally
        {
            _isRefreshing;
            _refreshLock.Release();
        }
    }

    private async void RefreshTimerCallback(object? state)
    {
        await RefreshTokenAsync().ConfigureAwait(false);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", "Centralized token provider started").ConfigureAwait(false);
        await RefreshTokenAsync().ConfigureAwait(false);
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "TokenProvider", "Centralized token provider stopped").ConfigureAwait(false);
        if (_refreshTimer != null)
        {
            await _refreshTimer.DisposeAsync().ConfigureAwait(false);
        }
        _refreshLock.Dispose();
    }
}