using System;
using System.IdentityModel.Tokens.Jwt;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// JWT Lifecycle Manager for TopstepX connection tokens
/// Tracks token expiration and triggers refresh at 75% of lifetime
/// Provides events for SignalR reconnection coordination
/// </summary>
public interface IJwtLifecycleManager
{
    Task<bool> ValidateTokenAsync(string? token);
    bool IsTokenValid(string? token);
    DateTime? GetTokenExpiry(string? token);
    double GetTokenLifetimePercentage(string? token);
    event Action<string> TokenNeedsRefresh;
    event Action<string> TokenRefreshed;
}

public class JwtLifecycleManager : IJwtLifecycleManager, IHostedService, IDisposable
{
    // JWT validation constants
    private const int TokenValidityBufferMinutes = 5;
    private const int HealthCheckIntervalSeconds = 100;
    private const int RefreshThresholdPercentage = 75;
    private const int CriticalThresholdPercentage = 50;
    
    private readonly ILogger<JwtLifecycleManager> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly Timer _expirationCheckTimer;
    private readonly SemaphoreSlim _validationLock = new(1, 1);
    private string? _currentToken;
    private DateTime? _currentTokenExpiry;
    private bool _disposed = false;

    /// <summary>
    /// Gets the current token expiry time for monitoring purposes
    /// </summary>
    public DateTime? CurrentTokenExpiry => _currentTokenExpiry;

    public event Action<string>? TokenNeedsRefresh;
    public event Action<string>? TokenRefreshed;

    public JwtLifecycleManager(
        ILogger<JwtLifecycleManager> logger,
        ITradingLogger tradingLogger)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        
        // Check token expiration every 5 minutes
        _expirationCheckTimer = new Timer(CheckTokenExpiration, null, 
            TimeSpan.FromMinutes(5), TimeSpan.FromMinutes(5));
    }

    public async Task<bool> ValidateTokenAsync(string? token)
    {
        if (string.IsNullOrEmpty(token)) return false;

        await _validationLock.WaitAsync();
        try
        {
            var isValid = IsTokenValid(token);
            if (isValid)
            {
                _currentToken = token;
                _currentTokenExpiry = GetTokenExpiry(token);
                
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "JwtLifecycleManager",
                    $"Token validated successfully. Expires: {_currentTokenExpiry:yyyy-MM-dd HH:mm:ss} UTC");
            }
            else
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "JwtLifecycleManager",
                    "Token validation failed - token is expired or invalid");
            }
            
            return isValid;
        }
        finally
        {
            _validationLock.Release();
        }
    }

    public bool IsTokenValid(string? token)
    {
        if (string.IsNullOrEmpty(token)) return false;

        try
        {
            var handler = new JwtSecurityTokenHandler();
            if (!handler.CanReadToken(token)) return false;

            var jsonToken = handler.ReadJwtToken(token);
            var expiry = jsonToken.ValidTo;

            // Token is valid if it expires more than the buffer time from now
            return expiry > DateTime.UtcNow.AddMinutes(TokenValidityBufferMinutes);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error validating JWT token");
            return false;
        }
    }

    public DateTime? GetTokenExpiry(string? token)
    {
        if (string.IsNullOrEmpty(token)) return null;

        try
        {
            var handler = new JwtSecurityTokenHandler();
            if (!handler.CanReadToken(token)) return null;

            var jsonToken = handler.ReadJwtToken(token);
            return jsonToken.ValidTo;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error getting token expiry");
            return null;
        }
    }

    public double GetTokenLifetimePercentage(string? token)
    {
        if (string.IsNullOrEmpty(token)) return 0.0;

        try
        {
            var handler = new JwtSecurityTokenHandler();
            if (!handler.CanReadToken(token)) return 0.0;

            var jsonToken = handler.ReadJwtToken(token);
            var issuedAt = jsonToken.IssuedAt;
            var expiry = jsonToken.ValidTo;
            var now = DateTime.UtcNow;

            if (issuedAt == DateTime.MinValue || expiry <= issuedAt) return 0.0;

            var totalLifetime = expiry - issuedAt;
            var elapsed = now - issuedAt;

            if (elapsed.TotalSeconds < 0) return 0.0;
            if (elapsed >= totalLifetime) return HealthCheckIntervalSeconds;

            return (elapsed.TotalSeconds / totalLifetime.TotalSeconds) * HealthCheckIntervalSeconds;
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Error calculating token lifetime percentage");
            return 0.0;
        }
    }

    private void CheckTokenExpiration(object? state)
    {
        if (_disposed || string.IsNullOrEmpty(_currentToken)) return;

        // Fire-and-forget is acceptable here as we handle exceptions internally
        _ = Task.Run(async () => await CheckTokenExpirationAsync());
    }

    private async Task CheckTokenExpirationAsync()
    {
        try
        {
            var lifetimePercentage = GetTokenLifetimePercentage(_currentToken!);
            
            // Trigger refresh at the configured threshold of token lifetime
            if (lifetimePercentage >= RefreshThresholdPercentage)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "JwtLifecycleManager",
                    $"Token is {lifetimePercentage:F1}% through its lifetime - triggering refresh");
                
                TokenNeedsRefresh?.Invoke(_currentToken!);
            }
            else if (lifetimePercentage >= CriticalThresholdPercentage)
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "JwtLifecycleManager",
                    $"Token is {lifetimePercentage:F1}% through its lifetime");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error during token expiration check");
        }
    }

    public void NotifyTokenRefreshed(string newToken)
    {
        _currentToken = newToken;
        _currentTokenExpiry = GetTokenExpiry(newToken);
        TokenRefreshed?.Invoke(newToken);
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[JWT-LIFECYCLE] JWT Lifecycle Manager starting...");
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "JwtLifecycleManager",
            "JWT Lifecycle Manager service started - monitoring token expiration every 5 minutes");
        
        // Perform initial token validation if we have one
        var currentToken = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
        if (!string.IsNullOrEmpty(currentToken))
        {
            var isValid = await ValidateTokenAsync(currentToken);
            if (!isValid)
            {
                _logger.LogWarning("[JWT-LIFECYCLE] Current JWT token is invalid or expired");
                await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "JwtLifecycleManager",
                    "Initial JWT token validation failed - token refresh required");
            }
        }
    }

    public async Task StopAsync(CancellationToken cancellationToken)
    {
        _logger.LogInformation("[JWT-LIFECYCLE] JWT Lifecycle Manager stopping...");
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "JwtLifecycleManager",
            "JWT Lifecycle Manager service stopped");
        
        _expirationCheckTimer?.Change(Timeout.Infinite, 0);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            if (disposing)
            {
                _expirationCheckTimer?.Dispose();
                _validationLock?.Dispose();
            }
            _disposed = true;
        }
    }
}