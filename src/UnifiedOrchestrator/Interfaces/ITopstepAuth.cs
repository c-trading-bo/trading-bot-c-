using System;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// TopstepX authentication service interface
/// </summary>
internal interface ITopstepAuth
{
    /// <summary>
    /// Gets the current authentication token
    /// </summary>
    Task<string?> GetTokenAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Refreshes the authentication token
    /// </summary>
    Task<bool> RefreshTokenAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Checks if the current token is valid
    /// </summary>
    Task<bool> IsTokenValidAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Gets the account ID associated with the token
    /// </summary>
    Task<string?> GetAccountIdAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Simple TopstepX authentication service implementation
/// </summary>
internal sealed class TopstepAuth : ITopstepAuth
{
    private readonly Microsoft.Extensions.Logging.ILogger<TopstepAuth> _logger;
    private readonly Microsoft.Extensions.Configuration.IConfiguration _configuration;
    private volatile string? _cachedToken;
    private DateTime _tokenExpiry = DateTime.MinValue;
    
    public TopstepAuth(
        Microsoft.Extensions.Logging.ILogger<TopstepAuth> logger,
        Microsoft.Extensions.Configuration.IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;
    }
    
    public async Task<string?> GetTokenAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        // Try cached token first
        if (!string.IsNullOrEmpty(_cachedToken) && DateTime.UtcNow < _tokenExpiry)
        {
            return _cachedToken;
        }
        
        // Get from configuration
        var token = _configuration.GetValue<string>("TOPSTEPX_JWT") ?? 
                   _configuration.GetValue<string>("TOPSTEPX_AUTH_TOKEN") ??
                   Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
        
        if (!string.IsNullOrEmpty(token))
        {
            _cachedToken = token;
            _tokenExpiry = DateTime.UtcNow.AddHours(1); // Cache for 1 hour
            return token;
        }
        
        _logger.LogWarning("No TopstepX authentication token found");
        return null;
    }
    
    public async Task<bool> RefreshTokenAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        // Clear cached token to force refresh
        _cachedToken = null;
        _tokenExpiry = DateTime.MinValue;
        
        // Get fresh token
        var token = await GetTokenAsync(cancellationToken).ConfigureAwait(false);
        return !string.IsNullOrEmpty(token);
    }
    
    public async Task<bool> IsTokenValidAsync(CancellationToken cancellationToken = default)
    {
        var token = await GetTokenAsync(cancellationToken).ConfigureAwait(false);
        
        if (string.IsNullOrEmpty(token))
        {
            return false;
        }
        
        // Basic token validation - check expiry from cache
        return DateTime.UtcNow < _tokenExpiry;
    }
    
    public async Task<string?> GetAccountIdAsync(CancellationToken cancellationToken = default)
    {
        await Task.CompletedTask.ConfigureAwait(false);
        
        return _configuration.GetValue<string>("TOPSTEPX_ACCOUNT_ID") ?? 
               Environment.GetEnvironmentVariable("TOPSTEPX_ACCOUNT_ID");
    }
}