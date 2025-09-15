using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Security;

/// <summary>
/// Interface for JWT token refresh and expiry management
/// </summary>
public interface ITokenManager
{
    /// <summary>
    /// Initialize token management with configuration
    /// </summary>
    Task InitializeAsync(TokenConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get a valid token, refreshing if necessary
    /// </summary>
    Task<string> GetValidTokenAsync(TokenScope scope, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check if token needs refresh
    /// </summary>
    Task<bool> NeedsRefreshAsync(TokenScope scope, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Force token refresh
    /// </summary>
    Task<TokenRefreshResult> RefreshTokenAsync(TokenScope scope, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register callback for token expiry events
    /// </summary>
    void RegisterExpiryCallback(Func<TokenScope, TimeSpan, Task> callback);
    
    /// <summary>
    /// Get token status and metrics
    /// </summary>
    Task<Dictionary<TokenScope, TokenStatus>> GetTokenStatusAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for token management
/// </summary>
public record TokenConfiguration(
    Dictionary<TokenScope, ScopedTokenConfig> ScopeConfigurations,
    TimeSpan RefreshThreshold,
    TimeSpan RetryInterval = default,
    int MaxRetryAttempts = 3
)
{
    public TimeSpan RetryInterval { get; init; } = RetryInterval == default ? TimeSpan.FromSeconds(30) : RetryInterval;
}

/// <summary>
/// Configuration for specific token scope
/// </summary>
public record ScopedTokenConfig(
    string ClientId,
    string ClientSecret,
    string TokenEndpoint,
    List<string> Scopes,
    TimeSpan TokenLifetime = default
)
{
    public TimeSpan TokenLifetime { get; init; } = TokenLifetime == default ? TimeSpan.FromHours(1) : TokenLifetime;
}

/// <summary>
/// Token scopes for least-privilege access
/// </summary>
public enum TokenScope
{
    DataAccess,
    Trading,
    Administration,
    Monitoring
}

/// <summary>
/// Result of token refresh operation
/// </summary>
public record TokenRefreshResult(
    bool Success,
    string? NewToken,
    DateTime? ExpiresAt,
    string? Error = null,
    TimeSpan RefreshDuration = default
);

/// <summary>
/// Current status of a token
/// </summary>
public record TokenStatus(
    TokenScope Scope,
    bool IsValid,
    DateTime? ExpiresAt,
    TimeSpan? TimeToExpiry,
    DateTime LastRefreshed,
    int RefreshAttempts,
    string? LastError = null
);