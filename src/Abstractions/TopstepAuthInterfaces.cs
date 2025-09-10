using System;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Abstractions;

/// <summary>
/// Interface for TopstepX authentication services
/// </summary>
public interface ITopstepAuth
{
    Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct = default);
    Task EnsureFreshTokenAsync(CancellationToken ct = default);
}

/// <summary>
/// TopstepX credential information
/// </summary>
public class TopstepXCredentials
{
    public string Username { get; set; } = string.Empty;
    public string ApiKey { get; set; } = string.Empty;
    public string? JwtToken { get; set; }
    public string? AccountId { get; set; }
    public DateTime? LastUpdated { get; set; } = DateTime.UtcNow;
    public string Source { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime ExpiresAt { get; set; } = DateTime.UtcNow.AddHours(24);
    public string EnvironmentName { get; set; } = string.Empty;
    public bool IsValid => !string.IsNullOrEmpty(Username) && !string.IsNullOrEmpty(ApiKey);
}

/// <summary>
/// Report of credential discovery from various sources
/// </summary>
public class CredentialDiscoveryReport
{
    public bool HasEnvironmentCredentials { get; set; }
    public bool HasFileCredentials { get; set; }
    public TopstepXCredentials? EnvironmentCredentials { get; set; }
    public TopstepXCredentials? FileCredentials { get; set; }
    public TopstepXCredentials? RecommendedCredentials { get; set; }
    public string? RecommendedSource { get; set; }
    public string? FileErrorMessage { get; set; }
    public string? DiscoveryError { get; set; }

    public bool HasAnyCredentials => HasEnvironmentCredentials || HasFileCredentials;
    public int TotalSourcesFound => (HasEnvironmentCredentials ? 1 : 0) + (HasFileCredentials ? 1 : 0);
}