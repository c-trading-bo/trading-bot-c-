using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Safety.Security;

/// <summary>
/// Interface for hot-reloading secrets and configurations without restart
/// </summary>
public interface ISecretRotationManager
{
    /// <summary>
    /// Initialize secret rotation monitoring
    /// </summary>
    Task InitializeAsync(SecretRotationConfiguration configuration, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current secret value with automatic rotation handling
    /// </summary>
    Task<string?> GetSecretAsync(string secretName, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Register callback for secret rotation events
    /// </summary>
    void RegisterRotationCallback(string secretName, Func<string, string, Task> rotationCallback);
    
    /// <summary>
    /// Force rotation of specific secret
    /// </summary>
    Task<SecretRotationResult> ForceRotationAsync(string secretName, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check rotation status of all secrets
    /// </summary>
    Task<Dictionary<string, SecretStatus>> GetSecretStatusAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Validate that all required secrets are available and valid
    /// </summary>
    Task<SecretValidationReport> ValidateSecretsAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// Configuration for secret rotation
/// </summary>
public record SecretRotationConfiguration(
    Dictionary<string, SecretConfig> SecretConfigurations,
    TimeSpan CheckInterval,
    bool EnableAutoRotation = true,
    string? SecretProvider = null
);

/// <summary>
/// Configuration for individual secret
/// </summary>
public record SecretConfig(
    string SecretName,
    string Source, // Environment, KeyVault, SecretManager, etc.
    TimeSpan RotationInterval,
    bool IsRequired = true,
    string? ValidationPattern = null,
    Dictionary<string, object>? ProviderSpecificConfig = null
);

/// <summary>
/// Result of secret rotation operation
/// </summary>
public record SecretRotationResult(
    string SecretName,
    bool Success,
    DateTime RotatedAt,
    string? PreviousVersion,
    string? NewVersion,
    string? Error = null
);

/// <summary>
/// Current status of a secret
/// </summary>
public record SecretStatus(
    string SecretName,
    bool IsAvailable,
    DateTime LastRotated,
    DateTime? NextRotation,
    bool RotationRequired,
    string? LastError = null,
    SecretHealth Health = SecretHealth.Unknown
);

/// <summary>
/// Health status of secrets
/// </summary>
public enum SecretHealth
{
    Healthy,
    Warning,
    Critical,
    Unknown
}

/// <summary>
/// Validation report for all secrets
/// </summary>
public record SecretValidationReport(
    DateTime GeneratedAt,
    bool AllSecretsValid,
    Dictionary<string, SecretValidationResult> Results,
    List<string> CriticalIssues,
    List<string> Warnings
);

/// <summary>
/// Validation result for individual secret
/// </summary>
public record SecretValidationResult(
    string SecretName,
    bool IsValid,
    bool IsAvailable,
    bool PassesValidation,
    List<string> Issues
);