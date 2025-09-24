using Microsoft.Extensions.Logging;
using System.Security.Cryptography;
using System.Text;

namespace OrchestratorAgent.ML;

/// <summary>
/// Simple A/A testing for RL vs baseline CVaR sizer
/// Provides deterministic traffic splitting based on signal ID hash
/// </summary>
internal sealed class SizerCanary
{
    private readonly ILogger<SizerCanary> _log;
    private readonly bool _enabled;
    private readonly double _rlTrafficFraction;

    public SizerCanary(ILogger<SizerCanary> log)
    {
        _log = log;
        _enabled = EnvFlag("RL_SIZER_CANARY_ENABLED", true);
        const double defaultRlFraction = 0.5; // Default 50/50 split between RL and baseline
        const double minFraction = 0.0; // No RL traffic
        const double maxFraction = 1.0; // All RL traffic
        _rlTrafficFraction = Math.Clamp(EnvDouble("RL_SIZER_CANARY_RL_FRACTION", defaultRlFraction), minFraction, maxFraction);

        _log.LogInformation("[SizerCanary] Enabled={Enabled} RLFraction={Fraction:F2}",
            _enabled, _rlTrafficFraction);
    }

    /// <summary>
    /// Determines whether to use RL sizer for given signal
    /// Uses deterministic hash-based traffic splitting
    /// </summary>
    public bool ShouldUseRl(string signalId)
    {
        if (!_enabled) return false;
        if (string.IsNullOrEmpty(signalId)) return false;

        // Deterministic hash-based splitting
        using var sha = SHA256.Create();
        var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(signalId));
        var hashValue = BitConverter.ToUInt32(hash, 0);
        var fraction = (double)hashValue / uint.MaxValue;

        bool useRl = fraction < _rlTrafficFraction;

        _log.LogDebug("[SizerCanary] Signal={SignalId} Hash={Hash:F6} UseRL={UseRL}",
            signalId, fraction, useRl);

        return useRl;
    }

    /// <summary>
    /// Gets current canary configuration for diagnostics
    /// </summary>
    public object GetConfig() => new
    {
        enabled = _enabled,
        rlTrafficFraction = _rlTrafficFraction,
        timestamp = DateTime.UtcNow
    };

    private static bool EnvFlag(string key, bool defaultValue = false)
    {
        var value = Environment.GetEnvironmentVariable(key);
        if (string.IsNullOrWhiteSpace(value)) return defaultValue;
        value = value.Trim().ToLowerInvariant();
        return value is "1" or "true" or "yes" or "on";
    }

    private static double EnvDouble(string key, double defaultValue)
    {
        var value = Environment.GetEnvironmentVariable(key);
        return double.TryParse(value, out var result) ? result : defaultValue;
    }
}
