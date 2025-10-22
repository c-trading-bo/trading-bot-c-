using System;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Helper class to check if the application is running in Lab Mode
/// and suppress warnings/errors that are expected in offline training
/// </summary>
public static class LabModeHelper
{
    private static bool? _isLabMode;
    
    /// <summary>
    /// Check if application is running in Lab Mode
    /// </summary>
    public static bool IsLabMode
    {
        get
        {
            if (_isLabMode.HasValue)
                return _isLabMode.Value;
                
            var labMode = Environment.GetEnvironmentVariable("LAB_MODE");
            _isLabMode = labMode == "1" || labMode?.ToLowerInvariant() == "true";
            return _isLabMode.Value;
        }
    }
    
    /// <summary>
    /// Should suppress TopstepX connection warnings in Lab Mode
    /// </summary>
    public static bool ShouldSuppressTopstepXWarnings => IsLabMode;
    
    /// <summary>
    /// Should suppress missing model file warnings in Lab Mode
    /// </summary>
    public static bool ShouldSuppressMissingModelWarnings => IsLabMode;
    
    /// <summary>
    /// Should suppress cloud/GitHub warnings in Lab Mode
    /// </summary>
    public static bool ShouldSuppressCloudWarnings => IsLabMode;
    
    /// <summary>
    /// Should suppress API health check errors in Lab Mode
    /// </summary>
    public static bool ShouldSuppressAPIErrors => IsLabMode;
}
