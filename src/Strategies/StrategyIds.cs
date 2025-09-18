using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Trading.Strategies;

/// <summary>
/// Generates deterministic strategy IDs that are stable across runs with the same configuration
/// </summary>
public static class StrategyIds
{
    /// <summary>
    /// Generates a deterministic strategy ID based on strategy name and current date
    /// Format: StrategyName_YYYYMMDD
    /// </summary>
    /// <param name="strategyName">The name of the strategy</param>
    /// <param name="date">Optional date override (uses today if null)</param>
    /// <returns>Deterministic strategy ID</returns>
    public static string GenerateStrategyId(string strategyName, DateTime? date = null)
    {
        if (string.IsNullOrWhiteSpace(strategyName))
            throw new ArgumentException("Strategy name cannot be null or empty", nameof(strategyName));
            
        var targetDate = date ?? DateTime.UtcNow;
        var dateString = targetDate.ToString("yyyyMMdd", CultureInfo.InvariantCulture);
        
        // Sanitize strategy name to ensure valid ID
        var sanitizedName = SanitizeStrategyName(strategyName);
        
        return $"{sanitizedName}_{dateString}";
    }
    
    /// <summary>
    /// Generates a deterministic strategy ID based on configuration hash
    /// This ensures the same configuration always produces the same ID
    /// </summary>
    /// <param name="strategyName">The name of the strategy</param>
    /// <param name="configuration">The strategy configuration object</param>
    /// <param name="date">Optional date override (uses today if null)</param>
    /// <returns>Deterministic strategy ID with config hash</returns>
    public static string GenerateStrategyIdWithConfig<T>(string strategyName, T configuration, DateTime? date = null)
        where T : class
    {
        if (string.IsNullOrWhiteSpace(strategyName))
            throw new ArgumentException("Strategy name cannot be null or empty", nameof(strategyName));
            
        if (configuration == null)
            throw new ArgumentNullException(nameof(configuration));
            
        var baseId = GenerateStrategyId(strategyName, date);
        var configHash = GenerateConfigHash(configuration);
        
        return $"{baseId}_{configHash}";
    }
    
    /// <summary>
    /// Generates a deterministic hash from a configuration object
    /// </summary>
    /// <typeparam name="T">Type of the configuration object</typeparam>
    /// <param name="configuration">The configuration object</param>
    /// <returns>8-character hex hash of the configuration</returns>
    public static string GenerateConfigHash<T>(T configuration)
        where T : class
    {
        if (configuration == null)
            throw new ArgumentNullException(nameof(configuration));
            
        // Serialize configuration to JSON for consistent hashing
        var jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            WriteIndented = false,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };
        
        var configJson = JsonSerializer.Serialize(configuration, jsonOptions);
        
        // Generate SHA256 hash and take first 8 characters
        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(configJson));
        var fullHash = Convert.ToHexString(hashBytes);
        
        // Defensive bounds checking to prevent ArgumentOutOfRangeException
        if (string.IsNullOrEmpty(fullHash))
        {
            return "00000000"; // Fallback hash
        }
        
        var hashLength = Math.Min(8, fullHash.Length);
        return fullHash.Substring(0, hashLength).ToLowerInvariant();
    }
    
    /// <summary>
    /// Sanitizes a strategy name to ensure it's valid for use in IDs
    /// </summary>
    /// <param name="strategyName">The strategy name to sanitize</param>
    /// <returns>Sanitized strategy name</returns>
    private static string SanitizeStrategyName(string strategyName)
    {
        if (string.IsNullOrWhiteSpace(strategyName))
            return "Unknown";
            
        // Replace invalid characters with underscores
        var sanitized = new StringBuilder();
        
        foreach (var c in strategyName)
        {
            if (char.IsLetterOrDigit(c))
            {
                sanitized.Append(c);
            }
            else if (c == '_' || c == '-')
            {
                sanitized.Append(c);
            }
            else
            {
                sanitized.Append('_');
            }
        }
        
        var result = sanitized.ToString();
        
        // Ensure it doesn't start or end with underscore
        result = result.Trim('_');
        
        // Ensure it's not empty after sanitization
        if (string.IsNullOrEmpty(result))
            result = "Strategy";
            
        return result;
    }
    
    /// <summary>
    /// Parses a strategy ID to extract the strategy name and date
    /// </summary>
    /// <param name="strategyId">The strategy ID to parse</param>
    /// <returns>Tuple containing strategy name and date, or null if parsing fails</returns>
    public static (string StrategyName, DateTime Date)? ParseStrategyId(string strategyId)
    {
        if (string.IsNullOrWhiteSpace(strategyId))
            return null;
            
        var parts = strategyId.Split('_');
        
        if (parts.Length < 2)
            return null;
            
        var datePart = parts[^1]; // Last part should be the date
        
        if (datePart.Length == 8 && DateTime.TryParseExact(datePart, "yyyyMMdd", CultureInfo.InvariantCulture, DateTimeStyles.None, out var date))
        {
            var strategyName = string.Join("_", parts.Take(parts.Length - 1));
            return (strategyName, date);
        }
        
        return null;
    }
}