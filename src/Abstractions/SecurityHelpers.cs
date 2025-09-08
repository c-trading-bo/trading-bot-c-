using System;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace TradingBot.Abstractions;

/// <summary>
/// Security utilities for safely handling sensitive information in logs and API responses
/// </summary>
public static class SecurityHelpers
{
    private static readonly Regex AccountIdPattern = new(@"\b\d{5,12}\b", RegexOptions.Compiled);
    
    /// <summary>
    /// Masks account IDs in log messages for security
    /// </summary>
    /// <param name="message">The log message that may contain account IDs</param>
    /// <returns>The message with account IDs masked as ****1234</returns>
    public static string MaskAccountId(string message)
    {
        if (string.IsNullOrEmpty(message))
            return message;
            
        return AccountIdPattern.Replace(message, match =>
        {
            var accountId = match.Value;
            if (accountId.Length <= 4)
                return "****";
            
            var lastFour = accountId.Substring(accountId.Length - 4);
            return $"****{lastFour}";
        });
    }
    
    /// <summary>
    /// Masks sensitive data in API responses before logging
    /// </summary>
    /// <param name="response">The API response that may contain sensitive data</param>
    /// <returns>The response with sensitive data masked</returns>
    public static string MaskSensitiveData(string response)
    {
        if (string.IsNullOrEmpty(response))
            return response;
            
        // Mask common sensitive patterns
        var patterns = new[]
        {
            (@"""token"":\s*""[^""]+""", @"""token"": ""****"""),
            (@"""apiKey"":\s*""[^""]+""", @"""apiKey"": ""****"""),
            (@"""password"":\s*""[^""]+""", @"""password"": ""****"""),
            (@"""jwt"":\s*""[^""]+""", @"""jwt"": ""****"""),
            (@"""authorization"":\s*""[^""]+""", @"""authorization"": ""****""")
        };
        
        var maskedResponse = response;
        foreach (var (pattern, replacement) in patterns)
        {
            maskedResponse = Regex.Replace(maskedResponse, pattern, replacement, RegexOptions.IgnoreCase);
        }
        
        return maskedResponse;
    }
    
    /// <summary>
    /// Returns a generic error message for API responses, logging the actual exception server-side
    /// </summary>
    /// <param name="ex">The exception that occurred</param>
    /// <param name="logger">Optional logger to record the full exception details</param>
    /// <returns>A generic error message safe for client consumption</returns>
    public static string GetGenericErrorMessage(Exception ex, Microsoft.Extensions.Logging.ILogger? logger = null)
    {
        // Log the full exception details server-side for debugging
        logger?.LogError(ex, "An error occurred during API operation");
        
        // Return a generic message that doesn't expose internal details
        return "An internal error occurred. Please try again or contact support if the issue persists.";
    }
}