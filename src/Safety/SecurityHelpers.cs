using System;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace Trading.Safety;

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
    /// Masks a specific account ID value
    /// </summary>
    /// <param name="accountId">The account ID to mask</param>
    /// <returns>The masked account ID as ****1234</returns>
    public static string MaskAccountId(long accountId)
    {
        return MaskAccountId(accountId.ToString());
    }
    
    /// <summary>
    /// Masks order IDs for security - shows only last 4 characters
    /// </summary>
    /// <param name="orderId">The order ID to mask</param>
    /// <returns>The masked order ID as ****abcd or [REDACTED] if null/empty</returns>
    public static string MaskOrderId(string? orderId)
    {
        if (string.IsNullOrEmpty(orderId))
            return "[REDACTED]";
            
        if (orderId.Length <= 4)
            return "****";
            
        var lastFour = orderId.Substring(orderId.Length - 4);
        return $"****{lastFour}";
    }
    
    /// <summary>
    /// Completely removes sensitive identifiers from logs - returns generic placeholder
    /// </summary>
    /// <param name="sensitiveValue">Any sensitive value that should not appear in logs</param>
    /// <returns>A generic placeholder that reveals no information</returns>
    public static string RedactSensitiveValue(string? sensitiveValue)
    {
        return "[REDACTED]";
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