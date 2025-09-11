using System;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.Extensions.Logging;

namespace TradingBot.Abstractions;

/// <summary>
/// Security utilities for safely handling sensitive information in logs and API responses
/// CodeQL compliant with proper hashing for sensitive identifiers
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
            return HashAccountId(accountId);
        });
    }
    
    /// <summary>
    /// Creates a CodeQL-compliant hashed representation of an account ID for secure logging
    /// Uses SHA256 to create a consistent but non-reversible identifier
    /// </summary>
    /// <param name="accountId">The account ID to hash</param>
    /// <returns>A SHA256 hash prefix for secure logging</returns>
    public static string HashAccountId(string? accountId)
    {
        if (string.IsNullOrEmpty(accountId))
            return "[REDACTED]";

        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(accountId + "TRADING_BOT_SALT"));
        var hashString = Convert.ToHexString(hashBytes);
        
        // Return first 8 characters of hash for log correlation while maintaining security
        return $"acc_{hashString[..8].ToLowerInvariant()}";
    }
    
    /// <summary>
    /// Masks a specific account ID value for secure logging - CodeQL compliant
    /// </summary>
    /// <param name="accountId">The account ID to mask</param>
    /// <returns>The hashed account ID for secure correlation</returns>
    public static string MaskSpecificAccountId(string? accountId)
    {
        return HashAccountId(accountId);
    }
    
    /// <summary>
    /// Creates a CodeQL-compliant hashed representation of an order ID for secure logging
    /// </summary>
    /// <param name="orderId">The order ID to hash</param>
    /// <returns>A SHA256 hash prefix for secure logging</returns>
    public static string HashOrderId(string? orderId)
    {
        if (string.IsNullOrEmpty(orderId))
            return "[REDACTED]";

        using var sha256 = SHA256.Create();
        var hashBytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(orderId + "ORDER_SALT"));
        var hashString = Convert.ToHexString(hashBytes);
        
        return $"ord_{hashString[..8].ToLowerInvariant()}";
    }
    
    /// <summary>
    /// Masks order IDs in log messages for security
    /// </summary>
    /// <param name="message">The log message that may contain order IDs</param>
    /// <returns>The message with order IDs masked</returns>
    public static string MaskOrderId(string message)
    {
        if (string.IsNullOrEmpty(message))
            return message;
            
        // Mask GUIDs and long numeric IDs that look like order IDs
        var guidPattern = new Regex(@"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b", RegexOptions.Compiled);
        var numericOrderIdPattern = new Regex(@"\b\d{10,20}\b", RegexOptions.Compiled);
        
        var masked = guidPattern.Replace(message, match => HashOrderId(match.Value));
        masked = numericOrderIdPattern.Replace(masked, match => HashOrderId(match.Value));
        
        return masked;
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