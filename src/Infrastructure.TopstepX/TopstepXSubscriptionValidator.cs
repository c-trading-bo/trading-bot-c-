using Microsoft.Extensions.Logging;
using System;
using System.Text.RegularExpressions;

namespace TradingBot.Infrastructure.TopstepX;

/// <summary>
/// Validates subscription parameters for TopstepX SignalR hub compliance
/// Ensures all subscription calls match the exact TopstepX API specification
/// </summary>
public static class TopstepXSubscriptionValidator
{
    private static readonly Regex AccountIdPattern = new(@"^\d{5,12}$", RegexOptions.Compiled);
    private static readonly Regex ContractIdPattern = new(@"^[A-Z]{2,4}[0-9]{0,4}$", RegexOptions.Compiled);
    
    /// <summary>
    /// Validates account ID format for TopstepX user hub subscriptions
    /// </summary>
    /// <param name="accountId">The account ID to validate</param>
    /// <returns>True if valid, false otherwise</returns>
    public static bool IsValidAccountId(string? accountId)
    {
        if (string.IsNullOrWhiteSpace(accountId))
            return false;
            
        return AccountIdPattern.IsMatch(accountId);
    }
    
    /// <summary>
    /// Validates contract ID format for TopstepX market hub subscriptions
    /// </summary>
    /// <param name="contractId">The contract ID to validate (e.g., "ES", "NQ", "ES2412")</param>
    /// <returns>True if valid, false otherwise</returns>
    public static bool IsValidContractId(string? contractId)
    {
        if (string.IsNullOrWhiteSpace(contractId))
            return false;
            
        return ContractIdPattern.IsMatch(contractId);
    }
    
    /// <summary>
    /// Validates and sanitizes account ID for user hub subscription calls
    /// </summary>
    /// <param name="accountId">The account ID to validate</param>
    /// <param name="logger">Logger for validation failures</param>
    /// <returns>Validated account ID or throws ArgumentException</returns>
    public static string ValidateAccountIdForSubscription(string? accountId, ILogger? logger = null)
    {
        if (string.IsNullOrWhiteSpace(accountId))
        {
            logger?.LogError("AccountId is null or empty for user hub subscription");
            throw new ArgumentException("AccountId cannot be null or empty", nameof(accountId));
        }
        
        if (!IsValidAccountId(accountId))
        {
            logger?.LogError("Invalid AccountId format for user hub subscription: {AccountIdHash}", 
                TradingBot.Abstractions.SecurityHelpers.HashAccountId(accountId));
            throw new ArgumentException($"Invalid AccountId format: must be 5-12 digits", nameof(accountId));
        }
        
        return accountId;
    }
    
    /// <summary>
    /// Validates and sanitizes contract ID for market hub subscription calls
    /// </summary>
    /// <param name="contractId">The contract ID to validate</param>
    /// <param name="logger">Logger for validation failures</param>
    /// <returns>Validated contract ID or throws ArgumentException</returns>
    public static string ValidateContractIdForSubscription(string? contractId, ILogger? logger = null)
    {
        if (string.IsNullOrWhiteSpace(contractId))
        {
            logger?.LogError("ContractId is null or empty for market hub subscription");
            throw new ArgumentException("ContractId cannot be null or empty", nameof(contractId));
        }
        
        if (!IsValidContractId(contractId))
        {
            logger?.LogError("Invalid ContractId format for market hub subscription: {ContractId}", contractId);
            throw new ArgumentException($"Invalid ContractId format: must match pattern {ContractIdPattern}", nameof(contractId));
        }
        
        return contractId.ToUpperInvariant(); // Normalize to uppercase per TopstepX spec
    }
    
    /// <summary>
    /// Gets the standard TopstepX user hub subscription methods with parameter validation
    /// </summary>
    /// <returns>Array of supported subscription method names</returns>
    public static string[] GetSupportedUserHubMethods()
    {
        return new[]
        {
            "SubscribeOrders",
            "SubscribeTrades", 
            "SubscribePositions",
            "SubscribeAccount"
        };
    }
    
    /// <summary>
    /// Gets the standard TopstepX market hub subscription methods with parameter validation
    /// </summary>
    /// <returns>Array of supported subscription method names</returns>
    public static string[] GetSupportedMarketHubMethods()
    {
        return new[]
        {
            "SubscribeContractQuotes",
            "SubscribeContractTrades",
            "SubscribeContractMarketDepth"
        };
    }
    
    /// <summary>
    /// Validates that a subscription method name is supported by TopstepX specification
    /// </summary>
    /// <param name="methodName">The method name to validate</param>
    /// <param name="isUserHub">True for user hub methods, false for market hub methods</param>
    /// <returns>True if method is supported, false otherwise</returns>
    public static bool IsValidSubscriptionMethod(string methodName, bool isUserHub)
    {
        var supportedMethods = isUserHub ? GetSupportedUserHubMethods() : GetSupportedMarketHubMethods();
        return Array.Exists(supportedMethods, m => string.Equals(m, methodName, StringComparison.OrdinalIgnoreCase));
    }
}