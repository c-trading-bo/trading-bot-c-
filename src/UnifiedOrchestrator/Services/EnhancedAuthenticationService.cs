using System;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Hosting;
using TradingBot.Abstractions;
using TopstepX.Bot.Authentication;
using System.Globalization;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Enhanced authentication service with comprehensive logging
/// Tracks authentication attempts, JWT token refresh, account information retrieval
/// </summary>
public class EnhancedAuthenticationService : IHostedService
{
    private readonly ILogger<EnhancedAuthenticationService> _logger;
    private readonly ITradingLogger _tradingLogger;
    private readonly ITokenProvider _tokenProvider;
    private readonly /* Legacy removed: TopstepXCredentialManager */object _credentialManager;
    private readonly TopstepAuthAgent _authAgent;
    
    public EnhancedAuthenticationService(
        ILogger<EnhancedAuthenticationService> logger,
        ITradingLogger tradingLogger,
        ITokenProvider tokenProvider,
        /* Legacy removed: TopstepXCredentialManager */object credentialManager,
        TopstepAuthAgent authAgent)
    {
        _logger = logger;
        _tradingLogger = tradingLogger;
        _tokenProvider = tokenProvider;
        _credentialManager = credentialManager;
        _authAgent = authAgent;
        
        // Subscribe to token refresh events
        _tokenProvider.TokenRefreshed += OnTokenRefreshed;
    }

    public async Task StartAsync(CancellationToken cancellationToken)
    {
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService", 
            "Enhanced authentication service started").ConfigureAwait(false);

        // Log credential discovery
        var credentialDiscovery = _credentialManager.DiscoverAllCredentialSources();
        
        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService", 
            "Credential discovery completed", new
            {
                totalSourcesFound = credentialDiscovery.TotalSourcesFound,
                hasEnvironmentCredentials = credentialDiscovery.HasEnvironmentCredentials,
                hasFileCredentials = credentialDiscovery.HasFileCredentials,
                hasAnyCredentials = credentialDiscovery.HasAnyCredentials,
                recommendedSource = credentialDiscovery.RecommendedSource
            }).ConfigureAwait(false);

        if (!credentialDiscovery.HasAnyCredentials)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "AuthService", 
                "No TopstepX credentials found - authentication will fail").ConfigureAwait(false);
        }

        // Attempt initial authentication
        await AttemptAuthenticationAsync().ConfigureAwait(false);
    }

    public Task StopAsync(CancellationToken cancellationToken)
    {
        return _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService",
            "Enhanced authentication service stopped");
    }

    private async Task AttemptAuthenticationAsync()
    {
        try
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService", 
                "Starting authentication attempt").ConfigureAwait(false);

            var startTime = DateTime.UtcNow;
            
            // Try to get account information to verify authentication
            var token = await _tokenProvider.GetTokenAsync().ConfigureAwait(false);
            if (string.IsNullOrEmpty(token))
            {
                await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "AuthService", 
                    "Authentication failed - no valid JWT token available").ConfigureAwait(false);
                return;
            }

            var authDuration = DateTime.UtcNow - startTime;
            
            await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService", 
                "Authentication successful", new
                {
                    durationMs = authDuration.TotalMilliseconds,
                    tokenLength = token.Length,
                    hasToken = !string.IsNullOrEmpty(token)
                }).ConfigureAwait(false);

            // Log JWT token timing information (without logging the actual token)
            await LogJwtTokenInfo(token).ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            await _tradingLogger.LogSystemAsync(TradingLogLevel.ERROR, "AuthService", 
                $"Authentication attempt failed: {ex.Message}", new
                {
                    exceptionType = ex.GetType().Name,
                    stackTrace = ex.StackTrace
                }).ConfigureAwait(false);
        }
    }

    private async Task LogJwtTokenInfo(string token)
    {
        try
        {
            // Parse JWT token to get expiration info (basic parsing, not full verification)
            var parts = token.Split('.');
            if (parts.Length >= 2)
            {
                var payload = parts[1];
                // Add padding if necessary for Base64 decoding
                switch (payload.Length % 4)
                {
                    case 2: payload += "=="; break;
                    case 3: payload += "="; break;
                }
                
                var payloadBytes = Convert.FromBase64String(payload);
                var payloadJson = System.Text.Encoding.UTF8.GetString(payloadBytes);
                
                if (payloadJson.Contains("exp"))
                {
                    var expMatch = System.Text.RegularExpressions.Regex.Match(payloadJson, @"""exp"":(\d+)");
                    if (expMatch.Success && long.TryParse(expMatch.Groups[1].Value, out var expUnix))
                    {
                        var expiration = DateTimeOffset.FromUnixTimeSeconds(expUnix).DateTime;
                        var timeToExpiry = expiration - DateTime.UtcNow;
                        
                        await _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService", 
                            "JWT token timing analysis", new
                            {
                                expiresAt = expiration.ToString("yyyy-MM-dd HH:mm:ss UTC", CultureInfo.InvariantCulture),
                                timeToExpiryMinutes = Math.Round(timeToExpiry.TotalMinutes, 1),
                                isExpiringSoon = timeToExpiry.TotalMinutes < 30
                            }).ConfigureAwait(false);

                        if (timeToExpiry.TotalMinutes < 30)
                        {
                            await _tradingLogger.LogSystemAsync(TradingLogLevel.WARN, "AuthService", 
                                "JWT token expiring soon - refresh recommended").ConfigureAwait(false);
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Could not parse JWT token for timing info");
        }
    }

    private Task OnTokenRefreshed(string newToken)
    {
        return _tradingLogger.LogSystemAsync(TradingLogLevel.INFO, "AuthService",
            "JWT token refreshed successfully", new
            {
                tokenLength = newToken.Length,
                refreshTime = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss UTC", CultureInfo.InvariantCulture)
            });
    }
}