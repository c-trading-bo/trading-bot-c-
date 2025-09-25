using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;
using BotCore.Services;
using BotCore.Auth;
using TopstepX.Bot.Authentication;

namespace BotCore.Extensions;

/// <summary>
/// Wrapper to make SimpleTopstepAuth implement ITopstepAuth interface
/// </summary>
internal sealed class SimpleTopstepAuthWrapper : ITopstepAuth, IAsyncDisposable
{
    private readonly Auth.SimpleTopstepAuth _auth;

    public SimpleTopstepAuthWrapper(HttpClient httpClient, ILogger<Auth.SimpleTopstepAuth> logger)
    {
        _auth = new Auth.SimpleTopstepAuth(httpClient, logger);
    }

    public Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct = default)
    {
        return _auth.GetFreshJwtAsync(ct);
    }

    public Task EnsureFreshTokenAsync(CancellationToken ct = default)
    {
        return _auth.EnsureFreshTokenAsync(ct);
    }

    public ValueTask DisposeAsync()
    {
        return _auth.DisposeAsync();
    }
}

/// <summary>
/// Wrapper for custom auth provider functions
/// </summary>
internal sealed class FuncTopstepAuthWrapper : ITopstepAuth
{
    private readonly Func<CancellationToken, Task<string>> _authProvider;
    private string _lastJwt = string.Empty;
    private DateTimeOffset _lastExpiry = DateTimeOffset.MinValue;

    public FuncTopstepAuthWrapper(Func<CancellationToken, Task<string>> authProvider)
    {
        _authProvider = authProvider ?? throw new ArgumentNullException(nameof(authProvider));
    }

    public async Task<(string jwt, DateTimeOffset expiresUtc)> GetFreshJwtAsync(CancellationToken ct = default)
    {
        var jwt = await _authProvider(ct).ConfigureAwait(false);
        var expiry = GetJwtExpiryUtc(jwt);
        
        _lastJwt = jwt;
        _lastExpiry = expiry;
        
        return (jwt, expiry);
    }

    public async Task EnsureFreshTokenAsync(CancellationToken ct = default)
    {
        await GetFreshJwtAsync(ct).ConfigureAwait(false);
    }

    private static DateTimeOffset GetJwtExpiryUtc(string jwt)
    {
        try
        {
            var parts = jwt.Split('.');
            if (parts.Length < 2) return DateTimeOffset.UtcNow.AddHours(1); // Default fallback
            
            string payloadJson = System.Text.Encoding.UTF8.GetString(Base64UrlDecode(parts[1]));
            using var doc = System.Text.Json.JsonDocument.Parse(payloadJson);
            if (doc.RootElement.TryGetProperty("exp", out var expProp))
            {
                long exp = expProp.GetInt64();
                return DateTimeOffset.FromUnixTimeSeconds(exp);
            }
        }
        catch
        {
            // Fallback on parse errors
        }
        return DateTimeOffset.UtcNow.AddHours(1);
    }

    private static byte[] Base64UrlDecode(string s)
    {
        s = s.Replace('-', '+').Replace('_', '/');
        switch (s.Length % 4) 
        { 
            case 2: s += "=="; break; 
            case 3: s += "="; break; 
        }
        return Convert.FromBase64String(s);
    }
}

/// <summary>
/// Extension methods for registering authentication and HTTP client services
/// Implements singleton pattern with proper DI registration
/// </summary>
public static class AuthenticationServiceExtensions
{
    /// <summary>
    /// Register authentication and HTTP client services as singletons
    /// </summary>
    public static IServiceCollection AddTopstepAuthentication(
        this IServiceCollection services, 
        IConfiguration configuration)
    {
        // Register HTTP client factory for proper socket management
        services.AddHttpClient("Topstep", httpClient =>
        {
            var baseAddress = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
            httpClient.BaseAddress = new Uri(baseAddress);
            httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot/1.0");
            httpClient.Timeout = TimeSpan.FromSeconds(30);
        });

        // Register auth service factory function
        services.AddSingleton<Func<CancellationToken, Task<string>>>(serviceProvider =>
        {
            return new Func<CancellationToken, Task<string>>(async (cancellationToken) =>
            {
                // First try environment variable for pre-existing JWT
                var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (!string.IsNullOrEmpty(jwt))
                {
                    return jwt;
                }
                
                // Use TopstepAuthAgent for username/apiKey authentication
                var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
                
                if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey))
                {
                    var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
                    var httpClient = httpClientFactory.CreateClient("Topstep");
                    var authAgent = new TopstepAuthAgent(httpClient);
                    
                    try
                    {
                        return await authAgent.GetJwtAsync(username, apiKey, cancellationToken).ConfigureAwait(false);
                    }
                    catch (Exception ex)
                    {
                        var logger = serviceProvider.GetRequiredService<ILogger<TopstepXHttpClient>>();
                        logger.LogError(ex, "Failed to authenticate with TopstepX API");
                        throw;
                    }
                }
                
                throw new InvalidOperationException("No authentication credentials available (TOPSTEPX_JWT or TOPSTEPX_USERNAME/TOPSTEPX_API_KEY)");
            });
        });

        // Register cached auth service as singleton using SimpleTopstepAuth
        services.AddSingleton<ITopstepAuth>(serviceProvider =>
        {
            var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
            var httpClient = httpClientFactory.CreateClient("Topstep");
            var logger = serviceProvider.GetRequiredService<ILogger<BotCore.Auth.SimpleTopstepAuth>>();
            
            // Create wrapper that implements ITopstepAuth interface
            return new SimpleTopstepAuthWrapper(httpClient, logger);
        });

        // Register HTTP client service as singleton using factory
        services.AddSingleton<ITopstepXHttpClient>(serviceProvider =>
        {
            var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
            var httpClient = httpClientFactory.CreateClient("Topstep");
            var logger = serviceProvider.GetRequiredService<ILogger<TopstepXHttpClient>>();
            var authService = serviceProvider.GetRequiredService<ITopstepAuth>();
            
            return new TopstepXHttpClient(httpClient, logger, authService);
        });

        return services;
    }

    /// <summary>
    /// Register authentication services with custom auth provider
    /// </summary>
    public static IServiceCollection AddTopstepAuthentication(
        this IServiceCollection services,
        IConfiguration configuration,
        Func<CancellationToken, Task<string>> customAuthProvider)
    {
        services.AddHttpClient("Topstep", httpClient =>
        {
            var baseAddress = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
            httpClient.BaseAddress = new Uri(baseAddress);
            httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot/1.0");
            httpClient.Timeout = TimeSpan.FromSeconds(30);
        });

        // Use custom auth provider with proper wrapper
        services.AddSingleton(customAuthProvider);
        services.AddSingleton<ITopstepAuth>(serviceProvider => 
        {
            var authProvider = serviceProvider.GetRequiredService<Func<CancellationToken, Task<string>>>();
            return new FuncTopstepAuthWrapper(authProvider);
        });

        services.AddSingleton<ITopstepXHttpClient>(serviceProvider =>
        {
            var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
            var httpClient = httpClientFactory.CreateClient("Topstep");
            var logger = serviceProvider.GetRequiredService<ILogger<TopstepXHttpClient>>();
            var authService = serviceProvider.GetRequiredService<ITopstepAuth>();
            
            return new TopstepXHttpClient(httpClient, logger, authService);
        });

        return services;
    }

    /// <summary>
    /// Register authentication services without DI (legacy support)
    /// </summary>
    public static IServiceCollection AddTopstepAuthenticationLegacy(
        this IServiceCollection services,
        IConfiguration configuration)
    {
        // Register without auth service integration for backward compatibility
        services.AddHttpClient("Topstep", httpClient =>
        {
            var baseAddress = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
            httpClient.BaseAddress = new Uri(baseAddress);
            httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot/1.0");
            httpClient.Timeout = TimeSpan.FromSeconds(30);
        });

        services.AddSingleton<ITopstepXHttpClient>(serviceProvider =>
        {
            var httpClientFactory = serviceProvider.GetRequiredService<IHttpClientFactory>();
            var httpClient = httpClientFactory.CreateClient("Topstep");
            var logger = serviceProvider.GetRequiredService<ILogger<TopstepXHttpClient>>();
            
            return new TopstepXHttpClient(httpClient, logger, null);
        });

        return services;
    }
}