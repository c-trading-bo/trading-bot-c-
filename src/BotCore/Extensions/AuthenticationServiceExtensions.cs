using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;
using BotCore.Services;
using TopstepX.Bot.Authentication;

namespace BotCore.Extensions;

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
                        var logger = serviceProvider.GetRequiredService<ILogger<CachedTopstepAuth>>();
                        logger.LogError(ex, "Failed to authenticate with TopstepX API");
                        throw;
                    }
                }
                
                throw new InvalidOperationException("No authentication credentials available (TOPSTEPX_JWT or TOPSTEPX_USERNAME/TOPSTEPX_API_KEY)");
            });
        });

        // Register cached auth service as singleton
        services.AddSingleton<ITopstepAuth, CachedTopstepAuth>();

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

        // Use custom auth provider
        services.AddSingleton(customAuthProvider);
        services.AddSingleton<ITopstepAuth, CachedTopstepAuth>();

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