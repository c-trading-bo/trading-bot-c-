using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.Abstractions;
using Infrastructure.TopstepX;
using BotCore.Services;

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
            return async (cancellationToken) =>
            {
                // This would typically call the actual TopstepAuthAgent
                // For now, fallback to environment variable
                var jwt = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (!string.IsNullOrEmpty(jwt))
                {
                    return jwt;
                }
                
                // In production, this would authenticate with username/password
                var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");
                
                if (!string.IsNullOrEmpty(username) && !string.IsNullOrEmpty(apiKey))
                {
                    // Would call actual authentication endpoint
                    var logger = serviceProvider.GetRequiredService<ILogger<CachedTopstepAuth>>();
                    logger.LogWarning("Auth function placeholder - implement actual authentication logic");
                    throw new NotImplementedException("Implement actual TopstepX authentication");
                }
                
                throw new InvalidOperationException("No authentication credentials available");
            };
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