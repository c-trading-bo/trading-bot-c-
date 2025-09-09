using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http;
using Microsoft.Extensions.Logging;
using System;
using System.Net.Http;
using System.Net.Http.Headers;

namespace BotCore.Infrastructure
{
    /// <summary>
    /// HttpClient DI configuration and token handling review
    /// Addresses requirement: "HttpClient DI & token handling review - confirm single HttpClient registration via DI and robust JWT refresh without logging secrets"
    /// </summary>
    public static class HttpClientConfiguration
    {
        /// <summary>
        /// Configure HttpClient with proper DI registration and token handling
        /// </summary>
        public static IServiceCollection AddTopstepXHttpClient(this IServiceCollection services)
        {
            // Register named HttpClient for TopstepX API with proper configuration
            services.AddHttpClient("TopstepX", client =>
            {
                var apiBase = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
                client.BaseAddress = new Uri(apiBase);
                client.DefaultRequestHeaders.Accept.Clear();
                client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                client.DefaultRequestHeaders.UserAgent.Add(new ProductInfoHeaderValue("TradingBot", "1.0"));
                client.Timeout = TimeSpan.FromSeconds(30);
            })
            .ConfigurePrimaryHttpMessageHandler(() => new HttpClientHandler());
            
            // Register token handler as singleton for JWT management
            services.AddSingleton<ITopstepXTokenHandler, TopstepXTokenHandler>();
            
            // Register factory for creating authenticated clients
            services.AddSingleton<ITopstepXHttpClientFactory, TopstepXHttpClientFactory>();

            return services;
        }
    }

    /// <summary>
    /// Token handler interface for JWT management
    /// </summary>
    public interface ITopstepXTokenHandler
    {
        Task<string?> GetValidTokenAsync(CancellationToken cancellationToken = default);
        Task RefreshTokenAsync(CancellationToken cancellationToken = default);
        void ClearToken();
    }

    /// <summary>
    /// Secure JWT token handler that doesn't log secrets
    /// </summary>
    public class TopstepXTokenHandler : ITopstepXTokenHandler
    {
        private readonly ILogger<TopstepXTokenHandler> _logger;
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly SemaphoreSlim _refreshSemaphore = new(1, 1);
        private string? _currentToken;
        private DateTime _tokenExpiry = DateTime.MinValue;
        
        public TopstepXTokenHandler(ILogger<TopstepXTokenHandler> logger, IHttpClientFactory httpClientFactory)
        {
            _logger = logger;
            _httpClientFactory = httpClientFactory;
        }

        public async Task<string?> GetValidTokenAsync(CancellationToken cancellationToken = default)
        {
            // Check if current token is still valid (with 5 minute buffer)
            if (!string.IsNullOrEmpty(_currentToken) && DateTime.UtcNow < _tokenExpiry.AddMinutes(-5))
            {
                return _currentToken;
            }

            // Refresh token if needed
            await RefreshTokenAsync(cancellationToken);
            return _currentToken;
        }

        public async Task RefreshTokenAsync(CancellationToken cancellationToken = default)
        {
            await _refreshSemaphore.WaitAsync(cancellationToken);
            try
            {
                // Double-check pattern - another thread might have refreshed the token
                if (!string.IsNullOrEmpty(_currentToken) && DateTime.UtcNow < _tokenExpiry.AddMinutes(-5))
                {
                    return;
                }

                _logger.LogInformation("Refreshing TopstepX authentication token");

                var username = Environment.GetEnvironmentVariable("TOPSTEPX_USERNAME");
                var apiKey = Environment.GetEnvironmentVariable("TOPSTEPX_API_KEY");

                if (string.IsNullOrEmpty(username) || string.IsNullOrEmpty(apiKey))
                {
                    _logger.LogError("Missing TopstepX credentials in environment variables");
                    throw new InvalidOperationException("TopstepX credentials not configured");
                }

                using var client = _httpClientFactory.CreateClient("TopstepX");
                
                var loginRequest = new
                {
                    username = username,
                    apiKey = apiKey // Note: We use apiKey, not password - safer for API authentication
                };

                var response = await client.PostAsJsonAsync("/auth/login", loginRequest, cancellationToken);
                
                if (response.IsSuccessStatusCode)
                {
                    var responseContent = await response.Content.ReadAsStringAsync(cancellationToken);
                    
                    // Parse JWT token from response (implementation depends on API response format)
                    var tokenResponse = System.Text.Json.JsonSerializer.Deserialize<TokenResponse>(responseContent);
                    
                    if (tokenResponse?.Token != null)
                    {
                        _currentToken = tokenResponse.Token;
                        _tokenExpiry = tokenResponse.ExpiresAt ?? DateTime.UtcNow.AddHours(1); // Default 1 hour if not specified
                        
                        // Log success without exposing token
                        _logger.LogInformation("Token refresh successful, expires at {ExpiryTime}", _tokenExpiry);
                    }
                    else
                    {
                        _logger.LogError("Token refresh failed: Invalid response format");
                        throw new InvalidOperationException("Invalid token response format");
                    }
                }
                else
                {
                    // Log error without exposing credentials
                    _logger.LogError("Token refresh failed: {StatusCode} {ReasonPhrase}", 
                        response.StatusCode, response.ReasonPhrase);
                    throw new HttpRequestException($"Token refresh failed: {response.StatusCode}");
                }
            }
            finally
            {
                _refreshSemaphore.Release();
            }
        }

        public void ClearToken()
        {
            _currentToken = null;
            _tokenExpiry = DateTime.MinValue;
            _logger.LogInformation("Token cleared");
        }

        private class TokenResponse
        {
            public string? Token { get; set; }
            public DateTime? ExpiresAt { get; set; }
        }
    }

    /// <summary>
    /// Factory for creating authenticated HttpClient instances
    /// </summary>
    public interface ITopstepXHttpClientFactory
    {
        Task<HttpClient> CreateAuthenticatedClientAsync(CancellationToken cancellationToken = default);
    }

    /// <summary>
    /// Factory implementation that creates HttpClient instances with automatic token injection
    /// </summary>
    public class TopstepXHttpClientFactory : ITopstepXHttpClientFactory
    {
        private readonly IHttpClientFactory _httpClientFactory;
        private readonly ITopstepXTokenHandler _tokenHandler;
        private readonly ILogger<TopstepXHttpClientFactory> _logger;

        public TopstepXHttpClientFactory(
            IHttpClientFactory httpClientFactory, 
            ITopstepXTokenHandler tokenHandler,
            ILogger<TopstepXHttpClientFactory> logger)
        {
            _httpClientFactory = httpClientFactory;
            _tokenHandler = tokenHandler;
            _logger = logger;
        }

        public async Task<HttpClient> CreateAuthenticatedClientAsync(CancellationToken cancellationToken = default)
        {
            var client = _httpClientFactory.CreateClient("TopstepX");
            
            var token = await _tokenHandler.GetValidTokenAsync(cancellationToken);
            if (!string.IsNullOrEmpty(token))
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
                _logger.LogDebug("HttpClient configured with valid authentication token");
            }
            else
            {
                _logger.LogWarning("No valid authentication token available for HttpClient");
            }

            return client;
        }
    }
}