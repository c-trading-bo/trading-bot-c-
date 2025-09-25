using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore.Services
{
    /// <summary>
    /// Complete TopstepX API client with full error handling and retry logic
    /// No null returns - all methods throw meaningful exceptions
    /// </summary>
    public class ProductionTopstepXApiClient
    {
        private readonly HttpClient _httpClient;
        private readonly ILogger<ProductionTopstepXApiClient> _logger;
        private readonly string _baseUrl = "https://api.topstepx.com";

        public ProductionTopstepXApiClient(HttpClient httpClient, ILogger<ProductionTopstepXApiClient> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
            
            // Configure HttpClient for production use
            _httpClient.BaseAddress = new Uri(_baseUrl);
            _httpClient.Timeout = TimeSpan.FromSeconds(30);
            _httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot-Production/1.0");
        }

        /// <summary>
        /// Get account information with comprehensive error handling
        /// </summary>
        public Task<JsonElement> GetAccountAsync(string accountId, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));

            var endpoint = $"/accounts/{accountId}";
            return ExecuteWithRetryAsync(endpoint, cancellationToken);
        }

        /// <summary>
        /// Get account balance with proper error handling
        /// </summary>
        public Task<JsonElement> GetAccountBalanceAsync(string accountId, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));

            var endpoint = $"/accounts/{accountId}/balance";
            return ExecuteWithRetryAsync(endpoint, cancellationToken);
        }

        /// <summary>
        /// Get positions with comprehensive error handling
        /// </summary>
        public Task<JsonElement> GetPositionsAsync(string accountId, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));

            var endpoint = $"/accounts/{accountId}/positions";
            return ExecuteWithRetryAsync(endpoint, cancellationToken);
        }

        /// <summary>
        /// Get orders with full error handling
        /// </summary>
        public Task<JsonElement> GetOrdersAsync(string accountId, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));

            var endpoint = $"/accounts/{accountId}/orders";
            return ExecuteWithRetryAsync(endpoint, cancellationToken);
        }

        /// <summary>
        /// Get trades with comprehensive error handling
        /// </summary>
        public Task<JsonElement> GetTradesAsync(string accountId, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));

            var endpoint = $"/accounts/{accountId}/trades";
            return ExecuteWithRetryAsync(endpoint, cancellationToken);
        }

        /// <summary>
        /// Place order with full error handling and validation
        /// </summary>
        public Task<JsonElement> PlaceOrderAsync(string accountId, object orderRequest, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));
            
            if (orderRequest == null)
                throw new ArgumentException("Order request cannot be null", nameof(orderRequest));

            var endpoint = $"/accounts/{accountId}/orders";
            return ExecutePostWithRetryAsync(endpoint, orderRequest, cancellationToken);
        }

        /// <summary>
        /// Cancel order with proper error handling
        /// </summary>
        public Task<JsonElement> CancelOrderAsync(string accountId, string orderId, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrEmpty(accountId))
                throw new ArgumentException("Account ID cannot be null or empty", nameof(accountId));
            
            if (string.IsNullOrEmpty(orderId))
                throw new ArgumentException("Order ID cannot be null or empty", nameof(orderId));

            var endpoint = $"/accounts/{accountId}/orders/{orderId}";
            return ExecuteDeleteWithRetryAsync(endpoint, cancellationToken);
        }

        /// <summary>
        /// Execute HTTP request with exponential backoff retry logic
        /// </summary>
        private async Task<JsonElement> ExecuteWithRetryAsync(string endpoint, CancellationToken cancellationToken)
        {
            const int maxRetries = 3;
            var baseDelay = TimeSpan.FromSeconds(1);

            for (int attempt = 1; attempt <= maxRetries; attempt++)
            {
                try
                {
                    _logger.LogDebug("[API-CLIENT] Executing GET request to {Endpoint}, Attempt {Attempt}/{MaxRetries}",
                        endpoint, attempt, maxRetries);

                    using var response = await _httpClient.GetAsync(endpoint, cancellationToken).ConfigureAwait(false);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                        var jsonElement = JsonSerializer.Deserialize<JsonElement>(content);
                        
                        _logger.LogDebug("[API-CLIENT] GET request to {Endpoint} succeeded", endpoint);
                        return jsonElement;
                    }

                    // Handle specific HTTP status codes
                    var errorContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                    await HandleHttpErrorAsync(response.StatusCode, errorContent, endpoint, attempt, maxRetries).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    _logger.LogWarning("[API-CLIENT] Request to {Endpoint} was cancelled", endpoint);
                    throw;
                }
                catch (HttpRequestException ex)
                {
                    _logger.LogError(ex, "[API-CLIENT] HTTP error on request to {Endpoint}, Attempt {Attempt}/{MaxRetries}",
                        endpoint, attempt, maxRetries);
                    
                    if (attempt == maxRetries)
                        throw new HttpRequestException($"HTTP request to {endpoint} failed after {maxRetries} attempts", ex);
                }
                catch (TaskCanceledException ex) when (!cancellationToken.IsCancellationRequested)
                {
                    _logger.LogError(ex, "[API-CLIENT] Request timeout on {Endpoint}, Attempt {Attempt}/{MaxRetries}",
                        endpoint, attempt, maxRetries);
                    
                    if (attempt == maxRetries)
                        throw new TimeoutException($"Request to {endpoint} timed out after {maxRetries} attempts", ex);
                }

                // Exponential backoff with jitter
                if (attempt < maxRetries)
                {
                    var delay = TimeSpan.FromMilliseconds(baseDelay.TotalMilliseconds * Math.Pow(2, attempt - 1));
                    var jitter = TimeSpan.FromMilliseconds(Random.Shared.Next(0, 1000));
                    var totalDelay = delay + jitter;

                    _logger.LogInformation("[API-CLIENT] Retrying request to {Endpoint} in {DelayMs}ms",
                        endpoint, totalDelay.TotalMilliseconds);
                    
                    await Task.Delay(totalDelay, cancellationToken).ConfigureAwait(false);
                }
            }

            throw new InvalidOperationException($"Request to {endpoint} failed after {maxRetries} attempts");
        }

        /// <summary>
        /// Execute POST request with retry logic
        /// </summary>
        private async Task<JsonElement> ExecutePostWithRetryAsync(string endpoint, object requestData, CancellationToken cancellationToken)
        {
            var json = JsonSerializer.Serialize(requestData);
            var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

            const int maxRetries = 3;
            var baseDelay = TimeSpan.FromSeconds(1);

            for (int attempt = 1; attempt <= maxRetries; attempt++)
            {
                try
                {
                    _logger.LogDebug("[API-CLIENT] Executing POST request to {Endpoint}, Attempt {Attempt}/{MaxRetries}",
                        endpoint, attempt, maxRetries);

                    using var response = await _httpClient.PostAsync(endpoint, content, cancellationToken).ConfigureAwait(false);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var responseContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                        var jsonElement = JsonSerializer.Deserialize<JsonElement>(responseContent);
                        
                        _logger.LogInformation("[API-CLIENT] POST request to {Endpoint} succeeded", endpoint);
                        return jsonElement;
                    }

                    var errorContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                    await HandleHttpErrorAsync(response.StatusCode, errorContent, endpoint, attempt, maxRetries).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    _logger.LogWarning("[API-CLIENT] POST request to {Endpoint} was cancelled", endpoint);
                    throw;
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "[API-CLIENT] Error on POST request to {Endpoint}, Attempt {Attempt}/{MaxRetries}",
                        endpoint, attempt, maxRetries);
                    
                    if (attempt == maxRetries)
                        throw;
                }

                if (attempt < maxRetries)
                {
                    var delay = TimeSpan.FromMilliseconds(baseDelay.TotalMilliseconds * Math.Pow(2, attempt - 1));
                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }
            }

            throw new InvalidOperationException($"POST request to {endpoint} failed after {maxRetries} attempts");
        }

        /// <summary>
        /// Execute DELETE request with retry logic
        /// </summary>
        private async Task<JsonElement> ExecuteDeleteWithRetryAsync(string endpoint, CancellationToken cancellationToken)
        {
            const int maxRetries = 3;
            var baseDelay = TimeSpan.FromSeconds(1);

            for (int attempt = 1; attempt <= maxRetries; attempt++)
            {
                try
                {
                    using var response = await _httpClient.DeleteAsync(endpoint, cancellationToken).ConfigureAwait(false);
                    
                    if (response.IsSuccessStatusCode)
                    {
                        var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                        var jsonElement = string.IsNullOrEmpty(content) 
                            ? JsonSerializer.Deserialize<JsonElement>("{}") 
                            : JsonSerializer.Deserialize<JsonElement>(content);
                        
                        _logger.LogInformation("[API-CLIENT] DELETE request to {Endpoint} succeeded", endpoint);
                        return jsonElement;
                    }

                    var errorContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                    await HandleHttpErrorAsync(response.StatusCode, errorContent, endpoint, attempt, maxRetries).ConfigureAwait(false);
                }
                catch (Exception ex) when (attempt < maxRetries)
                {
                    _logger.LogWarning(ex, "[API-CLIENT] DELETE request to {Endpoint} failed, retrying...", endpoint);
                    await Task.Delay(baseDelay, cancellationToken).ConfigureAwait(false);
                }
            }

            throw new InvalidOperationException($"DELETE request to {endpoint} failed after {maxRetries} attempts");
        }

        /// <summary>
        /// Handle HTTP error responses with specific exception types
        /// </summary>
        private async Task HandleHttpErrorAsync(System.Net.HttpStatusCode statusCode, string errorContent, string endpoint, int attempt, int maxRetries)
        {
            _logger.LogWarning("[API-CLIENT] HTTP {StatusCode} error on {Endpoint}, Attempt {Attempt}/{MaxRetries}: {ErrorContent}",
                (int)statusCode, endpoint, attempt, maxRetries, errorContent);

            // Only retry on specific status codes
            var shouldRetry = statusCode switch
            {
                System.Net.HttpStatusCode.InternalServerError => true,
                System.Net.HttpStatusCode.BadGateway => true,
                System.Net.HttpStatusCode.ServiceUnavailable => true,
                System.Net.HttpStatusCode.GatewayTimeout => true,
                System.Net.HttpStatusCode.RequestTimeout => true,
                System.Net.HttpStatusCode.TooManyRequests => true,
                _ => false
            };

            if (!shouldRetry || attempt >= maxRetries)
            {
                // Throw specific exceptions based on status code
                Exception exception = statusCode switch
                {
                    System.Net.HttpStatusCode.Unauthorized => new UnauthorizedAccessException($"Authentication failed for {endpoint}: {errorContent}"),
                    System.Net.HttpStatusCode.Forbidden => new InvalidOperationException($"Access forbidden for {endpoint}: {errorContent}"),
                    System.Net.HttpStatusCode.NotFound => new InvalidOperationException($"Resource not found at {endpoint}: {errorContent}"),
                    System.Net.HttpStatusCode.TooManyRequests => new InvalidOperationException($"Rate limit exceeded for {endpoint}: {errorContent}"),
                    _ => new HttpRequestException($"HTTP {(int)statusCode} error at {endpoint}: {errorContent}")
                };

                throw exception;
            }

            await Task.Yield(); // Allow retry logic to continue
        }
    }
}