using System;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TradingBot.Abstractions;

namespace BotCore.Services;

/// <summary>
/// Hardened HTTP client service with exponential backoff and proper error handling
/// Implements centralized base address, auth, retries with jitter, and JSON tolerance
/// </summary>
public interface ITopstepXHttpClient
{
    Task<HttpResponseMessage> GetAsync(string requestUri, CancellationToken cancellationToken = default);
    Task<HttpResponseMessage> PostAsync(string requestUri, HttpContent? content, CancellationToken cancellationToken = default);
    Task<T?> GetJsonAsync<T>(string requestUri, CancellationToken cancellationToken = default) where T : class;
    Task<T?> PostJsonAsync<T>(string requestUri, object? content, CancellationToken cancellationToken = default) where T : class;
}

public class TopstepXHttpClient : ITopstepXHttpClient, IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly ILogger<TopstepXHttpClient> _logger;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly Random _jitterRandom = new();
    private readonly ITopstepAuth? _authService;

    public TopstepXHttpClient(HttpClient httpClient, ILogger<TopstepXHttpClient> logger, ITopstepAuth? authService = null)
    {
        _httpClient = httpClient;
        _logger = logger;
        _authService = authService;

        // Set up HTTP base address centrally
        var baseAddress = Environment.GetEnvironmentVariable("TOPSTEPX_API_BASE") ?? "https://api.topstepx.com";
        _httpClient.BaseAddress = new Uri(baseAddress);

        // Set up default headers
        _httpClient.DefaultRequestHeaders.Add("User-Agent", "TradingBot/1.0");

        // JSON tolerance with snake_case policy 
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = true,
            PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            UnmappedMemberHandling = JsonUnmappedMemberHandling.Skip, // Ignore unknown fields
            ReadCommentHandling = JsonCommentHandling.Skip,
            AllowTrailingCommas = true
        };

        _logger.LogInformation("TopstepXHttpClient initialized with base address: {BaseAddress}, AuthService: {HasAuth}", 
            baseAddress, _authService != null);
    }

    public async Task<HttpResponseMessage> GetAsync(string requestUri, CancellationToken cancellationToken = default)
    {
        await EnsureFreshTokenAsync(cancellationToken).ConfigureAwait(false);
        return await ExecuteWithRetryAsync(
            () => _httpClient.GetAsync(requestUri, cancellationToken),
            "GET",
            requestUri,
            cancellationToken
        ).ConfigureAwait(false);
    }

    public async Task<HttpResponseMessage> PostAsync(string requestUri, HttpContent? content, CancellationToken cancellationToken = default)
    {
        await EnsureFreshTokenAsync(cancellationToken).ConfigureAwait(false);
        return await ExecuteWithRetryAsync(
            () => _httpClient.PostAsync(requestUri, content, cancellationToken),
            "POST",
            requestUri,
            cancellationToken
        ).ConfigureAwait(false);
    }

    public async Task<T?> GetJsonAsync<T>(string requestUri, CancellationToken cancellationToken = default) where T : class
    {
        var response = await GetAsync(requestUri, cancellationToken).ConfigureAwait(false);
        return await DeserializeResponseAsync<T>(response, cancellationToken).ConfigureAwait(false);
    }

    public async Task<T?> PostJsonAsync<T>(string requestUri, object? content, CancellationToken cancellationToken = default) where T : class
    {
        HttpContent? httpContent = null;
        if (content != null)
        {
            var json = JsonSerializer.Serialize(content, _jsonOptions);
            httpContent = new StringContent(json, Encoding.UTF8, "application/json");
        }

        var response = await PostAsync(requestUri, httpContent, cancellationToken).ConfigureAwait(false);
        return await DeserializeResponseAsync<T>(response, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>
    /// Ensure fresh JWT token before making requests
    /// </summary>
    private async Task EnsureFreshTokenAsync(CancellationToken cancellationToken)
    {
        if (_authService == null)
        {
            // Fall back to environment variable if no auth service
            await SetAuthHeaderFromEnvironmentAsync().ConfigureAwait(false);
            return;
        }

        try
        {
            await _authService.EnsureFreshTokenAsync(cancellationToken).ConfigureAwait(false);
            var (jwt, _) = await _authService.GetFreshJwtAsync(cancellationToken).ConfigureAwait(false);
            
            if (!string.IsNullOrEmpty(jwt))
            {
                _httpClient.DefaultRequestHeaders.Authorization = 
                    new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", jwt);
            }
        }
        catch (Exception ex)
        {
            var errorData = new
            {
                timestamp = DateTimeOffset.UtcNow,
                component = "topstepx_http_client",
                operation = "ensure_fresh_token_failed",
                error_type = ex.GetType().Name,
                sanitized_message = SanitizeErrorMessage(ex.Message)
            };

            _logger.LogError("Failed to ensure fresh token: {ErrorData}", JsonSerializer.Serialize(errorData));
            throw;
        }
    }

    /// <summary>
    /// Fallback method to set auth header from environment
    /// </summary>
    private async Task SetAuthHeaderFromEnvironmentAsync()
    {
        var token = GetAuthToken();
        if (!string.IsNullOrEmpty(token))
        {
            _httpClient.DefaultRequestHeaders.Authorization = 
                new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
        }
        await Task.CompletedTask.ConfigureAwait(false);
    }

    /// <summary>
    /// Execute HTTP request with exponential backoff + jitter for 5xx/408 errors only
    /// </summary>
    private async Task<HttpResponseMessage> ExecuteWithRetryAsync(
        Func<Task<HttpResponseMessage>> operation,
        string method,
        string requestUri,
        CancellationToken cancellationToken)
    {
        const int maxRetries = 3;
        
        for (int attempt = 0; attempt <= maxRetries; attempt++)
        {
            try
            {
                var response = await operation().ConfigureAwait(false);

                // Only retry on 5xx server errors or 408 timeout
                if (ShouldRetry(response.StatusCode) && attempt < maxRetries)
                {
                    var delay = CalculateBackoffDelay(attempt);
                    
                    var retryData = new
                    {
                        timestamp = DateTime.UtcNow,
                        component = "topstepx_http_client",
                        operation = "http_retry",
                        method = method,
                        uri = RedactSensitiveInfo(requestUri),
                        attempt = attempt + 1,
                        max_attempts = maxRetries + 1,
                        status_code = (int)response.StatusCode,
                        delay_ms = delay.TotalMilliseconds
                    };

                    _logger.LogWarning("HTTP_RETRY: {RetryData}", JsonSerializer.Serialize(retryData));

                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                    response.Dispose(); // Clean up the failed response
                    continue;
                }

                // Success or non-retryable error
                return response;
            }
            catch (HttpRequestException ex) when (attempt < maxRetries)
            {
                var delay = CalculateBackoffDelay(attempt);
                
                var retryData = new
                {
                    timestamp = DateTime.UtcNow,
                    component = "topstepx_http_client",
                    operation = "http_retry_exception",
                    method = method,
                    uri = RedactSensitiveInfo(requestUri),
                    attempt = attempt + 1,
                    max_attempts = maxRetries + 1,
                    error_type = ex.GetType().Name,
                    delay_ms = delay.TotalMilliseconds
                };

                _logger.LogWarning("HTTP_RETRY_EXCEPTION: {RetryData}", JsonSerializer.Serialize(retryData));

                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
            catch (TaskCanceledException ex) when (ex.InnerException is TimeoutException && attempt < maxRetries)
            {
                var delay = CalculateBackoffDelay(attempt);
                
                var retryData = new
                {
                    timestamp = DateTime.UtcNow,
                    component = "topstepx_http_client",
                    operation = "http_retry_timeout",
                    method = method,
                    uri = RedactSensitiveInfo(requestUri),
                    attempt = attempt + 1,
                    max_attempts = maxRetries + 1,
                    delay_ms = delay.TotalMilliseconds
                };

                _logger.LogWarning("HTTP_RETRY_TIMEOUT: {RetryData}", JsonSerializer.Serialize(retryData));

                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
        }

        // If we get here, all retries failed
        throw new HttpRequestException($"HTTP request failed after {maxRetries + 1} attempts");
    }

    /// <summary>
    /// Determine if HTTP status code should trigger a retry (5xx/408 only)
    /// </summary>
    private static bool ShouldRetry(HttpStatusCode statusCode)
    {
        return statusCode == HttpStatusCode.RequestTimeout || // 408
               statusCode == HttpStatusCode.InternalServerError || // 500
               statusCode == HttpStatusCode.BadGateway || // 502
               statusCode == HttpStatusCode.ServiceUnavailable || // 503
               statusCode == HttpStatusCode.GatewayTimeout; // 504
    }

    /// <summary>
    /// Calculate exponential backoff delay with jitter
    /// </summary>
    private TimeSpan CalculateBackoffDelay(int attempt)
    {
        // Base delay: 1000ms * 2^attempt
        var baseDelay = TimeSpan.FromMilliseconds(1000 * Math.Pow(2, attempt));
        
        // Add jitter: ±25% of base delay
        var jitterMs = (int)(baseDelay.TotalMilliseconds * 0.25 * (_jitterRandom.NextDouble() * 2 - 1));
        var finalDelay = baseDelay.Add(TimeSpan.FromMilliseconds(jitterMs));
        
        // Cap at 30 seconds
        return finalDelay > TimeSpan.FromSeconds(30) ? TimeSpan.FromSeconds(30) : finalDelay;
    }

    /// <summary>
    /// Deserialize HTTP response with error handling
    /// </summary>
    private async Task<T?> DeserializeResponseAsync<T>(HttpResponseMessage response, CancellationToken cancellationToken) where T : class
    {
        try
        {
            if (!response.IsSuccessStatusCode)
            {
                var errorContent = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                _logger.LogWarning("HTTP request failed: {StatusCode} {ReasonPhrase} - {Content}", 
                    response.StatusCode, response.ReasonPhrase, RedactSensitiveInfo(errorContent));
                return null;
            }

            var json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            return JsonSerializer.Deserialize<T>(json, _jsonOptions);
        }
        catch (JsonException ex)
        {
            _logger.LogError(ex, "JSON deserialization failed for type {Type}", typeof(T).Name);
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Unexpected error deserializing response");
            return null;
        }
    }

    /// <summary>
    /// Get auth token from environment with token redaction in logs
    /// </summary>
    private string? GetAuthToken()
    {
        try
        {
            var token = Environment.GetEnvironmentVariable("TOPSTEPX_AUTH_TOKEN");
            if (!string.IsNullOrEmpty(token))
            {
                _logger.LogDebug("Auth token loaded (length: {Length})", token.Length);
                return token;
            }
            
            _logger.LogWarning("No auth token found in TOPSTEPX_AUTH_TOKEN environment variable");
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to load auth token");
            return null;
        }
    }

    /// <summary>
    /// Redact sensitive information from URLs and content for logging
    /// </summary>
    private static string RedactSensitiveInfo(string input)
    {
        if (string.IsNullOrEmpty(input))
            return input;

        // Redact common sensitive patterns
        var patterns = new[]
        {
            (@"(token|key|secret|password|auth)=([^&\s]+)", "$1=[REDACTED]"),
            (@"(bearer\s+)([a-zA-Z0-9\.\-_]+)", "$1[REDACTED]"),
            (@"(\d{4})(\d{4,})(\d{4})", "$1****$3"), // Account numbers
        };

        var result = input;
        foreach (var (pattern, replacement) in patterns)
        {
            result = System.Text.RegularExpressions.Regex.Replace(
                result, pattern, replacement, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        }

        return result;
    }

    /// <summary>
    /// Sanitize error messages to prevent token leakage
    /// </summary>
    private static string SanitizeErrorMessage(string message)
    {
        if (string.IsNullOrEmpty(message))
            return message;

        // Remove potential token/secret patterns from error messages
        var patterns = new[]
        {
            (@"(token|key|secret|password|auth)[=:]\s*[^\s,}]+", "$1=[REDACTED]"),
            (@"(bearer\s+)[a-zA-Z0-9\.\-_]+", "$1[REDACTED]"),
            (@"[a-zA-Z0-9]{50,}", "[LONG_STRING_REDACTED]") // Potential tokens
        };

        var result = message;
        foreach (var (pattern, replacement) in patterns)
        {
            result = System.Text.RegularExpressions.Regex.Replace(
                result, pattern, replacement, System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        }

        return result;
    }

    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}