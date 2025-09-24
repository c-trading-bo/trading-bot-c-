using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using System.Net;
using System.Net.Http;
using System.Net.Sockets;
using System.Text.Json;

namespace BotCore.Services;

/// <summary>
/// Production-grade resilience service implementing retry logic, circuit breakers, and graceful degradation
/// Essential for reliable financial trading operations
/// </summary>
public class ProductionResilienceService
{
    private readonly ILogger<ProductionResilienceService> _logger;
    private readonly ResilienceConfig _config;
    private readonly Dictionary<string, CircuitBreakerState> _circuitBreakers = new();

    public ProductionResilienceService(ILogger<ProductionResilienceService> logger, IOptions<ResilienceConfig> config)
    {
        _logger = logger;
        _config = config.Value;
    }

    /// <summary>
    /// Execute operation with retry logic and circuit breaker protection
    /// </summary>
    public async Task<T> ExecuteWithResilienceAsync<T>(
        string operationName,
        Func<CancellationToken, Task<T>> operation,
        CancellationToken cancellationToken = default)
    {
        var circuitBreaker = GetOrCreateCircuitBreaker(operationName);
        
        // Check circuit breaker state
        if (circuitBreaker.State == CircuitState.Open)
        {
            if (DateTime.UtcNow - circuitBreaker.LastFailure < _config.CircuitBreakerTimeout)
            {
                _logger.LogWarning("🚫 [RESILIENCE] Circuit breaker OPEN for {Operation}, falling back to default", operationName);
                throw new CircuitBreakerOpenException($"Circuit breaker is open for {operationName}");
            }
            else
            {
                // Try half-open state
                circuitBreaker.State = CircuitState.HalfOpen;
                _logger.LogInformation("🔄 [RESILIENCE] Circuit breaker HALF-OPEN for {Operation}, attempting recovery", operationName);
            }
        }

        var lastException = (Exception?)null;
        
        for (int attempt = 1; attempt <= _config.MaxRetries; attempt++)
        {
            try
            {
                cancellationToken.ThrowIfCancellationRequested();
                
                var result = await operation(cancellationToken).ConfigureAwait(false);
                
                // Success - reset circuit breaker
                if (circuitBreaker.State == CircuitState.HalfOpen)
                {
                    circuitBreaker.State = CircuitState.Closed;
                    circuitBreaker.FailureCount;
                    _logger.LogInformation("✅ [RESILIENCE] Circuit breaker CLOSED for {Operation} - recovery successful", operationName);
                }
                
                if (attempt > 1)
                {
                    _logger.LogInformation("✅ [RESILIENCE] Operation {Operation} succeeded on attempt {Attempt}", operationName, attempt);
                }
                
                return result;
            }
            catch (Exception ex) when (IsRetriableException(ex))
            {
                lastException = ex;
                circuitBreaker.FailureCount++;
                
                _logger.LogWarning(ex, "⚠️ [RESILIENCE] Operation {Operation} failed on attempt {Attempt}/{MaxAttempts}: {Error}", 
                    operationName, attempt, _config.MaxRetries, ex.Message);
                
                // Check if we should open the circuit breaker
                if (circuitBreaker.FailureCount >= _config.CircuitBreakerThreshold)
                {
                    circuitBreaker.State = CircuitState.Open;
                    circuitBreaker.LastFailure = DateTime.UtcNow;
                    _logger.LogError("🚫 [RESILIENCE] Circuit breaker OPENED for {Operation} after {Failures} failures", 
                        operationName, circuitBreaker.FailureCount);
                }
                
                if (attempt < _config.MaxRetries)
                {
                    var delay = CalculateExponentialBackoff(attempt);
                    _logger.LogDebug("⏳ [RESILIENCE] Waiting {Delay}ms before retry {Attempt}", delay.TotalMilliseconds, attempt + 1);
                    await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
                }
            }
            catch (Exception ex)
            {
                // Non-retriable exception
                _logger.LogError(ex, "❌ [RESILIENCE] Non-retriable error in {Operation}: {Error}", operationName, ex.Message);
                throw;
            }
        }
        
        // All retries exhausted
        _logger.LogError(lastException, "❌ [RESILIENCE] Operation {Operation} failed after {Attempts} attempts", operationName, _config.MaxRetries);
        throw lastException ?? new InvalidOperationException($"Operation {operationName} failed after {_config.MaxRetries} attempts");
    }

    /// <summary>
    /// Execute HTTP operation with proper timeout and retry handling
    /// </summary>
    public Task<T> ExecuteHttpOperationAsync<T>(
        string operationName,
        Func<HttpClient, CancellationToken, Task<T>> httpOperation,
        HttpClient httpClient,
        CancellationToken cancellationToken = default)
    {
        return ExecuteWithResilienceAsync(operationName, async (ct) =>
        {
            using var timeoutCts = CancellationTokenSource.CreateLinkedTokenSource(ct).ConfigureAwait(false);
            timeoutCts.CancelAfter(_config.HttpTimeout);
            
            return await httpOperation(httpClient, timeoutCts.Token).ConfigureAwait(false);
        }, cancellationToken);
    }

    /// <summary>
    /// Fallback mechanism for critical trading operations
    /// </summary>
    public T ExecuteWithFallback<T>(
        string operationName,
        Func<T> primaryOperation,
        Func<T> fallbackOperation,
        bool logFallback = true)
    {
        try
        {
            return primaryOperation();
        }
        catch (Exception ex)
        {
            if (logFallback)
            {
                _logger.LogWarning(ex, "⚠️ [RESILIENCE] Primary operation {Operation} failed, using fallback: {Error}", 
                    operationName, ex.Message);
            }
            
            try
            {
                var result = fallbackOperation();
                _logger.LogInformation("🔄 [RESILIENCE] Fallback operation {Operation} succeeded", operationName);
                return result;
            }
            catch (Exception fallbackEx)
            {
                _logger.LogError(fallbackEx, "❌ [RESILIENCE] Both primary and fallback operations failed for {Operation}", operationName);
                throw new InvalidOperationException($"Both primary and fallback operations failed for {operationName}", ex);
            }
        }
    }

    private CircuitBreakerState GetOrCreateCircuitBreaker(string operationName)
    {
        if (!_circuitBreakers.TryGetValue(operationName, out var state))
        {
            state = new CircuitBreakerState();
            _circuitBreakers[operationName] = state;
        }
        return state;
    }

    private bool IsRetriableException(Exception ex)
    {
        return ex switch
        {
            HttpRequestException httpEx => IsRetriableHttpError(httpEx),
            TaskCanceledException => true, // Timeout
            SocketException => true,
            TimeoutException => true,
            JsonException => false, // Data corruption, don't retry
            ArgumentException => false, // Bad input, don't retry
            UnauthorizedAccessException => false, // Auth issues, don't retry
            _ => false
        };
    }

    private bool IsRetriableHttpError(HttpRequestException httpEx)
    {
        // Retry on server errors, not client errors
        var statusCode = GetHttpStatusCode(httpEx);
        return statusCode >= HttpStatusCode.InternalServerError || 
               statusCode == HttpStatusCode.RequestTimeout ||
               statusCode == HttpStatusCode.TooManyRequests;
    }

    private HttpStatusCode GetHttpStatusCode(HttpRequestException httpEx)
    {
        // Extract status code from HTTP exception message if available
        // This is a simplified implementation
        if (httpEx.Message.Contains("500")) return HttpStatusCode.InternalServerError;
        if (httpEx.Message.Contains("502")) return HttpStatusCode.BadGateway;
        if (httpEx.Message.Contains("503")) return HttpStatusCode.ServiceUnavailable;
        if (httpEx.Message.Contains("504")) return HttpStatusCode.GatewayTimeout;
        if (httpEx.Message.Contains("408")) return HttpStatusCode.RequestTimeout;
        if (httpEx.Message.Contains("429")) return HttpStatusCode.TooManyRequests;
        
        return HttpStatusCode.InternalServerError; // Default to retryable
    }

    private TimeSpan CalculateExponentialBackoff(int attempt)
    {
        var baseDelay = _config.BaseRetryDelay;
        var exponentialDelay = TimeSpan.FromMilliseconds(baseDelay.TotalMilliseconds * Math.Pow(2, attempt - 1));
        var jitteredDelay = TimeSpan.FromMilliseconds(exponentialDelay.TotalMilliseconds * (0.8 + Random.Shared.NextDouble() * 0.4));
        
        return jitteredDelay > _config.MaxRetryDelay ? _config.MaxRetryDelay : jitteredDelay;
    }
}

#region Configuration and Data Models

public class ResilienceConfig
{
    public int MaxRetries { get; set; } = 3;
    public TimeSpan BaseRetryDelay { get; set; } = TimeSpan.FromMilliseconds(500);
    public TimeSpan MaxRetryDelay { get; set; } = TimeSpan.FromSeconds(30);
    public TimeSpan HttpTimeout { get; set; } = TimeSpan.FromSeconds(30);
    public int CircuitBreakerThreshold { get; set; } = 5;
    public TimeSpan CircuitBreakerTimeout { get; set; } = TimeSpan.FromMinutes(1);
}

public class CircuitBreakerState
{
    public CircuitState State { get; set; } = CircuitState.Closed;
    public int FailureCount { get; set; }
    public DateTime LastFailure { get; set; }
}

public enum CircuitState
{
    Closed,   // Normal operation
    Open,     // Failing, don't try
    HalfOpen  // Testing if service is back
}

public class CircuitBreakerOpenException : Exception
{
    public CircuitBreakerOpenException(string message) : base(message) { }

    public CircuitBreakerOpenException()
    {
    }

    public CircuitBreakerOpenException(string message, Exception innerException) : base(message, innerException)
    {
    }
}

#endregion