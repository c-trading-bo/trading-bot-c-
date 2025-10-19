using System;
using System.IO;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Retry service with exponential backoff for training operations
/// Handles transient failures with intelligent retry logic
/// </summary>
internal sealed class TrainingRetryService
{
    private readonly ILogger<TrainingRetryService> _logger;
    private const int MaxRetries = 3;
    private static readonly TimeSpan[] RetryDelays = new[]
    {
        TimeSpan.FromMinutes(5),  // First retry: 5 minutes
        TimeSpan.FromMinutes(15), // Second retry: 15 minutes
        TimeSpan.FromMinutes(30)  // Third retry: 30 minutes
    };

    public TrainingRetryService(ILogger<TrainingRetryService> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Execute an operation with retry logic and exponential backoff
    /// </summary>
    public async Task<T> ExecuteWithRetryAsync<T>(
        Func<CancellationToken, Task<T>> operation,
        string operationName,
        Func<Exception, bool> isTransientError,
        CancellationToken cancellationToken = default)
    {
        var attempt = 0;
        Exception? lastException = null;

        while (attempt < MaxRetries)
        {
            try
            {
                if (attempt > 0)
                {
                    _logger.LogInformation("[RETRY] Attempting {Operation} (Retry {Attempt}/{Max})",
                        operationName, attempt, MaxRetries);
                }

                return await operation(cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (isTransientError(ex) && attempt < MaxRetries - 1)
            {
                lastException = ex;
                attempt++;
                
                var delay = RetryDelays[Math.Min(attempt - 1, RetryDelays.Length - 1)];
                
                _logger.LogWarning(ex,
                    "[RETRY] {Operation} failed with transient error (Attempt {Attempt}/{Max}). Retrying in {Delay}...",
                    operationName, attempt, MaxRetries, delay);

                await Task.Delay(delay, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                // Non-transient error or max retries reached
                _logger.LogError(ex,
                    "[RETRY] {Operation} failed permanently after {Attempt} attempts",
                    operationName, attempt + 1);
                throw;
            }
        }

        // This should never be reached, but just in case
        var finalException = lastException ?? new InvalidOperationException("Retry logic error");
        _logger.LogError(finalException,
            "[RETRY] {Operation} failed after {Max} retries",
            operationName, MaxRetries);
        throw finalException;
    }

    /// <summary>
    /// Determine if an exception represents a transient error that should be retried
    /// </summary>
    public static bool IsTransientError(Exception ex)
    {
        // Network-related errors (transient)
        if (ex is System.Net.Http.HttpRequestException ||
            ex is System.Net.Sockets.SocketException ||
            ex is TimeoutException ||
            ex is TaskCanceledException)
        {
            return true;
        }

        // IO errors that might be transient (file locks, transient unavailability)
        if (ex is IOException ioEx)
        {
            // Check for specific transient IO errors
            var message = ioEx.Message.ToLowerInvariant();
            if (message.Contains("being used by another process") ||
                message.Contains("temp") ||
                message.Contains("network"))
            {
                return true;
            }
        }

        // Database/resource transiently unavailable
        if (ex.Message.Contains("unavailable", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("resource busy", StringComparison.OrdinalIgnoreCase) ||
            ex.Message.Contains("connection timeout", StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        // Not a transient error - likely a code bug or permanent failure
        return false;
    }

    /// <summary>
    /// Determine if an exception is permanent and should not be retried
    /// </summary>
    public static bool IsPermanentError(Exception ex)
    {
        // Argument/validation errors (permanent - code bug)
        if (ex is ArgumentException ||
            ex is ArgumentNullException ||
            ex is InvalidOperationException ||
            ex is NotImplementedException ||
            ex is NotSupportedException)
        {
            return true;
        }

        // File not found, path errors (permanent - configuration issue)
        if (ex is FileNotFoundException ||
            ex is DirectoryNotFoundException ||
            ex is UnauthorizedAccessException)
        {
            return true;
        }

        // Serialization errors (permanent - data corruption or code bug)
        if (ex is JsonException ||
            ex is System.Text.Json.JsonException ||
            ex is FormatException)
        {
            return true;
        }

        return false;
    }
}
