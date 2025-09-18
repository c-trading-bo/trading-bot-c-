using System;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;

namespace TradingBot.Infrastructure.TopstepX
{
    /// <summary>
    /// Production-grade constants for SignalR safe invoker
    /// </summary>
    internal static class SignalRSafeInvokerConstants
    {
        public const int DEFAULT_WAIT_TIMEOUT_MS = 30000; // 30 seconds default timeout
        public const int RECONNECT_POLL_INTERVAL_MS = 300; // 300ms polling interval
        public const int DEFAULT_CONNECTION_TIMEOUT_SECONDS = 30; // 30 seconds connection timeout
        public const int DEFAULT_MAX_ATTEMPTS = 12; // Maximum retry attempts
    }

    /// <summary>
    /// Utility for safe SignalR hub invocation with automatic reconnect and retry logic.
    /// This is a copy of BotCore.HubSafe to avoid circular dependencies.
    /// </summary>
    public static class SignalRSafeInvoker
    {
        public static async Task InvokeWhenConnected(
            HubConnection hub,
            Func<Task> call,
            ILogger log,
            CancellationToken ct,
            int maxAttempts = 12,
            int waitMs = SignalRSafeInvokerConstants.DEFAULT_WAIT_TIMEOUT_MS)
        {
            await WaitForConnected(hub, TimeSpan.FromSeconds(SignalRSafeInvokerConstants.DEFAULT_CONNECTION_TIMEOUT_SECONDS), ct, log);

            for (int attempt = 1; attempt <= maxAttempts; attempt++)
            {
                ct.ThrowIfCancellationRequested();
                try
                {
                    await call();
                    if (attempt > 1)
                        log.LogDebug("Invoke succeeded after {Attempts} attempt(s).", attempt);
                    return;
                }
                catch (InvalidOperationException ioe)
                {
                    var msg = ioe.Message ?? string.Empty;
                    if (msg.Contains("not active", StringComparison.OrdinalIgnoreCase)
                        || msg.Contains("cannot be called", StringComparison.OrdinalIgnoreCase)
                        || msg.Contains("The 'InvokeCoreAsync' method cannot be called", StringComparison.OrdinalIgnoreCase))
                    {
                        log.LogDebug(ioe, "Invoke attempt {Attempt} while hub state={State}; retrying.", attempt, hub.State);
                    }
                    else
                    {
                        // non-transient InvalidOperation
                        log.LogWarning(ioe, "Invoke failed (non-transient). Rethrowing.");
                        throw;
                    }
                }
                catch (Microsoft.AspNetCore.SignalR.HubException hex)
                {
                    log.LogDebug(hex, "HubException on attempt {Attempt}; will retry.", attempt);
                }
                catch (TaskCanceledException tce)
                {
                    log.LogDebug(tce, "Invoke task canceled on attempt {Attempt}; will retry.", attempt);
                }
                catch (OperationCanceledException oce)
                {
                    // During reconnects, transient cancels can bubble up from transport/handshake
                    log.LogDebug(oce, "Invoke canceled on attempt {Attempt}; will retry.", attempt);
                }
                catch (Exception ex)
                {
                    // Other exceptions: try a few times as well, but warn
                    log.LogWarning(ex, "Invoke attempt {Attempt} failed; will retry (state={State}).", attempt, hub.State);
                }

                // Backoff before next attempt; let automatic reconnect handle restarts
                try
                {
                    await Task.Delay(TimeSpan.FromMilliseconds(waitMs), ct);
                }
                catch { /* ignore delay errors and continue */ }
            }

            throw new TimeoutException($"Invoke failed after {maxAttempts} attempts while state={hub.State}.");
        }

        public static async Task WaitForConnected(
            HubConnection hub,
            TimeSpan timeout,
            CancellationToken ct,
            ILogger log)
        {
            if (hub.State == HubConnectionState.Connected) return;

            // Fast-path: if we're Disconnected, try starting immediately (no throw if it fails).
            // If the hub hasn't started yet (Disconnected), attempt a StartAsync once to kick things off.
            if (hub.State == HubConnectionState.Disconnected)
            {
                try
                {
                    await hub.StartAsync(ct);
                }
                catch (Exception ex)
                {
                    // Log at debug; automatic reconnect may still recover if StartAsync races
                    log.LogDebug(ex, "Hub StartAsync failed in WaitForConnected (state={State}); will rely on reconnect.", hub.State);
                }
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if (sw.Elapsed > timeout)
                    throw new TimeoutException($"Hub stayed {hub.State} for {timeout.TotalSeconds:N0}s.");

                // Do not force StartAsync during reconnect; rely on automatic reconnect policy
                await Task.Delay(SignalRSafeInvokerConstants.RECONNECT_POLL_INTERVAL_MS, ct);
            }
            log.LogDebug("Hub is Connected (ConnectionId={Id}).", hub.ConnectionId);
        }
    }
}