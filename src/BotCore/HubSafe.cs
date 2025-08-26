using System;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;

namespace BotCore
{
    /// <summary>
    /// Utility for safe SignalR hub invocation with automatic reconnect and retry logic.
    /// </summary>
    public static class HubSafe
    {
        public static async Task InvokeWhenConnected(
            HubConnection hub,
            Func<Task> call,
            ILogger log,
            CancellationToken ct,
            int maxAttempts = 12,
            int waitMs = 30000)
        {
            await WaitForConnected(hub, TimeSpan.FromMilliseconds(waitMs), ct, log);

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
                    await Task.Delay(TimeSpan.FromSeconds(1), ct);
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
            // Do not call StartAsync here; rely on automatic reconnect to handle restarts

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if (sw.Elapsed > timeout)
                    throw new TimeoutException($"Hub stayed {hub.State} for {timeout.TotalSeconds:N0}s.");

                // Do not force StartAsync during reconnect; rely on automatic reconnect policy
                await Task.Delay(100, ct);
            }
            log.LogDebug("Hub is Connected (ConnectionId={Id}).", hub.ConnectionId);
        }
    }
}
