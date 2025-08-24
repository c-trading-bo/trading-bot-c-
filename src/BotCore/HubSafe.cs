using System;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;

namespace BotCore
{
    /// <summary>
    /// Utility for safe SignalR hub invocation with retries that does not throw when hub is null/closed.
    /// </summary>
    public static class HubSafe
    {
        /// <summary>
        /// Preferred safe invoker. Tries a few times only when hub is connected; returns false on failure.
        /// </summary>
        public static async Task<bool> InvokeIfConnected(
            HubConnection? hub,
            Func<Task> call,
            ILogger log,
            CancellationToken ct,
            int tries = 3)
        {
            for (int i = 1; i <= tries; i++)
            {
                ct.ThrowIfCancellationRequested();
                if (hub is null || hub.State != HubConnectionState.Connected)
                {
                    await Task.Delay(150 * i, ct);
                    continue;
                }
                try
                {
                    await call();
                    return true;
                }
                catch (InvalidOperationException)
                {
                    // transient: hub not active
                    await Task.Delay(200 * i, ct);
                }
                catch (Exception ex)
                {
                    log.LogWarning(ex, "HubSafe.InvokeIfConnected: invoke failed (attempt {Attempt})", i);
                    await Task.Delay(300 * i, ct);
                }
            }
            return false;
        }

        /// <summary>
        /// Backwards-compatible wrapper used by existing call sites. Now safe and non-throwing.
        /// </summary>
        public static Task<bool> InvokeWhenConnected(
            HubConnection hub,
            Func<Task> call,
            ILogger log,
            CancellationToken ct,
            int maxAttempts = 3,
            int waitMs = 10000)
            => InvokeIfConnected(hub, call, log, ct, maxAttempts);

        // Kept for compatibility; no longer throws or blocks aggressively.
        public static Task WaitForConnected(
            HubConnection hub,
            TimeSpan timeout,
            CancellationToken ct,
            ILogger log)
        {
            // No-op lightweight wait that resolves quickly if not connected; callers use InvokeIfConnected loop.
            if (hub.State == HubConnectionState.Connected)
            {
                log.LogDebug("Hub is Connected (ConnectionId={Id}).", hub.ConnectionId);
            }
            return Task.CompletedTask;
        }
    }
}
