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
            int maxAttempts = 3,
            int waitMs = 30000)
        {
            await WaitForConnected(hub, TimeSpan.FromMilliseconds(waitMs), ct, log);
            try
            {
                await call();
            }
            catch (InvalidOperationException ioe)
            {
                log.LogWarning(ioe, "Invoke failed: hub not active.");
                throw;
            }
            catch (Exception ex)
            {
                log.LogWarning(ex, "Invoke failed.");
                throw;
            }
        }

        public static async Task WaitForConnected(
            HubConnection hub,
            TimeSpan timeout,
            CancellationToken ct,
            ILogger log)
        {
            if (hub.State == HubConnectionState.Connected) return;

            // Fast-path: if we're Disconnected, try starting immediately (no throw if it fails).
            if (hub.State == HubConnectionState.Disconnected)
            {
                try { await hub.StartAsync(ct); } catch { /* allow retry loop below */ }
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            while (hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if (sw.Elapsed > timeout)
                    throw new TimeoutException($"Hub stayed {hub.State} for {timeout.TotalSeconds:N0}s.");

                if (hub.State == HubConnectionState.Disconnected)
                {
                    try { await hub.StartAsync(ct); } catch { /* swallow; next loop will re-check */ }
                }

                await Task.Delay(100, ct);
            }
            log.LogDebug("Hub is Connected (ConnectionId={Id}).", hub.ConnectionId);
        }
    }
}
