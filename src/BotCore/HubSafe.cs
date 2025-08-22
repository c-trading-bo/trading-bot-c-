using System;
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
        public static async Task InvokeAsync(HubConnection hub, Func<Task> call, CancellationToken ct, int maxAttempts = 5)
        {
            for (var a = 1; ; a++)
            {
                await WaitUntilConnected(hub, ct);
                try { await call(); return; }
                catch (InvalidOperationException) when (a < maxAttempts)
                { await Task.Delay(150 * a, ct); }
            }
        }

        private static async Task WaitUntilConnected(HubConnection hub, CancellationToken ct)
        {
            while (hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if (hub.State == HubConnectionState.Disconnected)
                    try { await hub.StartAsync(ct); } catch { /* let it retry */ }
                await Task.Delay(120, ct);
            }
        }
    }
}
