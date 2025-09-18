using System;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Legacy
{
    public static class HubSafe
    {
        public static async Task<bool> InvokeAsync(HubConnection hub, Func<Task> call, CancellationToken ct, int maxAttempts = 10)
        {
            for (var a = 1; a <= maxAttempts; a++)
            {
                await WaitUntilConnected(hub, ct);
                try { await call(); return true; }
                catch (InvalidOperationException ioe) when (ioe.Message.Contains("not active"))
                {
                    await Task.Delay(150 * a, ct); // backoff
                    continue;
                }
            }
            Console.WriteLine("Invoke aborted: connection never became active.");
            return false; // don’t crash app
        }

        private static async Task WaitUntilConnected(HubConnection hub, CancellationToken ct)
        {
            while (hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if (hub.State == HubConnectionState.Disconnected)
                {
                    try { await hub.StartAsync(ct); } catch { /* let autoreconnect handle */ }
                }
                await Task.Delay(120, ct);
            }
        }
    }
}
