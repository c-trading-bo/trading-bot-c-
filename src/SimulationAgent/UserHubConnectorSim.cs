using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Net.WebSockets;
using System.Threading;
using System.Threading.Tasks;
using System;

namespace SimulationAgent
{
    public static class UserHubConnectorSim
    {
        public static HubConnection Build(string jwt, ILogger log)
        {
            var url = "https://rtc.topstepx.com/hubs/user";

            var hub = new HubConnectionBuilder()
                .WithUrl(url, options =>
                {
                    options.AccessTokenProvider = () => Task.FromResult(jwt);
                    options.Transports = HttpTransportType.WebSockets;
                    options.SkipNegotiation = true;
                })
                .WithAutomaticReconnect(new[]
                {
                    TimeSpan.Zero,
                    TimeSpan.FromSeconds(2),
                    TimeSpan.FromSeconds(5),
                    TimeSpan.FromSeconds(10)
                })
                .Build();

            hub.ServerTimeout = TimeSpan.FromSeconds(60);
            hub.KeepAliveInterval = TimeSpan.FromSeconds(15);
            hub.HandshakeTimeout = TimeSpan.FromSeconds(15);

            hub.Closed += async (ex) =>
            {
                log.LogWarning(ex, "UserHub CLOSED: {Message}", ex?.Message);
                await Task.Delay(TimeSpan.FromSeconds(1));
            };

            hub.Reconnecting += (ex) =>
            {
                log.LogWarning(ex, "UserHub RECONNECTING: {Message}", ex?.Message);
                return Task.CompletedTask;
            };

            hub.Reconnected += (connId) =>
            {
                log.LogInformation("UserHub RECONNECTED. ConnectionId={ConnId}", connId);
                // Re-subscribe or re-join groups here if needed
                return Task.CompletedTask;
            };

            return hub;
        }

        public static async Task StartAndReadyAsync(HubConnection hub, ILogger log, CancellationToken ct)
        {
            await hub.StartAsync(ct);
            log.LogInformation("UserHub connected. State={State}", hub.State);
        }

        public static async Task<bool> SafeInvoke(HubConnection hub, Func<Task> call, ILogger log, CancellationToken ct, int maxAttempts = 3)
        {
            for (int i = 1; i <= maxAttempts; i++)
            {
                if (hub.State != HubConnectionState.Connected)
                {
                    log.LogWarning("Invoke skipped: Hub state is {State}", hub.State);
                    await Task.Delay(250, ct);
                    continue;
                }

                try
                {
                    await call();
                    return true;
                }
                catch (Exception ex)
                {
                    log.LogWarning(ex, "Invoke attempt {Attempt} failed.", i);
                    await Task.Delay(TimeSpan.FromMilliseconds(500 * i), ct);
                }
            }
            return false;
        }
    }
}
