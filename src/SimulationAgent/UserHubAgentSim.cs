using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Threading;
using System.Threading.Tasks;
using System;

namespace SimulationAgent
{
    public class UserHubAgentSim
    {
        private HubConnection? _hub;
        private readonly ILogger _logger;
        private bool _subscribed;

        public UserHubAgentSim(ILogger logger)
        {
            _logger = logger;
        }

        public async Task ConnectAsync(string jwtToken, long accountId, CancellationToken ct)
        {
            _hub ??= UserHubConnectorSim.Build(jwtToken, _logger);
            await UserHubConnectorSim.StartAndReadyAsync(_hub, _logger, ct);

            // Wire up event handlers for order/trade events
            _hub.On<object>("GatewayUserOrder", data =>
            {
                _logger.LogInformation($"ORDER => {data}");
            });
            _hub.On<object>("GatewayUserTrade", data =>
            {
                _logger.LogInformation($"TRADE => {data}");
            });

            // Subscribe to orders/trades only after connected
            if (!_subscribed && _hub.State == HubConnectionState.Connected)
            {
                var ok1 = await UserHubConnectorSim.SafeInvoke(
                    _hub,
                    () => _hub.InvokeAsync("SubscribeOrders", accountId, ct),
                    _logger, ct);
                var ok2 = await UserHubConnectorSim.SafeInvoke(
                    _hub,
                    () => _hub.InvokeAsync("SubscribeTrades", accountId, ct),
                    _logger, ct);
                _subscribed = ok1 && ok2;
                if (!_subscribed)
                    _logger.LogError("Failed to subscribe to orders/trades on UserHub.");
            }
        }
    }
}
