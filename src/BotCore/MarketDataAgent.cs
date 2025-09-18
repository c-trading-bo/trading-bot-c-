// Agent: MarketDataAgent
// Role: Handles market data streaming, aggregation, and event distribution.
// Integration: Feeds data to strategies, orchestrator, and risk agents.
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;
using BotCore.Models;

namespace BotCore
{
    public class MarketDataAgent(string jwt)
    {
        private readonly HubConnection _hub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", o => o.AccessTokenProvider = () => Task.FromResult<string?>(jwt))
                .WithAutomaticReconnect()
                .Build();
        public event Action<Bar>? OnBar;
        public event Action<JsonElement>? OnQuote;
        public event Action<JsonElement>? OnTrade;
        public int BarsSeen { get; private set; }
        public int QuotesSeen { get; private set; }
        public int TradesSeen { get; private set; }

        public async Task StartAsync(string contractId, string barTf = "1m")
        {
            // Normalize to both plain and Gateway* event names, and support two-arg variants
            _hub.On<JsonElement>("Bar", data =>
            {
                BarsSeen++;
                var bar = JsonSerializer.Deserialize<Bar>(data.GetRawText());
                if (bar is not null)
                    OnBar?.Invoke(bar);
            });
            _hub.On<JsonElement>("Quote", data =>
            {
                QuotesSeen++;
                OnQuote?.Invoke(data);
            });
            _hub.On<string, JsonElement>("GatewayQuote", (cid, data) =>
            {
                QuotesSeen++;
                OnQuote?.Invoke(data);
            });
            _hub.On<JsonElement>("Trade", data =>
            {
                TradesSeen++;
                OnTrade?.Invoke(data);
            });
            _hub.On<string, JsonElement>("GatewayTrade", (cid, data) =>
            {
                TradesSeen++;
                OnTrade?.Invoke(data);
            });
            await _hub.StartAsync().ConfigureAwait(false);
            if (_hub.State != HubConnectionState.Connected)
            {
                Console.WriteLine($"Market Hub connection state: {_hub.State}. Waiting for connection...");
                int retries = 0;
                while (_hub.State != HubConnectionState.Connected && retries < 10)
                {
                    await Task.Delay(500).ConfigureAwait(false);
                    retries++;
                }
                if (_hub.State != HubConnectionState.Connected)
                {
                    Console.WriteLine("Market Hub failed to connect after retries.");
                    throw new InvalidOperationException("Market Hub not connected.");
                }
            }
            Console.WriteLine("Market Hub connected. Subscribing...");
            // Subscribe using both legacy and Contract* method names for compatibility
            await _hub.SendAsync("SubscribeQuote", contractId).ConfigureAwait(false);
            await _hub.SendAsync("SubscribeContractQuotes", contractId).ConfigureAwait(false);

            await _hub.SendAsync("SubscribeTrade", contractId).ConfigureAwait(false);
            await _hub.SendAsync("SubscribeContractTrades", contractId).ConfigureAwait(false);

            await _hub.SendAsync("SubscribeBars", contractId, barTf).ConfigureAwait(false);
            await _hub.SendAsync("SubscribeContractBars", contractId, barTf).ConfigureAwait(false);
        }

        public async Task StopAsync()
        {
            await _hub.DisposeAsync().ConfigureAwait(false);
        }
    }
}
