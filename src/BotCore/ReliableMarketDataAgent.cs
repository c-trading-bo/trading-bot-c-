// Agent: ReliableMarketDataAgent
// Role: Provides fault-tolerant market data streaming and recovery.
// Integration: Used by orchestrator and strategies for robust data feeds.
using System;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using BotCore.Models;

namespace BotCore
{
    /// <summary>
    /// Resilient SignalR market data client:
    ///  - awaits StartAsync before sending
    ///  - gates sends on Connected state
    ///  - resubscribes after automatic reconnect
    ///  - wires both "Gateway*" and plain event names
    /// </summary>
    public sealed class ReliableMarketDataAgent : IAsyncDisposable
    {
        private HubConnection? _hub;
        private readonly string _jwt;
        private string? _contractId;
        private string? _barTf;

        public event Action<Bar>? OnBar;
        public event Action<JsonElement>? OnQuote;
        public event Action<JsonElement>? OnTrade;

        public ReliableMarketDataAgent(string jwt)
        {
            _jwt = jwt ?? throw new ArgumentNullException(nameof(jwt));
        }

        public async Task StartAsync(string contractId, string barTf, CancellationToken ct = default)
        {
            _contractId = contractId ?? throw new ArgumentNullException(nameof(contractId));
            _barTf = barTf ?? throw new ArgumentNullException(nameof(barTf));

            _hub = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", o =>
                {
                    o.AccessTokenProvider = () => Task.FromResult(_jwt);
                })
                .WithAutomaticReconnect(new[] { TimeSpan.Zero, TimeSpan.FromSeconds(2), TimeSpan.FromSeconds(10), TimeSpan.FromSeconds(30) })
                .Build();

            WireHandlers(_hub);

            _hub.Reconnected += async _ =>
            {
                try { await SubscribeAll(ct).ConfigureAwait(false); }
                catch (Exception ex) { Console.WriteLine($"[ReliableMarketDataAgent] Resubscribe failed: {ex.Message}"); }
            };

            await _hub.StartAsync(ct).ConfigureAwait(false);
            await WaitForConnectedAsync(ct).ConfigureAwait(false);
            await SubscribeAll(ct).ConfigureAwait(false);
        }

        private void WireHandlers(HubConnection hub)
        {
            hub.On<JsonElement>("Bar", data =>
            {
                try
                {
                    var b = new Bar
                    {
                        Ts = data.TryGetProperty("ts", out var ts) ? ts.GetInt64() :
                             data.TryGetProperty("timestamp", out var t2) ? t2.GetInt64() : 0,
                        Open = data.TryGetProperty("open", out var o) ? o.GetDecimal() : 0m,
                        High = data.TryGetProperty("high", out var h) ? h.GetDecimal() : 0m,
                        Low = data.TryGetProperty("low", out var l) ? l.GetDecimal() : 0m,
                        Close = data.TryGetProperty("close", out var c) ? c.GetDecimal() : 0m,
                        Volume = data.TryGetProperty("volume", out var v) ? v.GetInt32() : 0,
                        Symbol = data.TryGetProperty("symbol", out var s) ? s.GetString() ?? "" : ""
                    };
                    OnBar?.Invoke(b);
                }
                catch { }
            });
            hub.On<JsonElement>("GatewayBars", data => OnQuote?.Invoke(data)); // fallback

            hub.On<JsonElement>("Quote", data => OnQuote?.Invoke(data));
            hub.On<JsonElement>("GatewayQuote", data => OnQuote?.Invoke(data));

            hub.On<JsonElement>("Trade", data => OnTrade?.Invoke(data));
            hub.On<JsonElement>("GatewayTrade", data => OnTrade?.Invoke(data));
        }

        private async Task SubscribeAll(CancellationToken ct)
        {
            if (_hub is null) throw new InvalidOperationException("Hub is not built.");
            if (_contractId is null || _barTf is null) throw new InvalidOperationException("Call StartAsync(contractId, barTf) first.");
            await WaitForConnectedAsync(ct).ConfigureAwait(false);

            await TrySendAsync("SubscribeQuote", new object?[] { _contractId }, ct);
            await TrySendAsync("SubscribeContractQuotes", new object?[] { _contractId }, ct);

            await TrySendAsync("SubscribeTrade", new object?[] { _contractId }, ct);
            await TrySendAsync("SubscribeContractTrades", new object?[] { _contractId }, ct);

            await TrySendAsync("SubscribeBars", new object?[] { _contractId, _barTf }, ct);
            await TrySendAsync("SubscribeContractBars", new object?[] { _contractId, _barTf }, ct);

            Console.WriteLine($"[ReliableMarketDataAgent] Subscribed to {_contractId} ({_barTf}).");
        }

        private async Task<bool> TrySendAsync(string method, object?[] args, CancellationToken ct)
        {
            try
            {
                if (_hub is null) return false;
                if (_hub.State != HubConnectionState.Connected) await WaitForConnectedAsync(ct).ConfigureAwait(false);
                await _hub.SendAsync(method, args, ct).ConfigureAwait(false);
                return true;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[ReliableMarketDataAgent] {method} failed: {ex.Message}");
                return false;
            }
        }

        private async Task WaitForConnectedAsync(CancellationToken ct)
        {
            if (_hub is null) throw new InvalidOperationException("Hub is null");
            var start = DateTime.UtcNow;
            while (_hub.State != HubConnectionState.Connected)
            {
                ct.ThrowIfCancellationRequested();
                if ((DateTime.UtcNow - start) > TimeSpan.FromSeconds(30))
                    throw new TimeoutException("SignalR connection did not reach Connected state within 30s.");
                await Task.Delay(200, ct).ConfigureAwait(false);
            }
        }

        public async ValueTask DisposeAsync()
        {
            if (_hub is not null)
            {
                try { await _hub.DisposeAsync(); } catch { }
            }
        }

        public async Task StopAsync(CancellationToken ct = default)
        {
            if (_hub is not null)
            {
                try { await _hub.StopAsync(ct).ConfigureAwait(false); } catch { }
                await DisposeAsync();
            }
        }
    }
}
