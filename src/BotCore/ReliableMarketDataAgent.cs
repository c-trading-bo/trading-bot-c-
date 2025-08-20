// Remove stray closing brace at the top
using System;
using Microsoft.AspNetCore.SignalR.Client;
using System.Collections.Concurrent;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;
using System.Net.Http;

namespace BotCore.MarketData
{
    public class ReliableMarketDataAgent : IAsyncDisposable
    {
        public int BarsSeen { get; private set; }
        public decimal LastPrice { get; private set; }
        private HubConnection? _conn;
        private readonly SemaphoreSlim _connectLock = new(1, 1);
        private readonly ConcurrentQueue<Func<CancellationToken, Task>> _pendingSubs = new();
        private readonly string _marketHubUrl;
        private readonly Func<Task<string?>> _jwtProvider;

        public ReliableMarketDataAgent(string? marketHubUrl = null, Func<Task<string?>>? jwtProvider = null)
        {
            _marketHubUrl = string.IsNullOrWhiteSpace(marketHubUrl)
                ? (Environment.GetEnvironmentVariable("TOPSTEPX_MARKET_HUB")?.Trim()
                    ?? "https://rtc.topstepx.com/hubs/market")
                : marketHubUrl.Trim();

            _jwtProvider = jwtProvider ?? (async () => {
                await Task.Yield(); // Ensure truly async
                var tok = Environment.GetEnvironmentVariable("TOPSTEPX_JWT");
                if (string.IsNullOrWhiteSpace(tok))
                    throw new InvalidOperationException("TOPSTEPX_JWT is empty. Fetch JWT first.");
                return tok.Trim();
            });
        }

        private HubConnection BuildConnection()
        {
            return new HubConnectionBuilder()
                .WithUrl(_marketHubUrl, options =>
                {
                    options.AccessTokenProvider = async () => await _jwtProvider();
                    options.SkipNegotiation = false;
                    options.HttpMessageHandlerFactory = _ => new SocketsHttpHandler
                    {
                        AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate,
                        PooledConnectionIdleTimeout = TimeSpan.FromMinutes(2),
                        ConnectTimeout = TimeSpan.FromSeconds(15),
                    };
                })
                .WithAutomaticReconnect(new[]
                {
                    TimeSpan.Zero, TimeSpan.FromSeconds(2),
                    TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10)
                })
                .ConfigureLogging(b => b.AddConsole().SetMinimumLevel(LogLevel.Information))
                .Build();
        }

        public async Task ConnectAsync(CancellationToken ct = default)
        {
            await _connectLock.WaitAsync(ct);
            try
            {
                if (_conn is { State: HubConnectionState.Connected }) return;
                _conn ??= BuildConnection();
                WireEvents(_conn!);
                Console.WriteLine("[ReliableMarketDataAgent] Starting connection…");
                await _conn!.StartAsync(ct);
                await WaitForStateAsync(HubConnectionState.Connected, TimeSpan.FromSeconds(20), ct);
                Console.WriteLine("[ReliableMarketDataAgent] Connected.");
            }
            finally
            {
                _connectLock.Release();
            }
        }
        // ...existing code...

        private void WireEvents(HubConnection conn)
        {
            conn.Reconnecting += ex => {
                Console.WriteLine($"[ReliableMarketDataAgent] Reconnecting… {ex?.Message}");
                return Task.CompletedTask;
            };

            conn.Reconnected += _ => Task.Run(async () => {
                Console.WriteLine("[ReliableMarketDataAgent] Reconnected. Replaying subscriptions…");
                while (_pendingSubs.TryDequeue(out var sub))
                {
                    try { await sub(CancellationToken.None); }
                    catch (Exception ex) { Console.WriteLine($"[ReliableMarketDataAgent] Re-subscribe failed: {ex.Message}"); }
                }
            });

            conn.Closed += ex => {
                Console.WriteLine($"[ReliableMarketDataAgent] Closed: {ex?.Message}");
                return Task.CompletedTask;
            };

            // Bar-ish events with different names
            foreach (var evt in new[] { "Bar", "Bars", "OnBar", "OnBars", "RealtimeBar" })
            {
                conn.On<object>(evt, payload =>
                {
                    try
                    {
                        var close = TryGetDecimal(payload, "close", "Close", "c", "last", "Last", "price", "Price");
                        if (close.HasValue) LastPrice = close.Value;
                        BarsSeen++;
                        Console.WriteLine($"[MD] {evt} -> BarsSeen={BarsSeen} Last={LastPrice}");
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"[MD] {evt} parse fail: {ex.Message}");
                    }
                });
            }

                                // Quotes (helpful heartbeat even if market closed)
                                foreach (var evt in new[] { "Quote", "Quotes", "OnQuote", "OnQuotes", "RealtimeQuote" })
                                {
                                    conn.On<object>(evt, payload =>
                                    {
                                        var last = TryGetDecimal(payload, "last", "Last", "price", "Price", "p", "c");
                                        if (last.HasValue) LastPrice = last.Value;
                                    });
                                }
                            }

                            private static decimal? TryGetDecimal(object obj, params string[] keys)
                            {
                                if (obj == null) return null;

                                // flat
                                foreach (var k in keys)
                                {
                                    var p = obj.GetType().GetProperty(k);
                                    if (p?.GetValue(obj) is { } v && decimal.TryParse(v.ToString(), out var d)) return d;
                                }
                                // nested { bar = { close = ... } }
                                var bp = obj.GetType().GetProperty("bar");
                                if (bp?.GetValue(obj) is { } inner)
                                {
                                    foreach (var k in keys)
                                    {
                                        var p2 = inner.GetType().GetProperty(k);
                                        if (p2?.GetValue(inner) is { } v && decimal.TryParse(v.ToString(), out var d)) return d;
                                    }
                                }
                                return null;
                            }

                            private async Task WaitForStateAsync(HubConnectionState desired, TimeSpan timeout, CancellationToken ct)
                            {
                                var start = DateTime.UtcNow;
                                while (_conn!.State != desired)
                                {
                                    ct.ThrowIfCancellationRequested();
                                    if (DateTime.UtcNow - start > timeout)
                                        throw new TimeoutException($"SignalR did not reach {desired} within {timeout.TotalSeconds}s.");
                                    await Task.Delay(150, ct);
                                }
                            }

                            private void EnqueueOrInvoke(Func<CancellationToken, Task> invoker, CancellationToken ct)
                            {
                                if (_conn!.State == HubConnectionState.Connected)
                                {
                                    _ = Task.Run(async () =>
                                    {
                                        try { await invoker(ct); }
                                        catch (Exception ex) { Console.WriteLine($"[MD] invoke failed: {ex.Message}"); }
                                    }, ct);
                                }
                                else
                                {
                                    _pendingSubs.Enqueue(invoker);
                                }
                            }

                            public async ValueTask DisposeAsync()
                            {
                                if (_conn is not null)
                                {
                                    await _conn.DisposeAsync();
                                }
                            }
                        }
                    }


