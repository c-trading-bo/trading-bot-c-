using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Collections.Generic;
using System.Linq;
using BotCore;
using BotCore.Models;
using BotCore.Config;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
public static class Program {
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.Extensions.Logging;

namespace LegacyRunner{{
    public static class Program {
        // --- Minimal BotState and BotPhase for runner ---
        public enum BotPhase { Startup, Scanning, Armed, PlacingOrder, WaitingFill, ManagingPosition }
        public class BotState {
            public volatile bool MarketHubUp;
            public volatile bool UserHubUp;
            public long SignalsEvaluated;
            public long SignalsTriggered;
            public long OrdersPlaced;
            public static class Program {
            public DateTimeOffset LastSignalAt;
            public DateTimeOffset LastOrderAt;
            public DateTimeOffset LastFillAt;
            public BotPhase Phase = BotPhase.Startup;
            public bool IsTrading => OpenPositionsByContract.Count > 0;
            public ConcurrentDictionary<long, (string contractId, string status)> OpenOrders = new();
            public ConcurrentDictionary<string, decimal> OpenPositionsByContract = new();
        }
        // ---------- STATE FOR HEARTBEAT AND RUNNER ----------
        static volatile bool MarketHubUp;
        static long Bars;
        static decimal LastPrice;
        static BarAggregator barAggES = new BarAggregator();
        static BarAggregator barAggNQ = new BarAggregator();
        static List<Bar> barsES = new List<Bar>();
        static List<Bar> barsNQ = new List<Bar>();
        static BotState S = new BotState();

        public static async Task Main(string[] args) {
            // ...existing code...
            // --- Instantiate agents and runner state ---
            var strategyAgent = new StrategyAgent.StrategyAgent(config);
            var riskEngine = new BotCore.Risk.RiskEngine();
            var apiClient = new ApiClient(token);
            // ...existing code...
            // --- Wire BarAggregator event handlers after runner state is initialized ---
            barAggES.OnBar += bar => {
                barsES.Add(bar);
                Console.WriteLine($"[BarGen] ES bar: start={bar.Start:HH:mm:ss} close={bar.Close:0.00} vol={bar.Volume}");
                System.Threading.Interlocked.Increment(ref S.SignalsEvaluated);
                var snap = new MarketSnapshot { Symbol = "ES", UtcNow = DateTime.UtcNow };
                var signals = strategyAgent.RunAll(snap, barsES, riskEngine);
                Console.WriteLine($"[SignalEval] ES signals={signals?.Count ?? 0}");
                if (signals != null && signals.Count > 0)
                {
                    System.Threading.Interlocked.Add(ref S.SignalsTriggered, signals.Count);
                    S.LastSignalAt = DateTimeOffset.UtcNow;
                    S.Phase = BotPhase.Armed;
                }
                foreach (var sig in signals ?? new List<Signal>()) {
                    if (sig.ExpR < 1.0m) continue;
                    var side = sig.Side == "BUY" ? 0 : 1;
                    Task.Run(async () => {
                        S.Phase = BotPhase.PlacingOrder;
                        var orderId = await apiClient.PlaceLimit(accountId, es.id, side, sig.Size, sig.Entry);
                        if (orderId.HasValue)
                        {
                            System.Threading.Interlocked.Increment(ref S.OrdersPlaced);
                            S.LastOrderAt = DateTimeOffset.UtcNow;
                        }
                        S.Phase = BotPhase.WaitingFill;
                        Console.WriteLine($"[StrategyOrder] ES {sig.Side} id={orderId} entry={sig.Entry} R={sig.ExpR:0.00}");
                    });
                }
            };
            barAggNQ.OnBar += bar => {
                barsNQ.Add(bar);
                Console.WriteLine($"[BarGen] NQ bar: start={bar.Start:HH:mm:ss} close={bar.Close:0.00} vol={bar.Volume}");
                System.Threading.Interlocked.Increment(ref S.SignalsEvaluated);
                var snap = new MarketSnapshot { Symbol = "NQ", UtcNow = DateTime.UtcNow };
                var signals = strategyAgent.RunAll(snap, barsNQ, riskEngine);
                Console.WriteLine($"[SignalEval] NQ signals={signals?.Count ?? 0}");
                if (signals != null && signals.Count > 0)
                {
                    System.Threading.Interlocked.Add(ref S.SignalsTriggered, signals.Count);
                    S.LastSignalAt = DateTimeOffset.UtcNow;
                    S.Phase = BotPhase.Armed;
                }
                foreach (var sig in signals ?? new List<Signal>()) {
                    if (sig.ExpR < 1.0m) continue;
                    var side = sig.Side == "BUY" ? 0 : 1;
                    Task.Run(async () => {
                        S.Phase = BotPhase.PlacingOrder;
                        var orderId = await apiClient.PlaceLimit(accountId, nq.id, side, sig.Size, sig.Entry);
                        if (orderId.HasValue)
                        {
                            System.Threading.Interlocked.Increment(ref S.OrdersPlaced);
                            S.LastOrderAt = DateTimeOffset.UtcNow;
                        }
                        S.Phase = BotPhase.WaitingFill;
                        Console.WriteLine($"[StrategyOrder] NQ {sig.Side} id={orderId} entry={sig.Entry} R={sig.ExpR:0.00}");
                    });
                }
            };
            // ...existing code...
        }
        // ...existing code...
    }

        public static async Task Main(string[] args)
                // --- Wire BarAggregator event handlers after runner state is initialized ---
                barAggES.OnBar += bar => {
                    barsES.Add(bar);
                    Console.WriteLine($"[BarGen] ES bar: start={bar.Start:HH:mm:ss} close={bar.Close:0.00} vol={bar.Volume}");
                    System.Threading.Interlocked.Increment(ref S.SignalsEvaluated);
                    var snap = new MarketSnapshot { Symbol = "ES", UtcNow = DateTime.UtcNow };
                    var signals = strategyAgent.RunAll(snap, barsES, riskEngine);
                    {
                        System.Threading.Interlocked.Add(ref S.SignalsTriggered, signals.Count);
                        S.LastSignalAt = DateTimeOffset.UtcNow;
                        S.Phase = BotPhase.Armed;
                    }
                    foreach (var sig in signals ?? new List<Signal>()) {
                        if (sig.ExpR < 1.0m) continue;
                        var side = sig.Side == "BUY" ? 0 : 1;
                        Task.Run(async () => {
                            S.Phase = BotPhase.PlacingOrder;
                            var orderId = await apiClient.PlaceLimit(accountId, nq.id, side, sig.Size, sig.Entry);
                            if (orderId.HasValue)
                            {
                                System.Threading.Interlocked.Increment(ref S.OrdersPlaced);
                                S.LastOrderAt = DateTimeOffset.UtcNow;
                            }
                            S.Phase = BotPhase.WaitingFill;
                            Console.WriteLine($"[StrategyOrder] NQ {sig.Side} id={orderId} entry={sig.Entry} R={sig.ExpR:0.00}");
                        });
                    }
                };
        {
            Console.WriteLine("[Boot] TopstepX connectivity check starting...");
            LoadEnvFromFile(".env.local");

            // Set up long-running process until Ctrl+C or process exit
            var runTcs = new TaskCompletionSource<object?>();
            Console.CancelKeyPress += (s, e) => { e.Cancel = true; runTcs.TrySetResult(null); };
            try
            {
                Console.WriteLine("[Bot] Starting authentication...");
                var token = await Auth.EnsureSessionTokenAsync();
                Console.WriteLine("[Auth] Session token acquired.");

                // --- Load strategy profile from JSON ---
                var profilePath = "c:\\Users\\kevin\\Downloads\\C# ai bot\\src\\BotCore\\Config\\high_win_rate_profile.json";
                Console.WriteLine($"[Config] Using profile: {profilePath}");
                var config = BotCore.Config.ConfigLoader.FromFile(profilePath);

                // --- Instantiate agents and runner state ---
                var strategyAgent = new StrategyAgent.StrategyAgent(config);
                var riskEngine = new BotCore.Risk.RiskEngine();
                var apiClient = new ApiClient(token);
                var http = new System.Net.Http.HttpClient { BaseAddress = new Uri("https://api.topstepx.com") };
                http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);

                // --- Find a tradable eval account ---
                var accountResp = await http.PostAsJsonAsync("/api/Account/search", new { onlyActiveAccounts = true });
                accountResp.EnsureSuccessStatusCode();
                var accountJson = await accountResp.Content.ReadAsStringAsync();
                Console.WriteLine($"[AccountSearch] Response: {accountJson}");
                var accountDoc = JsonDocument.Parse(accountJson);
                var all = new List<AccountDto>();
                foreach (var a in accountDoc.RootElement.GetProperty("accounts").EnumerateArray())
                {
                    all.Add(new AccountDto {
                        id = a.GetProperty("id").GetInt32(),
                        name = a.GetProperty("name").GetString() ?? "",
                        balance = a.GetProperty("balance").GetDecimal(),
                        canTrade = a.GetProperty("canTrade").GetBoolean(),
                        isVisible = a.GetProperty("isVisible").GetBoolean(),
                        simulated = a.GetProperty("simulated").GetBoolean()
                    });
                }
                var evalAccount = all.FirstOrDefault(a => a.simulated && a.canTrade);
                if (evalAccount == null) throw new Exception("No tradable SIM account found.");
                int accountId = evalAccount.id;

                // --- Live market data and heartbeat ---
                bool live = !evalAccount.simulated ? true : false;
                var targets = new[] { "ES", "NQ" };
                var contracts = await LegacyRunner.ContractResolver.ResolveContractsAsync(http, live, targets);

                if (!contracts.TryGetValue("ES", out var es))
                    throw new InvalidOperationException("ES not found from /available (live=false).");
                if (!contracts.TryGetValue("NQ", out var nq))
                    throw new InvalidOperationException("NQ not found from /available (live=false).");

                Console.WriteLine($"Picked SIM front: {es.name} ({es.symbolId}) id={es.id}");
                Console.WriteLine($"Using contractId={es.id}");

                // --- Strategic trading runner state ---
                var barAggES = new BotCore.BarAggregator();
                var barAggNQ = new BotCore.BarAggregator();
                var barsES = new List<BotCore.Models.Bar>();
                var barsNQ = new List<BotCore.Models.Bar>();
                var S = new BotCore.Models.BotState();

                // --- Minimal heartbeat harness ---
                using var cts = new CancellationTokenSource();
                Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };
                await StartMarketHubAndHeartbeatAsync(token, new[] { es.id, nq.id }, cts.Token);
                try { await Task.Delay(Timeout.Infinite, cts.Token); }
                catch (OperationCanceledException) { Console.WriteLine("[Shutdown] canceled"); }
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine($"[Error] {ex}");
                            }
                        }

                        // ---------- helpers for minimal harness ----------
                        static async Task StartMarketHubAndHeartbeatAsync(string jwt, IEnumerable<string> contractIds, CancellationToken ct)
                        {
                            var hub = new HubConnectionBuilder()
                                .WithUrl("https://rtc.topstepx.com/hubs/market", o => {
                                    o.AccessTokenProvider = () => Task.FromResult(jwt)!;
                                    o.SkipNegotiation = true;
                                    o.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                                })
                                .WithAutomaticReconnect()
                                .Build();

                            hub.Reconnecting += _ => { MarketHubUp = false; return Task.CompletedTask; };
                            hub.Reconnected  += _ => { MarketHubUp = true;  return Task.CompletedTask; };
                            hub.Closed       += _ => { MarketHubUp = false; return Task.CompletedTask; };

                            // Any market event means the hub is alive; also try to update "last" for heartbeat
                            hub.On<string, object>("GatewayQuote",  (cid, payload) => { MarketHubUp = true; TryUpdateLast(payload); });
                            hub.On<string, object>("GatewayTrade",  (cid, payload) => { MarketHubUp = true; TryUpdateLast(payload); });

                            await hub.StartAsync(ct);

                            foreach (var cid in contractIds)
                            {
                                await hub.InvokeAsync("SubscribeContractQuotes",      cid, ct);
                                await hub.InvokeAsync("SubscribeContractTrades",      cid, ct);
                                await hub.InvokeAsync("SubscribeContractMarketDepth", cid, ct);
                            }

                            Console.WriteLine("Market hub started. Processing signals...");

                            _ = Task.Run(async () =>
                            {
                                while (!ct.IsCancellationRequested)
                                {
                                    Console.WriteLine($"heartbeat: bars={Bars} last={LastPrice:0.00}");
                                    await Task.Delay(TimeSpan.FromSeconds(10), ct);
                                }
                            }, ct);
                        }

                        static void TryUpdateLast(object payload)
                        {
                            try
                            {
                                var json = JsonSerializer.Serialize(payload);
                                using var doc = JsonDocument.Parse(json);
                                var r = doc.RootElement;

                                if (r.TryGetProperty("lastPrice", out var lp) && lp.TryGetDecimal(out var d)) LastPrice = d;
                                else if (r.TryGetProperty("price", out var p)   && p.TryGetDecimal(out d))     LastPrice = d;
                                else if (r.TryGetProperty("ask", out var a)     && a.TryGetDecimal(out d))     LastPrice = d;
                                else if (r.TryGetProperty("bid", out var b)     && b.TryGetDecimal(out d))     LastPrice = d;
                            }
                            catch { /* ignore parse errors for heartbeat */ }
                        }

                        private static async Task ProbeMarketHubAsync(string token)
                        {
                            using var http = new HttpClient();
                            var req = new HttpRequestMessage(HttpMethod.Post,
                                "https://rtc.topstepx.com/hubs/market/negotiate?negotiateVersion=1");
                            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
                            req.Content = new StringContent("");
                            var resp = await http.SendAsync(req);
                            var body = await resp.Content.ReadAsStringAsync();
                            Console.WriteLine($"[Negotiate] {resp.StatusCode} {body}");
                        }

                        private static void LoadEnvFromFile(string path)
                        {
                            try
                            {
                                if (!System.IO.File.Exists(path)) return;
                                foreach (var line in System.IO.File.ReadAllLines(path))
                                {
                                    if (string.IsNullOrWhiteSpace(line)) continue;
                                    var trimmed = line.Trim();
                                    if (trimmed.StartsWith("#")) continue;
                                    var idx = trimmed.IndexOf('=');
                                    if (idx <= 0) continue;
                                    var key = trimmed.Substring(0, idx).Trim();
                                    var value = trimmed.Substring(idx + 1).Trim();
                                    Environment.SetEnvironmentVariable(key, value);
                                }
                                Console.WriteLine("[Env] .env.local loaded.");
                            }
                            catch (Exception ex)
                            {
                                Console.WriteLine("[Env] load failed: " + ex.Message);
                            }
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(15) };
            var body = new { userName, apiKey };
            var r = await http.PostAsJsonAsync("https://api.topstepx.com/api/Auth/loginKey", body);
            var txt = await r.Content.ReadAsStringAsync();
            if (!r.IsSuccessStatusCode)
            {
                Console.WriteLine($"[Auth/loginKey] {(int)r.StatusCode} {r.ReasonPhrase} {txt}");
                r.EnsureSuccessStatusCode();
            }
            using var doc = JsonDocument.Parse(txt);
            var tok = doc.RootElement.GetProperty("token").GetString();
            if (string.IsNullOrWhiteSpace(tok)) throw new Exception("loginKey returned no token.");
            return tok!;
        }

        public static async Task<string?> ValidateAndRefreshAsync(string token)
        {
            using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
            http.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", token);
            // empty JSON body per docs
            using var r = await http.PostAsync("https://api.topstepx.com/api/Auth/validate",
                                               new StringContent("{}", System.Text.Encoding.UTF8, "application/json"));
            var txt = await r.Content.ReadAsStringAsync();
            if (!r.IsSuccessStatusCode)
            {
                Console.WriteLine($"[Auth/validate] {(int)r.StatusCode} {r.ReasonPhrase} {txt}");
                return null;
            }
            using var doc = JsonDocument.Parse(txt);
            if (doc.RootElement.TryGetProperty("newToken", out var nt) && nt.ValueKind == JsonValueKind.String)
                return nt.GetString(); // refreshed token
            return token; // still valid, no newToken provided
        }
    }

    static class CFG
    {
        // ---- Endpoints (Topstep production) ----
        public const string API        = "https://api.topstepx.com";
        public const string MARKET_HUB = "https://rtc.topstepx.com/hubs/market"; // token-only URL

        // ---- Session / Behavior ----
        public static readonly string[] SYMBOLS = new[] { "ES", "NQ" }; // change as needed
        public const bool LIVE = true;                 // live trading mode
        public static bool FORCE_BUY = false;          // set true to force BUY order for testing
        public const int  UNIT = 2;                    // 1=Sec, 2=Min, 3=Hour, 4=Day, 5=Week, 6=Month
        public const int  UNITNUM = 5;                 // 5-minute bars
        public const int  HISTORY_DAYS = 7;            // bootstrap window
        public const int  LIMIT_FALLBACK = 4000;       // used when date-range returns empty
        public const int  LIMIT_MAX = 20000;           // server-side cap safety
        public const bool VERBOSE_STREAM_LOG = true;   // flip off if chatty
    }

    // ----------------------- REST API Client -----------------------
    sealed class ApiClient : IDisposable
    {
        private readonly HttpClient _http;

        public ApiClient(string bearer)
        {
            _http = new HttpClient();
            _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", bearer);
            _http.Timeout = TimeSpan.FromSeconds(30);
        }

        public void Dispose() => _http.Dispose();

        public async Task<int> GetTradableAccountId()
        {
            var body = new { onlyActiveAccounts = true };
            var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Account/search", body);
            r.EnsureSuccessStatusCode();
            using var doc = JsonDocument.Parse(await r.Content.ReadAsStringAsync());
            foreach (var a in doc.RootElement.GetProperty("accounts").EnumerateArray())
            {
                if (a.GetProperty("canTrade").GetBoolean())
                {
                    int id = a.GetProperty("id").GetInt32();
                    string name = a.GetProperty("name").GetString() ?? "";
                    Console.WriteLine($"[Account] Using {id} ({name})");
                    return id;
                }
            }
            throw new Exception("No tradable account found (canTrade=false).");
        }

    // public async Task<string> ResolveContractId(string search) (removed)
    // (removed old ResolveContractId, now handled by ResolveContractsAsync)

        // Remove old RetrieveBars. Use HistoryApi for robust bar retrieval.

        // Optional: place/cancel helpers (you’ll call later from your strategy)
        public async Task<long?> PlaceLimit(int accountId, string contractId, int sideBuy0Sell1, int size, decimal limitPrice)
        {
            var body = new {
                accountId, contractId,
                type = 1,             // 1 = Limit
                side = sideBuy0Sell1, // 0=Buy,1=Sell
        {
            var body = new { accountId, orderId };
            var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Order/cancel", body);
            Console.WriteLine("[Order/cancel] " + await r.Content.ReadAsStringAsync());
        }
    }

    // ----------------------- Robust Market Hub Client -----------------------

    public sealed class MarketHubClient : IAsyncDisposable
    {
        private readonly HubConnection _conn;
        private readonly Func<Task<string>> _getTokenAsync;
        private readonly SemaphoreSlim _sendLock = new(1, 1);
        private readonly HashSet<string> _subs = new(StringComparer.OrdinalIgnoreCase);
        private volatile bool _disposed;

        public MarketHubClient(Func<Task<string>> getTokenAsync)
        {
            _getTokenAsync = getTokenAsync;

            _conn = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/market", opts =>
                {
                    opts.AccessTokenProvider = async () => await _getTokenAsync();
                })
                .WithAutomaticReconnect(new[]
                {
                    TimeSpan.Zero, TimeSpan.FromSeconds(2),
                    TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10)
                })
                .Build();

            _conn.ServerTimeout = TimeSpan.FromSeconds(60);
            _conn.KeepAliveInterval = TimeSpan.FromSeconds(15);

            _conn.Reconnecting += ex =>
            {
                Console.WriteLine($"[MarketHub RECONNECTING] {ex?.Message ?? "(no reason)"}");
                return Task.CompletedTask;
            };

            _conn.Reconnected += async newConnId =>
            {
                Console.WriteLine($"[MarketHub RECONNECTED] id={newConnId}");
                await ResubscribeAsync();
            };

            _conn.Closed += ex =>
            {
                Console.WriteLine($"[MarketHub CLOSED] {ex?.Message ?? "(server closed)"}");
                if (_disposed)
                {
                    Console.WriteLine("[MarketHub CLOSED] not restarting because client is disposing.");
                    return Task.CompletedTask;
                }
                _ = Task.Run(async () =>
                {
                    try
                    {
                        await TryStartLoopAsync(default);
                        Console.WriteLine("[MarketHub CLOSED] restart attempt completed.");
                    }
                    catch (Exception rex)
                    {
                        Console.WriteLine("[MarketHub CLOSED] restart failed: " + rex.Message);
                    }
                });
                return Task.CompletedTask;
            };

            _conn.On<string>("Heartbeat", hb => Console.WriteLine($"[HB] {hb}"));
            // Add more handlers as needed
        }

        public void OnHeartbeat(Action<string> handler)
        {
            _conn.On<string>("Heartbeat", handler);
        }

        public async Task StartAsync(CancellationToken ct = default)
        {
            await TryStartLoopAsync(ct);
            Console.WriteLine("[MarketHub] connected.");
        }

        public async Task SubscribeAllAsync(IEnumerable<string> contractIds, CancellationToken ct = default)
        {
            foreach (var id in contractIds) _subs.Add(id);
            await InvokeSafeAsync("SubscribeInstruments", new object?[] { _subs.ToArray() }, ct);
            Console.WriteLine($"[MarketHub] subscribed: {string.Join(", ", _subs)}");
        }

        public void OnGatewayTrade(Action<string, System.Text.Json.JsonElement> handler)
        {
            _conn.On<string, System.Text.Json.JsonElement>("GatewayTrade", handler);
        }

        public async Task SubscribeTradesAsync(string contractId, CancellationToken ct = default)
        {
            await InvokeSafeAsync("SubscribeContractTrades", new object?[] { contractId }, ct);
            Console.WriteLine($"[MarketHub] SubscribeContractTrades -> {contractId}");
        }

        private async Task ResubscribeAsync(CancellationToken ct = default)
        {
            if (_subs.Count == 0) return;
            await InvokeSafeAsync("SubscribeInstruments", new object?[] { _subs.ToArray() }, ct);
            Console.WriteLine("[MarketHub] resubscribed after reconnect.");
        }

        private async Task TryStartLoopAsync(CancellationToken ct = default)
        {
            while (_conn.State != HubConnectionState.Connected && !ct.IsCancellationRequested)
            {
                try
                {
                    Console.WriteLine("[MarketHub START] attempting StartAsync...");
                    await _conn.StartAsync(ct);
                    Console.WriteLine("[MarketHub START] StartAsync successful.");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[MarketHub START RETRY] {ex.Message}");
                    await Task.Delay(1000, ct);
                    continue;
                }
            }
        }

        private async Task InvokeSafeAsync(string method, object?[] args, CancellationToken ct)
        {
            await EnsureConnectedAsync(ct);
            await _sendLock.WaitAsync(ct);
            try
            {
                await _conn.InvokeCoreAsync(method, typeof(void), args, ct);
            }
            catch (InvalidOperationException ioe) when (_conn.State != HubConnectionState.Connected)
            {
                Console.WriteLine($"[MarketHub INVOKE RACE] {ioe.Message}");
                await EnsureConnectedAsync(ct);
                await _conn.InvokeCoreAsync(method, typeof(void), args, ct);
            }
            finally
            {
                _sendLock.Release();
            }
        }

        private async Task EnsureConnectedAsync(CancellationToken ct)
        {
            if (_conn.State == HubConnectionState.Connected) return;
            await TryStartLoopAsync(ct);
        }

        public async Task StopAsync()
        {
            _disposed = true;
            try
            {
                await _conn.StopAsync();
            }
            catch { /* ignore */ }
        }

        public async ValueTask DisposeAsync()
        {
            _disposed = true;
            _sendLock.Dispose();
            await _conn.DisposeAsync();
        }
    }



    // ...existing code...
        }

        private static void TryUpdateLast(object payload)
        {
            try
            {
                var json = JsonSerializer.Serialize(payload);
                using var doc = JsonDocument.Parse(json);
                var r = doc.RootElement;

                if (r.TryGetProperty("lastPrice", out var lp) && lp.TryGetDecimal(out var d)) LastPrice = d;
                else if (r.TryGetProperty("price", out var p)   && p.TryGetDecimal(out d))     LastPrice = d;
                else if (r.TryGetProperty("ask", out var a)     && a.TryGetDecimal(out d))     LastPrice = d;
                else if (r.TryGetProperty("bid", out var b)     && b.TryGetDecimal(out d))     LastPrice = d;
            }
            catch { /* ignore parse errors for heartbeat */ }
        }
// ---------- STATE FOR HEARTBEAT ----------



                // --- Strategic trading: wire BarAggregator, run StrategyAgent, place trades only on signals ---
                // ...existing code...
                barAggES.OnBar += bar => {
                    barsES.Add(bar);
                    Console.WriteLine($"[BarGen] ES bar: start={bar.Start:HH:mm:ss} close={bar.Close:0.00} vol={bar.Volume}");
                    System.Threading.Interlocked.Increment(ref S.SignalsEvaluated);
                    var snap = new BotCore.Config.MarketSnapshot { Symbol = "ES", UtcNow = DateTime.UtcNow };
                    var signals = strategyAgent.RunAll(snap, barsES, riskEngine);
                    Console.WriteLine($"[SignalEval] ES signals={signals?.Count ?? 0}");
                    if (signals != null && signals.Count > 0)
                    {
                        System.Threading.Interlocked.Add(ref S.SignalsTriggered, signals.Count);
                        S.LastSignalAt = DateTimeOffset.UtcNow;
                        S.Phase = BotPhase.Armed;
                    }
                    foreach (var sig in signals) {
                        if (sig.ExpR < 1.0m) continue;
                        var side = sig.Side == "BUY" ? 0 : 1;
                        Task.Run(async () => {
                            S.Phase = BotPhase.PlacingOrder;
                            var orderId = await apiClient.PlaceLimit(accountId, es.id, side, sig.Size, sig.Entry);
                            if (orderId.HasValue)
                            {
                                System.Threading.Interlocked.Increment(ref S.OrdersPlaced);
                                S.LastOrderAt = DateTimeOffset.UtcNow;
                            }
                            S.Phase = BotPhase.WaitingFill;
                            Console.WriteLine($"[StrategyOrder] ES {sig.Side} id={orderId} entry={sig.Entry} R={sig.ExpR:0.00}");
                        });
                    }
                };
                barAggNQ.OnBar += bar => {
                    barsNQ.Add(bar);
                    Console.WriteLine($"[BarGen] NQ bar: start={bar.Start:HH:mm:ss} close={bar.Close:0.00} vol={bar.Volume}");
                    System.Threading.Interlocked.Increment(ref S.SignalsEvaluated);
                    var snap = new BotCore.Config.MarketSnapshot { Symbol = "NQ", UtcNow = DateTime.UtcNow };
                    var signals = strategyAgent.RunAll(snap, barsNQ, riskEngine);
                    Console.WriteLine($"[SignalEval] NQ signals={signals?.Count ?? 0}");
                    if (signals != null && signals.Count > 0)
                    {
                        System.Threading.Interlocked.Add(ref S.SignalsTriggered, signals.Count);
                        S.LastSignalAt = DateTimeOffset.UtcNow;
                        S.Phase = BotPhase.Armed;
                    }
                    foreach (var sig in signals) {
                        if (sig.ExpR < 1.0m) continue;
                        var side = sig.Side == "BUY" ? 0 : 1;
                        Task.Run(async () => {
                            S.Phase = BotPhase.PlacingOrder;
                            var orderId = await apiClient.PlaceLimit(accountId, nq.id, side, sig.Size, sig.Entry);
                            if (orderId.HasValue)
                            {
                                System.Threading.Interlocked.Increment(ref S.OrdersPlaced);
                                S.LastOrderAt = DateTimeOffset.UtcNow;
                            }
                            S.Phase = BotPhase.WaitingFill;
                            Console.WriteLine($"[StrategyOrder] NQ {sig.Side} id={orderId} entry={sig.Entry} R={sig.ExpR:0.00}");
                        });
                    }
                };

        marketHub.On<string, object>("GatewayTrade", (contractId, payload) =>
        {
            MarketHubUp = true;
            try
            {
                var json = System.Text.Json.JsonSerializer.Serialize(payload);
                using var doc = System.Text.Json.JsonDocument.Parse(json);
                var r = doc.RootElement;
                DateTimeOffset ts =
                    r.TryGetProperty("time", out var tEl) && tEl.ValueKind == JsonValueKind.String && DateTimeOffset.TryParse(tEl.GetString(), out var tVal) ? tVal :
                    r.TryGetProperty("timestamp", out var tsEl) && tsEl.ValueKind == JsonValueKind.Number ? DateTimeOffset.FromUnixTimeMilliseconds(tsEl.GetInt64()) :
                    DateTimeOffset.UtcNow;
                decimal px =
                    r.TryGetProperty("price", out var pEl) && pEl.TryGetDecimal(out var pVal) ? pVal :
                    r.TryGetProperty("lastPrice", out var lpEl) && lpEl.TryGetDecimal(out var lpVal) ? lpVal : 0m;
                decimal sz =
                    r.TryGetProperty("size", out var sEl) && sEl.TryGetDecimal(out var sVal) ? sVal :
                    r.TryGetProperty("qty",  out var qEl) && qEl.TryGetDecimal(out var qVal) ? qVal :
                    r.TryGetProperty("volume", out var vEl) && vEl.TryGetDecimal(out var vVal) ? vVal : 0m;
                Console.WriteLine($"[TradeTick] {contractId} {ts:HH:mm:ss.fff} px={px} sz={sz}");
                if (contractId == es.id) barAggES.OnTrade(ts.UtcDateTime, px, (int)sz);
                if (contractId == nq.id) barAggNQ.OnTrade(ts.UtcDateTime, px, (int)sz);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[TradeTick] parse error: {ex.Message}");
                Console.WriteLine($"[TradeTickRaw] {System.Text.Json.JsonSerializer.Serialize(payload)}");
            }
        });
        marketHub.Reconnected += async _ =>
        {
            Console.WriteLine("[MarketHub] reconnected — resubscribing trades");
            await SubscribeTradesAsync(marketHub, contractIds);
        };
        _ = Task.Run(async () =>
        {
            Console.WriteLine("[HB] started");
            while (true)
            {
                Console.WriteLine($"[Status] MarketHubUp={MarketHubUp} (expect True once ticks arrive)");
                await Task.Delay(TimeSpan.FromSeconds(15));
            }
        });

                // --- Wire UserHub for order/fill events ---
                var userHub = new HubConnectionBuilder()
                    .WithUrl("https://rtc.topstepx.com/hubs/user", o => {
                        o.AccessTokenProvider = () => Task.FromResult(token)!;
                        o.SkipNegotiation = true;
                        o.Transports = Microsoft.AspNetCore.Http.Connections.HttpTransportType.WebSockets;
                    })
                    .WithAutomaticReconnect()
                    .Build();

                await userHub.StartAsync();
                await userHub.InvokeAsync("SubscribeAccounts");
                await userHub.InvokeAsync("SubscribeOrders",   accountId);
                await userHub.InvokeAsync("SubscribePositions",accountId);
                await userHub.InvokeAsync("SubscribeTrades",   accountId);
                Console.WriteLine($"[Diagnostic] Subscribing to contracts: ES={es.id} NQ={nq.id}");
                Console.WriteLine("[Diagnostic] MarketHub and UserHub subscriptions are active. Bot is running.");

                userHub.On<object>("GatewayUserOrder", o => {
                    S.UserHubUp = true;
                    var json = System.Text.Json.JsonSerializer.Serialize(o);
                    using var doc = System.Text.Json.JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    long orderId = root.TryGetProperty("id", out var idEl) && idEl.TryGetInt64(out var idVal) ? idVal : 0;
                    string status = root.TryGetProperty("status", out var stEl) ? stEl.GetString() ?? "" : "";
                    string contractId = root.TryGetProperty("contractId", out var cEl) ? cEl.GetString() ?? "" : "";
                    if (orderId > 0)
                    {
                        if (status is "Working" or "Accepted" or "PendingNew" or "PartiallyFilled")
                            S.OpenOrders[orderId] = (contractId, status);
                        if (status is "Filled" or "Cancelled" or "Rejected" or "Expired")
                            S.OpenOrders.TryRemove(orderId, out _);
                    }
                    Console.WriteLine($"[OrderEvent] {o}");
                });
                userHub.On<object>("GatewayUserPosition", p => {
                    var json = System.Text.Json.JsonSerializer.Serialize(p);
                    using var doc = System.Text.Json.JsonDocument.Parse(json);
                    var root = doc.RootElement;
                    string contractId = root.TryGetProperty("contractId", out var cEl) ? cEl.GetString() ?? "" : "";
                    decimal qty = root.TryGetProperty("netPosition", out var qEl) ? qEl.GetDecimal() : 0m;
                    if (!string.IsNullOrWhiteSpace(contractId))
                    {
                        if (qty == 0) S.OpenPositionsByContract.TryRemove(contractId, out _);
                        else S.OpenPositionsByContract[contractId] = qty;
                    }
                });
                userHub.On<object>("GatewayUserTrade", t => {
                    Interlocked.Increment(ref S.Fills);
                    S.LastFillAt = DateTimeOffset.UtcNow;
                    Console.WriteLine($"[Fill] {t}");
                });
                // Heartbeat: print bot status every 30 seconds
                Console.WriteLine("[Heartbeat] Starting status print loop...");
                _ = Task.Run(async () => {
                    try {
                        // Immediate status print for diagnostics
                        var phase = S.IsTrading ? BotPhase.ManagingPosition : (S.Phase == BotPhase.Startup ? BotPhase.Scanning : S.Phase);
                        var openOrders = S.OpenOrders.Count;
                        var openPos = S.OpenPositionsByContract.Count;
                        Console.WriteLine(
                            $"[Status] phase={phase} hub(M={S.MarketHubUp},U={S.UserHubUp}) " +
                            $"signals eval={S.SignalsEvaluated} trig={S.SignalsTriggered} " +
                            $"orders open={openOrders} placed={S.OrdersPlaced} fills={S.Fills} " +
                            $"pos open={openPos} " +
                            $"last(signal={S.LastSignalAt:HH:mm:ss}, order={S.LastOrderAt:HH:mm:ss}, fill={S.LastFillAt:HH:mm:ss})");
                        while (true)
                        {
                            phase = S.IsTrading ? BotPhase.ManagingPosition : (S.Phase == BotPhase.Startup ? BotPhase.Scanning : S.Phase);
                            openOrders = S.OpenOrders.Count;
                            openPos = S.OpenPositionsByContract.Count;
                            Console.WriteLine(
                                $"[Status] phase={phase} hub(M={S.MarketHubUp},U={S.UserHubUp}) " +
                                $"signals eval={S.SignalsEvaluated} trig={S.SignalsTriggered} " +
                                $"orders open={openOrders} placed={S.OrdersPlaced} fills={S.Fills} " +
                                $"pos open={openPos} " +
                                $"last(signal={S.LastSignalAt:HH:mm:ss}, order={S.LastOrderAt:HH:mm:ss}, fill={S.LastFillAt:HH:mm:ss})");
                            await Task.Delay(TimeSpan.FromSeconds(30));
                        }
                    } catch (Exception ex) {
                        Console.WriteLine($"[HeartbeatError] {ex}");
                    }
                });

                Console.WriteLine($"[Live] Subscribed to ES and NQ market data. Orders placed. Press Ctrl+C to exit.");
                await Task.Delay(TimeSpan.FromMinutes(5));
            }
            catch (Exception) { /* suppress legacy pipeline errors */ }
        }

        private static async Task ProbeMarketHubAsync(string token)
        {
            using var http = new HttpClient();
            var req = new HttpRequestMessage(HttpMethod.Post,
                "https://rtc.topstepx.com/hubs/market/negotiate?negotiateVersion=1");
            req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
            req.Content = new StringContent("");
            var resp = await http.SendAsync(req);
            var body = await resp.Content.ReadAsStringAsync();
            Console.WriteLine($"[Negotiate] {resp.StatusCode} {body}");
        }

        private static void LoadEnvFromFile(string path)
        {
            try
            {
                if (!System.IO.File.Exists(path)) return;
                foreach (var line in System.IO.File.ReadAllLines(path))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var trimmed = line.Trim();
                    if (trimmed.StartsWith("#")) continue;
                    var idx = trimmed.IndexOf('=');
                    if (idx <= 0) continue;
                    var key = trimmed.Substring(0, idx).Trim();
                    var value = trimmed.Substring(idx + 1).Trim();
                    Environment.SetEnvironmentVariable(key, value);
                }
                Console.WriteLine("[Env] .env.local loaded.");
            }
            catch (Exception ex)
            {
                Console.WriteLine("[Env] load failed: " + ex.Message);
            }
        }
    }
}