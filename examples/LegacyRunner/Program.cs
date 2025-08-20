using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.SignalR.Client;
using BotCore.Config;
using BotCore.Models;
using BotCore.Strategy;
using StrategyAgent;

namespace LegacyRunner
{
        public static class ContractResolver
        {
            public sealed class ContractHit
            {
                public string id { get; set; } = "";
                public string name { get; set; } = "";
                public string description { get; set; } = "";
                public bool activeContract { get; set; }
                public string symbolId { get; set; } = "";
                public decimal tickSize { get; set; }
                public decimal tickValue { get; set; }
            }
            public sealed class ContractListResponse
            {
                public List<ContractHit>? contracts { get; set; }
                public bool success { get; set; }
                public int errorCode { get; set; }
                public string? errorMessage { get; set; }
            }

            public static readonly Dictionary<string, string> RootToSymbol = new(StringComparer.OrdinalIgnoreCase)
            {
                ["ES"]  = "F.US.EP",
                ["NQ"]  = "F.US.ENQ",
                ["MES"] = "F.US.MES",
                ["MNQ"] = "F.US.MNQ"
            };

            public static async Task<Dictionary<string, ContractHit>> ResolveContractsAsync(HttpClient http, bool live, IEnumerable<string> roots, CancellationToken ct = default)
            {
                var body = new Dictionary<string, object> { ["live"] = live };
                using var res = await http.PostAsJsonAsync("/api/Contract/available", body, ct);
                res.EnsureSuccessStatusCode();
                var doc = await res.Content.ReadFromJsonAsync<ContractListResponse>(cancellationToken: ct);
                var list = doc?.contracts ?? new();

                var map = new Dictionary<string, ContractHit>(StringComparer.OrdinalIgnoreCase);
                foreach (var root in roots)
                {
                    var want = RootToSymbol.TryGetValue(root, out var s) ? s : root;
                    var pick = list
                        .Where(c => c.activeContract && c.symbolId == want)
                        .OrderByDescending(c => c.name)
                        .FirstOrDefault();

                    if (pick != null) map[root] = pick;
                }
                return map;
            }
    }

    // ----------------------- Auth Helpers -----------------------
    static class Auth
    {
        public static async Task<string> EnsureSessionTokenAsync()
        {
            var token = Environment.GetEnvironmentVariable("PROJECTX_TOKEN");
            if (!string.IsNullOrWhiteSpace(token))
            {
                var refreshed = await ValidateAndRefreshAsync(token);
                if (!string.IsNullOrEmpty(refreshed)) {
                    Environment.SetEnvironmentVariable("PROJECTX_TOKEN", refreshed);
                    return refreshed;
                }
                // if validate didn’t return newToken, keep the old one
                return token;
            }

            var user = Environment.GetEnvironmentVariable("PROJECTX_USERNAME")
                       ?? Environment.GetEnvironmentVariable("TSX_USERNAME");
            var key  = Environment.GetEnvironmentVariable("PROJECTX_APIKEY")
                       ?? Environment.GetEnvironmentVariable("TSX_API_KEY");
            if (string.IsNullOrWhiteSpace(user) || string.IsNullOrWhiteSpace(key))
                throw new Exception("Set either PROJECTX_TOKEN (session JWT) or both PROJECTX_USERNAME/TSX_USERNAME and PROJECTX_APIKEY/TSX_API_KEY.");

            var newToken = await LoginWithApiKeyAsync(user, key);
            Environment.SetEnvironmentVariable("PROJECTX_TOKEN", newToken);
            return newToken;
        }

        public static async Task<string> LoginWithApiKeyAsync(string userName, string apiKey)
        {
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
                size,
                limitPrice
            };
            var r = await _http.PostAsJsonAsync($"{CFG.API}/api/Order/place", body);
            var txt = await r.Content.ReadAsStringAsync();
            Console.WriteLine("[Order/place] " + txt);
            if (!r.IsSuccessStatusCode) return null;

            try {
                using var doc = JsonDocument.Parse(txt);
                bool success = doc.RootElement.TryGetProperty("success", out var s) && s.GetBoolean();
                if (!success) return null;
                return doc.RootElement.TryGetProperty("orderId", out var id) ? id.GetInt64() : null;
            } catch { return null; }
        }

        public async Task CancelOrder(int accountId, long orderId)
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



    public sealed partial class Program
    {
        // ==== AUTO-EXECUTE STATE ====
        private static volatile bool MarketHubUp;
        private static volatile bool UserHubUp;
        private static long BarsSeen;

        private static bool EnvTrue(string n)
        {
            var v = (Environment.GetEnvironmentVariable(n) ?? "").Trim().ToLowerInvariant();
            return v is "1" or "true" or "yes";
        }

        private static bool ReadyToArm(bool canTrade, string? contractId)
            => canTrade
               && !string.IsNullOrWhiteSpace(contractId)
               && MarketHubUp && UserHubUp
               && BarsSeen >= 10; // saw at least 10 trade ticks → data flowing

        private static void Heartbeat(Func<decimal> getLastClose)
        {
            _ = Task.Run(async () =>
            {
                while (true)
                {
                    await Task.Delay(TimeSpan.FromSeconds(10));
                    var bars = System.Threading.Interlocked.Read(ref BarsSeen);
                    Console.WriteLine($"heartbeat: bars={bars} last={getLastClose():0.00}");
                }
            });
        }

        // Supporting type for account search
        public class AccountListResponse {
            public List<AccountDto>? accounts { get; set; }
        }
        public class AccountDto
        {
            public int id { get; set; }
            public string name { get; set; } = "";
            public decimal balance { get; set; }
            public bool canTrade { get; set; }
            public bool isVisible { get; set; }
            public bool simulated { get; set; }
        }

        static bool IsPractice(AccountDto a) =>
            a.name.StartsWith("PRAC-", StringComparison.OrdinalIgnoreCase);

        static AccountDto? PickEvaluationAccount(IEnumerable<AccountDto> accts)
        {
            // Evaluation/Combine: simulated + canTrade + not PRAC
            return accts
                .Where(a => a.canTrade && a.isVisible && a.simulated && !IsPractice(a))
                .OrderByDescending(a => a.balance)
                .FirstOrDefault();
        }

        static AccountDto? PickPracticeAccount(IEnumerable<AccountDto> accts)
        {
            return accts
                .Where(a => a.canTrade && a.isVisible && a.simulated && IsPractice(a))
                .OrderByDescending(a => a.balance)
                .FirstOrDefault();
        }

        // ----------------------- Program Entry -----------------------
        // Only one Program class should exist at the end of the file. All entry logic is inside it.
        public static async Task Main(string[] args)
        {
            Console.WriteLine("[Boot] TopstepX connectivity check starting...");
            LoadEnvFromFile(".env.local");

            // Set up long-running process until Ctrl+C or process exit
            var runTcs = new TaskCompletionSource<object?>();
            Console.CancelKeyPress += (s, e) => { e.Cancel = true; runTcs.TrySetResult(null); };
            AppDomain.CurrentDomain.ProcessExit += (s, e) => runTcs.TrySetResult(null);
            try
            {
                Console.WriteLine("[Bot] Starting authentication...");
                var token = await Auth.EnsureSessionTokenAsync();
                Console.WriteLine("[Auth] Session token acquired.");

                // --- Load strategy profile from JSON ---
                var profilePath = "c:\\Users\\kevin\\Downloads\\C# ai bot\\src\\BotCore\\Config\\high_win_rate_profile.json";
                Console.WriteLine($"[Config] Using profile: {profilePath}");
                var config = BotCore.Config.ConfigLoader.FromFile(profilePath);

                // --- Instantiate agents ---
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
                var accountList = System.Text.Json.JsonSerializer.Deserialize<AccountListResponse>(accountJson);
                var all = accountList?.accounts ?? new List<AccountDto>();
                AccountDto? evalAccount = PickEvaluationAccount(all);
                int accountId;
                if (evalAccount == null)
                {
                    var prac = PickPracticeAccount(all);
                    if (prac != null)
                    {
                        Console.WriteLine($"[Account] Using PRACTICE {prac.name} ({prac.id}) bal={prac.balance}");
                        accountId = prac.id;
                    }
                    else
                    {
                        foreach (var a in all)
                            Console.WriteLine($"[AccountDebug] {a.name} canTrade={a.canTrade} visible={a.isVisible} simulated={a.simulated}");
                        throw new InvalidOperationException("No tradable eval account found.");
                    }
                }
                else
                {
                    Console.WriteLine($"[Account] Using EVAL {evalAccount.name} ({evalAccount.id}) bal={evalAccount.balance}");
                    accountId = evalAccount.id;
                }

                // --- Live market data and heartbeat ---
                bool live = !evalAccount.simulated;
                Console.WriteLine($"[Debug] live={live} (account simulated={evalAccount.simulated})");
                var targets = new[] { "ES", "NQ" };
                var contracts = await LegacyRunner.ContractResolver.ResolveContractsAsync(http, live, targets);

                if (!contracts.TryGetValue("ES", out var es))
                    throw new InvalidOperationException("ES not found from /available (live=false).");
                if (!contracts.TryGetValue("NQ", out var nq))
                    throw new InvalidOperationException("NQ not found from /available (live=false).");

                Console.WriteLine($"[Contract] ES {es.name} id={es.id} tick={es.tickSize} value={es.tickValue}");
                Console.WriteLine($"[Contract] NQ {nq.name} id={nq.id} tick={nq.tickSize} value={nq.tickValue}");

                var marketHub = new BotCore.MarketHubClient(token);
                var barAggES = new BotCore.BarAggregator(TimeSpan.FromMinutes(1));
                var barAggNQ = new BotCore.BarAggregator(TimeSpan.FromMinutes(1));
                decimal lastPriceES = 0, lastPriceNQ = 0;
                int tickCountES = 0, tickCountNQ = 0;

                marketHub.OnTrade += (price, vol) =>
                {
                    // Print generic tick for both ES and NQ
                    Console.WriteLine($"[TICK] {DateTime.UtcNow:O} price={price} vol={vol}");
                    barAggES.OnTrade(DateTime.UtcNow, price, vol);
                    barAggNQ.OnTrade(DateTime.UtcNow, price, vol);
                };
                barAggES.OnBar += bar =>
                {
                    Console.WriteLine($"[ES BAR] {bar.Start:O} O={bar.Open} H={bar.High} L={bar.Low} C={bar.Close} V={bar.Volume}");
                };
                barAggNQ.OnBar += bar =>
                {
                    Console.WriteLine($"[NQ BAR] {bar.Start:O} O={bar.Open} H={bar.High} L={bar.Low} C={bar.Close} V={bar.Volume}");
                };

                foreach (var contract in new[] { es, nq })
                {
                    await marketHub.StartAsync(contract.id, CancellationToken.None);
                }

                Console.WriteLine($"[Live] Subscribed to ES and NQ market data. Press Ctrl+C to exit.");
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
