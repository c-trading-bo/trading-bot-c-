#nullable enable
// Agent: OrchestratorAgent (standalone)
// Role: Standalone orchestration logic for strategy and agent coordination.
// Integration: Manages bot lifecycle and agent interactions.
// PURPOSE: Evaluation-account policy configuration and simple session window helpers.
using System.Globalization;
using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR.Client;

using Microsoft.Extensions.Logging;
using BotCore.Auth;

namespace OrchestratorAgent
{
    public sealed class EvalPolicy
    {
        public bool Enabled { get; set; } = true;
        public decimal MaxDailyLoss { get; set; } = 850m;
        public int MaxContractsPerSymbol { get; set; } = 1;
        public int MaxTotalContracts { get; set; } = 2;
        public string Timezone { get; set; } = "America/Chicago";
        public string Session { get; set; } = "05:00-15:00";                 // HH:mm-HH:mm in local TZ
        public string TradingDayRolloverLocalTime { get; set; } = "17:00";    // 5:00pm local (futures convention)

        public static EvalPolicy FromEnv()
        {
            var p = new EvalPolicy();
            if (decimal.TryParse(Environment.GetEnvironmentVariable("EVAL_MAX_DAILY_LOSS"), NumberStyles.Any, CultureInfo.InvariantCulture, out var mdl))
                p.MaxDailyLoss = mdl;
            if (int.TryParse(Environment.GetEnvironmentVariable("EVAL_MAX_CONTRACTS_PER_SYMBOL"), out var mcs))
                p.MaxContractsPerSymbol = Math.Max(1, mcs);
            if (int.TryParse(Environment.GetEnvironmentVariable("EVAL_MAX_TOTAL_CONTRACTS"), out var mtc))
                p.MaxTotalContracts = Math.Max(1, mtc);
            p.Timezone = Environment.GetEnvironmentVariable("EVAL_TIMEZONE") ?? p.Timezone;
            p.Session = Environment.GetEnvironmentVariable("EVAL_SESSION") ?? p.Session;
            p.TradingDayRolloverLocalTime = Environment.GetEnvironmentVariable("EVAL_ROLLOVER") ?? p.TradingDayRolloverLocalTime;
            return p;
        }

        public bool IsWithinSession(DateTimeOffset nowUtc)
        {
            try
            {
                var tz = TimeZoneInfo.FindSystemTimeZoneById(Timezone);
                var local = TimeZoneInfo.ConvertTime(nowUtc, tz);
                var parts = Session.Split('-', 2);
                var start = TimeSpan.Parse(parts[0], CultureInfo.InvariantCulture);
                var end = TimeSpan.Parse(parts[1], CultureInfo.InvariantCulture);
                return local.TimeOfDay >= start && local.TimeOfDay <= end;
            }
            catch
            {
                return true; // fail-open if misconfigured
            }
        }

        public DateTimeOffset GetTradingDayStart(DateTimeOffset nowUtc)
        {
            try
            {
                var tz = TimeZoneInfo.FindSystemTimeZoneById(Timezone);
                var local = TimeZoneInfo.ConvertTime(nowUtc, tz);
                var t = TimeSpan.Parse(TradingDayRolloverLocalTime, CultureInfo.InvariantCulture); // e.g., 17:00
                var rolloverToday = new DateTime(local.Year, local.Month, local.Day, t.Hours, t.Minutes, 0, DateTimeKind.Unspecified);
                var startLocal = local >= rolloverToday ? rolloverToday : rolloverToday.AddDays(-1);
                return TimeZoneInfo.ConvertTimeToUtc(startLocal, tz);
            }
            catch
            {
                // default to UTC midnight if timezone parse fails
                var d = nowUtc.Date;
                return new DateTimeOffset(d, TimeSpan.Zero);
            }
        }
    }

    public static class SymbolMeta
    {
        private sealed record Meta(decimal TickSize, decimal TickValueUSD);

        // ES: 0.25 tick = $12.50; NQ: 0.25 tick = $5.00
        private static readonly ConcurrentDictionary<string, Meta> _map = new(StringComparer.OrdinalIgnoreCase)
        {
            ["ES"] = new Meta(0.25m, 12.50m),
            ["NQ"] = new Meta(0.25m, 5.00m)
        };

        public static void Set(string root, decimal tickSize, decimal tickValueUsd)
            => _map[root] = new Meta(tickSize, tickValueUsd);

        public static decimal ToPnlUSD(string root, decimal pointsMoved, int qtySigned)
        {
            if (!_map.TryGetValue(root, out var m)) return 0m;
            var ticks = pointsMoved / m.TickSize;
            return ticks * m.TickValueUSD * qtySigned;
        }

        public static string RootFromName(string? contractNameOrSymbol)
        {
            if (string.IsNullOrWhiteSpace(contractNameOrSymbol)) return "?";
            var s = contractNameOrSymbol.ToUpperInvariant();
            if (s.StartsWith("ES")) return "ES";
            if (s.StartsWith("NQ") || s.Contains("MNQ")) return "NQ";
            return new string(s.TakeWhile(char.IsLetter).ToArray());
        }
    }
    // PURPOSE: Track positions and realized PnL per trading day from trade fills.
    public sealed class PnLTracker
    {
        private readonly object _lock = new();
        private readonly EvalPolicy _policy;

        private sealed class Pos
        {
            public int Qty;                 // signed
            public decimal AvgPrice;        // weighted avg
            public decimal Realized;        // USD
        }

        private readonly ConcurrentDictionary<string, Pos> _pos = new(StringComparer.OrdinalIgnoreCase);
        private DateTimeOffset _dayStartUtc;

        public PnLTracker(EvalPolicy policy)
        {
            _policy = policy;
            _dayStartUtc = policy.GetTradingDayStart(DateTimeOffset.UtcNow);
        }

        public void ResetIfNewDay(DateTimeOffset nowUtc)
        {
            var start = _policy.GetTradingDayStart(nowUtc);
            if (start > _dayStartUtc)
            {
                lock (_lock)
                {
                    _pos.Clear();
                    _dayStartUtc = start;
                }
            }
        }

        public void OnFill(JsonElement tradePayload)
        {
            // Tries common fields: symbol/name, side/direction, qty/quantity, price/fillPrice
            var name = TryStr(tradePayload, "symbol", "sym", "contractName", "name") ?? "?";
            var side = TryStr(tradePayload, "side", "direction") ?? "Buy";
            var qty = TryInt(tradePayload, "qty", "quantity", "filledQty", "fillQty") ?? 0;
            var px = TryDec(tradePayload, "price", "fillPrice", "avgPrice") ?? 0m;

            if (qty == 0 || px <= 0m) return;

            var root = SymbolMeta.RootFromName(name);
            var signed = side.StartsWith("B", StringComparison.OrdinalIgnoreCase) ? qty : -qty;

            lock (_lock)
            {
                var p = _pos.GetOrAdd(root, _ => new Pos());
                // Realized PnL from closing portion
                if (p.Qty != 0 && Math.Sign(p.Qty) != Math.Sign(signed))
                {
                    var closeQty = Math.Min(Math.Abs(p.Qty), Math.Abs(signed)) * Math.Sign(signed); // signed portion
                    var points = (px - p.AvgPrice) * Math.Sign(closeQty); // positive when profitable for the closing direction
                    var usd = SymbolMeta.ToPnlUSD(root, points, Math.Abs(closeQty));
                    p.Realized += usd;
                }

                // Update position & average
                var newQty = p.Qty + signed;
                if (newQty == 0)
                {
                    p.Qty = 0;
                    p.AvgPrice = 0m;
                }
                else if (Math.Sign(p.Qty) == Math.Sign(newQty))
                {
                    // same side add
                    p.AvgPrice = (p.AvgPrice * Math.Abs(p.Qty) + px * Math.Abs(signed)) / Math.Abs(newQty);
                    p.Qty = newQty;
                }
                else
                {
                    // reduced or flipped: set new avg at px for the flipped remainder
                    p.Qty = newQty;
                    if (Math.Sign(newQty) != 0) p.AvgPrice = px;
                    else p.AvgPrice = 0m;
                }
            }
        }

        public decimal RealizedTodayUSD()
        {
            lock (_lock)
            {
                return _pos.Values.Sum(p => p.Realized);
            }
        }

        public int TotalAbsQty()
        {
            lock (_lock)
            {
                return _pos.Values.Sum(p => Math.Abs(p.Qty));
            }
        }

        public int QtyForRoot(string root)
        {
            lock (_lock)
            {
                return _pos.TryGetValue(root, out var p) ? p.Qty : 0;
            }
        }

        private static string? TryStr(JsonElement e, params string[] names)
        {
            foreach (var n in names)
                if (e.TryGetProperty(n, out var p) && p.ValueKind == JsonValueKind.String)
                    return p.GetString();
            return null;
        }
        private static int? TryInt(JsonElement e, params string[] names)
        {
            foreach (var n in names)
                if (e.TryGetProperty(n, out var p))
                {
                    if (p.ValueKind == JsonValueKind.Number && p.TryGetInt32(out var i)) return i;
                    if (p.ValueKind == JsonValueKind.String && int.TryParse(p.GetString(), out var j)) return j;
                }
            return null;
        }
        private static decimal? TryDec(JsonElement e, params string[] names)
        {
            foreach (var n in names)
                if (e.TryGetProperty(n, out var p))
                {
                    if (p.ValueKind == JsonValueKind.Number && p.TryGetDecimal(out var d)) return d;
                    if (p.ValueKind == JsonValueKind.String && decimal.TryParse(p.GetString(), out var s)) return s;
                }
            return null;
        }
    }

    // PURPOSE: Evaluation-mode gating: session window, daily loss cap, per-symbol and total size caps.
    public sealed class EvalGuard
    {
        private readonly EvalPolicy _policy;
        private readonly PnLTracker _pnl;

        public EvalGuard(EvalPolicy policy, PnLTracker pnl)
        {
            _policy = policy;
            _pnl = pnl;
        }

        public bool CanOpen(string rootSymbol, int desiredQty, out string? reason)
        {
            reason = null;
            if (!_policy.Enabled) return true;

            _pnl.ResetIfNewDay(DateTimeOffset.UtcNow);

            if (!_policy.IsWithinSession(DateTimeOffset.UtcNow))
            {
                reason = "Outside session window.";
                return false;
            }

            var realized = _pnl.RealizedTodayUSD();
            if (realized <= -_policy.MaxDailyLoss)
            {
                reason = $"Daily loss reached ({realized:F2} USD).";
                return false;
            }

            var symAbs = Math.Abs(_pnl.QtyForRoot(rootSymbol));
            if (symAbs + desiredQty > _policy.MaxContractsPerSymbol)
            {
                reason = $"Per-symbol size cap ({_policy.MaxContractsPerSymbol}).";
                return false;
            }

            var total = _pnl.TotalAbsQty();
            if (total + desiredQty > _policy.MaxTotalContracts)
            {
                reason = $"Total contracts cap ({_policy.MaxTotalContracts}).";
                return false;
            }

            return true;
        }
    }

    // PURPOSE: Subscribe to user hub and stream trades into PnLTracker.
    public sealed class UserHubAgent : IAsyncDisposable
    {
        private readonly ILogger<UserHubAgent> _log;
        private HubConnection? _conn;
        private readonly PnLTracker _pnl;

        public UserHubAgent(ILogger<UserHubAgent> log, PnLTracker pnl)
        {
            _log = log;
            _pnl = pnl;
        }

        public async Task ConnectAsync(ITopstepAuth auth, int accountId, CancellationToken ct = default)
        {
            async Task<string?> TokenProvider()
            {
                var (jwt, expUtc) = await auth.GetFreshJwtAsync(ct);
                var ttl = expUtc - DateTimeOffset.UtcNow;
                if (ttl < TimeSpan.FromSeconds(30))
                    throw new InvalidOperationException($"JWT TTL too low ({ttl.TotalSeconds:F0}s). Refresh logic failed.");
                return jwt;
            }

            _conn = new HubConnectionBuilder()
                .WithUrl("https://rtc.topstepx.com/hubs/user", options =>
                {
                    options.AccessTokenProvider = TokenProvider;
                    options.SkipNegotiation = false;
                    options.Transports = HttpTransportType.WebSockets;
                })
                .WithAutomaticReconnect(new[]
                {
                    TimeSpan.Zero, TimeSpan.FromSeconds(2),
                    TimeSpan.FromSeconds(5), TimeSpan.FromSeconds(10)
                })
                .ConfigureLogging(logging =>
                {
                    logging.ClearProviders();
                    logging.AddConsole();
                    logging.SetMinimumLevel(LogLevel.Information);
                    var concise = (Environment.GetEnvironmentVariable("APP_CONCISE_CONSOLE") ?? "true").Trim().ToLowerInvariant() is "1" or "true" or "yes";
                    if (concise)
                    {
                        logging.AddFilter("Microsoft", LogLevel.Warning);
                        logging.AddFilter("System", LogLevel.Warning);
                        logging.AddFilter("Microsoft.AspNetCore.SignalR", LogLevel.Warning);
                        logging.AddFilter("Microsoft.AspNetCore.Http.Connections", LogLevel.Warning);
                    }
                    // Always suppress verbose client transport logs that can echo access_token in URLs
                    logging.AddFilter("Microsoft.AspNetCore.SignalR.Client", LogLevel.Error);
                    logging.AddFilter("Microsoft.AspNetCore.Http.Connections.Client", LogLevel.Error);
                    logging.AddFilter("Microsoft.AspNetCore.Http.Connections.Client.Internal.WebSocketsTransport", LogLevel.Error);
                })
                .Build();

            _conn.ServerTimeout = TimeSpan.FromSeconds(45);
            _conn.KeepAliveInterval = TimeSpan.FromSeconds(15);

            // Wire events ONCE BEFORE StartAsync
            bool handlersWired = false;
            void WireHandlers()
            {
                if (handlersWired) return;
                _conn.On<object>("GatewayUserOrder", data =>
                {
                    try
                    {
                        if (data is JsonElement je) _log.LogInformation($"Order event: {je}");
                    }
                    catch (Exception ex)
                    {
                        _log.LogWarning(ex, "Order parse failed.");
                    }
                });
                _conn.On<object>("GatewayUserTrade", data => Console.WriteLine($"TRADE => {data}"));
                handlersWired = true;
            }
            WireHandlers();

            _conn.Reconnecting += error =>
            {
                _log.LogWarning(error, "UserHub reconnecting: {Message}", error?.Message);
                return Task.CompletedTask;
            };
            _conn.Reconnected += connectionId =>
            {
                _log.LogInformation("UserHub reconnected: {ConnectionId}", connectionId);
                // Re-subscribe after reconnect
                _ = ReliableInvokeAsync(_conn, (c, ct2) => c.InvokeAsync("SubscribeAccounts", accountId, ct2), ct);
                _ = ReliableInvokeAsync(_conn, (c, ct2) => c.InvokeAsync("SubscribeOrders", accountId, ct2), ct);
                _ = ReliableInvokeAsync(_conn, (c, ct2) => c.InvokeAsync("SubscribePositions", accountId, ct2), ct);
                _ = ReliableInvokeAsync(_conn, (c, ct2) => c.InvokeAsync("SubscribeTrades", accountId, ct2), ct);
                return Task.CompletedTask;
            };
            _conn.Closed += error =>
            {
                _log.LogWarning(error, "UserHub closed: {Message}", error?.Message);
                return Task.CompletedTask;
            };

            using (var connectCts = CancellationTokenSource.CreateLinkedTokenSource(ct))
            {
                connectCts.CancelAfter(TimeSpan.FromSeconds(15));
                await _conn.StartAsync(connectCts.Token);
            }
            if (_conn.State != HubConnectionState.Connected)
                throw new InvalidOperationException("UserHub failed to connect.");

            // Log JWT TTL at connect time
            {
                var (_, expUtc) = await auth.GetFreshJwtAsync(ct);
                var ttl = expUtc - DateTimeOffset.UtcNow;
                _log.LogInformation($"UserHub connected. JWT TTL ≈ {(int)ttl.TotalSeconds}s");
            }

            // Only subscribe if connection is active
            if (_conn.State == HubConnectionState.Connected)
            {
                await ReliableInvokeAsync(_conn, (conn, token) => conn.InvokeAsync("SubscribeAccounts", accountId, token), ct);
                await ReliableInvokeAsync(_conn, (conn, token) => conn.InvokeAsync("SubscribeOrders", accountId, token), ct);
                await ReliableInvokeAsync(_conn, (conn, token) => conn.InvokeAsync("SubscribePositions", accountId, token), ct);
                await ReliableInvokeAsync(_conn, (conn, token) => conn.InvokeAsync("SubscribeTrades", accountId, token), ct);
                _log.LogInformation("[UserHub] subscribed for account {AccountId}", accountId);
            }
        }
#nullable enable
        private async Task ReliableInvokeAsync(HubConnection conn, Func<HubConnection, CancellationToken, Task> call, CancellationToken ct)
        {
            var delay = TimeSpan.FromMilliseconds(300);
            for (int attempt = 1; attempt <= 5; attempt++)
            {
                ct.ThrowIfCancellationRequested();

                if (conn.State == HubConnectionState.Connected)
                {
                    try { await call(conn, ct); return; }
                    catch (Exception ex) { _log.LogWarning(ex, $"Invoke attempt {attempt} failed: {ex.Message}"); }
                }
                else
                {
                    _log.LogWarning($"Connection state is {conn.State}, waiting before retry...");
                }

                await Task.Delay(delay, ct);
                delay = TimeSpan.FromMilliseconds(Math.Min(delay.TotalMilliseconds * 2, 5000));
            }
            throw new InvalidOperationException("UserHub invoke could not complete after multiple retries.");
        }

        public async ValueTask DisposeAsync()
        {
            if (_conn is not null)
            {
                try { await _conn.DisposeAsync(); } catch { }
            }
        }
    }


}
