// PURPOSE: Evaluation-account policy configuration and simple session window helpers.
#nullable enable
using System.Globalization;
using System.Collections.Concurrent;
using System.Text.Json;
using Microsoft.AspNetCore.Http.Connections;
using Microsoft.AspNetCore.SignalR.Client;
using Microsoft.Extensions.Logging;

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
            var qty  = TryInt(tradePayload, "qty", "quantity", "filledQty", "fillQty") ?? 0;
            var px   = TryDec(tradePayload, "price", "fillPrice", "avgPrice") ?? 0m;

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

        public async Task ConnectAsync(string jwt, int accountId, CancellationToken ct = default)
        {
            var url = $"https://rtc.topstepx.com/hubs/user?access_token={Uri.EscapeDataString(jwt)}";
            _conn = new HubConnectionBuilder()
                .WithUrl(url, o =>
                {
                    o.AccessTokenProvider = () => Task.FromResult<string?>(jwt);
                    o.Transports = HttpTransportType.WebSockets;
                })
                .WithAutomaticReconnect()
                .Build();

            _conn.On<object>("GatewayUserTrade", data =>
            {
                try
                {
                    if (data is JsonElement je) _pnl.OnFill(je);
                }
                catch (Exception ex)
                {
                    _log.LogWarning(ex, "Trade parse failed.");
                }
            });

            await _conn.StartAsync(ct);
            await _conn.InvokeAsync("SubscribeAccounts");
            await _conn.InvokeAsync("SubscribeOrders", accountId);
            await _conn.InvokeAsync("SubscribePositions", accountId);
            await _conn.InvokeAsync("SubscribeTrades", accountId);
            _log.LogInformation("[UserHub] subscribed for account {AccountId}", accountId);
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
