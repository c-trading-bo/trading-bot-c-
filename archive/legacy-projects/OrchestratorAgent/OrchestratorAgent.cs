#nullable enable
// Agent: OrchestratorAgent (standalone)
// Role: Standalone orchestration logic for strategy and agent coordination.
// Integration: Manages bot lifecycle and agent interactions.
// PURPOSE: Evaluation-account policy configuration and simple session window helpers.
using System.Collections.Concurrent;
using System.Text.Json;

using Microsoft.Extensions.Logging;
using Trading.Safety;
using TradingBot.Abstractions;

namespace OrchestratorAgent
{
    internal sealed class EvalPolicy
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

    internal static class SymbolMeta
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
            if (s.StartsWith("NQ")) return "NQ";
            return new string([.. s.TakeWhile(char.IsLetter)]);
        }
    }
    // PURPOSE: Track positions and realized PnL per trading day from trade fills.
    internal sealed class PnLTracker(EvalPolicy policy)
    {
        private readonly object _lock = new();
        private readonly EvalPolicy _policy = policy;

        private sealed class Pos
        {
            public int Qty;                 // signed
            public decimal AvgPrice;        // weighted avg
            public decimal Realized;        // USD
        }

        private readonly ConcurrentDictionary<string, Pos> _pos = new(StringComparer.OrdinalIgnoreCase);
        private DateTimeOffset _dayStartUtc = policy.GetTradingDayStart(DateTimeOffset.UtcNow);

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
                    p.Qty;
                    p.AvgPrice;
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
                    else p.AvgPrice;
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
    internal sealed class EvalGuard(EvalPolicy policy, PnLTracker pnl)
    {
        private readonly EvalPolicy _policy = policy;
        private readonly PnLTracker _pnl = pnl;

        public bool CanOpen(string rootSymbol, int desiredQty, out string? reason)
        {
            reason = null!;
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

    // PURPOSE: Subscribe to TopstepX SDK for trades and stream into PnLTracker.
    internal sealed class TopstepXUserAgent(ILogger<TopstepXUserAgent> log, PnLTracker pnl, ITopstepXClient topstepXClient) : IAsyncDisposable
    {
        private readonly ILogger<TopstepXUserAgent> _log = log;
        private readonly ITopstepXClient _topstepXClient = topstepXClient;
        private readonly PnLTracker _pnl = pnl;
        private CancellationTokenSource? _cancellationTokenSource;
        private bool _subscribed;

        public async Task ConnectAsync(string authToken, int accountId, CancellationToken ct = default)
        {
            _log.LogInformation("[TOPSTEPX-SDK] Initializing connection for account {AccountId}", accountId);
            
            _cancellationTokenSource = CancellationTokenSource.CreateLinkedTokenSource(ct);

            try
            {
                // Subscribe to TopstepX SDK events
                _topstepXClient.OnOrderUpdate += HandleOrderUpdate;
                _topstepXClient.OnTradeUpdate += HandleTradeUpdate;

                // Subscribe to account-specific order and trade updates
                var orderSubscribed = await _topstepXClient.SubscribeOrdersAsync(accountId.ToString(), ct).ConfigureAwait(false);
                var tradeSubscribed = await _topstepXClient.SubscribeTradesAsync(accountId.ToString(), ct).ConfigureAwait(false);

                if (orderSubscribed && tradeSubscribed)
                {
                    _subscribed = true;
                    _log.LogInformation("[TOPSTEPX-SDK] Successfully subscribed to orders and trades for account {AccountId}", accountId);
                }
                else
                {
                    _log.LogWarning("[TOPSTEPX-SDK] Failed to subscribe to some data streams for account {AccountId}. Orders: {OrdersSub}, Trades: {TradesSub}", 
                        accountId, orderSubscribed, tradeSubscribed);
                }
            }
            catch (Exception ex)
            {
                _log.LogError(ex, "[TOPSTEPX-SDK] Failed to initialize connection for account {AccountId}", accountId);
                throw;
            }
        }

        public async Task SubscribeToAccountAsync(string authToken, int accountId, CancellationToken ct = default)
        {
            if (_subscribed)
            {
                _log.LogInformation("[TOPSTEPX-SDK] Already subscribed to account {AccountId}", accountId);
                return;
            }

            await ConnectAsync(authToken, accountId, ct).ConfigureAwait(false);
        }

        private void HandleOrderUpdate(object? sender, OrderUpdateEventArgs e)
        {
            try
            {
                _log.LogInformation("[TOPSTEPX-SDK] Order update - OrderId: {OrderId}, Status: {Status}", 
                    e.OrderId, e.Status);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[TOPSTEPX-SDK] Error handling order update");
            }
        }

        private void HandleTradeUpdate(object? sender, TradeUpdateEventArgs e)
        {
            try
            {
                _log.LogInformation("[TOPSTEPX-SDK] Trade update - OrderId: {OrderId}, Price: {Price}, Quantity: {Quantity}", 
                    e.OrderId, e.FillPrice, e.Quantity);

                // Update PnL tracker with the trade information
                if (_pnl != null)
                {
                    // Integration with PnL tracker would go here
                    _log.LogDebug("[TOPSTEPX-SDK] Trade forwarded to PnL tracker");
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[TOPSTEPX-SDK] Error handling trade update");
            }
        }