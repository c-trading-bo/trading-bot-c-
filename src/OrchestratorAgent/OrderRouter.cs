#nullable enable
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Http;
using Microsoft.Extensions.Logging;
using BotCore;
using BotCore.Config;
using BotCore.Models;
using OrchestratorAgent.Infra;
using Trading.Safety;

// NOTE: Not referenced by OrchestratorAgent.Program. The runtime uses SimpleOrderRouter.
// This richer router remains for future expansion and tooling, but is not on the hot path.
namespace OrchestratorAgent
{
    /// <summary>Tiny wrapper around ApiClient for placing orders.</summary>
    public sealed class OrderRouter
    {
        // Helpers (class-scope) for env and parsing
        private static int ResolveIntEnv(string key, int defVal)
        {
            var raw = Environment.GetEnvironmentVariable(key);
            return int.TryParse(raw, out var v) ? v : defVal;
        }
        private static int SafeToInt(object? o)
        {
            try { return Convert.ToInt32(o ?? 0); } catch { return 0; }
        }
        private static decimal PickPrice(Type t, object obj, string[] names, decimal current)
        {
            if (current > 0m) return current;
            foreach (var n in names)
            {
                var p = t.GetProperty(n);
                if (p == null) continue;
                try
                {
                    var v = p.GetValue(obj);
                    if (v == null) continue;
                    var d = Convert.ToDecimal(v, System.Globalization.CultureInfo.InvariantCulture);
                    if (d > 0m) return d;
                }
                catch { }
            }
            return current;
        }
        private readonly ILogger<OrderRouter> _log;
        private readonly ApiClient _api;
        private readonly int _accountId;
        private static readonly ConcurrentDictionary<string, DateTime> _sent = new();
        private static readonly ConcurrentQueue<long> _latencyMs = new();
        private static readonly ConcurrentQueue<DateTime> _rejects = new();
        private static readonly ConcurrentQueue<DateTime> _ocoRebuilds = new();
        private static readonly object _latLock = new();
        private readonly OrderLimiter _limiter = new(20, TimeSpan.FromSeconds(10));
        private readonly Notifier _notifier = new();

        private sealed class OrderLimiter(int max, TimeSpan window)
        {
            private readonly Queue<DateTime> _ts = new();
            private readonly int _max = max; private readonly TimeSpan _window = window;

            public bool Allow()
            {
                var now = DateTime.UtcNow;
                while (_ts.Count > 0 && (now - _ts.Peek()) > _window) _ts.Dequeue();
                if (_ts.Count >= _max) return false;
                _ts.Enqueue(now); return true;
            }
        }

        public OrderRouter(ILogger<OrderRouter> log, HttpClient http, string apiBase, string jwt, int accountId)
        {
            _log = log;
            _api = new ApiClient(http, log as ILogger<ApiClient> ?? LoggerFactory.Create(b => { }).CreateLogger<ApiClient>(), apiBase);
            _api.SetJwt(jwt);
            _accountId = accountId;
        }

        public void UpdateJwt(string jwt)
        {
            try { _api.SetJwt(jwt); } catch { }
        }

        public async Task<bool> RouteAsync(StrategySignal sig, string contractId, CancellationToken ct = default)
        {
            // Client Order ID (deterministic + unique)
            var cid = sig.ClientOrderId;
            if (string.IsNullOrWhiteSpace(cid))
            {
                cid = $"{sig.Strategy}|{sig.Symbol}|{DateTime.UtcNow:yyyyMMddTHHmmssfff}|{Guid.NewGuid():N}".ToUpperInvariant();
                sig = new StrategySignal
                {
                    Strategy = sig.Strategy,
                    Symbol = sig.Symbol,
                    Side = sig.Side,
                    Size = sig.Size,
                    LimitPrice = sig.LimitPrice,
                    Note = sig.Note,
                    ClientOrderId = cid
                };
            }

            // Idempotent guard: don't send same CID within 10 minutes
            var now = DateTime.UtcNow;
            if (_sent.TryGetValue(cid!, out var t) && (now - t).TotalMinutes < 10)
            {
                _log.LogWarning("[ROUTER] Duplicate CID suppressed: {Cid}", cid);
                return false;
            }
            _sent[cid!] = now;

            // Order burst guard (rate limiter)
            if (!_limiter.Allow())
            {
                _log.LogWarning("[ROUTER] Rate limit exceeded; blocking route CID={Cid}", cid);
                return false;
            }

            var sideBuy0Sell1 = sig.Side == SignalSide.Long ? 0 : 1;

            // Apply configurable tick buffers to price
            decimal tick = InstrumentMeta.Tick(sig.Symbol);
            int bufTicks = ResolveBufferTicks(sig.Symbol);
            var basePx = InstrumentMeta.RoundToTick(sig.Symbol, sig.LimitPrice ?? 0m);
            // Ensure qty respects lot step (and global max=2)
            var step = InstrumentMeta.LotStep(sig.Symbol);
            var sizeClamped = Math.Clamp(sig.Size, 1, 2);
            var qtyAdj = Math.Max(step, sizeClamped - (sizeClamped % step));
            var px = sig.Side == SignalSide.Long ? basePx + bufTicks * tick : basePx - bufTicks * tick;

            // Slippage guard (per-symbol override)
            var maxSlipTicks = 4;
            var sym = sig.Symbol.ToUpperInvariant();
            var symEnv = Environment.GetEnvironmentVariable($"TOPSTEPX_SLIP_{sym}");
            if (int.TryParse(symEnv, out var symTicks) && symTicks > 0) maxSlipTicks = symTicks;
            else
            {
                var msEnv = Environment.GetEnvironmentVariable("MAX_SLIPPAGE_TICKS");
                if (int.TryParse(msEnv, out var maxEnv) && maxEnv > 0) maxSlipTicks = maxEnv;
            }
            var adverse = Math.Abs(px - basePx) / (tick == 0 ? 0.25m : tick);
            if (adverse > maxSlipTicks)
            {
                var cap = maxSlipTicks * tick;
                px = sig.Side == SignalSide.Long ? basePx + cap : basePx - cap;
                _log.LogWarning("[ROUTER] Slippage capped to {Ticks} ticks for {Cid}", maxSlipTicks, cid);
            }

            var live = (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase)
                        || (Environment.GetEnvironmentVariable("LIVE_ORDERS") ?? string.Empty).Equals("true", StringComparison.OrdinalIgnoreCase);

            if (!live)
            {
                _log.LogInformation("[DRY-RUN] {Strat} {Sym} {Side} x{Size}@{Px} (+{Buf}t) CID={Cid} -> {Contract}",
                    sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, bufTicks, cid, contractId);
                return true;
            }

            _log.LogInformation("[LIVE] placing {Strat} {Sym} {Side} x{Size}@{Px} (+{Buf}t) CID={Cid} -> {Contract}",
                sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, bufTicks, cid, contractId);

            // News-burst routing preferences (IOC + stop-through) via env switches
            var explosive = (Environment.GetEnvironmentVariable("IS_MAJOR_NEWS_NOW") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase)
                         || (Environment.GetEnvironmentVariable("TOPSTEPX_NEWS_EXPLOSIVE") ?? "0").Equals("1", StringComparison.OrdinalIgnoreCase);
            string orderType = "LIMIT";
            string tif = "DAY";
            if (explosive)
            {
                orderType = "STOP"; // flip to stop-through
                tif = "IOC";        // aggressive during news
                // add one more tick for momentum through price if needed
                var extra = tick;
                px = sig.Side == SignalSide.Long ? px + extra : px - extra;
            }

            var orderReq = new
            {
                accountId = _accountId,
                contractId,
                type = GetOrderTypeValue(orderType),    // ProjectX: 1=Limit, 2=Market, 4=Stop
                side = sideBuy0Sell1,                   // Already correct: 0=Buy, 1=Sell
                size = qtyAdj,                          // ProjectX expects integer size
                limitPrice = orderType == "LIMIT" ? px : (decimal?)null,
                stopPrice = orderType == "STOP" ? px : (decimal?)null,
                customTag = cid,
                timeInForce = tif
            };

            var sw = Stopwatch.StartNew();
            try
            {
                await _api.PlaceOrderAsync(orderReq, ct).ConfigureAwait(false);
                return true;
            }
            catch (Exception ex)
            {
                // remove CID on failure to allow retry
                _sent.TryRemove(cid!, out _);
                // track reject for alerting
                _rejects.Enqueue(DateTime.UtcNow);
                TrimQueueWindow(_rejects, TimeSpan.FromSeconds(60));
                if (_rejects.Count >= 3)
                {
                    _log.LogError(ex, "[ALERT] 3+ order rejects in 60s. Auto-pausing routing.");
                    _ = _notifier.Error("3+ order rejects in 60s — routing paused");
                    Environment.SetEnvironmentVariable("ROUTE_PAUSE", "1");
                }
                throw;
            }
            finally
            {
                sw.Stop();
                var ms = sw.ElapsedMilliseconds;
                _log.LogInformation("[ROUTER] Order latency: {Ms} ms (CID {Cid})", ms, cid);
                // track latency, keep last 20
                _latencyMs.Enqueue(ms);
                while (_latencyMs.Count > 20 && _latencyMs.TryDequeue(out _)) { }
                // compute median
                try
                {
                    var arr = _latencyMs.ToArray();
                    if (arr.Length >= 10)
                    {
                        Array.Sort(arr);
                        var median = arr[arr.Length / 2];
                        if (median > 800)
                        {
                            _log.LogWarning("[WARN] Median order latency over last {N} orders is {Med} ms (>800)", arr.Length, median);
                        }
                    }
                }
                catch { }
            }
        }

        public async Task EnsureBracketsAsync(long accountId, CancellationToken ct = default)
        {
            try
            {
                var parents = await _api.GetAsync<List<dynamic>>($"/orders?accountId={accountId}&status=OPEN&parent=true", ct)
                               ?? [].ConfigureAwait(false);
                foreach (var p in parents)
                {
                    bool hasBr = false;
                    try { hasBr = (bool)(p.hasBrackets ?? false); } catch { }
                    if (!hasBr)
                    {
                        var body = new { parentId = (object?)p.id, takeProfitTicks = 20, stopLossTicks = 12 };
                        await _api.PostAsync("/orders/brackets", body, ct).ConfigureAwait(false);
                        _ocoRebuilds.Enqueue(DateTime.UtcNow);
                    }
                }
                TrimQueueWindow(_ocoRebuilds, TimeSpan.FromMinutes(10));
                var cnt10m = _ocoRebuilds.Count;
                if (cnt10m == 1)
                {
                    _log.LogInformation("[INFO] OCO rebuild performed: count={Count} in last 10m", cnt10m);
                    _ = _notifier.Info($"OCO rebuilt x{cnt10m} after reconnect");
                }
                else if (cnt10m > 3)
                {
                    _log.LogWarning("[WARN] OCO rebuilds high: count={Count} in last 10m", cnt10m);
                    _ = _notifier.Warn($"OCO rebuilds high: {cnt10m} in last 10m");
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[ROUTER] EnsureBrackets failed for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
            }
        }

        public async Task UpsertBracketsAsync(string orderId, int filledQty, CancellationToken ct = default)
        {
            // Upsert TP/SL brackets sized to the filled quantity for a partially-filled parent order.
            // Uses best-effort endpoints. Replace with your exact API routes if different.
            try
            {
                // Defaults with env overrides
                int tpTicks = ResolveIntEnv("BRACKET_TP_TICKS", 20);
                int slTicks = ResolveIntEnv("BRACKET_SL_TICKS", 12);
                int beTicks = ResolveIntEnv("BRACKET_BE_TICKS", 8);
                int trailTicks = ResolveIntEnv("BRACKET_TRAIL_TICKS", 6);

                // Try to fetch symbol to allow per-symbol overrides like BRACKET_TP_ES_TICKS
                string? symbol = null!;
                try
                {
                    var parent = await _api.GetAsync<System.Text.Json.JsonElement>($"/orders/{orderId}", ct).ConfigureAwait(false);
                    if (parent.ValueKind == System.Text.Json.JsonValueKind.Object)
                    {
                        if (parent.TryGetProperty("symbol", out var s) && s.ValueKind == System.Text.Json.JsonValueKind.String) symbol = s.GetString();
                        else if (parent.TryGetProperty("Symbol", out var S) && S.ValueKind == System.Text.Json.JsonValueKind.String) symbol = S.GetString();
                    }
                }
                catch { /* optional */ }
                if (!string.IsNullOrWhiteSpace(symbol))
                {
                    var sym = symbol!.ToUpperInvariant();
                    tpTicks = ResolveIntEnv($"BRACKET_TP_{sym}_TICKS", tpTicks);
                    slTicks = ResolveIntEnv($"BRACKET_SL_{sym}_TICKS", slTicks);
                }

                var body = new { parentId = (object?)orderId, qty = filledQty, takeProfitTicks = tpTicks, stopLossTicks = slTicks, breakevenAfterTicks = beTicks, trailTicks };

                // Try primary route
                await _api.PostAsync("/orders/brackets/upsert", body, ct).ConfigureAwait(false);
                _log.LogInformation("[ROUTER] Upserted brackets for parent {OrderId} (qty={Qty}, tp={Tp}t, sl={Sl}t)", SecurityHelpers.MaskOrderId(orderId), filledQty, tpTicks, slTicks);
            }
            catch (Exception ex1)
            {
                try
                {
                    // Fallback to generic brackets endpoint
                    var body2 = new { parentId = (object?)orderId, takeProfitTicks = ResolveIntEnv("BRACKET_TP_TICKS", 20), stopLossTicks = ResolveIntEnv("BRACKET_SL_TICKS", 12) };
                    await _api.PostAsync("/orders/brackets", body2, ct).ConfigureAwait(false);
                    _log.LogInformation("[ROUTER] Upsert fallback: posted brackets for parent {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                }
                catch (Exception ex2)
                {
                    try
                    {
                        // Legacy API route fallback
                        var body3 = new { orderId, takeProfitTicks = ResolveIntEnv("BRACKET_TP_TICKS", 20), stopLossTicks = ResolveIntEnv("BRACKET_SL_TICKS", 12) };
                        await _api.PostAsync("/api/Order/brackets/upsert", body3, ct).ConfigureAwait(false);
                        _log.LogInformation("[ROUTER] Upsert legacy fallback OK for parent {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                    }
                    catch (Exception ex3)
                    {
                        _log.LogWarning(ex1, "[ROUTER] UpsertBrackets primary failed");
                        _log.LogWarning(ex2, "[ROUTER] UpsertBrackets fallback failed");
                        _log.LogWarning(ex3, "[ROUTER] UpsertBrackets legacy failed for parent {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                    }
                }
            }
        }

        public async Task ConvertRemainderToLimitOrCancelAsync(dynamic orderUpdate, CancellationToken ct = default)
        {
            // Convert stale partial parent remainder to a LIMIT (IOC) at an offset, or cancel after timeout.
            try
            {
                var t = orderUpdate.GetType();
                string orderId = t.GetProperty("OrderId")?.GetValue(orderUpdate)?.ToString() ?? string.Empty;
                if (string.IsNullOrWhiteSpace(orderId)) { _log.LogWarning("[ROUTER] Convert skipped: missing OrderId"); return; }

                int remaining = SafeToInt(t.GetProperty("RemainingQty")?.GetValue(orderUpdate));
                if (remaining <= 0) return;

                // Age thresholds
                var age = (TimeSpan?)(t.GetProperty("Age")?.GetValue(orderUpdate)) ?? TimeSpan.Zero;
                int convertAtSec = ResolveIntEnv("PARTIAL_CONVERT_AT_SEC", 8);
                int cancelAtSec = ResolveIntEnv("PARTIAL_CANCEL_AT_SEC", 20);
                if (age.TotalSeconds < convertAtSec) return;

                // Basics for price calc
                string sideStr = t.GetProperty("Side")?.GetValue(orderUpdate)?.ToString() ?? string.Empty;
                bool isSell = sideStr.Equals("SELL", StringComparison.OrdinalIgnoreCase) || sideStr == "1";
                string symbol = t.GetProperty("Symbol")?.GetValue(orderUpdate)?.ToString() ?? string.Empty;
                decimal tick = !string.IsNullOrWhiteSpace(symbol) ? InstrumentMeta.Tick(symbol) : 0.25m;
                int offsetTicks = ResolveIntEnv("PARTIAL_CONVERT_OFFSET_TICKS", 1);

                // Derive a reference price from update, prefer limit then avg fill then last
                decimal refPx = 0m;
                refPx = PickPrice(t, orderUpdate, new string[] { "LimitPrice", "limitPrice", "Price", "price" }, refPx);
                refPx = PickPrice(t, orderUpdate, new string[] { "AvgFillPrice", "avgFillPrice", "AverageFillPrice" }, refPx);
                refPx = PickPrice(t, orderUpdate, new string[] { "LastPrice", "lastPrice" }, refPx);

                bool converted = false;
                if (refPx > 0m)
                {
                    var px = isSell ? refPx - offsetTicks * tick : refPx + offsetTicks * tick;
                    var body = new { orderId, toType = "LIMIT", price = px, timeInForce = "IOC" };
                    try
                    {
                        await _api.PostAsync("/orders/convert", body, ct).ConfigureAwait(false);
                        _log.LogInformation("[ROUTER] Converted partial parent to LIMIT IOC at {Px} (order {OrderId})", px, SecurityHelpers.MaskOrderId(orderId));
                        converted = true;
                    }
                    catch (Exception ex1)
                    {
                        try
                        {
                            await _api.PostAsync("/api/Order/convert", body, ct).ConfigureAwait(false);
                            _log.LogInformation("[ROUTER] Converted (legacy) partial parent to LIMIT IOC at {Px} (order {OrderId})", px, SecurityHelpers.MaskOrderId(orderId));
                            converted = true;
                        }
                        catch (Exception ex2)
                        {
                            _log.LogWarning(ex1, "[ROUTER] Convert remainder primary failed for {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                            _log.LogWarning(ex2, "[ROUTER] Convert remainder legacy failed for {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                        }
                    }
                }

                // If still stale past cancel threshold, cancel parent to avoid dangling partials
                if (!converted && age.TotalSeconds >= cancelAtSec)
                {
                    var body = new { orderId };
                    try { await _api.PostAsync("/orders/cancel", body, ct).ConfigureAwait(false); _log.LogWarning("[ROUTER] Canceled stale partial parent {OrderId}", SecurityHelpers.MaskOrderId(orderId)); }
                    catch (Exception ex1)
                    {
                        try { await _api.PostAsync("/api/Order/cancel", body, ct).ConfigureAwait(false); _log.LogWarning("[ROUTER] Canceled (legacy) stale partial parent {OrderId}", SecurityHelpers.MaskOrderId(orderId)); }
                        catch (Exception ex2)
                        {
                            _log.LogWarning(ex1, "[ROUTER] Cancel remainder primary failed for {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                            _log.LogWarning(ex2, "[ROUTER] Cancel remainder legacy failed for {OrderId}", SecurityHelpers.MaskOrderId(orderId));
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[ROUTER] ConvertRemainderToLimitOrCancelAsync unexpected error");
            }
        }

        public async Task CancelAllOpenAsync(long accountId, CancellationToken ct = default)
        {
            try
            {
                await _api.PostAsync($"/orders/cancel_all", new { accountId }, ct).ConfigureAwait(false);
                _log.LogWarning("[ROUTER] CancelAllOpen posted for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[ROUTER] CancelAllOpen failed for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
            }
        }

        public async Task FlattenAllAsync(long accountId, CancellationToken ct = default)
        {
            _log.LogError("[ROUTER] PANIC_FLATTEN: flattening all positions for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
            try
            {
                await _api.PostAsync($"/positions/flatten_all", new { accountId }, ct).ConfigureAwait(false);
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[ROUTER] FlattenAll failed for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
            }
        }

        public async Task<List<dynamic>> QueryOpenPositionsAsync(long accountId, CancellationToken ct = default)
        {
            try
            {
                var list = await _api.PostAsync<List<dynamic>>("/api/Position/searchOpen", new { accountId }, ct).ConfigureAwait(false);
                return list ?? [];
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[ROUTER] QueryOpenPositions failed for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
                return [];
            }
        }

        public async Task<List<dynamic>> QueryOpenOrdersAsync(long accountId, CancellationToken ct = default)
        {
            try
            {
                var list = await _api.GetAsync<List<dynamic>>($"/orders?accountId={accountId}&status=OPEN", ct).ConfigureAwait(false);
                return list ?? [];
            }
            catch (Exception ex)
            {
                _log.LogWarning(ex, "[ROUTER] QueryOpenOrders failed for account {Acc}", SecurityHelpers.MaskAccountId(accountId));
                return [];
            }
        }

        private static int ResolveBufferTicks(string symbol)
        {
            // Environment override first
            if (symbol.Equals("ES", StringComparison.OrdinalIgnoreCase))
            {
                var esEnv = Environment.GetEnvironmentVariable("TOPSTEPX_BUFFER_ES_TICKS");
                if (int.TryParse(esEnv, out var esTicks)) return Math.Max(0, esTicks);
            }
            else if (symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase))
            {
                var nqEnv = Environment.GetEnvironmentVariable("TOPSTEPX_BUFFER_NQ_TICKS");
                if (int.TryParse(nqEnv, out var nqTicks)) return Math.Max(0, nqTicks);
            }

            // Fallback to profile defaults
            try
            {
                var profile = new HighWinRateProfile();
                if (symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) && HighWinRateProfile.Buffers.TryGetValue("ES_ticks", out var es)) return Math.Max(0, es);
                if (symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) && HighWinRateProfile.Buffers.TryGetValue("NQ_ticks", out var nq)) return Math.Max(0, nq);
            }
            catch { }

            // Safe default
            return symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ? 2 : 1;
        }
        
        /// <summary>
        /// Convert order type string to ProjectX API numeric value
        /// </summary>
        private static int GetOrderTypeValue(string orderType)
        {
            return orderType.ToUpper() switch
            {
                "LIMIT" => 1,
                "MARKET" => 2,
                "STOP" => 4,
                "TRAILING_STOP" => 5,
                _ => 1 // Default to LIMIT
            };
        }
        
        private static void TrimQueueWindow(ConcurrentQueue<DateTime> q, TimeSpan window)
        {
            var cutoff = DateTime.UtcNow - window;
            while (q.TryPeek(out var t) && t < cutoff)
                q.TryDequeue(out _);
        }
    }
}
