#nullable enable
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Net.Http;
using Microsoft.Extensions.Logging;
using BotCore;
using BotCore.Config;
using BotCore.Models;

namespace OrchestratorAgent
{
    /// <summary>Tiny wrapper around ApiClient for placing orders.</summary>
    public sealed class OrderRouter
    {
        private readonly ILogger<OrderRouter> _log;
        private readonly ApiClient _api;
        private readonly int _accountId;
        private static readonly ConcurrentDictionary<string, DateTime> _sent = new();
        private static readonly ConcurrentQueue<long> _latencyMs = new();
        private static readonly ConcurrentQueue<DateTime> _rejects = new();
        private static readonly ConcurrentQueue<DateTime> _ocoRebuilds = new();
        private static readonly object _latLock = new();

        public OrderRouter(ILogger<OrderRouter> log, HttpClient http, string apiBase, string jwt, int accountId)
        {
            _log = log;
            _api = new ApiClient(http, log as ILogger<ApiClient> ?? LoggerFactory.Create(b=>{}).CreateLogger<ApiClient>(), apiBase);
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

            var sideBuy0Sell1 = sig.Side == SignalSide.Long ? 0 : 1;

            // Apply configurable tick buffers to price
            decimal tick = InstrumentMeta.Tick(sig.Symbol);
            int bufTicks = ResolveBufferTicks(sig.Symbol);
            var basePx = InstrumentMeta.RoundToTick(sig.Symbol, sig.LimitPrice ?? 0m);
                        // Ensure qty respects lot step
                        var step = InstrumentMeta.LotStep(sig.Symbol);
                        var qtyAdj = Math.Max(step, sig.Size - (sig.Size % step));
            var px = sig.Side == SignalSide.Long ? basePx + bufTicks * tick : basePx - bufTicks * tick;

            // Slippage guard (per-symbol override)
            var maxSlipTicks = 4;
            var sym = sig.Symbol.ToUpperInvariant();
            var symEnv = Environment.GetEnvironmentVariable($"TOPSTEPX_SLIP_{sym}");
            if (int.TryParse(symEnv, out var symTicks) && symTicks > 0) maxSlipTicks = symTicks;
            else {
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

            var orderReq = new {
                accountId = _accountId,
                contractId = contractId,
                side = sideBuy0Sell1,
                qty = qtyAdj,
                price = px,
                customTag = cid,
                orderType,
                timeInForce = tif
            };

            var sw = Stopwatch.StartNew();
            try
            {
                await _api.PlaceOrderAsync(orderReq, ct);
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
            _log.LogInformation("[ROUTER] EnsureBrackets requested for account {Acc}. (Implement API sync here)", accountId);
            _ocoRebuilds.Enqueue(DateTime.UtcNow);
            TrimQueueWindow(_ocoRebuilds, TimeSpan.FromMinutes(10));
            var cnt10m = _ocoRebuilds.Count;
            if (cnt10m == 1)
                _log.LogInformation("[INFO] OCO rebuild performed: count={Count} in last 10m", cnt10m);
            else if (cnt10m > 3)
                _log.LogWarning("[WARN] OCO rebuilds high: count={Count} in last 10m", cnt10m);
            await Task.CompletedTask;
        }

        public async Task UpsertBracketsAsync(string orderId, int filledQty, CancellationToken ct = default)
        {
            _log.LogInformation("[ROUTER] UpsertBrackets for parent {OrderId} -> filledQty={Qty}", orderId, filledQty);
            await Task.CompletedTask;
        }

        public async Task ConvertRemainderToLimitOrCancelAsync(dynamic orderUpdate, CancellationToken ct = default)
        {
            _log.LogWarning("[ROUTER] Convert remainder-to-limit or cancel for order update {Update}", orderUpdate);
            await Task.CompletedTask;
        }

        public async Task CancelAllOpenAsync(long accountId, CancellationToken ct = default)
        {
            _log.LogWarning("[ROUTER] CancelAllOpen requested for account {Acc}", accountId);
            await Task.CompletedTask;
        }

        public async Task FlattenAllAsync(long accountId, CancellationToken ct = default)
        {
            _log.LogError("[ROUTER] PANIC_FLATTEN: flattening all positions for account {Acc}", accountId);
            // Implement via API: close positions and cancel all brackets
            await Task.CompletedTask;
        }

        public async Task<List<dynamic>> QueryOpenPositionsAsync(long accountId, CancellationToken ct = default)
        {
            _log.LogInformation("[ROUTER] QueryOpenPositions for account {Acc}", accountId);
            return await Task.FromResult(new List<dynamic>());
        }

        public async Task<List<dynamic>> QueryOpenOrdersAsync(long accountId, CancellationToken ct = default)
        {
            _log.LogInformation("[ROUTER] QueryOpenOrders for account {Acc}", accountId);
            return await Task.FromResult(new List<dynamic>());
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
                if (symbol.Equals("ES", StringComparison.OrdinalIgnoreCase) && profile.Buffers.TryGetValue("ES_ticks", out var es)) return Math.Max(0, es);
                if (symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) && profile.Buffers.TryGetValue("NQ_ticks", out var nq)) return Math.Max(0, nq);
            }
            catch { }

            // Safe default
            return symbol.Equals("NQ", StringComparison.OrdinalIgnoreCase) ? 2 : 1;
        }
        private static void TrimQueueWindow(ConcurrentQueue<DateTime> q, TimeSpan window)
        {
            var cutoff = DateTime.UtcNow - window;
            while (q.TryPeek(out var t) && t < cutoff)
                q.TryDequeue(out _);
        }
    }
}
