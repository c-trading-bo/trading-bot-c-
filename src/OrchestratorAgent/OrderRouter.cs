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

        public OrderRouter(ILogger<OrderRouter> log, HttpClient http, string apiBase, string jwt, int accountId)
        {
            _log = log;
            _api = new ApiClient(http, log as ILogger<ApiClient> ?? LoggerFactory.Create(b=>{}).CreateLogger<ApiClient>(), apiBase);
            _api.SetJwt(jwt);
            _accountId = accountId;
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
            var basePx = sig.LimitPrice ?? 0m;
            var px = sig.Side == SignalSide.Long ? basePx + bufTicks * tick : basePx - bufTicks * tick;

            // Slippage guard
            var maxSlipTicks = 4;
            var msEnv = Environment.GetEnvironmentVariable("MAX_SLIPPAGE_TICKS");
            if (int.TryParse(msEnv, out var maxEnv) && maxEnv > 0) maxSlipTicks = maxEnv;
            var adverse = Math.Abs(px - basePx) / (tick == 0 ? 0.25m : tick);
            if (adverse > maxSlipTicks)
            {
                var cap = maxSlipTicks * tick;
                px = sig.Side == SignalSide.Long ? basePx + cap : basePx - cap;
                _log.LogWarning("[ROUTER] Slippage capped to {Ticks} ticks for {Cid}", maxSlipTicks, cid);
            }

            var live = (Environment.GetEnvironmentVariable("LIVE_TRADING") ??
                        Environment.GetEnvironmentVariable("TOPSTEPX_LIVE") ?? "false")
                        .Equals("true", StringComparison.OrdinalIgnoreCase);

            if (!live)
            {
                _log.LogInformation("[DRY-RUN] {Strat} {Sym} {Side} x{Size}@{Px} (+{Buf}t) CID={Cid} -> {Contract}",
                    sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, bufTicks, cid, contractId);
                return true;
            }

            _log.LogInformation("[LIVE] placing {Strat} {Sym} {Side} x{Size}@{Px} (+{Buf}t) CID={Cid} -> {Contract}",
                sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, bufTicks, cid, contractId);

            var orderReq = new {
                accountId = _accountId,
                contractId = contractId,
                side = sideBuy0Sell1,
                qty = sig.Size,
                price = px,
                customTag = cid
            };

            var sw = Stopwatch.StartNew();
            try
            {
                await _api.PlaceOrderAsync(orderReq, ct);
                return true;
            }
            catch
            {
                // remove CID on failure to allow retry
                _sent.TryRemove(cid!, out _);
                throw;
            }
            finally
            {
                sw.Stop();
                _log.LogInformation("[ROUTER] Order latency: {Ms} ms (CID {Cid})", sw.ElapsedMilliseconds, cid);
            }
        }

        public async Task EnsureBracketsAsync(long accountId, CancellationToken ct = default)
        {
            // Placeholder for OCO reconstruction: query open orders and ensure TP/SL exist
            // Minimal safe log until server-side endpoints wired
            _log.LogInformation("[ROUTER] EnsureBrackets requested for account {Acc}. (Implement API sync here)", accountId);
            await Task.CompletedTask;
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
    }
}
