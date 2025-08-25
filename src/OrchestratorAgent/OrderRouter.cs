#nullable enable
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

        public OrderRouter(ILogger<OrderRouter> log, HttpClient http, string apiBase, string jwt, int accountId)
        {
            _log = log;
            _api = new ApiClient(http, log as ILogger<ApiClient> ?? LoggerFactory.Create(b=>{}).CreateLogger<ApiClient>(), apiBase);
            _api.SetJwt(jwt);
            _accountId = accountId;
        }

        public async Task RouteAsync(StrategySignal sig, string contractId, CancellationToken ct = default)
        {
            var sideBuy0Sell1 = sig.Side == SignalSide.Long ? 0 : 1;

            // Apply configurable tick buffers to price
            decimal tick = InstrumentMeta.Tick(sig.Symbol);
            int bufTicks = ResolveBufferTicks(sig.Symbol);
            var basePx = sig.LimitPrice ?? 0m;
            var px = sig.Side == SignalSide.Long ? basePx + bufTicks * tick : basePx - bufTicks * tick;

            var live = (Environment.GetEnvironmentVariable("LIVE_TRADING") ??
                        Environment.GetEnvironmentVariable("TOPSTEPX_LIVE") ?? "false")
                        .Equals("true", StringComparison.OrdinalIgnoreCase);

            if (!live)
            {
                _log.LogInformation("[DRY-RUN] {Strat} {Sym} {Side} x{Size}@{Px} (+{Buf}t) -> {Cid}",
                    sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, bufTicks, contractId);
                return;
            }

            _log.LogInformation("[LIVE] placing {Strat} {Sym} {Side} x{Size}@{Px} (+{Buf}t) -> {Cid}",
                sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, bufTicks, contractId);

            var orderReq = new {
                accountId = _accountId,
                contractId = contractId,
                side = sideBuy0Sell1,
                qty = sig.Size,
                price = px,
                customTag = $"{sig.Strategy}-{sig.Symbol}-{DateTime.UtcNow:yyyyMMdd-HHmmss}"
            };
            await _api.PlaceOrderAsync(orderReq, ct);
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
