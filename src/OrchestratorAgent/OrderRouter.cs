#nullable enable
using System.Net.Http;
using Microsoft.Extensions.Logging;
using BotCore;

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
            var px = sig.LimitPrice ?? 0m;
            var live = (Environment.GetEnvironmentVariable("LIVE_TRADING") ??
                        Environment.GetEnvironmentVariable("TOPSTEPX_LIVE") ?? "false")
                        .Equals("true", StringComparison.OrdinalIgnoreCase);

            if (!live)
            {
                _log.LogInformation("[DRY-RUN] {Strat} {Sym} {Side} x{Size}@{Px} -> {Cid}",
                    sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, contractId);
                return;
            }

            _log.LogInformation("[LIVE] placing {Strat} {Sym} {Side} x{Size}@{Px} -> {Cid}",
                sig.Strategy, sig.Symbol, sig.Side, sig.Size, px, contractId);

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
    }
}
