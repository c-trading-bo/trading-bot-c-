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
            // EVAL SAFE: log only. Uncomment when ready to send.
            _log.LogInformation("[ROUTE] {Strat} {Sym} {Side} x{Size}@{Px} -> {Cid}",
                sig.Strategy, sig.Symbol, sig.Side, sig.Size, sig.LimitPrice, contractId);

            // If you want to place a protective far-away limit in eval, uncomment:
            // var sideBuy0Sell1 = sig.Side == SignalSide.Long ? 0 : 1;
            // var px = sig.LimitPrice ?? 0m;
            // await _api.PlaceLimit(_accountId, contractId, sideBuy0Sell1, sig.Size, px, ct);
        }
    }
}
