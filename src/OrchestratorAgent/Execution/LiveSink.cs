using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Execution
{
    public sealed class LiveSink : IExecutionSink
    {
        private readonly BotCore.ApiClient _api;
        public LiveSink(BotCore.ApiClient api) => _api = api;

        public async Task<object> HandleAsync(NewOrder o, CancellationToken ct)
        {
            // Pass-through to REST place order. Adapt property names if API requires.
            var payload = new
            {
                accountId = o.AccountId,
                contractId = o.ContractId,
                type = o.Type,
                side = o.Side,
                size = o.Size,
                price = o.Price,
                stop = o.Stop,
                target = o.Target
            };
            var orderId = await _api.PlaceOrderAsync(payload, ct);
            return new { status = "sent", orderId };
        }
    }
}
