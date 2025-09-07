using System.Threading;
using System.Threading.Tasks;
using OrchestratorAgent.Ops;
using OrchestratorAgent.Execution.DrySim;

namespace OrchestratorAgent.Execution
{
    public sealed class OrderRouter
    {
        private readonly TradeMode _mode;
        private readonly IExecutionSink _live;
        private readonly IExecutionSink _shadow;
        private readonly ISimEngine _sim;

        public OrderRouter(TradeMode mode, IExecutionSink live, IExecutionSink shadow, ISimEngine sim)
            => (_mode, _live, _shadow, _sim) = (mode, live, shadow, sim);

        // Provide best bid/ask/tick from callsite
        public Task<object> ExecuteAsync(NewOrder o, (decimal bid, decimal ask, decimal tick) mkt, CancellationToken ct)
            => _mode switch
            {
                TradeMode.Dry => _sim.FillAsync(o, mkt.bid, mkt.ask, mkt.tick, ct),
                TradeMode.Shadow => _shadow.HandleAsync(o, ct),
                TradeMode.Live => _live.HandleAsync(o, ct),
                _ => Task.FromResult<object>(new { status = "noop" })
            };
    }
}
