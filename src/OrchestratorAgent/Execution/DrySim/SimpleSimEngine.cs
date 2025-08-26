using System.Threading;
using System.Threading.Tasks;
using OrchestratorAgent.Infra;

namespace OrchestratorAgent.Execution.DrySim
{
    public sealed class SimpleSimEngine : ISimEngine
    {
        private readonly PositionTracker _pos;
        public SimpleSimEngine(PositionTracker pos) => _pos = pos;

        public Task<object> FillAsync(OrchestratorAgent.Execution.NewOrder o, decimal bid, decimal ask, decimal tick, CancellationToken ct)
        {
            var slip = tick > 0m ? tick / 2m : 0m;
            var px = o.Side == 0 /* buy */ ? ask + slip : bid - slip;
            var signedQty = o.Side == 0 ? o.Size : -o.Size;

            _pos.ApplySimFill(o.ContractId, signedQty, px);
            return Task.FromResult<object>(new { status = "filled", price = px, o.AccountId, o.ContractId });
        }
    }
}
