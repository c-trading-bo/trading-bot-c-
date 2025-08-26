using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Execution.DrySim
{
    public interface ISimEngine
    {
        Task<object> FillAsync(OrchestratorAgent.Execution.NewOrder order, decimal bestBid, decimal bestAsk, decimal tickSize, CancellationToken ct);
    }
}
