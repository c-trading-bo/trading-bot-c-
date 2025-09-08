using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Execution
{
    public interface IExecutionSink
    {
        Task<object> HandleAsync(NewOrder order, CancellationToken ct);
    }

    // Minimal order skeleton to adapt to models expected by the REST API
    public sealed record NewOrder(long AccountId, string ContractId, int Side, int Type, int Size, decimal? Price, decimal? Stop, decimal? Target);
}
