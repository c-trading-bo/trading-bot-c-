using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services.State
{
    public interface ILoginCompletionState
    {
        Task WaitForLoginCompletion();
        void SetLoginCompleted();
    }
}
