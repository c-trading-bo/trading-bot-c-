using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Services.State
{
    internal interface ILoginCompletionState
    {
        Task WaitForLoginCompletion();
        void SetLoginCompleted();
    }
}
