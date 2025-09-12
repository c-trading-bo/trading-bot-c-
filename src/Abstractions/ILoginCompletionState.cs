namespace TradingBot.Abstractions
{
    public interface ILoginCompletionState
    {
        Task WaitForLoginCompletion();
        void SetLoginCompleted();
    }
}
