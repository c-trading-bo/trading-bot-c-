namespace TradingBot.Abstractions
{
    public class LoginCompletionState : ILoginCompletionState
    {
        private readonly TaskCompletionSource<bool> _loginCompleted = new TaskCompletionSource<bool>();

        public Task WaitForLoginCompletion()
        {
            return _loginCompleted.Task;
        }

        public void SetLoginCompleted()
        {
            _loginCompleted.TrySetResult(true);
        }
    }
}
