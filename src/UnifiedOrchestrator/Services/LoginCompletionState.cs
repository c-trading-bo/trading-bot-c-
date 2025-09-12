namespace TradingBot.UnifiedOrchestrator.Services;

/// <summary>
/// Local interface for login completion state management
/// </summary>
public interface ILoginCompletionState
{
    Task WaitForLoginCompletion();
    void SetLoginCompleted();
}

/// <summary>
/// Simple implementation of ILoginCompletionState for the UnifiedOrchestrator
/// </summary>
public class SimpleLoginCompletionState : ILoginCompletionState
{
    private readonly TaskCompletionSource _loginCompletedTcs = new();
    private volatile bool _isLoginCompleted = false;

    public async Task WaitForLoginCompletion()
    {
        if (_isLoginCompleted)
            return;
            
        await _loginCompletedTcs.Task;
    }

    public void SetLoginCompleted()
    {
        if (!_isLoginCompleted)
        {
            _isLoginCompleted = true;
            _loginCompletedTcs.SetResult();
        }
    }
}