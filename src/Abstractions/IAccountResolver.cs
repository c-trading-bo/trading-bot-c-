using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.Abstractions;

/// <summary>
/// Interface for resolving and managing the active TopstepX account
/// </summary>
public interface IAccountResolver
{
    /// <summary>
    /// The currently active account for this session
    /// </summary>
    ActiveAccount? CurrentAccount { get; }

    /// <summary>
    /// Resolve the active account from TopstepX using intelligent selection rules
    /// </summary>
    Task<ActiveAccount?> ResolveActiveAccountAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Check if the current account is still valid/active
    /// </summary>
    Task<bool> ValidateCurrentAccountAsync(CancellationToken cancellationToken = default);

    /// <summary>
    /// Event fired when the active account changes
    /// </summary>
    event EventHandler<AccountChangedEventArgs>? AccountChanged;
}

/// <summary>
/// Event arguments for account change events
/// </summary>
public class AccountChangedEventArgs : EventArgs
{
    public ActiveAccount? Account { get; }

    public AccountChangedEventArgs(ActiveAccount? account)
    {
        Account = account;
    }
}
