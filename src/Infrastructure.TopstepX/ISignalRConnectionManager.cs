using System;
using System.Threading.Tasks;

namespace Infrastructure.TopstepX;

/// <summary>
/// Interface for token providers that support forced refresh
/// </summary>
public interface ITokenRefresher
{
    Task ForceRefreshAsync();
}

// NOTE: ISignalRConnectionManager interface was moved to TradingBot.Abstractions
// to avoid duplication. Use TradingBot.Abstractions.ISignalRConnectionManager instead.
