using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;
using AbstractionsTradingDecision = TradingBot.Abstractions.TradingDecision;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for trading brain adapter that bridges old and new architectures
/// </summary>
public interface ITradingBrainAdapter
{
    /// <summary>
    /// Make trading decision using champion brain with challenger shadow testing
    /// </summary>
    Task<AbstractionsTradingDecision> DecideAsync(TradingContext context, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Manually promote challenger to primary (for testing/demonstration)
    /// </summary>
    Task<bool> PromoteToChallengerAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Rollback to champion
    /// </summary>
    Task<bool> RollbackToChampionAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get current adapter statistics
    /// </summary>
    AdapterStatistics GetStatistics();
}