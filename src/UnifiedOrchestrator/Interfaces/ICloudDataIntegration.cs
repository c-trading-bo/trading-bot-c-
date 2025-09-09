using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for cloud data integration - links GitHub workflows to trading decisions
/// </summary>
public interface ICloudDataIntegration
{
    /// <summary>
    /// Sync all cloud data from GitHub workflows to the trading system
    /// </summary>
    Task SyncCloudDataForTradingAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get trading recommendation based on all cloud intelligence
    /// </summary>
    Task<CloudTradingRecommendation> GetTradingRecommendationAsync(string symbol, CancellationToken cancellationToken = default);
}