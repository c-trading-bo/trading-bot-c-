using System.Threading;
using System.Threading.Tasks;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for unified data integration service that validates historical and live data connections
/// </summary>
public interface IUnifiedDataIntegrationService
{
    /// <summary>
    /// Validate data consistency across all sources
    /// </summary>
    Task<bool> ValidateDataConsistencyAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check historical data availability
    /// </summary>
    Task<bool> CheckHistoricalDataAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Check live data connectivity
    /// </summary>
    Task<bool> CheckLiveDataAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get data integration status report
    /// </summary>
    Task<object> GetDataIntegrationStatusAsync(CancellationToken cancellationToken = default);
}