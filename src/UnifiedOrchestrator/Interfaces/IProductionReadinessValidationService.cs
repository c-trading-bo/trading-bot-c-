using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Services;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for production readiness validation service
/// </summary>
public interface IProductionReadinessValidationService
{
    /// <summary>
    /// Run complete production readiness validation and generate all requested artifacts
    /// </summary>
    Task<ProductionReadinessReport> RunCompleteValidationAsync(CancellationToken cancellationToken = default);
}