using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for rollback drill service
/// </summary>
public interface IRollbackDrillService
{
    /// <summary>
    /// Execute comprehensive rollback drill under simulated trading load
    /// </summary>
    Task<RollbackDrillResult> ExecuteRollbackDrillAsync(RollbackDrillConfig config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Execute quick rollback drill for demonstration
    /// </summary>
    Task<RollbackDrillResult> ExecuteQuickDrillAsync(CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get rollback drill history
    /// </summary>
    List<RollbackDrillResult> GetDrillHistory(int maxCount = 20);
    
    /// <summary>
    /// Generate summary report of all rollback drills
    /// </summary>
    RollbackDrillSummary GenerateSummaryReport();
}