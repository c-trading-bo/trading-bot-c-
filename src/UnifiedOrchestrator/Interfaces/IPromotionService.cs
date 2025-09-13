using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for shadow testing challenger models against champions
/// </summary>
public interface IShadowTester
{
    /// <summary>
    /// Run shadow A/B test between challenger and champion
    /// </summary>
    Task<ValidationReport> RunShadowTestAsync(string algorithm, string challengerVersionId, ShadowTestConfig config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get ongoing shadow test status
    /// </summary>
    Task<ShadowTestStatus> GetTestStatusAsync(string testId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Cancel an ongoing shadow test
    /// </summary>
    Task<bool> CancelTestAsync(string testId, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for model promotion service with atomic swaps and rollback
/// </summary>
public interface IPromotionService
{
    /// <summary>
    /// Evaluate whether a challenger should be promoted
    /// </summary>
    Task<PromotionDecision> EvaluatePromotionAsync(string algorithm, string challengerVersionId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Promote a challenger to champion with atomic swap
    /// </summary>
    Task<bool> PromoteToChampionAsync(string algorithm, string challengerVersionId, string reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Rollback to previous champion (instant rollback < 100ms)
    /// </summary>
    Task<bool> RollbackToPreviousAsync(string algorithm, string reason, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get promotion status and history
    /// </summary>
    Task<PromotionStatus> GetPromotionStatusAsync(string algorithm, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Schedule automatic promotion for challenger
    /// </summary>
    Task<string> SchedulePromotionAsync(string algorithm, string challengerVersionId, PromotionSchedule schedule, CancellationToken cancellationToken = default);
}

/// <summary>
/// Shadow test configuration
/// </summary>
public class ShadowTestConfig
{
    public int MinTrades { get; set; } = 50;
    public int MinSessions { get; set; } = 5;
    public TimeSpan MaxTestDuration { get; set; } = TimeSpan.FromDays(7);
    public decimal SignificanceLevel { get; set; } = 0.05m; // p < 0.05
    public bool RequireBehaviorAlignment { get; set; } = true;
    public decimal MinAlignmentThreshold { get; set; } = 0.8m; // 80% alignment
    public Dictionary<string, object> Parameters { get; set; } = new();
}

/// <summary>
/// Shadow test status
/// </summary>
public class ShadowTestStatus
{
    public string TestId { get; set; } = string.Empty;
    public string Algorithm { get; set; } = string.Empty;
    public string ChallengerVersionId { get; set; } = string.Empty;
    public string ChampionVersionId { get; set; } = string.Empty;
    public string Status { get; set; } = "UNKNOWN"; // RUNNING, COMPLETED, FAILED, CANCELLED
    public DateTime StartTime { get; set; }
    public DateTime? EndTime { get; set; }
    public decimal Progress { get; set; } // 0.0 to 1.0
    public int TradesRecorded { get; set; }
    public int SessionsRecorded { get; set; }
    public Dictionary<string, object> IntermediateResults { get; set; } = new();
}

/// <summary>
/// Promotion status
/// </summary>
public class PromotionStatus
{
    public string Algorithm { get; set; } = string.Empty;
    public string CurrentChampionVersionId { get; set; } = string.Empty;
    public DateTime? LastPromotionTime { get; set; }
    public string LastPromotionReason { get; set; } = string.Empty;
    public bool HasPendingPromotion { get; set; }
    public string? PendingChallengerVersionId { get; set; }
    public DateTime? ScheduledPromotionTime { get; set; }
    public List<string> RecentPromotions { get; set; } = new();
    public bool CanRollback { get; set; }
}

/// <summary>
/// Promotion schedule
/// </summary>
public class PromotionSchedule
{
    public DateTime? ScheduledTime { get; set; } // null = next safe window
    public bool RequireFlat { get; set; } = true;
    public bool RequireSafeWindow { get; set; } = true;
    public TimeSpan MaxDelay { get; set; } = TimeSpan.FromHours(24);
    public string ApprovedBy { get; set; } = string.Empty;
    public Dictionary<string, object> Conditions { get; set; } = new();
}