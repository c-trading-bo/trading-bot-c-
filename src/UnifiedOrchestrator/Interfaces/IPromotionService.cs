using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TradingBot.UnifiedOrchestrator.Models;
using TradingBot.Abstractions;

namespace TradingBot.UnifiedOrchestrator.Interfaces;

/// <summary>
/// Interface for shadow testing challenger models against champions
/// </summary>
internal interface IShadowTester
{
    /// <summary>
    /// Run shadow A/B test between challenger and champion
    /// </summary>
    Task<PromotionTestReport> RunShadowTestAsync(string algorithm, string challengerVersionId, ShadowTestConfig config, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get ongoing shadow test status
    /// </summary>
    Task<ShadowTestStatus> GetTestStatusAsync(string testId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Cancel an ongoing shadow test
    /// </summary>
    Task<bool> CancelTestAsync(string testId, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Record a decision for shadow testing comparison
    /// </summary>
    Task RecordDecisionAsync(string algorithm, TradingContext context, TradingBot.Abstractions.TradingDecision decision, CancellationToken cancellationToken = default);
    
    /// <summary>
    /// Get recent shadow test results for analysis
    /// </summary>
    Task<IReadOnlyList<ShadowTestResult>> GetRecentResultsAsync(string algorithm, TimeSpan timeWindow, CancellationToken cancellationToken = default);
}

/// <summary>
/// Interface for model promotion service with atomic swaps and rollback
/// </summary>
internal interface IPromotionService
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
internal class ShadowTestConfig
{
    public int MinTrades { get; set; } = 50;
    public int MinSessions { get; set; } = 5;
    public TimeSpan MaxTestDuration { get; set; } = TimeSpan.FromDays(7);
    public decimal SignificanceLevel { get; set; } = 0.05m; // p < 0.05
    public bool RequireBehaviorAlignment { get; set; } = true;
    public decimal MinAlignmentThreshold { get; set; } = 0.8m; // 80% alignment
    public Dictionary<string, object> Parameters { get; } = new();
}

/// <summary>
/// Shadow test status
/// </summary>
internal class ShadowTestStatus
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
    public Dictionary<string, object> IntermediateResults { get; } = new();
}

/// <summary>
/// Promotion status
/// </summary>
internal class PromotionStatus
{
    public string Algorithm { get; set; } = string.Empty;
    public string CurrentChampionVersionId { get; set; } = string.Empty;
    public DateTime? LastPromotionTime { get; set; }
    public string LastPromotionReason { get; set; } = string.Empty;
    public bool HasPendingPromotion { get; set; }
    public string? PendingChallengerVersionId { get; set; }
    public DateTime? ScheduledPromotionTime { get; set; }
    public List<string> RecentPromotions { get; } = new();
    public bool CanRollback { get; set; }
}

/// <summary>
/// Promotion schedule
/// </summary>
internal class PromotionSchedule
{
    public DateTime? ScheduledTime { get; set; } // null = next safe window
    public bool RequireFlat { get; set; } = true;
    public bool RequireSafeWindow { get; set; } = true;
    public TimeSpan MaxDelay { get; set; } = TimeSpan.FromHours(24);
    public string ApprovedBy { get; set; } = string.Empty;
    public Dictionary<string, object> Conditions { get; } = new();
    
    // Required properties per production specification  
    public string Algorithm { get; set; } = string.Empty;
    public string ChallengerVersionId { get; set; } = string.Empty;
    public Models.ValidationReport ValidationReport { get; set; } = new(); // Use the detailed ValidationReport from Models
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public string Status { get; set; } = "Pending"; // Use string instead of enum to avoid conflicts
}