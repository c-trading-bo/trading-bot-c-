using System;
using System.Threading;
using System.Threading.Tasks;

namespace OrchestratorAgent.Infra;

/// <summary>
/// Interface for self-healing recovery actions that can be automatically executed when health checks fail
/// </summary>
public interface ISelfHealingAction
{
    /// <summary>
    /// Unique name for this recovery action
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Description of what this recovery action does
    /// </summary>
    string Description { get; }
    
    /// <summary>
    /// Which health check failure this action can potentially fix
    /// </summary>
    string TargetHealthCheck { get; }
    
    /// <summary>
    /// Risk level of this recovery action (Low, Medium, High)
    /// </summary>
    RecoveryRiskLevel RiskLevel { get; }
    
    /// <summary>
    /// Maximum number of times this action can be attempted in a 24-hour period
    /// </summary>
    int MaxAttemptsPerDay { get; }
    
    /// <summary>
    /// Attempt to fix the detected issue
    /// </summary>
    /// <param name="healthCheckResult">The failed health check result that triggered this recovery</param>
    /// <param name="cancellationToken">Cancellation token</param>
    /// <returns>Recovery result indicating success/failure and actions taken</returns>
    Task<RecoveryResult> ExecuteRecoveryAsync(HealthCheckResult healthCheckResult, CancellationToken cancellationToken = default);
}

/// <summary>
/// Result of a self-healing recovery attempt
/// </summary>
public class RecoveryResult
{
    public bool Success { get; set; }
    public string Message { get; set; } = string.Empty;
    public string[] ActionsPerformed { get; set; } = Array.Empty<string>();
    public Exception? Exception { get; set; }
    public bool RequiresEscalation { get; set; }
    public bool RequiresManualIntervention { get; set; }
    public TimeSpan Duration { get; set; }

    public static RecoveryResult Successful(string message, params string[] actions)
        => new() { Success = true, Message = message, ActionsPerformed = actions };

    public static RecoveryResult Failed(string message, Exception? ex = null, bool escalate = false)
        => new() { Success = false, Message = message, Exception = ex, RequiresEscalation = escalate };

    public static RecoveryResult PartialSuccess(string message, params string[] actions)
        => new() { Success = false, Message = message, ActionsPerformed = actions, RequiresEscalation = true };
}

/// <summary>
/// Risk level for recovery actions
/// </summary>
public enum RecoveryRiskLevel
{
    /// <summary>
    /// Safe actions like clearing cache, reloading config files, restarting background tasks
    /// </summary>
    Low,
    
    /// <summary>
    /// Moderate actions like restarting services, recreating connections, fixing file permissions
    /// </summary>
    Medium,
    
    /// <summary>
    /// High-risk actions like modifying system settings, restarting processes, emergency procedures
    /// </summary>
    High
}

/// <summary>
/// Attribute to mark self-healing actions for automatic discovery
/// </summary>
[AttributeUsage(AttributeTargets.Class)]
public class SelfHealingActionAttribute : Attribute
{
    public string? Category { get; set; }
    public bool Enabled { get; set; } = true;
    public RecoveryRiskLevel MaxRiskLevel { get; set; } = RecoveryRiskLevel.Medium;
}
