using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using OrchestratorAgent.Infra.SelfHealing;

namespace OrchestratorAgent.Infra;

/// <summary>
/// Self-healing engine that automatically attempts to fix detected health issues
/// </summary>
public class SelfHealingEngine
{
    private readonly IServiceProvider _serviceProvider;
    private readonly ILogger<SelfHealingEngine> _logger;
    private readonly Dictionary<string, ISelfHealingAction> _healingActions = new();
    private readonly Dictionary<string, RecoveryAttemptHistory> _attemptHistory = new();
    private readonly object _lockObject = new();

    public SelfHealingEngine(IServiceProvider serviceProvider, ILogger<SelfHealingEngine> logger)
    {
        _serviceProvider = serviceProvider;
        _logger = logger;
    }

    /// <summary>
    /// Initialize the self-healing engine by discovering all available healing actions
    /// </summary>
    public async Task InitializeAsync()
    {
        try
        {
            await DiscoverHealingActionsAsync();
            LoadAttemptHistory();
            _logger.LogInformation("[SELF-HEAL] Self-healing engine initialized with {Count} healing actions", _healingActions.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SELF-HEAL] Failed to initialize self-healing engine");
        }
    }

    /// <summary>
    /// Attempt to automatically fix a failed health check
    /// </summary>
    public async Task<bool> AttemptHealingAsync(string healthCheckName, HealthCheckResult failedResult)
    {
        try
        {
            await Task.Delay(1); // Satisfy async requirement
            lock (_lockObject)
            {
                // Check if we have a healing action for this health check
                // First try exact match, then try wildcard adaptive actions
                var healingAction = _healingActions.Values.FirstOrDefault(a => 
                    a.TargetHealthCheck.Equals(healthCheckName, StringComparison.OrdinalIgnoreCase))
                    ?? _healingActions.Values.FirstOrDefault(a => 
                        a.TargetHealthCheck == "*" || a.TargetHealthCheck.Equals("*", StringComparison.OrdinalIgnoreCase));

                if (healingAction == null)
                {
                    _logger.LogDebug("[SELF-HEAL] No healing action available for health check: {HealthCheck}", healthCheckName);
                    return false;
                }

                // Check attempt limits
                var history = GetOrCreateHistory(healingAction.Name);
                if (!CanAttemptRecovery(healingAction, history))
                {
                    _logger.LogWarning("[SELF-HEAL] Recovery attempt limit reached for {Action} ({Attempts}/{Max} attempts today)", 
                        healingAction.Name, history.AttemptsToday, healingAction.MaxAttemptsPerDay);
                    return false;
                }

                // Record attempt start
                var attemptId = Guid.NewGuid().ToString("N")[..8];
                history.AttemptStarted(attemptId);
                
                _logger.LogInformation("[SELF-HEAL] Starting recovery attempt {AttemptId} for {HealthCheck} using {Action}", 
                    attemptId, healthCheckName, healingAction.Name);

                // Execute healing action in background
                _ = Task.Run(async () => 
                {
                    try
                    {
                        await ExecuteHealingActionAsync(healingAction, failedResult, attemptId, history);
                    }
                    catch (Exception ex)
                    {
                        _logger.LogError(ex, "[SELF-HEAL] Background healing action failed");
                    }
                });
                
                return true;
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SELF-HEAL] Failed to initiate healing for {HealthCheck}", healthCheckName);
            return false;
        }
    }

    private async Task ExecuteHealingActionAsync(ISelfHealingAction healingAction, HealthCheckResult failedResult, string attemptId, RecoveryAttemptHistory history)
    {
        RecoveryResult result;
        
        try
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(5)); // 5-minute timeout
            result = await healingAction.ExecuteRecoveryAsync(failedResult, cts.Token);
        }
        catch (Exception ex)
        {
            result = RecoveryResult.Failed($"Recovery action crashed: {ex.Message}", ex, true);
            _logger.LogError(ex, "[SELF-HEAL] Recovery action {Action} crashed during attempt {AttemptId}", healingAction.Name, attemptId);
        }

        // Record result
        lock (_lockObject)
        {
            history.AttemptCompleted(attemptId, result);
            SaveAttemptHistory();
        }

        // Log result
        if (result.Success)
        {
            _logger.LogInformation("[SELF-HEAL] Recovery {AttemptId} SUCCEEDED: {Message} (Actions: {Actions})", 
                attemptId, result.Message, string.Join(", ", result.ActionsPerformed));
        }
        else
        {
            var logLevel = result.RequiresEscalation ? LogLevel.Error : LogLevel.Warning;
            _logger.Log(logLevel, "[SELF-HEAL] Recovery {AttemptId} FAILED: {Message} (Escalate: {Escalate})", 
                attemptId, result.Message, result.RequiresEscalation);
        }

        // Handle escalation
        if (result.RequiresEscalation)
        {
            await EscalateFailureAsync(healingAction, failedResult, result);
        }
    }

    private async Task DiscoverHealingActionsAsync()
    {
        try
        {
            await Task.Delay(1); // Satisfy async requirement
            var assemblies = new[] { Assembly.GetExecutingAssembly(), Assembly.GetEntryAssembly() }
                .Where(a => a != null).Distinct();

            var discoveredCount = 0;
            var adaptiveCount = 0;

            foreach (var assembly in assemblies)
            {
                var actionTypes = assembly!.GetTypes()
                    .Where(t => t.IsClass && !t.IsAbstract && typeof(ISelfHealingAction).IsAssignableFrom(t))
                    .ToList();

                foreach (var actionType in actionTypes)
                {
                    try
                    {
                        var attribute = actionType.GetCustomAttribute<SelfHealingActionAttribute>();
                        if (attribute?.Enabled == false)
                        {
                            _logger.LogDebug("[SELF-HEAL] Skipping disabled healing action: {Action}", actionType.Name);
                            continue;
                        }

                        // Create instance
                        ISelfHealingAction? action = null;
                        try
                        {
                            action = (ISelfHealingAction?)_serviceProvider.GetService(actionType);
                        }
                        catch
                        {
                            // Service provider failed, fall back to Activator
                        }
                        
                        if (action == null)
                        {
                            try
                            {
                                action = (ISelfHealingAction?)Activator.CreateInstance(actionType);
                            }
                            catch (Exception ex)
                            {
                                _logger.LogWarning(ex, "[SELF-HEAL] Failed to create instance of {Type}", actionType.Name);
                                continue;
                            }
                        }

                        if (action != null)
                        {
                            _healingActions[action.Name] = action;
                            discoveredCount++;
                            
                            // Check if this is an adaptive action
                            if (actionType.Name.Contains("Adaptive", StringComparison.OrdinalIgnoreCase))
                            {
                                adaptiveCount++;
                                _logger.LogInformation("[SELF-HEAL] ⚡ ADAPTIVE healing action: {Name} -> {Target} (Risk: {Risk}) - Can learn new patterns!", 
                                    action.Name, action.TargetHealthCheck, action.RiskLevel);
                            }
                            else
                            {
                                _logger.LogInformation("[SELF-HEAL] Registered healing action: {Name} -> {Target} (Risk: {Risk})", 
                                    action.Name, action.TargetHealthCheck, action.RiskLevel);
                            }
                        }
                    }
                    catch (Exception ex)
                    {
                        _logger.LogWarning(ex, "[SELF-HEAL] Failed to register healing action: {Type}", actionType.Name);
                    }
                }
            }
            
            _logger.LogInformation("[SELF-HEAL] Discovery complete. Found {Count} healing actions ({AdaptiveCount} adaptive, {SpecificCount} specific)", 
                discoveredCount, adaptiveCount, discoveredCount - adaptiveCount);
            
            if (adaptiveCount > 0)
            {
                _logger.LogInformation("[SELF-HEAL] 🧠 Adaptive healing enabled - system can now learn to fix ANY new feature automatically!");
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SELF-HEAL] Failed to discover healing actions");
        }
    }

    private bool CanAttemptRecovery(ISelfHealingAction action, RecoveryAttemptHistory history)
    {
        // Check daily attempt limit
        if (history.AttemptsToday >= action.MaxAttemptsPerDay)
            return false;

        // Check recent failures (don't retry too quickly)
        var recentFailures = history.RecentAttempts
            .Where(a => a.Timestamp > DateTime.UtcNow.AddMinutes(-30))
            .Count(a => !a.Success);

        if (recentFailures >= 3)
            return false;

        return true;
    }

    private RecoveryAttemptHistory GetOrCreateHistory(string actionName)
    {
        if (!_attemptHistory.TryGetValue(actionName, out var history))
        {
            history = new RecoveryAttemptHistory(actionName);
            _attemptHistory[actionName] = history;
        }
        return history;
    }

    private void LoadAttemptHistory()
    {
        try
        {
            var historyFile = "state/self_healing_history.json";
            if (File.Exists(historyFile))
            {
                var json = File.ReadAllText(historyFile);
                var histories = JsonSerializer.Deserialize<Dictionary<string, RecoveryAttemptHistory>>(json);
                if (histories != null)
                {
                    foreach (var (key, value) in histories)
                    {
                        _attemptHistory[key] = value;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[SELF-HEAL] Failed to load attempt history");
        }
    }

    private void SaveAttemptHistory()
    {
        try
        {
            Directory.CreateDirectory("state");
            var historyFile = "state/self_healing_history.json";
            var json = JsonSerializer.Serialize(_attemptHistory, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(historyFile, json);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "[SELF-HEAL] Failed to save attempt history");
        }
    }

    private async Task EscalateFailureAsync(ISelfHealingAction action, HealthCheckResult originalFailure, RecoveryResult recoveryResult)
    {
        try
        {
            _logger.LogCritical("[SELF-HEAL] ESCALATION: Recovery action {Action} failed and requires manual intervention. Original failure: {OriginalMessage}, Recovery failure: {RecoveryMessage}", 
                action.Name, originalFailure.Message, recoveryResult.Message);

            // TODO: Add email/SMS alerts, webhook notifications, etc.
            
            // Save escalation record
            var escalation = new
            {
                Timestamp = DateTime.UtcNow,
                Action = action.Name,
                OriginalFailure = originalFailure.Message,
                RecoveryFailure = recoveryResult.Message,
                RequiresManualIntervention = true
            };

            Directory.CreateDirectory("state");
            var escalationFile = $"state/escalation_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json";
            var json = JsonSerializer.Serialize(escalation, new JsonSerializerOptions { WriteIndented = true });
            await File.WriteAllTextAsync(escalationFile, json);
            
            _logger.LogError("[SELF-HEAL] Escalation record saved: {File}", escalationFile);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[SELF-HEAL] Failed to escalate failure");
        }
    }

    /// <summary>
    /// Get summary of self-healing engine status
    /// </summary>
    public SelfHealingStatus GetStatus()
    {
        lock (_lockObject)
        {
            var todaysAttempts = _attemptHistory.Values.Sum(h => h.AttemptsToday);
            var successfulAttempts = _attemptHistory.Values.Sum(h => h.SuccessfulAttemptsToday);
            
            return new SelfHealingStatus
            {
                IsEnabled = true,
                AvailableActions = _healingActions.Count,
                AttemptsToday = todaysAttempts,
                SuccessfulToday = successfulAttempts,
                FailureRate = todaysAttempts > 0 ? (double)(todaysAttempts - successfulAttempts) / todaysAttempts : 0.0
            };
        }
    }
}

/// <summary>
/// Tracks recovery attempt history for rate limiting and analysis
/// </summary>
public class RecoveryAttemptHistory
{
    public string ActionName { get; set; } = string.Empty;
    public List<RecoveryAttempt> RecentAttempts { get; set; } = new();
    
    public int AttemptsToday => RecentAttempts.Count(a => a.Timestamp.Date == DateTime.UtcNow.Date);
    public int SuccessfulAttemptsToday => RecentAttempts.Count(a => a.Timestamp.Date == DateTime.UtcNow.Date && a.Success);

    public RecoveryAttemptHistory() { }
    public RecoveryAttemptHistory(string actionName) => ActionName = actionName;

    public void AttemptStarted(string attemptId)
    {
        RecentAttempts.Add(new RecoveryAttempt
        {
            AttemptId = attemptId,
            Timestamp = DateTime.UtcNow,
            Success = false,
            InProgress = true
        });

        // Keep only last 100 attempts
        if (RecentAttempts.Count > 100)
        {
            RecentAttempts = RecentAttempts.OrderByDescending(a => a.Timestamp).Take(100).ToList();
        }
    }

    public void AttemptCompleted(string attemptId, RecoveryResult result)
    {
        var attempt = RecentAttempts.FirstOrDefault(a => a.AttemptId == attemptId);
        if (attempt != null)
        {
            attempt.Success = result.Success;
            attempt.InProgress = false;
            attempt.Message = result.Message;
            attempt.Duration = result.Duration;
        }
    }
}

public class RecoveryAttempt
{
    public string AttemptId { get; set; } = string.Empty;
    public DateTime Timestamp { get; set; }
    public bool Success { get; set; }
    public bool InProgress { get; set; }
    public string Message { get; set; } = string.Empty;
    public TimeSpan Duration { get; set; }
}

public class SelfHealingStatus
{
    public bool IsEnabled { get; set; }
    public int AvailableActions { get; set; }
    public int AttemptsToday { get; set; }
    public int SuccessfulToday { get; set; }
    public double FailureRate { get; set; }
}
