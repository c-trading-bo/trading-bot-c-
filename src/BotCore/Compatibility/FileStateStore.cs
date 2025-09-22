using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using System.Text.Json;

namespace BotCore.Compatibility;

/// <summary>
/// File-based state store that tracks Neural UCB learning progress across restarts
/// 
/// State Persistence Addition: Builds institutional memory without interfering 
/// with your existing state management.
/// </summary>
public class FileStateStore : IDisposable
{
    private readonly ILogger<FileStateStore> _logger;
    private readonly FileStateStoreConfig _config;
    private readonly Timer _persistenceTimer;
    
    // In-memory cache
    private readonly Dictionary<string, EnhancedTradingDecision> _decisionCache = new();
    private readonly Dictionary<string, LearningState> _learningStateCache = new();
    private readonly object _cacheLock = new();
    
    public FileStateStore(ILogger<FileStateStore> logger, FileStateStoreConfig config)
    {
        _logger = logger;
        _config = config;
        
        // Ensure state directory exists
        Directory.CreateDirectory(_config.StateDirectory);
        
        // Initialize periodic persistence
        _persistenceTimer = new Timer(PersistStateToFiles, null, 
            TimeSpan.FromMinutes(_config.PersistenceIntervalMinutes),
            TimeSpan.FromMinutes(_config.PersistenceIntervalMinutes));
        
        // Load existing state
        LoadStateFromFiles();
        
        _logger.LogInformation("FileStateStore initialized with directory: {StateDirectory}", 
            _config.StateDirectory);
    }
    
    /// <summary>
    /// Save decision state for learning
    /// </summary>
    public async Task SaveDecisionStateAsync(
        EnhancedTradingDecision decision,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var decisionId = GenerateDecisionId(decision);
            
            lock (_cacheLock)
            {
                _decisionCache[decisionId] = decision;
            }
            
            // Also save learning state
            var learningState = new LearningState
            {
                BundleId = decision.ParameterBundle.BundleId,
                MarketContext = new
                {
                    decision.OriginalDecision.Symbol,
                    Timestamp = decision.TimestampUtc,
                    Parameters = decision.ParameterBundle
                },
                DecisionOutcome = new
                {
                    decision.OriginalDecision.Action,
                    decision.OriginalDecision.Quantity,
                    decision.OriginalDecision.Confidence
                }
            };
            
            lock (_cacheLock)
            {
                _learningStateCache[decisionId] = learningState;
            }
            
            _logger.LogDebug("Saved decision state: {DecisionId}", decisionId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving decision state");
        }
    }
    
    /// <summary>
    /// Load decision state for outcome processing
    /// </summary>
    public async Task<EnhancedTradingDecision?> LoadDecisionStateAsync(
        string decisionId,
        CancellationToken cancellationToken = default)
    {
        try
        {
            lock (_cacheLock)
            {
                return _decisionCache.GetValueOrDefault(decisionId);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading decision state for {DecisionId}", decisionId);
            return null;
        }
    }
    
    /// <summary>
    /// Save learning progress state
    /// </summary>
    public async Task SaveLearningProgressAsync(
        string bundleId,
        LearningProgress progress,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var progressFile = Path.Combine(_config.StateDirectory, "learning_progress.json");
            
            Dictionary<string, LearningProgress> allProgress;
            if (File.Exists(progressFile))
            {
                var json = await File.ReadAllTextAsync(progressFile, cancellationToken);
                allProgress = JsonSerializer.Deserialize<Dictionary<string, LearningProgress>>(json) ?? new();
            }
            else
            {
                allProgress = new Dictionary<string, LearningProgress>();
            }
            
            allProgress[bundleId] = progress;
            
            var updatedJson = JsonSerializer.Serialize(allProgress, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            
            await File.WriteAllTextAsync(progressFile, updatedJson, cancellationToken);
            
            _logger.LogDebug("Saved learning progress for bundle: {BundleId}", bundleId);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving learning progress for {BundleId}", bundleId);
        }
    }
    
    /// <summary>
    /// Load learning progress for all bundles
    /// </summary>
    public async Task<Dictionary<string, LearningProgress>> LoadLearningProgressAsync(
        CancellationToken cancellationToken = default)
    {
        try
        {
            var progressFile = Path.Combine(_config.StateDirectory, "learning_progress.json");
            
            if (!File.Exists(progressFile))
            {
                return new Dictionary<string, LearningProgress>();
            }
            
            var json = await File.ReadAllTextAsync(progressFile, cancellationToken);
            return JsonSerializer.Deserialize<Dictionary<string, LearningProgress>>(json) ?? 
                   new Dictionary<string, LearningProgress>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading learning progress");
            return new Dictionary<string, LearningProgress>();
        }
    }
    
    /// <summary>
    /// Save bracket-specific performance metrics for learning optimization
    /// </summary>
    public async Task SaveBracketPerformanceAsync(
        string bundleId,
        BracketPerformanceMetrics metrics,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var bracketFile = Path.Combine(_config.StateDirectory, "bracket_performance.json");
            
            Dictionary<string, BracketPerformanceMetrics> allMetrics;
            if (File.Exists(bracketFile))
            {
                var json = await File.ReadAllTextAsync(bracketFile, cancellationToken);
                allMetrics = JsonSerializer.Deserialize<Dictionary<string, BracketPerformanceMetrics>>(json) ?? new();
            }
            else
            {
                allMetrics = new Dictionary<string, BracketPerformanceMetrics>();
            }
            
            allMetrics[bundleId] = metrics;
            
            var updatedJson = JsonSerializer.Serialize(allMetrics, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            await File.WriteAllTextAsync(bracketFile, updatedJson, cancellationToken);
            
            _logger.LogDebug("Saved bracket performance for {BundleId}: {Metrics}", 
                bundleId, $"Win Rate: {metrics.BracketWinRate:P1}, Profit Factor: {metrics.ProfitFactor:F2}");
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error saving bracket performance for {BundleId}", bundleId);
        }
    }
    
    /// <summary>
    /// Load bracket performance metrics for all bundles
    /// </summary>
    public async Task<Dictionary<string, BracketPerformanceMetrics>> LoadBracketPerformanceAsync(
        CancellationToken cancellationToken = default)
    {
        try
        {
            var bracketFile = Path.Combine(_config.StateDirectory, "bracket_performance.json");
            
            if (!File.Exists(bracketFile))
            {
                return new Dictionary<string, BracketPerformanceMetrics>();
            }
            
            var json = await File.ReadAllTextAsync(bracketFile, cancellationToken);
            return JsonSerializer.Deserialize<Dictionary<string, BracketPerformanceMetrics>>(json) ?? 
                   new Dictionary<string, BracketPerformanceMetrics>();
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading bracket performance");
            return new Dictionary<string, BracketPerformanceMetrics>();
        }
    }
    
    /// <summary>
    /// Update bracket performance with position outcome
    /// Integrates with existing Neural UCB learning for bracket optimization
    /// </summary>
    public async Task UpdateBracketOutcomeAsync(
        string bundleId,
        BracketOutcome outcome,
        decimal pnl,
        TimeSpan holdingTime,
        CancellationToken cancellationToken = default)
    {
        try
        {
            var allMetrics = await LoadBracketPerformanceAsync(cancellationToken);
            
            if (!allMetrics.TryGetValue(bundleId, out var metrics))
            {
                // Create new metrics for this bundle
                var bundle = BotCore.Bandits.ParameterBundleFactory.ParseBundle(bundleId);
                metrics = new BracketPerformanceMetrics
                {
                    BracketModeType = bundle?.BracketMode.ModeType ?? "Unknown",
                    StopTicks = bundle?.BracketMode.StopTicks ?? 12,
                    TargetTicks = bundle?.BracketMode.TargetTicks ?? 18,
                    RiskRewardRatio = bundle?.BracketMode.RiskRewardRatio ?? 1.5m
                };
            }
            
            metrics.UpdateWithOutcome(outcome, pnl, holdingTime);
            
            await SaveBracketPerformanceAsync(bundleId, metrics, cancellationToken);
            
            _logger.LogInformation("[BRACKET_LEARNING] Updated bracket outcome for {BundleId}: " +
                                 "outcome={Outcome}, pnl={Pnl:F2}, winRate={WinRate:P1}",
                bundleId, outcome, pnl, metrics.BracketWinRate);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error updating bracket outcome for {BundleId}", bundleId);
        }
    }
    
    private void LoadStateFromFiles()
    {
        try
        {
            // Load decisions
            var decisionsFile = Path.Combine(_config.StateDirectory, "decisions.json");
            if (File.Exists(decisionsFile))
            {
                var json = File.ReadAllText(decisionsFile);
                var decisions = JsonSerializer.Deserialize<Dictionary<string, EnhancedTradingDecision>>(json);
                if (decisions != null)
                {
                    lock (_cacheLock)
                    {
                        foreach (var kvp in decisions)
                        {
                            _decisionCache[kvp.Key] = kvp.Value;
                        }
                    }
                }
            }
            
            // Load learning states
            var learningFile = Path.Combine(_config.StateDirectory, "learning_states.json");
            if (File.Exists(learningFile))
            {
                var json = File.ReadAllText(learningFile);
                var learningStates = JsonSerializer.Deserialize<Dictionary<string, LearningState>>(json);
                if (learningStates != null)
                {
                    lock (_cacheLock)
                    {
                        foreach (var kvp in learningStates)
                        {
                            _learningStateCache[kvp.Key] = kvp.Value;
                        }
                    }
                }
            }
            
            _logger.LogInformation("Loaded {DecisionCount} decisions and {LearningStateCount} learning states from files",
                _decisionCache.Count, _learningStateCache.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error loading state from files");
        }
    }
    
    private void PersistStateToFiles(object? state)
    {
        try
        {
            Dictionary<string, EnhancedTradingDecision> decisions;
            Dictionary<string, LearningState> learningStates;
            
            lock (_cacheLock)
            {
                decisions = new Dictionary<string, EnhancedTradingDecision>(_decisionCache);
                learningStates = new Dictionary<string, LearningState>(_learningStateCache);
            }
            
            // Save decisions
            var decisionsFile = Path.Combine(_config.StateDirectory, "decisions.json");
            var decisionsJson = JsonSerializer.Serialize(decisions, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            File.WriteAllText(decisionsFile, decisionsJson);
            
            // Save learning states
            var learningFile = Path.Combine(_config.StateDirectory, "learning_states.json");
            var learningJson = JsonSerializer.Serialize(learningStates, new JsonSerializerOptions 
            { 
                WriteIndented = true 
            });
            File.WriteAllText(learningFile, learningJson);
            
            _logger.LogDebug("Persisted {DecisionCount} decisions and {LearningStateCount} learning states to files",
                decisions.Count, learningStates.Count);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error persisting state to files");
        }
    }
    
    private string GenerateDecisionId(EnhancedTradingDecision decision)
    {
        return $"{decision.OriginalDecision.Symbol}_{decision.TimestampUtc:yyyyMMdd_HHmmss}_{Guid.NewGuid():N}";
    }
    
    public void Dispose()
    {
        _persistenceTimer?.Dispose();
        PersistStateToFiles(null); // Final persistence
        
        _logger.LogInformation("FileStateStore disposed");
    }
}

/// <summary>
/// Learning state for tracking decision outcomes
/// </summary>
public class LearningState
{
    public string BundleId { get; set; } = string.Empty;
    public object MarketContext { get; set; } = new();
    public object DecisionOutcome { get; set; } = new();
    public DateTime CreatedUtc { get; set; } = DateTime.UtcNow;
}

/// <summary>
/// Learning progress tracking with bracket-specific metrics
/// </summary>
public class LearningProgress
{
    public string BundleId { get; set; } = string.Empty;
    public int TotalDecisions { get; set; }
    public int SuccessfulDecisions { get; set; }
    public decimal TotalReward { get; set; }
    public decimal AverageReward => TotalDecisions > 0 ? TotalReward / TotalDecisions : 0;
    public double SuccessRate => TotalDecisions > 0 ? (double)SuccessfulDecisions / TotalDecisions : 0;
    public DateTime LastUpdatedUtc { get; set; } = DateTime.UtcNow;
    
    /// <summary>
    /// Bracket-specific performance metrics
    /// </summary>
    public BracketPerformanceMetrics BracketMetrics { get; set; } = new();
}

/// <summary>
/// Bracket-specific performance tracking for learning
/// </summary>
public class BracketPerformanceMetrics
{
    /// <summary>
    /// Bracket mode used for this performance tracking
    /// </summary>
    public string BracketModeType { get; set; } = string.Empty;
    
    /// <summary>
    /// Stop ticks configuration
    /// </summary>
    public int StopTicks { get; set; }
    
    /// <summary>
    /// Target ticks configuration
    /// </summary>
    public int TargetTicks { get; set; }
    
    /// <summary>
    /// Risk-to-reward ratio for this bracket
    /// </summary>
    public decimal RiskRewardRatio { get; set; }
    
    /// <summary>
    /// Number of positions that hit stop loss
    /// </summary>
    public int StopHits { get; set; }
    
    /// <summary>
    /// Number of positions that hit take profit
    /// </summary>
    public int TargetHits { get; set; }
    
    /// <summary>
    /// Number of positions closed at breakeven
    /// </summary>
    public int BreakevenExits { get; set; }
    
    /// <summary>
    /// Number of positions that were trailed to profit
    /// </summary>
    public int TrailedExits { get; set; }
    
    /// <summary>
    /// Total profit/loss for positions using this bracket mode
    /// </summary>
    public decimal TotalPnL { get; set; }
    
    /// <summary>
    /// Average holding time for positions using this bracket mode
    /// </summary>
    public TimeSpan AverageHoldingTime { get; set; }
    
    /// <summary>
    /// Win rate specifically for this bracket mode
    /// </summary>
    public double BracketWinRate => TotalPositions > 0 ? (double)(TargetHits + TrailedExits) / TotalPositions : 0.0;
    
    /// <summary>
    /// Total positions using this bracket mode
    /// </summary>
    public int TotalPositions => StopHits + TargetHits + BreakevenExits + TrailedExits;
    
    /// <summary>
    /// Profit factor for this bracket mode
    /// </summary>
    public decimal ProfitFactor
    {
        get
        {
            var totalWins = (TargetHits + TrailedExits);
            var totalLosses = StopHits;
            
            if (totalLosses == 0) return totalWins > 0 ? decimal.MaxValue : 0m;
            if (totalWins == 0) return 0m;
            
            var avgWin = TotalPnL > 0 ? TotalPnL / totalWins : 0m;
            var avgLoss = TotalPnL < 0 ? Math.Abs(TotalPnL) / totalLosses : 0m;
            
            return avgLoss > 0 ? avgWin / avgLoss : 0m;
        }
    }
    
    /// <summary>
    /// Update bracket metrics with new position outcome
    /// </summary>
    public void UpdateWithOutcome(BracketOutcome outcome, decimal pnl, TimeSpan holdingTime)
    {
        switch (outcome)
        {
            case BracketOutcome.StopHit:
                StopHits++;
                break;
            case BracketOutcome.TargetHit:
                TargetHits++;
                break;
            case BracketOutcome.BreakevenExit:
                BreakevenExits++;
                break;
            case BracketOutcome.TrailedExit:
                TrailedExits++;
                break;
        }
        
        TotalPnL += pnl;
        
        // Update average holding time
        var totalPositions = TotalPositions;
        if (totalPositions > 1)
        {
            var totalMs = AverageHoldingTime.TotalMilliseconds * (totalPositions - 1) + holdingTime.TotalMilliseconds;
            AverageHoldingTime = TimeSpan.FromMilliseconds(totalMs / totalPositions);
        }
        else
        {
            AverageHoldingTime = holdingTime;
        }
    }
}

/// <summary>
/// Bracket outcome types for performance tracking
/// </summary>
public enum BracketOutcome
{
    StopHit,
    TargetHit,
    BreakevenExit,
    TrailedExit
}

/// <summary>
/// Configuration for file state store
/// </summary>
public class FileStateStoreConfig
{
    public string StateDirectory { get; set; } = Path.Combine(Directory.GetCurrentDirectory(), "state", "compatibility");
    public int PersistenceIntervalMinutes { get; set; } = 5;
    public int MaxCacheSize { get; set; } = 10000;
}