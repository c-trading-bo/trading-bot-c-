using System.Text.Json;
using Microsoft.Extensions.Logging;
using Trading.Safety.Models;

namespace Trading.Safety.Persistence;

/// <summary>
/// Production-grade position state persistence for disaster recovery
/// Handles graceful shutdown/startup with atomic operations and corruption protection
/// </summary>
public interface IPositionStatePersistence
{
    Task SavePositionStateAsync(PositionStateSnapshot snapshot);
    Task<PositionStateSnapshot?> LoadPositionStateAsync();
    Task SaveRiskStateAsync(Models.RiskState riskState);
    Task<Models.RiskState?> LoadRiskStateAsync();
    Task ArchiveCompletedSessionAsync(string sessionId, SessionSummary summary);
    Task<List<SessionSummary>> GetRecentSessionsAsync(int count = 10);
    Task CreateBackupAsync(string reason);
    Task<bool> ValidateIntegrityAsync();
}

/// <summary>
/// Complete position state snapshot for persistence
/// </summary>
public class PositionStateSnapshot
{
    public Guid SnapshotId { get; set; } = Guid.NewGuid();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string Version { get; set; } = "1.0";
    public Dictionary<string, PositionInfo> OpenPositions { get; set; } = new();
    public Dictionary<string, OrderIntent> PendingOrders { get; set; } = new();
    public List<TradeEvent> RecentTrades { get; set; } = new();
    public decimal TotalExposure { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public decimal RealizedPnL { get; set; }
    public string SessionId { get; set; } = string.Empty;
    public Dictionary<string, object> CustomState { get; set; } = new();
    
    // Integrity protection
    public string HashCode { get; set; } = string.Empty;
    public string SourceSystem { get; set; } = "TradingBot";
}

/// <summary>
/// Individual position information
/// </summary>
public class PositionInfo
{
    public string Symbol { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal AveragePrice { get; set; }
    public decimal CurrentPrice { get; set; }
    public decimal UnrealizedPnL { get; set; }
    public DateTime OpenTime { get; set; }
    public DateTime LastUpdate { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> Metadata { get; set; } = new();
}

/// <summary>
/// Order intent for reconstruction on restart
/// </summary>
public class OrderIntent
{
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public string OrderType { get; set; } = string.Empty;
    public DateTime CreatedTime { get; set; }
    public string Status { get; set; } = string.Empty;
    public decimal FilledQuantity { get; set; }
    public string Strategy { get; set; } = string.Empty;
}

/// <summary>
/// Trade event for audit trail
/// </summary>
public class TradeEvent
{
    public Guid TradeId { get; set; } = Guid.NewGuid();
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public string OrderId { get; set; } = string.Empty;
    public string CustomTag { get; set; } = string.Empty;
    public string Symbol { get; set; } = string.Empty;
    public string Side { get; set; } = string.Empty;
    public decimal Quantity { get; set; }
    public decimal Price { get; set; }
    public decimal PnL { get; set; }
    public string Strategy { get; set; } = string.Empty;
    public Dictionary<string, object> Context { get; set; } = new();
}

/// <summary>
/// Session summary for historical analysis
/// </summary>
public class SessionSummary
{
    public string SessionId { get; set; } = string.Empty;
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public TimeSpan Duration { get; set; }
    public decimal TotalPnL { get; set; }
    public decimal MaxDrawdown { get; set; }
    public int TradeCount { get; set; }
    public int WinningTrades { get; set; }
    public int LosingTrades { get; set; }
    public decimal WinRate { get; set; }
    public decimal SharpeRatio { get; set; }
    public List<string> Strategies { get; set; } = new();
    public Dictionary<string, decimal> StrategyPnL { get; set; } = new();
    public string TerminationReason { get; set; } = string.Empty;
}

/// <summary>
/// File-based position state persistence with atomic operations
/// </summary>
public class FilePositionStatePersistence : IPositionStatePersistence
{
    private readonly ILogger<FilePositionStatePersistence> _logger;
    private readonly string _baseDirectory;
    private readonly string _positionStateFile;
    private readonly string _riskStateFile;
    private readonly string _backupDirectory;
    private readonly string _sessionDirectory;
    private readonly object _fileLock = new object();
    
    private readonly JsonSerializerOptions _jsonOptions = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    public FilePositionStatePersistence(ILogger<FilePositionStatePersistence> logger)
    {
        _logger = logger;
        _baseDirectory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), 
                                     "TradingBot", "State");
        _positionStateFile = Path.Combine(_baseDirectory, "position_state.json");
        _riskStateFile = Path.Combine(_baseDirectory, "risk_state.json");
        _backupDirectory = Path.Combine(_baseDirectory, "Backups");
        _sessionDirectory = Path.Combine(_baseDirectory, "Sessions");
        
        // Ensure directories exist
        Directory.CreateDirectory(_baseDirectory);
        Directory.CreateDirectory(_backupDirectory);
        Directory.CreateDirectory(_sessionDirectory);
    }

    public async Task SavePositionStateAsync(PositionStateSnapshot snapshot)
    {
        try
        {
            snapshot.HashCode = CalculateHashCode(snapshot);
            var json = JsonSerializer.Serialize(snapshot, _jsonOptions);
            
            lock (_fileLock)
            {
                // Atomic save with temp file
                var tempFile = _positionStateFile + ".tmp";
                File.WriteAllText(tempFile, json);
                
                // Create backup before overwriting
                if (File.Exists(_positionStateFile))
                {
                    var backupFile = Path.Combine(_backupDirectory, 
                        $"position_state_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
                    File.Copy(_positionStateFile, backupFile);
                }
                
                // Atomic replace
                File.Move(tempFile, _positionStateFile);
            }
            
            _logger.LogInformation("[PERSISTENCE] Position state saved: {Positions} positions, {Orders} orders, " +
                                 "Exposure: {Exposure:C}", 
                snapshot.OpenPositions.Count, snapshot.PendingOrders.Count, snapshot.TotalExposure);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to save position state");
            throw;
        }
    }

    public async Task<PositionStateSnapshot?> LoadPositionStateAsync()
    {
        try
        {
            if (!File.Exists(_positionStateFile))
            {
                _logger.LogInformation("[PERSISTENCE] No existing position state found");
                return null;
            }

            string json;
            lock (_fileLock)
            {
                json = File.ReadAllText(_positionStateFile);
            }
            
            var snapshot = JsonSerializer.Deserialize<PositionStateSnapshot>(json, _jsonOptions);
            if (snapshot == null)
            {
                _logger.LogWarning("[PERSISTENCE] Failed to deserialize position state");
                return null;
            }
            
            // Validate integrity
            var expectedHash = CalculateHashCode(snapshot);
            if (snapshot.HashCode != expectedHash)
            {
                _logger.LogError("[PERSISTENCE] Position state integrity check failed - hash mismatch");
                
                // Try to load from backup
                var backupSnapshot = await TryLoadFromBackupAsync();
                if (backupSnapshot != null)
                {
                    _logger.LogInformation("[PERSISTENCE] Loaded position state from backup");
                    return backupSnapshot;
                }
                
                throw new InvalidDataException("Position state corrupted and no valid backup found");
            }
            
            _logger.LogInformation("[PERSISTENCE] Position state loaded: {Positions} positions, {Orders} orders, " +
                                 "Age: {Age:hh\\:mm\\:ss}", 
                snapshot.OpenPositions.Count, snapshot.PendingOrders.Count, 
                DateTime.UtcNow - snapshot.Timestamp);
            
            return snapshot;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to load position state");
            return null;
        }
    }

    public async Task SaveRiskStateAsync(Models.RiskState riskState)
    {
        try
        {
            var json = JsonSerializer.Serialize(riskState, _jsonOptions);
            
            lock (_fileLock)
            {
                var tempFile = _riskStateFile + ".tmp";
                File.WriteAllText(tempFile, json);
                File.Move(tempFile, _riskStateFile);
            }
            
            _logger.LogDebug("[PERSISTENCE] Risk state saved: PnL={PnL:C}, Risk Level={Level}", 
                riskState.DailyPnL, riskState.RiskLevel);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to save risk state");
            throw;
        }
    }

    public async Task<Models.RiskState?> LoadRiskStateAsync()
    {
        try
        {
            if (!File.Exists(_riskStateFile))
            {
                return null;
            }

            string json;
            lock (_fileLock)
            {
                json = File.ReadAllText(_riskStateFile);
            }
            
            var riskState = JsonSerializer.Deserialize<Models.RiskState>(json, _jsonOptions);
            
            _logger.LogInformation("[PERSISTENCE] Risk state loaded: PnL={PnL:C}, Risk Level={Level}", 
                riskState?.DailyPnL ?? 0, riskState?.RiskLevel ?? "UNKNOWN");
            
            return riskState;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to load risk state");
            return null;
        }
    }

    public async Task ArchiveCompletedSessionAsync(string sessionId, SessionSummary summary)
    {
        try
        {
            var sessionFile = Path.Combine(_sessionDirectory, $"session_{sessionId}_{DateTime.UtcNow:yyyyMMdd_HHmmss}.json");
            var json = JsonSerializer.Serialize(summary, _jsonOptions);
            
            await File.WriteAllTextAsync(sessionFile, json);
            
            _logger.LogInformation("[PERSISTENCE] Session archived: {SessionId}, PnL={PnL:C}, Duration={Duration}",
                sessionId, summary.TotalPnL, summary.Duration);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to archive session {SessionId}", sessionId);
        }
    }

    public async Task<List<SessionSummary>> GetRecentSessionsAsync(int count = 10)
    {
        try
        {
            var sessionFiles = Directory.GetFiles(_sessionDirectory, "session_*.json")
                                       .OrderByDescending(f => File.GetCreationTime(f))
                                       .Take(count);
            
            var sessions = new List<SessionSummary>();
            
            foreach (var file in sessionFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(file);
                    var session = JsonSerializer.Deserialize<SessionSummary>(json, _jsonOptions);
                    if (session != null)
                    {
                        sessions.Add(session);
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "[PERSISTENCE] Failed to load session file {File}", file);
                }
            }
            
            return sessions;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to get recent sessions");
            return new List<SessionSummary>();
        }
    }

    public async Task CreateBackupAsync(string reason)
    {
        try
        {
            var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            var backupName = $"full_backup_{timestamp}_{reason}";
            var backupPath = Path.Combine(_backupDirectory, backupName);
            
            Directory.CreateDirectory(backupPath);
            
            // Backup position state
            if (File.Exists(_positionStateFile))
            {
                File.Copy(_positionStateFile, Path.Combine(backupPath, "position_state.json"));
            }
            
            // Backup risk state
            if (File.Exists(_riskStateFile))
            {
                File.Copy(_riskStateFile, Path.Combine(backupPath, "risk_state.json"));
            }
            
            // Create backup manifest
            var manifest = new
            {
                Timestamp = DateTime.UtcNow,
                Reason = reason,
                Files = Directory.GetFiles(backupPath).Select(Path.GetFileName).ToArray()
            };
            
            var manifestJson = JsonSerializer.Serialize(manifest, _jsonOptions);
            await File.WriteAllTextAsync(Path.Combine(backupPath, "manifest.json"), manifestJson);
            
            _logger.LogInformation("[PERSISTENCE] Backup created: {BackupName}", backupName);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to create backup for reason: {Reason}", reason);
        }
    }

    public async Task<bool> ValidateIntegrityAsync()
    {
        try
        {
            var snapshot = await LoadPositionStateAsync();
            if (snapshot == null)
            {
                return true; // No state to validate
            }
            
            // Additional integrity checks can be added here
            var isValid = !string.IsNullOrEmpty(snapshot.SessionId) &&
                         snapshot.Timestamp > DateTime.MinValue &&
                         snapshot.OpenPositions != null &&
                         snapshot.PendingOrders != null;
            
            _logger.LogInformation("[PERSISTENCE] Integrity validation: {Result}", isValid ? "PASSED" : "FAILED");
            return isValid;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Integrity validation failed");
            return false;
        }
    }

    private string CalculateHashCode(PositionStateSnapshot snapshot)
    {
        // Create a deterministic hash based on critical fields
        var hashInput = $"{snapshot.Timestamp:O}|{snapshot.TotalExposure}|{snapshot.UnrealizedPnL}|" +
                       $"{snapshot.OpenPositions.Count}|{snapshot.PendingOrders.Count}";
        
        using var sha256 = System.Security.Cryptography.SHA256.Create();
        var hashBytes = sha256.ComputeHash(System.Text.Encoding.UTF8.GetBytes(hashInput));
        return Convert.ToHexString(hashBytes);
    }

    private async Task<PositionStateSnapshot?> TryLoadFromBackupAsync()
    {
        try
        {
            var backupFiles = Directory.GetFiles(_backupDirectory, "position_state_*.json")
                                      .OrderByDescending(f => File.GetCreationTime(f))
                                      .Take(5);
            
            foreach (var backupFile in backupFiles)
            {
                try
                {
                    var json = await File.ReadAllTextAsync(backupFile);
                    var snapshot = JsonSerializer.Deserialize<PositionStateSnapshot>(json, _jsonOptions);
                    
                    if (snapshot != null && CalculateHashCode(snapshot) == snapshot.HashCode)
                    {
                        _logger.LogInformation("[PERSISTENCE] Valid backup found: {BackupFile}", backupFile);
                        return snapshot;
                    }
                }
                catch
                {
                    // Continue to next backup
                }
            }
            
            return null;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "[PERSISTENCE] Failed to load from backup");
            return null;
        }
    }
}