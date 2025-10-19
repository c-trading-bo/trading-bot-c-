# 🔧 Bot/Trainer Split - Technical Specifications

**Purpose**: Detailed technical specifications for data structures, protocols, and interfaces  
**Version**: 1.0  
**Date**: October 19, 2025

---

## 📦 Brain Bundle Format

### Directory Structure
```
/opt/models/
├── active -> v49/           (Symlink to current version)
├── v47/                     (Old version, kept for rollback)
│   ├── manifest.json
│   ├── cvar_ppo_policy.onnx
│   ├── cvar_ppo_value.onnx
│   ├── cvar_ppo_cvar.onnx
│   ├── ucb_network.onnx
│   ├── lstm_predictor.onnx
│   ├── sac_actor.onnx
│   ├── sac_critic.onnx
│   ├── meta_learner.onnx
│   └── config.json
├── v48/                     (Previous version)
└── v49/                     (Current version)
    ├── manifest.json
    ├── cvar_ppo_policy.onnx
    ├── cvar_ppo_value.onnx
    ├── cvar_ppo_cvar.onnx
    ├── ucb_network.onnx
    ├── lstm_predictor.onnx
    ├── sac_actor.onnx
    ├── sac_critic.onnx
    ├── meta_learner.onnx
    └── config.json
```

### Manifest Schema (`manifest.json`)

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["version", "created_at", "models"],
  "properties": {
    "version": {
      "type": "string",
      "pattern": "^v[0-9]+$",
      "description": "Version identifier (e.g., v49)"
    },
    "created_at": {
      "type": "string",
      "format": "date-time",
      "description": "ISO 8601 timestamp of brain creation"
    },
    "training_duration_minutes": {
      "type": "integer",
      "minimum": 0,
      "description": "Total training time in minutes"
    },
    "experience_count": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of experiences used for training"
    },
    "historical_bars": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of historical bars used for training"
    },
    "models": {
      "type": "object",
      "description": "Dictionary of model name to model info",
      "additionalProperties": {
        "$ref": "#/definitions/ModelInfo"
      }
    },
    "performance": {
      "$ref": "#/definitions/PerformanceMetrics"
    },
    "training_config": {
      "$ref": "#/definitions/TrainingConfig"
    }
  },
  "definitions": {
    "ModelInfo": {
      "type": "object",
      "required": ["file", "checksum", "size_bytes"],
      "properties": {
        "file": {
          "type": "string",
          "description": "Filename of the model"
        },
        "checksum": {
          "type": "string",
          "pattern": "^sha256:[a-f0-9]{64}$",
          "description": "SHA-256 checksum of the file"
        },
        "size_bytes": {
          "type": "integer",
          "minimum": 0,
          "description": "File size in bytes"
        },
        "input_shape": {
          "type": "array",
          "items": { "type": "integer" },
          "description": "Input tensor shape"
        },
        "output_shape": {
          "type": "array",
          "items": { "type": "integer" },
          "description": "Output tensor shape"
        }
      }
    },
    "PerformanceMetrics": {
      "type": "object",
      "properties": {
        "backtest_sharpe": {
          "type": "number",
          "description": "Sharpe ratio on historical backtest"
        },
        "backtest_winrate": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Win rate on historical backtest"
        },
        "backtest_total_pnl": {
          "type": "number",
          "description": "Total PnL on historical backtest"
        },
        "validation_loss": {
          "type": "number",
          "description": "Validation loss from training"
        },
        "cvar_ppo_policy_loss": {
          "type": "number",
          "description": "Policy network loss"
        },
        "cvar_ppo_value_loss": {
          "type": "number",
          "description": "Value network loss"
        },
        "lstm_rmse": {
          "type": "number",
          "description": "LSTM prediction RMSE"
        }
      }
    },
    "TrainingConfig": {
      "type": "object",
      "properties": {
        "learning_rate": {
          "type": "number",
          "description": "Learning rate used"
        },
        "batch_size": {
          "type": "integer",
          "description": "Batch size used"
        },
        "epochs": {
          "type": "integer",
          "description": "Number of training epochs"
        },
        "optimizer": {
          "type": "string",
          "description": "Optimizer type (e.g., Adam)"
        }
      }
    }
  }
}
```

### Example Manifest

```json
{
  "version": "v49",
  "created_at": "2025-10-19T21:00:00Z",
  "training_duration_minutes": 180,
  "experience_count": 15247,
  "historical_bars": 6989,
  "models": {
    "cvar_ppo_policy": {
      "file": "cvar_ppo_policy.onnx",
      "checksum": "sha256:abc123def456789012345678901234567890123456789012345678901234567890",
      "size_bytes": 5242880,
      "input_shape": [1, 64],
      "output_shape": [1, 5]
    },
    "cvar_ppo_value": {
      "file": "cvar_ppo_value.onnx",
      "checksum": "sha256:def456789012345678901234567890123456789012345678901234567890abc123",
      "size_bytes": 3145728,
      "input_shape": [1, 64],
      "output_shape": [1, 1]
    },
    "cvar_ppo_cvar": {
      "file": "cvar_ppo_cvar.onnx",
      "checksum": "sha256:789012345678901234567890123456789012345678901234567890abc123def456",
      "size_bytes": 2097152,
      "input_shape": [1, 64],
      "output_shape": [1, 10]
    },
    "ucb_network": {
      "file": "ucb_network.onnx",
      "checksum": "sha256:345678901234567890123456789012345678901234567890abc123def456789012",
      "size_bytes": 4194304,
      "input_shape": [1, 32],
      "output_shape": [1, 17]
    },
    "lstm_predictor": {
      "file": "lstm_predictor.onnx",
      "checksum": "sha256:567890123456789012345678901234567890abc123def456789012345678901234",
      "size_bytes": 6291456,
      "input_shape": [1, 50, 11],
      "output_shape": [1, 1]
    }
  },
  "performance": {
    "backtest_sharpe": 1.85,
    "backtest_winrate": 0.68,
    "backtest_total_pnl": 12500.50,
    "validation_loss": 0.042,
    "cvar_ppo_policy_loss": 0.015,
    "cvar_ppo_value_loss": 0.023,
    "lstm_rmse": 2.34
  },
  "training_config": {
    "learning_rate": 0.0003,
    "batch_size": 256,
    "epochs": 50,
    "optimizer": "Adam"
  }
}
```

---

## 🗄️ Experience Database Schema

### SQLite Database: `experience.db`

#### Table: `experiences`

```sql
CREATE TABLE experiences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,                    -- ISO 8601 format: 2025-10-19T14:30:00Z
    symbol TEXT NOT NULL,                       -- "NQ" or "ES"
    strategy TEXT NOT NULL,                     -- "S2", "S3", "S6", etc.
    state_json TEXT NOT NULL,                   -- JSON array of state features
    action INTEGER NOT NULL,                    -- 0=Wait, 1=Long, 2=Short, 3=Close, 4=Reduce
    reward REAL NOT NULL,                       -- Calculated reward value
    next_state_json TEXT,                       -- JSON array (null if terminal)
    done INTEGER NOT NULL,                      -- 0=false, 1=true (position closed)
    brain_version TEXT NOT NULL,                -- "v49"
    market_regime TEXT,                         -- "Trending", "Ranging", "Volatile"
    pnl REAL,                                   -- Actual PnL in dollars
    confidence REAL,                            -- Decision confidence (0-1)
    ucb_value REAL,                             -- UCB value for strategy selection
    cvar_value REAL,                            -- CVaR risk estimate
    
    -- Metadata
    entry_price REAL,                           -- Entry price (if action is entry)
    exit_price REAL,                            -- Exit price (if action is exit)
    position_size INTEGER,                      -- Number of contracts
    hold_time_seconds INTEGER,                  -- Time position was held
    
    -- Context
    market_hour INTEGER,                        -- Hour of day (0-23)
    market_day_of_week INTEGER,                 -- Day of week (0=Monday, 6=Sunday)
    volatility REAL,                            -- Market volatility (ATR)
    volume_ratio REAL,                          -- Current volume / average volume
    
    -- Indexes for fast querying
    CONSTRAINT experiences_pk PRIMARY KEY (id)
);

CREATE INDEX idx_timestamp ON experiences(timestamp);
CREATE INDEX idx_symbol ON experiences(symbol);
CREATE INDEX idx_strategy ON experiences(strategy);
CREATE INDEX idx_brain_version ON experiences(brain_version);
CREATE INDEX idx_done ON experiences(done);
CREATE INDEX idx_composite ON experiences(timestamp, symbol, strategy);
```

#### Table: `metadata`

```sql
CREATE TABLE metadata (
    key TEXT PRIMARY KEY NOT NULL,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL                    -- ISO 8601 format
);

-- Predefined keys:
-- "current_brain_version": "v49"
-- "last_training_date": "2025-10-19T21:00:00Z"
-- "total_experiences": "15247"
-- "schema_version": "1.0"
```

#### Table: `training_runs`

```sql
CREATE TABLE training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    brain_version TEXT NOT NULL,
    experience_count INTEGER NOT NULL,
    historical_bars INTEGER NOT NULL,
    duration_minutes INTEGER,
    status TEXT NOT NULL,                       -- "running", "completed", "failed"
    error_message TEXT,
    performance_json TEXT,                      -- JSON of PerformanceMetrics
    
    CONSTRAINT training_runs_pk PRIMARY KEY (id)
);

CREATE INDEX idx_brain_version ON training_runs(brain_version);
CREATE INDEX idx_status ON training_runs(status);
```

### Example Experience Record

```json
{
  "id": 12345,
  "timestamp": "2025-10-19T14:30:00Z",
  "symbol": "NQ",
  "strategy": "S3",
  "state_json": "[0.45, 0.78, -0.23, 0.91, ...]",  // 64 features
  "action": 1,  // Long
  "reward": 2.5,
  "next_state_json": "[0.47, 0.76, -0.21, 0.89, ...]",
  "done": 0,  // Position still open
  "brain_version": "v49",
  "market_regime": "Trending",
  "pnl": 125.00,
  "confidence": 0.85,
  "ucb_value": 0.72,
  "cvar_value": 0.15,
  "entry_price": 16250.50,
  "exit_price": null,
  "position_size": 1,
  "hold_time_seconds": 180,
  "market_hour": 14,
  "market_day_of_week": 1,  // Tuesday
  "volatility": 25.5,
  "volume_ratio": 1.35
}
```

---

## 📡 Redis Notification Protocol

### Channel: `brain:updated`

**Published by**: Trainer (after successful brain publishing)  
**Subscribed by**: Live Bot

### Message Format

```json
{
  "event": "brain_updated",
  "version": "v49",
  "timestamp": "2025-10-19T21:00:00Z",
  "models_path": "/opt/models/v49",
  "active_path": "/opt/models/active",
  "manifest": {
    "experience_count": 15247,
    "historical_bars": 6989,
    "training_duration_minutes": 180,
    "performance": {
      "backtest_sharpe": 1.85,
      "backtest_winrate": 0.68
    }
  },
  "action": "hot_reload"
}
```

### Channel: `brain:rollback`

**Published by**: Admin/Operator (manual rollback)  
**Subscribed by**: Live Bot

```json
{
  "event": "brain_rollback",
  "version": "v48",
  "timestamp": "2025-10-19T22:00:00Z",
  "reason": "Performance degradation detected",
  "action": "hot_reload"
}
```

### Channel: `trainer:status`

**Published by**: Trainer (progress updates)  
**Subscribed by**: Monitoring/Dashboard

```json
{
  "event": "training_progress",
  "run_id": 123,
  "status": "running",
  "progress_percent": 45,
  "current_step": "cvar_training",
  "experiences_processed": 6800,
  "total_experiences": 15247,
  "eta_minutes": 90,
  "timestamp": "2025-10-19T19:30:00Z"
}
```

---

## 🔌 Interface Definitions

### IBrainLoader

```csharp
namespace QBot.Contracts.Interfaces;

/// <summary>
/// Loads brain bundles from disk
/// </summary>
public interface IBrainLoader
{
    /// <summary>
    /// Load complete brain bundle from directory
    /// </summary>
    /// <param name="path">Path to brain directory (e.g., /opt/models/active/)</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Loaded brain bundle</returns>
    Task<BrainBundle> LoadAsync(string path, CancellationToken ct = default);
    
    /// <summary>
    /// Get brain metadata without loading models
    /// </summary>
    /// <param name="path">Path to brain directory</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Brain metadata</returns>
    Task<BrainMetadata> GetMetadataAsync(string path, CancellationToken ct = default);
    
    /// <summary>
    /// Validate brain bundle integrity
    /// </summary>
    /// <param name="path">Path to brain directory</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>True if valid, false otherwise</returns>
    Task<bool> ValidateAsync(string path, CancellationToken ct = default);
}
```

### IExperienceStore

```csharp
namespace QBot.Contracts.Interfaces;

/// <summary>
/// Store and retrieve experiences from database
/// </summary>
public interface IExperienceStore
{
    /// <summary>
    /// Write single experience to database
    /// </summary>
    Task WriteExperienceAsync(Experience experience, CancellationToken ct = default);
    
    /// <summary>
    /// Write batch of experiences (more efficient)
    /// </summary>
    Task WriteBatchAsync(IEnumerable<Experience> experiences, CancellationToken ct = default);
    
    /// <summary>
    /// Read experiences within date range
    /// </summary>
    Task<List<Experience>> ReadExperiencesAsync(
        DateTime startDate, 
        DateTime endDate, 
        string? symbol = null,
        string? strategy = null,
        CancellationToken ct = default);
    
    /// <summary>
    /// Read experiences since last training
    /// </summary>
    Task<List<Experience>> ReadNewExperiencesAsync(
        string lastBrainVersion, 
        CancellationToken ct = default);
    
    /// <summary>
    /// Get statistics about stored experiences
    /// </summary>
    Task<ExperienceStatistics> GetStatisticsAsync(CancellationToken ct = default);
}
```

### IBrainPublisher

```csharp
namespace QBot.Contracts.Interfaces;

/// <summary>
/// Publishes trained brain bundles
/// </summary>
public interface IBrainPublisher
{
    /// <summary>
    /// Publish brain bundle to production
    /// </summary>
    /// <param name="bundle">Brain bundle to publish</param>
    /// <param name="ct">Cancellation token</param>
    /// <returns>Version identifier (e.g., "v49")</returns>
    Task<string> PublishAsync(BrainBundle bundle, CancellationToken ct = default);
    
    /// <summary>
    /// Rollback to previous brain version
    /// </summary>
    /// <param name="version">Version to rollback to (e.g., "v48")</param>
    /// <param name="ct">Cancellation token</param>
    Task RollbackAsync(string version, CancellationToken ct = default);
    
    /// <summary>
    /// List all available brain versions
    /// </summary>
    Task<List<string>> ListVersionsAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Delete old brain versions (keep last N)
    /// </summary>
    Task CleanupOldVersionsAsync(int keepCount = 5, CancellationToken ct = default);
}
```

### ITrainer

```csharp
namespace QBot.Contracts.Interfaces;

/// <summary>
/// Base interface for all trainers
/// </summary>
public interface ITrainer
{
    /// <summary>
    /// Trainer name (e.g., "CVaR-PPO", "Neural UCB")
    /// </summary>
    string Name { get; }
    
    /// <summary>
    /// Add single experience for training
    /// </summary>
    Task AddExperienceAsync(Experience experience, CancellationToken ct = default);
    
    /// <summary>
    /// Add batch of experiences
    /// </summary>
    Task AddExperiencesBatchAsync(IEnumerable<Experience> experiences, CancellationToken ct = default);
    
    /// <summary>
    /// Train on accumulated experiences
    /// </summary>
    /// <returns>Training result with metrics</returns>
    Task<TrainingResult> TrainAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Get trained model(s)
    /// </summary>
    Task<Dictionary<string, byte[]>> GetTrainedModelsAsync(CancellationToken ct = default);
    
    /// <summary>
    /// Reset trainer state
    /// </summary>
    Task ResetAsync(CancellationToken ct = default);
}
```

---

## 📊 Data Models

### BrainBundle

```csharp
namespace QBot.Contracts.Models;

public class BrainBundle
{
    public string Version { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public Dictionary<string, byte[]> Models { get; set; } = new();
    public BrainManifest Manifest { get; set; } = new();
    public Dictionary<string, string> Config { get; set; } = new();
}
```

### BrainManifest

```csharp
public class BrainManifest
{
    public string Version { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public int TrainingDurationMinutes { get; set; }
    public int ExperienceCount { get; set; }
    public int HistoricalBars { get; set; }
    public Dictionary<string, ModelInfo> Models { get; set; } = new();
    public PerformanceMetrics? Performance { get; set; }
    public TrainingConfig? TrainingConfig { get; set; }
}

public class ModelInfo
{
    public string File { get; set; } = string.Empty;
    public string Checksum { get; set; } = string.Empty;
    public long SizeBytes { get; set; }
    public int[]? InputShape { get; set; }
    public int[]? OutputShape { get; set; }
}

public class PerformanceMetrics
{
    public double? BacktestSharpe { get; set; }
    public double? BacktestWinrate { get; set; }
    public double? BacktestTotalPnl { get; set; }
    public double? ValidationLoss { get; set; }
    public double? CVaRPPOPolicyLoss { get; set; }
    public double? CVaRPPOValueLoss { get; set; }
    public double? LstmRmse { get; set; }
}

public class TrainingConfig
{
    public double? LearningRate { get; set; }
    public int? BatchSize { get; set; }
    public int? Epochs { get; set; }
    public string? Optimizer { get; set; }
}
```

### Experience

```csharp
public class Experience
{
    public long Id { get; set; }
    public DateTime Timestamp { get; set; }
    public string Symbol { get; set; } = string.Empty;
    public string Strategy { get; set; } = string.Empty;
    public List<double> State { get; set; } = new();
    public int Action { get; set; }
    public double Reward { get; set; }
    public List<double>? NextState { get; set; }
    public bool Done { get; set; }
    public string BrainVersion { get; set; } = string.Empty;
    public string? MarketRegime { get; set; }
    public double? Pnl { get; set; }
    public double? Confidence { get; set; }
    public double? UcbValue { get; set; }
    public double? CVaRValue { get; set; }
    
    // Context
    public double? EntryPrice { get; set; }
    public double? ExitPrice { get; set; }
    public int? PositionSize { get; set; }
    public int? HoldTimeSeconds { get; set; }
    public int? MarketHour { get; set; }
    public int? MarketDayOfWeek { get; set; }
    public double? Volatility { get; set; }
    public double? VolumeRatio { get; set; }
}
```

### TrainingResult

```csharp
public class TrainingResult
{
    public int Episode { get; set; }
    public bool Success { get; set; }
    public string? ErrorMessage { get; set; }
    public DateTime StartTime { get; set; }
    public DateTime EndTime { get; set; }
    public int DurationMinutes => (int)(EndTime - StartTime).TotalMinutes;
    
    // Metrics
    public double TotalLoss { get; set; }
    public double AverageReward { get; set; }
    public Dictionary<string, double> AdditionalMetrics { get; set; } = new();
}
```

### ExperienceStatistics

```csharp
public class ExperienceStatistics
{
    public long TotalCount { get; set; }
    public DateTime OldestTimestamp { get; set; }
    public DateTime NewestTimestamp { get; set; }
    public Dictionary<string, long> CountBySymbol { get; set; } = new();
    public Dictionary<string, long> CountByStrategy { get; set; } = new();
    public Dictionary<string, long> CountByBrainVersion { get; set; } = new();
    public long DatabaseSizeBytes { get; set; }
}
```

---

## 🔐 Security Considerations

### Checksum Validation

All model files must be validated using SHA-256 checksums:

```csharp
public static string CalculateChecksum(byte[] data)
{
    using var sha256 = SHA256.Create();
    var hash = sha256.ComputeHash(data);
    return "sha256:" + BitConverter.ToString(hash).Replace("-", "").ToLowerInvariant();
}

public static bool ValidateChecksum(byte[] data, string expectedChecksum)
{
    var actualChecksum = CalculateChecksum(data);
    return actualChecksum == expectedChecksum;
}
```

### Atomic Publishing

Brain publishing must be atomic to prevent partial updates:

```csharp
public async Task<string> PublishAsync(BrainBundle bundle)
{
    // 1. Write to temporary directory
    var tempDir = $"/opt/models/{bundle.Version}.tmp";
    await WriteBundleAsync(tempDir, bundle);
    
    // 2. Validate integrity
    if (!await ValidateAsync(tempDir))
    {
        Directory.Delete(tempDir, true);
        throw new InvalidOperationException("Brain validation failed");
    }
    
    // 3. Rename to final directory (atomic on POSIX)
    var finalDir = $"/opt/models/{bundle.Version}";
    Directory.Move(tempDir, finalDir);
    
    // 4. Update symlink (atomic)
    var activeLink = "/opt/models/active";
    var tempLink = "/opt/models/active.tmp";
    
    if (File.Exists(tempLink))
        File.Delete(tempLink);
    
    // Create symlink
    CreateSymbolicLink(tempLink, finalDir);
    
    // Atomic rename
    File.Move(tempLink, activeLink, overwrite: true);
    
    return bundle.Version;
}
```

### Access Control

```
/opt/models/
├── active/       (read: live-bot, write: trainer)
├── v*/           (read: live-bot, write: trainer)
└── ...

/opt/data/
├── experience.db (read: trainer, write: live-bot)
└── ...
```

---

## ⚡ Performance Specifications

### Live Bot Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Startup Time | < 3 seconds | < 5 seconds |
| Brain Load Time | < 2 seconds | < 3 seconds |
| Decision Latency | < 5ms | < 10ms |
| Experience Write Latency | < 1ms | < 5ms |
| Memory Usage | < 1.5GB | < 2GB |
| CPU Usage (idle) | < 10% | < 20% |
| CPU Usage (active) | < 25% | < 30% |

### Trainer Requirements

| Metric | Target | Maximum |
|--------|--------|---------|
| Experience Read (10k) | < 5 seconds | < 10 seconds |
| Historical Load (6989 bars) | < 3 seconds | < 5 seconds |
| CVaR-PPO Training (10k exp) | < 20 minutes | < 30 minutes |
| Neural UCB Training | < 15 minutes | < 20 minutes |
| LSTM Training | < 10 minutes | < 15 minutes |
| Full Training Cycle | < 3 hours | < 4 hours |
| Brain Packaging | < 3 seconds | < 5 seconds |
| Brain Publishing | < 2 seconds | < 3 seconds |

### Database Performance

| Operation | Target | Maximum |
|-----------|--------|---------|
| Write Single Experience | < 1ms | < 5ms |
| Write Batch (100 exp) | < 10ms | < 50ms |
| Read 10k Experiences | < 5 seconds | < 10 seconds |
| Read by Date Range (1 day) | < 1 second | < 2 seconds |
| Database Size (1 month) | < 500MB | < 1GB |

---

## 🔍 Monitoring & Metrics

### Live Bot Metrics

```csharp
public class LiveBotMetrics
{
    // Decision metrics
    public long TotalDecisions { get; set; }
    public double AverageDecisionLatencyMs { get; set; }
    public double P95DecisionLatencyMs { get; set; }
    public double P99DecisionLatencyMs { get; set; }
    
    // Experience logging
    public long ExperiencesLogged { get; set; }
    public long ExperienceWriteErrors { get; set; }
    
    // Brain loading
    public string CurrentBrainVersion { get; set; } = string.Empty;
    public DateTime BrainLoadedAt { get; set; }
    public int HotReloadCount { get; set; }
    
    // Resource usage
    public long MemoryUsageBytes { get; set; }
    public double CpuUsagePercent { get; set; }
}
```

### Trainer Metrics

```csharp
public class TrainerMetrics
{
    // Training run
    public int RunId { get; set; }
    public DateTime StartedAt { get; set; }
    public DateTime? CompletedAt { get; set; }
    public string Status { get; set; } = "running";
    
    // Data
    public int ExperiencesLoaded { get; set; }
    public int HistoricalBarsLoaded { get; set; }
    
    // Training progress
    public Dictionary<string, TrainerProgress> Trainers { get; set; } = new();
    
    // Results
    public string? ProducedBrainVersion { get; set; }
    public PerformanceMetrics? Performance { get; set; }
}

public class TrainerProgress
{
    public string Name { get; set; } = string.Empty;
    public string Status { get; set; } = "pending";
    public int Epoch { get; set; }
    public int TotalEpochs { get; set; }
    public double Loss { get; set; }
    public double? ValidationLoss { get; set; }
}
```

---

## 📖 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-19 | Initial specification |

---

**Next Steps**: Use these specifications to implement the Bot/Trainer split according to the Implementation Checklist.
