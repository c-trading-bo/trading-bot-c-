# ✅ Bot/Trainer Split - Implementation Checklist

**Purpose**: Detailed task-by-task checklist for implementing the bot/trainer split  
**Timeline**: 4-6 weeks (160-240 hours)  
**Status**: 📋 Planning Phase

---

## 📝 How to Use This Checklist

- [ ] Mark tasks as complete with `[x]`
- [ ] Add notes under each task as you work
- [ ] Update time estimates based on actual time spent
- [ ] Track blockers in the "Blockers" section at the end

---

## Phase 1: Project Setup (Day 1 - 8 hours)

### 1.1 Create QBot.Contracts Project (1 hour)

- [ ] Create directory: `src/QBot.Contracts/`
- [ ] Create `QBot.Contracts.csproj`
  ```xml
  <Project Sdk="Microsoft.NET.Sdk">
    <PropertyGroup>
      <TargetFramework>net8.0</TargetFramework>
      <Nullable>enable</Nullable>
    </PropertyGroup>
  </Project>
  ```
- [ ] Add to solution: `dotnet sln add src/QBot.Contracts/QBot.Contracts.csproj`
- [ ] Create folder structure:
  ```
  QBot.Contracts/
  ├── Interfaces/
  ├── Models/
  └── Constants/
  ```

**Deliverables**: 
- QBot.Contracts project compiles
- Appears in solution file

---

### 1.2 Create Shared Interfaces (1 hour)

- [ ] Create `Interfaces/IBrainLoader.cs`
  ```csharp
  public interface IBrainLoader
  {
      Task<BrainBundle> LoadAsync(string path, CancellationToken ct = default);
      Task<BrainMetadata> GetMetadataAsync(string path, CancellationToken ct = default);
  }
  ```

- [ ] Create `Interfaces/IExperienceStore.cs`
  ```csharp
  public interface IExperienceStore
  {
      Task WriteExperienceAsync(Experience exp, CancellationToken ct = default);
      Task<List<Experience>> ReadExperiencesAsync(DateTime start, DateTime end, CancellationToken ct = default);
  }
  ```

- [ ] Create `Interfaces/IBrainPublisher.cs`
  ```csharp
  public interface IBrainPublisher
  {
      Task<string> PublishAsync(BrainBundle bundle, CancellationToken ct = default);
      Task RollbackAsync(string version, CancellationToken ct = default);
  }
  ```

**Deliverables**: 
- 3 interface files created
- All interfaces compile without errors

---

### 1.3 Create Shared Models (1 hour)

- [ ] Create `Models/BrainBundle.cs`
  ```csharp
  public class BrainBundle
  {
      public string Version { get; set; } = string.Empty;
      public Dictionary<string, byte[]> Models { get; set; } = new();
      public BrainManifest Manifest { get; set; } = new();
  }
  ```

- [ ] Create `Models/BrainManifest.cs`
  ```csharp
  public class BrainManifest
  {
      public string Version { get; set; } = string.Empty;
      public DateTime CreatedAt { get; set; }
      public int TrainingDurationMinutes { get; set; }
      public Dictionary<string, ModelInfo> Models { get; set; } = new();
      public PerformanceMetrics? Performance { get; set; }
  }
  
  public class ModelInfo
  {
      public string File { get; set; } = string.Empty;
      public string Checksum { get; set; } = string.Empty;
      public long SizeBytes { get; set; }
  }
  ```

- [ ] Create `Models/Experience.cs`
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
  }
  ```

**Deliverables**: 
- 3 model files created
- All models serialize/deserialize correctly

---

### 1.4 Create QBot.Trainer Project (2 hours)

- [ ] Create directory: `src/QBot.Trainer/`
- [ ] Create `QBot.Trainer.csproj`
  ```xml
  <Project Sdk="Microsoft.NET.Sdk">
    <PropertyGroup>
      <OutputType>Exe</OutputType>
      <TargetFramework>net8.0</TargetFramework>
      <Nullable>enable</Nullable>
    </PropertyGroup>
    
    <ItemGroup>
      <PackageReference Include="Microsoft.Extensions.Hosting" Version="9.0.0" />
      <PackageReference Include="Microsoft.Extensions.Logging" Version="9.0.0" />
      <PackageReference Include="Microsoft.Extensions.Configuration" Version="9.0.0" />
      <PackageReference Include="Microsoft.Data.Sqlite" Version="9.0.0" />
      <PackageReference Include="StackExchange.Redis" Version="2.8.16" />
    </ItemGroup>
    
    <ItemGroup>
      <ProjectReference Include="../QBot.Contracts/QBot.Contracts.csproj" />
      <ProjectReference Include="../BotCore/BotCore.csproj" />
      <ProjectReference Include="../RLAgent/RLAgent.csproj" />
      <ProjectReference Include="../ML/ML.csproj" />
      <ProjectReference Include="../IntelligenceStack/IntelligenceStack.csproj" />
    </ItemGroup>
  </Project>
  ```

- [ ] Add to solution: `dotnet sln add src/QBot.Trainer/QBot.Trainer.csproj`
- [ ] Create folder structure:
  ```
  QBot.Trainer/
  ├── Program.cs
  ├── Infrastructure/
  ├── Trainers/
  ├── Services/
  └── appsettings.json
  ```

**Deliverables**: 
- QBot.Trainer project compiles
- All dependencies resolve correctly

---

### 1.5 Create Trainer Program.cs (2 hours)

- [ ] Create basic `Program.cs` with DI setup
  ```csharp
  var builder = Host.CreateApplicationBuilder(args);
  
  // Configure logging
  builder.Logging.AddConsole();
  builder.Logging.SetMinimumLevel(LogLevel.Information);
  
  // Register services
  builder.Services.AddSingleton<IExperienceStore, ExperienceStore>();
  builder.Services.AddSingleton<IBrainLoader, BrainLoader>();
  builder.Services.AddSingleton<IBrainPublisher, BrainPublisher>();
  
  // Register trainers
  builder.Services.AddSingleton<CVaRTrainer>();
  builder.Services.AddSingleton<UcbTrainer>();
  
  var app = builder.Build();
  
  var logger = app.Services.GetRequiredService<ILogger<Program>>();
  logger.LogInformation("🎓 QBot Trainer starting...");
  
  await app.RunAsync();
  ```

- [ ] Create `appsettings.json`
  ```json
  {
    "Logging": {
      "LogLevel": {
        "Default": "Information"
      }
    },
    "Trainer": {
      "ModelPath": "/opt/models",
      "ExperienceDbPath": "/opt/data/experience.db",
      "HistoricalDataPath": "/opt/data/historical_cache",
      "RedisConnectionString": "localhost:6379"
    }
  }
  ```

**Deliverables**: 
- Trainer program runs without errors
- Shows "QBot Trainer starting..." in logs

---

### 1.6 Verify Build and References (1 hour)

- [ ] Build entire solution: `dotnet build`
- [ ] Verify no errors
- [ ] Verify no circular dependencies
- [ ] Run UnifiedOrchestrator (Live Bot) - should still work unchanged
- [ ] Run Trainer - should start and exit cleanly

**Deliverables**: 
- Clean build of entire solution
- Both programs run without errors

---

## Phase 2: Infrastructure Layer (Days 2-4 - 24 hours)

### 2.1 Experience Database Schema (2 hours)

- [ ] Create `Infrastructure/DatabaseSchema.cs`
  ```csharp
  public class DatabaseSchema
  {
      public const string CreateExperiencesTable = @"
          CREATE TABLE IF NOT EXISTS experiences (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT NOT NULL,
              symbol TEXT NOT NULL,
              strategy TEXT NOT NULL,
              state_json TEXT NOT NULL,
              action INTEGER NOT NULL,
              reward REAL NOT NULL,
              next_state_json TEXT,
              done INTEGER NOT NULL,
              brain_version TEXT NOT NULL,
              market_regime TEXT,
              pnl REAL
          );
          
          CREATE INDEX IF NOT EXISTS idx_timestamp ON experiences(timestamp);
          CREATE INDEX IF NOT EXISTS idx_symbol ON experiences(symbol);
          CREATE INDEX IF NOT EXISTS idx_brain_version ON experiences(brain_version);
      ";
      
      public const string CreateMetadataTable = @"
          CREATE TABLE IF NOT EXISTS metadata (
              key TEXT PRIMARY KEY,
              value TEXT NOT NULL,
              updated_at TEXT NOT NULL
          );
      ";
  }
  ```

- [ ] Create `Infrastructure/ExperienceStore.cs` (implements `IExperienceStore`)
- [ ] Implement `WriteExperienceAsync()` - batch writes for performance
- [ ] Implement `ReadExperiencesAsync()` - paginated reads
- [ ] Add connection pooling
- [ ] Add error handling (disk full, write failures)

**Deliverables**: 
- experience.db can be created
- Can write 1000 experiences/sec
- Can read experiences by date range

---

### 2.2 Experience Reader (Trainer) (4 hours)

- [ ] Create `Infrastructure/ExperienceReader.cs`
- [ ] Implement `LoadAllExperiences()` - load all from DB
- [ ] Implement `LoadExperiencesSince()` - incremental loading
- [ ] Implement `GetStatistics()` - count by strategy, symbol
- [ ] Add progress reporting (log every 10k experiences)
- [ ] Add memory-efficient streaming (don't load all at once)

**Test**:
- [ ] Load 100k experiences in < 10 seconds
- [ ] Memory usage < 500MB for 100k experiences

**Deliverables**: 
- Can read experiences from DB
- Performance acceptable

---

### 2.3 Historical Data Loader (4 hours)

- [ ] Create `Infrastructure/HistoricalDataLoader.cs`
- [ ] Integrate with existing `HistoricalDataSeedService`
- [ ] Implement `LoadHistoricalBars()` - load from cache
- [ ] Support 90-day rolling window
- [ ] Add parallel loading (one thread per symbol)
- [ ] Add caching (don't reload same data)

**Test**:
- [ ] Load 6989 bars in < 5 seconds
- [ ] Memory usage < 1GB

**Deliverables**: 
- Can load historical data efficiently
- Same data as Live Bot uses

---

### 2.4 Brain Loader (6 hours)

- [ ] Create `Infrastructure/BrainLoader.cs` (implements `IBrainLoader`)
- [ ] Implement `LoadAsync()` - load brain from directory
  - [ ] Read `manifest.json`
  - [ ] Validate checksums (SHA-256)
  - [ ] Load ONNX models into memory
  - [ ] Validate model compatibility
- [ ] Implement `GetMetadataAsync()` - read manifest only
- [ ] Add error handling (corrupt files, missing models)
- [ ] Add caching (reuse loaded models)

**Test**:
- [ ] Load brain bundle in < 2 seconds
- [ ] Detect corrupted files
- [ ] Fail gracefully if model missing

**Deliverables**: 
- Can load brain from /opt/models/active/
- Validates integrity before loading

---

### 2.5 Brain Packager (4 hours)

- [ ] Create `Infrastructure/BrainPackager.cs`
- [ ] Implement `PackageBrainAsync()` - create brain bundle
  - [ ] Serialize ONNX models to byte arrays
  - [ ] Generate `manifest.json` with metadata
  - [ ] Calculate SHA-256 checksums
  - [ ] Add version number (increment from previous)
- [ ] Implement `ValidateBundleAsync()` - check integrity
- [ ] Add compression (optional, for network transfer)

**Test**:
- [ ] Package brain in < 5 seconds
- [ ] Checksums match files
- [ ] Bundle can be unpacked successfully

**Deliverables**: 
- Can create valid brain bundle
- Manifest is correct

---

### 2.6 Brain Publisher (4 hours)

- [ ] Create `Infrastructure/BrainPublisher.cs` (implements `IBrainPublisher`)
- [ ] Implement `PublishAsync()` - atomic publishing
  - [ ] Write to `/opt/models/v{N}/`
  - [ ] Validate all files written
  - [ ] Update symlink `/opt/models/active/` atomically
  - [ ] Keep last 5 versions for rollback
- [ ] Implement `RollbackAsync()` - revert to previous version
- [ ] Add locking (prevent concurrent publishes)
- [ ] Add validation (health check before publishing)

**Test**:
- [ ] Atomic publishing (no partial states)
- [ ] Rollback works correctly
- [ ] Can't publish corrupted brain

**Deliverables**: 
- Can publish brain atomically
- Rollback capability works

---

### 2.7 Redis Notifier (2 hours)

- [ ] Create `Infrastructure/RedisNotifier.cs`
- [ ] Implement `NotifyBrainUpdatedAsync()`
  - [ ] Publish to Redis channel: `brain:updated`
  - [ ] Include version, timestamp, manifest
- [ ] Add retry logic (network failures)
- [ ] Add connection pooling

**Test**:
- [ ] Notification received by Live Bot
- [ ] Works even if Redis temporarily down

**Deliverables**: 
- Can notify Live Bot of brain updates
- Reliable delivery

---

## Phase 3: Training Components (Days 5-8 - 32 hours)

### 3.1 CVaR-PPO Trainer (8 hours)

- [ ] Create `Trainers/CVaRTrainer.cs`
- [ ] Load current CVaR-PPO model from brain
- [ ] Load experiences from `ExperienceReader`
- [ ] Reuse existing `CVaRPPO.TrainAsync()` logic
- [ ] Add training metrics logging
  - [ ] Policy loss
  - [ ] Value loss
  - [ ] CVaR loss
  - [ ] Average reward
- [ ] Add convergence checking (early stopping)
- [ ] Save trained model to staging directory

**Test**:
- [ ] Trains on 10k experiences in < 30 minutes
- [ ] Loss decreases over epochs
- [ ] Model improves from baseline

**Deliverables**: 
- CVaR-PPO training works offline
- Produces improved model

---

### 3.2 Neural UCB Trainer (8 hours)

- [ ] Create `Trainers/UcbTrainer.cs`
- [ ] Load current Neural UCB model from brain
- [ ] Batch update with all experiences
- [ ] Retrain neural network weights
  - [ ] Context embedding
  - [ ] Reward prediction
  - [ ] Uncertainty estimation
- [ ] Export to ONNX format
- [ ] Add training metrics logging

**Test**:
- [ ] Trains on 10k experiences in < 20 minutes
- [ ] Network converges
- [ ] Strategy selection probabilities reasonable

**Deliverables**: 
- Neural UCB training works offline
- Produces improved strategy selector

---

### 3.3 LSTM Trainer (6 hours)

- [ ] Create `Trainers/LstmTrainer.cs`
- [ ] Load current LSTM model from brain
- [ ] Train on historical price sequences
- [ ] Add sequence padding for variable lengths
- [ ] Export to ONNX format
- [ ] Add metrics: RMSE, MAE, directional accuracy

**Test**:
- [ ] Trains on 6989 bars in < 15 minutes
- [ ] Prediction accuracy > baseline

**Deliverables**: 
- LSTM training works
- Improves price prediction

---

### 3.4 SAC Trainer (5 hours)

- [ ] Create `Trainers/SacTrainer.cs`
- [ ] Implement Soft Actor-Critic training loop
- [ ] Load experiences from DB
- [ ] Train actor and critic networks
- [ ] Add entropy regularization
- [ ] Export to ONNX

**Test**:
- [ ] Trains on 10k experiences in < 30 minutes

**Deliverables**: 
- SAC training works

---

### 3.5 Meta-Learner Trainer (5 hours)

- [ ] Create `Trainers/MetaTrainer.cs`
- [ ] Implement MAML training loop
- [ ] Support few-shot learning
- [ ] Train on multiple tasks (strategies)
- [ ] Export meta-initialized model

**Test**:
- [ ] Adapts to new strategies quickly

**Deliverables**: 
- Meta-learning training works

---

## Phase 4: Historical Replay Migration (Days 9-11 - 24 hours)

### 4.1 Move EnhancedBacktestLearningService (8 hours)

- [ ] Copy `EnhancedBacktestLearningService.cs` to `QBot.Trainer/Services/`
- [ ] Remove from `UnifiedOrchestrator` project
- [ ] Update namespace
- [ ] Refactor constructor:
  - [ ] Remove live `UnifiedTradingBrain` dependency
  - [ ] Add `BrainLoader` dependency
  - [ ] Add `ExperienceWriter` to feed experiences to DB
- [ ] Update DI registration in Trainer

**Deliverables**: 
- Service compiles in Trainer project
- No longer in Live Bot

---

### 4.2 Refactor to Use Loaded Brain (8 hours)

- [ ] Change brain initialization:
  ```csharp
  // BEFORE
  private readonly UnifiedTradingBrain _unifiedBrain;
  
  // AFTER
  private UnifiedTradingBrain? _brain;
  
  protected override async Task ExecuteAsync(CancellationToken ct)
  {
      var brainBundle = await _brainLoader.LoadAsync("/opt/models/active/");
      _brain = await BrainFactory.CreateFromBundle(brainBundle);
      ...
  }
  ```
- [ ] Update all references to use loaded brain
- [ ] Ensure decisions are identical to before

**Test**:
- [ ] Historical replay produces same decisions as before

**Deliverables**: 
- Historical replay works with loaded brain
- Results match old system

---

### 4.3 Integrate with Trainers (8 hours)

- [ ] Feed historical experiences to all trainers:
  ```csharp
  foreach (var experience in historicalExperiences)
  {
      await _cvarTrainer.AddExperienceAsync(experience);
      await _ucbTrainer.AddExperienceAsync(experience);
      await _sacTrainer.AddExperienceAsync(experience);
  }
  
  await _cvarTrainer.TrainAsync();
  await _ucbTrainer.TrainAsync();
  await _sacTrainer.TrainAsync();
  ```
- [ ] Add progress reporting (log every 1000 bars)
- [ ] Optimize performance (parallel processing)

**Test**:
- [ ] Full 90-day replay completes in < 4 hours
- [ ] All trainers receive experiences

**Deliverables**: 
- Historical replay feeds all trainers
- Training completes successfully

---

## Phase 5: Live Bot Modifications (Days 12-14 - 24 hours)

### 5.1 Add Runtime Mode Configuration (4 hours)

- [ ] Update `appsettings.json`:
  ```json
  {
    "RLConfiguration": {
      "RuntimeMode": "InferenceOnly",
      "ModelPath": "/opt/models/active/",
      "ExperienceDbPath": "/opt/data/experience.db",
      "EnableHotReload": true
    }
  }
  ```
- [ ] Add configuration class:
  ```csharp
  public class RLConfiguration
  {
      public RlRuntimeMode RuntimeMode { get; set; }
      public string ModelPath { get; set; } = "/opt/models/active/";
      public string ExperienceDbPath { get; set; } = "/opt/data/experience.db";
      public bool EnableHotReload { get; set; } = true;
  }
  ```
- [ ] Register in DI: `services.Configure<RLConfiguration>(config.GetSection("RLConfiguration"))`
- [ ] Propagate to all RL components (CVaR-PPO, SAC, Meta)

**Deliverables**: 
- Configuration loads correctly
- Runtime mode set to InferenceOnly

---

### 5.2 Implement Brain Loader in Live Bot (8 hours)

- [ ] Copy `BrainLoader.cs` to `UnifiedOrchestrator/Infrastructure/`
- [ ] Add to DI: `services.AddSingleton<IBrainLoader, BrainLoader>()`
- [ ] Update `Program.cs` startup:
  ```csharp
  var brainLoader = app.Services.GetRequiredService<IBrainLoader>();
  logger.LogInformation("Loading brain from {Path}", config["RLConfiguration:ModelPath"]);
  
  var brainBundle = await brainLoader.LoadAsync(config["RLConfiguration:ModelPath"]);
  logger.LogInformation("Loaded brain version {Version}", brainBundle.Version);
  ```
- [ ] Update `UnifiedTradingBrain` to accept pre-loaded models:
  ```csharp
  public UnifiedTradingBrain(
      ...,
      CVaRPPO cvarPPO,  // Pre-loaded from brain bundle
      ...
  )
  ```
- [ ] Add brain version logging in decision logs

**Test**:
- [ ] Bot starts with loaded brain
- [ ] All models loaded successfully
- [ ] Version appears in logs

**Deliverables**: 
- Live Bot loads brain at startup
- Version tracking works

---

### 5.3 Disable Training Services (2 hours)

- [ ] Update `Program.cs`:
  ```csharp
  // Conditional registration
  if (config["RLConfiguration:RuntimeMode"] == "Training")
  {
      services.AddHostedService<EnhancedBacktestLearningService>();
  }
  else
  {
      logger.LogInformation("Training disabled (InferenceOnly mode)");
  }
  ```
- [ ] Verify service doesn't run in production
- [ ] Update health checks to not expect training service

**Test**:
- [ ] EnhancedBacktestLearningService not registered in InferenceOnly mode
- [ ] Bot runs without it

**Deliverables**: 
- Training service disabled in Live Bot
- No performance impact

---

### 5.4 Add Experience Logging (8 hours)

- [ ] Copy `ExperienceStore.cs` to `UnifiedOrchestrator/Infrastructure/`
- [ ] Add to DI: `services.AddSingleton<IExperienceStore, ExperienceStore>()`
- [ ] Update `UnifiedTradingBrain.LearnFromResultAsync()`:
  ```csharp
  public async Task LearnFromResultAsync(...)
  {
      // Existing lightweight learning stays
      await _strategySelector.UpdateArmAsync(...);
      
      // NEW: Log to experience DB
      if (_experienceStore != null)
      {
          await _experienceStore.WriteExperienceAsync(new Experience
          {
              Timestamp = DateTime.UtcNow,
              Symbol = symbol,
              Strategy = strategy,
              State = _lastCVaRState?.ToList() ?? new(),
              Action = _lastCVaRAction,
              Reward = (double)CalculateReward(pnl, wasCorrect, holdTime),
              NextState = null,  // Will be filled next trade
              Done = true,
              BrainVersion = _currentBrainVersion
          });
      }
  }
  ```
- [ ] Add batching (write every 100 experiences)
- [ ] Add error handling (disk full)

**Test**:
- [ ] All decisions logged to experience.db
- [ ] Performance impact < 1ms per decision
- [ ] Handles disk full gracefully

**Deliverables**: 
- Experience logging works
- No performance degradation

---

### 5.5 Implement Hot-Reload (2 hours)

- [ ] Create `Infrastructure/RedisListener.cs`:
  ```csharp
  public class RedisListener : BackgroundService
  {
      protected override async Task ExecuteAsync(CancellationToken ct)
      {
          await foreach (var message in _redis.SubscribeAsync("brain:updated", ct))
          {
              _logger.LogInformation("New brain version available: {Version}", message.Version);
              // Trigger hot-reload
              await _brainLoader.LoadAsync("/opt/models/active/");
              _logger.LogInformation("Brain hot-reloaded successfully");
          }
      }
  }
  ```
- [ ] Add to DI: `services.AddHostedService<RedisListener>()`
- [ ] Test hot-reload doesn't interrupt trading

**Test**:
- [ ] Receives Redis notification
- [ ] Loads new brain without restart
- [ ] No trading interruption

**Deliverables**: 
- Hot-reload works
- Zero downtime

---

## Phase 6: End-to-End Testing (Days 15-18 - 32 hours)

### 6.1 Live Bot Testing (16 hours)

#### Startup Testing (2 hours)
- [ ] Test cold start with brain loading
  - [ ] Measure startup time (should be < 5 seconds)
  - [ ] Verify all 17 components initialized
  - [ ] Check brain version in logs
- [ ] Test with missing brain (should fail gracefully)
- [ ] Test with corrupted brain (should detect and fail)

#### Decision Testing (4 hours)
- [ ] Run in DRY_RUN mode (no real orders)
- [ ] Compare decisions with previous system
  - [ ] Same inputs should produce same outputs
  - [ ] Decision latency should be < 10ms
- [ ] Verify no training calls in logs
- [ ] Check experience logging works

#### Stress Testing (6 hours)
- [ ] Run for full 6-hour market session
- [ ] Monitor memory usage (should be stable)
- [ ] Monitor CPU usage (should be < 30%)
- [ ] Check for memory leaks
- [ ] Verify experience.db grows as expected

#### Failure Testing (4 hours)
- [ ] Test Redis connection failure
- [ ] Test experience DB write failure
- [ ] Test brain loading failure at startup
- [ ] Test hot-reload with corrupted brain
- [ ] Verify graceful degradation

**Test Checklist**:
```
✓ Startup time < 5 seconds
✓ Decision latency < 10ms
✓ All 17 components working
✓ experience.db receives all decisions
✓ Zero training calls
✓ Memory stable (< 2GB)
✓ CPU usage < 30%
✓ Runs full session without crash
✓ Handles failures gracefully
```

---

### 6.2 Trainer Testing (16 hours)

#### Startup Testing (2 hours)
- [ ] Test with sample experience.db
- [ ] Test with no experiences (should handle gracefully)
- [ ] Test with corrupted brain
- [ ] Verify historical data loading

#### Training Testing (8 hours)
- [ ] Run full training cycle with 10k experiences
  - [ ] Measure training duration (should be < 4 hours)
  - [ ] Verify all trainers run
  - [ ] Check training metrics improve
- [ ] Test CVaR-PPO training
  - [ ] Loss decreases
  - [ ] Reward increases
- [ ] Test Neural UCB training
  - [ ] Network converges
  - [ ] Strategy selection improves
- [ ] Test LSTM training
  - [ ] Prediction accuracy improves
- [ ] Test with 90-day historical replay
  - [ ] Completes in reasonable time
  - [ ] All 6989 bars processed

#### Publishing Testing (4 hours)
- [ ] Test brain bundle creation
  - [ ] Manifest correct
  - [ ] Checksums match
  - [ ] All models included
- [ ] Test publishing to /opt/models/
  - [ ] Atomic symlink update works
  - [ ] Old versions preserved
  - [ ] Version numbering correct
- [ ] Test Redis notification
  - [ ] Live Bot receives notification
  - [ ] Hot-reload triggered
- [ ] Test rollback
  - [ ] Can revert to previous version
  - [ ] Live Bot loads old brain

#### Failure Testing (2 hours)
- [ ] Test training failure (should rollback)
- [ ] Test publish failure (should not corrupt active brain)
- [ ] Test Redis failure (should complete anyway)
- [ ] Test disk full (should fail gracefully)

**Test Checklist**:
```
✓ Loads brain from /opt/models/active/
✓ Reads all experiences from DB
✓ Loads 6989 historical bars
✓ Completes training in < 4 hours
✓ Produces valid brain bundle
✓ Manifest checksums correct
✓ Atomic publishing works
✓ Redis notification works
✓ Live Bot hot-reloads
✓ Rollback works
```

---

### 6.3 Integration Testing (8 hours)

- [ ] Test Day 1 → Day 2 flow
  - [ ] Day 1: Live Bot logs experiences
  - [ ] Evening: Trainer runs, produces new brain
  - [ ] Day 2: Live Bot loads new brain
  - [ ] Verify decisions use new brain
- [ ] Test version incrementing
  - [ ] v48 → v49 → v50
  - [ ] All versions preserved
- [ ] Test side-by-side comparison
  - [ ] Old system vs new split system
  - [ ] Same inputs, same outputs
- [ ] Test multi-machine deployment
  - [ ] Live Bot on Machine A
  - [ ] Trainer on Machine B
  - [ ] Shared /opt/models/ via NFS
- [ ] Test rollback scenario
  - [ ] New brain performs poorly
  - [ ] Rollback to previous version
  - [ ] Live Bot picks up old brain

**Test Checklist**:
```
✓ Day 2 decisions match Day 1 brain expectations
✓ Brain versions increment correctly
✓ No file corruption observed
✓ Redis notifications reliable
✓ Rollback works
✓ Both programs run on same machine
✓ Both programs run on different machines
```

---

## Phase 7: Documentation & Deployment (Days 19-20 - 16 hours)

### 7.1 Deployment Guide (4 hours)

- [ ] Write deployment guide covering:
  - [ ] Prerequisites (Redis, disk space)
  - [ ] Directory structure setup
  - [ ] Configuration files
  - [ ] Initial brain creation
  - [ ] Live Bot startup
  - [ ] Trainer startup
  - [ ] Monitoring & logging
  - [ ] Troubleshooting

**Deliverables**: 
- 20-page deployment guide

---

### 7.2 Runbook (4 hours)

- [ ] Write runbook covering:
  - [ ] Daily operations
  - [ ] Starting Live Bot
  - [ ] Running Trainer
  - [ ] Monitoring brain versions
  - [ ] Checking training progress
  - [ ] Handling failures
  - [ ] Rolling back brain
  - [ ] Emergency procedures

**Deliverables**: 
- 15-page runbook

---

### 7.3 Troubleshooting Guide (2 hours)

- [ ] Document common issues:
  - [ ] Brain loading fails
  - [ ] Training takes too long
  - [ ] experience.db grows too large
  - [ ] Redis connection fails
  - [ ] Hot-reload doesn't work
  - [ ] Performance degradation
- [ ] Add solutions for each

**Deliverables**: 
- 10-page troubleshooting guide

---

### 7.4 Automation Scripts (4 hours)

- [ ] Create `start-live-bot.ps1`:
  ```powershell
  # Check brain exists
  # Start Live Bot
  # Monitor logs
  ```

- [ ] Create `start-trainer.ps1`:
  ```powershell
  # Check experience.db has data
  # Start Trainer
  # Monitor progress
  # Publish brain if successful
  ```

- [ ] Create `rollback-brain.ps1`:
  ```powershell
  # List available versions
  # Prompt for version
  # Update symlink
  # Notify Live Bot
  ```

- [ ] Setup Windows Task Scheduler:
  - [ ] Live Bot: Start at 9:00 AM ET
  - [ ] Trainer: Start at 5:00 PM ET

**Deliverables**: 
- 3 PowerShell scripts
- Task Scheduler configuration

---

### 7.5 Architecture Documentation (2 hours)

- [ ] Update architecture diagrams
- [ ] Document data flow
- [ ] Document brain format
- [ ] Document versioning scheme
- [ ] Update existing documentation

**Deliverables**: 
- 5 architecture diagrams
- Updated documentation

---

## 📊 Progress Tracking

### Phases Completed
- [ ] Phase 1: Project Setup
- [ ] Phase 2: Infrastructure Layer
- [ ] Phase 3: Training Components
- [ ] Phase 4: Historical Replay Migration
- [ ] Phase 5: Live Bot Modifications
- [ ] Phase 6: End-to-End Testing
- [ ] Phase 7: Documentation & Deployment

### Time Tracking
| Phase | Estimated | Actual | Delta |
|-------|-----------|--------|-------|
| 1 | 8h | | |
| 2 | 24h | | |
| 3 | 32h | | |
| 4 | 24h | | |
| 5 | 24h | | |
| 6 | 32h | | |
| 7 | 16h | | |
| **Total** | **160h** | | |

---

## 🚧 Blockers & Issues

### Current Blockers
_(Add blockers as you encounter them)_

- [ ] 

### Resolved Issues
_(Add resolved issues for reference)_

- [ ] 

---

## ✅ Definition of Done

Each phase is considered complete when:

✅ All tasks in phase marked complete  
✅ Tests pass  
✅ Code reviewed  
✅ Documentation updated  
✅ Performance requirements met  
✅ No critical bugs  

---

## 🎯 Final Acceptance Criteria

The split is considered successful when:

✅ Live Bot runs in InferenceOnly mode  
✅ Decision latency < 10ms  
✅ Memory usage < 2GB  
✅ Trainer completes in < 4 hours  
✅ Brain hot-reload works  
✅ Rollback capability works  
✅ All tests pass  
✅ Documentation complete  
✅ Side-by-side comparison shows identical behavior  
✅ System runs in production for 1 week without issues  

---

**Status**: 📋 Ready to start implementation  
**Next Step**: Begin Phase 1: Project Setup
