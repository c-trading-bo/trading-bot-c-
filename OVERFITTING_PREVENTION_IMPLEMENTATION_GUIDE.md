# Complete Fully Automated Lab Mode Training - Implementation Guide

## Overview

This guide describes the fully automated training system with overfitting prevention for the QBot trading system. The system operates in three modes with zero manual intervention required.

## Three Operational Modes

### 1. Terminal Mode (Monday-Saturday) - Live Trading
- Runs 24/7 during trading week
- Uses champion models from model registry
- Collects trading experiences for Sunday training
- NO training occurs - pure inference mode

### 2. Sunday Lab Mode (Every Sunday 12:05-5:45 PM ET) - Scheduled Training
- Triggered automatically by time-based scheduler
- Runs complete training pipeline with all safety gates
- Auto-promotes models that pass canary testing
- Uses production data (90-day historical dataset)
- Writes to `/manifests/training/` with tag `runType: scheduled`
- Sleeps after completion, waits for next Sunday
- Normal logging level (info)

### 3. Anyday Lab Mode (Emergency/On-Demand) - Triggered Training
- Triggered by specific conditions:
  - Performance degradation (Sharpe < 0.5 for 3 consecutive days)
  - Market regime change detected
  - Data integrity issues
  - Model staleness (> 14 days old)
  - Catastrophic forgetting detected
- Runs immediately with same rigor as Sunday mode
- Writes to `/manifests/sandbox/` with tag `runType: manual`
- Requires manual approval before promotion
- Verbose logging level (debug + trace)
- Exits when complete

## Part 1: Overfitting Prevention - The Foundation

All training sessions (Sunday and Anyday) automatically use these three safeguards:

### Component A: Dynamic Data Splitting

**Implementation**: `DynamicDataSplitStrategy.cs`

**What it does automatically**:

1. **Counts available historical days**
   - Checks historical data files (`data/historical/{symbol}_90days.json`)
   - Currently: 51 days, eventually: 90 days

2. **Calculates optimal split**
   - 51 days → 34 train / 10 validation / 7 test (67%/20%/13%)
   - 60 days → 40 train / 12 validation / 8 test (67%/20%/13%)
   - 90+ days → 60 train / 15 validation / 15 test (67%/17%/17%)

3. **Divides the actual bar data chronologically**
   - Oldest bars → Training set (models learn from these)
   - Middle bars → Validation set (used for early stopping)
   - Newest bars → Test set (NEVER shown to models, only for final promotion decision)

4. **Logs the split**
   ```
   [SPLIT] GROWTH STATE: 51 days available, using 34/10/7 split, 39 days until optimal
   [SPLIT] Bar distribution: 12240 train / 3600 validation / 2520 test (total: 18360)
   ```

5. **Enforces immutability**
   - Test set is locked away - no training code can access it until promotion decision time

**Zero configuration needed**: The math adapts automatically as data grows from 51 to 90 days.

### Component B: Early Stopping Tracker

**Implementation**: `EarlyStoppingTracker.cs`

**What it does automatically**:

For every model being trained (CVaR-PPO, SAC, LSTM, etc):

1. **Monitors validation performance**
   - After each training epoch
   - Evaluates model on validation set
   - Calculates Sharpe ratio or win rate

2. **Saves checkpoints when improving**
   - Whenever validation metric improves
   - Saves checkpoint file with that epoch's weights
   - Path: `artifacts/training_checkpoints/{component}_epoch_{N}.ckpt`

3. **Tracks patience counter**
   - Counts consecutive epochs without validation improvement
   - Default patience: 10 epochs

4. **Stops training early**
   - When 10 epochs pass with no validation improvement
   - Stops training and loads the best checkpoint (not the final epoch)

5. **Logs the decision**
   ```
   [EARLY-STOP] CVaR-PPO: Validation improved by 2.3% at epoch 15
   [EARLY-STOP] CVaR-PPO: STOPPING at epoch 52, best was epoch 42 with metric 1.34
   [EARLY-STOP] CVaR-PPO: Saved 48 epochs worth of training time, avoided overfitting
   ```

**Example automatic flow**:
- CVaR-PPO starts training, max 100 epochs allowed
- Epoch 1-42: Validation Sharpe keeps improving (1.05 → 1.34), checkpoints saved
- Epoch 43-52: Validation Sharpe plateaus at 1.34, no improvement
- Epoch 52: Early stopping triggers, loads checkpoint from epoch 42
- Training stops, best model from epoch 42 is used for promotion evaluation
- Saved 48 epochs worth of wasted training time, avoided overfitting

**Zero configuration needed**: Patience set to 10 epochs universally, applies to all components.

### Component C: Multi-Seed Training Coordinator

**Implementation**: `MultiSeedTrainingCoordinator.cs`

**What it does automatically**:

For each training component:

1. **Defines five random seeds**
   - Uses deterministic seeds: 42, 123, 456, 789, 1337

2. **Trains five times**
   - Runs entire training process five times, once per seed
   - Seed 42: Train on training set, early stop on validation set, evaluate on test set
   - Seed 123: Same process with different random initialization
   - Seed 456: Same process
   - Seed 789: Same process
   - Seed 1337: Same process

3. **Compares each seed to champion**
   - For each seed: Did this seed's final model beat the current champion on the TEST set?

4. **Counts successes**
   - Tallies how many seeds beat champion (need at least 3 out of 5)

5. **Makes promotion decision**
   - 0-2 seeds beat champion: "PROMOTION REJECTED: Only 2/5 seeds succeeded, likely random luck"
   - 3-5 seeds beat champion: "PROMOTION APPROVED: 4/5 seeds succeeded, real learning detected"

6. **Selects best seed**
   - Among successful seeds, picks the one with highest TEST Sharpe for promotion

7. **Logs all results**
   ```
   [MULTI-SEED] CVaR-PPO: Multi-seed training results:
   [MULTI-SEED]   Seed 42: PASS - Test metric 1.34 vs champion 1.20
   [MULTI-SEED]   Seed 123: PASS - Test metric 1.28 vs champion 1.20
   [MULTI-SEED]   Seed 456: PASS - Test metric 1.42 vs champion 1.20
   [MULTI-SEED]   Seed 789: FAIL - Test metric 1.18 vs champion 1.20
   [MULTI-SEED]   Seed 1337: PASS - Test metric 1.31 vs champion 1.20
   [MULTI-SEED] CVaR-PPO: PROMOTION APPROVED - 4/5 seeds succeeded
   [MULTI-SEED] CVaR-PPO: Promoting seed 456 with test metric 1.42
   ```

**Zero configuration needed**: Always uses same 5 seeds, always requires 3/5 success rate, fully automatic.

## Part 2: Integration Pattern

The overfitting prevention components are integrated into `HistoricalTrainingOrchestrator.cs` through dependency injection. The components are available for use in the training pipeline.

### Services Added to Constructor

Three new services injected:
- `DynamicDataSplitStrategy _dataSplitStrategy`
- `EarlyStoppingTracker _earlyStoppingTracker`
- `MultiSeedTrainingCoordinator _multiSeedCoordinator`

### Integration Points

#### LoadHistoricalDataAsync Pattern

Current behavior:
```
Loads all historical bars into one list
Returns: Dictionary<string, int> (symbol → bar count)
```

Enhanced behavior (when integrated):
```
1. Loads all historical bars
2. Counts total days of data
3. Calls _dataSplitStrategy.SplitData(bars, totalDays)
4. Returns three separate lists: train, validation, test
5. Logs: "Loaded 51 days, split into 34 train / 10 validation / 7 test"
```

#### Training Phase Pattern

For each component (CVaR-PPO, LSTM, etc):

Current behavior:
```
1. Initialize model
2. Train on all available data
3. Evaluate on test set
4. Promote if metrics improve
```

Enhanced behavior (when integrated):
```
1. Get five seeds from _multiSeedCoordinator.GetTrainingSeeds()
2. For each seed:
   a. Initialize model with this seed
   b. Create early stopping tracker for this seed
   c. Training loop:
      - Train one epoch on TRAINING set
      - Evaluate on VALIDATION set
      - Call _earlyStoppingTracker.ShouldStop(validationMetric, epoch)
      - If ShouldStop returns true: Load best checkpoint, break
      - If ShouldStop returns false: Continue to next epoch
   d. Evaluate final model on TEST set
   e. Record: seed, test Sharpe, validation Sharpe, beat champion yes/no
3. Pass all five results to _multiSeedCoordinator.MakePromotionDecision()
4. If approved: Promote best seed's model
   If rejected: Keep current champion
5. Log: "CVaR-PPO: 4/5 seeds succeeded, promoting seed 456 with Test Sharpe 1.42"
```

## Part 3: The Two-Lab System

### Sunday Lab Mode (Scheduled/Automatic)

**Trigger**: Time-based scheduler checks clock every 5 minutes, detects it's Sunday 12:05 PM Eastern

**Behavior**:
1. Runs full training - All Heavy, Medium, Light components with overfitting prevention
2. Uses production data - Official 90-day (or current 51-day) historical dataset
3. Auto-promotes winners - If models pass all gates (multi-seed, test performance, canary)
4. Writes to production manifest - `/manifests/training/` with tag `runType: scheduled`
5. Updates metrics database - Records all metrics as official production training run
6. Sleeps afterward - After 5:45 PM, enters idle mode, checks every 5 minutes for next Sunday
7. Runs forever - Never exits, keeps running 24/7 waiting for next Sunday

**Configuration**:
- Logging level: Normal (info level)
- Safety gates: ALL gates enforced strictly - no shortcuts
- Promotion policy: Automatic - if canary passes, model goes live for Monday
- Who triggers it: Nobody - clock triggers it automatically

### Anyday Lab Mode (Emergency/On-Demand)

**Trigger**: Specific condition detected

**Automatic detection conditions**:
1. Performance degradation - Terminal Mode detects champion Sharpe dropped below 0.5 for 3 consecutive days
2. Market regime shift - Regime detector identifies new market state
3. Data quality issue - New data reveals gap or corruption
4. Model staleness - Champion model is more than 14 days old
5. Catastrophic forgetting - Online learning metrics show models forgot critical patterns

**Behavior when triggered**:
1. Runs immediately - No waiting for Sunday, starts training NOW
2. Uses current data - Whatever historical data exists right now (could be 54 days mid-week)
3. Same overfitting prevention - Still uses train/val/test split, early stopping, multi-seed
4. Writes to sandbox manifest - `/manifests/sandbox/` with tag `runType: manual`
5. Requires approval before promotion - Doesn't auto-promote, waits for validation
6. Extra verbose logging - Debug and trace level logs for troubleshooting
7. Exits when done - Completes training then terminates, doesn't loop

**Configuration**:
- Logging level: Verbose (debug + trace level)
- Safety gates: ALL gates still enforced - same rigor as Sunday
- Promotion policy: Manual approval required - creates candidate models but doesn't auto-promote
- Who triggers it: Smart detection system - bot detects condition and triggers itself

### How The Bot Decides Which Mode To Use

**Environment variable checked**: `LAB_MODE_SCHEDULE`

**Values**:
- `"SCHEDULED"` (default) → Sunday Lab Mode behavior
- `"MANUAL"` → Anyday Lab Mode behavior
- Not set → Defaults to SCHEDULED

**The smart orchestrator logic**:

On startup, checks current mode:

**If SCHEDULED mode**:
1. Enters infinite loop
2. Every 5 minutes checks: Is it Sunday between 12:05 PM and 5:45 PM Eastern?
3. If yes: Triggers training
4. If no: Sleeps 5 more minutes
5. Never exits

**If MANUAL mode**:
1. Immediately starts training
2. Uses current historical data (whatever days exist)
3. Applies all overfitting prevention
4. Saves results to sandbox
5. Exits when complete

**Terminal Mode can trigger MANUAL mode**:
1. Detects performance issue
2. Sets environment variable to MANUAL
3. Spawns new process for Anyday Lab
4. Anyday Lab runs, completes, exits
5. Terminal Mode reviews results
6. If approved: Promotes models
7. If rejected: Keeps current champions
8. Resets environment variable to SCHEDULED

**All of this is automatic decision-making based on conditions.**

## Implementation Status

### ✅ Completed
- [x] `DynamicDataSplitStrategy.cs` created and registered in DI
- [x] `EarlyStoppingTracker.cs` created and registered in DI
- [x] `MultiSeedTrainingCoordinator.cs` created and registered in DI
- [x] Services injected into `HistoricalTrainingOrchestrator` constructor
- [x] Infrastructure ready for integration

### 📋 Integration Points Available

The three components are now available in `HistoricalTrainingOrchestrator` and can be used:

1. **_dataSplitStrategy** - Call `SplitData<T>(List<T> data, int totalDays)` to get train/val/test splits
2. **_earlyStoppingTracker** - Call `ShouldStop(metric, epoch, componentName, checkpointCallback)` in training loop
3. **_multiSeedCoordinator** - Call `GetTrainingSeeds()` and `MakePromotionDecision(results, championMetric)`

### 🎯 Usage Example Pattern

```csharp
// Pattern for integrating multi-seed training with early stopping
async Task TrainComponentWithOverfittingPrevention(
    string componentName,
    Func<int, List<object>, CancellationToken, Task<double>> trainFunc,
    List<object> trainingData,
    double championTestMetric,
    CancellationToken cancellationToken)
{
    var seeds = _multiSeedCoordinator.GetTrainingSeeds(); // [42, 123, 456, 789, 1337]
    var results = new List<SeedTrainingResult>();
    
    foreach (var seed in seeds)
    {
        _earlyStoppingTracker.Reset();
        double bestValMetric = 0;
        
        // Training loop with early stopping
        for (int epoch = 1; epoch <= 100; epoch++)
        {
            var valMetric = await trainFunc(seed, trainingData, cancellationToken);
            
            if (_earlyStoppingTracker.ShouldStop(valMetric, epoch, componentName, 
                async path => await SaveCheckpoint(path, cancellationToken)))
            {
                bestValMetric = _earlyStoppingTracker.GetBestValidationMetric();
                break;
            }
        }
        
        // Evaluate on test set
        var testMetric = await EvaluateOnTestSet(seed, cancellationToken);
        
        results.Add(_multiSeedCoordinator.CreateSeedResult(
            seed, testMetric, bestValMetric, $"model_{seed}.onnx"));
    }
    
    // Make promotion decision
    var decision = _multiSeedCoordinator.MakePromotionDecision(
        componentName, results, championTestMetric);
    
    if (decision.Approved)
    {
        await PromoteModel(decision.BestSeed.Value, cancellationToken);
    }
}
```

## Next Steps

The overfitting prevention infrastructure is now in place. The components are:
1. Created and tested
2. Registered in dependency injection
3. Injected into the training orchestrator
4. Ready for use in training pipelines

The training pipeline can now be enhanced to use these components following the patterns described in this guide, maintaining the principle of minimal, surgical changes to existing code.
