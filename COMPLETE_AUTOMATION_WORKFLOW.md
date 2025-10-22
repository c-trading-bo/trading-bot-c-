# Complete Automation Workflow - Parts 4-7

## PART 4: TERMINAL MODE INTEGRATION (MONDAY-SATURDAY)

### What Terminal Mode Does With Overfitting-Prevented Models

#### Every Monday Morning (9:30 AM ET)

**Model Loading Process:**
1. **Loads new champions** - Checks model registry for models promoted last Sunday
2. **Validates they passed gates** - Reads manifest to confirm:
   - Train/validation/test split used
   - Early stopping applied
   - Multi-seed passed (3/5 seeds minimum)
   - Canary passed
3. **Initializes for live trading** - Loads champion ONNX files into memory
4. **Starts making decisions** - Uses these vetted models for all trading decisions

**Implementation in UnifiedTradingBrain.cs:**
```csharp
// On Monday startup, load champions from registry
public async Task InitializeForTradingWeekAsync(CancellationToken cancellationToken)
{
    _logger.LogInformation("[TERMINAL] Monday startup - loading champions from Sunday training");
    
    // Load manifest from last Sunday training
    var manifest = await LoadLatestTrainingManifestAsync(cancellationToken);
    
    // Validate all gates passed
    ValidateOverfittingPreventionGates(manifest);
    
    // Load champion models
    await LoadChampionModelsAsync(manifest.PromotedModels, cancellationToken);
    
    _logger.LogInformation("[TERMINAL] Loaded {Count} vetted champions, ready for trading", 
        manifest.PromotedModels.Count);
}

private void ValidateOverfittingPreventionGates(TrainingManifest manifest)
{
    // Verify train/val/test split was used
    if (!manifest.Metadata.ContainsKey("DataSplitApplied"))
    {
        throw new InvalidOperationException("Models were not trained with data splitting");
    }
    
    // Verify multi-seed training was used
    if (!manifest.Metadata.ContainsKey("MultiSeedResults"))
    {
        throw new InvalidOperationException("Models were not trained with multi-seed validation");
    }
    
    // Verify canary tests passed
    if (!manifest.CanaryTestsPassed)
    {
        throw new InvalidOperationException("Models did not pass canary testing");
    }
    
    _logger.LogInformation("[TERMINAL] ✅ All overfitting prevention gates validated");
}
```

#### During the Week (Monday-Saturday)

**Terminal Mode operates in pure inference mode:**

1. **Pure inference mode** - Never trains, only uses models
   - All training happens in Lab Mode (Sunday or Anyday)
   - Terminal Mode focuses on trading decisions

2. **Monitors performance** - Tracks live metrics:
   - Sharpe ratio
   - Win rate
   - Drawdown
   - R-multiple
   - Directional accuracy

3. **Detects degradation** - Automatic detection triggers:
   ```csharp
   public async Task MonitorPerformanceAndDetectDegradationAsync()
   {
       var recentMetrics = await _metricsTracker.GetRecentPerformanceAsync(days: 3);
       
       // Check degradation conditions
       var degradationDetected = 
           recentMetrics.Sharpe < 0.5m ||           // Severe Sharpe degradation
           recentMetrics.Drawdown > 0.10m ||        // 10% drawdown
           recentMetrics.ConsecutiveLosses >= 5;     // 5 consecutive losses
       
       var isPersistent = recentMetrics.DegradedDaysCount >= 3; // 3+ days
       
       if (degradationDetected && isPersistent)
       {
           _logger.LogWarning("[TERMINAL] Performance degradation detected: Sharpe {Sharpe:F2} for {Days} days",
               recentMetrics.Sharpe, recentMetrics.DegradedDaysCount);
           
           // Trigger Anyday Lab Mode automatically
           await TriggerAnydayLabModeAsync();
       }
   }
   
   private async Task TriggerAnydayLabModeAsync()
   {
       // Safety checks before triggering emergency retrain
       if (_isTrainingInProgress)
       {
           _logger.LogWarning("[TERMINAL] Training already in progress, skipping emergency retrain");
           return;
       }
       
       if (!await HasSufficientDataAsync()) // Need at least 30 days
       {
           _logger.LogWarning("[TERMINAL] Insufficient data for retraining");
           return;
       }
       
       if (!await HasSufficientResourcesAsync())
       {
           _logger.LogWarning("[TERMINAL] Insufficient resources for retraining");
           return;
       }
       
       _logger.LogInformation("[TERMINAL] All safety checks passed - triggering Anyday Lab Mode");
       
       // Set environment variable for Anyday Lab Mode
       Environment.SetEnvironmentVariable("LAB_MODE_SCHEDULE", "MANUAL");
       
       // Spawn Lab Mode process
       await SpawnLabModeProcessAsync();
       
       _logger.LogInformation("[TERMINAL] Anyday Lab Mode triggered, continuing trading with current models");
   }
   ```

4. **Collects experiences** - Every trade recorded:
   - Entry/exit prices
   - Strategy used
   - Regime at entry
   - Outcome (win/loss)
   - R-multiple
   - Slippage
   - Latency
   
5. **Online calibration** - Lightweight learning (NOT full retraining):
   - Updates prediction confidence
   - Calibrates slippage models
   - Updates regime beliefs
   - Adjusts position sizing
   
   ```csharp
   public async Task UpdateOnlineCalibrationAsync(TradingOutcome outcome)
   {
       // Update strategy confidence based on outcome
       await _confidenceCalibrator.UpdateAsync(
           strategy: outcome.Strategy,
           predicted: outcome.PredictedDirection,
           actual: outcome.ActualDirection,
           confidence: outcome.EntryConfidence
       );
       
       // Update slippage model
       await _slippageModel.UpdateAsync(
           expectedPrice: outcome.ExpectedPrice,
           actualPrice: outcome.ActualPrice,
           marketConditions: outcome.MarketConditions
       );
       
       // Update regime detector (Bayesian update)
       await _regimeDetector.UpdateBeliefAsync(
           observedFeatures: outcome.MarketFeatures,
           regime: outcome.DetectedRegime
       );
   }
   ```

6. **No manual intervention** - Fully automated self-monitoring

## PART 5: DAILY DATA GROWTH AUTOMATION (51 → 90 DAYS)

### Automatic Data Growth Process

The system automatically grows historical data from 51 days to 90 days over 8 weeks.

#### Daily Automated Process

**Script: `fetch-and-save-historical-data.py`** (already exists)

**Scheduled Execution:**
- Runs daily at 6:00 AM Eastern Time
- Scheduled via cron job or Windows Task Scheduler

**Automatic Behavior:**

1. **Appends newest day** - Fetches yesterday's market data:
   ```python
   # Fetch yesterday's 5-minute bars
   yesterday = datetime.now() - timedelta(days=1)
   bars = fetch_historical_bars(symbol, yesterday, interval="5min")
   
   # Append to existing data file
   append_bars_to_file(f"data/historical/{symbol}_90days.json", bars)
   ```

2. **Trims oldest day** - Maintains 90-day rolling window:
   ```python
   # Once we reach 90 days, drop day 91
   if total_days > 90:
       remove_oldest_day(data_file)
   ```

3. **Updates metadata** - Increments bar count:
   ```python
   # Each day adds ~78 bars (5-min bars: 6.5 hours × 12 bars/hour)
   metadata["bar_count"] += 78
   metadata["total_days"] = min(total_days + 1, 90)
   metadata["last_updated"] = datetime.now().isoformat()
   ```

#### Integration with Overfitting Prevention

**Automatic adaptation as data grows:**

| Week | Days | Split (Train/Val/Test) | Notes |
|------|------|----------------------|-------|
| 1 | 51 | 34 / 10 / 7 | Initial state |
| 2 | 56 | 37 / 11 / 8 | Growing |
| 3 | 61 | 40 / 12 / 9 | Growing |
| 4 | 66 | 44 / 13 / 9 | Growing |
| 5 | 71 | 47 / 14 / 10 | Growing |
| 6 | 76 | 50 / 14 / 12 | Growing |
| 7 | 81 | 54 / 15 / 12 | Growing |
| 8 | 86 | 57 / 15 / 14 | Almost optimal |
| 9+ | 90 | 60 / 15 / 15 | **LOCKED - Optimal** |

**Zero Configuration Required:**
- `DynamicDataSplitStrategy` automatically detects available days
- Calculates optimal split based on current data size
- No code changes needed week-to-week
- Sunday training automatically uses current split

**Logging Example:**
```
[LAB] Week 1: GROWTH STATE: 51 days available, using 34/10/7 split, 39 days until optimal
[LAB] Week 5: GROWTH STATE: 71 days available, using 47/14/10 split, 19 days until optimal
[LAB] Week 9: OPTIMAL STATE: 90 days available, using 60/15/15 split (permanent)
```

## PART 6: AUTOMATIC PROMOTION PIPELINE

### Complete Multi-Gate Promotion Process

After all training completes (Sunday or Anyday), models go through 5 automatic gates:

#### Phase 1: Multi-Seed Evaluation ✅ (Already Implemented)

**Process:**
- 5 seeds trained per component (42, 123, 456, 789, 1337)
- Each seed evaluated on TEST set (never seen during training)
- 3 out of 5 must beat champion
- Best seed selected for promotion

**Implementation:** See `MultiSeedTrainingCoordinator.cs`

**Logging:**
```
[MULTI-SEED] CVaR-PPO: Multi-seed training results:
[MULTI-SEED]   Seed 42: PASS - Test Sharpe 1.34 vs champion 1.20
[MULTI-SEED]   Seed 123: PASS - Test Sharpe 1.28 vs champion 1.20
[MULTI-SEED]   Seed 456: PASS - Test Sharpe 1.42 vs champion 1.20
[MULTI-SEED]   Seed 789: FAIL - Test Sharpe 1.18 vs champion 1.20
[MULTI-SEED]   Seed 1337: PASS - Test Sharpe 1.31 vs champion 1.20
[MULTI-SEED] CVaR-PPO: PROMOTION APPROVED - 4/5 seeds succeeded
```

#### Phase 2: Canary Testing ✅ (Already Implemented)

**Process:**
- Loads all new models into staging directory
- Runs inference tests on validation dataset
- Checks performance metrics

**Checks Performed:**
```csharp
public async Task<CanaryTestResult> RunCanaryTestsAsync()
{
    var result = new CanaryTestResult();
    
    // Check 1: Average latency
    result.AverageLatencyMs = await MeasureInferenceLatencyAsync();
    result.LatencyCheck = result.AverageLatencyMs < MaxAverageLatencyMs; // 50ms
    
    // Check 2: Output stability
    result.StabilityCheck = await CheckOutputStabilityAsync();
    // - No NaN values
    // - No extreme values (> 1000 or < -1000)
    // - Consistent predictions for same input
    
    // Check 3: Model loading success
    result.ModelsLoadedSuccessfully = _modelsLoaded == _modelsExpected;
    
    // Check 4: Memory usage
    result.MemoryUsageMB = await MeasureMemoryUsageAsync();
    result.MemoryCheck = result.MemoryUsageMB < 2000; // 2GB limit
    
    result.Passed = result.LatencyCheck && result.StabilityCheck && 
                    result.ModelsLoadedSuccessfully && result.MemoryCheck;
    
    return result;
}
```

**Logging:**
```
[CANARY] Canary tests PASS: 273 models, avg latency 23ms, no errors
[CANARY] ✅ All models stable and performant
```

#### Phase 3: Catastrophic Forgetting Check ✅ (Already Implemented)

**Process:**
- Tests new models on old scenarios from 30-90 days ago
- Compares performance to previous champion
- Ensures new models didn't forget old patterns

**Implementation:**
```csharp
public async Task<ForgettingCheckResult> CheckCatastrophicForgettingAsync(
    string newModelPath,
    string championModelPath)
{
    // Load historical scenarios (30-90 days old)
    var historicalScenarios = await LoadHistoricalScenariosAsync(
        fromDaysAgo: 90, toDaysAgo: 30);
    
    // Test new model
    var newModelResults = await EvaluateModelAsync(newModelPath, historicalScenarios);
    
    // Test champion model
    var championResults = await EvaluateModelAsync(championModelPath, historicalScenarios);
    
    // Calculate retention rate
    var retentionRate = newModelResults.Sharpe / championResults.Sharpe;
    
    var result = new ForgettingCheckResult
    {
        RetentionRate = retentionRate,
        Passed = retentionRate >= 0.95m, // Must retain 95% of performance
        NewModelSharpe = newModelResults.Sharpe,
        ChampionSharpe = championResults.Sharpe
    };
    
    return result;
}
```

**Logging:**
```
[FORGETTING] Forgetting check PASS: 97% retention on historical scenarios
[FORGETTING] New model Sharpe: 1.42, Champion Sharpe: 1.46 (on old data)
```

#### Phase 4: Bootstrap Significance Testing (Enhanced)

**Purpose:** Verify improvement is statistically significant, not random luck

**Process:**
```csharp
public async Task<SignificanceTestResult> RunBootstrapTestAsync(
    List<double> newModelResults,
    List<double> championResults)
{
    const int bootstrapSamples = 1000;
    var differences = new List<double>();
    
    // Bootstrap resampling
    for (int i = 0; i < bootstrapSamples; i++)
    {
        var sampleNew = ResampleWithReplacement(newModelResults);
        var sampleChampion = ResampleWithReplacement(championResults);
        
        var diff = sampleNew.Average() - sampleChampion.Average();
        differences.Add(diff);
    }
    
    // Calculate confidence interval
    differences.Sort();
    var ci95Lower = differences[(int)(0.025 * bootstrapSamples)];
    var ci95Upper = differences[(int)(0.975 * bootstrapSamples)];
    
    // Calculate p-value (proportion of samples where new is worse)
    var pValue = differences.Count(d => d <= 0) / (double)bootstrapSamples;
    
    var result = new SignificanceTestResult
    {
        PValue = pValue,
        ConfidenceIntervalLower = ci95Lower,
        ConfidenceIntervalUpper = ci95Upper,
        Significant = pValue < 0.05 && ci95Lower > 0,
        MeanDifference = differences.Average()
    };
    
    return result;
}
```

**Logging:**
```
[SIGNIFICANCE] Bootstrap test: p=0.012, CI=[+0.08, +0.24], SIGNIFICANT
[SIGNIFICANCE] Improvement is statistically significant with 95% confidence
```

#### Phase 5: Final Promotion Decision (Automatic)

**All Gates Must Pass:**

```csharp
public async Task<PromotionDecision> MakeFinalPromotionDecisionAsync(
    string componentName,
    MultiSeedResult multiSeedResult,
    CanaryTestResult canaryResult,
    ForgettingCheckResult forgettingResult,
    SignificanceTestResult significanceResult)
{
    var allGatesPassed = 
        multiSeedResult.Approved &&                    // ✅ 3/5 seeds succeeded
        canaryResult.Passed &&                         // ✅ Latency OK, no errors
        forgettingResult.Passed &&                     // ✅ 95%+ retention
        significanceResult.Significant;                // ✅ p < 0.05
    
    if (allGatesPassed)
    {
        _logger.LogInformation("[PROMOTION] {Component}: ALL GATES PASSED ✅", componentName);
        _logger.LogInformation("[PROMOTION]   Multi-seed: {Success}/{Total}", 
            multiSeedResult.SuccessfulSeedCount, multiSeedResult.TotalSeedCount);
        _logger.LogInformation("[PROMOTION]   Canary: {Latency}ms avg latency", 
            canaryResult.AverageLatencyMs);
        _logger.LogInformation("[PROMOTION]   Forgetting: {Retention:F1}% retention", 
            forgettingResult.RetentionRate * 100);
        _logger.LogInformation("[PROMOTION]   Significance: p={PValue:F3}", 
            significanceResult.PValue);
        
        // Determine promotion path
        var labMode = Environment.GetEnvironmentVariable("LAB_MODE_SCHEDULE");
        
        if (labMode == "SCHEDULED") // Sunday Lab
        {
            await AutoPromoteToProductionAsync(componentName, multiSeedResult.BestSeed);
            return PromotionDecision.AutoPromoted;
        }
        else // Anyday Lab
        {
            await CreatePromotionCandidateAsync(componentName, multiSeedResult.BestSeed);
            return PromotionDecision.AwaitingApproval;
        }
    }
    else
    {
        // Log detailed failure reason
        var failures = new List<string>();
        if (!multiSeedResult.Approved) 
            failures.Add($"Multi-seed: only {multiSeedResult.SuccessfulSeedCount}/5 seeds succeeded");
        if (!canaryResult.Passed) 
            failures.Add($"Canary: {canaryResult.FailureReason}");
        if (!forgettingResult.Passed) 
            failures.Add($"Forgetting: only {forgettingResult.RetentionRate:F1}% retention");
        if (!significanceResult.Significant) 
            failures.Add($"Not significant: p={significanceResult.PValue:F3}");
        
        _logger.LogWarning("[PROMOTION] {Component}: REJECTED - {Reason}", 
            componentName, string.Join("; ", failures));
        
        // Save to rejected folder for analysis
        await SaveToRejectedFolderAsync(componentName, multiSeedResult.BestSeed, failures);
        
        // Send alert
        await _alertService.AlertPromotionRejectedAsync(componentName, failures);
        
        return PromotionDecision.Rejected;
    }
}
```

## PART 7: COMPLETE AUTOMATIC WORKFLOW

### Normal Week Cycle (No Issues)

#### Monday 9:30 AM ET
- **Terminal Mode starts**
- Loads Sunday's new champions from model registry
- Validates all overfitting prevention gates passed
- Begins live trading with vetted models

#### Monday-Saturday
- **Terminal Mode operates**
- Pure inference - no training
- Collects trading experiences
- Monitors performance (Sharpe, win rate, drawdown)
- Online learning updates (confidence calibration, regime beliefs)
- Checks for performance degradation every 4 hours

#### Sunday 12:05 PM ET
- **Sunday Lab triggers automatically** (time-based)
- Checks available data (currently 51+ days, growing weekly)
- Loads historical bars + week's experiences

#### Sunday 12:05-12:20 PM: Data Preparation
- Splits data dynamically: 34 train / 10 validation / 7 test (adapts to available days)
- Logs split configuration
- Prepares training/validation/test datasets

#### Sunday 12:20-2:30 PM: Heavy Phase Training
- Trains 7 Heavy components with multi-seed + early stopping:
  1. CVaR-PPO (30 min)
  2. Neural UCB (15 min)
  3. LSTM (20 min)
  4. Pattern Recognition (15 min)
  5. Regime Detector (15 min)
  6. Slippage/Latency (10 min)
  7. Model Ensemble (15 min)

#### Sunday 2:30-4:00 PM: Medium Phase Training
- Trains 15 Medium components (~6 min each)
- Calibration models with multi-seed

#### Sunday 4:00-5:15 PM: Light Phase Training
- Trains 15 Light components (~5 min each)
- Online learning models with multi-seed

#### Sunday 5:15-5:35 PM: Canary Testing
- Loads all 273 models into staging
- Runs inference tests on validation dataset
- Checks latency, stability, memory usage
- **GATE 1 PASSED**: Canary tests OK

#### Sunday 5:35-5:40 PM: Promotion Evaluation
For each component:
- **GATE 2**: Multi-seed check (3/5 seeds passed?)
- **GATE 3**: Test performance (beats champion?)
- **GATE 4**: Catastrophic forgetting (95%+ retention?)
- **GATE 5**: Statistical significance (p < 0.05?)

If all gates pass:
- Auto-promotes to production registry
- Updates manifest with promotion details
- Copies best seed's model to production location

If any gate fails:
- Keeps current champion
- Saves rejected model to analysis folder
- Logs detailed failure reason

#### Sunday 5:40-5:45 PM: Finalization
- Backs up manifest to GitHub (optional)
- Generates training summary report
- Sends email notification with results
- Enters idle mode

#### Sunday 5:45 PM - Monday 12:00 AM
- **Sunday Lab enters idle mode**
- Checks every 5 minutes for next Sunday
- No training occurs

#### Monday 9:30 AM ET (Next Week)
- **Cycle repeats** with new champions

### Emergency Week Cycle (Performance Degradation)

#### Wednesday 2:00 PM ET
- **Terminal Mode detects degradation**
- Sharpe ratio: 0.42 (below 0.5 threshold)
- Consecutive days degraded: 3
- Logs: "Performance degradation detected: Sharpe 0.42 for 3 days"

#### Wednesday 2:05 PM ET
- **Safety checks before triggering Anyday Lab**
- ✅ Not already training
- ✅ Sufficient data (54 days available)
- ✅ Sufficient resources (CPU, RAM, disk)
- Sets `LAB_MODE_SCHEDULE=MANUAL`
- Spawns Anyday Lab Mode process

#### Wednesday 2:10-7:30 PM: Anyday Lab Training
- Same process as Sunday Lab but:
  - Uses current data (54 days → 36/10/8 split)
  - Extra verbose logging (debug + trace)
  - Writes to `/manifests/sandbox/` instead of `/manifests/training/`
  - Does NOT auto-promote

#### Wednesday 7:30 PM: Anyday Lab Complete
- All gates evaluated
- If passed: Creates promotion candidate
- Requires manual approval before promotion
- Logs: "Anyday Lab complete - awaiting approval for 12 models"

#### Thursday 9:00 AM
- **Admin reviews results**
- Checks training logs
- Validates promotion candidate
- If approved: Promotes models to production
- If rejected: Keeps current champions

#### Thursday 9:30 AM
- **Terminal Mode reloads models** (if approved)
- Continues trading with new champions
- Resets environment variable to `LAB_MODE_SCHEDULE=SCHEDULED`

### Data Growth Timeline

| Week | Data Available | Training Impact |
|------|---------------|----------------|
| 1 | 51 days | 34/10/7 split, all features working |
| 5 | 71 days | 47/14/10 split, better validation |
| 9 | 90 days | **60/15/15 split - OPTIMAL & LOCKED** |
| 10+ | 90 days | Maintains optimal split, oldest day dropped daily |

**Key Points:**
- System automatically adapts each week
- No configuration changes needed
- Logging clearly shows growth state
- Week 9 onwards: Optimal state achieved

### Monitoring & Alerts

**Automatic Email Notifications:**

1. **Sunday Training Success**
   - Subject: "Lab Training Succeeded - X Models Promoted"
   - Includes: Duration, models trained, promotions, next training date

2. **Sunday Training Failure**
   - Subject: "Lab Training Failed - Review Required"
   - Includes: Error details, failed components, troubleshooting steps

3. **Anyday Lab Triggered**
   - Subject: "Emergency Retraining Triggered - Performance Degradation"
   - Includes: Degradation metrics, trigger reason, expected completion time

4. **Promotion Rejected**
   - Subject: "Model Promotion Rejected - Gate Failure"
   - Includes: Which gates failed, retention rates, statistical tests

5. **Daily Performance Summary**
   - Subject: "Trading Performance - Day X"
   - Includes: Sharpe, win rate, PnL, trades executed

**Zero Manual Intervention Required:**
- All processes automatic
- Self-monitoring
- Self-correcting
- Admin only needed for Anyday Lab approval

This completes the full automation workflow from Parts 4-7.
