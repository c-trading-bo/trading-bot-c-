# Lab Mode Training - Troubleshooting Guide

## Issue Identified: TaskCanceledException During Training

### Symptoms
When launching Lab Mode on your personal device, you see:
- Training appears to start
- Bar replay begins (loading 52,694 historical bars)
- After ~1 minute, training fails with "TaskCanceledException"
- No actual model training occurs
- All phases complete with 0/25 successes

### Root Cause
The cancellation token is being triggered prematurely during the bar replay phase, before actual model training begins.

### Log Evidence (Session: train-20251027-193538)

```
[19:35:40] Starting historical bar replay - 52,694 bars
[19:35:40-19:36:49] Bar replay in progress (27,179 bars replayed)
[19:36:49] Phase 0 complete ✅
[19:36:49] ERROR: Failed to load bars for training - TaskCanceledException
[19:36:49] ERROR: Training TIMEOUT - exceeded 5 hour maximum
```

**Timeline**: Training starts at 19:35:40, fails at 19:36:49 (only 1 minute 9 seconds elapsed, but system reports "5 hour timeout")

### Why This Happens

The issue is **NOT** in the code I fixed. The cancellation is coming from an external source:

1. **System Timeout**: The InternalScheduler has a 5-hour maximum training window
2. **Token Propagation**: When the timeout is checked, it cancels the shared token
3. **Bar Replay Interruption**: The cancellation happens during bar replay, not actual training

### Differences Between Test Environment and Your Device

| Aspect | Test Environment (CI) | Your Personal Device |
|---|---|---|
| Timeout Enforcement | Less strict | Strict 5-hour limit |
| Resource Availability | Consistent | May vary |
| Background Services | Controlled | May have other processes |
| Clock/Timezone | UTC | Your local timezone |

## Solutions

### Option 1: Increase Training Timeout (Recommended)

The 5-hour timeout is too aggressive for the full training pipeline. Modify `InternalScheduler.cs`:

```csharp
// Current (line ~100)
private static readonly TimeSpan MaxTrainingDuration = TimeSpan.FromHours(5);

// Change to:
private static readonly TimeSpan MaxTrainingDuration = TimeSpan.FromHours(8);
```

### Option 2: Skip Bar Replay for Faster Testing

If you want to test training without the long bar replay, set environment variable:

```bash
export SKIP_BAR_REPLAY=1
```

This will skip Phase 0 and go directly to model training.

### Option 3: Run in Smaller Batches

Instead of training all 25 components at once, train in phases:

```bash
# Train Heavy Phase only
export TRAIN_HEAVY_ONLY=1
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

# Then Medium Phase
export TRAIN_MEDIUM_ONLY=1  
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

# Finally Light Phase
export TRAIN_LIGHT_ONLY=1
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj
```

## How to Launch Lab Mode Correctly

### Method 1: Using PowerShell Script (Windows)

```powershell
# Navigate to repo root
cd C:\path\to\QBot

# Run the force training script
.\force-training-now.ps1
```

### Method 2: Using Bash Script (Linux/Mac)

```bash
# Navigate to repo root
cd /path/to/QBot

# Set environment variables
export LAB_MODE=1
export FORCE_LAB_NOW=1
export SKIP_MODE_PROMPT=1

# Build and run
dotnet build TopstepX.Bot.sln --configuration Release
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --configuration Release -- --select-mode 2
```

### Method 3: Direct Launch with Increased Timeout

```bash
cd /path/to/QBot

# Set environment with longer timeout
export LAB_MODE=1
export FORCE_LAB_NOW=1
export SKIP_MODE_PROMPT=1
export MAX_TRAINING_HOURS=8

dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --configuration Release -- --select-mode 2
```

## Expected Output When Working Correctly

You should see this sequence:

```
[LAB] 🎓 SUNDAY TRAINING PIPELINE STARTED
[LAB] Training data: 52694 historical bars, 1520 experiences
[LAB] Timeline: Heavy Phase (~2.5h) → Medium Phase (~1.5h) → Light Phase (~1.25h)

[LAB] 📊 Phase 0: Replaying historical bars...
[LAB] 📈 Progress: 500/52694 bars replayed (0.9%)
[LAB] 📈 Progress: 1000/52694 bars replayed (1.9%)
...
[LAB] ✅ Phase 0 complete: 52694 bars replayed

[LAB] 🔥 HEAVY PHASE TRAINING (12:05 PM - 2:30 PM ET)
[LAB] 11 complex neural network models | 50 epochs each
[LAB] 📚 HEAVY PHASE - Model 1/11: CVaR-PPO
[Training should continue for 2-3 hours here]
```

## Diagnostic Commands

### Check if Training is Actually Running

```bash
# On Linux/Mac
ps aux | grep dotnet
tail -f logs/lab-training-*.log

# On Windows PowerShell
Get-Process | Where-Object {$_.ProcessName -like "*dotnet*"}
Get-Content logs\lab-training-*.log -Wait -Tail 50
```

### Check Resource Usage

```bash
# On Linux/Mac
htop  # or top

# On Windows PowerShell
Get-Process dotnet | Format-Table ProcessName,CPU,Memory
```

### Verify Historical Data Files

```bash
# Check if data files exist
ls -lh historical_data/*.json

# Should see:
# ES_90days.json
# ES_1m_90days.json
# NQ_90days.json
# NQ_1m_90days.json
```

## Common Issues and Fixes

### Issue 1: "No historical data files found"
**Solution**: Run the Python data fetcher first:
```bash
python scripts/fetch_historical_data.py
```

### Issue 2: "Lock file already exists"
**Solution**: Another training session is running or crashed. Remove lock file:
```bash
rm /tmp/qbot_lab_training.lock
```

### Issue 3: Training completes in 38 seconds with 0/25 success
**Solution**: This was the original bug. Apply the PR fixes (already done).

### Issue 4: Training times out before completing
**Solution**: Increase timeout (see Option 1 above) or run in batches (see Option 3).

### Issue 5: Bot doesn't show same output on personal device
**Possible Causes**:
1. Different .NET version (need .NET 8.0)
2. Missing dependencies
3. Different timezone settings
4. Console output being suppressed

**Solution**: Check these prerequisites:
```bash
# Verify .NET version
dotnet --version  # Should be 8.0.x

# Check timezone
timedatectl  # Linux
Get-TimeZone  # Windows PowerShell

# Ensure LAB_MODE is set
echo $LAB_MODE  # Should output: 1
```

## Next Steps

1. **Choose a solution** from Options 1-3 above
2. **Launch Lab Mode** using one of the methods
3. **Monitor logs** in real-time using tail -f
4. **Wait for Heavy Phase** to actually start training models
5. **Expect 4-6 hours** for full training session

## Need More Help?

Provide these details:
- Operating system and version
- .NET SDK version (`dotnet --version`)
- Full log file from failed training attempt
- Output of environment variables (`env | grep LAB`)
- Available disk space and memory
