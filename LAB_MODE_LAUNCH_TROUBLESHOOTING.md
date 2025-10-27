# QBot Lab Mode - Troubleshooting Guide

## Quick Launch Instructions

### For Windows Users:
```powershell
# Option 1: Using the automatic launch script (RECOMMENDED)
.\launch-lab-auto.ps1

# Option 2: Manual launch with environment variables
$env:LAB_MODE = "1"
$env:FORCE_LAB_NOW = "1"
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release
echo "2`n2" | dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -c Release
```

### For Linux/Mac Users:
```bash
# Option 1: Using the automatic launch script (RECOMMENDED)
./launch-lab-mode.sh

# Option 2: Manual launch with environment variables
export LAB_MODE=1
export FORCE_LAB_NOW=1
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release
echo -e "2\n2" | dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj --no-build -c Release
```

## What You Should See

When Lab Mode launches successfully, you will see:

### 1. Menu Selection (Automatic)
```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    TopstepX Trading Bot - Mode Selection                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  [1] Terminal Mode (Live Trading)                                             ║
║  [2] Lab Mode (Historical Training)      ← AUTO-SELECTED                      ║
║  [3] Backtest Mode (Strategy Testing)                                         ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

### 2. Training Schedule Selection (Automatic)
```
╔════════════════════════════════════════════════════════════════════════════════╗
║                      Lab Mode - Training Schedule Options                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  [1] Scheduled Training (Sunday Only)                                         ║
║  [2] Manual Training (Run Now)           ← AUTO-SELECTED                      ║
║  [3] Back to Main Menu                                                        ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

### 3. Training Dashboard
```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     🧪 LAB MODE - SUNDAY TRAINING SESSION                         ║
║                        Session ID: train-20251027-XXXXXX                         ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

⏰ Time: X:XX:XX PM ET | Elapsed: Xs | ETA: XXXXs

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📈 OVERALL PROGRESS                                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10.0%                            │
│ Components: 25/250 completed (225 remaining)                                  │
│ Phase: 🔴 HEAVY PHASE (Large Neural Networks)                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. Training Progress
You should see components being trained one by one:
- CVaR-PPO Trainer (30 min)
- Neural-UCB Bandit (15 min)
- LSTM Time-Series (20 min)
- Pattern Recognition CNN (60 min)
- Regime Detector MLP (60 min)
- And more...

## Common Issues & Solutions

### Issue 1: "Nothing happens when I run the script"

**Symptoms:**
- Script runs but bot doesn't start
- No menu appears
- Command prompt returns immediately

**Solutions:**
1. Make sure you built the project first:
   ```bash
   dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release
   ```

2. Check if .NET 8.0 SDK is installed:
   ```bash
   dotnet --version  # Should show 8.0.x
   ```

3. Try running without input redirection to see error messages:
   ```bash
   dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release
   ```

### Issue 2: "Menu appears but bot doesn't start training"

**Symptoms:**
- Main menu shows up
- You can see options [1] [2] [3]
- Bot waits for input

**Solutions:**
1. Make sure you're selecting option 2 twice:
   - First: Select [2] Lab Mode
   - Second: Select [2] Manual Training

2. Use the automatic launch scripts provided:
   - Windows: `launch-lab-auto.ps1`
   - Linux/Mac: `launch-lab-mode.sh`

### Issue 3: "Training starts but stops immediately"

**Symptoms:**
- Dashboard appears briefly
- Training session ID shown
- Bot exits without training

**Possible Causes:**
1. **Missing Historical Data**: Bot needs historical data files
   - Check: `data/historical/` directory should have ES and NQ data
   - Solution: Run `ensure-historical-data.ps1` or `fetch-and-save-historical-data.py`

2. **Failed Health Checks**: Pre-training validation failed
   - Check logs for specific failures (disk space, RAM, CPU)
   - Solution: Free up resources and try again

3. **Lock File Exists**: Another instance is running
   - Check: `/tmp/qbot_lab_training.lock` exists
   - Solution: Delete the lock file or wait for other instance to finish

### Issue 4: "Build fails with errors"

**Symptoms:**
- `dotnet build` shows errors
- Missing packages or dependencies

**Solutions:**
1. Restore NuGet packages:
   ```bash
   dotnet restore
   ```

2. Clean and rebuild:
   ```bash
   dotnet clean
   dotnet build
   ```

3. Check .NET SDK version matches project requirements (net8.0)

## Verification Steps

### Step 1: Verify Environment
```bash
# Check .NET version
dotnet --version

# Check Python version (for some trainers)
python --version

# Check if .env file exists
ls -la .env
```

### Step 2: Verify Build
```bash
# Build the project
dotnet build src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release

# Should show: "Build succeeded. 0 Warning(s) 0 Error(s)"
```

### Step 3: Test Launch (Manual)
```bash
# Set environment variables
export LAB_MODE=1
export FORCE_LAB_NOW=1

# Run and manually select options
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj -c Release

# When menu appears:
# Type: 2 (then press Enter)
# Type: 2 (then press Enter again)
```

### Step 4: Verify Training Starts
Look for these log messages:
- `Training session train-XXXXXXXX-XXXXXX started`
- `Starting Heavy phase with 11 components`
- `HEAVY PHASE TRAINING - Model 1/11: CVaR-PPO`

## Getting Help

If you're still having issues:

1. **Check the logs**:
   - Logs are in: `logs/training/`
   - Look for: `session-summary-*.json`

2. **Run diagnostics**:
   ```bash
   ./dev-helper.sh test        # Run all tests
   ./verify-lab-mode.sh        # Verify Lab Mode setup
   ```

3. **Capture full output**:
   ```bash
   # Windows
   .\launch-lab-auto.ps1 2>&1 | Tee-Object -FilePath lab-output.log

   # Linux
   ./launch-lab-mode.sh 2>&1 | tee lab-output.log
   ```

4. **Check system resources**:
   - Disk space: Need at least 20 GB free
   - RAM: Need at least 4 GB available
   - CPU: Should be under 80% usage

## Expected Training Duration

| Phase | Components | Duration | Description |
|-------|-----------|----------|-------------|
| Heavy | 11 models | ~5.3 hours | Deep neural networks (CNNs, RNNs, RL agents) |
| Medium | 7 models | ~1.5 hours | Calibration, optimization |
| Light | 7 models | ~15 minutes | Online learning, fine-tuning |
| Validation | - | ~55 minutes | Testing and promotion |
| **TOTAL** | **25+** | **~7-8 hours** | **Complete training session** |

## What Happens During Training

1. **Pre-Training (5 minutes)**
   - Health checks (disk, RAM, CPU, data)
   - Load historical data
   - Load recent experiences
   - Initialize models

2. **Heavy Phase (5.3 hours)**
   - Train 11 deep neural networks
   - Each model: 50-270 epochs
   - Real backpropagation and gradient descent
   - ~3 million parameters trained

3. **Medium Phase (1.5 hours)**
   - Calibration of parameters
   - Optimization of position management
   - Statistical validation

4. **Light Phase (15 minutes)**
   - Online learning setup
   - Shadow model initialization
   - Adaptive learning configuration

5. **Validation & Promotion (55 minutes)**
   - Test all models
   - Compare against baseline
   - Promote if passing all tests
   - Create backups

## Success Indicators

Training is working correctly if you see:

✅ Dashboard updates every 5 seconds
✅ Progress bars moving forward
✅ Component completion messages
✅ No critical errors in activity log
✅ CPU/Memory usage shows activity
✅ Training processes listed in dashboard
✅ Session ID displayed at top
✅ Elapsed time incrementing

## Need More Help?

See these additional guides:
- `LAB_MODE_QUICK_REFERENCE.md` - Quick reference card
- `LAB_MODE_TRAINING_GUIDE.md` - Detailed training guide
- `QUICK_START_BOT_LAUNCH.md` - General bot launch guide
- `AI_AGENT_DEBUG_GUIDE.md` - Debugging guide
