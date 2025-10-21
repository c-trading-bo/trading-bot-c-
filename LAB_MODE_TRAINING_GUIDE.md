# 🧪 Lab Mode Training Guide

## Overview

Lab Mode is the offline training environment for QBot's ML/RL models. It runs on Sundays (12:00 PM - 5:45 PM Eastern) and uses historical data fetched via Python scripts, maintaining complete segregation from live trading APIs.

## 🎯 ML/RL Component Training Status

### ✅ TRAINABLE COMPONENTS (2/9)
Components with dedicated trainers that produce new ONNX models:

1. **CVaR-PPO**
   - Trainer: `TradingBot.RLAgent.CVaRPPOTrainer`
   - Output: `artifacts/models/cvar_ppo_v{version}.onnx`
   - Training: `HistoricalTrainingOrchestrator.TrainCVarPPOAsync()`

2. **Neural-UCB**
   - Trainer: `TradingBot.RLAgent.NeuralUcbBanditTrainer`
   - Output: `artifacts/models/neural_ucb_v{version}.onnx`
   - Training: `HistoricalTrainingOrchestrator.TrainNeuralUCBAsync()`

### ⚠️ INTEGRATED COMPONENTS (7/9)
Components registered as champions but without dedicated trainers:

3. **Regime-Detector**: Bootstrap placeholder model
4. **Model-Ensemble**: Bootstrap placeholder model
5. **Online-Learning-System**: Uses IncrementalBayesianOptimizer (optimization only)
6. **Slippage-Latency-Model**: Bootstrap placeholder model
7. **S15-RL-Policy**: Validated but not trained (uses external S15 system)
8. **Pattern-Recognition**: Bootstrap placeholder model
9. **PM-Optimizer**: Uses PositionManagementOptimizer (optimization only)

## 🚀 Running Training

### Option 1: Sunday Automatic Schedule
Training runs automatically on Sundays 12:00 PM - 5:45 PM Eastern Time (no configuration needed).

### Option 2: Force Training Immediately
Set environment variable to bypass schedule:

**Windows (PowerShell):**
```powershell
$env:FORCE_LAB_NOW = "1"
dotnet run --project src/UnifiedOrchestrator
```

**Linux/Mac:**
```bash
FORCE_LAB_NOW=1 dotnet run --project src/UnifiedOrchestrator
```

**Docker/Docker Compose:**
```yaml
environment:
  - FORCE_LAB_NOW=1
```

## 📁 Model Registry Structure

```
model_registry/
├── CVaR-PPO_champion.txt          # Points to current champion version
├── Neural-UCB_champion.txt
├── [7 other champion pointers]
├── models/
│   ├── CVaR-PPO_v1.0.0-bootstrap.json
│   ├── CVaR-PPO_v{version}.json
│   └── [other model metadata]
├── promotions/
│   └── [promotion records]
└── artifacts/
    └── [ONNX model files]
```

### Champion Pointer Format
Each `*_champion.txt` file contains a single line with the version ID:
```
v1.0.0-bootstrap
```

## 🔄 Training Pipeline Flow

1. **Data Loading** (HistoricalTrainingOrchestrator)
   - Load experiences from Terminal Mode (last 7 days)
   - Load 90-day historical bars from JSON files
   - Python script: `fetch-and-save-historical-data.py`

2. **Sequential Training**
   - Train CVaR-PPO model
   - Train Neural-UCB model
   - Optimize Position Management
   - Run S15 Shadow Validation

3. **Model Registration**
   - Register trained models as challengers
   - Generate version IDs: `v{timestamp}_{hash}`
   - Save ONNX artifacts to registry

4. **Promotion Evaluation**
   - Compare challengers vs champions
   - Promote if metrics improve (Sharpe, WinRate, MaxDrawdown)
   - Update champion pointers atomically

## 🐛 Troubleshooting

### Issue: "Bootstrap runs on every startup"
**Cause**: Missing champion pointer files  
**Fix**: Verify all 9 `*_champion.txt` files exist in `model_registry/`

### Issue: "Training never starts"
**Cause**: Not Sunday or outside time window  
**Fix**: Set `FORCE_LAB_NOW=1` environment variable

### Issue: "No ONNX models generated"
**Cause**: Training errors or incomplete runs  
**Check**: 
- View logs for training errors
- Verify Python script runs successfully
- Check disk space for artifact storage

### Issue: "Old log messages appearing"
**Not a DLL cache issue**: The log messages exist in the current source code. If you see unexpected messages, check `HistoricalTrainingOrchestrator.cs` line 787.

## 📊 Monitoring Training Progress

Watch for these log patterns:

```
🌱 [MODEL-BOOTSTRAP] Registry already initialized - skipping bootstrap
[LAB-DEBUG] ⏰ IsTrainingTime() called at {timestamp}
[LAB-DEBUG] FORCE_LAB_NOW=1 detected - forcing training to START NOW
[LAB] Fetching historical data using Python script...
[LAB] Historical data fetch completed successfully
🧠 [TRAINING] Starting CVaR-PPO training...
🧠 [TRAINING] CVaR-PPO training completed
```

## 🎓 Next Steps

1. **Verify Setup**: Run with `FORCE_LAB_NOW=1` to test training pipeline
2. **Monitor Outputs**: Check `artifacts/models/` for new ONNX files
3. **Review Metrics**: Examine champion vs challenger comparison logs
4. **Production Use**: Let Sunday schedule run automatically

## 📚 Related Documentation

- `src/UnifiedOrchestrator/Services/HistoricalTrainingOrchestrator.cs` - Main training orchestrator
- `src/UnifiedOrchestrator/Services/ModelRegistryBootstrapService.cs` - Bootstrap service with component documentation
- `src/UnifiedOrchestrator/Scheduling/InternalScheduler.cs` - Training schedule logic
- `src/UnifiedOrchestrator/Runtime/FileModelRegistry.cs` - Model registry implementation
