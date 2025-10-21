# 🚀 Lab Mode Training Quick Reference

## Instant Commands

### Force Training Now
```bash
# Linux/Mac/WSL
FORCE_LAB_NOW=1 dotnet run --project src/UnifiedOrchestrator

# Windows PowerShell
$env:FORCE_LAB_NOW="1"; dotnet run --project src/UnifiedOrchestrator

# Docker/Compose
docker run -e FORCE_LAB_NOW=1 qbot
```

### Check Champion Pointers
```bash
cat model_registry/CVaR-PPO_champion.txt
cat model_registry/Neural-UCB_champion.txt
```

### Verify ONNX Models
```bash
ls -lh artifacts/models/*.onnx
```

## Training Schedule

**Automatic:** Sunday 12:00 PM - 5:45 PM Eastern Time  
**Override:** Set `FORCE_LAB_NOW=1` environment variable

## ML/RL Components

| Component | Trainable | Output |
|-----------|-----------|--------|
| CVaR-PPO | ✅ Yes | `cvar_ppo_v{version}.onnx` |
| Neural-UCB | ✅ Yes | `neural_ucb_v{version}.onnx` |
| Regime-Detector | ❌ Bootstrap only | N/A |
| Model-Ensemble | ❌ Bootstrap only | N/A |
| Online-Learning-System | ❌ Optimization only | N/A |
| Slippage-Latency-Model | ❌ Bootstrap only | N/A |
| S15-RL-Policy | ❌ External system | N/A |
| Pattern-Recognition | ❌ Bootstrap only | N/A |
| PM-Optimizer | ❌ Optimization only | N/A |

## Model Registry Structure

```
model_registry/
├── *_champion.txt       # Version pointers (9 files)
├── models/              # Model metadata JSON files
├── promotions/          # Promotion history
└── artifacts/           # ONNX model binaries
```

## Key Log Messages

```
✅ Good:
🌱 [MODEL-BOOTSTRAP] Registry already initialized - skipping bootstrap
[LAB-DEBUG] FORCE_LAB_NOW=1 detected - forcing training to START NOW
[LAB] Historical data fetch completed successfully
🧠 [TRAINING] CVaR-PPO training completed

⚠️ Warning:
[LAB] Python executable not found - historical data fetch skipped
[LAB] Historical data fetch script not found

❌ Error:
[LAB] Historical data fetch failed with exit code {code}
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training never starts | Not Sunday | Set `FORCE_LAB_NOW=1` |
| Bootstrap re-runs | Missing champion files | Check `model_registry/*.txt` |
| No ONNX files | Training incomplete | Check logs for errors |
| Build fails | Hardcoded test code | Remove keywords like "HARDCODED" |

## File Locations

- **Training Logic:** `src/UnifiedOrchestrator/Services/HistoricalTrainingOrchestrator.cs`
- **Schedule Logic:** `src/UnifiedOrchestrator/Scheduling/InternalScheduler.cs`
- **Bootstrap:** `src/UnifiedOrchestrator/Services/ModelRegistryBootstrapService.cs`
- **Registry:** `src/UnifiedOrchestrator/Runtime/FileModelRegistry.cs`

## Environment Variables

```bash
FORCE_LAB_NOW=1           # Force training to run immediately
ASPNETCORE_ENVIRONMENT    # Set to 'Development' for verbose logs
```

## Documentation

- 📘 **Full Guide:** `LAB_MODE_TRAINING_GUIDE.md`
- 🔍 **Issue Analysis:** `LAB_MODE_ISSUE_ANALYSIS.md`
- 📋 **This Card:** `LAB_MODE_QUICK_REF.md`
