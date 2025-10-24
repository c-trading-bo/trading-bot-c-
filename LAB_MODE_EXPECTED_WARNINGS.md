# Lab Mode Expected Warnings

This document lists all warnings and informational messages that are **expected and normal** when running in Lab Mode.

## ✅ Expected Warnings (Not Errors)

### 1. TopstepX Connection Warnings
```
⚠️ TopstepX adapter health degraded: 0%
⚠️ TopstepX adapter not connected - cannot get live prices
❌ [CONFIG-PROOF] TopstepXClient:ClientType = '(null)' (NOT PRODUCTION)
❌ [HEALTH-CHECK] TopstepX adapter unhealthy: TopstepX adapter health: 0%, connected: False
```
**Reason:** Lab mode operates in **offline training mode** and does not require TopstepX API connections. The system generates synthetic market data for training purposes.

### 2. Missing Historical Data Files
```
⚠️ Historical data files not found:
   • ES_90days.json: MISSING
   • NQ_90days.json: MISSING
⚠️ [PROD-READY] Historical data seeding failed
```
**Reason:** Lab mode generates **synthetic historical data** on-the-fly for training. Pre-downloaded historical files are optional.

### 3. Missing ML Model Files
```
[ML-Memory] Model file not found: models/rl_model.onnx
[ML-Memory] Failed to load model from: models/rl/test_cvar_ppo.onnx
⚠️ [FEATURE-SPEC] Feature spec file not found at artifacts/current/feature_spec.json, using default
⚠️ [S15-RL] Model file not found at artifacts/current/rl_policy.onnx
```
**Reason:** Lab mode **trains new models from scratch**. ONNX inference models are generated during training, not required before training starts.

### 4. Model Registry Bootstrap Warnings
```
⚠️ Failed to register CVaR-PPO champion: The file already exists.
⚠️ Failed to register Neural-UCB champion: The file already exists.
...
```
**Reason:** Model registry files were created on first run. The system attempts to bootstrap them again but they already exist. This is **harmless and expected**.

### 5. GitHub Backup Disabled
```
[GITHUB BACKUP] No GitHub token configured - backups disabled
🌐 [CLOUD-SYNC] GitHub API request failed: Unauthorized
```
**Reason:** GitHub model backup is **optional**. Lab mode can run without GitHub integration.

### 6. API Health Check Failures
```
❌ [HEALTH-CHECK] API health check failed: API error: NotFound
❌ [HEALTH-CHECK] One or more health checks failed
```
**Reason:** Lab mode operates **offline**. API health checks are expected to fail since no live API connection is established.

### 7. Resource Constraints
```
[RESOURCE-MANAGER] CONSTRAINED system - MINIMAL strategy: Only 10 critical components, 7-day data
[RESOURCE-MANAGER] CRITICAL: Heavily reduced training - consider upgrading hardware
```
**Reason:** The system adapts training to available resources. This is a warning about **reduced training scope**, not a failure.

### 8. Lab Mode Safety Warning
```
⚠️ [LAB-SAFETY] WARNING: Lab training mode in Production environment!
```
**Reason:** Informational warning to remind users that Lab mode is for training only, not live trading.

## ❌ Real Errors to Investigate

If you see these errors, they indicate actual problems:

1. **Build Failures:** Compilation errors, missing dependencies
2. **Training Lock Conflicts:** Multiple training sessions running simultaneously (resolved by removing `/tmp/qbot_training.lock`)
3. **Out of Memory:** Process crashes due to insufficient RAM
4. **File Permission Errors:** Cannot write to required directories

## 🎯 Summary

Lab mode is designed to run **completely offline** without requiring:
- TopstepX API credentials
- Pre-downloaded historical data
- Pre-trained ML models
- GitHub tokens

All warnings listed above are **expected** and do not indicate failures. The system will:
- ✅ Generate synthetic market data
- ✅ Train ML/RL models from scratch
- ✅ Validate and test components
- ✅ Produce training metrics and logs

## 🔍 Verification

To verify lab mode is working correctly:

```bash
# Run lab mode
export LAB_MODE=1
export HISTORICAL_MODE=0
export DRY_RUN=1
export SKIP_MODE_PROMPT=1
dotnet run --project src/UnifiedOrchestrator -c Release

# Expected output:
# - Continuous market data processing
# - Brain decisions (BRAIN-DECISION logs)
# - CVaR-PPO actions
# - Neural-UCB strategy selection
# - Position sizing calculations
# - No FATAL errors or crashes
```

## 📊 Success Indicators

Lab mode is working correctly if you see:
- ✅ `[BRAIN-DECISION]` logs showing trading decisions
- ✅ `[CVAR-PPO]` action selections
- ✅ `[NEURAL-UCB]` strategy choices
- ✅ Market context updates
- ✅ Position sizing calculations
- ✅ Training metrics accumulation
- ✅ No process crashes
