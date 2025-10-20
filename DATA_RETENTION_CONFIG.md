# Data Retention & Cleanup Configuration

## Overview
The **DataRetentionService** runs daily at **3:00 AM** to prevent bloat across all system components. It cleans up old files while preserving recent data and audit trails.

---

## 📋 What Gets Cleaned Up

| Data Type | Default Retention | Environment Variable | What Gets Removed |
|-----------|------------------|---------------------|-------------------|
| **Promotion Records** | 90 days | `PROMOTION_RETENTION_DAYS` | Old model promotion JSON files |
| **Training Checkpoints** | 30 days | `CHECKPOINT_RETENTION_DAYS` | Old training session checkpoints |
| **Training Artifacts** | 14 days | `TRAINING_ARTIFACT_RETENTION_DAYS` | Old manifests, summaries, reports |
| **Model Versions** | Keep 10 | `MODEL_VERSIONS_TO_KEEP` | Old model artifacts (keeps recent + champion) |
| **Temp Files** | 24 hours | `TEMP_FILE_RETENTION_HOURS` | `.tmp`, `.lock`, `.staging` files |
| **Historical Data Cache** | 30 days | *Fixed* | Cached parquet files |
| **Validation Reports** | 14 days | *Fixed* | Old validation result files |
| **Experience Files** | 90 days | `EXPERIENCE_RETENTION_DAYS` | Old Terminal trading experiences |
| **Live Training Data** | 7 days | `LIVE_TRAINING_DATA_RETENTION_DAYS` | Old JSONL training files |
| **Session Backups** | 30 days | `SESSION_BACKUP_RETENTION_DAYS` | Old position state sessions |

---

## 🛠️ Configuration

### Default Configuration (No Setup Required)
The service works out-of-the-box with sensible defaults:
- Promotion records: **90 days** (audit compliance)
- Training checkpoints: **30 days** (disaster recovery)
- Training artifacts: **14 days** (debugging recent training)
- Model versions: **10 per algorithm** (rollback capability)
- Temp files: **24 hours** (active operations)
- Experience files: **90 days** (training window)
- Live training data: **7 days** (recent training only)
- Session backups: **30 days** (position state recovery)

### Custom Configuration (Optional)
Add to your `.env` file to customize retention policies:

```bash
# Promotion Records (audit trail)
PROMOTION_RETENTION_DAYS=90

# Training Checkpoints (crash recovery)
CHECKPOINT_RETENTION_DAYS=30

# Training Artifacts (manifests, summaries, reports)
TRAINING_ARTIFACT_RETENTION_DAYS=14

# Model Versions to Keep (per algorithm)
MODEL_VERSIONS_TO_KEEP=10

# Temp Files (locks, staging)
TEMP_FILE_RETENTION_HOURS=24

# Experience Files (Terminal trading experiences)
EXPERIENCE_RETENTION_DAYS=90

# Live Training Data (JSONL files)
LIVE_TRAINING_DATA_RETENTION_DAYS=7

# Session Backups (position state)
SESSION_BACKUP_RETENTION_DAYS=30
```

---

## 📂 Directories Monitored

### 1. Model Registry Promotions
**Path**: `model_registry/promotions/*.json`  
**Action**: Removes promotion records older than retention period  
**Preserved**: Recent promotions for audit trail  
**Bloat Risk**: **HIGH** - 2 promotions × 8 models × every training run = rapid accumulation

### 2. Model Versions
**Path**: `model_registry/models/*.json` + artifacts  
**Action**: Keeps most recent N versions per algorithm + current champion  
**Preserved**: Champion model always kept regardless of age  
**Bloat Risk**: **MEDIUM** - Old versions accumulate with each training run

### 3. Training Checkpoints
**Path**: `checkpoints/**/*.json`  
**Action**: Removes checkpoints older than retention period  
**Preserved**: Recent checkpoints for crash recovery  
**Bloat Risk**: **MEDIUM** - Weekly training sessions create new checkpoints

### 4. Training Artifacts
**Path**: `training_artifacts/**/*.{json,md,txt}`  
**Action**: Removes old manifests, summaries, and reports  
**Preserved**: GitHub backup manifests (always kept for audit)  
**Bloat Risk**: **LOW** - Small files, but accumulate over time

### 5. Temp Files
**Path**: `**/*.{tmp,lock,staging}`  
**Action**: Removes stale temporary files older than 24 hours  
**Preserved**: Active temp files (< 24 hours old)  
**Bloat Risk**: **LOW** - Usually auto-cleaned, but stale files can accumulate

### 6. Historical Data Cache
**Path**: `data/historical/**/*.parquet`  
**Action**: Removes cached files older than 30 days  
**Preserved**: Recent 30 days for training  
**Bloat Risk**: **HIGH** - Large parquet files can consume significant disk space

### 7. Validation Reports
**Path**: `validation_reports/**/*.json`  
**Action**: Removes old validation reports  
**Preserved**: Recent 14 days for debugging  
**Bloat Risk**: **LOW** - Small JSON files

### 8. Experience Files
**Path**: `data/experiences/*.json`  
**Action**: Removes old Terminal trading experiences  
**Preserved**: Recent 90 days for Lab training  
**Bloat Risk**: **HIGH** - 50-200 files/week × 52 weeks = 2,600-10,400 files/year

### 9. Live Training Data
**Path**: `data/live_trades/live_trades_*.jsonl`  
**Action**: Removes old daily training data files  
**Preserved**: Recent 7 days for immediate training  
**Bloat Risk**: **MEDIUM** - Daily files with full trade context

### 10. Position State Session Backups
**Path**: `AppData/TradingBot/State/Sessions/**` and `Backups/**`  
**Action**: Removes old session directories and full backups  
**Preserved**: Recent 30 days for position recovery  
**Bloat Risk**: **LOW** - Created per session, small files

---

## 🕐 Schedule

### Primary Cleanup
- **Time**: Daily at **3:00 AM local time**
- **Duration**: ~5-30 seconds (depends on file count)
- **Impact**: Zero - runs when bot is idle (Terminal mode) or idle state (Lab mode)

### Initial Cleanup
- **Time**: 5 minutes after service startup
- **Purpose**: Clean up accumulated files after bot restart

### Health Check
- **Frequency**: Hourly (background monitoring)
- **Purpose**: Ensure cleanup service is responsive

---

## 📊 Monitoring

### Logs
Check logs for cleanup activity:
```bash
grep "DATA-RETENTION" logs/system/*.log
```

### Sample Log Output
```
[DATA-RETENTION] ========== Daily Cleanup Started ==========
[DATA-RETENTION] Cleaned up 16 promotion records older than 90 days (0.01 MB)
[DATA-RETENTION] Cleaned up old model versions (kept 10 per algorithm)
[DATA-RETENTION] Cleaned up 5 training checkpoints older than 30 days
[DATA-RETENTION] Cleaned up 12 training artifacts older than 14 days
[DATA-RETENTION] Cleaned up 3 temp files older than 24 hours
[DATA-RETENTION] Cleaned up 24 cached historical data files (freed 1250.45 MB)
[DATA-RETENTION] Cleaned up 8 validation reports older than 14 days
[DATA-RETENTION] Cleaned up 2847 experience files older than 90 days (145.23 MB)
[DATA-RETENTION] Cleaned up 8 live training data files older than 7 days (23.45 MB)
[DATA-RETENTION] Cleaned up 12 position state backups older than 30 days (0.85 MB)
[DATA-RETENTION] ========== Daily Cleanup Complete ========== 
Duration: 18.7s | Files Removed: 2935 | Space Freed: 1419.99 MB
```

---

## 🚨 What's Preserved

### Always Kept (Never Deleted)
1. **Current Champion Models** - Production models in use
2. **Recent Model Versions** (10 per algorithm) - Rollback capability
3. **Recent Promotions** (90 days) - Audit compliance
4. **Recent Checkpoints** (30 days) - Disaster recovery
5. **GitHub Backup Manifests** - Audit trail
6. **Active Temp Files** (< 24 hours) - Ongoing operations
7. **Recent Historical Data** (30 days) - Training readiness
8. **Recent Experiences** (90 days) - Lab training window
9. **Recent Live Training Data** (7 days) - Immediate use
10. **Recent Session Backups** (30 days) - Position recovery

---

## 🔧 Manual Cleanup (Emergency)

If disk space is critical, you can manually trigger cleanup:

### Option 1: Adjust Retention Policies
```bash
# Reduce retention periods temporarily
export PROMOTION_RETENTION_DAYS=30
export CHECKPOINT_RETENTION_DAYS=7
export TRAINING_ARTIFACT_RETENTION_DAYS=7

# Restart bot to apply
```

### Option 2: Manual File Deletion
```bash
# Remove old promotions (older than 60 days)
find model_registry/promotions -name "*.json" -mtime +60 -delete

# Remove old checkpoints (older than 14 days)
find checkpoints -name "*.json" -mtime +14 -delete

# Remove all temp files
find . -name "*.tmp" -delete
find . -name "*.lock" -delete
```

### Option 3: Clean Historical Data Cache
```bash
# WARNING: This will require re-downloading historical data
rm -rf data/historical/*.parquet
```

---

## 🎯 Recommendations

### Development Environment
```bash
# Aggressive cleanup for limited disk space
PROMOTION_RETENTION_DAYS=30
CHECKPOINT_RETENTION_DAYS=14
TRAINING_ARTIFACT_RETENTION_DAYS=7
MODEL_VERSIONS_TO_KEEP=5
TEMP_FILE_RETENTION_HOURS=12
```

### Production Environment (Lab Mode)
```bash
# Balanced cleanup for audit compliance
PROMOTION_RETENTION_DAYS=90
CHECKPOINT_RETENTION_DAYS=30
TRAINING_ARTIFACT_RETENTION_DAYS=14
MODEL_VERSIONS_TO_KEEP=10
TEMP_FILE_RETENTION_HOURS=24
```

### Production Environment (Terminal Mode)
```bash
# Minimal cleanup (Terminal has less accumulation)
PROMOTION_RETENTION_DAYS=90
CHECKPOINT_RETENTION_DAYS=7
TRAINING_ARTIFACT_RETENTION_DAYS=7
MODEL_VERSIONS_TO_KEEP=5
TEMP_FILE_RETENTION_HOURS=24
```

---

## ⚠️ Important Notes

1. **Promotion Records**: These are **audit trail** files. Keep retention period long enough for compliance (default: 90 days).

2. **Model Versions**: The service always keeps the **current champion** model regardless of age. Only old **challenger** versions are cleaned up.

3. **Checkpoints**: Used for **crash recovery**. If your training sessions are reliable, you can reduce retention (7-14 days).

4. **Historical Data Cache**: These are **large files**. Cleanup can free significant disk space but requires re-downloading data for training.

5. **Temp Files**: If you see many stale `.lock` files, it indicates unclean shutdowns. Fix the underlying issue rather than just cleaning up.

---

## 🔍 Troubleshooting

### Issue: "Cleanup not running"
**Check**: Service registration in `Program.cs`
```bash
grep "DataRetentionService" logs/system/*.log
```

### Issue: "Disk space still growing"
**Possible Causes**:
1. Log files (use `LogRetentionService` for logs)
2. Large historical data cache (check `data/historical/`)
3. Database files (not covered by this service)
4. Docker volumes (not accessible to cleanup service)

### Issue: "Important files deleted"
**Recovery**:
- Champion models are never deleted
- Recent promotions (< 90 days) are preserved
- Check GitHub backup for training artifacts

---

## 📈 Expected Impact

### Before Cleanup Service
- **48 promotion files** (accumulating indefinitely)
- **Multiple old model versions** per algorithm
- **Stale checkpoints** from old training runs
- **Temp files** from interrupted operations
- **2,000+ experience files** (from months of Terminal trading)
- **Daily JSONL files** accumulating
- **Session backups** from every trading session
- **Result**: Gradual disk bloat over weeks/months

### After Cleanup Service
- **~10-20 recent promotions** (audit window)
- **10 model versions** per algorithm (rollback capability)
- **Recent checkpoints** only (crash recovery)
- **No stale temp files**
- **~200-400 recent experiences** (90-day training window)
- **~7 recent JSONL files** (1 week)
- **~30 recent session backups**
- **Result**: Stable disk usage, predictable growth

---

## 🎉 Summary

The DataRetentionService provides **automatic, safe cleanup** with:
- ✅ **Zero configuration** required (works with defaults)
- ✅ **Preserves critical data** (champions, recent promotions, audit trails, training data)
- ✅ **Configurable retention** policies via environment variables
- ✅ **Daily automated** cleanup (3:00 AM)
- ✅ **Comprehensive coverage** (10 different data types)
- ✅ **Comprehensive monitoring** via logs
- ✅ **Production-safe** (never deletes in-use or critical files)

**Estimated Disk Savings**: 200-800 MB per week (depends on trading frequency and data accumulation)
