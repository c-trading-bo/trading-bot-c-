# Lab Mode Phase 10 & 11 - Quick Reference Card

## 🎯 What's New

### Phase 10: Idle State Management
Between training sessions, Lab Mode now provides:
- ⏰ **Smart Countdown** - Shows exact time until next Sunday training
- 🏥 **Hourly Health Checks** - Monitors disk space, files, registry
- 🚀 **Pre-Warming** - Starts 5 min before training for faster startup
- 📊 **Market Status** - Shows if market is open/closed/pre-market
- 🎮 **Watchdog Display** - Shows monitoring is active
- 🛑 **Graceful Shutdown** - Clean exit with next session info

### Phase 11: GitHub Cloud Backup
Optional cloud backup system:
- 📦 **Manifest Backup** - Training session metadata to GitHub
- 📝 **Summary Backup** - Results, timings, component status
- 💾 **Local Archiving** - ZIP models locally (keeps last 3)
- 🔄 **Disaster Recovery** - Restore manifests from GitHub
- 📊 **History Tracking** - All backups recorded in JSON
- ⚡ **Non-Blocking** - Training succeeds even if backup fails

---

## 🚀 Quick Start

### View Idle State
```bash
# Just run Lab Mode on a non-Sunday
./run-lab-mode.sh

# You'll see:
# ╔═══════════════════════════════════════════╗
# ║        LAB MODE - IDLE STATE              ║
# ║ Next Training: Sunday Oct 26, 12:00 PM   ║
# ║ Countdown: 6 days 2h 30m                  ║
# ╚═══════════════════════════════════════════╝
```

### Enable GitHub Backup
```bash
# 1. Create GitHub PAT at https://github.com/settings/tokens
#    Scopes needed: repo (full control)

# 2. Set environment variable
export GITHUB_BACKUP_TOKEN=ghp_your_token_here

# 3. Update appsettings.json
{
  "GitHub": {
    "BackupOwner": "your-org",
    "BackupRepository": "your-backup-repo",
    "BackupBranch": "main"
  }
}

# 4. Run training - backups will upload automatically
```

### Enable Verbose Logging
```bash
# See all DEBUG messages during idle
export IDLE_VERBOSE_LOGGING=1
```

---

## 📅 Timeline of Events

### Monday - Saturday (Idle State)
```
09:30 AM - Lab Mode starts, enters idle state
          - Displays countdown to Sunday
          - Shows market status
          - Watchdog active

10:30 AM - Hourly health check runs
          - Checks disk space (>20GB required)
          - Verifies data files readable
          - Tests model registry writable
          - Logs: "All systems nominal"

11:30 AM - Hourly countdown update
          - Logs: "Next Training: Sunday... (in 5 days 0h 30m)"
          
... (continues every hour)

08:30 PM - Market closed, idle continues
```

### Sunday (Training Day)
```
11:55 AM - Pre-warming phase starts
          - Warms filesystem cache
          - Initializes database paths
          - Compacts memory
          - Logs: "System pre-warming complete"

12:00 PM - Training window opens
          - Health checks run
          - Resource checks run
          - Training starts with watchdog
          
12:01 PM - Training session begins
          - Loads 90 days historical data
          - Loads 7 days experiences
          - Runs training pipeline
          
05:30 PM - Training completes
          - Saves models to registry
          - Runs promotion evaluations
          - [GitHub Backup runs if enabled]
          
05:30 PM - Returns to idle state
          - Shows next Sunday countdown
          - Hourly health checks resume
```

---

## 🏥 Health Check Details

### What's Monitored (Every Hour)
| Check | Threshold | Action on Failure |
|-------|-----------|-------------------|
| Disk Space | <20 GB | WARNING + Alert |
| Historical Data | Missing | WARNING + Alert |
| Model Registry | Not Writable | WARNING + Alert |
| Experience DB | Missing | WARNING (non-critical) |
| Lock Files | Stale | Auto-cleanup |

### Health Check Results
- ✅ **All Pass**: `[DEBUG] All systems nominal`
- ⚠️ **1-2 Issues**: `[WARN] Issues detected - [details]` + Alert
- ❌ **3+ Issues**: `[ERROR] System unhealthy` + Critical Alert

---

## 📦 GitHub Backup Details

### What Gets Backed Up
| Item | Location | Size | GitHub Upload |
|------|----------|------|---------------|
| Manifest | `manifests/manifest-{id}.json` | ~10 KB | ✅ Yes |
| Summary | `artifacts/summaries/summary-{id}.json` | ~5 KB | ✅ Yes |
| Models | `artifacts/backups/models-{id}.zip` | 4-10 GB | ❌ No (local only) |

### Backup Locations
- **GitHub**: `lab-backups/manifests/` and `lab-backups/summaries/`
- **Local**: `artifacts/backups/models-{sessionId}.zip`
- **History**: `artifacts/backups/backup-history.json`

### Backup Process
```
1. Training completes successfully
2. Generate manifest.json
3. Generate training-summary.json
4. [If GitHub enabled]
   - Upload manifest (with retry, compression if >1MB)
   - Upload summary (with retry)
   - Archive models locally (ZIP, keep last 3)
   - Log all operations
5. Continue (don't fail if GitHub upload fails)
```

### Restore Process
```bash
# Restore latest manifest from GitHub
./restore-from-github.sh

# Or restore specific session
./restore-from-github.sh --session a1b2c3d4
```

---

## 🎨 Console Output Examples

### Idle State Display
```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        LAB MODE - IDLE STATE                               ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Status:               IDLE - Waiting for next Sunday training             ║
║ Current Time:         Monday, Oct 20 2025, 9:30:05 AM ET                  ║
║ Next Training:        Sunday, Oct 26 2025, 12:00 PM ET                    ║
║ Countdown:            6 days 2h 29m                                        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Watchdog:             Active (will wake automatically)                    ║
║ Health Checks:        Running hourly (ensuring system readiness)          ║
║ Lock File:            Cleared (no concurrent session prevention)          ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Market Status:        Regular Trading Hours (9:30 AM - 4:00 PM ET)        ║
╠═══════════════════════════════════════════════════════════════════════════╣
║ Press Ctrl+C to exit gracefully                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

### Hourly Update
```
[INFO] Next Training: Sunday, Oct 26 2025, 12:00 PM ET (in 5 days 23h 30m) - Current: 10:30:00 AM ET
[DEBUG] Watchdog monitoring active - System ready for next session
```

### Health Check Success
```
[DEBUG] Running hourly health check during idle state...
[DEBUG] ✓ Disk space: 45.3 GB available
[DEBUG] ✓ Model registry writable
[DEBUG] ✓ Historical data directory accessible
[DEBUG] ✓ Experiences directory accessible
[DEBUG] Hourly health check: All systems nominal
```

### Health Check Warning
```
[WARN] Hourly health check: Issues detected - Low disk space: 18.5 GB (critical below 20 GB)
[WARN] [ALERT] Health check failure detected
```

### Pre-Warming
```
[INFO] Pre-warming systems (5 minutes before training window)...
[DEBUG] ✓ Data directory warmed
[DEBUG] ✓ Experience database paths cached
[DEBUG] ✓ Model registry warmed
[DEBUG] ✓ Memory compacted and ready
[INFO] System pre-warming complete - ready for training
```

### GitHub Backup
```
[INFO] GITHUB SYNC (Optional Cloud Backup) - started
[INFO] [GITHUB BACKUP] Uploading manifest for session a1b2c3d4...
[INFO] [GITHUB BACKUP] ✓ Manifest uploaded: lab-backups/manifests/manifest-a1b2c3d4.json
[INFO] [GITHUB BACKUP] ✓ Summary uploaded: lab-backups/summaries/summary-a1b2c3d4.json
[INFO] [GITHUB BACKUP] ✓ Models archived locally: models-a1b2c3d4.zip (5432.8 MB)
[INFO] Note: Terminal Mode will use local registry (no GitHub dependency)
```

### Graceful Shutdown
```
^C
[INFO] Shutdown requested during idle state
[INFO] Lab Mode shutdown complete - next session: Sunday, Oct 26 2025, 12:00 PM ET
[INFO] Graceful shutdown complete
```

---

## 🔧 Troubleshooting

### Issue: Low disk space warning
**Solution**: Clean up old logs/models
```bash
# Find large files
du -h artifacts/ | sort -h | tail -20

# Clean up old archives (keeps last 3 automatically)
# Manual cleanup if needed:
rm artifacts/backups/models-old*.zip
```

### Issue: GitHub backup fails
**Check**: 
1. Token is valid: `curl -H "Authorization: token $GITHUB_BACKUP_TOKEN" https://api.github.com/user`
2. Repository exists and accessible
3. Rate limits not exceeded (5000/hour for authenticated)

**Note**: Training will succeed even if backup fails (non-critical)

### Issue: Health checks failing
**Check**:
1. Disk space: `df -h`
2. Data directory: `ls -lh data/historical/`
3. Model registry: `ls -lh model_registry/`
4. Permissions: All directories writable

### Issue: No idle state display
**Check**:
1. Running on a non-Sunday (Monday-Saturday)
2. Not within training window (before 12 PM or after 5:45 PM ET on Sunday)
3. Check logs for errors

---

## 📊 Monitoring

### Key Log Messages to Watch

**Success Indicators:**
- `[INFO] LAB MODE - IDLE STATE` - Idle state active
- `[DEBUG] All systems nominal` - Health checks passing
- `[INFO] System pre-warming complete` - Ready for training
- `[INFO] ✓ Manifest uploaded` - GitHub backup working

**Warning Indicators:**
- `[WARN] Issues detected` - Minor health issues
- `[WARN] [GITHUB BACKUP] Rate limit hit` - Slow down backups
- `[WARN] GitHub backup failed (non-critical)` - Backup issue

**Error Indicators:**
- `[ERROR] System unhealthy` - Critical health issues
- `[ERROR] [GITHUB BACKUP] Failed to upload` - Backup completely failed
- `[ERROR] Health checks failed` - Pre-training checks failed

### Metrics to Track
- Idle state uptime (Monday-Saturday)
- Health check success rate (should be 100%)
- Pre-warming completion time (should be <10 seconds)
- GitHub backup success rate (if enabled)
- Training startup time (should be faster after pre-warming)

---

## 🎓 Best Practices

### 1. Monitor Disk Space
- Set up alerts for <30 GB free space
- Clean up old backups regularly
- Last 3 model archives are kept automatically

### 2. GitHub Backup Strategy
- Use private repository for sensitive training data
- Enable for compliance/audit requirements
- Monitor backup success rate
- Test restore process periodically

### 3. Health Check Review
- Review hourly health logs weekly
- Address warnings before they become errors
- Keep at least 50 GB free space for safety

### 4. Timezone Awareness
- All times in Eastern Time (ET)
- Handles DST automatically
- Training window: Sunday 12:00 PM - 5:45 PM ET

### 5. Graceful Operations
- Always use Ctrl+C for shutdown (not kill)
- Wait for "Graceful shutdown complete" message
- Check next training time on exit

---

## 📚 Related Documentation

- **PHASE_10_11_IMPLEMENTATION_SUMMARY.md** - Complete technical details
- **IDLE_STATE_SAMPLE_OUTPUT.md** - More console output examples
- **InternalScheduler.cs** - Source code with inline comments
- **GitHubBackupService.cs** - Backup service implementation

---

## ✅ Checklist for Production

### Before First Run
- [ ] Understand idle state behavior
- [ ] Know training schedule (Sunday 12-5:45 PM ET)
- [ ] Monitor disk space (>20 GB free)
- [ ] Optionally configure GitHub backup

### If Using GitHub Backup
- [ ] Create GitHub PAT (repo write scope)
- [ ] Set GITHUB_BACKUP_TOKEN environment variable
- [ ] Configure repository in appsettings.json
- [ ] Test backup after first training session
- [ ] Verify files appear in GitHub repo

### Weekly Maintenance
- [ ] Review health check logs
- [ ] Check disk space trends
- [ ] Verify backups if enabled
- [ ] Clean up old archives if needed

---

## 🎉 Benefits

### Phase 10 Benefits
- 🎯 **Better Visibility** - Know exactly when next training happens
- 🏥 **Proactive Monitoring** - Catch issues before they impact training
- 🚀 **Faster Startup** - Pre-warming eliminates cold-start delays
- 📊 **Market Context** - See market hours for Terminal Mode planning
- 🛑 **Clean Operations** - Graceful shutdown preserves state

### Phase 11 Benefits
- 📦 **Audit Trail** - Complete history of all training sessions
- 🔄 **Disaster Recovery** - Restore metadata if needed
- 💾 **Local Backups** - Easy rollback to previous models
- ⚡ **Optional Feature** - Enable only if needed
- 🔒 **Non-Disruptive** - Training succeeds even if backup fails

---

**Version**: 1.0  
**Last Updated**: 2025-10-20  
**Status**: Production Ready ✅
