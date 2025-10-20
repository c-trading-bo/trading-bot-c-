# Lab Mode Idle State - Sample Console Output

This document shows what the console output will look like when Lab Mode enters idle state between training sessions.

## Initial Idle State Display (Example: Monday Morning)

```
[2025-10-20 09:30:00] [INFO] [LAB] Scheduler starting - Training Sunday 12:00 PM - 5:45 PM America/New_York
[2025-10-20 09:30:05] [INFO] [LAB] 
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

[2025-10-20 09:30:05] [DEBUG] [LAB] Watchdog Status:
[2025-10-20 09:30:05] [DEBUG] [LAB]   - Active: YES (will wake for next session automatically)
[2025-10-20 09:30:05] [DEBUG] [LAB]   - Health checks: Every 1 hour (ensuring readiness)
[2025-10-20 09:30:05] [DEBUG] [LAB]   - Lock file: Cleared
[2025-10-20 09:30:05] [DEBUG] [LAB]   - Next check: 2025-10-20 10:30:00 UTC
```

## Hourly Countdown Update (Example: 1 hour later)

```
[2025-10-20 10:30:00] [INFO] [LAB] Next Training: Sunday, Oct 26 2025, 12:00 PM ET (in 6 days 1h 30m) - Current: 10:30:00 AM ET
[2025-10-20 10:30:00] [DEBUG] [LAB] Watchdog monitoring active - System ready for next session
```

## Hourly Health Check (Example: Same time)

```
[2025-10-20 10:30:00] [DEBUG] [LAB] Running hourly health check during idle state...
[2025-10-20 10:30:01] [DEBUG] [LAB] ✓ Disk space: 45.3 GB available
[2025-10-20 10:30:01] [DEBUG] [LAB] ✓ Model registry writable
[2025-10-20 10:30:01] [DEBUG] [LAB] ✓ Historical data directory accessible
[2025-10-20 10:30:01] [DEBUG] [LAB] ✓ Experiences directory accessible
[2025-10-20 10:30:01] [DEBUG] [LAB] Hourly health check: All systems nominal
```

## Health Check with Warnings (Example: Low disk space detected)

```
[2025-10-20 14:30:00] [DEBUG] [LAB] Running hourly health check during idle state...
[2025-10-20 14:30:01] [WARN] [LAB] Hourly health check: Issues detected - Low disk space: 18.5 GB (critical below 20 GB)
[2025-10-20 14:30:01] [WARN] [ALERT] Health check failure detected - Idle health check: Low disk space: 18.5 GB (critical below 20 GB)
```

## Pre-Warming Phase (Example: Sunday 11:55 AM - 5 minutes before training)

```
[2025-10-26 11:55:00] [INFO] [LAB] Pre-warming systems (5 minutes before training window)...
[2025-10-26 11:55:01] [DEBUG] [LAB] ✓ Data directory warmed
[2025-10-26 11:55:01] [DEBUG] [LAB] ✓ Experience database paths cached
[2025-10-26 11:55:02] [DEBUG] [LAB] ✓ Model registry warmed
[2025-10-26 11:55:02] [DEBUG] [LAB] ✓ Memory compacted and ready
[2025-10-26 11:55:02] [INFO] [LAB] System pre-warming complete - ready for training
```

## Training Window Opens (Example: Sunday 12:00 PM)

```
[2025-10-26 12:00:00] [INFO] [LAB] Training window OPEN - Starting training with watchdog
[2025-10-26 12:00:01] [INFO] [LAB] Health checks passed
[2025-10-26 12:00:02] [INFO] [LAB] Resource checks passed
[2025-10-26 12:00:03] [INFO] [LAB] Training session started - RunID: a1b2c3d4, Sunday Oct 26, 12:00 PM ET
...
```

## Graceful Shutdown During Idle (Example: User presses Ctrl+C)

```
^C
[2025-10-20 15:45:23] [INFO] [LAB] Shutdown requested during idle state
[2025-10-20 15:45:23] [INFO] [LAB] Lab Mode shutdown complete - next session: Sunday, Oct 26 2025, 12:00 PM ET
[2025-10-20 15:45:23] [INFO] [LAB] Graceful shutdown complete
```

## Market Status Examples

### Pre-Market (Example: 5:00 AM)
```
║ Market Status:        Pre-Market (4:00 AM - 9:30 AM ET)                    ║
```

### After-Hours (Example: 5:00 PM)
```
║ Market Status:        After-Hours (4:00 PM - 8:00 PM ET)                   ║
```

### Weekend (Example: Saturday)
```
║ Market Status:        Closed (Weekend)                                     ║
```

### Overnight (Example: 2:00 AM)
```
║ Market Status:        Closed (Outside Trading Hours)                       ║
```

## GitHub Backup Integration (If Enabled)

After training completes successfully:

```
[2025-10-26 17:30:00] [INFO] [LAB] Training session complete - 2 promoted, 0 discarded
[2025-10-26 17:30:00] [INFO] [LAB] Next training: Sunday, Nov 02 2025, 12:00 PM ET
[2025-10-26 17:30:00] [INFO] [LAB] Entering idle mode

[2025-10-26 17:30:01] [INFO] [LAB] GITHUB SYNC (Optional Cloud Backup) - started
[2025-10-26 17:30:02] [INFO] [GITHUB BACKUP] Uploading manifest for session a1b2c3d4...
[2025-10-26 17:30:05] [INFO] [GITHUB BACKUP] ✓ Manifest uploaded: lab-backups/manifests/manifest-a1b2c3d4.json
[2025-10-26 17:30:06] [INFO] [GITHUB BACKUP] Uploading training summary for session a1b2c3d4...
[2025-10-26 17:30:08] [INFO] [GITHUB BACKUP] ✓ Summary uploaded: lab-backups/summaries/summary-a1b2c3d4.json
[2025-10-26 17:30:09] [INFO] [GITHUB BACKUP] Archiving models locally for session a1b2c3d4...
[2025-10-26 17:30:45] [INFO] [GITHUB BACKUP] ✓ Models archived locally: /path/to/artifacts/backups/models-a1b2c3d4.zip (5432.8 MB)
[2025-10-26 17:30:45] [INFO] [GITHUB BACKUP] Cleaning up 1 old archives (keeping last 3)
[2025-10-26 17:30:45] [DEBUG] [GITHUB BACKUP] Deleted old archive: models-old123.zip
[2025-10-26 17:30:45] [INFO] [LAB] Note: Terminal Mode will use local registry (no GitHub dependency)
```

## GitHub Backup Failure (Non-blocking)

```
[2025-10-26 17:30:02] [INFO] [GITHUB BACKUP] Uploading manifest for session a1b2c3d4...
[2025-10-26 17:30:05] [ERROR] [GITHUB BACKUP] Failed to upload manifest: API rate limit exceeded
[2025-10-26 17:30:05] [WARN] [LAB] GitHub backup failed (non-critical) - training completed successfully
```

## Key Features Demonstrated

### 1. Clear Visual Hierarchy
- Box drawing characters for important status displays
- INFO/DEBUG/WARN/ERROR log levels
- Emoji/checkmarks for success indicators

### 2. Time Zone Awareness
- All times displayed in ET (Eastern Time)
- Handles DST automatically
- Countdown shows days, hours, minutes

### 3. Non-Intrusive Monitoring
- Hourly updates (not spamming console)
- DEBUG level for routine checks
- Only INFO/WARN/ERROR for important events

### 4. Production-Ready Error Handling
- Health check failures logged with context
- Alerts sent for critical issues
- GitHub backup failures don't block training

### 5. User-Friendly Messages
- Clear status: IDLE, Active, Waiting
- Next training time prominently displayed
- Graceful exit instructions
- Market status context

## Configuration

### Verbose Logging (Optional)
Set environment variable to see all DEBUG messages:
```bash
export IDLE_VERBOSE_LOGGING=1
```

### GitHub Backup (Optional)
Set environment variable to enable cloud backups:
```bash
export GITHUB_BACKUP_TOKEN=ghp_your_token_here
```

## Notes

1. **Default Behavior**: Quiet operation with hourly INFO messages
2. **Debug Logging**: Available for troubleshooting
3. **Alerts**: Critical issues trigger alert notifications
4. **Non-Blocking**: GitHub backup failures don't stop training
5. **Resource Efficient**: Pre-warming reduces cold start time
6. **Timezone Safe**: All calculations use America/New_York timezone
