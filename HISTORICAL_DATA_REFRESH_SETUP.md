# Historical Data Refresh Automation Setup

This guide explains how to set up automated weekly refresh of historical data for the trading bot.

## Overview

The bot requires fresh historical data (90 days of ES and NQ bars) for training. The `refresh-historical-data.bat` script automates fetching new data weekly to keep the training data current.

## Requirements

- Python 3.8+ with TopstepX SDK installed
- TOPSTEPX_API_KEY and TOPSTEPX_USERNAME in .env file
- Windows (for Task Scheduler) or Linux/WSL (for cron)

## Windows Setup (Task Scheduler)

### Step 1: Test the Script

Before scheduling, verify the script works:

```batch
cd C:\path\to\QBot
refresh-historical-data.bat
```

You should see output indicating data is being fetched and saved.

### Step 2: Open Task Scheduler

1. Press `Win + R`
2. Type `taskschd.msc` and press Enter
3. Click "Create Task" in the right panel

### Step 3: Configure General Settings

**Name:** Trading Bot - Refresh Historical Data

**Description:** Fetch latest 90 days of ES and NQ historical bars for training

**Security Options:**
- Select "Run whether user is logged on or not"
- Check "Run with highest privileges"
- Configure for: Windows 10/11

### Step 4: Configure Triggers

Click "Triggers" tab, then "New..."

**Settings:**
- Begin the task: On a schedule
- Settings: Weekly
- Recur every: 1 week
- Days: Monday (recommended - after market closed Sunday)
- Start: 1:00:00 AM
- Stop task if it runs longer than: 1 hour
- Enabled: ✓

**Why Monday 1 AM?**
- Markets closed Sunday evening
- Avoids interference with live trading (weekday hours)
- Allows time for completion before market open Monday

### Step 5: Configure Actions

Click "Actions" tab, then "New..."

**Action:** Start a program

**Program/script:** 
```
C:\Windows\System32\cmd.exe
```

**Add arguments:**
```
/c "cd /d C:\path\to\QBot && refresh-historical-data.bat"
```

Replace `C:\path\to\QBot` with your actual bot directory.

**Start in:** (leave blank)

### Step 6: Configure Conditions

Click "Conditions" tab:

- ✓ Start only if the computer is on AC power
- ✗ Stop if the computer switches to battery power
- ✓ Wake the computer to run this task (optional)

### Step 7: Configure Settings

Click "Settings" tab:

- ✓ Allow task to be run on demand
- ✓ Run task as soon as possible after a scheduled start is missed
- ✓ Stop the task if it runs longer than: 1 hour
- If the task fails, restart every: 10 minutes
- Attempt to restart up to: 3 times
- If the task is already running, then the following rule applies: Do not start a new instance

### Step 8: Save and Test

1. Click "OK"
2. Enter your Windows password if prompted
3. Right-click the task and select "Run" to test
4. Check the "Last Run Result" column (should be 0x0 for success)

### Verification

After the first scheduled run, check:

1. **Log file**: Check for errors in console output
2. **Data files**: 
   - `data/historical/ES_90days.json` 
   - `data/historical/NQ_90days.json`
3. **File timestamps**: Should be recent
4. **File sizes**: ES ~50-150 MB, NQ ~50-150 MB
5. **Task history**: Task Scheduler shows successful completion

---

## Linux/WSL Setup (Cron)

### Step 1: Test the Script

```bash
cd /path/to/QBot
./refresh-historical-data.bat
```

Note: On Linux/WSL, you may need to create a bash version or use Python directly:

```bash
python3 fetch-and-save-historical-data.py
```

### Step 2: Create Wrapper Script (Optional)

Create `refresh-historical-data.sh`:

```bash
#!/bin/bash
cd /path/to/QBot
export REFRESH_MODE=incremental
python3 fetch-and-save-historical-data.py >> logs/historical-refresh.log 2>&1
```

Make it executable:

```bash
chmod +x refresh-historical-data.sh
```

### Step 3: Edit Crontab

```bash
crontab -e
```

Add this line:

```bash
# Refresh historical data every Monday at 1 AM ET
0 1 * * 1 /path/to/QBot/refresh-historical-data.sh
```

**Note:** Adjust for your timezone if not running in ET.

For Pacific Time (3 hours behind):
```bash
0 22 * * 0 /path/to/QBot/refresh-historical-data.sh
```
(Sunday 10 PM PT = Monday 1 AM ET)

### Step 4: Verify Cron Setup

List your cron jobs:

```bash
crontab -l
```

Check cron logs:

```bash
# Ubuntu/Debian
grep CRON /var/log/syslog

# CentOS/RHEL
grep CRON /var/log/cron
```

---

## Monitoring and Maintenance

### Check Data Freshness

Before each training session, the bot checks:

1. **File existence**: ES_90days.json and NQ_90days.json exist
2. **Date range**: Approximately 90 days of data
3. **Recency**: Data fetched within last 7 days
4. **Bar count**: ~630,000 bars per symbol (5-min bars, 90 days)

If data is stale (>7 days old), the bot will warn but may continue training.

### Manual Refresh

If needed, run manually:

**Windows:**
```batch
cd C:\path\to\QBot
refresh-historical-data.bat
```

**Linux/WSL:**
```bash
cd /path/to/QBot
python3 fetch-and-save-historical-data.py
```

### Troubleshooting

**Problem:** Task shows success but no new data

**Solution:** 
- Check that .env file has correct API credentials
- Verify Python and TopstepX SDK are installed
- Check logs for errors

**Problem:** Task fails with timeout

**Solution:**
- Increase timeout in Task Scheduler settings
- Check network connectivity
- Verify API is accessible

**Problem:** Data files are empty or corrupted

**Solution:**
- Delete existing files and run full refresh:
  ```bash
  set REFRESH_MODE=full
  refresh-historical-data.bat
  ```
- Check Python script output for validation errors

**Problem:** High disk space usage

**Solution:**
- Each symbol uses ~50-150 MB
- Total: ~100-300 MB (negligible)
- If logs are large, rotate or delete old log files

---

## Advanced Configuration

### Refresh Modes

The script supports two modes (set via `REFRESH_MODE` environment variable):

**Incremental (default):**
- Fetches only new bars since last update
- Fast (typically 1-2 minutes)
- Recommended for scheduled tasks

**Full:**
- Fetches entire 90-day window
- Slower (5-10 minutes)
- Use for first run or if data corrupted

Set in script or environment:

```bash
# Windows
set REFRESH_MODE=full
refresh-historical-data.bat

# Linux
export REFRESH_MODE=full
python3 fetch-and-save-historical-data.py
```

### Multiple Schedules

You can create multiple tasks for different purposes:

1. **Daily incremental** (before trading):
   - Time: 5:00 AM ET (before market open)
   - Mode: incremental
   - Quick data catch-up

2. **Weekly full refresh** (Sunday):
   - Time: 1:00 AM ET Monday
   - Mode: full
   - Complete data validation

### Email Notifications (Windows)

Configure Task Scheduler to send email on failure:

1. Task properties > Actions > New
2. Action: Send an e-mail
3. Configure SMTP settings
4. Set condition: "On failure"

### Slack Notifications

Integrate with monitoring:

1. Set `SLACK_WEBHOOK_URL` in .env
2. The bot will alert on data integrity failures
3. Check #trading-bot channel for alerts

---

## Best Practices

1. **Run during off-hours**: Avoid running during market hours to prevent interference
2. **Monitor regularly**: Check logs weekly to ensure data is fresh
3. **Test after changes**: After any system updates, test the scheduled task
4. **Keep credentials secure**: Ensure .env file has proper permissions (600 on Linux)
5. **Backup data**: Occasionally back up historical data files for disaster recovery

---

## Support

If you encounter issues:

1. Check logs in `logs/` directory
2. Review Python script output
3. Verify API credentials in .env
4. Check network connectivity
5. Consult TopstepX SDK documentation

For critical failures, the bot will:
- Log errors to console and files
- Send Slack alerts (if configured)
- Refuse to start training with stale data
