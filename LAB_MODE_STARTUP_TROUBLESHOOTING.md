# 🔧 Lab Mode Startup Troubleshooting Guide

## Overview

This guide helps diagnose and fix issues when Lab Mode fails to load historical data and start training.

## Common Issues and Solutions

### Issue 1: Historical Data Files Missing

**Symptoms:**
- Lab mode starts but fails immediately
- Error: "Historical data files are missing"
- Training doesn't begin

**Solution:**
```bash
# Check if data files exist
ls -lh data/historical/*.json

# If missing, fetch data using Python script
python fetch-and-save-historical-data.py

# Or on Windows
python.exe fetch-and-save-historical-data.py
```

**Expected Files:**
- `data/historical/ES_90days.json` (required)
- `data/historical/NQ_90days.json` (required)
- `data/historical/ES_1m_90days.json` (optional, for multi-timeframe)
- `data/historical/NQ_1m_90days.json` (optional, for multi-timeframe)

### Issue 2: Empty or Corrupted JSON Files

**Symptoms:**
- Files exist but have 0 bytes
- Error: "Historical data file is empty"
- Error: "Historical data file contains no bars"

**Solution:**
```bash
# Check file sizes
ls -lh data/historical/

# If files are 0 bytes or very small (< 100 KB), delete and re-fetch
rm data/historical/*_90days.json
python fetch-and-save-historical-data.py
```

**Validation:**
```bash
# Check JSON structure
head -20 data/historical/ES_90days.json

# Should see:
# {
#   "symbol": "ES",
#   "bars": [
#     {
#       "timestamp": "...",
#       "open": ...,
#       ...
```

### Issue 3: Python Not Found

**Symptoms:**
- Error: "Python executable not found"
- Data fetch fails with "command not found"

**Solution:**

**Linux/Mac:**
```bash
# Check Python installation
which python3
python3 --version

# If not installed, install Python 3
# Ubuntu/Debian:
sudo apt-get install python3 python3-pip

# macOS:
brew install python3
```

**Windows:**
```powershell
# Check Python installation
python --version

# If not installed, download from python.org
# Or install via winget:
winget install Python.Python.3.12
```

**Add to PATH:**
Make sure Python is in your system PATH environment variable.

### Issue 4: Training Starts with No Data

**Symptoms:**
- Lab mode starts but shows "0 bars loaded"
- Training proceeds but immediately fails
- No error about missing files

**Fixed in Latest Version:**
The bot now validates data BEFORE starting training and will fail fast with clear error messages if data is missing or invalid.

**Verification:**
```bash
# Check logs for validation messages
# You should see:
# [LAB] ✅ Loaded 4928 5m bars for ES from ES_90days.json
# [LAB] ✅ Multi-timeframe: Loaded 17280 1m bars for ES from ES_1m_90days.json
# [LAB] 📊 MULTI-TIMEFRAME DATA LOADED - Total: 9856 5m bars + 34560 1m bars
```

### Issue 5: Insufficient Data for Training

**Symptoms:**
- Warning: "Low bar count"
- Training quality is poor
- Models don't converge

**Solution:**
Ensure you have at least 1000 bars (approximately 3 days of 5-minute data):

```bash
# Check bar count in JSON files
grep -o "timestamp" data/historical/ES_90days.json | wc -l

# Should be > 1000 for quality training
```

If bar count is low, fetch more data:
```bash
# Modify fetch script to get more days (default is 90 days)
python fetch-and-save-historical-data.py
```

## Pre-Flight Checklist

Before launching Lab Mode, verify:

- [ ] Python is installed and in PATH
- [ ] Data directory exists: `data/historical/`
- [ ] Required files exist: `ES_90days.json` and `NQ_90days.json`
- [ ] Files have reasonable size (> 100 KB each)
- [ ] JSON files have valid structure with "bars" array
- [ ] Bar count is sufficient (> 1000 bars per symbol)

## Launch Commands

### Linux/Mac/WSL
```bash
# Force immediate training (any day)
FORCE_LAB_NOW=1 dotnet run --project src/UnifiedOrchestrator

# Or use launch script
./launch-lab-auto.ps1
```

### Windows PowerShell
```powershell
# Force immediate training (any day)
$env:FORCE_LAB_NOW="1"
dotnet run --project src/UnifiedOrchestrator

# Or use launch script
.\launch-lab-auto.ps1
```

## Automatic Fixes

The latest version includes these automatic fixes:

1. **Pre-Flight Validation**: Checks all required files exist before loading
2. **Retry Mechanism**: Attempts to fetch data 3 times with exponential backoff
3. **Data Validation**: Verifies JSON structure and bar count
4. **Fail Fast**: Stops immediately with clear errors if data cannot be loaded
5. **Helpful Messages**: Provides specific instructions on how to fix issues

## Error Messages Explained

### ❌ "Historical data files are missing and could not be fetched"
**Cause:** Required JSON files don't exist and Python script failed to fetch them.
**Fix:** Manually run `python fetch-and-save-historical-data.py` and check for errors.

### ❌ "Historical data file is empty"
**Cause:** JSON file exists but has no content.
**Fix:** Delete the file and re-fetch: `rm data/historical/ES_90days.json && python fetch-and-save-historical-data.py`

### ❌ "Historical data file contains no bars"
**Cause:** JSON file exists but the "bars" array is empty.
**Fix:** Check API credentials and re-fetch data.

### ❌ "No historical data loaded. Cannot proceed with training."
**Cause:** Data loading failed for all symbols.
**Fix:** Follow the troubleshooting steps above for file validation and Python setup.

## Success Indicators

When everything is working correctly, you should see:

```
[LAB] 📊 Loading historical data for training session...
[LAB] ✅ Loaded 4928 5m bars for ES from ES_90days.json
[LAB] ✅ Multi-timeframe: Loaded 17280 1m bars for ES from ES_1m_90days.json
[LAB] ✅ Loaded 4928 5m bars for NQ from NQ_90days.json
[LAB] ✅ Multi-timeframe: Loaded 17280 1m bars for NQ from NQ_1m_90days.json
[LAB] 📊 MULTI-TIMEFRAME DATA LOADED - Total: 9856 5m bars + 34560 1m bars
[LAB] ═══════════════════════════════════════════════════════
[LAB] 🎓 SUNDAY TRAINING PIPELINE STARTED
```

## Contact Support

If issues persist after following this guide:
1. Check the repository issues on GitHub
2. Review logs in `logs/` directory
3. Ensure you have the latest version of the code
4. File a new issue with full error logs

## Related Documentation

- `LAB_MODE_TRAINING_GUIDE.md` - Complete training guide
- `LAB_MODE_QUICK_REF.md` - Quick reference card
- `QUICK_START_BOT_LAUNCH.md` - Bot launch guide
