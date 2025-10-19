# Historical Data Acquisition Strategy

## Overview

Lab Mode training requires 90 days of historical 5-minute bars for ES and NQ futures to perform backtesting and validation. This document describes the strategy for acquiring and maintaining this historical data.

## Data Requirements

- **Symbols:** ES (E-mini S&P 500) and NQ (E-mini NASDAQ-100)
- **Timeframe:** 5-minute bars
- **History:** 90 days (approximately 6,989 bars per symbol)
- **Format:** JSON files saved to disk
- **Update Frequency:** Daily or weekly (markets are open)

## Acquisition Methods

### Option A: TopstepX API (Recommended)

#### Using Automated Scripts

The repository includes automated scripts for fetching historical data:

**Windows:**
```batch
refresh-historical-data.bat
```

**Unix/Linux:**
```bash
python3 fetch-and-save-historical-data.py
```

#### What the Script Does

1. Connects to TopstepX REST API using credentials from `.env`
2. Fetches last 90 days of 5-minute bars for ES and NQ
3. Saves data to JSON files:
   - `data/historical/ES_90days.json`
   - `data/historical/NQ_90days.json`
4. Includes incremental mode to fetch only new bars since last update

#### Configuration

The script reads from environment variables (`.env`):
```env
TOPSTEP_API_KEY=your_api_key_here
TOPSTEP_API_SECRET=your_api_secret_here
TOPSTEP_BASE_URL=https://api.topstepx.com
```

#### Timing Constraints

**⚠️ IMPORTANT: TopstepX API Availability**

- **Markets Open (Monday-Friday):** ✅ API available, data fetching works
- **Markets Closed (Saturday-Sunday):** ❌ API may timeout or return stale data
- **Sunday Lab Training:** ⚠️ Data must be fetched BEFORE Sunday training session

**Recommended Schedule:**

1. **Daily Updates (Monday-Friday):**
   - Run `refresh-historical-data.bat` during market hours (9:30 AM - 4:00 PM ET)
   - Or run overnight Sunday-Thursday (6:00 PM - 5:00 AM ET)
   - Script uses incremental mode to fetch only new bars

2. **Weekly Preparation (Friday):**
   - Run full refresh on Friday afternoon (4:00 PM - 5:00 PM ET)
   - Ensures complete 90-day dataset is available for Sunday training
   - This is the "weekend insurance" - data is ready before markets close

3. **Sunday Training:**
   - Lab Mode reads cached JSON files from disk
   - No API calls needed during training session
   - Uses data prepared on Friday/Saturday

#### Error Handling

If API is unavailable (Sunday timeout):
- Lab Mode uses last cached dataset (Friday's data)
- Training proceeds with 89 days instead of 90 days
- Logs warning about stale data
- System continues to function (graceful degradation)

### Option B: Historical Data Seed Files (Fallback)

If TopstepX API is unavailable, use pre-populated seed files:

**Location:**
- `data/historical/seed/ES_90day_seed.json` (3,529 bars)
- `data/historical/seed/NQ_90day_seed.json` (3,460 bars)

**Usage:**
1. Copy seed files to main directory:
   ```bash
   cp data/historical/seed/ES_90day_seed.json data/historical/ES_90days.json
   cp data/historical/seed/NQ_90day_seed.json data/historical/NQ_90days.json
   ```

2. Lab Mode automatically uses these files if API data is missing

**Limitations:**
- Data becomes stale over time (not real-time)
- Should only be used as emergency fallback
- Update seed files monthly from production data

## Data Format

### JSON Structure

```json
{
  "symbol": "ES",
  "timeframe": "5min",
  "bars": [
    {
      "timestamp": "2025-07-21T09:30:00Z",
      "open": 5532.50,
      "high": 5535.25,
      "low": 5531.75,
      "close": 5534.00,
      "volume": 12543
    },
    ...
  ],
  "metadata": {
    "count": 6989,
    "startDate": "2025-04-21",
    "endDate": "2025-07-21",
    "source": "TopstepX",
    "fetchedAt": "2025-07-21T17:00:00Z"
  }
}
```

### Storage Location

- **Production Data:** `data/historical/ES_90days.json`, `data/historical/NQ_90days.json`
- **Seed Files:** `data/historical/seed/ES_90day_seed.json`, etc.
- **Backup Archives:** `data/historical/archive/` (weekly snapshots)

## Integration with Lab Mode

### How Lab Mode Uses Historical Data

1. **Load Phase (5 minutes):**
   - `HistoricalTrainingOrchestrator` starts training session
   - Reads JSON files via `IHistoricalDataBridgeService`
   - Validates data integrity (bar count, date range, gaps)

2. **Replay Phase (2 hours):**
   - `EnhancedBacktestLearningService` processes bars sequentially
   - Simulates live trading conditions
   - Generates synthetic experiences for training

3. **Training Phase (2-3 hours):**
   - Uses replayed experiences + live experiences
   - Trains CVaR-PPO, Neural UCB, LSTM models
   - Historical data not accessed directly during training

### Data Dependencies

Lab Mode training depends on:
1. **Live Experiences:** Collected from Terminal Mode (last 7 days)
2. **Historical Bars:** 90-day dataset from TopstepX (6,989 bars)
3. **Combination:** Training uses both sources for robustness

**Minimum Requirements:**
- At least 10,000 live experiences OR
- At least 6,000 historical bars OR
- Combination of both (preferred)

## Maintenance Tasks

### Daily Maintenance (Automated)

**Recommended:** Schedule as Windows Task or Linux cron job

**Windows Task Scheduler:**
```powershell
# Run daily at 5:30 PM ET (after market close)
schtasks /create /tn "QBot Historical Data Refresh" /tr "C:\path\to\refresh-historical-data.bat" /sc daily /st 17:30
```

**Linux Cron:**
```bash
# Run daily at 5:30 PM ET (after market close)
30 17 * * 1-5 cd /home/qbot && ./refresh-historical-data.bat
```

### Weekly Backup

**Friday Snapshot:**
```bash
# Backup current dataset before weekend
mkdir -p data/historical/archive
cp data/historical/ES_90days.json data/historical/archive/ES_90days_$(date +%Y%m%d).json
cp data/historical/NQ_90days.json data/historical/archive/NQ_90days_$(date +%Y%m%d).json

# Keep only last 4 weeks
find data/historical/archive -name "*.json" -mtime +28 -delete
```

### Monthly Seed Update

**Update seed files from production:**
```bash
# Copy current production data to seed files (for fallback)
cp data/historical/ES_90days.json data/historical/seed/ES_90day_seed.json
cp data/historical/NQ_90days.json data/historical/seed/NQ_90day_seed.json
```

## Troubleshooting

### Issue: API Timeout on Sunday

**Symptom:** `refresh-historical-data.bat` fails with timeout error on Sunday

**Solution:**
- This is expected - TopstepX API is unavailable when markets are closed
- Use Friday's cached data (already on disk)
- Lab Mode will log warning but continue training
- Next week: Schedule refresh for Monday-Friday only

### Issue: Stale Data Warning

**Symptom:** Lab Mode logs "Historical data is X days old"

**Solution:**
- Run `refresh-historical-data.bat` during market hours (Monday-Friday)
- Check `.env` file has valid API credentials
- Verify network connectivity to TopstepX API
- If persistent, use seed files as fallback

### Issue: Missing Historical Files

**Symptom:** Lab Mode fails with "Historical data files not found"

**Solution:**
1. Check if files exist:
   ```bash
   ls -la data/historical/ES_90days.json
   ls -la data/historical/NQ_90days.json
   ```

2. If missing, run refresh script:
   ```bash
   python3 fetch-and-save-historical-data.py
   ```

3. If script fails, use seed files:
   ```bash
   cp data/historical/seed/*.json data/historical/
   ```

### Issue: Corrupted JSON Files

**Symptom:** Lab Mode fails with "JSON parse error"

**Solution:**
1. Validate JSON structure:
   ```bash
   python3 -m json.tool data/historical/ES_90days.json > /dev/null
   ```

2. If corrupted, restore from archive:
   ```bash
   cp data/historical/archive/ES_90days_*.json data/historical/ES_90days.json
   ```

3. If no archive, fetch fresh data:
   ```bash
   python3 fetch-and-save-historical-data.py
   ```

## Best Practices

1. **Always Run Refresh on Friday:** Ensures weekend availability
2. **Enable Incremental Mode:** Faster updates, less API load
3. **Monitor Data Age:** Lab Mode should log data freshness
4. **Keep Backups:** Weekly archives for disaster recovery
5. **Update Seed Files:** Monthly refresh for fallback strategy
6. **Test Fallback Path:** Periodically verify seed files work

## API Rate Limits

TopstepX API rate limits:
- **Requests per minute:** 60
- **Daily requests:** Unlimited
- **Burst allowance:** 10 requests/second

The `fetch-and-save-historical-data.py` script respects these limits:
- Uses single request per symbol (2 total)
- No parallel requests
- Implements exponential backoff on errors

## Data Quality Checks

Lab Mode validates historical data:

1. **Bar Count:** Should be ~6,989 bars (90 days × 77.7 bars/day)
2. **Date Range:** Should span exactly 90 days
3. **Gaps:** Identifies missing bars (weekends, holidays expected)
4. **Outliers:** Detects abnormal price movements
5. **Volume:** Validates non-zero volume for liquid contracts

If validation fails:
- Lab Mode logs warnings
- Training continues with available data
- Marks training session with "data quality issue" flag

## Future Enhancements

### Planned Improvements

1. **Real-time Validation:** Check data quality during fetch
2. **Automatic Retry:** Retry failed API calls with backoff
3. **Multi-source Aggregation:** Combine TopstepX + backup sources
4. **Compression:** Store historical data in compressed format
5. **Cloud Backup:** Sync historical data to S3/Azure Blob

### Nice-to-Have

- **Web Dashboard:** Monitor data freshness and quality
- **Email Alerts:** Notify on stale data or failed refreshes
- **API Health Check:** Test TopstepX availability before training
- **Data Comparison:** Validate TopstepX data against other sources

## Summary

**Key Takeaways:**

1. ✅ **Use TopstepX API** for automated daily updates (Monday-Friday)
2. ⚠️ **Run on Friday** before market close to prepare for Sunday training
3. 🔄 **Fallback to seed files** if API unavailable
4. 📦 **Keep weekly backups** for disaster recovery
5. 🚫 **Don't run on Sunday** - API times out when markets closed

**Next Steps:**

1. Set up automated daily refresh (Monday-Friday, 5:30 PM ET)
2. Create weekly backup script (Friday, 4:30 PM ET)
3. Test fallback path with seed files
4. Monitor data freshness in Lab Mode logs
5. Update seed files monthly from production data

---

**Document Version:** 1.0  
**Last Updated:** October 19, 2025  
**Owner:** Lab Mode Infrastructure Team
