# Historical Training Mode - Feature Demo

## Mode Selection UI

When the bot launches, users see an interactive prompt:

```
================================================================================
                    🎯 TRADING MODE SELECTION 🎯
================================================================================

Please select your trading mode:

  [1] 📊 HISTORICAL TRAINING MODE
      - Replay 90 days of historical data at high speed
      - Train models on simulated trading
      - No API calls, no real money
      - Comprehensive terminal audit logs

  [2] 🚀 LIVE MODE
      - Real trading with TopstepX API
      - ⚠️  REAL MONEY AT RISK ⚠️
      - Requires DRY_RUN=0 in .env

  [3] 📝 DRY-RUN MODE (Paper Trading)
      - Real live market data
      - Simulated trades (no real money)
      - Safe for testing strategies

  [Q] Quit

Enter your choice [1-3 or Q]:
```

## Selection Flow

### Option 1 - Historical Training Mode
When selected, the bot:
- Sets `HISTORICAL_MODE=1`
- Forces `DRY_RUN=1` (safety)
- Displays: `✅ Historical Training Mode selected`
- Displays: `📊 Bot will replay historical data and train models`

### Option 2 - Live Mode
When selected, the bot:
- Sets `HISTORICAL_MODE=0`
- Checks current `DRY_RUN` setting
- If `DRY_RUN=1`, prompts: `Do you want to disable DRY_RUN and trade with REAL MONEY? [yes/NO]:`
  - Requires typing `yes` to enable live trading
  - Any other input keeps dry-run enabled
- Displays appropriate warning about real money at risk

### Option 3 - Dry-Run Mode
When selected, the bot:
- Sets `HISTORICAL_MODE=0`
- Sets `DRY_RUN=1`
- Displays: `✅ Dry-Run Mode (Paper Trading) selected`
- Displays: `📝 Bot will use live data but simulate trades`

## Automation Mode

For CI/CD and automation, set `SKIP_MODE_PROMPT=1` in `.env` to bypass the interactive prompt and use environment variable settings directly.

## Service Registration

When `HISTORICAL_MODE=1`, the bot displays:

```
✅ [HISTORICAL-MODE] Historical replay orchestrator ENABLED
   📊 Bot will replay 90 days of historical data at high speed
   🎓 Models will be trained on simulated trading
   📝 Complete audit trail will be logged to terminal
```

## Implementation Status

✅ Interactive mode selection prompt implemented
✅ Three-way mode selection (Historical/Live/Dry-Run)
✅ Safety confirmations for live mode
✅ Automation-friendly skip option
✅ HistoricalReplayOrchestrator service created
✅ Service conditionally registered based on mode
✅ Bar loading from HistoricalDataSeedService
✅ Simulated execution with SlippageLatencyModel
✅ Progress tracking and metrics
✅ Terminal audit logging

## Testing

The implementation has been tested and verified:
- ✅ Build passes successfully
- ✅ Service registration works correctly
- ✅ Historical mode is detected and enabled
- ✅ Bot launches without errors in historical mode
