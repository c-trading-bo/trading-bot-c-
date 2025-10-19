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

  [2] 📝 DRY-RUN MODE (Paper Trading)
      - Real live market data from TopstepX API
      - Simulated trades (no real money)
      - Safe for testing strategies
      - Models learn from paper trades

  [3] 🚀 LIVE MODE
      - Real trading with TopstepX API
      - ⚠️  REAL MONEY AT RISK ⚠️
      - Real orders sent to broker
      - Requires explicit YES confirmation

  [4] Exit

Enter your choice [1-4]:
```

## Selection Flow

### Option 1 - Historical Training Mode
When selected, the bot:
- Sets `HISTORICAL_MODE=1`
- Forces `DRY_RUN=1` (safety)
- Displays: `✅ Historical Training Mode selected`
- Displays: `📊 Bot will replay 90 days of historical data at high speed`
- Displays: `🎓 Models will be trained on simulated trading`

### Option 2 - Dry-Run Mode
When selected, the bot:
- Sets `HISTORICAL_MODE=0`
- Sets `DRY_RUN=1`
- Displays: `✅ Dry-Run Mode (Paper Trading) selected`
- Displays: `📝 Bot will connect to TopstepX API for live data`
- Displays: `💡 Trades will be simulated (paper trading)`

### Option 3 - Live Mode
When selected, the bot:
- Shows warning: `⚠️  WARNING: You are about to enable LIVE TRADING with REAL MONEY`
- Prompts: `Type YES in all capitals to confirm live trading:`
- If user types `YES` exactly:
  - Sets `HISTORICAL_MODE=0`
  - Sets `DRY_RUN=0`
  - Displays: `🚨 LIVE TRADING ENABLED - REAL MONEY AT RISK 🚨`
  - Displays: `💰 Real orders will be placed`
- If user types anything else:
  - Displays: `❌ Live trading NOT enabled (you must type YES exactly)`
  - Returns to menu

### Option 4 - Exit
When selected, the bot:
- Displays: `👋 Exiting...`
- Exits immediately

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
✅ Four-option menu (Historical/Dry-Run/Live/Exit)
✅ Safety confirmations for live mode (requires "YES" in capitals)
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
- ✅ Menu displays in correct order (Historical, Dry-Run, Live, Exit)
- ✅ Live mode requires "YES" confirmation in capitals
