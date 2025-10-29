# Backtest Mode - Quick Launch Guide

## Overview

The bot has **3 modes**: Terminal (live trading), Lab (training), and **Backtest** (strategy testing with UI).

Backtest mode allows you to:
- ✅ Test strategies on historical data **without API access**
- ✅ See tick-by-tick replay with professional DOM (Depth of Market)
- ✅ Watch bot decisions in real-time as if market was live
- ✅ Track P&L, position, and account stats
- ✅ **No log spam** - clean UI-only display

## Quick Start

### Option 1: Interactive Menu (Recommended for First Time)

```bash
cd src/UnifiedOrchestrator
dotnet run
```

Then select **option 3** from the menu:
```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    TopstepX Trading Bot - Mode Selection                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  [1] Terminal Mode (Live Trading)                                             ║
║  [2] Lab Mode (Historical Training)                                           ║
║  [3] Backtest Mode (Strategy Testing)         <-- Choose this                 ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝
Select mode [1-3]: 3
```

### Option 2: Direct Launch (Automated/Scripts)

```bash
# Set environment variables
export BACKTEST_MODE=1
export ENABLE_BACKTEST_UI=1
export SKIP_MODE_PROMPT=1
export DRY_RUN=1

# Optional: Configure backtest parameters
export BACKTEST_SYMBOL=ES          # ES or NQ
export BACKTEST_MODEL=CVaR-PPO     # Model to test
export BACKTEST_DAYS=1             # How many days back

# Run
cd src/UnifiedOrchestrator
dotnet run
```

Or use the test script:
```bash
./test-backtest-mode.sh
```

## What You'll See

The backtest UI displays:

```
╔══════════════════════════════════════════════════════════════════════╗
║              ES BACKTEST - LIVE TICK REPLAY                         ║
║              Jan 01, 2024  09:30:00 CT                              ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│ DEPTH OF MARKET (Last 10 Ticks)                    Speed: 1x Real   │
├─────────────────────────────────────────────────────────────────────┤
│ 09:30:05.000   4700.50   ↑   VOL: 3      BID:  4700.25  ASK:  4700.75│
│ 09:30:04.000   4700.25   ↑   VOL: 2      BID:  4700.00  ASK:  4700.50│
│ 09:30:03.000   4700.00   →   VOL: 1      BID:  4699.75  ASK:  4700.25│
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 🤖 BOT BRAIN THINKING...                                            │
├─────────────────────────────────────────────────────────────────────┤
│ 🚨 [SIGNAL] BOT DECISION: ENTER LONG ES!                           │
│ ├─ Entry: 4700.50                                                   │
│ ├─ Stop: 4695.00 (-5.5 pts = -$275)                                │
│ ├─ Target: 4711.00 (+10.5 pts = +$525)                             │
│ ├─ Risk/Reward: 1.91:1                                              │
│ ├─ Confidence: 72%                                                  │
│ └─ Reason: CVaR-PPO - Strong momentum signal                       │
│                                                                      │
│ ⚡ [ORDER] Submitting MARKET BUY 1 ES...                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ OPEN POSITION                                                       │
├─────────────────────────────────────────────────────────────────────┤
│ ES LONG 1 @ 4700.50                                                │
│ Current Price: 4702.25  ↑                                          │
│ P&L: +$87.50 (+1.24%)  🟢                                          │
│ Stop: 4695.00 (-5.5 pts away)                                      │
│ Target: 4711.00 (+8.8 pts away)                                    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┬───────────────────────────────────┐
│ 💼 ACCOUNT                      │ 📊 TODAY'S STATS                  │
├─────────────────────────────────┼───────────────────────────────────┤
│ Equity: $100,087 (+0.09%)       │ Trades: 3                         │
│ Open P&L: $87.50                │ Winners: 2 (66.7%)                │
│ Daily P&L: +$87.50              │ P&L: +87                          │
│ Buying Power: $100,087          │ Best: +150                        │
└─────────────────────────────────┴───────────────────────────────────┘

🎬 [TICK] Price: 4702.25 ↑ | Bot sees this tick and analyzes...
```

## Key Features

### ✅ No API Required
- Uses local historical data from `datasets/quotes/`
- Falls back gracefully if API is unavailable
- Sample data included for ES and NQ

### ✅ Clean UI (No Log Spam)
- All logs redirected to file: `logs/backtest-*.log`
- Only the UI writes to console
- In-place rendering (no scrolling)

### ✅ Real-Time Replay
- Ticks play at configurable speed (default 1x = real-time)
- Bot sees each tick and makes decisions
- Just like watching live trading, but with historical data

### ✅ Position Tracking
- Opens/closes positions automatically
- Shows real-time P&L
- Tracks stop loss and target distances

## Configuration

### Backtest Parameters (via environment variables)

```bash
# Core backtest settings
BACKTEST_MODE=1                # Enable backtest mode
ENABLE_BACKTEST_UI=1           # Show visual UI (recommended)
SKIP_MODE_PROMPT=1             # Skip interactive menu

# Backtest parameters
BACKTEST_SYMBOL=ES             # Symbol: ES or NQ
BACKTEST_MODEL=CVaR-PPO        # Model: CVaR-PPO, NeuralUCB, etc.
BACKTEST_DAYS=7                # Days of history (1-365)

# Safety (always enabled in backtest)
DRY_RUN=1                      # Always set - no real orders
```

### Data Sources

Backtest mode tries data sources in this order:
1. **Features data**: `datasets/features/{symbol}_features.json`
2. **Quotes data**: `datasets/quotes/{symbol}_quotes.json` ✅ Included
3. **TopstepX API**: Falls back to API if available (optional)

Sample data is provided in `datasets/quotes/`:
- `es_quotes.json` - 500 ticks of ES data
- `nq_quotes.json` - 500 ticks of NQ data

To generate more sample data:
```bash
python3 generate-sample-ticks.py
```

## Troubleshooting

### No data available error
```bash
# Generate sample tick data
python3 generate-sample-ticks.py

# Or convert existing data
python3 convert-tick-data.py
```

### Logs appearing on screen
Check that `ENABLE_BACKTEST_UI=1` is set. All logs go to:
```
logs/backtest-YYYYMMDD-HHMMSS.log  (e.g., backtest-20241029-093000.log)
```

### UI not rendering
- Make sure terminal supports UTF-8
- Try running in a full terminal window (not embedded)
- Check that `ENABLE_BACKTEST_UI=1` is set

## Advanced Usage

### Custom Replay Speed
Edit `appsettings.backtest.json`:
```json
{
  "BacktestOptions": {
    "ReplaySpeed": 2,        // 2x speed
    "EnableTickReplayUI": true
  }
}
```

### Different Models
Test different models by setting:
```bash
export BACKTEST_MODEL=NeuralUCB  # or LSTM, CVaR-PPO, etc.
```

### Longer History
```bash
export BACKTEST_DAYS=30  # Test on 30 days of data
```

Note: You'll need to generate more sample data or use real API data for longer periods.

## Log Files

All diagnostic logs are saved to `logs/backtest-*.log` for debugging:
- Service initialization
- Model loading
- Decision making
- Trade execution

You can tail the most recent log file in another terminal:
```bash
# Tail the most recent backtest log
tail -f $(ls -t logs/backtest-*.log | head -1)

# Or use wildcard if only one backtest is running
tail -f logs/backtest-*.log
```

## Next Steps

After running backtest mode:
1. Review performance metrics in the final summary
2. Check `reports/bt/` for detailed reports
3. Analyze bot decision patterns
4. Tune model parameters if needed
5. Test different strategies

## Key Differences from Terminal Mode

| Feature | Terminal Mode | Backtest Mode |
|---------|---------------|---------------|
| API Connection | ✅ Required | ❌ Not needed |
| Real Orders | ✅ Yes (if DRY_RUN=0) | ❌ Never |
| Data Source | 🌐 Live WebSocket | 💾 Local files |
| Speed | ⏱️ Real-time only | ⚡ Configurable |
| UI | 📊 Dashboard | 🎬 Tick replay |
| Logs | 📝 Console + file | 📝 File only |

---

**Ready to test your strategies?**
```bash
./test-backtest-mode.sh
```
