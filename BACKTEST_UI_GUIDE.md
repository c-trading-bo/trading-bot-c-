# Backtest Live Tick Replay UI Guide

## Overview

The backtest system now includes a **Live Tick Replay UI** that displays market data playback in real-time with a clean, visual interface similar to live trading. This replaces the previous verbose logging with an interactive display.

## Features

### 📊 Depth of Market (DOM)
- Last 10 ticks displayed with timestamps
- Bid/Ask prices and spreads
- Volume for each tick
- Tick direction indicators (↑ ↓ →)

### 🤖 Bot Brain Thinking
- Real-time strategy analysis
- Signal detection with entry/stop/target details
- Risk/Reward calculations
- Entry signals with dollar amounts
- Order submission status

### 📍 Open Position Panel (NEW!)
- Live position tracking
- Real-time P&L updates with percentage
- Current price with direction indicator
- Stop loss distance (points and dollars)
- Target distance (points and dollars)
- Visual P&L indicator (🟢 profit / 🔴 loss)

### 💼 Account & Stats Panel
- Current equity and daily P&L
- Open and realized P&L
- Trade statistics (total, winners, win rate)
- Best trade tracking

## Configuration

### Enable/Disable UI

**Option 1: Environment Variable (Recommended)**
```bash
export ENABLE_BACKTEST_UI=1  # Enable
export ENABLE_BACKTEST_UI=0  # Disable
```

**Option 2: Configuration File**
Edit `appsettings.backtest.json`:
```json
{
  "BacktestOptions": {
    "EnableTickReplayUI": true,
    "ReplaySpeed": 1
  }
}
```

### Replay Speed

Control how fast ticks are replayed:
- `1` = Real-time (default)
- `2` = 2x speed
- `5` = 5x speed
- `10` = 10x speed

Set in `appsettings.backtest.json`:
```json
{
  "BacktestOptions": {
    "ReplaySpeed": 1
  }
}
```

## Running Backtest with UI

### Windows (PowerShell)
```powershell
# Set environment variable
$env:ENABLE_BACKTEST_UI = "1"
$env:BACKTEST_MODE = "1"

# Run backtest
./start-bot-with-backtest.ps1
```

### Linux/Mac (Bash)
```bash
# Set environment variable
export ENABLE_BACKTEST_UI=1
export BACKTEST_MODE=1

# Run backtest
./start-bot-with-backtest.sh
```

## Example Output

```
╔══════════════════════════════════════════════════════════════════════╗
║              ES BACKTEST - LIVE TICK REPLAY                          ║
║              Oct 15, 2025  14:32:47 CT                               ║
╚══════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────┐
│ DEPTH OF MARKET (Last 10 Ticks)                    Speed: 1x Real   │
├─────────────────────────────────────────────────────────────────────┤
│ 14:32:47.892   5,848.00   ↑   VOL: 12      BID: 5,847.75  ASK: 5,848.00
│ 14:32:47.654   5,847.75   ↓   VOL: 8       BID: 5,847.50  ASK: 5,847.75
│ 14:32:47.432   5,848.00   ↑   VOL: 15      BID: 5,847.75  ASK: 5,848.00
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ 🤖 BOT BRAIN THINKING...                                            │
├─────────────────────────────────────────────────────────────────────┤
│ 🧠 Analyzing tick flow...                                           │
│ 📊 Pattern detected: Bull Flag (Confidence: 78%)                    │
│ 📈 Strategy: S3-MeanReversion (Confidence: 85%)                     │
│ ⚖️  Risk/Reward: 2.8:1 (Good setup)                                 │
│                                                                      │
│ ⏳ WATCHING... Waiting for entry trigger                            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┬───────────────────────────────────┐
│ 💼 ACCOUNT                      │ 📊 TODAY'S STATS                  │
├─────────────────────────────────┼───────────────────────────────────┤
│ Equity: $52,340 (+4.68%)        │ Trades: 3                         │
│ Open P&L: $0.00                 │ Winners: 2 (66.7%)                │
│ Daily P&L: +$450.00             │ P&L: +$450                        │
│ Buying Power: $52,340           │ Best: +$275                       │
└─────────────────────────────────┴───────────────────────────────────┘

🎬 [TICK] Price: 5,848.00 ↑ | Bot sees this tick and analyzes...
```

## Disabling Verbose Logging

When the UI is enabled, most verbose logging is automatically suppressed. The system only logs:
- Critical errors
- Important state changes
- Final backtest results

If you need full logging for debugging, disable the UI:
```bash
export ENABLE_BACKTEST_UI=0
```

## Troubleshooting

### UI Not Showing
1. Verify `ENABLE_BACKTEST_UI=1` is set
2. Check `appsettings.backtest.json` has `EnableTickReplayUI: true`
3. Ensure running in backtest mode (`BACKTEST_MODE=1`)

### Too Much Logging
1. Set `ENABLE_BACKTEST_UI=1`
2. Verify logging levels in `appsettings.backtest.json`:
   - Default: Warning
   - TopstepX: Warning
   - Microsoft: Warning

### TopstepX Connection Attempts
When in backtest mode with UI enabled, the system should NOT attempt to connect to TopstepX live API. All data comes from historical sources.

If you see connection attempts:
1. Verify `BACKTEST_MODE=1` is set
2. Check that historical data provider is configured
3. Review logs for TopstepX initialization attempts

## Architecture

The backtest UI system consists of:

1. **BacktestConsoleUI** - Renders the formatted console output
2. **BacktestTickReplayService** - Manages tick-by-tick playback
3. **BacktestHarnessService** - Integrates UI with backtest execution
4. **EnhancedBacktestLearningService** - Suppresses logs when UI enabled

## Performance Notes

- UI rendering occurs every 5 ticks by default
- Rendering also triggers on trading signals
- Replay delay scales with `ReplaySpeed` setting
- Minimum delay: 10ms per tick
- Default delay: 100ms per tick (at 1x speed)

## See Also

- `appsettings.backtest.json` - Configuration file
- `src/Backtest/UI/BacktestConsoleUI.cs` - UI implementation
- `src/Backtest/BacktestHarnessService.cs` - Backtest harness
