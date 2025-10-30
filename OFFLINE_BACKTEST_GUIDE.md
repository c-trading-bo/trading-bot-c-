# Offline Backtest Mode - Complete Guide

## Overview

The QBot trading system includes a **fully offline backtest mode** that allows you to test trading strategies using saved historical market data without requiring any API connection. This mode is perfect for:

- Testing strategies without live market access
- Validating bot behavior on historical data
- Developing and debugging trading logic offline
- Demonstrating bot capabilities without credentials

## Architecture

### Data Flow (Offline Mode)
```
Saved Historical Bars (JSON Files)
    ↓
LocalQuotesProvider
    ↓
BacktestHarnessService
    ↓
UnifiedDecisionRouter / Fallback Logic
    ↓
ExecutionSimulator
    ↓
Trade Execution (Simulated)
    ↓
Metrics & Reports
```

### Data Sources (Priority Order)
1. **Features Data** (if available): `datasets/features/{symbol}_features.json`
2. **Quotes Data** ✅ **PRIMARY**: `datasets/quotes/{symbol}_quotes.json`
3. **TopstepX API** (fallback, requires connection - NOT used in offline mode)

The offline backtest uses **saved historical bars** stored in JSON format, eliminating any need for API connectivity.

## Quick Start

### Method 1: Using the Convenience Script (Recommended)

```bash
./run-offline-backtest.sh
```

This script automatically:
- Sets all required environment variables
- Configures the bot for offline operation
- Runs the backtest with informative logging
- Shows trade execution in the console

### Method 2: Manual Execution

```bash
# Set environment variables
export BACKTEST_MODE=1
export ENABLE_BACKTEST_UI=0  # Disable UI for clearer logging
export SKIP_MODE_PROMPT=1
export DRY_RUN=1
export BACKTEST_SYMBOL=ES
export BACKTEST_MODEL=CVaR-PPO
export BACKTEST_DAYS=1
export ASPNETCORE_ENVIRONMENT=backtest

# Run the bot
cd src/UnifiedOrchestrator
dotnet run --configuration Release
```

## Historical Data Files

### Location
- ES (E-mini S&P 500): `datasets/quotes/es_quotes.json`
- NQ (E-mini NASDAQ-100): `datasets/quotes/nq_quotes.json`

### Format
Each file contains an array of quote objects with OHLCV data:

```json
[
  {
    "Time": "2024-10-29T09:30:00Z",
    "Symbol": "ES",
    "Bid": 4699.78,
    "Ask": 4700.03,
    "Last": 4699.91,
    "Volume": 145,
    "Open": 4700.0,
    "High": 4700.29,
    "Low": 4698.74,
    "Close": 4699.91
  },
  ...
]
```

### Current Data
- **ES**: ~500 minute bars
- **NQ**: ~500 minute bars
- **Date Range**: Recent market data (see files for exact dates)

## What You'll See

When running the offline backtest, you'll see output like:

```
📊 [BACKTEST] Running in silent mode (no UI)
📊 [BACKTEST] Loading historical data from ES (2024-10-28 to 2024-10-29)
✅ [BACKTEST] Historical data available for ES using provider: LocalQuotesProvider
📊 [BACKTEST] Processed 100 bars | Current Price: 4705.25 | P&L: $0.00
✅ TRADE EXECUTED: LONG 1 @ 4705.50 | Reason: Market order fill | Total P&L: -$2.50
📊 [BACKTEST] Processed 200 bars | Current Price: 4708.75 | P&L: $165.00
✅ TRADE EXECUTED: SHORT 1 @ 4708.75 | Reason: Stop loss triggered | Total P&L: $162.50
📊 [BACKTEST] Processed 300 bars | Current Price: 4710.25 | P&L: $240.00
📊 [BACKTEST] Completed processing 500 historical bars
Backtest completed successfully. Final PnL: $240.00, Trades: 5
```

### Key Indicators of Offline Operation

1. **Data Provider**: Look for `LocalQuotesProvider` (not `TopstepXHistoricalDataProvider`)
2. **No API Errors**: No connection errors or API timeouts
3. **Trade Execution**: `✅ TRADE EXECUTED` messages showing the bot is making trades
4. **Progress Updates**: Regular updates showing bars being processed
5. **Final Summary**: P&L and trade count at completion

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKTEST_MODE` | - | Set to `1` to enable backtest mode |
| `ENABLE_BACKTEST_UI` | `1` | Set to `0` for console logging, `1` for visual UI |
| `SKIP_MODE_PROMPT` | - | Set to `1` to skip interactive mode selection |
| `DRY_RUN` | - | Set to `1` to ensure no real orders (always use in backtest) |
| `BACKTEST_SYMBOL` | `ES` | Symbol to backtest (ES or NQ) |
| `BACKTEST_MODEL` | `CVaR-PPO` | Model to use for decisions |
| `BACKTEST_DAYS` | `7` | Number of days of history to process |
| `ASPNETCORE_ENVIRONMENT` | - | Set to `backtest` to use `appsettings.backtest.json` |

### Configuration File: `appsettings.backtest.json`

Key settings:
```json
{
  "Logging": {
    "LogLevel": {
      "TradingBot.Backtest": "Information"  // Shows trade execution
    }
  },
  "BacktestOptions": {
    "CommissionPerContract": 2.50,
    "InitialCapital": 100000,
    "EnableTickReplayUI": true
  }
}
```

## Trade Execution Logic

### Decision Making
1. **Primary**: Uses `UnifiedDecisionRouter` if available
   - Integrates all ML/RL models
   - Real production trading logic
   - Confidence-based decisions

2. **Fallback**: Simple momentum-based logic
   - Used for testing when router unavailable
   - Ensures backtest can run standalone

### Execution Simulation
- **Realistic fills**: Considers spread, slippage, and market impact
- **Commission modeling**: Applies per-contract fees
- **Bracket orders**: Automatic stop-loss and take-profit
- **Position tracking**: Real-time P&L calculation

## Verifying Offline Operation

To confirm the backtest is running completely offline:

1. **Disconnect network** (optional but proves offline capability):
   ```bash
   # Linux/macOS
   sudo ifconfig en0 down  # WiFi
   sudo ifconfig eth0 down # Ethernet
   
   # Then run backtest
   ./run-offline-backtest.sh
   
   # Re-enable network after test
   sudo ifconfig en0 up
   ```

2. **Check data provider**: Look for "LocalQuotesProvider" in logs
3. **Monitor network**: No HTTP requests should be made
4. **Verify file access**: The only I/O should be reading `datasets/quotes/*.json`

## Troubleshooting

### "Historical data not available"
- **Cause**: Quote files missing or empty
- **Solution**: Verify files exist in `datasets/quotes/`
- **Check**: Run `ls -lh datasets/quotes/` to see files

### "No historical model available"
- **Cause**: Model registry can't find model for date
- **Solution**: This is expected - the fallback logic will be used
- **Impact**: Bot uses simple test logic instead of ML models

### No trades executed
- **Cause**: Data doesn't meet trading criteria
- **Solution**: Normal behavior - not every bar triggers a trade
- **Check**: Look for "Hold - no clear signal" messages

### Build errors
- **Cause**: Missing dependencies or .NET SDK issues
- **Solution**: Run `dotnet restore` first
- **Check**: Ensure .NET 8.0+ SDK is installed

## Output Files

### Backtest Results
- **Location**: `reports/bt/`
- **Format**: JSON metrics and trade logs
- **Contents**: 
  - All trading decisions
  - Fill details with P&L
  - Performance statistics

### Log Files
- **Location**: `logs/backtest-*.log` (if file logging enabled)
- **Contents**: Detailed execution trace

## Performance Expectations

- **Processing speed**: ~1000-2000 bars/second (no UI)
- **Memory usage**: <500MB for typical backtest
- **CPU usage**: Single core, <50% utilization
- **Disk I/O**: Minimal (read quotes, write results)

## Extending the System

### Adding More Historical Data

1. **Manual creation**:
   ```bash
   # Edit datasets/quotes/es_quotes.json
   # Add more quote objects following the format
   ```

2. **From API** (when online):
   ```bash
   python3 fetch-and-save-historical-data.py
   ```

3. **Generate synthetic data** (for testing):
   ```bash
   python3 generate-sample-ticks.py
   ```

### Custom Trading Logic

Edit `BacktestHarnessService.cs`:
```csharp
private async Task<DecisionLog> MakeTradingDecisionAsync(...)
{
    // Add your custom logic here
    // This will be used in fallback mode
}
```

## Comparison: Offline vs Online Backtest

| Feature | Offline Mode | Online Mode |
|---------|--------------|-------------|
| API Required | ❌ No | ✅ Yes |
| Data Source | Local JSON files | TopstepX API |
| Network Access | Not needed | Required |
| Historical Range | Limited to saved data | Up to 365 days |
| Setup Time | Instant | Needs credentials |
| Reliability | Always works | Depends on API |
| Speed | Fast (local I/O) | Slower (network) |

## Summary

The offline backtest mode provides a **complete, self-contained** way to:
- ✅ Test trading strategies without API access
- ✅ Execute trades on historical data
- ✅ See bot decision-making in action
- ✅ Validate strategy performance
- ✅ Demonstrate bot capabilities

All using **saved historical bars** stored locally in JSON format - **no API connection required**.

## Next Steps

1. Run the offline backtest: `./run-offline-backtest.sh`
2. Review the output to see trades being executed
3. Check the results in `reports/bt/`
4. Experiment with different symbols and date ranges
5. Add more historical data if needed

For visual replay mode, set `ENABLE_BACKTEST_UI=1` to see tick-by-tick market replay with a professional trading interface.
