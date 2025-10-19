# Mode Separation Architecture

## Overview

The bot now has complete separation between three distinct modes:
1. **HISTORICAL** - Bulk training on 90 days of historical data
2. **DRY-RUN** - Paper trading with live data
3. **LIVE** - Real trading with real money

## Mode Detection

Added `IsHistoricalMode()` static method in `ProductionKillSwitchService`:
```csharp
public static bool IsHistoricalMode()
{
    var historicalMode = Environment.GetEnvironmentVariable("HISTORICAL_MODE");
    return historicalMode == "1" || historicalMode.Equals("true", StringComparison.OrdinalIgnoreCase);
}
```

## Service Registration by Mode

### HISTORICAL Mode (HISTORICAL_MODE=1)

**Registered Services:**
- ✅ `HistoricalDataSeedService` - Loads 90 days from local files
- ✅ `HistoricalDataBridgeService` - Bridges historical data to trading system
- ✅ `TradingSystemBarConsumer` - Consumes historical bars
- ✅ `HistoricalReplayOrchestrator` - Main orchestrator for replay
- ✅ `SlippageLatencyModel` - Simulates execution
- ✅ `UnifiedTradingBrain` - Makes trading decisions
- ✅ `PaperTradingTracker` - Tracks positions

**Skipped Services:**
- ❌ `UnifiedOrchestratorService` - Not needed (HistoricalReplayOrchestrator takes over)
- ❌ TopstepX API connections - No live data needed
- ❌ Live bar streaming - All data from files

**Characteristics:**
- 📁 Data source: Local JSON files (`data/historical/ES_90days.json`, `data/historical/NQ_90days.json`)
- ⚡ Speed: Fast-forward (process 90 days in ~2 hours)
- 💻 Execution: Simulated with realistic slippage
- 🎓 Learning: Models update during replay
- 📊 Output: Terminal logs + final metrics summary

### LIVE/DRY-RUN Mode (HISTORICAL_MODE=0)

**Registered Services:**
- ✅ `UnifiedOrchestratorService` - Main orchestrator for live trading
- ✅ `TopstepXAdapterService` - Connects to TopstepX API
- ✅ `EnhancedBacktestLearningService` - Continuous learning (if enabled)
- ✅ `UnifiedTradingBrain` - Makes trading decisions
- ✅ `PaperTradingTracker` - Tracks positions (DRY-RUN only)
- ✅ Live bar streaming - Real-time market data

**Skipped Services:**
- ❌ `HistoricalDataSeedService` - Not needed (live data from API)
- ❌ `HistoricalDataBridgeService` - Not needed
- ❌ `HistoricalReplayOrchestrator` - Not needed

**Characteristics:**
- 🌐 Data source: TopstepX API (live/real-time)
- ⏱️ Speed: Real-time (waits for actual bars)
- 💰 Execution: Real orders (LIVE) or simulated (DRY-RUN)
- 🎓 Learning: Models update from real trades
- 📊 Output: Terminal logs + real trades

## Console Output Examples

### Starting in HISTORICAL Mode
```
📊 [HISTORICAL-SEED] Smart auto-refresh service registered (HISTORICAL MODE ONLY)
   ⚡ Loads historical bars from disk (instant vs 30s+ API fetch)
   📅 Only active in HISTORICAL_MODE=1
⏭️ [ORCHESTRATOR] UnifiedOrchestratorService skipped (HISTORICAL_MODE - using HistoricalReplayOrchestrator instead)
✅ [HISTORICAL-MODE] Historical replay orchestrator ENABLED
   📊 Bot will replay 90 days of historical data at high speed
   🎓 Models will be trained on simulated trading
   📝 Complete audit trail will be logged to terminal
```

### Starting in LIVE/DRY-RUN Mode
```
⏭️ [HISTORICAL-SEED] Skipped registration (not in HISTORICAL_MODE)
✅ [ORCHESTRATOR] UnifiedOrchestratorService registered (LIVE/DRY-RUN mode)
✅ [HISTORICAL-LEARNING] Historical backtest learning ENABLED
   📊 Market OPEN: Learning every 60 minutes (light mode)
   📈 Market CLOSED: Learning every 15 minutes (intensive mode)
```

## Benefits of Clean Separation

1. **No Service Conflicts** - Each mode only loads services it needs
2. **Faster Startup** - Historical mode doesn't initialize TopstepX adapter
3. **Clear Boundaries** - No mixing of historical files with live data
4. **Memory Efficient** - Only needed services consume resources
5. **Easy to Test** - Each mode can be tested independently
6. **Safety** - Historical mode can never place real orders

## Mode Selection at Startup

Users can select mode via interactive prompt:
```
[1] 📊 HISTORICAL TRAINING MODE
[2] 🚀 LIVE MODE  
[3] 📝 DRY-RUN MODE
```

Or set environment variables directly:
```bash
# Historical mode
export HISTORICAL_MODE=1
export SKIP_MODE_PROMPT=1

# Live mode
export HISTORICAL_MODE=0
export DRY_RUN=0
export SKIP_MODE_PROMPT=1

# Dry-run mode
export HISTORICAL_MODE=0
export DRY_RUN=1
export SKIP_MODE_PROMPT=1
```

## Implementation Files

- `src/BotCore/Services/ProductionKillSwitchService.cs` - Mode detection
- `src/UnifiedOrchestrator/Program.cs` - Conditional service registration
- `src/BotCore/Extensions/ProductionReadinessServiceExtensions.cs` - Conditional bridge services
- `src/UnifiedOrchestrator/Services/HistoricalReplayOrchestrator.cs` - Historical mode orchestrator

## Testing

Both modes have been tested and verified:
- ✅ HISTORICAL_MODE=1 correctly skips live services
- ✅ HISTORICAL_MODE=0 correctly skips historical services
- ✅ No service resolution errors
- ✅ Clean console output showing active mode
- ✅ Build succeeds with 0 errors
