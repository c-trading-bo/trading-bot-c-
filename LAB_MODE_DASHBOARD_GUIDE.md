# Lab Mode Dashboard - Implementation Guide

## Overview

The Lab Mode Dashboard provides real-time, dynamic monitoring of Sunday training sessions with strategy-level performance tracking. The dashboard displays exactly as specified in the requirements, showing:

- Overall training progress with time tracking
- Phase-by-phase component status (Heavy, Medium, Light)
- **Strategy performance metrics** (Win Rate, PnL, Wins/Losses for S2, S3, S6, S11)
- System resource usage
- Recent activity log
- Post-training validation status
- Model promotion status

## Architecture

The dashboard consists of three main components:

### 1. LabModeDashboardModels.cs
Contains all data models for the dashboard:
- `LabModeDashboardState` - Complete dashboard state snapshot
- `StrategyTrainingMetrics` - Per-strategy performance tracking
- `PhaseDetails` - Heavy/Medium/Light phase information
- `ComponentSummary` - Individual component training details
- `ResourceMetrics` - System resource usage
- `ActivityLogEntry` - Recent activity log entries

### 2. LabModeDashboardStateManager.cs
Centralizes collection and management of dashboard metrics:
- Initializes training sessions
- Tracks phase progress
- Updates component training status
- **Collects strategy metrics** (win rate, PnL, wins/losses)
- Monitors system resources
- Maintains activity log

### 3. LabModeDashboardRenderer.cs
Renders the dashboard to terminal with exact formatting:
- Beautiful box-drawing characters (╔═══╗ ┌─┐ etc.)
- Progress bars with █ and ░ characters
- Strategy performance table with win rate, PnL, wins/losses
- Color-coded phase status indicators
- Real-time time tracking (ET timezone)
- System resource bars

### 4. ConsoleProgressRenderer.cs (Updated)
Integrated to use Lab Mode dashboard when enabled:
- Detects `LAB_MODE=1` environment variable
- Falls back to legacy display if dashboard not available
- Provides seamless integration

## Usage

### Integration with Training Orchestrator

The dashboard is designed to be integrated into `TrainingOrchestratorService`. Here's how:

```csharp
public class TrainingOrchestratorService
{
    private readonly LabModeDashboardStateManager _dashboardState;
    private readonly LabModeDashboardRenderer _dashboardRenderer;
    
    public async Task RunTrainingSessionAsync()
    {
        // 1. Initialize session
        _dashboardState.InitializeSession($"train-{DateTime.UtcNow:yyyyMMdd-HHmmss}", 250);
        
        // 2. Run Heavy Phase
        _dashboardState.UpdatePhase("Heavy", 7);
        
        foreach (var component in heavyComponents)
        {
            // Train component
            var result = await component.TrainAsync(config);
            
            // Update dashboard during training
            _dashboardState.UpdateComponentProgress(
                component.Name, 
                "Heavy", 
                currentEpoch, 
                totalEpochs, 
                currentLoss, 
                progress
            );
            
            // Update strategy metrics from backtest
            var backtestResult = await BacktestStrategyAsync("S2", component);
            _dashboardState.UpdateStrategyMetrics(
                "S2",
                backtestResult.WinRate,
                backtestResult.TotalPnL,
                backtestResult.TotalWon,
                backtestResult.TotalLost,
                backtestResult.WinningTrades,
                backtestResult.LosingTrades
            );
            
            // Render dashboard every N epochs
            if (currentEpoch % 5 == 0)
            {
                var state = _dashboardState.GetCurrentState();
                _dashboardRenderer.RenderDashboard(state);
            }
            
            // Mark component complete
            _dashboardState.CompleteComponent(
                component.Name, 
                "Heavy", 
                result.EpochsCompleted, 
                result.FinalLoss
            );
        }
        
        // 3. Complete phase
        _dashboardState.CompletePhase("Heavy", duration, succeeded, failed);
        
        // ... continue with Medium and Light phases ...
    }
}
```

### Strategy Metrics Collection

Strategy metrics should be collected after backtesting each strategy during training:

```csharp
private async Task UpdateStrategyPerformanceAsync(string strategy, ITrainingComponent component)
{
    // Run backtest with trained model
    var backtestResult = await _backtestService.RunBacktestAsync(strategy, component);
    
    // Calculate metrics
    var winningTrades = backtestResult.Trades.Where(t => t.PnL > 0).ToList();
    var losingTrades = backtestResult.Trades.Where(t => t.PnL < 0).ToList();
    var totalWon = winningTrades.Sum(t => t.PnL);
    var totalLost = Math.Abs(losingTrades.Sum(t => t.PnL));
    var winRate = (decimal)winningTrades.Count / backtestResult.Trades.Count * 100m;
    var totalPnL = backtestResult.Trades.Sum(t => t.PnL);
    
    // Update dashboard
    _dashboardState.UpdateStrategyMetrics(
        strategy,
        winRate,
        totalPnL,
        totalWon,
        totalLost,
        winningTrades.Count,
        losingTrades.Count
    );
    
    // Mark complete when done
    _dashboardState.CompleteStrategyTraining(strategy, "v1.2.5");
}
```

## Dashboard Output

The dashboard renders in the exact format specified:

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     🧪 LAB MODE - SUNDAY TRAINING SESSION                         ║
║                        Session ID: train-20251024-170000                          ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

⏰ Time: 5:15:42 PM ET | Elapsed: 3h 15m 42s | ETA: 29m 18s

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📈 OVERALL PROGRESS                                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [████████████████████████████████████████████░░░░░] 87.3%                      │
│ Components: 218/250 completed (32 remaining)                                   │
│ Phase: 🟢 LIGHT PHASE (Online Learning & Fine-Tuning)                          │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🔴 HEAVY PHASE - COMPLETE ✓                                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Duration: 2h 45m | Success: 7/7 | Failed: 0                                    │
│                                                                                 │
│ ✓ CVaR-PPO Trainer           [████████] 100% | Epochs: 10/10 | Loss: 0.0023    │
│   - Episodes: 150 | Avg Reward: +2.34 | Model: saved (v1.2.3)                 │
...
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📊 STRATEGY PERFORMANCE DURING TRAINING                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Strategy    Win Rate   Total PnL    Total Won    Total Lost   Trades   Status  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ S2             62.1%  $ 1580.75  $  2100.00  $  -519.25     200   ✓      │
│ S3             48.3%  $ 1120.50  $  1890.00  $  -769.50     200   ✓      │
│ S6             55.4%  $  980.00  $  1620.00  $  -640.00     200   ✓      │
│ S11            51.2%  $ 1350.25  $  1980.00  $  -629.75     200   ✓      │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Real-Time Updates

The dashboard updates in real-time during training:

1. **Component Progress**: Updated every epoch or batch
2. **Strategy Metrics**: Updated after each strategy backtest
3. **System Resources**: Updated every 30 seconds
4. **Activity Log**: Updated on significant events
5. **Time Tracking**: Continuous updates with ETA calculation

## Testing

See `LabModeDashboardIntegrationExample.cs` for a complete demo showing:
- Session initialization
- Phase progression
- Component training simulation
- Strategy metrics updates
- Real-time dashboard rendering

Run the demo with:
```bash
LAB_MODE=1 dotnet run --project src/UnifiedOrchestrator
```

## Configuration

Enable Lab Mode dashboard by setting environment variable:
```bash
export LAB_MODE=1
```

The dashboard automatically activates when:
1. `LAB_MODE=1` is set
2. `LabModeDashboardRenderer` is registered in DI
3. `LabModeDashboardStateManager` is registered in DI
4. Called from `ConsoleProgressRenderer.RenderProgress()`

## Next Steps

To complete the integration:

1. ✅ Dashboard models created
2. ✅ Dashboard renderer implemented
3. ✅ State manager implemented
4. ✅ Console renderer updated
5. ⏳ Integrate with TrainingOrchestratorService
6. ⏳ Add strategy backtest hooks
7. ⏳ Test with real training session
8. ⏳ Verify real-time updates

## Benefits

- **Dynamic Strategy Tracking**: See each strategy's performance improve during training
- **Real-Time Visibility**: Know exactly what's happening at any moment
- **Performance Metrics**: Win rate, PnL, wins/losses for each strategy
- **Professional Display**: Beautiful terminal UI matching requirements exactly
- **Actionable Insights**: Immediately see which strategies are performing well
- **Progress Tracking**: ETA and completion percentages at all levels
