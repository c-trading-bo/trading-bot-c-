# Backtest Order Service - Realistic Position Management in Backtesting

## Overview

The `BacktestOrderService` is a mock implementation of `IOrderService` designed specifically for backtesting. It enables realistic simulation of position management features like breakeven stops, trailing stops, and dynamic stop-loss modifications that were previously only available in live trading.

## Problem It Solves

Prior to this implementation, the backtesting framework couldn't properly test advanced position management features because:

1. **No Broker Simulation**: Methods like `ModifyStopLossAsync()` and `CancelOrderAsync()` returned `null` or `false` in backtest mode
2. **Silent Failures**: Professional features (breakeven, trailing stops, regime-based stop updates) were never exercised during backtests
3. **Testing Gap**: Live trading code had sophisticated features that couldn't be validated before going live

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    BacktestHarnessService                    │
│  • Drives historical data replay                            │
│  • Updates BacktestOrderService on each tick                │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   BacktestOrderService                       │
│  • Tracks simulated orders and positions                    │
│  • Processes order fills based on price movements           │
│  • Logs all order management events                         │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              UnifiedPositionManagementService                │
│  • Manages breakeven stops                                  │
│  • Manages trailing stops                                   │
│  • Adjusts stops based on regime changes                    │
│  • Calls IOrderService methods (BacktestOrderService)       │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **BacktestOrderService** (`src/Backtest/Services/BacktestOrderService.cs`)
   - Implements `IOrderService` interface
   - Maintains internal order and position tracking
   - Simulates order fills when price triggers occur
   - Logs all order management events to JSON

2. **Order Management Logging** (`src/Backtest/IMetricSink.cs`)
   - New `OrderManagementLog` record type
   - Captures stop modifications, breakeven activations, trailing stops
   - Saved to `order_management_{timestamp}.json`

3. **JsonMetricSink Enhancement** (`src/Backtest/Metrics/JsonMetricSink.cs`)
   - Extended to track order management events
   - Provides breakdown of order management activities
   - Enables post-backtest analysis of position management behavior

## Features Enabled

With `BacktestOrderService`, your backtests can now accurately simulate:

### 1. Dynamic 1:1 Breakeven System
- Moves stop to entry +1 tick when profit target is reached
- Prevents stopped out trades that reached breakeven
- Logs breakeven activation for analysis

### 2. Trailing Stops
- Automatically updates stop as position moves in your favor
- Respects trailing distance settings
- Captures maximum favorable excursion

### 3. Stop Order Management
- Cancel old stop orders before placing new ones
- Simulate realistic broker order lifecycle
- Track stop modification count

### 4. Regime Change Detection
- Update stops when market regime changes
- Tighter stops in uncertain regimes
- Wider stops in trending regimes

### 5. Professional Logging
- All order placements logged
- All stop modifications logged
- All order cancellations logged
- Comprehensive position management history

## Usage

### Registration (Automatic)

The service is automatically registered in `BacktestServiceExtensions.cs`:

```csharp
services.AddSingleton<BacktestOrderService>();
services.AddSingleton<IOrderService>(sp => sp.GetRequiredService<BacktestOrderService>());
```

### Integration with Backtest

The `BacktestHarnessService` automatically:
1. Injects `BacktestOrderService` 
2. Updates it on each tick with current market state
3. Processes pending orders
4. Resets state between backtest runs

### Position Management

Your existing `UnifiedPositionManagementService` code works unchanged:

```csharp
// This now works in backtest mode!
await orderService.ModifyStopLossAsync(positionId, newStopPrice);
await orderService.ModifyTakeProfitAsync(positionId, newTargetPrice);
await orderService.CancelOrderAsync(orderId);
```

## Output Files

After running a backtest, you'll find these files in the `reports/bt/` directory:

### 1. `order_management_{timestamp}.json`
Contains all position management events:
```json
[
  {
    "timestamp": "2024-11-25T10:15:23",
    "positionId": "pos-001",
    "symbol": "MES",
    "eventType": "StopModified",
    "oldPrice": 4500.25,
    "newPrice": 4501.00,
    "reason": "Breakeven"
  },
  {
    "timestamp": "2024-11-25T10:18:45",
    "positionId": "pos-001",
    "symbol": "MES",
    "eventType": "StopModified",
    "oldPrice": 4501.00,
    "newPrice": 4503.50,
    "reason": "Trailing"
  }
]
```

### 2. `summary_{timestamp}.json`
Includes order management breakdown:
```json
{
  "totalOrderManagementEvents": 127,
  "orderManagementBreakdown": {
    "StopModified": 89,
    "Breakeven": 23,
    "TrailingStop": 11,
    "OrderCancelled": 4
  }
}
```

## Benefits

### For Development
- Test all position management features before going live
- Catch bugs in breakeven/trailing stop logic
- Validate regime-based stop adjustments
- Build confidence in live trading readiness

### For Analysis
- Understand how often breakeven triggers
- Analyze trailing stop effectiveness
- Identify over-aggressive stop management
- Optimize position management parameters

### For Production
- Realistic backtests that match live trading behavior
- No surprises when transitioning from backtest to live
- Same code paths exercised in both environments
- Professional-grade position management simulation

## Implementation Notes

### Thread Safety
The service uses `Dictionary<>` for internal state, which is accessed only from the single-threaded backtest loop. No additional locking is needed.

### Performance
Order processing is O(n) where n = number of pending orders. For typical backtests with <100 concurrent orders, this is negligible.

### Accuracy
Stop triggers use `currentQuote.Last` price, which may differ slightly from bid/ask fills. This is acceptable for backtesting and matches most broker implementations.

## Future Enhancements

Potential improvements (not required for current functionality):

1. **Slippage Simulation**: Add configurable slippage to stop fills
2. **Partial Fills**: Support partial order fills for large positions
3. **Order Queue**: Simulate order queue delays and priority
4. **Book Integration**: Use `BookAwareExecutionSimulator` for more realistic fills

## Validation

To verify the BacktestOrderService is working correctly, check that:

1. Console logs show `[BACKTEST-ORDER]` entries for stop modifications
2. `order_management_{timestamp}.json` file is created
3. Summary shows non-zero `totalOrderManagementEvents`
4. Backtest P&L reflects stop modifications (fewer losses, better risk management)

## Conclusion

The `BacktestOrderService` bridges the gap between basic backtesting and professional trading simulation. It enables you to test sophisticated position management strategies with confidence before deploying them in live markets.

All features that work in live trading now work in backtesting - giving you realistic, reliable backtest results you can trust.
