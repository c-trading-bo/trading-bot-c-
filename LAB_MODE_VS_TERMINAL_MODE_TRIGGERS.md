# Lab Mode vs Terminal Mode - Trigger Mechanisms

## Overview

This document clarifies the **trigger mechanisms** for each mode in the QBot trading system.

## Automatic Data Collection

**Daily Data Collection (6 AM):**
- ✅ **Automatic** - runs every day
- Collects 5-minute bars → `ES_90days.json`, `NQ_90days.json`
- Collects 1-minute bars → `ES_1m_90days.json`, `NQ_1m_90days.json`
- Trims to 90-day rolling window
- NO user action required

## Sunday Lab Mode (Automatic Training)

**Trigger:** ✅ **Automatic** - clock-based schedule
- Runs every Sunday 12:00 PM - 5:45 PM ET
- Triggered by time/day of week
- NO user action required

**What Happens:**
- Loads full 90 days of multi-timeframe data (5m + 1m bars)
- Trains all models (CVaR-PPO, SAC, LSTM, etc.)
- Applies overfitting prevention (early stopping, multi-seed, test holdout)
- Promotes models to production if they beat champion
- Exports frozen ONNX models for Terminal Mode

**Philosophy:** "Hands-off weekly improvement" - consistent model updates without user intervention

## Anyday Lab Mode (Manual Training Only)

**Trigger:** ❌ **Manual Only** - user must explicitly launch
- Set `FORCE_LAB_NOW=1` environment variable
- Restart bot or run explicit command
- NO automatic triggers of any kind

**What Does NOT Trigger It Automatically:**
- ❌ NOT triggered by performance degradation
- ❌ NOT triggered by Sharpe ratio dropping
- ❌ NOT triggered by drawdown increasing
- ❌ NOT triggered by consecutive losses
- ❌ NOT triggered by regime shifts
- ❌ NOT triggered by data quality issues
- ❌ NOT triggered by catastrophic forgetting detection
- ❌ NOT triggered by ANY automatic condition

**What Happens When User Launches It:**
- Uses SAME training pipeline as Sunday Lab (HistoricalTrainingOrchestrator)
- Loads whatever multi-timeframe data exists (5m + 1m bars)
  - Example: 54 days on Wednesday vs 90 days on Sunday
- Trains with same rigorous validation (multi-seed + bootstrap)
- Requires user approval before promoting to production
- Can run on ANY day of the week (Monday, Wednesday, Friday, etc.)

**Philosophy:** "Emergency retrain button" - user decides when market conditions warrant manual intervention

## Terminal Mode (Automatic Trading)

**Trigger:** ✅ **Automatic** - starts Monday morning after Sunday Lab
- Runs continuously Monday-Saturday
- NO user action required for trading execution

**What Happens Automatically:**
- Uses frozen ONNX models from last Sunday Lab (or approved Anyday Lab)
- Executes trades based on multi-timeframe inference (5m + 1m context)
- Performs lightweight online calibration (NO weight updates)
- Monitors performance metrics passively
- Logs warnings when performance degrades

**What Does NOT Happen Automatically:**
- ❌ Does NOT automatically trigger Anyday Lab retraining
- ❌ Does NOT modify model weights
- ❌ Does NOT retrain models
- ❌ Performance monitoring is passive reporting only

**User Responsibility:**
- Review performance metrics periodically
- Evaluate if degradation warrants manual Anyday Lab intervention
- Decide when to trigger Anyday Lab based on market conditions

**Philosophy:** "Hands-off trading" - models execute strategies, user monitors and decides if manual intervention needed

## Performance Degradation Detector

**Role:** ❌ **Monitoring Only** - NO automatic triggers
- Monitors Sharpe ratio, drawdown, consecutive losses
- Logs warnings when degradation conditions detected
- Reports safety check results (for user information)
- Suggests manual intervention in logs

**What It Does:**
- ✅ Checks performance metrics every 4 hours
- ✅ Logs degradation warnings
- ✅ Runs safety checks (training not in progress, sufficient data, resources available)
- ✅ Informs user: "To manually trigger Anyday Lab retraining, set FORCE_LAB_NOW=1"

**What It Does NOT Do:**
- ❌ Does NOT automatically set FORCE_LAB_NOW=1
- ❌ Does NOT automatically spawn Lab Mode process
- ❌ Does NOT trigger retraining of any kind
- ❌ Does NOT modify any environment variables

## Key Distinctions

| Mode | Trigger Type | User Action Required | Training Happens | Trading Happens |
|------|--------------|---------------------|------------------|-----------------|
| **Sunday Lab** | ✅ Automatic (clock) | Zero | ✅ Yes | ❌ No |
| **Anyday Lab** | ❌ Manual Only | Explicit (set FORCE_LAB_NOW=1) | ✅ Yes | ❌ No |
| **Terminal Mode** | ✅ Automatic | Zero for trading | ❌ No | ✅ Yes |

## How to Manually Trigger Anyday Lab

**Step 1:** Review performance metrics
```bash
# Check logs for degradation warnings
tail -f logs/trading.log | grep DEGRADATION
```

**Step 2:** Decide if manual retraining is warranted
- Evaluate market conditions
- Review recent trades
- Consider regime changes

**Step 3:** Trigger Anyday Lab manually
```bash
# Set environment variable
export FORCE_LAB_NOW=1

# Restart bot (or run explicit lab mode command)
./restart-in-lab-mode.sh
```

**Step 4:** Monitor training
- Lab mode runs same pipeline as Sunday
- Review training metrics
- Approve model promotion if performance improved

## Why This Design?

### Automatic Sunday Lab
- **Benefit:** Consistent weekly improvement without human intervention
- **Use Case:** Regular model updates based on new data
- **Trust:** Proven pipeline with rigorous overfitting prevention

### Manual Anyday Lab
- **Benefit:** User maintains full control over emergency retraining
- **Use Case:** Unexpected market regime shifts, rare events, user discretion
- **Protection:** Prevents "false alarms" from automatic triggers
- **Flexibility:** User evaluates context before deciding to retrain

### Automatic Terminal Mode Trading
- **Benefit:** Hands-off execution once models are trained
- **Use Case:** Daily trading with proven models
- **Safety:** Models are frozen (no automatic weight updates)

## Summary

**Automatic Operations:**
1. ✅ Daily data collection (6 AM)
2. ✅ Sunday Lab training (Sunday 12 PM - 5:45 PM ET)
3. ✅ Terminal Mode trading (Monday-Saturday)
4. ✅ Performance monitoring (passive reporting)

**Manual Operations:**
1. ❌ Anyday Lab training (user sets FORCE_LAB_NOW=1)
2. ❌ Model promotion approval (for Anyday Lab)
3. ❌ Emergency intervention decisions

**No Automatic Triggers For:**
- ❌ Anyday Lab Mode
- ❌ Emergency retraining
- ❌ Model weight updates during Terminal Mode

The user is always in control of when emergency retraining happens.
