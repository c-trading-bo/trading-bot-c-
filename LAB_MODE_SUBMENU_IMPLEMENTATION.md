# Lab Mode Sub-Menu Implementation - UI Flow Documentation

## Overview
This document shows the expected user interface flow for the new Lab Mode sub-menu feature.

## Main Menu (Before Change)
The main menu remains clean with 3 options:

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    TopstepX Trading Bot - Mode Selection                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  [1] Terminal Mode (Live Trading)                                             ║
║      • Real-time market execution                                             ║
║      • Inference-only operations                                              ║
║      • Safety systems active                                                  ║
║      • Model loading from registry                                            ║
║                                                                                ║
║  [2] Lab Mode (Historical Training)                                           ║
║      • Historical data replay                                                 ║
║      • Model training & optimization                                          ║
║      • Scheduled Sunday 12:00 PM - 5:45 PM ET                                 ║
║      • No live trading                                                        ║
║                                                                                ║
║  [3] Backtest Mode (Strategy Testing)                                         ║
║      • Historical strategy validation                                         ║
║      • Performance metrics                                                    ║
║      • No training or live execution                                          ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

Select mode [1-3]:
```

## NEW: Lab Mode Sub-Menu (After Selecting Option 2)
When user selects option 2 (Lab Mode), they now see this sub-menu:

```
╔════════════════════════════════════════════════════════════════════════════════╗
║                      Lab Mode - Training Schedule Options                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║                                                                                ║
║  [1] Scheduled Training (Sunday Only)                                         ║
║      • Runs Sunday 12:00 PM - 5:45 PM ET                                      ║
║      • Waits for Sunday if not today                                          ║
║      • Automatic weekly retraining                                            ║
║      • Best for: Production automation                                        ║
║                                                                                ║
║  [2] Manual Training (Run Now)                                                ║
║      • Starts immediately (any day/time)                                      ║
║      • No schedule restrictions                                               ║
║      • Run as many times as you want                                          ║
║      • Best for: Testing, emergencies, experiments                            ║
║                                                                                ║
║  [3] Back to Main Menu                                                        ║
║                                                                                ║
╚════════════════════════════════════════════════════════════════════════════════╝

Select training schedule [1-3]:
```

## Option 1: Scheduled Training (Sunday Only)
When user selects option 1 in the sub-menu:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                   🧪 LAB MODE - SCHEDULED TRAINING 🧪                     ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  SCHEDULED TRAINING MODE - Automated Sunday training                      ║
║                                                                            ║
║  ✓ Scheduled training: Sundays 12:00 PM - 5:45 PM ET (DST-aware)         ║
║  ✓ Pre-training health checks (disk, RAM, CPU, data integrity)            ║
║  ✓ Retry logic with exponential backoff (5m, 15m, 30m)                    ║
║  ✓ Watchdog timeout enforcement (5 hour maximum)                          ║
║  ✓ Post-training canary tests before model promotion                      ║
║  ✓ Artifact manifests with SHA256 checksums                               ║
║  ✓ Atomic file operations and rollback safety                             ║
║  ✓ Structured logging with unique run IDs                                 ║
║  ✓ Alert notifications for all training events                            ║
║  ✓ Graceful shutdown with checkpoint saving                               ║
║  ✓ Metrics collection and export                                          ║
║                                                                            ║
║  ⚠️  NO LIVE TRADING in Lab mode - Training pipeline only                 ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

🧪 Lab scheduler will activate and wait for Sunday training window

Press Enter to continue...
```

**Environment Variables Set:**
- `LAB_MODE=1`
- `HISTORICAL_MODE=0`
- `DRY_RUN=1`
- `FORCE_LAB_NOW=0` (waits for Sunday schedule)

## Option 2: Manual Training (Run Now)
When user selects option 2 in the sub-menu:

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                     🧪 LAB MODE - MANUAL TRAINING 🧪                      ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  MANUAL TRAINING MODE - Immediate training execution                      ║
║                                                                            ║
║  ✓ Starts IMMEDIATELY (no wait for Sunday)                                ║
║  ✓ All training features enabled (same as scheduled)                      ║
║  ✓ Pre-training health checks (disk, RAM, CPU, data integrity)            ║
║  ✓ Watchdog timeout enforcement (5 hour maximum)                          ║
║  ✓ Post-training canary tests before model promotion                      ║
║  ✓ Artifact manifests with SHA256 checksums                               ║
║  ✓ Atomic file operations and rollback safety                             ║
║  ✓ Structured logging with unique run IDs                                 ║
║  ✓ Alert notifications for all training events                            ║
║                                                                            ║
║  ⚠️  NO LIVE TRADING in Lab mode - Training pipeline only                 ║
║                                                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝

🧪 Manual training mode activated - Training will start IMMEDIATELY
⚡ Bypassing Sunday schedule restriction

Press Enter to continue...
```

**Environment Variables Set:**
- `LAB_MODE=1`
- `HISTORICAL_MODE=0`
- `DRY_RUN=1`
- `FORCE_LAB_NOW=1` (starts immediately, bypasses Sunday schedule)

## Option 3: Back to Main Menu
When user selects option 3 in the sub-menu:

```
🔄 Returning to main menu...

[Returns to the main menu with 3 options]
```

## User Experience Examples

### Example 1: Production User Wants Scheduled Training
1. Main menu appears
2. User types `2` for Lab Mode
3. Sub-menu appears asking scheduled or manual
4. User types `1` for scheduled
5. Bot checks if today is Sunday
   - If not Sunday: Bot waits until next Sunday noon
   - If Sunday in window: Training starts immediately

### Example 2: Developer Wants Manual Training
1. Main menu appears
2. User types `2` for Lab Mode
3. Sub-menu appears asking scheduled or manual
4. User types `2` for manual
5. Training starts immediately right now (no waiting)

### Example 3: User Picks Wrong Mode
1. Main menu appears
2. User types `2` for Lab Mode (meant to pick Terminal Mode)
3. Sub-menu appears
4. User sees option 3 to go back
5. User types `3`
6. Main menu appears again
7. User types `1` for Terminal Mode
8. Problem solved without restarting

## Benefits of This Implementation

1. **Cleaner main menu**: Lab Mode stays as one simple option (3 total, not 4)
2. **Progressive disclosure**: Users only see the schedule choice if they actually want Lab Mode
3. **Option to go back**: Users can return to main menu if they pick Lab Mode by accident
4. **Grouped related options**: Both training schedules are under Lab Mode where they logically belong
5. **Easier to explain**: "Pick Lab Mode, then it asks you when you want to train"

## Technical Implementation

- Main menu method: `PromptForTradingModeAsync()`
- New sub-menu method: `PromptForLabModeScheduleAsync()`
- When user selects option 2 in main menu, it calls the sub-menu
- Sub-menu handles all Lab Mode configuration
- Uses existing `FORCE_LAB_NOW` environment variable (already in `InternalScheduler.cs`)
- Maintains all existing training infrastructure
- No breaking changes to existing functionality
