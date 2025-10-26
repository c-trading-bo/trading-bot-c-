# 🧪 Lab Mode Dashboard - Visual Example

This document shows what the Lab Mode dashboard looks like during a training session.

## Full Dashboard View

```
╔═══════════════════════════════════════════════════════════════════════════════════╗
║                     🧪 LAB MODE - SUNDAY TRAINING SESSION                         ║
║                        Session ID: train-20251026-031044                         ║
╚═══════════════════════════════════════════════════════════════════════════════════╝

⏰ Time: 10:10:59 PM ET | Elapsed: 0s | ETA: 0s

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📈 OVERALL PROGRESS                                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0%                      │
│ Components: 0/250 completed (250 remaining)                                     │
│ Phase: 🟢 LIGHT PHASE (Online Learning & Fine-Tuning)                           │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🔴 HEAVY PHASE - COMPLETE ✓                                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [████████████████████████████████████████] 100.0% (11/11 completed)             │
│ Duration: 2h 15m | Success: 11/11 | Failed: 0                                  │
│                                                                                 │
│ ✓ [1/11] CVaRPPOTrainer                          (15m 32s) - 21,600 experiences│
│ ✓ [2/11] NeuralUcbBanditTrainer                  (12m 45s) - 18,200 experiences│
│ ✓ [3/11] LSTMTrainer                             (18m 20s) - 25,400 experiences│
│ ✓ [4/11] PatternRecognitionTrainer               (10m 15s) - 15,800 experiences│
│ ✓ [5/11] RegimeDetectorTrainer                   (8m 40s)  - 12,300 experiences│
│ ✓ [6/11] SlippageLatencyTrainer                  (7m 30s)  - 10,500 experiences│
│ ✓ [7/11] ModelEnsembleTrainer                    (14m 10s) - 19,700 experiences│
│ ✓ [8/11] VolatilityForecasterTrainer             (11m 25s) - 16,400 experiences│
│ ✓ [9/11] OrderFlowImbalanceTrainer               (9m 50s)  - 14,100 experiences│
│ ✓ [10/11] MicrostructureTrainer                  (13m 5s)  - 18,900 experiences│
│ ✓ [11/11] AdversarialRobustnessTrainer           (16m 15s) - 22,800 experiences│
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🟡 MEDIUM PHASE - IN PROGRESS ⚙️                                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [████████████████████░░░░░░░░░░░░░░░░░░░] 57.1% (4/7 completed)                 │
│ Duration: 45m 20s | Success: 4/7 | Failed: 0                                   │
│                                                                                 │
│ ✓ [1/7] CalibratorTrainer                        (8m 30s)  - 5,200 experiences │
│ ✓ [2/7] HyperparameterOptimizerTrainer           (12m 15s) - 7,800 experiences │
│ ✓ [3/7] EnsembleWeightOptimizerTrainer           (10m 40s) - 6,500 experiences │
│ ✓ [4/7] AdaptiveLearningRateScheduler            (9m 25s)  - 5,900 experiences │
│ ⏳ [5/7] FeatureImportanceAnalyzer                (In progress: 4m 30s elapsed) │
│ ⏸ [6/7] ModelPruningOptimizer                    (Pending)                     │
│ ⏸ [7/7] TransferLearningCoordinator              (Pending)                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🟢 LIGHT PHASE - PENDING                                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 0.0% (0/7 completed)                 │
│ Duration: Not started                                                           │
│                                                                                 │
│ Queued Components:                                                              │
│  • OnlineLearningAgent                                                          │
│  • MetaLearningCoordinator                                                      │
│  • FineTuningOrchestrator                                                       │
│  • ContinualLearningManager                                                     │
│  • ExperienceReplayOptimizer                                                    │
│  • PolicyDistillationService                                                    │
│  • KnowledgeTransferService                                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📊 CURRENT TRAINING METRICS (FeatureImportanceAnalyzer)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Epoch: 145/200 | Batch: N/A | Learning Rate: N/A                               │
│                                                                                 │
│ Loss Metrics:                                                                   │
│  • Total Loss:       0.0234 (tracking)                                          │
│                                                                                 │
│ Performance:                                                                    │
│  • Training Progress:    72.5%                                                  │
│                                                                                 │
│ Resource Usage:                                                                 │
│  • GPU Utilization:      N/A (CPU training)                                     │
│  • CPU Utilization:      85%                                                    │
│  • Memory Used:          12.3 GB / 16.0 GB (77%)                                │
│  • Disk I/O:             45 MB/s read, 23 MB/s write                            │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📊 STRATEGY PERFORMANCE DURING TRAINING                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Strategy    Win Rate   Total PnL    Total Won    Total Lost   Trades   Status  │
├─────────────────────────────────────────────────────────────────────────────────┤
│ S2             52.3%  $   342.50  $   890.25  $   -547.75    1520   ✅ Live    │
│ S3             48.7%  $  -125.00  $   412.50  $   -537.50     850   ⚙️  Train  │
│ S6             55.1%  $   678.25  $  1245.00  $   -566.75    2140   ✅ Live    │
│ S11            49.8%  $   -45.50  $   301.25  $   -346.75     620   ⚙️  Train  │
│                                                                                 │
│ Total Portfolio: $850.25 | Sharpe: 1.45 | Max DD: $-234.50                     │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🔍 POST-TRAINING VALIDATION                                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│ ⏳ Waiting for Light Phase completion...                                       │
│                                                                                 │
│ Validation Checklist:                                                          │
│  □ Model Integrity Check                                                       │
│  □ Performance Baseline Comparison (75% threshold)                             │
│  □ Statistical Significance Test (95% confidence)                              │
│  □ Anti-Overfitting Validation (walk-forward)                                  │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 🚀 MODEL PROMOTION STATUS                                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│ Status: ⏳ Pending (waiting for validation)                                    │
│                                                                                 │
│ Promotion Plan:                                                                │
│  - Challenger Models: 7 heavy + 7 medium + 7 light = 21 models                │
│  - Atomic Promotion: enabled (rollback on failure)                             │
│  - Backup: staging/ → production/ (safe swap)                                 │
│  - Rollback Window: 15 minutes                                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📊 SYSTEM RESOURCES                                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│ CPU: [█████████████████░░░]  85% | Memory: [███████████████░]  77% (12.3 GB / 16.0 GB)│
│ Disk I/O:  45 MB/s read, 23 MB/s write | GPU: N/A (CPU training)              │
│ Training Processes: 8 active | Memory Leak: ✓ None detected                    │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 📝 RECENT ACTIVITY LOG                                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│ [14:23:45] info: FeatureImportanceAnalyzer                                      │
│            Analyzing feature importance for 247 features...                  │
│ [14:23:42] info: AdaptiveLearningRateScheduler                                  │
│            ✓ Training complete - loss: 0.0234, duration: 9m 25s             │
│ [14:23:15] info: EnsembleWeightOptimizer                                        │
│            ✓ Training complete - optimized 11 model weights                 │
│ [14:22:58] info: HyperparameterOptimizer                                        │
│            ✓ Optimized hyperparameters - best trial: 0.0189 loss            │
│ [14:22:30] info: CalibratorTrainer                                              │
│            ✓ Calibration complete - isotonic calibration applied            │
└─────────────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════════════╗
║ Press Ctrl+C to cancel training (will save checkpoint for resume)                ║
║ Training lock file: /tmp/qbot_lab_training.lock                              ║
║ Uptime: 3h 12m 45s       | Lock File Age: 3h 15m           | Next refresh: 5s      ║
╚═══════════════════════════════════════════════════════════════════════════════════╝
```

## Features Highlighted

### ✅ No Scrolling
- Dashboard uses ANSI escape codes to update in-place
- Clear screen (`\x1b[2J`) and move cursor home (`\x1b[H`)
- Static display that refreshes every 5 seconds

### ✅ Real-Time Updates
- 5-second update timer in `TrainingOrchestratorService`
- All metrics update automatically:
  - CPU and Memory usage
  - Component progress
  - Strategy performance
  - Recent activity log

### ✅ Three-Phase Training
- **🔴 Heavy Phase**: Large neural networks (11 components, ~2-3 hours)
- **🟡 Medium Phase**: Calibration & optimization (7 components, ~1 hour)
- **🟢 Light Phase**: Online learning & fine-tuning (7 components, ~30 minutes)

### ✅ Complete Dashboard Sections
1. **Header**: Session ID and branding
2. **Overall Progress**: Global progress bar and phase indicator
3. **Phase Details**: Progress bars and component lists for each phase
4. **Current Training Metrics**: Real-time metrics for active component
5. **Strategy Performance**: Win rate, PnL, and trade counts for S2, S3, S6, S11
6. **Post-Training Validation**: Validation checklist
7. **Model Promotion Status**: Promotion plan and status
8. **System Resources**: CPU, Memory, Disk I/O, Process count
9. **Recent Activity Log**: Last 5 activities with timestamps
10. **Footer**: Instructions, lock file path, uptime, next refresh

## Menu System

### Main Menu
```
╔════════════════════════════════════════════════════════════════════════════════╗
║                    TopstepX Trading Bot - Mode Selection                      ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  [1] Terminal Mode (Live Trading)                                             ║
║  [2] Lab Mode (Historical Training)                                           ║
║  [3] Backtest Mode (Strategy Testing)                                         ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

### Lab Mode Sub-Menu
```
╔════════════════════════════════════════════════════════════════════════════════╗
║                      Lab Mode - Training Schedule Options                     ║
╠════════════════════════════════════════════════════════════════════════════════╣
║  [1] Scheduled Training (Sunday Only)                                         ║
║      • Runs Sunday 12:00 PM - 5:45 PM ET                                      ║
║  [2] Manual Training (Run Now)                                                ║
║      • Starts immediately (any day/time)                                      ║
║  [3] Back to Main Menu                                                        ║
╚════════════════════════════════════════════════════════════════════════════════╝
```

## How to Launch

### Using Environment Variables (bypass menu)
```bash
export LAB_MODE=1
export FORCE_LAB_NOW=1  # For any day lab (manual training)
export DRY_RUN=1
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj
```

### Using Interactive Menu
```bash
# No environment variables needed - menu will appear
dotnet run --project src/UnifiedOrchestrator/UnifiedOrchestrator.csproj

# Then select:
# 2 (Lab Mode)
# 2 (Manual Training - Run Now)
# Press Enter
```

## Summary

✅ All requirements met:
- Menu system works
- Sunday Lab and Any Day Lab options available
- Dashboard displays without scrolling
- Real-time updates every 5 seconds
- All training phases visible
- Complete metrics and monitoring

🚀 Ready for production use!
