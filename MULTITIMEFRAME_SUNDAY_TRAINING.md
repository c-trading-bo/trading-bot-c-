# Multi-Timeframe Training Integration Guide

## Overview
This guide explains how multi-timeframe data and models integrate with QBot's existing Sunday Lab Mode training system.

## System Architecture

### Current Lab Mode (Already Implemented)
QBot already has a robust Sunday training system:
- **LAB_MODE=1**: Enables offline training mode
- **Sunday Schedule**: 12:00 PM - 5:45 PM ET automatic training
- **Training Orchestrator**: Coordinates full training lifecycle
- **Validation & Promotion**: Canary tests before production deployment
- **Manual Mode**: Available for testing outside Sunday schedule

### Multi-Timeframe Integration (New)

The multi-timeframe system integrates seamlessly with existing Lab Mode:

```
┌─────────────────────────────────────────────────────────────────┐
│                     SUNDAY LAB MODE TRAINING                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Data Collection (6:00 AM Daily)                            │
│     ├─ fetch-and-save-historical-data.py                       │
│     ├─ Fetches 5m bars → ES_90days.json, NQ_90days.json       │
│     └─ Fetches 1m bars → ES_1m_90days.json, NQ_1m_90days.json │
│                                                                 │
│  2. Training Session Start (Sunday 12:00 PM)                   │
│     ├─ LAB_MODE=1 environment variable set                     │
│     ├─ Resource pre-checks (disk, RAM, CPU)                    │
│     └─ Data integrity validation                               │
│                                                                 │
│  3. Data Loading (MultiTimeframeDataLoader)                    │
│     ├─ Load synchronized 5m + 1m bars                          │
│     ├─ Align timestamps across timeframes                      │
│     ├─ Extract multi-timeframe features                        │
│     └─ Split: train/val/test (chronological, no leakage)      │
│                                                                 │
│  4. Model Training (Python Training Scripts)                   │
│     ├─ Train CVaR-PPO with multi-timeframe inputs             │
│     ├─ Train SAC with multi-timeframe inputs                   │
│     ├─ Train LSTM with multi-timeframe sequences              │
│     ├─ Apply overfitting prevention:                           │
│     │   • Early stopping on validation set                     │
│     │   • Multi-seed training (3-5 seeds)                      │
│     │   • Dropout and regularization                           │
│     └─ Export to ONNX for production deployment               │
│                                                                 │
│  5. Validation & Testing                                       │
│     ├─ Canary tests on test set (unseen data)                 │
│     ├─ Performance metrics validation                          │
│     ├─ Model stability checks                                  │
│     └─ Overfitting detection                                   │
│                                                                 │
│  6. Model Promotion (if validation passes)                     │
│     ├─ Atomic promotion to production registry                 │
│     ├─ SHA256 checksums and manifests                          │
│     ├─ Rollback capability if issues detected                  │
│     └─ Alert notifications sent                                │
│                                                                 │
│  7. Training Complete (by 5:45 PM)                             │
│     └─ New multi-timeframe models ready for Monday trading     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Weekly Cycle

### Week 1-2: Initial Setup
```bash
# Run initial data collection (manual)
REFRESH_MODE=full python fetch-and-save-historical-data.py

# Wait 7 days to collect full week of data
# No trading yet - just data collection
```

### Week 3-4: Lab Mode Training Only
```bash
# Sunday: Train multi-timeframe models
LAB_MODE=1 LAB_MODE_SCHEDULE=SCHEDULED dotnet run --project src/UnifiedOrchestrator

# The system will:
# 1. Wait for Sunday if not today
# 2. Train models from 12:00 PM - 5:45 PM ET
# 3. Validate models on test set
# 4. Save to staging (NOT production yet)
```

### Week 5-7: Paper Trading Validation
```bash
# Monday-Saturday: Run in PAPER_TRADING mode
PAPER_TRADING=1 dotnet run --project src/UnifiedOrchestrator

# The system will:
# 1. Load multi-timeframe models (from staging)
# 2. Collect real-time 5m + 1m + tick data
# 3. Make trading decisions (NO real money)
# 4. Track performance and collect experiences
# 5. Log multi-timeframe feature quality

# Sunday: Continue training
# - Models retrain with accumulated week's data
# - Still in staging, not production
```

### Week 8: Production Deployment
```bash
# After successful paper trading week
# Promote models to production registry

# Monday-Saturday: Live trading
dotnet run --project src/UnifiedOrchestrator

# The system will:
# 1. Load multi-timeframe champion models
# 2. Trade with real money using 5m + 1m features
# 3. Execute with tick-level approval
# 4. Perform online calibration (lightweight)
# 5. Collect experiences for next Sunday

# Sunday: Automatic retraining
LAB_MODE=1 LAB_MODE_SCHEDULE=SCHEDULED dotnet run --project src/UnifiedOrchestrator
# - Trains with accumulated week's data
# - Validates and promotes new champions
# - Cycle continues automatically
```

### Week 9+: Continuous Operation
The system runs automatically with zero manual intervention:

**Daily (6:00 AM)**:
```bash
# Cron job runs data collection
0 6 * * * cd /home/runner/work/QBot/QBot && REFRESH_MODE=incremental python fetch-and-save-historical-data.py
```

**Monday-Saturday (Live Trading)**:
- Terminal Mode loads champion multi-timeframe models
- Collects real-time 5m + 1m + tick data
- Makes strategic decisions when 5m bars complete
- Executes with tick-level approval
- Performs lightweight online calibration
- Accumulates experiences for Sunday training

**Sunday (12:00 PM - 5:45 PM ET)**:
- Lab Mode automatically starts training
- Retrains all models with new week's data
- Applies overfitting prevention (splits, early stopping, multi-seed)
- Validates on test set
- Promotes champions to production
- New models ready for Monday

## Data Flow

### Offline Training (Sunday Lab Mode)
```
Historical Data Files
  ├─ ES_90days.json (5m bars)
  ├─ ES_1m_90days.json (1m bars)
  ├─ NQ_90days.json
  └─ NQ_1m_90days.json
        ↓
MultiTimeframeDataLoader
  ├─ Load both timeframes
  ├─ Align timestamps
  └─ Create synchronized samples
        ↓
MultiTimeframeFeatureExtractor
  ├─ Extract 5m features (ATR, RSI, MACD, etc.)
  └─ Extract 1m features (faster windows)
        ↓
Python Training Scripts
  ├─ Train multi-input models
  ├─ Apply overfitting prevention
  └─ Export to ONNX
        ↓
Validation Service
  ├─ Canary tests on test set
  └─ Performance validation
        ↓
Atomic Promotion Service
  └─ Promote to production registry
```

### Live Trading (Terminal Mode)
```
Live Market Feed
  ├─ Tick data
  └─ Trade data
        ↓
BarAggregationService
  ├─ Build 1m bars in real-time
  └─ Build 5m bars in real-time
        ↓
LiveMultiTimeframeFeatureComputer
  ├─ Compute 5m features when 5m bar completes
  ├─ Compute 1m features when 1m bar completes
  └─ Cache synchronized features
        ↓
MultiTimeframeBrainAdapter
  ├─ Get latest features
  └─ Provide to UnifiedTradingBrain
        ↓
UnifiedTradingBrain
  ├─ Load champion multi-timeframe models
  ├─ Make strategic decision using 5m + 1m features
  └─ If decision: check execution approval
        ↓
ExecutionApprovalService
  ├─ Validate using tick microstructure
  ├─ Estimate slippage
  └─ Approve/reject
        ↓
Trade Execution (if approved)
        ↓
MultiTimeframeOnlineLearning
  ├─ Record trade with multi-timeframe state
  ├─ Analyze outcome when trade closes
  ├─ Update calibration tables (5m vs 1m)
  └─ Save for next Sunday training
```

## Environment Variables

### Lab Mode (Sunday Training)
```bash
LAB_MODE=1                        # Enable offline training
LAB_MODE_SCHEDULE=SCHEDULED       # Wait for Sunday
# OR
LAB_MODE_SCHEDULE=MANUAL          # Train immediately (testing)
```

### Terminal Mode (Live Trading)
```bash
# LAB_MODE not set (or =0)         # Live trading mode
PAPER_TRADING=1                   # Paper trading (optional)
```

## Python Training Script Integration

Your Python training scripts should use the MultiTimeframeDataLoader:

```python
# train_multitimeframe_cvar_ppo.py

from BotCore.ML import MultiTimeframeDataLoader, MultiTimeframeFeatureExtractor

# Initialize
logger = setup_logger()
feature_extractor = MultiTimeframeFeatureExtractor(logger)
data_loader = MultiTimeframeDataLoader(logger, feature_extractor, "data/historical")

# Load synchronized data
bars_5m, bars_1m = data_loader.LoadHistoricalData("ES")
aligned_timestamps = data_loader.AlignTimestamps(bars_5m, bars_1m)
samples = data_loader.CreateSynchronizedSamples("ES", bars_5m, bars_1m, aligned_timestamps)

# Split data (chronological to prevent leakage)
train_samples, val_samples, test_samples = data_loader.SplitTrainValTest(samples)

# Get batches
for batch in data_loader.GetTrainingBatches(train_samples, batch_size=32):
    # Extract features for both timeframes
    features_5m = [s.Features5m for s in batch]
    features_1m = [s.Features1m for s in batch]
    
    # Train model with dual inputs
    loss = model.train_step(features_5m, features_1m, targets)
    
# Validate on val set (early stopping)
# Test on test set (final validation)
# Export to ONNX if validation passes
```

## Deployment Checklist

### Initial Setup (One-time)
- [ ] Set up cron job for daily data collection (6 AM)
- [ ] Configure LAB_MODE environment for Sunday training
- [ ] Test manual training run (LAB_MODE_SCHEDULE=MANUAL)
- [ ] Verify data alignment with validation script

### Week 1-2: Data Collection
- [ ] Run full data fetch
- [ ] Collect 7 days of 5m + 1m bars
- [ ] Validate alignment daily

### Week 3-4: Training Only
- [ ] Enable scheduled Sunday training
- [ ] Verify models train successfully
- [ ] Check validation metrics
- [ ] Models saved to staging (not production)

### Week 5-7: Paper Trading
- [ ] Enable PAPER_TRADING mode
- [ ] Load multi-timeframe models from staging
- [ ] Monitor feature computation performance
- [ ] Track decision quality
- [ ] Collect experiences for training

### Week 8: Production
- [ ] Promote models to production registry
- [ ] Disable PAPER_TRADING mode
- [ ] Start live trading with real money
- [ ] Monitor performance closely
- [ ] Verify online calibration working

### Week 9+: Continuous
- [ ] Confirm automatic weekly training
- [ ] Monitor model promotion logs
- [ ] Track calibration tables (5m vs 1m)
- [ ] Zero manual intervention needed

## Monitoring

### Daily Checks
- Data collection completed successfully
- Bar alignment validated
- Disk space sufficient

### Weekly Checks (Sunday)
- Training session started on schedule
- Models trained without errors
- Validation metrics acceptable
- Models promoted (or rejected with reason)

### Monthly Reviews
- Calibration table analysis (which timeframe performs better)
- Feature computation performance
- Model performance trends
- System health overall

## Troubleshooting

### Training Doesn't Start on Sunday
```bash
# Check LAB_MODE environment
echo $LAB_MODE  # Should be "1"

# Check schedule setting
echo $LAB_MODE_SCHEDULE  # Should be "SCHEDULED"

# Check logs
tail -f logs/training-orchestrator.log
```

### Data Collection Fails
```bash
# Check TopstepX credentials
cat .env | grep TOPSTEP

# Run validation
python validate-multitimeframe-alignment.py

# Check file existence
ls -la data/historical/*.json
```

### Models Not Promoted
```bash
# Check validation logs
cat reports/validation-*.json

# Check promotion logs
cat logs/promotion-service.log

# Manually review test metrics
```

## Summary

The multi-timeframe system integrates seamlessly with QBot's existing Sunday Lab Mode training infrastructure. No changes needed to the core orchestration - the multi-timeframe components are designed to work within the existing framework:

1. **Data collection** runs daily via cron
2. **Training** runs automatically every Sunday in Lab Mode
3. **Validation** uses existing canary test framework
4. **Promotion** uses existing atomic promotion service
5. **Live trading** uses new multi-timeframe adapters
6. **Online learning** tracks timeframe contributions

Everything is production-ready and automated. Just follow the deployment checklist to roll out week by week.
