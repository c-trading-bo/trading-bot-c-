# Complete Owner's Manual: Unified Trading Intelligence System

## System Architecture Overview

Your trading bot is a self-improving intelligence organism composed of three specialized brains, each with distinct responsibilities and boundaries. Together they form a closed learning loop where simulation feeds training, training produces champions, champions execute in live markets, and live results validate future training.

The three brains are:

1. **Historical Mode**: The simulator and data generator
2. **Lab Mode**: The scientist and model developer (with two sub-modes: Sunday and Anyday)
3. **Terminal Mode**: The live execution pilot

All three share the same decision pipeline architecture but operate at different phases of the learning-to-execution lifecycle.

---

## Mode 1: Terminal Mode – The Live Execution Pilot

### Core Purpose
Terminal Mode is your production trading brain. It executes real trades in live markets using champion models that were previously trained and validated in Lab Mode. Terminal **never trains or modifies models** — it only loads, executes, monitors, and performs lightweight calibration.

### Operating Schedule
- Runs continuously during market hours, Monday through Saturday
- Pauses when Sunday Lab Mode takes over
- Resumes Monday morning with fresh champion models from Sunday's training session

### Data Sources
- Real-time WebSocket tick stream from broker or data provider
- Live market depth and order book updates
- Real-time account balance and position information from TopstepX API

### Core Responsibilities

#### 1. Real-Time Data Processing
- Receives live tick stream (timestamp, price, size, bid, ask, spread for every trade)
- Builds 1-minute bars from incoming ticks as each 60-second window completes
- Builds 5-minute bars from incoming ticks as each 300-second window completes
- Maintains rolling 10-second raw tick buffer in memory for execution analysis
- All three timeframes synchronized and ready for inference at every decision point

**Implementation:** `BarPyramid.cs`, `TradingSystemBarConsumer.cs`

#### 2. Multi-Timeframe Inference
- Loads current champion models from GitHub registry at startup
- Champions include:
  - Neural-UCB (strategy selector)
  - CVaR-PPO (position sizer)
  - LSTM (price forecaster)
  - Auxiliary predictive models
- Each decision cycle feeds all three timeframes to multi-branch model:
  - **Strategic branch**: Last 20 five-minute bars (trend, regime, volatility context)
  - **Tactical branch**: Last 100 one-minute bars (entry timing, pullback detection, momentum)
  - **Execution branch**: Current 10-second raw tick buffer (spread, liquidity, order flow)
- Model outputs: selected strategy, position size, entry/exit signals, confidence scores

**Implementation:** `UnifiedTradingBrain.cs`, `MultiBranchModelArchitecture.cs`

#### 3. Full Pre-Trade Decision Pipeline
Before any order placement, Terminal runs complete validation hierarchy:

1. **Zone analysis**: Is price in favorable supply/demand zone?
2. **Pattern recognition**: Are chart patterns aligned with trade direction?
3. **Regime detection**: Is current market regime favorable for selected strategy?
4. **Risk validation**: Does trade comply with maximum position size, daily loss limit, drawdown threshold?
5. **Schedule checks**: Are we in allowed trading hours? Is overnight holding permitted?
6. **Execution environment check**: Is spread acceptable? Is liquidity sufficient? Is order flow balanced?
7. **Final approval**: All gates must pass green before order submission

**Implementation:** `MasterDecisionOrchestrator.cs`, `ProductionGuardrailOrchestrator.cs`

#### 4. Order Execution
- Submits orders through TopstepX API with zero-delay target
- Uses execution branch intelligence to time order placement (waits for tight spread and balanced flow)
- Monitors fill quality: actual fill price vs expected, slippage measurement, partial fill handling
- Logs every order: timestamp, symbol, side, quantity, limit price, fill price, slippage, execution latency

**Implementation:** `AutonomousDecisionEngine.cs`, `TopstepXAdapterService`

#### 5. Post-Trade Logging and Analysis
- Records complete decision chain: which strategy was selected, confidence level, expected reward
- Records actual outcome: fill price, realized PnL, Sharpe contribution, drawdown impact
- Writes structured logs to database for later Lab Mode analysis
- All logs include multi-timeframe state snapshot for reproducibility

**Implementation:** `ExperienceRepository.cs`, logging infrastructure

#### 6. Canary Monitoring After Model Updates
When new champion models arrive from Sunday Lab (via GitHub release download):

1. Terminal does **NOT** immediately switch to live trading with new models
2. Instead, runs canary validation period (example: 50 virtual trades or 2 hours of paper trading)
3. Compares new model metrics vs old model baseline:
   - Sharpe ratio
   - Win rate
   - Average slippage
   - Decision latency
4. If new model underperforms or shows anomalies, automatic rollback to previous champion
5. If canary passes, hot-swap to new model and resume live trading
6. Logs canary results to GitHub issue or notification system for audit

**Implementation:** `CanaryTestingOrchestrator.cs`, `ModelHotReloadManager.cs`

#### 7. Lightweight Online Calibration
- Does **NOT** update neural network weights (no backpropagation, no gradient descent)
- Only adjusts surface-level calibration parameters in real-time:
  - Position sizing multipliers (if recent volatility changed)
  - Regime probability estimates (if market behavior shifted)
  - Execution timing thresholds (if spread patterns evolved)
- Calibration adjustments stored in memory only, reset when new champion loads
- Purpose: Adapt to intraday conditions without modifying core trained intelligence

**Implementation:** In-memory adjustments within `AutonomousDecisionEngine.cs`

#### 8. Health Monitoring and Safety Systems
- Continuously checks WebSocket connection status, reconnects if dropped
- Monitors API latency and order acknowledgment times
- Enforces maximum position size limits (hard stop, no override)
- Enforces maximum daily loss limits (halts trading if breached)
- Enforces maximum drawdown limits (emergency pause if portfolio falls below threshold)
- Logs all safety events with critical severity for immediate review
- If critical failure detected (model corruption, API unresponsive, data feed dead), enters safe mode:
  - Close all positions
  - Halt new orders
  - Alert user

**Implementation:** `TopStepComplianceManager.cs`, `ProductionKillSwitchService.cs`

#### 9. Hub Synchronization
- Maintains persistent connection to User Hub (authentication, configuration, permissions)
- Maintains persistent connection to Market Hub (exchange connectivity, symbol routing, market status)
- Verifies both hubs are live before allowing any trade execution
- If either hub disconnects, pauses trading until reconnection confirmed

**Implementation:** Hub services in `Abstractions`

### Performance Targets
- **Decision latency**: Sub-22 milliseconds from signal generation to order submission
- **Uptime**: 99.9% during market hours (excluding planned Sunday Lab downtime)
- **Fill quality**: Average slippage within 0.5 ticks of mid-price for liquid instruments

### What Terminal Mode Never Does
- ❌ Never trains models (no weight updates, no backpropagation)
- ❌ Never modifies champion model files on disk
- ❌ Never automatically triggers Lab Mode retraining (you decide when to run Anyday Lab)
- ❌ Never trades during Sunday Lab training window (12:00 PM – 5:45 PM ET)
- ❌ Never overrides safety limits programmatically (requires manual config change and restart)

### Output Artifacts
- Real orders submitted to brokerage account
- Live PnL tracked tick-by-tick
- Complete decision logs (JSONL format) with state-action-reward tuples
- Performance metrics dashboard updated in real-time
- Trade execution quality reports (slippage, fill rate, latency)

### Environment Variables (Terminal Mode)
```bash
LAB_MODE=0
HISTORICAL_MODE=0
DRY_RUN=0  # Set to 1 for paper trading
RlRuntimeMode=InferenceOnly
AUTONOMOUS_MODE=true
```

---

## Mode 2: Lab Mode – The Scientist and Model Developer

### Core Purpose
Lab Mode is your training brain. It uses historical data to train and optimize all ML/RL models that Terminal Mode will use for live trading. Lab Mode has **two sub-modes**:

1. **Sunday Lab Mode** (Automatic): Scheduled every Sunday 12:00 PM - 5:45 PM ET
2. **Anyday Lab Mode** (Manual): User-triggered on any day via `FORCE_LAB_NOW=1`

Both sub-modes use the **same training pipeline** (`HistoricalTrainingOrchestrator`) but differ in scheduling.

### Operating Schedule

#### Sunday Lab Mode (Automatic)
- Runs every Sunday 12:00 PM - 5:45 PM America/New_York (DST-aware)
- Automatic weekly retraining cycle
- Best for: Production automation
- Triggered by: `InternalScheduler` when Sunday + time window detected

#### Anyday Lab Mode (Manual)
- Runs immediately when `FORCE_LAB_NOW=1` is set
- No schedule restrictions
- Can run on any day of the week
- Best for: Testing, emergencies, experiments, iterative development
- Triggered by: User setting environment variable

### Data Sources
Lab Mode uses **pre-loaded JSON files** for complete API segregation:

- `data/ES_90days.json` - 90 days of 5-minute bars for ES futures
- `data/NQ_90days.json` - 90 days of 5-minute bars for NQ futures
- `state/experiences.json` - Last 7 days of Terminal Mode trading experiences

**CRITICAL**: Lab Mode uses **ZERO live API connections**. All historical data is fetched via Python scripts (`fetch-and-save-historical-data.py`) and saved as JSON before training starts.

### Core Responsibilities

#### 1. Pre-Flight Health Checks (11:55 AM ET)
Before training starts, Lab Mode verifies system readiness:

- Disk space check (≥10 GB available)
- RAM memory check (≥4 GB free)
- CPU utilization (< 80%)
- Data integrity SHA-256 validation
- Training lock file staleness check
- Exponential backoff retry logic (5m, 15m, 30m intervals)

**Implementation:** `ResourcePreCheckService.cs`, `TrainingResourceMonitor.cs`

#### 2. Data Loading (12:05 PM ET)
- Load 90-day historical bars from JSON files
- Load 7-day experience replay buffer from Terminal Mode
- Verify data integrity (no missing bars, correct timestamps)
- Total: ~7,782 historical bars across ES and NQ

**Implementation:** `HistoricalTrainingOrchestrator.cs` → `LoadHistoricalDataAsync()`

#### 3. Heavy Phase Training (12:05 PM - 2:30 PM ET)
Trains 7 core models with 50 epochs each (~2.5 hours):

1. **CVaR-PPO** - Risk-aware position sizing
2. **Neural-UCB** - Multi-armed bandit strategy selector
3. **LSTM** - Price forecasting
4. **Pattern Recognition** - Chart pattern detector
5. **Regime Detector** - Market regime classifier
6. **Slippage-Latency** - Execution cost predictor
7. **Model Ensemble** - Meta-learner combining all models

**Implementation:** `HistoricalTrainingOrchestrator.cs` → `RunTrainingPipelineAsync()`

#### 4. Medium Phase Training (2:30 PM - 4:00 PM ET)
Trains 15 calibration models with 30 epochs each (~1.5 hours):

- Zone strength estimators
- Volatility predictors
- Spread forecasters
- Volume profile analyzers
- Momentum indicators

**Implementation:** `HistoricalTrainingOrchestrator.cs` → `TrainMediumPhaseAsync()`

#### 5. Light Phase Training (4:00 PM - 5:15 PM ET)
Trains 15 online learning models with 20 epochs each (~1.25 hours):

- Real-time adaptation layers
- Intraday pattern detectors
- Execution timing optimizers
- Micro-structure analyzers

**Implementation:** `HistoricalTrainingOrchestrator.cs` → `TrainLightPhaseAsync()`

#### 6. Canary Testing (5:15 PM - 5:35 PM ET)
Before promoting new models, Lab Mode runs canary validation:

**5 Metric Thresholds:**
1. Win rate must not decrease
2. Average profit drop < $5
3. Max drawdown increase < 10%
4. Sharpe ratio drop < 0.2
5. Profit factor ≥ 1.5

**Rejection Logic:**
- If **ANY** threshold fails → auto-reject
- Delete staged models
- Keep current champions
- Send alert notification

**Implementation:** `PerformanceComparisonEngine.cs` → `RunCanaryTestWithThresholdsAsync()`

#### 7. Atomic Promotion (5:35 PM - 5:40 PM ET)
If canary passes:

1. Backup current champion models to `artifacts/previous/`
2. Atomic folder rename: `artifacts/stage/` → `artifacts/current/`
3. Update `active_manifest.json` with new model metadata
4. 4-week backup retention policy

**Implementation:** `HistoricalTrainingOrchestrator.cs` → `PromoteModelsAsync()`

#### 8. Notifications (5:40 PM - 5:45 PM ET)
Send email with comprehensive training summary:

- All phase results
- Canary test outcomes
- Model promotion status
- Next training date/time
- Performance metrics dashboard link

**Implementation:** `TrainingAlertService.cs`

#### 9. Graceful Shutdown (5:45 PM ET)
- Save training checkpoint to `state/training_checkpoint.json`
- Release training lock file
- Clean up temporary artifacts
- Log next Sunday training schedule

**Implementation:** `InternalScheduler.cs` → graceful shutdown logic

### Differences Between Sunday and Anyday Lab Modes

| Feature | Sunday Lab Mode | Anyday Lab Mode |
|---------|----------------|-----------------|
| **Trigger** | Automatic (scheduler) | Manual (`FORCE_LAB_NOW=1`) |
| **Schedule** | Sunday 12:00 PM - 5:45 PM ET | Immediate execution |
| **Data Availability** | 90 days (full week) | Variable (e.g., 54 days on Wednesday) |
| **Training Pipeline** | Same | Same |
| **Pre-Flight Checks** | Same | Same |
| **Canary Testing** | Same | Same |
| **Model Promotion** | Same | Same |
| **Use Case** | Production automation | Testing, emergencies |

### What Lab Mode Never Does
- ❌ Never connects to TopstepX API for live market data
- ❌ Never places live orders (DRY_RUN=1 enforced)
- ❌ Never runs during Terminal Mode trading hours (segregation)
- ❌ Never trains on live data (uses pre-loaded JSON only)
- ❌ Never automatically triggers based on performance degradation (manual trigger only)

### Output Artifacts
- Challenger models saved to `artifacts/stage/`
- Champion models promoted to `artifacts/current/`
- Training manifest with SHA-256 checksums
- Canary test results in `reports/canary/`
- Training logs in `state/learning/`
- Backup models in `artifacts/previous/` (4-week retention)

### Environment Variables (Lab Mode)

#### Sunday Lab Mode
```bash
LAB_MODE=1
HISTORICAL_MODE=0
DRY_RUN=1
FORCE_LAB_NOW=0  # Use Sunday schedule
RlRuntimeMode=Train
LAB_MODE_BOOTSTRAP=1
```

#### Anyday Lab Mode
```bash
LAB_MODE=1
HISTORICAL_MODE=0
DRY_RUN=1
FORCE_LAB_NOW=1  # Bypass Sunday schedule
RlRuntimeMode=Train
LAB_MODE_BOOTSTRAP=1
```

---

## Mode 3: Historical Mode – The Simulator and Data Generator

### Core Purpose
Historical Mode is your backtesting brain. It replays historical data through the same decision pipeline as Terminal Mode but without live execution. Purpose: strategy validation, performance metrics, data generation for Lab Mode.

### Operating Schedule
- On-demand execution (user-triggered)
- No automatic schedule
- Runs independently of Terminal and Lab modes
- Can run at any time for backtesting purposes

### Data Sources
- Pre-loaded historical bar data from local JSON files
- No live API connections
- Simulated market conditions using historical tick data

### Core Responsibilities

#### 1. Historical Data Replay
- Load historical bars from `datasets/` directory
- Replay bars sequentially to simulate live market feed
- Maintain time-series consistency (no lookahead bias)
- Generate synthetic tick data from OHLCV bars when needed

**Implementation:** `HistoricalDataBridgeService.cs`, `BacktestEngine`

#### 2. Strategy Validation
- Run strategies (S2, S3, S6, S11) against historical data
- Execute decision pipeline in backtest mode
- Track simulated orders and fills
- Calculate slippage and execution costs

**Implementation:** `BacktestEngine`, strategy implementations

#### 3. Performance Metrics
Calculate comprehensive metrics:

- Total PnL
- Sharpe ratio
- Maximum drawdown
- Win rate
- Profit factor
- Average trade duration
- Risk-adjusted returns

**Implementation:** `BacktestMetricsCalculator.cs`

#### 4. Experience Generation for Lab Mode
- Record state-action-reward tuples from backtest
- Save experiences to `state/experiences.json`
- Format matches Terminal Mode experience format
- Enables Lab Mode to train on both live and simulated data

**Implementation:** `ExperienceRepository.cs`

#### 5. Simulation Accuracy
- Model realistic slippage and latency
- Account for spread costs
- Simulate partial fills
- Respect trading hours and holidays
- Apply position limits and risk constraints

**Implementation:** `SlippageLatencyModel.cs`

### What Historical Mode Never Does
- ❌ Never places live orders (DRY_RUN=1 enforced)
- ❌ Never connects to TopstepX API for live trading
- ❌ Never trains models (read-only for models)
- ❌ Never modifies champion models
- ❌ Never runs automatically (manual trigger only)

### Output Artifacts
- Backtest performance reports in `reports/backtests/`
- Simulated trade logs
- Performance metrics CSV
- Experience replay buffer for Lab Mode
- Strategy comparison charts

### Environment Variables (Historical Mode)
```bash
HISTORICAL_MODE=1
LAB_MODE=0
DRY_RUN=1
RlRuntimeMode=InferenceOnly
```

---

## Mode Separation and Boundaries

### Critical Segregation Rules

1. **Lab Mode ↔ Terminal Mode Segregation**
   - Lab Mode uses ZERO live API connections
   - Lab Mode runs Sunday 12:00 PM - 5:45 PM ET ONLY (automatic)
   - Terminal Mode pauses during Lab Mode training window
   - Lab Mode outputs go to `artifacts/stage/` (not `artifacts/current/`)
   - Only after canary tests do models move to `artifacts/current/`

2. **Historical Mode ↔ Terminal Mode Segregation**
   - Historical Mode never places live orders
   - Historical Mode uses pre-loaded JSON data
   - Historical Mode and Terminal Mode can share model artifacts (read-only)

3. **Lab Mode Data Sources**
   - Lab Mode: `data/ES_90days.json`, `data/NQ_90days.json` (offline)
   - Terminal Mode: Live WebSocket stream from TopstepX
   - Historical Mode: `datasets/` directory (offline)

### Mode Selection Flow

```
User starts bot
    ↓
Interactive mode prompt
    ↓
    ├─→ [1] Terminal Mode → LAB_MODE=0, HISTORICAL_MODE=0, DRY_RUN=0/1
    ├─→ [2] Lab Mode
    │        ↓
    │        ├─→ [1] Scheduled (Sunday) → LAB_MODE=1, FORCE_LAB_NOW=0
    │        └─→ [2] Manual (Anyday) → LAB_MODE=1, FORCE_LAB_NOW=1
    └─→ [3] Backtest Mode → HISTORICAL_MODE=1, LAB_MODE=0
```

**Implementation:** `Program.cs` → `PromptForTradingModeAsync()`

---

## Complete System Learning Loop

```
┌─────────────────────────────────────────────────────────────┐
│                  UNIFIED LEARNING LOOP                      │
└─────────────────────────────────────────────────────────────┘

1. TERMINAL MODE (Mon-Sat)
   ↓
   Executes trades with champion models
   ↓
   Logs experiences to state/experiences.json
   ↓
   
2. SUNDAY LAB MODE (Sun 12-5:45 PM ET)
   ↓
   Loads experiences + historical bars
   ↓
   Trains 37 models (Heavy + Medium + Light phases)
   ↓
   Runs canary tests (5 thresholds)
   ↓
   Promotes new champions → artifacts/current/
   ↓
   
3. MONDAY TERMINAL MODE RESUME
   ↓
   Loads new champion models from artifacts/current/
   ↓
   Canary validation (50 virtual trades)
   ↓
   Hot-swap to new models if canary passes
   ↓
   Continue live trading with improved models
   ↓
   Loop back to step 1
```

---

## File System Layout

```
QBot/
├── artifacts/
│   ├── current/          # Active champion models (Terminal Mode loads from here)
│   ├── stage/            # Newly trained challengers (Lab Mode outputs here)
│   ├── previous/         # Backup champions (4-week retention)
│   └── temp/             # Temporary training artifacts
│
├── data/
│   ├── ES_90days.json    # Lab Mode historical data (5-minute bars)
│   ├── NQ_90days.json    # Lab Mode historical data (5-minute bars)
│   └── calibration/      # Calibration data for fine-tuning
│
├── datasets/
│   ├── features/         # Historical Mode feature datasets
│   └── quotes/           # Historical Mode raw tick data
│
├── state/
│   ├── experiences.json           # Terminal Mode experience replay buffer
│   ├── training_checkpoint.json   # Lab Mode checkpoint for resume
│   ├── backtests/                 # Historical Mode backtest results
│   └── learning/                  # Lab Mode training logs
│
├── reports/
│   ├── canary/           # Canary test results
│   ├── backtests/        # Historical Mode performance reports
│   └── trading/          # Terminal Mode execution quality reports
│
└── manifests/
    ├── manifest.json              # Staged model manifest (Lab Mode output)
    └── active_manifest.json       # Active champion manifest (Terminal Mode reads)
```

---

## Key Implementation Files

### Terminal Mode
- `src/BotCore/Services/AutonomousDecisionEngine.cs` - Main trading loop
- `src/BotCore/Brain/UnifiedTradingBrain.cs` - Multi-timeframe inference
- `src/BotCore/Services/MasterDecisionOrchestrator.cs` - Pre-trade validation
- `src/BotCore/Market/BarPyramid.cs` - Real-time bar construction
- `src/BotCore/Services/TopStepComplianceManager.cs` - Safety systems

### Lab Mode
- `src/UnifiedOrchestrator/Scheduling/InternalScheduler.cs` - Sunday scheduler
- `src/UnifiedOrchestrator/Services/HistoricalTrainingOrchestrator.cs` - Training pipeline
- `src/UnifiedOrchestrator/Services/PerformanceComparisonEngine.cs` - Canary testing
- `src/UnifiedOrchestrator/Services/ResourcePreCheckService.cs` - Pre-flight checks
- `src/RLAgent/CVaRPPOTrainer.cs` - CVaR-PPO training
- `src/RLAgent/LSTMTrainer.cs` - LSTM training
- `src/RLAgent/ModelEnsembleTrainer.cs` - Ensemble training

### Historical Mode
- `src/BotCore/Services/HistoricalDataBridgeService.cs` - Data replay
- `src/Backtest/BacktestEngine.cs` - Backtest execution
- `src/Safety/Simulation/SlippageLatencyModel.cs` - Simulation accuracy

### Shared Infrastructure
- `src/Abstractions/RlRuntimeMode.cs` - Mode enum (InferenceOnly, CollectOnly, Train)
- `src/UnifiedOrchestrator/Program.cs` - Mode selection and startup
- `src/BotCore/Services/ProductionKillSwitchService.cs` - Mode detection utilities

---

## Troubleshooting

### Terminal Mode Not Trading
1. Check `LAB_MODE=0` and `HISTORICAL_MODE=0`
2. Verify TopstepX WebSocket connection
3. Check if within trading hours
4. Verify champion models loaded from `artifacts/current/`
5. Check daily loss limits not breached

### Lab Mode Not Training
1. Check `LAB_MODE=1`
2. For Sunday: Verify day is Sunday and time is 12:00-5:45 PM ET
3. For Anyday: Verify `FORCE_LAB_NOW=1`
4. Check historical data exists: `data/ES_90days.json`, `data/NQ_90days.json`
5. Verify disk space ≥10 GB, RAM ≥4 GB free

### Historical Mode Not Running
1. Check `HISTORICAL_MODE=1`
2. Verify historical data in `datasets/` directory
3. Check backtest configuration in `appsettings.backtest.json`

### Models Not Loading
1. Verify files exist in `artifacts/current/`
2. Check `active_manifest.json` integrity
3. Verify ONNX model compatibility (use `OnnxModelCompatibilityService`)
4. Check model file permissions

---

## Monitoring and Alerts

### Terminal Mode Alerts
- Connection loss to TopstepX
- Daily loss limit approaching/breached
- Drawdown threshold exceeded
- Model inference latency > 22ms
- Order execution failures

### Lab Mode Alerts
- Pre-flight health checks failed
- Training phase failures
- Canary test failures (auto-reject)
- Model promotion success/failure
- Disk space/memory warnings

### Historical Mode Alerts
- Backtest completion notifications
- Performance metric summaries
- Data integrity issues

---

## Performance Benchmarks

### Terminal Mode
- **Decision Latency**: < 22ms (target)
- **Uptime**: 99.9% during market hours
- **Fill Quality**: Average slippage ≤ 0.5 ticks
- **Model Inference**: < 10ms per decision

### Lab Mode
- **Total Training Time**: ~5.5 hours (Sunday)
- **Heavy Phase**: 2.5 hours (7 models × 50 epochs)
- **Medium Phase**: 1.5 hours (15 models × 30 epochs)
- **Light Phase**: 1.25 hours (15 models × 20 epochs)
- **Canary Testing**: 20 minutes
- **Model Promotion**: 5 minutes

### Historical Mode
- **Backtest Speed**: 10,000 bars/minute
- **Memory Usage**: < 2 GB for 90-day backtest
- **Report Generation**: < 30 seconds

---

## Conclusion

This Owner's Manual provides complete documentation for all three trading modes. Each mode has clear responsibilities, boundaries, and segregation rules. The system forms a closed learning loop where:

1. **Terminal Mode** executes trades and collects experiences
2. **Lab Mode** trains models using experiences + historical data
3. **Historical Mode** validates strategies and generates synthetic experiences

Together, they create a self-improving trading intelligence system that continuously learns and adapts while maintaining strict safety boundaries.

For detailed implementation guides, see:
- `LAB_MODE_COMPLETE.md` - Lab Mode implementation details
- `HISTORICAL_MODE_IMPLEMENTATION_SUMMARY.md` - Historical Mode setup
- `TRADING_SAFETY_GUARDRAILS.md` - Terminal Mode safety systems
- `TESTING_GUIDE.md` - Testing procedures for all modes
