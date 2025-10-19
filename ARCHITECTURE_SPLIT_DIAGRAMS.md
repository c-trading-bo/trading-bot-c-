# Architecture Split: Visual Diagrams

## Current State vs Future State

### Current Architecture (Monolith)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     UnifiedOrchestrator (Single Process)                │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                        Program.cs                                │  │
│  │                    (Single Entry Point)                          │  │
│  └─────────────────────┬───────────────────────────────────────────┘  │
│                        │                                               │
│        ┌───────────────┴────────────────┐                             │
│        │                                │                             │
│  ┌─────▼─────────┐              ┌──────▼──────────┐                  │
│  │ InferenceBrain│              │ TrainingBrain   │                  │
│  │ (Read-Only)   │              │ (Write-Only)    │                  │
│  ├───────────────┤              ├─────────────────┤                  │
│  │• PPO Router   │              │• CVaR-PPO Train │                  │
│  │• UCB Router   │              │• LSTM Train     │                  │
│  │• LSTM Router  │              │• UCB Train      │                  │
│  │• DecideAsync()│              │• Artifact Build │                  │
│  └───────┬───────┘              └────────┬────────┘                  │
│          │                               │                            │
│          │    ⚠️ SAME MEMORY SPACE ⚠️    │                            │
│          │    ⚠️ SAME CPU/THREADS ⚠️     │                            │
│          │                               │                            │
│  ┌───────▼───────────────────────────────▼────────┐                  │
│  │         UnifiedOrchestratorService              │                  │
│  ├─────────────────────────────────────────────────┤                  │
│  │• SignalR Client (TopstepX)                      │                  │
│  │• Bar Builder                                    │                  │
│  │• Feature Engineering                            │                  │
│  │• Decision Fusion                                │                  │
│  │• Position Management                            │                  │
│  │• Safety Gates                                   │                  │
│  │• BrainHotReloadService (15-min file poll)       │                  │
│  └─────────────────────────────────────────────────┘                  │
│                                                                         │
│  Issues with Current Design:                                           │
│  ❌ Training bugs can crash live trading                               │
│  ❌ Can't deploy training updates independently                        │
│  ❌ Slow model updates (15-min polling)                                │
│  ❌ Resource contention (CPU/memory)                                   │
│  ❌ Hard to audit/debug (mixed logs)                                   │
│  ❌ Can't scale training independently                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

### Future Architecture (Clean Split)

```
┌─────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│      LIVE BOT (Inference Only)          │   │   TRAINER/GYM (Learning Only)           │
│        UnifiedOrchestrator               │   │     Standalone Trainer                  │
│                                         │   │                                         │
│  ┌───────────────────────────────────┐  │   │  ┌───────────────────────────────────┐  │
│  │       InferenceBrain              │  │   │  │       TrainingBrain               │  │
│  │       (Read-Only)                 │  │   │  │       (Write-Only)                │  │
│  ├───────────────────────────────────┤  │   │  ├───────────────────────────────────┤  │
│  │ • PPO Router → model.onnx         │  │   │  │ • CVaR-PPO Training Loop          │  │
│  │ • UCB Router → ucb.onnx           │  │   │  │ • LSTM Training Loop              │  │
│  │ • LSTM Router → lstm.onnx         │  │   │  │ • Neural UCB Training Loop        │  │
│  │ • DecideAsync() - FAST            │  │   │  │ • TrainChallengerAsync()          │  │
│  │ • Zero Training Code              │  │   │  │ • Zero Inference Code             │  │
│  └───────────────┬───────────────────┘  │   │  └───────────────┬───────────────────┘  │
│                  │                       │   │                  │                       │
│  ┌───────────────▼───────────────────┐  │   │  ┌───────────────▼───────────────────┐  │
│  │   UnifiedOrchestratorService      │  │   │  │   TrainerOrchestratorService      │  │
│  ├───────────────────────────────────┤  │   │  ├───────────────────────────────────┤  │
│  │ • SignalR Client (TopstepX)       │  │   │  │ • Historical Data Fetcher         │  │
│  │ • Bar Builder                     │  │   │  │ • Experience Consumer (SQLite)    │  │
│  │ • Feature Engineering             │  │   │  │ • Backtest Engine                 │  │
│  │ • Decision Fusion                 │  │   │  │ • Parameter Optimizer             │  │
│  │ • Position Management (execute)   │  │   │  │ • Artifact Packager (zip)         │  │
│  │ • Safety Gates                    │  │   │  │ • Artifact Publisher (Redis)      │  │
│  │ • ExperienceBufferService (write) │  │   │  │ • Canary Evaluator                │  │
│  │ • ModelLoaderService (Redis sub)  │  │   │  │ • Manifest Generator              │  │
│  └───────────────┬───────────────────┘  │   │  └───────────────┬───────────────────┘  │
│                  │                       │   │                  │                       │
└──────────────────┼───────────────────────┘   └──────────────────┼───────────────────────┘
                   │                                              │
                   │                                              │
     Writes        │                                              │  Reads & Rotates
     Experiences   │                                              │  Experiences
                   │                                              │
                   ▼                                              │
         ┌─────────────────────┐                                 │
         │  EXPERIENCE BUFFER  │◄────────────────────────────────┘
         │     (SQLite)        │
         ├─────────────────────┤
         │ • Trade details     │
         │ • Features (JSON)   │
         │ • Actions (JSON)    │
         │ • Rewards, PnL      │
         │ • MAE/MFE           │
         │ • Timestamps        │
         │ • Model IDs         │
         └─────────────────────┘
                   ▲
                   │
                   │ Persists to disk
                   │ (/var/trader/experience.db)
                   │
                   
                   
         ┌─────────────────────────────────────────┐
         │         ARTIFACT STORE                  │
         │      (/var/artifacts/models/)           │
         ├─────────────────────────────────────────┤
         │                                         │
         │  model_ppo_v20251019.zip                │
         │  ├── model.onnx                         │
         │  ├── scaler.json                        │
         │  ├── policy_config.json                 │
         │  ├── manifest.json                      │
         │  └── eval_report.json                   │
         │                                         │
         │  model_ppo_v20251019.manifest.json      │
         │  ├── model_id                           │
         │  ├── checksum (SHA256)                  │
         │  ├── canary: true                       │
         │  ├── performance metrics                │
         │  └── feature_spec_id                    │
         │                                         │
         └─────────────────────────────────────────┘
                   ▲                    │
                   │                    │
        Trainer    │                    │  Live Bot
        Publishes  │                    │  Downloads
                   │                    ▼
                   │         
         ┌─────────┴────────────────────────────────┐
         │      REDIS PUB/SUB                       │
         │      (localhost:6379)                    │
         ├──────────────────────────────────────────┤
         │  Channel: "models:updates"               │
         │                                          │
         │  Message: {                              │
         │    "type": "model_update",               │
         │    "model_id": "ppo_v20251019",          │
         │    "artifact_url": "file://...",         │
         │    "tag": "canary",                      │
         │    "created_at": "2025-10-19T..."        │
         │  }                                       │
         │                                          │
         │  Latency: < 1 second                     │
         └──────────────────────────────────────────┘
                   ▲                    │
                   │                    │
      Trainer      │                    │  Live Bot
      Publishes    │                    │  Subscribes
      (after zip)  │                    │  (hot-swap)


Benefits of Split Architecture:
✅ Training crashes don't affect live trading
✅ Deploy Trainer updates without restarting Live Bot
✅ Instant model updates (Redis < 1sec vs file poll 15min)
✅ Independent resource limits (CPU/memory)
✅ Separate logs for easier debugging
✅ Clear compliance/audit boundaries
✅ Can run multiple Trainers (different algorithms)
✅ Test training changes safely
```

---

## Data Flow Diagrams

### 1. Live Trading Flow (No Changes)

```
Market Data         Live Bot                    Output
───────────        ──────────                  ────────
                   
TopstepX           SignalR Client
WebSocket    ──►   (TopstepAuthAgent)
                         │
                         │ Tick data
                         ▼
                   Bar Builder
                   (1-min bars)
                         │
                         │ OHLCV bars
                         ▼
                   Feature Engineering
                   (50+ features)
                         │
                         │ TradingContext
                         ▼
                   InferenceBrain
                   ├─ PPO.Decide()
                   ├─ UCB.Decide()
                   └─ LSTM.Decide()
                         │
                         │ TradingDecision
                         ▼
                   Decision Fusion
                   (weighted ensemble)
                         │
                         │ Final decision
                         ▼
                   Safety Gates
                   (CVaR, limits)
                         │
                         │ Approved action
                         ▼
                   Position Manager
                   (execute trade)
                         │
                         ├──────────────────────┐
                         │                      │
                         ▼                      ▼
                   TopstepX API          Experience Buffer
                   (place order)         (append SQLite)
                                               │
                                               │ Trade details
                                               ▼
                                         Disk Storage
                                         (for Trainer)

✅ This entire flow stays UNCHANGED
✅ Zero modifications to decision logic
✅ Just adds experience buffer append at end
```

---

### 2. Training Flow (New - Separate Process)

```
Experience         Trainer                     Output
──────────        ────────                    ────────

SQLite DB          ExperienceConsumer
/var/trader/       (reads last 100k rows)
experience.db ──►        │
                         │ List<Experience>
                         ▼
                   TrainingBrain
                   ├─ CVaRPPO.Train()
                   ├─ LSTM.Train()
                   └─ NeuralUCB.Train()
                         │
                         │ Trained models
                         ▼
                   Backtest Engine
                   (validate on holdout)
                         │
                         │ Performance metrics
                         ▼
                   Artifact Packager
                   ├─ model.onnx
                   ├─ manifest.json
                   ├─ scaler.json
                   └─ eval_report.json
                         │
                         │ .zip bundle
                         ▼
                   SHA256 Checksum
                         │
                         │ verified bundle
                         ▼
                   ArtifactPublisher
                   ├─ Copy to /var/artifacts/
                   └─ Publish to Redis
                         │
                         ├──────────────────────┐
                         │                      │
                         ▼                      ▼
                   Artifact Store         Redis Pub/Sub
                   (disk storage)         (notification)
                                               │
                                               │ models:updates
                                               ▼
                                         Live Bot
                                         (hot-swaps model)

🆕 This is all new (separate process)
🆕 Runs independently on same/different machine
🆕 Can crash without affecting live trading
```

---

### 3. Model Hot-Swap Flow (Enhanced)

```
Current:          Future (with Redis):
────────          ─────────────────

Trainer           Trainer
   │                 │
   │ Save            │ Package + Publish
   ▼                 ▼
/var/artifacts/   /var/artifacts/ + Redis
   │                 │            │
   │                 │            │ Instant notification
   │ 15-min poll     │            ▼
   │                 │         Live Bot
   ▼                 │         (subscriber)
Live Bot            │            │
   │                 │            │ < 1 second latency
   │ Detected!       │            │
   │                 │            ▼
   ▼                 │         Download & Verify
Load Model          │         (SHA256 checksum)
                    │            │
                    │            │ Checksum OK
                    │            ▼
                    │         Atomic Swap
                    │         ├─ Load to staging/
                    │         ├─ Validate model
                    │         ├─ Move active -> .old
                    │         └─ Move staging -> active
                    │            │
                    │            │ Swap complete
                    │            ▼
                    └──────►  InferenceBrain
                              (uses new model)

Improvement: 15 minutes → <1 second (900x faster)
```

---

## Migration Timeline

### Week 1-2: Foundation (40 hours)

```
┌────────────────────────────────────────────────────────┐
│ Week 1-2: Add Infrastructure to UnifiedOrchestrator   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Day 1-2:  SQLite Experience Buffer                   │
│            └─ ExperienceBufferService.cs              │
│            └─ Hook into UnifiedOrchestrator           │
│                                                        │
│  Day 3-4:  Redis Pub/Sub                              │
│            └─ RedisModelNotificationService.cs        │
│            └─ Replace file polling                    │
│                                                        │
│  Day 5-6:  Artifact Store Formalization               │
│            └─ Enforce manifest schema                 │
│            └─ Add canary fields                       │
│                                                        │
│  Day 7-8:  Docker & Config Split                      │
│            └─ Dockerfile                              │
│            └─ appsettings.livebot.json                │
│                                                        │
│  Status:   Live Bot still trains BUT also writes      │
│            experiences and subscribes to Redis        │
│            (backward compatible)                      │
└────────────────────────────────────────────────────────┘
```

### Week 3-4: Trainer Creation (40 hours)

```
┌────────────────────────────────────────────────────────┐
│ Week 3-4: Create Standalone Trainer (Shadow Mode)     │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Day 1-2:  Trainer Project Setup                      │
│            └─ src/Trainer/Program.cs                  │
│            └─ TrainerOrchestratorService.cs           │
│                                                        │
│  Day 3-4:  Experience Consumer                        │
│            └─ Read SQLite buffer                      │
│            └─ Rotation logic                          │
│                                                        │
│  Day 5-6:  Training Pipeline                          │
│            └─ Hook CVaR-PPO, LSTM, UCB               │
│            └─ Backtest validation                     │
│                                                        │
│  Day 7-8:  Artifact Publisher                         │
│            └─ Zip + manifest + Redis                  │
│            └─ Test end-to-end                         │
│                                                        │
│  Status:   Trainer trains and publishes BUT           │
│            Live Bot ignores artifacts                 │
│            (shadow mode - no impact)                  │
└────────────────────────────────────────────────────────┘
```

### Week 5-6: Canary Testing (40 hours)

```
┌────────────────────────────────────────────────────────┐
│ Week 5-6: Enable Canary Deployment                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Day 1-2:  Canary Monitoring                          │
│            └─ Track metrics separately                │
│            └─ PnL, drawdown, win rate                 │
│                                                        │
│  Day 3-4:  Canary @ 10%                               │
│            └─ 10% of decisions use Trainer models     │
│            └─ 90% use UnifiedOrchestrator models      │
│                                                        │
│  Day 5-6:  Validate & Increase to 50%                 │
│            └─ Compare performance                     │
│            └─ Test rollback mechanism                 │
│                                                        │
│  Day 7-8:  Increase to 100%                           │
│            └─ All decisions use Trainer models        │
│            └─ UnifiedOrchestrator still trains        │
│                                                        │
│  Status:   Trainer is primary source but              │
│            UnifiedOrchestrator still has fallback     │
└────────────────────────────────────────────────────────┘
```

### Week 7-10: Production Hardening (30 hours + buffer)

```
┌────────────────────────────────────────────────────────┐
│ Week 7-10: Finalize Split                             │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Week 7:   Disable UnifiedOrchestrator Training       │
│            └─ Set RlRuntimeMode = InferenceOnly       │
│            └─ Remove training code paths              │
│            └─ Live Bot is now inference-only          │
│                                                        │
│  Week 8:   Monitoring & Dashboards                    │
│            └─ Prometheus exporters                    │
│            └─ Grafana dashboards                      │
│            └─ Alerting rules                          │
│                                                        │
│  Week 9:   Documentation                              │
│            └─ Runbooks (live & trainer)               │
│            └─ Troubleshooting guide                   │
│            └─ CI/CD updates                           │
│                                                        │
│  Week 10:  Security & Final Testing                   │
│            └─ Token separation audit                  │
│            └─ Load testing                            │
│            └─ Chaos engineering (failure tests)       │
│                                                        │
│  Status:   COMPLETE - Clean split with monitoring     │
└────────────────────────────────────────────────────────┘
```

---

## Rollback Plan (If Needed)

```
Emergency Rollback at Any Stage:
─────────────────────────────────

Stage 1 (Week 1-2):
  └─ No rollback needed - changes are additive only

Stage 2 (Week 3-4):
  └─ Stop Trainer process
  └─ UnifiedOrchestrator continues as before

Stage 3 (Week 5-6):
  └─ Set canary fraction to 0%
  └─ Or stop Trainer and use UnifiedOrchestrator models

Stage 4 (Week 7-10):
  └─ Re-enable training in UnifiedOrchestrator
  └─ Set RlRuntimeMode back to Training
  └─ Stop Trainer process

Rollback Time: < 5 minutes at any stage
```

---

## Summary: Before vs After

| Aspect | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Processes** | 1 (monolith) | 2 (split) | ✅ Isolation |
| **Training Impact** | Crashes affect trading | Zero impact | ✅ Stability |
| **Model Updates** | 15 min (polling) | <1 sec (Redis) | ✅ 900x faster |
| **Deploy Training** | Restart bot | Independent | ✅ Flexibility |
| **Resource Usage** | Shared | Dedicated | ✅ Performance |
| **Debugging** | Mixed logs | Separate logs | ✅ Clarity |
| **Testing Training** | Risky | Safe sandbox | ✅ Safety |
| **Compliance** | Hard to audit | Clear boundaries | ✅ Auditability |
| **Files Changed** | N/A | 95% unchanged | ✅ Minimal impact |
| **Logic Preserved** | N/A | 100% | ✅ No loss |

---

**For Full Details**: See [ARCHITECTURE_SPLIT_ANALYSIS.md](./ARCHITECTURE_SPLIT_ANALYSIS.md)
**For Quick Summary**: See [ARCHITECTURE_SPLIT_EXECUTIVE_SUMMARY.md](./ARCHITECTURE_SPLIT_EXECUTIVE_SUMMARY.md)
