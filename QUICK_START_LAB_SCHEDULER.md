# Quick Start: Lab Scheduler

## What Is This?
The Lab now has a built-in scheduler that automatically trains models every Sunday from 12:00 PM to 5:45 PM Eastern Time. **No Task Scheduler. No cron jobs. No external automation.** Just start the Lab once and leave it running.

## How to Start

### 1. Set Lab Mode
```bash
# Linux/Mac
export BOT_MODE=Lab

# Windows
set BOT_MODE=Lab
```

### 2. Start the Lab
```bash
cd /home/runner/work/QBot/QBot
dotnet run --project src/UnifiedOrchestrator
```

### 3. That's It!
The Lab will:
- Check the clock every hour when idle
- Automatically start training every Sunday at noon ET
- Run the complete training pipeline
- Return to idle mode after training
- Log everything with `[LAB]` prefix

## What You'll See

### During Idle (Monday-Saturday)
```
[LAB] Internal scheduler initialized - No external Task Scheduler needed
[LAB] Lab idle - next training: Sunday Oct 26, 12:00 PM ET
```

### On Sunday at Noon
```
[LAB] Training window OPEN - Starting training session
[LAB] Training session started - Sunday Oct 19, 12:00 PM ET
[LAB] Loading historical data - started
[LAB] Loading historical data - complete in 2.3 minutes
[LAB] Loading experiences - started
[LAB] Loading experiences - complete in 0.5 minutes
```

### During Training
```
[LAB] CVaR-PPO training - started
[LAB] CVaR-PPO: Epoch 1/10 (10%) - Loss: 0.092
[LAB] CVaR-PPO: Epoch 2/10 (20%) - Loss: 0.084
[LAB] CVaR-PPO: Epoch 3/10 (30%) - Loss: 0.076
...
[LAB] CVaR-PPO complete in 30 min - Sharpe: 2.45, Win Rate: 62%
[LAB] CVaR-PPO complete - Starting Neural UCB
```

### Promotion Decisions
```
[LAB] PROMOTED: cvar-ppo-v2025.10.19 (Sharpe improved 2.30 → 2.45)
[LAB] DISCARDED: lstm-v2025.10.19 (accuracy 57% vs champion 58%)
```

### After Training
```
[LAB] Training session complete - 2 promoted, 1 discarded
[LAB] Next training: Sunday Oct 26, 12:00 PM ET
[LAB] Entering idle mode
```

## Monitoring

### Check if Training is Active
```bash
dotnet run --project src/UnifiedOrchestrator | grep "Training in progress"
```

### View All Lab Logs
```bash
dotnet run --project src/UnifiedOrchestrator | grep "\[LAB\]"
```

### See Promotion Decisions
```bash
dotnet run --project src/UnifiedOrchestrator | grep "PROMOTED\|DISCARDED"
```

### Check Next Training Time
```bash
dotnet run --project src/UnifiedOrchestrator | grep "next training"
```

## Training Schedule

### When Does It Train?
- **Day:** Sunday only
- **Time:** 12:00 PM - 5:45 PM Eastern Time
- **Frequency:** Once per week
- **Duration:** ~2-3 hours (complete pipeline)

### What Gets Trained?
1. CVaR-PPO (30 min)
2. Neural UCB (15 min)
3. LSTM (20 min)
4. Position Management (30 min)
5. S15 Shadow Validation (30 min)

### What Happens After Training?
- Models are saved as challengers
- Promotion evaluation compares to champions
- Winners promoted, losers discarded
- Next training scheduled

## Optional: Daily Maintenance

By default, daily maintenance is **DISABLED**. If you want lightweight updates during market maintenance (5:00-5:15 PM ET Mon-Thu):

### Enable Maintenance
Edit `src/UnifiedOrchestrator/Scheduling/MaintenanceScheduler.cs`:
```csharp
// Line 33
_maintenanceEnabled = true; // Change from false
```

### What Does Maintenance Do?
- Drift detection (~5 min)
- Parameter adjustments (~5 min)
- Performance monitoring (~3 min)
- **Total:** Under 15 minutes
- **Must complete by:** 5:45 PM ET (market reopens at 6 PM)

### What Maintenance Does NOT Do
- ❌ No model training
- ❌ No neural network updates
- ❌ No gradient descent
- ❌ No hyperparameter optimization
- ✅ Only lightweight checks and small adjustments

## Architecture

```
Lab Process
├── InternalScheduler (BackgroundService)
│   ├── Checks clock every hour when idle
│   ├── Detects Sunday 12:00 PM - 5:45 PM ET
│   └── Triggers HistoricalTrainingOrchestrator
│
├── HistoricalTrainingOrchestrator
│   ├── Loads 90 days historical data (TopstepX SDK)
│   ├── Loads 7 days of experiences
│   ├── Runs training pipeline sequentially
│   ├── Saves challengers to registry
│   └── Evaluates promotions
│
└── MaintenanceScheduler (OPTIONAL, BackgroundService)
    ├── Checks clock every hour
    ├── Detects Mon-Thu 5:00-5:15 PM ET
    └── Runs lightweight maintenance operations
```

## Troubleshooting

### Lab Not Training on Sunday
1. Check BOT_MODE is set to "Lab"
2. Verify time is between 12:00 PM - 5:45 PM ET
3. Check logs for errors: `grep "\[LAB\] ERROR"`
4. Verify HistoricalTrainingOrchestrator is registered

### Training Starts But Fails
1. Check historical data connection (TopstepX SDK)
2. Verify experience repository is available
3. Check model registry is accessible
4. Look for error logs with stack traces

### Can't See Logs
1. Make sure you're grepping for `[LAB]`
2. Check log level is set to Information or lower
3. Redirect stderr and stdout: `2>&1`
4. Verify logging configuration in appsettings.json

### Wrong Timezone
1. InternalScheduler uses TimeZoneInfo for "America/New_York"
2. Automatically handles DST transitions
3. Falls back to UTC-5 if timezone not found
4. Check system timezone database is installed

## Production Deployment

### Running as Service

#### Linux (systemd)
```bash
# Create service file
sudo nano /etc/systemd/system/qbot-lab.service

# Add:
[Unit]
Description=QBot Lab Training
After=network.target

[Service]
Type=notify
WorkingDirectory=/home/runner/work/QBot/QBot
Environment=BOT_MODE=Lab
ExecStart=/usr/bin/dotnet run --project src/UnifiedOrchestrator
Restart=always
RestartSec=10
User=runner

[Install]
WantedBy=multi-user.target

# Enable and start
sudo systemctl enable qbot-lab
sudo systemctl start qbot-lab
sudo systemctl status qbot-lab
```

#### Windows (Windows Service)
```powershell
# Publish app
dotnet publish src/UnifiedOrchestrator -c Release -o C:\QBot\Lab

# Install as service
sc create QBotLab binPath="C:\QBot\Lab\UnifiedOrchestrator.exe" start=auto

# Set environment
reg add "HKLM\SYSTEM\CurrentControlSet\Services\QBotLab" /v Environment /t REG_MULTI_SZ /d "BOT_MODE=Lab"

# Start service
sc start QBotLab
```

### Docker
```dockerfile
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY publish/ .
ENV BOT_MODE=Lab
ENTRYPOINT ["dotnet", "UnifiedOrchestrator.dll"]
```

```bash
docker build -t qbot-lab .
docker run -d --restart always --name qbot-lab qbot-lab
docker logs -f qbot-lab | grep "\[LAB\]"
```

## Best Practices

### Do's
✅ Start once and leave running 24/7
✅ Monitor logs regularly
✅ Keep Lab on dedicated infrastructure
✅ Verify training completes each Sunday
✅ Review promotion decisions weekly
✅ Check for error logs after each session

### Don'ts
❌ Don't restart during training window
❌ Don't run Lab and Terminal in same process
❌ Don't modify code during active training
❌ Don't disable error logging
❌ Don't skip timezone configuration
❌ Don't run multiple Lab instances

## FAQ

**Q: Can I change the training schedule?**
A: Yes, edit `InternalScheduler.cs` and modify `TrainingWindowStart`, `TrainingWindowEnd`, and `TrainingDay`.

**Q: Can I run training manually?**
A: Yes, but it's not recommended. Use the scheduler for consistency. If needed, you can call `HistoricalTrainingOrchestrator.RunTrainingSessionAsync()` directly.

**Q: What if I miss a Sunday?**
A: Next Sunday's training will use the latest available data. No problem.

**Q: Can I train more frequently?**
A: Yes, modify `InternalScheduler` to train on additional days or enable daily maintenance for lighter updates.

**Q: Does this work across time zones?**
A: Yes, all times are in Eastern Time (America/New_York). The scheduler handles DST automatically.

**Q: What if training runs past 5:45 PM?**
A: Training will complete naturally. The 5:45 PM is the target end time, not a hard cutoff.

**Q: Can I see training progress in real-time?**
A: Yes, use `dotnet run | grep "\[LAB\]"` to watch logs live.

**Q: Where are trained models saved?**
A: Challengers are saved to the model registry (configured in appsettings.json).

## Support

For issues or questions:
1. Check logs: `grep "\[LAB\] ERROR"`
2. Review PHASE_5_IMPLEMENTATION_SUMMARY.md
3. Check HistoricalTrainingOrchestrator logs
4. Verify TopstepX SDK connectivity
5. Contact development team

## Related Documentation
- [PHASE_5_IMPLEMENTATION_SUMMARY.md](PHASE_5_IMPLEMENTATION_SUMMARY.md) - Technical details
- [PRODUCTION_ARCHITECTURE.md](PRODUCTION_ARCHITECTURE.md) - Overall architecture
- [TRADING_BOT_AUDIT_REPORT.md](TRADING_BOT_AUDIT_REPORT.md) - System audit
