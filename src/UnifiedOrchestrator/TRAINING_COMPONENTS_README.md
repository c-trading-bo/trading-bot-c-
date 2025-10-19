# Training Components Inventory

## Overview

The `training-components.json` file serves as the central catalog of all training methods that the Lab Mode orchestrator coordinates. It provides metadata about each training component including execution phase, estimated time, dependencies, and configuration parameters.

## Purpose

The HistoricalTrainingOrchestrator uses this inventory to:
1. **Plan training sessions** - Understand what components need to run and in what order
2. **Estimate duration** - Calculate expected training time to fit within Sunday window
3. **Resolve dependencies** - Ensure required data (experiences, historical bars) is available
4. **Configure components** - Pass appropriate parameters to each training method
5. **Track progress** - Monitor completion of training pipeline phases

## File Structure

```json
{
  "version": "1.0.0",
  "summary": {
    "documented": 25,
    "targetTotal": 273
  },
  "components": {
    "heavy": [ /* Large neural networks, RL algorithms */ ],
    "medium": [ /* Calibration, optimization, retraining */ ],
    "light": [ /* Online learning, inference */ ]
  }
}
```

### Component Schema

Each training component includes:

```json
{
  "name": "CVaRPPO.TrainAsync",
  "className": "TradingBot.RLAgent.CVaRPPO",
  "filePath": "src/RLAgent/CVaRPPO.cs",
  "phase": "heavy|medium|light",
  "category": "rl_algorithm|calibration|optimization|etc",
  "estimatedTimeMinutes": 30,
  "description": "Human-readable description",
  "dependencies": ["experience_database", "historical_data"],
  "configuration": {
    "epochs": 10,
    "batchSize": 128,
    "learningRate": 0.0003
  }
}
```

## Phase Definitions

### Heavy Phase (Sunday 12 PM - 5:45 PM)

**Duration:** 5 hours 45 minutes  
**Frequency:** Weekly (Sundays only)  
**Environment:** Historical Mode (offline, no broker connection)

**Components:**
- Core RL algorithms (CVaR-PPO, SAC, Meta-Learning)
- Large neural networks (LSTM predictors, ensemble meta-learners)
- Intensive training with gradient descent and backpropagation
- Requires full 90-day historical dataset
- Produces new model artifacts for promotion

**Example:** CVaR-PPO training with 10 epochs on 1,000+ experiences

### Medium Phase (Daily 5:00 PM - 5:15 PM)

**Duration:** 15 minutes  
**Frequency:** Daily (Monday-Friday)  
**Environment:** Maintenance window during market break

**Components:**
- Calibration services (isotonic, microstructure)
- Position management optimization
- Quick retraining cycles
- Statistical validation
- Hot-swap minor model updates

**Example:** Microstructure calibration using last 24 hours of data

### Light Phase (Always Active)

**Duration:** Milliseconds per operation  
**Frequency:** Continuous during live trading  
**Environment:** Live Mode (23 hours/day)

**Components:**
- Online learning weight updates
- Real-time feedback logging
- Shadow learning for S15
- Action selection (inference only)
- Immediate post-trade learning

**Example:** OnlineLearningSystem updates weights after each trade

## Usage

### Reading the Inventory

```csharp
using System.Text.Json;

// Load training components
var json = await File.ReadAllTextAsync("training-components.json");
var inventory = JsonSerializer.Deserialize<TrainingComponentsInventory>(json);

// Get heavy components for Sunday training
var heavyComponents = inventory.Components.Heavy;
foreach (var component in heavyComponents)
{
    Console.WriteLine($"{component.Name}: {component.EstimatedTimeMinutes} min");
}
```

### Filtering by Category

```csharp
// Get all RL algorithm trainers
var rlTrainers = inventory.Components.Heavy
    .Where(c => c.Category == "rl_algorithm")
    .ToList();

// Get calibration services
var calibrators = inventory.Components.Medium
    .Where(c => c.Category == "calibration")
    .ToList();
```

### Calculating Total Time

```csharp
// Calculate total heavy training time
var totalHeavyMinutes = inventory.Components.Heavy
    .Sum(c => c.EstimatedTimeMinutes);

Console.WriteLine($"Total heavy training: {totalHeavyMinutes} minutes");
// Output: Total heavy training: 345 minutes (5h 45m)
```

## Current Coverage

### Documented Components (25 of 273)

**Heavy Phase (11 of 67):**
- ✅ CVaR-PPO training
- ✅ Soft Actor-Critic (SAC) training
- ✅ Meta-learning
- ✅ Neural UCB bandit training
- ✅ Regime blend head training
- ✅ Algorithm wrappers (CVaR-PPO, SAC, Meta)
- ✅ Historical trainers (standard, with CV)
- ✅ Enhanced backtest learning service

**Medium Phase (7 of 177):**
- ✅ Microstructure calibration
- ✅ Isotonic calibration
- ✅ Position management optimization (breakeven, trailing stop)
- ✅ Daily retraining
- ✅ Retraining trigger detection
- ✅ Statistical validation

**Light Phase (7 of 29):**
- ✅ Online learning weight updates
- ✅ Adaptive learning commentary
- ✅ S15 shadow learning
- ✅ MAML live integration
- ✅ Unified brain immediate learning
- ✅ CVaR-PPO action selection (inference)
- ✅ SAC action selection (inference)

### Priority for Adding More Components

To reach 273 components, add in this order:

1. **High Priority (Next 50 components):**
   - Remaining RL algorithms (DQN, A3C, DDPG variants)
   - All LSTM/GRU predictors
   - Strategy-specific trainers (S2, S3, S6, S11, etc.)
   - Risk model calibrators
   - Regime detection trainers

2. **Medium Priority (Next 100 components):**
   - Feature engineering trainers
   - Signal generator optimizers
   - Entry/exit rule calibrators
   - Portfolio optimizer variations
   - Market microstructure analyzers

3. **Lower Priority (Remaining ~100 components):**
   - Diagnostic trainers
   - Experimental algorithms
   - Research prototypes
   - Deprecated but not removed trainers
   - Utility training helpers

## Adding New Components

### Step 1: Identify the Component

Search codebase for training methods:
```bash
# Find trainer classes
find src/ -name "*Trainer*.cs"

# Find training methods
grep -r "TrainAsync\|Train\|Optimize" src/ --include="*.cs"

# Check COMPLETE_TRAINING_INVENTORY.md
cat COMPLETE_TRAINING_INVENTORY.md | grep "Train"
```

### Step 2: Classify the Component

Determine phase based on:
- **Heavy:** Uses gradient descent, multi-epoch training, backpropagation
- **Medium:** Quick updates, calibration, statistical methods (seconds to minutes)
- **Light:** Online learning, immediate feedback, inference (milliseconds)

### Step 3: Add to JSON

```json
{
  "name": "YourTrainer.TrainAsync",
  "className": "TradingBot.YourNamespace.YourTrainer",
  "filePath": "src/YourPath/YourTrainer.cs",
  "phase": "heavy",
  "category": "choose_category",
  "estimatedTimeMinutes": 30,
  "description": "What this trainer does",
  "dependencies": ["what_data_it_needs"],
  "configuration": {
    "key": "value"
  }
}
```

### Step 4: Update Summary Counts

Increment the `documented` count in the `summary` section.

## Maintenance

### Weekly Review

Every Sunday after Lab Mode training:
1. Review which components actually ran
2. Compare actual vs. estimated times
3. Update `estimatedTimeMinutes` if significantly different
4. Add any new components discovered

### Monthly Audit

Once per month:
1. Scan codebase for new trainer classes
2. Check for deprecated/removed components
3. Update categories and classifications
4. Verify dependencies are still accurate

### Version Updates

Increment version when:
- **Patch (1.0.x):** Small fixes, time estimate updates
- **Minor (1.x.0):** Add 10+ new components, update categories
- **Major (x.0.0):** Restructure schema, change phase definitions

## Integration with Lab Mode

### HistoricalTrainingOrchestrator

The orchestrator reads this file to:
```csharp
// Load inventory
var inventory = await LoadTrainingInventory();

// Plan Sunday training session
var heavyComponents = inventory.Components.Heavy;
var totalTime = CalculateTotalTime(heavyComponents);

if (totalTime > TimeSpan.FromHours(5.75))
{
    _logger.LogWarning("Training may exceed Sunday window");
}

// Execute each component
foreach (var component in heavyComponents)
{
    await ExecuteTrainingComponent(component);
}
```

### Dependency Resolution

Before starting training:
```csharp
// Check dependencies for heavy components
foreach (var component in heavyComponents)
{
    foreach (var dependency in component.Dependencies)
    {
        if (!await IsDependencyAvailable(dependency))
        {
            throw new Exception($"{component.Name} requires {dependency}");
        }
    }
}
```

### Configuration Injection

Pass configuration to trainers:
```csharp
// Create trainer with configuration from inventory
var config = component.Configuration;
var trainer = CreateTrainer(component.ClassName, config);

await trainer.TrainAsync(
    epochs: config.Epochs,
    batchSize: config.BatchSize,
    learningRate: config.LearningRate
);
```

## Related Files

- **COMPLETE_TRAINING_INVENTORY.md** - Full audit of all 273 training methods
- **HISTORICAL_DATA_ACQUISITION.md** - How to get data for training
- **HistoricalTrainingOrchestrator.cs** - Uses this inventory to coordinate training

## Future Enhancements

1. **Auto-generation:** Script to parse codebase and auto-populate inventory
2. **Schema validation:** JSON schema file for component validation
3. **Dependency graph:** Visualize component dependencies
4. **Time tracking:** Record actual vs. estimated times
5. **Priority scheduling:** Optimize order of execution for dependencies

---

**Document Version:** 1.0  
**Last Updated:** October 19, 2025  
**Maintainer:** Lab Mode Infrastructure Team
